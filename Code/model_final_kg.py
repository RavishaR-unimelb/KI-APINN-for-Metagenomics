import torch
import torch.nn as nn
import os.path as osp
from typing import Dict, List, Union
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import numpy as np
from scipy.spatial.distance import braycurtis
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score
# Knowledge graph addition

class AttentionFusion(nn.Module): #TODO
    def __init__(self, dims, fused_dim):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(d, fused_dim) for d in dims])
        self.att_weights = nn.Linear(fused_dim, 1)

    def forward(self, embs):  # embs = list of tensors [N, d_i]
        projected = [proj(e) for e, proj in zip(embs, self.projectors)]  # [N, F] * M
        stacked = torch.stack(projected, dim=1)  # [N, M, F]
        scores = self.att_weights(stacked).squeeze(-1)  # [N, M]
        weights = torch.softmax(scores, dim=1)  # [N, M]
        print(f'attention weights: {weights}')
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [N, F]
        return fused


class TimeAwareMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: (15, 100, 1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1])  # use final hidden state
        return out  # shape: (15, output_dim)


class Encoder_PINN(torch.nn.Module):
    def __init__(self, out_features=64, is_aug=False, stats_only=False, include_sent_emb=False, temporal=False, attention_fusion=False, time_steps=100, use_comm_signature=False, sent_emb_dim=384):
        super(Encoder_PINN, self).__init__()
        self.is_aug = is_aug
        self.stats_only = stats_only
        self.out_features = out_features
        self.include_sent_emb = include_sent_emb
        self.use_comm_signature = use_comm_signature

        self.pinn_net = nn.Sequential(
                nn.Linear(1, 100),
                nn.Tanh(),
                nn.Linear(100, 1)
            )

        self.hidden_dim = 256
        self.stats_size = 10 # 12
        self.sent_embed_size = sent_emb_dim
        if self.include_sent_emb:
            self.name_proj = nn.Linear(self.sent_embed_size, self.out_features)

        

        # temporal aspect
        self.temporal = temporal
        self.time_aware_mlp = TimeAwareMLP()

        # Attention fusion
        self.attention_fusion = attention_fusion

        # Setup encoder if necessary
        input_size = time_steps
        if self.is_aug:
            if self.stats_only:
                self.encoder_input_size = self.stats_size
            else:
                self.encoder_input_size = input_size + self.stats_size
        else:
            self.encoder_input_size = input_size 

        if self.include_sent_emb:
            self.abundance_proj = nn.Linear(self.encoder_input_size, self.hidden_dim)

        
        self.resize_time = nn.Linear(time_steps, input_size)
        self.combine_x_and_t = nn.Sequential(
            nn.Linear(input_size*2, self.encoder_input_size),

        )

        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_size, 128),
            #nn.ReLU(), #changed to tanh
            nn.Tanh(),
            nn.Linear(128, self.out_features)
            )
        
        # To combine encoder with stats
        self.combine_enc_and_stats = nn.Sequential(
            nn.Linear(self.out_features + self.stats_size, self.out_features)
            )
        
        
        # Only if attention fusion is true - encode
        if self.attention_fusion and (self.temporal or self.include_sent_emb):
            # Only temporal
            if self.temporal and (not self.include_sent_emb):
                self.attention_fusion_block = AttentionFusion(dims=[self.out_features, 64], fused_dim=128)

            # Only sentence embeddings
            elif self.include_sent_emb and (not self.temporal):
                self.attention_fusion_block = AttentionFusion(dims=[self.out_features, self.out_features], fused_dim=128)

            # Both temporal and sentence embeddings
            elif self.include_sent_emb and self.temporal:
                self.attention_fusion_block = AttentionFusion(dims=[self.out_features, self.out_features, 64], fused_dim=192)


    def forward(self, x, t, sent_emb=None, comm_signature=None):
        """
        x: Tensor of shape (num_otus, 100), each row is a time series
        returns: Tensor of shape (num_otus, 64)
        """
        num_otus, time_steps = x.shape  
                

        # Create stats and store
        if self.is_aug:
            with torch.no_grad():
                # Convert to numpy for statistical ops not available in PyTorch
                x_np = x.cpu().numpy()

                features = []
                for row in x_np:
                    diff = np.diff(row)
                    feat = [
                        row.mean(),
                        row.std(),
                        row.min(),
                        row.max(),
                        np.median(row),
                        diff.mean(),
                        diff.std(),
                        np.sum(row ** 2),
                        row.max() - row.min(),
                        np.percentile(row, 75) - np.percentile(row, 25)
                    ]
                    features.append(feat)

                stats_tensor = torch.tensor(features, dtype=x.dtype, device=x.device)
                print(f'stats_tensor.shape={stats_tensor.shape}')


        
        pinn_out = self.pinn_net(t)
        #print(f'pinn_out={pinn_out.shape}')
        pinn_resized = self.resize_time(pinn_out.T).T #changed
        #pinn_resized = self.resize_time(t.T).T
        #print(f'1: pinn_resized shape={pinn_resized.shape}')
        # Process each OTU through the PINN
        pinn_outputs = []
        for i in range(num_otus):
            # Apply the PINN to t (fixed), scale by the input signal
            x_i = x[i].unsqueeze(1)
            '''
            # Method 1: Product
            xt_input = x_i * pinn_resized  
            print(f'xt_input shape={xt_input.shape}')
            pinn_outputs.append(xt_input.squeeze(-1))
            
            '''
            # Method 2: Linear projection
            concat_input = torch.cat([x_i.T, pinn_resized.T], dim=1)
            #print(f'concat_input shape = {concat_input.shape}')
            xt_input = self.combine_x_and_t(concat_input) # method 2: using mlp
            
            #print(f'xt_input shape={xt_input.shape}')
            pinn_outputs.append(xt_input.T.squeeze(-1))
            

        # Stack and encode
        pinn_outputs = torch.stack(pinn_outputs, dim=0)
        #print(f'pinn_outputs={pinn_outputs}, {pinn_outputs.shape}')
        encoded = self.encoder(pinn_outputs)  # (num_otus, 64)
        print(f'Before combining with stats, encoded shape={encoded.shape}')

        # Combine with stats
        encoded = self.combine_enc_and_stats(torch.cat([encoded, stats_tensor], dim=1))

        # Combine with sentence embeddings
        if self.include_sent_emb and (sent_emb is not None):
            # Combine pinn output and sentence embeddings for node features
            name_encoded = self.name_proj(sent_emb)
            print(f'name encoded:{name_encoded.shape}')
            print(f'encoded: {encoded.shape}')
            if not self.attention_fusion:
                encoded = torch.cat((encoded, name_encoded), dim=1)

        # Consider the temporal aspect
        if self.temporal:
            temporal_encodings = self.time_aware_mlp(x.unsqueeze(-1))
            print(f'temporal encodings: {temporal_encodings.shape}')
            if not self.attention_fusion:
                encoded = torch.cat((encoded, temporal_encodings), dim=1)

        
        # Only if attention fusion is true - encode
        if self.attention_fusion and (self.temporal or self.include_sent_emb):
            # Only temporal
            if self.temporal and (not self.include_sent_emb):
                encoded  = self.attention_fusion_block([encoded, temporal_encodings])
                print(f'Using attention fusion...only temporal')

            # Only sentence embeddings
            elif self.include_sent_emb and (not self.temporal):
                encoded  = self.attention_fusion_block([encoded, name_encoded])
                print(f'Using attention fusion...only sent')

            # Both temporal and sentence embeddings
            elif self.include_sent_emb and self.temporal:
                encoded  = self.attention_fusion_block([encoded, name_encoded, temporal_encodings])
                print(f'Using attention fusion...both temporal and sent')

        print(f'encoded after combining: {encoded.shape}')

        # Add OTU community signature
        if self.use_comm_signature:
            # Replicate
            repeated_signature = comm_signature.repeat(num_otus, 1)
            # Concatenate
            encoded = torch.cat([encoded, repeated_signature], dim=1)
        print(f'final encoded: {encoded.shape}')

        return encoded

    
class Decoder(torch.nn.Module):
    def __init__(self, in_decoder=32, out_decoder=100, heads=1):
        super(Decoder, self).__init__()
        # decoder
        self.decoder_lin = nn.Linear(in_decoder*heads, out_decoder) # heads=4, depends on GAT
        self.final_lin = nn.Linear(out_decoder, out_decoder)
        
    def forward(self, x):
        x_out = self.decoder_lin(x)
        #x_out = F.leaky_relu(x_out)#changed to tanh
        x_out = F.tanh(x_out)
        x_out = self.final_lin(x_out)
        #x_out = F.leaky_relu(x_out)
        return x_out
    
class SentDecoder(torch.nn.Module):
    def __init__(self, in_decoder=32, out_decoder=384, heads=1):
        super(SentDecoder, self).__init__()
        # decoder
        self.decoder_lin = nn.Linear(in_decoder*heads, out_decoder) # heads=4, depends on GAT
        self.final_lin = nn.Linear(out_decoder, out_decoder)
        
    def forward(self, x):
        x_out = self.decoder_lin(x)
        #x_out = F.leaky_relu(x_out)#changed to tanh
        x_out = F.tanh(x_out)
        x_out = self.final_lin(x_out)
        #x_out = F.leaky_relu(x_out)
        return x_out


# Our GNN with anchoring + mlp + attention based neighbour aggretation
class AnchorGraphNN(torch.nn.Module):
    def __init__(
        self,
        in_gcn: int = 64,
        out_gcn: int = 32,
        #num_anchors: int = 5,
        alpha_bias_init: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        #self.num_anchors = num_anchors
        self.out_ = out_gcn

        self.anchor_scorer = nn.Sequential(
            nn.Linear(in_gcn, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.q_proj = nn.Linear(in_gcn, out_gcn, bias=False)
        self.k_proj = nn.Linear(in_gcn, out_gcn, bias=False)
        self.v_proj = nn.Linear(in_gcn, out_gcn, bias=False)
        self.scale  = out_gcn ** -0.5

        self.self_transform = nn.Linear(in_gcn, out_gcn)

        self.alpha_layer = nn.Linear(in_gcn, 1)
        nn.init.constant_(self.alpha_layer.bias, alpha_bias_init)
        nn.init.zeros_(self.alpha_layer.weight)

        self.dropout = nn.Dropout(dropout)

    def _select_anchors(self, x: torch.Tensor) -> torch.Tensor:
        '''
        if x.shape[0] < 5:
            num_anchors = 1
        else:
            num_anchors = int(x.shape[0] * (40 / 100)) # Top 20%
        '''
        num_anchors = int(x.shape[0] * (40 / 100)) # Top 40%
        scores = self.anchor_scorer(x).squeeze(-1)
        k = min(num_anchors, x.size(0))
        print(f'num anchors:{k}/{x.shape[0]}')
        return torch.topk(scores, k).indices

    def _anchor_aggregate(self, x: torch.Tensor, anchor_idx: torch.Tensor) -> torch.Tensor:
        anchor_feats = x[anchor_idx]

        Q = self.q_proj(x)
        K = self.k_proj(anchor_feats)
        V = self.v_proj(anchor_feats)

        attn = (Q @ K.T) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        return attn @ V

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        anchor_idx = self._select_anchors(x)

        self_out = self.self_transform(x)
        agg_out  = self._anchor_aggregate(x, anchor_idx)

        alpha = torch.sigmoid(self.alpha_layer(x))
        print(f'alpha: {alpha}')
        x_out = alpha * self_out + (1.0 - alpha) * agg_out

        return x_out    


class GraphNN(torch.nn.Module):
    def __init__(self, gnn_type, in_gcn=64*2, out_gcn=32, out_mlp=1, heads=8, use_kg_emb=False):
        super(GraphNN, self).__init__()
        self.gnn_type = gnn_type
        self.use_kg_emb = use_kg_emb
        if gnn_type == 'GAT':
            self.out_ = out_gcn*heads
            self.gnn_1 = GATConv(in_gcn, out_gcn, heads=heads, concat=True)
            self.gnn_2 = GATConv(self.out_, out_gcn, heads=heads, concat=True)
            
        elif gnn_type == 'GCN':
            self.out_ = out_gcn
            self.gcn = GCNConv(in_gcn, out_gcn)
            self.gcn_2 = GCNConv(out_gcn, out_gcn)
            
        elif gnn_type == 'GraphSAGE':
            self.out_ = out_gcn
            self.graphsage = SAGEConv(in_gcn, out_gcn)
            #self.graphsage_2 = SAGEConv(out_gcn, out_gcn)
        
        elif gnn_type == 'AnchorGNN':
            self.out_ = out_gcn
            self.anchor_gnn = AnchorGraphNN(in_gcn, out_gcn, alpha_bias_init=4) # changed

        else:
            self.out_ = out_gcn*heads
            self.gnn_1 = GATConv(in_gcn, out_gcn, heads=heads, concat=True)
            self.gnn_2 = GATConv(self.out_, out_gcn, heads=heads, concat=True)
            


        # Final MLP to create edge embeddings
        self.lin = nn.Linear(self.out_, out_mlp)

        if self.use_kg_emb:
            mlp_in = 3*self.out_
        else:
            mlp_in = 2*self.out_
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 100),  # n*F → hidden
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(100, self.out_)  # hidden → final edge embedding
        )

        # For KG
        print(f'For the GNN: self.out_: {self.out_}')
        self.kg_adapt = nn.Linear(768, self.out_)
        

    def forward(self, x, edge_index, kg_embeddings=None):
        # Get node embeddings
        if self.gnn_type == 'GAT':
            x_out, (edge_index_used, attn_weights) = self.gnn_1(x, edge_index, return_attention_weights=True) #GAT
            # Attention
            attn_agg = attn_weights.mean(dim=1).unsqueeze(-1)
            #print(f'attn_agg={attn_agg}')
            #print(f'attn_agg.shape={attn_agg.shape}')
        elif self.gnn_type == 'GCN':
            x_out = self.gnn_2(x_out, edge_index)
            attn_agg = None
        elif self.gnn_type == 'GraphSAGE':
            x_out = self.graphsage(x, edge_index) #GraphSAGE
            #optional
            #x_out = self.graphsage_2(x_out, edge_index)
            attn_agg = None
        elif self.gnn_type == 'AnchorGNN': #domain-specific gnn with anchors
            x_out = self.anchor_gnn(x, edge_index)
            attn_agg = None
        else:
            x_out, (edge_index_used, attn_weights) = self.gnn_1(x, edge_index, return_attention_weights=True) #GAT
            # Attention
            attn_agg = attn_weights.mean(dim=1).unsqueeze(-1)
            #print(f'attn_agg={attn_agg}')
            #print(f'attn_agg.shape={attn_agg.shape}')
       
        #x_out = F.leaky_relu(x_out)#changed to tanh
        x_out = F.tanh(x_out)

        # Get edge embeddings
        src_node_emb = x_out[edge_index[0]]
        dst_node_emb = x_out[edge_index[1]]
        print(f'src:{src_node_emb.shape}, dst:{dst_node_emb.shape}')
        # Normalize: TODO

        # Method 1:
        # operation used: product
        #edge_embeddings = src_node_emb * dst_node_emb

        # Method 2:
        # operation used: concatenation + MLP
        if self.use_kg_emb and (kg_embeddings is not None):
            kg_edge_emb = self.kg_adapt(kg_embeddings)
            print(f'Shapes: src_node_emb:{src_node_emb.shape}, dst_node_emb:{dst_node_emb.shape}, kg_edge_emb:{kg_edge_emb.shape}')
            node_concat = torch.cat([src_node_emb, dst_node_emb, kg_edge_emb], dim=-1)
        else:
            node_concat = torch.cat([src_node_emb, dst_node_emb], dim=-1) 
        print(f'node_concat:{node_concat.shape}')
        edge_embeddings = self.mlp(node_concat)
        

        print(f'edge embeddings shape={edge_embeddings.shape}')
        edge_linear = self.lin(edge_embeddings)
        #edge_linear = F.leaky_relu(edge_linear)#changed to tanh
        edge_linear = F.tanh(edge_linear)
        #print(f'node_embeddings = {x_out}')

        return x_out, edge_linear, attn_agg
    


#  Main structure
class PIGNN(torch.nn.Module):
    def __init__(self, num_nodes=15, heads=8, timepoints=100, is_aug=True, stats_only=False, include_sent_emb=False, gnn_type='GAT', temporal=False, attention_fusion=False, use_comm_signature=False, comm_sig_dim=1, use_kg_emb=False):
        super(PIGNN, self).__init__()
        # Encoder
        #self.encoder = Encoder_RNN(num_otus=num_nodes)
        #self.encoder = MLPEncoder()
        #self.encoder = MLPEncoderStats()

        self.include_sent_emb = include_sent_emb
        self.use_comm_signature = use_comm_signature
        self.use_kg_emb = use_kg_emb
        self.encoder = Encoder_PINN(is_aug=is_aug, stats_only=stats_only, include_sent_emb=include_sent_emb, temporal=temporal, attention_fusion=attention_fusion, time_steps=timepoints, use_comm_signature=self.use_comm_signature)

        # PI-GNN
        self.decoder_input_size = 32
        if self.use_comm_signature:
            gnn_add = comm_sig_dim
        else:
            gnn_add = 0

        if temporal:
            if include_sent_emb:
                self.gnn = GraphNN(in_gcn=(64*3)+gnn_add, out_gcn=self.decoder_input_size, gnn_type=gnn_type, heads=heads, use_kg_emb=self.use_kg_emb)
            else:
                self.gnn = GraphNN(in_gcn=(64*2)+gnn_add, out_gcn=self.decoder_input_size, gnn_type=gnn_type, heads=heads, use_kg_emb=self.use_kg_emb)
        else:
            if include_sent_emb:
                self.gnn = GraphNN(in_gcn=(64*2)+gnn_add, out_gcn=self.decoder_input_size, gnn_type=gnn_type, heads=heads, use_kg_emb=self.use_kg_emb)
            else:
                self.gnn = GraphNN(in_gcn=64+gnn_add, out_gcn=self.decoder_input_size, gnn_type=gnn_type, heads=heads, use_kg_emb=self.use_kg_emb)

        # Decoder
        if gnn_type == 'GAT':
            self.decoder = Decoder(heads=heads, in_decoder=self.decoder_input_size, out_decoder=timepoints) # provide heads value if using GAT
            self.sentence_decoder = SentDecoder(heads=heads, in_decoder=self.decoder_input_size)
        else:
            self.decoder = Decoder(out_decoder=timepoints, in_decoder=self.decoder_input_size)
            self.sentence_decoder = SentDecoder(in_decoder=self.decoder_input_size)

        # MSE
        self.mse = nn.MSELoss()

        # Coefficients - static
        self.coeffs = {
            'mse_coeff': 1,
            'bcd_coeff': 1,
            'physics_coeff': 1,
            'sent_coeff': 1
        }

        # For the physics loss - growth rates
        num_out = 100
        self.combine_x_and_t = nn.Sequential(nn.Linear(timepoints*2, num_out)) # check
        if self.include_sent_emb:
            
            self.growth_rate_lin = nn.Sequential(
                    nn.Linear(num_out*2, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1)  # Output is scalar r_i
                )
            '''
            self.growth_rate_lin = nn.Sequential(
                nn.Linear(num_out*2, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            '''
        else:
            
            self.growth_rate_lin = nn.Sequential(
                    nn.Linear(num_out, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1)  # Output is scalar r_i
                )
            '''
            self.growth_rate_lin = nn.Sequential(
                nn.Linear(num_out, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            '''

        self.growth_rate_lin_alt = nn.Sequential(
            nn.Linear(self.decoder_input_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)  # Output is scalar r_i
        )
        self.sent_resize_for_r = nn.Linear(384, num_out)

        self.freeze_mask = None
        self.resize_time = None

        # Loss coefficients - learnable
        self.log_vars = torch.nn.Parameter(torch.zeros(3))
        self.loss_coeff = nn.Parameter(torch.ones(3))



    def forward(self, x, edge_index, t, sentence_embeddings=None, growth_rates=None, comm_signature=None, kg_embeddings=None):
        # Encoder
        encoded = self.encoder(x, t, sentence_embeddings, comm_signature=comm_signature) # PINN+Encoder
        print(f'encoded={encoded.shape}')

        # PI-GNN
        '''
        # mask input to gnn based on a condition to dynamically freeze certain node updates#####
        if self.freeze_mask is not None:
            encoded = encoded.clone()
            encoded[self.freeze_mask] = encoded[self.freeze_mask].detach()
        ########################################################################################
        '''
        x_out, edge_linear, att_mat = self.gnn(encoded, edge_index, kg_embeddings)
        print(f'edge_linear={edge_linear}, {edge_linear.shape}')
        print(f'x_out:{x_out.shape}')
        # Create adjacency matrix
        N = int(edge_linear.shape[0] ** 0.5)
        #print(f'N={N}')
        A = edge_linear.view(N, N)
        #A = att_mat.view(N, N)
        #print(f'A={A}, {A.shape}')

        # Decoder
        x_reconstructed = self.decoder(x_out)
        sent_reconstructed = self.sentence_decoder(x_out)

        # Normalize the reconstructed data before loss calculation
        #x_reconstructed_norm = x_reconstructed / (x_reconstructed.sum(dim=1, keepdim=True) + 1e-8)
        x_reconstructed_norm = (x_reconstructed - torch.min(x_reconstructed)) / (torch.max(x_reconstructed) - torch.min(x_reconstructed) + 1e-8) # for global
        #x_reconstructed_norm = x_reconstructed # use when data is not normalized
        #print(f'original x = {x}, {x.shape}')
        #print(f'x_reconstructed_norm = {x_reconstructed_norm}, {x_reconstructed_norm.shape}')
 
        # Calculate loss
        loss_1 = self.mse(x_reconstructed_norm, x)
        loss_2 = self.differentiable_bcd(x_reconstructed_norm, x)
        #print(f"t.requires_grad = {t.requires_grad}")
        #print(f"x_reconstructed_norm.requires_grad = {x_reconstructed_norm.requires_grad}")
        #print(f"x_reconstructed_norm grad_fn = {x_reconstructed_norm.grad_fn}")

        #loss_3, growth_rates = self.physics_loss(t, x_reconstructed_norm, A, x, sentence_embeddings, growth_rates=growth_rates)
        loss_3, growth_rates = self.physics_loss_gnn(t, x_reconstructed_norm, A, x, sentence_embeddings, node_embeddings=x_out, growth_rates=growth_rates)

        if self.include_sent_emb:
            #loss_4 = self.mse(sent_reconstructed, sentence_embeddings)

            cos_sim = F.cosine_similarity(sent_reconstructed, sentence_embeddings, dim=1)
            loss_4 = 1 - cos_sim.mean()
            print(f'mse_loss={loss_1}, bcd_loss={loss_2}, physics_loss={loss_3}, sent_mse_loss={loss_4}')
        else:
            loss_4 = None
            print(f'mse_loss={loss_1}, bcd_loss={loss_2}, physics_loss={loss_3}')
        

        lambda_l1 = 0.05
        l1_reg = lambda_l1 * torch.sum(torch.abs(edge_linear.view(-1))) # L1 regularization
        l1_growth_rates = lambda_l1 * torch.sum(torch.abs(growth_rates))
        print(f'l1_reg: {l1_reg}')
        print(f'l1_growth_rates: {l1_growth_rates}')



        eps = 1e-8
        if not self.include_sent_emb:
            loss = (self.coeffs['mse_coeff']*loss_1) + (self.coeffs['bcd_coeff']*loss_2) + (self.coeffs['physics_coeff']*loss_3)
        else:
            loss = (self.coeffs['mse_coeff']*loss_1) + (self.coeffs['bcd_coeff']*loss_2) + (self.coeffs['physics_coeff']*loss_3) + (self.coeffs['sent_coeff']*loss_4)
            #loss = (self.coeffs['mse_coeff']*loss_1) + (self.coeffs['bcd_coeff']*loss_2) + (self.coeffs['physics_coeff']*loss_3)

        
        #loss = self.coeffs['bcd_coeff']*loss_2 + self.coeffs['physics_coeff']*loss_3 #without mse
        #loss += l1_reg
        #loss += l1_growth_rates

        # Evaluate using BCD per timestep
        bcd_all, bcd_mean = self.mean_bcd_per_timestep_standard(x_reconstructed_norm.detach(), x.detach())
        #print(f'bcd per timestep={bcd_all}')
        #print(f'bcd_mean={bcd_mean}')

        # Evaluate BCD per OTU - might not make sense
        bcd_per_otu, bcd_per_otu_mean = self.bcd_per_otu(x_reconstructed_norm.detach(), x.detach())

        # Evaluate R2 score per OTU
        r2_per_otu, r2_per_otu_mean, r2_global, r2_per_time = self.evaluate_r2(x_reconstructed_norm.detach(), x.detach())
        print(f'r2_per_time:{r2_per_time}')

        # Update mask
        r2_per_otu_tensor = torch.tensor(r2_per_otu)
        self.freeze_mask = r2_per_otu_tensor > 0.3
        #print(f'self.freeze_mask={self.freeze_mask}')
        #print(f'r2_per_otu_mean={r2_per_otu_mean}')
        print(f'BCD based accuracy updated: {1 - bcd_mean}')


        return x_reconstructed_norm, loss, bcd_mean, bcd_all, loss_3, loss_1, bcd_per_otu, r2_per_otu, r2_global, loss_4, r2_per_time, edge_linear
    
    
    def mean_bcd_per_timestep(self, x_reconstructed, x_real):
        """
        Computes Bray-Curtis Dissimilarity per timepoint (community composition),
        and the mean BCD.
        Input shape: (num_otus, num_timesteps)
        """
        if hasattr(x_real, "cpu"):
            x_real = x_real.cpu().numpy()
        if hasattr(x_reconstructed, "cpu"):
            x_reconstructed = x_reconstructed.cpu().numpy()

        bcd_per_time = [braycurtis(x_real[:, t], x_reconstructed[:, t]) for t in range(x_real.shape[1])]
        mean_bcd = np.mean(bcd_per_time)

        return bcd_per_time, mean_bcd
    
    def mean_bcd_per_timestep_standard(self, x_reconstructed, x_real):
        """
        Computes Bray-Curtis Dissimilarity per timepoint (community composition),
        with per-timestep normalization, and returns the mean BCD.
        
        Input shape: (num_otus, num_timesteps)
        """
        # Convert tensors to numpy arrays if necessary
        if hasattr(x_real, "cpu"):
            x_real = x_real.cpu().numpy()
        if hasattr(x_reconstructed, "cpu"):
            x_reconstructed = x_reconstructed.cpu().numpy()

        num_timesteps = x_real.shape[1]
        bcd_per_time = []

        for t in range(num_timesteps):
            # Normalize per timestep
            x_real_t = x_real[:, t]
            x_reconstructed_t = x_reconstructed[:, t]

            # Avoid division by zero
            sum_real = x_real_t.sum()
            sum_recon = x_reconstructed_t.sum()
            if sum_real > 0:
                x_real_t = x_real_t / sum_real
            if sum_recon > 0:
                x_reconstructed_t = x_reconstructed_t / sum_recon

            # Compute BCD for this timestep
            bcd = braycurtis(x_real_t, x_reconstructed_t)
            bcd_per_time.append(bcd)

        mean_bcd = np.mean(bcd_per_time)
        return bcd_per_time, mean_bcd
    
    def bcd_per_otu(self, x_reconstructed, x_real):

        original = x_real.cpu().numpy()
        reconstructed = x_reconstructed.cpu().numpy()

        # Evaluate BCD per-OTU
        bcd_scores = [braycurtis(original[i], reconstructed[i]) for i in range(original.shape[0])]

        return bcd_scores, np.mean(bcd_scores)
    
    def differentiable_bcd(self, x_reconstructed, x_real, eps=1e-8):
        """
        Computes differentiable Bray-Curtis Dissimilarity loss.
        Inputs:
            x_reconstructed: torch.Tensor of shape [num_otus, time_points]
            x_real: torch.Tensor of same shape
        Returns:
            bcd_scores: Tensor of BCD values per OTU
            bcd_mean: Mean BCD across all OTUs
        """
        numerator = torch.abs(x_real - x_reconstructed).sum(dim=0) #changed
        #print(f'numerator={numerator.shape}')
        denominator = (torch.abs(x_real) + torch.abs(x_reconstructed)).sum(dim=0) + eps #changed
        bcd_scores = numerator / denominator
        bcd_mean = bcd_scores.mean()
        #print(f'differentiable bcd: {bcd_scores}')
        return bcd_mean
    
    
    def evaluate_r2(self, x_reconstructed, x_real):
        """
        Computes R² per OTU (comparing time series), and the mean R².
        Input shape: (num_otus, num_timesteps)
        """
        x_real = x_real.cpu().numpy()
        x_reconstructed = x_reconstructed.cpu().numpy()

        r2_per_otu = [r2_score(x_real[i], x_reconstructed[i]) for i in range(x_real.shape[0])]
        mean_r2 = np.mean(r2_per_otu)

        r2_global = r2_score(x_real.flatten(), x_reconstructed.flatten())

        r2_per_time = [r2_score(x_real[:, t], x_reconstructed[:, t]) for t in range(x_real.shape[1])]


        return r2_per_otu, mean_r2, r2_global, r2_per_time
    

    
    def physics_loss(self, t, x_reconstructed, A, x_original, sentence_embeddings, growth_rates=None):
        scaling_factor = 1
        # dx_i/dt = r_i * x_i(t) + x_i(t) * sum_j A_ij * x_j(t)

        x_rec = x_reconstructed.T           # (T, N)
        x_og = x_original.T
        T, N = x_rec.shape
        #print(f'x_rec={x_rec.shape}')
        #print(f't={t.shape}')

        # Use autograd to compute dX/dt for each OTU
        '''
        dxdt = []
        for i in range(N): # per OTU
            grad_i = torch.autograd.grad(
                x_rec[:, i].sum(),  # scalar to enable differentiation
                t,
                create_graph=True,
                retain_graph=True
            )[0]  # (T, 1)
            print(f'grad_i shape={grad_i.shape}')
            dxdt.append(grad_i)
        dxdt = torch.cat(dxdt, dim=1) # shape: (T, N)
        '''
        '''
        dxdt = torch.autograd.grad(
            x_rec, t,
            grad_outputs=torch.ones_like(x_rec),
            create_graph=True,
            retain_graph=True
        )[0]
        '''
        dxdt = []
        for i in range(N):
            # Must slice the output properly: (T,)
            #print(f'x_rec[:, i]={x_rec[:, i].shape}')
            grad_i = torch.autograd.grad(
                x_rec[:, i],   # This is (T,), function of t
                t,
                grad_outputs=torch.ones_like(x_rec[:, i]),
                create_graph=True,
                retain_graph=True
            )[0]  # Shape: (T, 1)
            dxdt.append(grad_i)

        dxdt = torch.cat(dxdt, dim=1)
        
        #print(f'dxdt after concat={dxdt}, {dxdt.shape}')

        # Compute right-hand side
        #rhs = self.r * x_rec + x_rec * (x_rec @ A.T)  # shape: (T, N)
        #rhs = x_rec * (x_rec @ A.T)

        #print(f'x_reconstructed.T={x_reconstructed.T.shape}')
        
        x_and_t_outputs = []
        for i in range(N):
            # Apply the PINN to t (fixed), scale by the input signal
            x_i = x_original[i].unsqueeze(1)

            #print(f'x_i.T:{x_i.T.shape}, t.T:{t.T.shape}')
            concat_input = torch.cat([x_i.T, t.T], dim=1)
            #print(f'concat_input shape = {concat_input.shape}')
            xt_input = self.combine_x_and_t(concat_input) # method 2: using mlp
            
            #print(f'xt_input shape={xt_input.shape}')
            x_and_t_outputs.append(xt_input.T.squeeze(-1))

        # Stack and encode
        x_and_t_outputs = torch.stack(x_and_t_outputs, dim=0)
        #print(f'x_and_t_outputs:{x_and_t_outputs}')
        #print(f'x_and_t_outputs:{x_and_t_outputs.shape}')
        if growth_rates is not None:
            print(f'r externally given: {growth_rates}, {growth_rates.shape}')
            if self.include_sent_emb:
                adjusted_sent = self.sent_resize_for_r(sentence_embeddings)
                x_augmented = torch.cat([x_and_t_outputs, adjusted_sent], dim=-1)
                rates = self.growth_rate_lin(x_augmented).view(1, -1)
            
            else:
                rates = self.growth_rate_lin(x_and_t_outputs).view(1, -1)

            print(f'rates:{rates.shape}')
            r = growth_rates*rates
        else:
            if self.include_sent_emb:
                adjusted_sent = self.sent_resize_for_r(sentence_embeddings)
                x_augmented = torch.cat([x_and_t_outputs, adjusted_sent], dim=-1)
                r = self.growth_rate_lin(x_augmented).view(1, -1)

            else:
                r = self.growth_rate_lin(x_and_t_outputs).view(1, -1)
        print(f'r={r},{r.shape}, x_rec:{x_rec.shape}')

        #x_and_t_outputs_norm = (x_and_t_outputs - x_and_t_outputs.mean(dim=1, keepdim=True)) / (x_and_t_outputs.std(dim=1, keepdim=True) + 1e-5)
        #print(f'x_and_t_outputs_norm: {x_and_t_outputs_norm}')
        
        rhs = (r * x_rec) + (x_rec * (x_rec @ A.T)) #original method
        
        #rhs = (r * x_og) + (x_og * (x_og @ A.T))
        #print(f'rhs= {rhs.shape}')
        #print(f'growth rates={self.r}')
        residual = (dxdt - rhs) * scaling_factor
        physics_loss = torch.mean(residual ** 2)


        return physics_loss, r
    
    def physics_loss_gnn(self, t, x_reconstructed, A, x_original, sentence_embeddings, node_embeddings, growth_rates=None):
        scaling_factor = 1
        # dx_i/dt = r_i * x_i(t) + x_i(t) * sum_j A_ij * x_j(t)

        x_rec = x_reconstructed.T           # (T, N)
        x_og = x_original.T
        T, N = x_rec.shape
        #print(f'x_rec={x_rec.shape}')
        #print(f't={t.shape}')

        # Use autograd to compute dX/dt for each OTU
        dxdt = []
        for i in range(N):
            # Must slice the output properly: (T,)
            #print(f'x_rec[:, i]={x_rec[:, i].shape}')
            grad_i = torch.autograd.grad(
                x_rec[:, i],   # This is (T,), function of t
                t,
                grad_outputs=torch.ones_like(x_rec[:, i]),
                create_graph=True,
                retain_graph=True
            )[0]  # Shape: (T, 1)
            dxdt.append(grad_i)

        dxdt = torch.cat(dxdt, dim=1)

        
        r = self.growth_rate_lin_alt(node_embeddings).view(1, -1)
        print(f'r={r},{r.shape}, x_rec:{x_rec.shape}')

        #x_and_t_outputs_norm = (x_and_t_outputs - x_and_t_outputs.mean(dim=1, keepdim=True)) / (x_and_t_outputs.std(dim=1, keepdim=True) + 1e-5)
        #print(f'x_and_t_outputs_norm: {x_and_t_outputs_norm}')
        
        rhs = (r * x_rec) + (x_rec * (x_rec @ A.T)) #original method
        
        #rhs = (r * x_og) + (x_og * (x_og @ A.T))
        #print(f'rhs= {rhs.shape}')
        #print(f'growth rates={self.r}')
        residual = (dxdt - rhs) * scaling_factor
        physics_loss = torch.mean(residual ** 2)


        return physics_loss, r
    
    



