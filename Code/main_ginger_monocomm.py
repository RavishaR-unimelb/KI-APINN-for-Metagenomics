from paper2.dataset import Dataset
#from paper2.loess import loessline
#from paper2.umap_util import run_umap
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch.nn as nn
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import itertools
import torch
from model_v7_final import PIGNN 
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from helper import plot_true_vs_predicted
from signatures import get_comm_signature_ginger
from paper2.dataset import Dataset
from kg_embeddings import get_kg_based_text_embeddings

print(f'Ginger data')

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Config
plot = False
dynamic_weights = False #changed
prune_graph = False #changed
dataname = 'G_rhiz_C_138_200X'
gnn_type = 'AnchorGNN' # changed

#N_all = 30
N_all = 30
N_test = 30 # for testing
N_list = [5, 10, 15, 20, 25]

#N_list = [5]
time_steps = 7

# load OTU names
df_raw_path = f'Data/ginger_data/{dataname}_preprocessed.csv'
df_raw = pd.read_csv(df_raw_path, index_col="OTU ID", sep=',')
df_raw = Dataset(df_raw)
df_ranked = df_raw.select_by_rank(count=N_all)

df_ranked_time = df_ranked.time_interval(start=0, delta_t=time_steps)
otu_names = df_ranked_time.get_OTUs
print(f'OTU names: {otu_names}')

for N in N_list:
    # for training
    print(f'For N: {N}-----------------------------------------------')
    
    preprocessed_data_path = f'Embeddings/preprocessed_data_global_{dataname}_{N_all}_{time_steps}.pt'
    embeddings_path = f'Embeddings/final_otu_embeddings_titles_and_abstracts_{dataname}_{N_all}.pt'
    results_path = 'Results/'
    is_aug=True 
    stats_only=False
    include_sent_emb=True #changed
    temporal = True 
    attention_fusion = True
    use_comm_signature = True  
    
    #info = f'{dataname}_{gnn_type}_train_{N}_test_{N_test}_{N_all}_{time_steps}_xaug_sent_ta' #{dataname}_..._xaug_sent or {dataname}_..._xaug_ta or t
    info = f'{dataname}_{gnn_type}_{N}_{time_steps}_sent_{include_sent_emb}_temp_{temporal}_ta' 
    #info = f'{dataname}_{gnn_type}_{N}_{time_steps}_sent_{include_sent_emb}'
    num_epochs = 200

    # Load preprocessed data
    data_original = torch.load(preprocessed_data_path)
    print(f'data all = {data_original.shape}')
    if N <= data_original.shape[0]:
        data = data_original[:N]
    else:
        continue
    otu_names_N = otu_names[:N]
    print(f'Abundance data for training: {data.shape}')

    # Load the sentence embeddings
    sent_emb_original = torch.load(embeddings_path).float()
    print(f'sent_emb_original.shape={sent_emb_original.shape}')
    sent_emb_dim = sent_emb_original.shape[1]
    # Get embeddings only for the required OTUs
    sent_emb = sent_emb_original[:N]
    print(f'sent_emb.shape for training={sent_emb.shape}')

    # Create edge index for a fully-connected graph
    num_nodes = data.shape[0]
    edges = list(itertools.product(range(num_nodes), repeat=2))
    #edges = list(itertools.combinations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
    #edge_index = edge_index[:, edge_index[0] != edge_index[1]] # removing the self loops
    print(f'edge index={edge_index}, {edge_index.shape}')

    # For the PINN
    t = torch.linspace(1, time_steps, time_steps).unsqueeze(1).requires_grad_(True)
    #t = torch.linspace(0.1, 1.0, time_steps).unsqueeze(1).requires_grad_(True)
    print(f't={t}, {t.shape}')


    # Community signature 
    ids = list(range(N))
    
    community_signature = get_comm_signature_ginger(dataname, N, time_steps, ids)
    #community_signature = get_comm_signature_all_levels(dataname, N, time_steps, ids)
    comm_sig_dim = len(community_signature)

    # Kg embeddings
    #get_kg_based_text_embeddings(df_raw_path, otu_names_N, edge_index)   



    ########################################################
    print(f'num_epochs={num_epochs}')

    # Initialize model
    growth_rates = nn.Parameter(torch.randn(num_nodes)) # initialize per community
    
    model = PIGNN(is_aug=is_aug, stats_only=stats_only, include_sent_emb=include_sent_emb, gnn_type=gnn_type, temporal=temporal, attention_fusion=attention_fusion, use_comm_signature=use_comm_signature, comm_sig_dim=comm_sig_dim, sent_emb_dim=sent_emb_dim, timepoints=time_steps)
    
    print(f'is_aug={is_aug}, stats_only={stats_only}, include_sent_emb={include_sent_emb}, gnn_type={gnn_type}, attention_fusion={attention_fusion}, use_comm_signature={use_comm_signature}')
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': [growth_rates]}
    ], lr=0.005)

    losses = []
    bcd_means = []
    bcd_all_per_epoch = []
    physics_loss_per_epoch = []
    mse_loss_per_epoch = []
    r2_per_otu_per_epoch = []
    sent_recon_loss_per_epoch = []

    print(f'Training for {N}...')
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} ------------------------------')
        if dynamic_weights:
            max_w = 0.5
            ramp_epochs = 80
    
            if epoch < ramp_epochs:
                w = max_w * (epoch / ramp_epochs)
            else:
                w = max_w
            
            model.coeffs['physics_coeff'] = 1-w
            model.coeffs['mse_coeff'] = w
            model.coeffs['bcd_coeff'] = w

        model.train()
        print(model.state_dict().keys())
        optimizer.zero_grad()

        reconstructed_data, loss, bcd_mean, bcd_all, physics_loss, mse_loss, bcd_per_otu, r2_per_otu, r2_global, sent_recon_loss, r2_per_time, edge_importance, pooled_r2, comp_r2 = model(data, edge_index, t=t, sentence_embeddings=sent_emb, comm_signature=community_signature) 
        print(f'Loss: {loss}')

        losses.append(loss.item())
        bcd_means.append(bcd_mean.item())
        bcd_all_per_epoch.append(bcd_per_otu)
        physics_loss_per_epoch.append(physics_loss.item())
        mse_loss_per_epoch.append(mse_loss.item())
        r2_per_otu_per_epoch.append(r2_per_otu)
        print(f'r2_per_otu={r2_per_otu}')
        print(f'r2_global={r2_global}')
        print(f'comp_r2:{comp_r2}')

        if include_sent_emb and (sent_recon_loss is not None):
            sent_recon_loss_per_epoch.append(sent_recon_loss.item())

        loss.backward()
        # Testing gradients

        #for name, param in model.named_parameters():
        #    if "growth_rate_lin" in name:
        #        print(name, param.grad)
        #   elif "phy_combine_x_and_t" in name:
        #        print(name, param.grad)
        #    else:
        #        continue

        optimizer.step()

    # edge importance values
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    edges_with_importance = [
        (f'{otu_names_N[src]} - {src}', f'{otu_names_N[dst]} - {dst}', float(imp))
        for (src, dst, imp) in zip(src_nodes, dst_nodes, edge_importance)
    ]
    sorted_edges = sorted(edges_with_importance, key=lambda x: abs(x[2]), reverse=True)
    #print(f'edges_with_importance:{edges_with_importance}')
    df_sorted_edges = pd.DataFrame(sorted_edges, columns=['source', 'target', 'importance'])
    #print(df_sorted_edges)
    df_sorted_edges.to_csv(results_path+f'{info}_edge_importances.csv')

    if plot:
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss over Epochs')
        plt.grid(True)
        plt.legend()

        # Plot BCD
        plt.subplot(1, 2, 2)
        plt.plot(bcd_means, label='BCD Mean', color='orange')
        plt.plot(physics_loss_per_epoch, label='Physics Loss', color='green')
        plt.plot(mse_loss_per_epoch, label='MSE Loss', color='purple')
        if sent_recon_loss_per_epoch != []:
            plt.plot(sent_recon_loss_per_epoch, label='Sentence MSE', color='cyan')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses over Epochs')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(results_path+f'epoch_loss_bcd_{info}.jpg', dpi=300)

        # For BCD per OTU
        bcd_all_per_epoch = np.array(bcd_all_per_epoch)
        num_epochs, num_otus = bcd_all_per_epoch.shape

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % cmap.N) for i in range(num_otus)]
        

        # For R2 per OTU
        r2_per_otu_per_epoch = np.array(r2_per_otu_per_epoch)
        num_epochs, num_otus = r2_per_otu_per_epoch.shape

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % cmap.N) for i in range(num_otus)]
        
        plt.figure(figsize=(12, 5))
        for otu_id in range(num_otus):
            plt.plot(range(num_epochs), r2_per_otu_per_epoch[:, otu_id], label=f'OTU {otu_id + 1}', color=colors[otu_id])

        plt.xlabel('Epoch')
        plt.ylabel('R2 Value')
        plt.title('R2 per OTU across Epochs')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(results_path+f'r2_all_OTUs_{info}.jpg', dpi=300)

        plot_true_vs_predicted(data.detach(), reconstructed_data.detach(), r2_global, mse_loss_per_epoch[-1], f'{info}', results_path)

        # Plot R2 per timestep
        plt.figure(figsize=(12, 5))
        plt.plot(range(time_steps), r2_per_time)
        plt.xlabel('Timestep')
        plt.ylabel('R2 Value')
        plt.title('R2 per timestep')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(results_path+f'r2_per_timestep_{info}.jpg', dpi=300)


    

