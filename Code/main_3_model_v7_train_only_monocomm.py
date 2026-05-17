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
import psutil, os
from signatures import get_comm_signature_baranwal

# Script for training and testing for multiple pre-defined communities 
print(f'training only for multiple pre-defined communities - baranwalclark - monocommunity definition')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Config
stats_save_name = 'base'
plot = True
test = True
dynamic_weights = False #changed
prune_graph = False #changed
dataname = 'baranwalclark'
gnn_type = 'AnchorGNN' #changed
time_steps = 4
preprocessed_data_path = f'Embeddings/preprocesses_data_global_{dataname}.pt'
embeddings_path = f'Embeddings/final_otu_embeddings_titles_and_abstracts_{dataname}.pt'
results_path = 'Results/'
is_aug=True
stats_only=False
include_sent_emb=True #changed
temporal = True 
attention_fusion = True

num_epochs = 200
use_comm_signature = True 

# Load preprocessed data
community_tensors = torch.load(preprocessed_data_path)
# Load the sentence embeddings
community_tensors_text = torch.load(embeddings_path)
# Convert to float
for k, v in community_tensors_text.items():
    community_tensors_text[k] = v.float()
    print(f'community_tensors_text.shape for comm {k}={community_tensors_text[k].shape}')



print(f'Number of communities: {len(community_tensors.keys())}')

train_data_time = [v for i, (k, v) in enumerate(list(community_tensors.items()))]
train_data_sent= [v for i, (k, v) in enumerate(list(community_tensors_text.items()))] 


# Get community signatures
train_signatures = [get_comm_signature_baranwal(k, v) for i, (k, v) in enumerate(list(community_tensors.items()))]
print(f'train_signatures:{train_signatures}')

comm_sig_dim = len(train_signatures[0])



########################################################
# Initialize
stats_communities = {}
N_train = len(train_data_time)
for comm in range(N_train):
    stats_communities[comm] = {
        'losses': [],
        'bcd_means': [],
        'physics_loss_per_epoch': [],
        'mse_loss_per_epoch': [],
        'r2_per_otu_per_epoch': [],
        'r2_global_per_epoch': [],
        'sent_recon_loss_per_epoch' : [],
        'comp_r2_per_epoch': [],
        'comm_size': []
    }


print(f'Training...')
print(f'num_epochs={num_epochs}')
# Training loop - train for each comm - model per community
for i, community in enumerate(train_data_time):

    # For the PINN
    t = torch.linspace(1, 4, time_steps).unsqueeze(1).requires_grad_(True)
    #t = torch.linspace(0.1, 1.0, time_steps).unsqueeze(1).requires_grad_(True)
    print(f't={t}, {t.shape}')

    model = PIGNN(num_nodes=train_data_time[i].shape[0], is_aug=is_aug, stats_only=stats_only, include_sent_emb=include_sent_emb, gnn_type=gnn_type, temporal=temporal, attention_fusion=attention_fusion, timepoints=time_steps, use_comm_signature=use_comm_signature, comm_sig_dim=comm_sig_dim)
    print(f'is_aug={is_aug}, stats_only={stats_only}, include_sent_emb={include_sent_emb}, gnn_type={gnn_type}, temporal={temporal}, attention_fusion={attention_fusion}, use_comm_signature={use_comm_signature}')
    optimizer = optim.Adam(model.parameters(), lr=0.005)

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

        # Train model
        print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f} GB")

        optimizer.zero_grad()
        # Reset t
        #t = t.clone().detach().requires_grad_(True)
        # Construct edge index
        num_nodes = train_data_time[i].shape[0]
        edges = list(itertools.product(range(num_nodes), repeat=2))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
        print(f'edge index for train comm {i}={edge_index.shape}')

        reconstructed_data, loss, bcd_mean, bcd_all, physics_loss, mse_loss, bcd_per_otu, r2_per_otu, r2_global, sent_recon_loss, r2_per_time, edge_importance, _, comp_r2 = model(train_data_time[i], edge_index, t=t, sentence_embeddings=train_data_sent[i], comm_signature=train_signatures[i]) 
        stats_communities[i]['losses'].append(loss.item())
        stats_communities[i]['bcd_means'].append(bcd_mean.item())
        stats_communities[i]['physics_loss_per_epoch'].append(physics_loss.item())
        stats_communities[i]['mse_loss_per_epoch'].append(mse_loss.item())
        stats_communities[i]['r2_per_otu_per_epoch'].append(r2_per_otu)
        stats_communities[i]['r2_global_per_epoch'].append(r2_global)
        stats_communities[i]['comp_r2_per_epoch'].append(comp_r2)
        stats_communities[i]['comm_size'].append(num_nodes)
        if include_sent_emb and (sent_recon_loss is not None):
            stats_communities[i]['sent_recon_loss_per_epoch'].append(sent_recon_loss.item())


            
        loss.backward(retain_graph=True)
        optimizer.step()



print(f'stats_community: {stats_communities}')   
N_train = len(train_data_time)
print(f'Average train values: ')
train_communities_bcd_avg = 0
train_communities_r2_avg = 0
train_communities_r2_avg_comp = 0
for comm in range(N_train):
    train_communities_bcd_avg += 1-stats_communities[comm]['bcd_means'][-1]
    train_communities_r2_avg += stats_communities[comm]['r2_global_per_epoch'][-1]
    train_communities_r2_avg_comp += stats_communities[comm]['comp_r2_per_epoch'][-1]
print(f'Training bcd accuracy mean: {train_communities_bcd_avg/N_train}')
print(f'Training r2 score mean: {train_communities_r2_avg/N_train}')
print(f'Training comp r2 score mean: {train_communities_r2_avg_comp/N_train}')


rows = []

for comm, stats in stats_communities.items():
    row = {
        "community": comm,
        "comm_size": stats['comm_size'][-1],
        "final_bcd_acc": 1-stats['bcd_means'][-1],
        "final_r2": stats['r2_global_per_epoch'][-1],
        "final_comp_r2": stats['comp_r2_per_epoch'][-1],
        
    }
    rows.append(row)

df_final = pd.DataFrame(rows)
df_final.to_csv(results_path+f"in_vitro_stats_communities_{stats_save_name}.csv", index=False)

