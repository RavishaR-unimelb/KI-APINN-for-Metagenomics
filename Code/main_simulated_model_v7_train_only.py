from paper2.dataset import Dataset
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
from sklearn.model_selection import KFold
from scipy.stats import linregress

# Script for training and testing for multiple pre-defined communities 
print(f'training only for multiple pre-defined communities - simulated - monocomm definition')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Config
plot = False
test = False
dataname = 'simulated'
gnn_type = 'AnchorGNN' #changed
time_steps = 4
preprocessed_data_path = f'Embeddings/preprocesses_data_global_{dataname}.pt'
results_path = 'Results/'
is_aug=True
stats_only=False
include_sent_emb=False #changed
temporal = True #changed
attention_fusion = True
info = f'{dataname}_{gnn_type}_{time_steps}'
num_epochs = 200
use_comm_signature = False #changed

# Load preprocessed data
community_tensors = torch.load(preprocessed_data_path)


# Convert community_tensors (dict) to list for indexing
all_data_time_train = list(community_tensors['train'].values())
print(f'Train comms: {len(all_data_time_train)}')
all_data_time_test = list(community_tensors['test'].values())
print(f'Test comms: {len(all_data_time_test)}')
#all_data_time_train = all_data_time_train + all_data_time_test


# Consider all communities: train and test
all_data_time_train = all_data_time_train + all_data_time_test
print(f'Updated Train comms: {len(all_data_time_train)}')


t = torch.linspace(1, time_steps, time_steps).unsqueeze(1).requires_grad_(True)

# Track CV results
final_results = {
    "train_bcd": [],
    "train_r2": [],
    "train_r2_pooled": [],
    "train_comp_r2": []
}






stats_communities = {comm: {"losses": [], "bcd_means": [], "r2_global_per_epoch": [], "pooled_r2_per_epoch": [], "comp_r2_per_epoch": []}
                        for comm in range(len(all_data_time_train))}

# Training loop
for i, community in enumerate(all_data_time_train):
    # Initialize model fresh for each comm
    model = PIGNN(
        is_aug=is_aug, stats_only=stats_only, include_sent_emb=include_sent_emb,
        gnn_type=gnn_type, temporal=temporal, attention_fusion=attention_fusion,
        timepoints=time_steps, use_comm_signature=use_comm_signature, comm_sig_dim=0, mono_species=True
    )
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(num_epochs):
        model.train()
        #print(f'Epoch {epoch} ===========')

        
        optimizer.zero_grad()
        num_nodes = community.shape[0]
        edges = list(itertools.product(range(num_nodes), repeat=2))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        outputs = model(community, edge_index, t=t)
        reconstructed_data, loss, bcd_mean, bcd_all, physics_loss, mse_loss, \
        bcd_per_otu, r2_per_otu, r2_global, sent_recon_loss, r2_per_time, edge_importance, pooled_r2,comp_r2 = outputs

        stats_communities[i]["losses"].append(loss.item())
        stats_communities[i]["bcd_means"].append(bcd_mean.item())
        stats_communities[i]["r2_global_per_epoch"].append(r2_global)
        stats_communities[i]["pooled_r2_per_epoch"].append(pooled_r2)
        stats_communities[i]["comp_r2_per_epoch"].append(comp_r2)


        loss.backward(retain_graph=True)
        optimizer.step()

# Evaluate on train                
train_bcd = np.mean([1 - stats_communities[i]["bcd_means"][-1] for i in range(len(all_data_time_train))])
train_r2 = np.mean([stats_communities[i]["r2_global_per_epoch"][-1] for i in range(len(all_data_time_train))])
train_pooled_r2 = np.mean([stats_communities[i]["pooled_r2_per_epoch"][-1] for i in range(len(all_data_time_train))])
train_comp_r2 = np.mean([stats_communities[i]["comp_r2_per_epoch"][-1] for i in range(len(all_data_time_train))])

print(f"On average --> Train BCD={train_bcd:.4f}, Train R²={train_r2:.4f}, pooled r2:{train_pooled_r2}, train_comp_r2:{train_comp_r2}")
with open(results_path+"training_metrics_baseline_model.txt", "a") as f:
    f.write(f"Train BCD={train_bcd:.4f}, Train R²={train_r2:.4f}, "
            f"Pooled R²={train_pooled_r2:.4f}, Comp R²={train_comp_r2:.4f}\n")



