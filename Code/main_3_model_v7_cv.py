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
from helper import sample_predefined_communities
import psutil, os
from signatures import get_comm_signature_baranwal
from sklearn.model_selection import KFold
from scipy.stats import linregress

# Script for training and testing for multiple pre-defined communities 
print(f'training and testing for multiple pre-defined communities - baranwalclark - CV pipeline')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Config
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
temporal = True ###### changed ######## 
attention_fusion = True
info = f'{dataname}_{gnn_type}_{time_steps}_sent_{include_sent_emb}_t'
num_epochs = 200
use_comm_signature = True # changed ####### check ##########

# Load preprocessed data
community_tensors = torch.load(preprocessed_data_path)
# Load the sentence embeddings
community_tensors_text = torch.load(embeddings_path)
# Convert to float
for k, v in community_tensors_text.items():
    community_tensors_text[k] = v.float()
    print(f'community_tensors_text.shape for comm {k}={community_tensors_text[k].shape}')


# Number of folds
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Convert community_tensors (dict) to list for indexing
all_data_time = list(community_tensors.values())
all_data_sent = [v.float() for v in community_tensors_text.values()]
all_comm_ids = list(community_tensors.keys())
print(f'Number of communities: {len(all_comm_ids)}')

# Get community signatures
all_signatures = [get_comm_signature_baranwal(k, v) for k, v in community_tensors.items()]
comm_sig_dim = len(all_signatures[0])

# split into train-test
train_n = int(0.8*len(all_comm_ids))
print(f'Number of train comms: {train_n}, test comms: {95-train_n}')
all_data_time_train = all_data_time[: train_n]
all_data_time_test = all_data_time[train_n:]
all_data_sent_train = all_data_sent[: train_n]
all_data_sent_test = all_data_sent[train_n:]
all_comm_ids_train = all_comm_ids[: train_n]
all_comm_ids_test = all_comm_ids[train_n: ]
all_signatures_train = all_signatures[: train_n]
all_signatures_test = all_signatures[train_n: ]

t = torch.linspace(1, time_steps, time_steps).unsqueeze(1).requires_grad_(True)

# Track CV results
cv_results = {
    "train_bcd": [],
    "train_r2": [],
    "train_r2_pooled": [],
    "val_bcd": [],
    "val_r2": [],
    "val_r2_pooled": [],
    "test_bcd": [],
    "test_r2": [],
    "test_r2_pooled": [],
    "test_r2_pearson": []
}

print(f"Running {k_folds}-fold cross validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(all_data_time_train)):
    print(f"\n========== Fold {fold+1}/{k_folds} ==========")

    # Split into train & val folds
    train_data_time = [all_data_time_train[i] for i in train_idx]
    train_data_sent = [all_data_sent_train[i] for i in train_idx]
    train_signatures = [all_signatures_train[i] for i in train_idx]

    val_data_time = [all_data_time_train[i] for i in val_idx]
    val_data_sent = [all_data_sent_train[i] for i in val_idx]
    val_signatures = [all_signatures_train[i] for i in val_idx]

    # Initialize model fresh for each fold
    model = PIGNN(
        is_aug=is_aug, stats_only=stats_only, include_sent_emb=include_sent_emb,
        gnn_type=gnn_type, temporal=temporal, attention_fusion=attention_fusion,
        timepoints=time_steps, use_comm_signature=use_comm_signature, comm_sig_dim=comm_sig_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    stats_communities = {comm: {"losses": [], "bcd_means": [], "r2_global_per_epoch": [], "pooled_r2_per_epoch": []}
                         for comm in range(len(train_data_time))}

    # Training loop
    for epoch in range(num_epochs):
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
        total_loss = 0.0

        for i, community in enumerate(train_data_time):
            optimizer.zero_grad()
            num_nodes = community.shape[0]
            edges = list(itertools.product(range(num_nodes), repeat=2))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            outputs = model(community, edge_index, t=t,
                            sentence_embeddings=train_data_sent[i],
                            comm_signature=train_signatures[i])
            reconstructed_data, loss, bcd_mean, bcd_all, physics_loss, mse_loss, \
            bcd_per_otu, r2_per_otu, r2_global, sent_recon_loss, r2_per_time, edge_importance, pooled_r2, comp_r2 = outputs

            total_loss += loss
            stats_communities[i]["losses"].append(loss.item())
            stats_communities[i]["bcd_means"].append(bcd_mean.item())
            stats_communities[i]["r2_global_per_epoch"].append(r2_global)
            stats_communities[i]["pooled_r2_per_epoch"].append(pooled_r2)

        avg_total_loss = total_loss
        avg_total_loss.backward(retain_graph=True)
        optimizer.step()

    # Evaluate on train
    train_bcd = np.mean([1 - stats_communities[i]["bcd_means"][-1] for i in range(len(train_data_time))])
    train_r2 = np.mean([stats_communities[i]["r2_global_per_epoch"][-1] for i in range(len(train_data_time))])
    train_pooled_r2 = np.mean([stats_communities[i]["pooled_r2_per_epoch"][-1] for i in range(len(train_data_time))])

    # Validation loop
    model.eval()
    val_bcd_acc = 0
    val_r2_acc = 0
    val_pooled_r2_acc = 0
    for j, community in enumerate(val_data_time):
        num_nodes_val = community.shape[0]
        edges_val = list(itertools.product(range(num_nodes_val), repeat=2))
        edge_index_val = torch.tensor(edges_val, dtype=torch.long).t().contiguous()

        outputs = model(community, edge_index_val, t=t,
                        sentence_embeddings=val_data_sent[j],
                        comm_signature=val_signatures[j])
        reconstructed_data, loss, bcd_mean, bcd_all, physics_loss, mse_loss, \
        bcd_per_otu, r2_per_otu, r2_global, sent_recon_loss, r2_per_time, edge_importance, pooled_r2, comp_r2 = outputs

        val_bcd_acc += 1 - bcd_mean.item()
        val_r2_acc += r2_global
        val_pooled_r2_acc += pooled_r2

    val_bcd = val_bcd_acc / len(val_data_time)
    val_r2 = val_r2_acc / len(val_data_time)
    val_pooled_r2 = val_pooled_r2_acc / len(val_data_time)

    print(f"[Fold {fold+1}] Train BCD={train_bcd:.4f}, Train R²={train_r2:.4f}, pooled r2:{train_pooled_r2}, "
          f"Val BCD={val_bcd:.4f}, Val R²={val_r2:.4f}, pooled r2:{val_pooled_r2}")

    cv_results["train_bcd"].append(train_bcd)
    cv_results["train_r2"].append(train_r2)
    cv_results["train_r2_pooled"].append(train_pooled_r2)
    cv_results["val_bcd"].append(val_bcd)
    cv_results["val_r2"].append(val_r2)
    cv_results["val_r2_pooled"].append(val_pooled_r2)

    # Testing
    model.eval()
    test_bcd_acc = 0
    test_r2_acc = 0
    test_pooled_r2_acc = 0
    pred_community_abundances = []
    for k, community in enumerate(all_data_time_test):
        num_nodes_val = community.shape[0]
        edges_val = list(itertools.product(range(num_nodes_val), repeat=2))
        edge_index_val = torch.tensor(edges_val, dtype=torch.long).t().contiguous()

        outputs = model(community, edge_index_val, t=t,
                        sentence_embeddings=all_data_sent_test[k],
                        comm_signature=all_signatures_test[k])
        reconstructed_data, loss, bcd_mean, bcd_all, physics_loss, mse_loss, \
        bcd_per_otu, r2_per_otu, r2_global, sent_recon_loss, r2_per_time, edge_importance, pooled_r2, comp_r2 = outputs

        test_bcd_acc += 1 - bcd_mean.item()
        test_r2_acc += r2_global
        test_pooled_r2_acc += pooled_r2

        pred_community_abundances.append(reconstructed_data)

    test_bcd = test_bcd_acc / len(all_data_time_test)
    test_r2 = test_r2_acc / len(all_data_time_test)
    test_pooled_r2 = test_pooled_r2_acc / len(all_data_time_test)

     # Calculate pearson r2
    true_all = []
    pred_all = []

    for real, pred in zip(all_data_time_test, pred_community_abundances):
        # Flatten both tensors
        real_flat = real.flatten().detach().cpu().numpy()
        pred_flat = pred.flatten().detach().cpu().numpy()

        true_all.append(real_flat)
        pred_all.append(pred_flat)

    # Concatenate across all communities
    true_all = np.concatenate(true_all)
    pred_all = np.concatenate(pred_all)

    # Compute Pearson R²
    _, _, r, _, _ = linregress(pred_all, true_all)
    pearson_r2_global = r**2
    print(f"Global Pearson R² across all communities: {pearson_r2_global:.4f}")


    cv_results["test_bcd"].append(test_bcd)
    cv_results["test_r2"].append(test_r2)
    cv_results["test_r2_pooled"].append(test_pooled_r2)
    cv_results["test_r2_pearson"].append(pearson_r2_global)


# Final CV summary
print("\n========== Cross Validation Summary ==========")
print(f"Train BCD: {np.mean(cv_results['train_bcd']):.4f} ± {np.std(cv_results['train_bcd']):.4f}")
print(f"Train R² : {np.mean(cv_results['train_r2']):.4f} ± {np.std(cv_results['train_r2']):.4f}, {np.mean(cv_results['train_r2_pooled']):.4f}")
print(f"Val BCD  : {np.mean(cv_results['val_bcd']):.4f} ± {np.std(cv_results['val_bcd']):.4f}")
print(f"Val R²   : {np.mean(cv_results['val_r2']):.4f} ± {np.std(cv_results['val_r2']):.4f}, {np.mean(cv_results['val_r2_pooled']):.4f}")
print(f"Test BCD  : {np.mean(cv_results['test_bcd']):.4f} ± {np.std(cv_results['test_bcd']):.4f}")
print(f"Test R²   : {np.mean(cv_results['test_r2']):.4f} ± {np.std(cv_results['test_r2']):.4f}")
print(f"Pooled R2: {np.mean(cv_results['test_r2_pooled']):.4f} ± {np.std(cv_results['test_r2_pooled']):.4f}")
print(f"Test pearson r2: {np.mean(cv_results['test_r2_pearson']):.4f} ± {np.std(cv_results['test_r2_pearson']):.4f}")