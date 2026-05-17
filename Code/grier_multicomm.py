import pandas as pd
import psutil, os, random, itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from model_v7_final import PIGNN 
from signatures import get_comm_signature_grier_all_test
from sklearn.model_selection import KFold
from scipy.stats import linregress

# -----------------------
#  Reproducibility
# -----------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print(f'Running Grier CV...')

# -----------------------
#  Config
# -----------------------
dataname = 'REC_L5'
dynamic_weights = False #changed
prune_graph = False #changed
gnn_type = 'AnchorGNN' #changed
time_steps = 7
num_epochs = 200
k_folds = 5
train_otu_count = 20 #changed
test_otu_count = 20 #changed
h_otu_count = 20 #changed
print(f'dataname:{dataname}, train_otu_count:{train_otu_count}, test_otu_count:{test_otu_count}')

preprocessed_data_path = f'Embeddings/preprocessed_data_global_{dataname}_train_test_{time_steps}.pt'
embeddings_path = f'Embeddings/final_otu_embeddings_titles_and_abstracts_{dataname}_{time_steps}_train_test.pt'
results_path = 'Results/'

is_aug = False ############################################################
stats_only = False
include_sent_emb = True #changed
temporal = True ###### changed ######## ############## check
attention_fusion = True
use_comm_signature = True # changed ############## check

# -----------------------
#  Load Data
# -----------------------
data_final = torch.load(preprocessed_data_path)
sent_embeddings_final = torch.load(embeddings_path)
all_pid_final = list(data_final.keys())
print(f'Total number of communities: {len(all_pid_final)}')

# Split
train_n = int(0.8*len(all_pid_final))
data_all = data_final
sent_embeddings_all = sent_embeddings_final
all_pid = all_pid_final[: train_n]
print(f'Total number of train communities: {len(all_pid)}')

data_all_h = data_final
sent_embeddings_all_h = sent_embeddings_final
all_pid_h = all_pid_final[train_n:]
print(f'Total number of hold-out communities: {len(all_pid_h)}')

# PINN input
t = torch.linspace(1, time_steps, time_steps).unsqueeze(1).requires_grad_(True)

# -----------------------
#  Cross Validation Setup
# -----------------------
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

# To store results across folds
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

# -----------------------
#  K-Fold Loop
# -----------------------
for fold, (train_idx, test_idx) in enumerate(kf.split(all_pid)):
    print(f"\n===== Fold {fold+1} / {k_folds} =====")
    train_pids = [all_pid[i] for i in train_idx]
    test_pids = [all_pid[i] for i in test_idx]

    # Build comm signatures
    comm_sig_dict = {}
    for pid in train_pids:
        otu_sig = get_comm_signature_grier_all_test(
            data_all[pid][0][:train_otu_count],
            data_all[pid][1][:train_otu_count].clone().detach().cpu().numpy()
        )
        comm_sig_dict[pid] = otu_sig
    for pid in test_pids:
        otu_sig = get_comm_signature_grier_all_test(
            data_all[pid][0][:test_otu_count],
            data_all[pid][1][:test_otu_count].clone().detach().cpu().numpy()
        )
        comm_sig_dict[pid] = otu_sig

    # Init model + optimizer
    model = PIGNN(
        is_aug=is_aug,
        stats_only=stats_only,
        include_sent_emb=include_sent_emb,
        gnn_type=gnn_type,
        temporal=temporal,
        timepoints=time_steps,
        attention_fusion=attention_fusion,
        use_comm_signature=use_comm_signature
    )
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # -------------------
    # Train Loop
    # -------------------
    stats_communities = {pid: {"losses": [], "bcd_means": [], "r2_global": [], "pooled_r2_per_epoch": []} for pid in train_pids}
    print("Training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for pid in train_pids:
            optimizer.zero_grad()

            otu_list, data = data_all[pid]
            otu_list = otu_list[:train_otu_count]
            data = data[:train_otu_count]

            # Graph edges (fully connected)
            num_nodes = data.shape[0]
            edges = list(itertools.product(range(num_nodes), repeat=2))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            sent_emb = sent_embeddings_all[pid][:train_otu_count].to(torch.float32)

            reconstructed_data, loss, bcd_mean, _, physics_loss, mse_loss, _, _, r2_global, sent_recon_loss, _, _, pooled_r2, comp_r2 = model(
                data, edge_index, t=t, sentence_embeddings=sent_emb, comm_signature=comm_sig_dict[pid]
            )

            total_loss += loss
            stats_communities[pid]["losses"].append(loss.item())
            stats_communities[pid]["bcd_means"].append(bcd_mean.item())
            stats_communities[pid]["r2_global"].append(r2_global)
            stats_communities[pid]["pooled_r2_per_epoch"].append(pooled_r2)

        avg_total_loss = total_loss
        print(f'avg_total_loss={avg_total_loss}')

        avg_total_loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: avg loss={total_loss/len(train_pids):.4f}")

    # -------------------
    # Train Summary
    # -------------------
    train_bcd_acc = np.mean([1 - stats_communities[pid]["bcd_means"][-1] for pid in train_pids])
    train_r2 = np.mean([stats_communities[pid]["r2_global"][-1] for pid in train_pids])
    train_pooled_r2 = np.mean([stats_communities[pid]["pooled_r2_per_epoch"][-1] for pid in train_pids])

    # -------------------
    # Validation Loop
    # -------------------
    model.eval()
    r2_test, bcd_acc_test, pooled_r2_test = [], [], []

    for pid in test_pids:
        otu_list, data = data_all[pid]
        otu_list = otu_list[:test_otu_count]
        data = data[:test_otu_count]

        num_nodes = data.shape[0]
        edges_test = list(itertools.product(range(num_nodes), repeat=2))
        edge_index_test = torch.tensor(edges_test, dtype=torch.long).t().contiguous()
        sent_emb = sent_embeddings_all[pid][:test_otu_count].to(torch.float32)

        _, loss, bcd_mean, _, physics_loss, mse_loss, _, _, r2_global, sent_recon_loss, _, _, pooled_r2, comp_r2= model(
            data, edge_index_test, t=t, sentence_embeddings=sent_emb, comm_signature=comm_sig_dict[pid]
        )

        bcd_acc_test.append(1 - bcd_mean.item())
        r2_test.append(r2_global)
        pooled_r2_test.append(pooled_r2)

    test_bcd_acc = np.mean(bcd_acc_test)
    test_r2_mean = np.mean(r2_test)
    test_pooled_r2_mean = np.mean(pooled_r2_test)

    # -------------------
    # Store fold results
    # -------------------
    cv_results["train_bcd"].append(train_bcd_acc)
    cv_results["train_r2"].append(train_r2)
    cv_results["train_r2_pooled"].append(train_pooled_r2)
    cv_results["val_bcd"].append(test_bcd_acc)
    cv_results["val_r2"].append(test_r2_mean)
    cv_results["val_r2_pooled"].append(test_pooled_r2_mean)

    print(f"Fold {fold+1} Summary:")
    print(f"  Train BCD Acc: {train_bcd_acc:.4f}, Train R2: {train_r2:.4f}, Train pooled r2: {train_pooled_r2}")
    print(f"  Val  BCD Acc: {test_bcd_acc:.4f}, Val  R2: {test_r2_mean:.4f}, Val pooled r2: {test_pooled_r2_mean}")

    # Hold out set
    model.eval()
    r2_h, bcd_acc_h, pooled_r2_h = [], [], []
    pred_community_abundances = []
    real_comm_abundances = []

    comm_sig_dict = {}
    for pid in all_pid_h:
        otu_sig = get_comm_signature_grier_all_test(
            data_all_h[pid][0][:h_otu_count],
            data_all_h[pid][1][:h_otu_count].clone().detach().cpu().numpy()
        )
        comm_sig_dict[pid] = otu_sig

    for pid in all_pid_h:
        otu_list, data = data_all_h[pid]
        otu_list = otu_list[:h_otu_count]
        data = data[:h_otu_count]
        data_og = data.clone().detach()

        num_nodes = data.shape[0]
        edges_test = list(itertools.product(range(num_nodes), repeat=2))
        edge_index_test = torch.tensor(edges_test, dtype=torch.long).t().contiguous()
        sent_emb = sent_embeddings_all[pid][:test_otu_count].to(torch.float32)

        reconstructed_data, loss, bcd_mean, _, physics_loss, mse_loss, _, _, r2_global, sent_recon_loss, _, _, pooled_r2, comp_r2 = model(
            data, edge_index_test, t=t, sentence_embeddings=sent_emb, comm_signature=comm_sig_dict[pid]
        )
        pred_community_abundances.append(reconstructed_data)
        real_comm_abundances.append(data_og)

        bcd_acc_h.append(1 - bcd_mean.item())
        r2_h.append(r2_global)
        pooled_r2_h.append(pooled_r2)

    h_bcd_acc = np.mean(bcd_acc_h)
    h_r2_mean = np.mean(r2_h)
    h_pooled_r2_mean = np.mean(pooled_r2_h)

    # Calculate pearson r2
    true_all = []
    pred_all = []

    for real, pred in zip(real_comm_abundances, pred_community_abundances):
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

    # -------------------
    # Store fold results
    # -------------------
    cv_results["test_bcd"].append(h_bcd_acc)
    cv_results["test_r2"].append(h_r2_mean)
    cv_results["test_r2_pooled"].append(h_pooled_r2_mean)
    cv_results["test_r2_pearson"].append(pearson_r2_global)


    print(f"Fold {fold+1} Summary:")
    print(f"  Test  BCD Acc: {h_bcd_acc:.4f}, Test  R2: {h_r2_mean:.4f}, Test  pooled R2: {h_pooled_r2_mean:.4f}")

# -----------------------
#  Final CV Summary
# -----------------------
print("\n===== Cross-Validation Summary =====")
for metric, values in cv_results.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
