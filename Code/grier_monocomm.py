import pandas as pd
import psutil, os, random, itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from model_v7_final import PIGNN 
from signatures import get_comm_signature_grier_all_test
from scipy.stats import linregress

print(f'Grier train only monocomm for multiple comms')

# -----------------------
#  Reproducibility
# -----------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# -----------------------
#  Config
# -----------------------
dataname = 'NAS_L5'
gnn_type = 'AnchorGNN' #changed
time_steps = 7
num_epochs = 200
N_list = [5, 10, 15, 20, 25]
comms_stats_save_name = f'{dataname}_base'


preprocessed_data_path = f'Embeddings/preprocessed_data_global_{dataname}_train_test_{time_steps}.pt'
embeddings_path = f'Embeddings/final_otu_embeddings_titles_and_abstracts_{dataname}_{time_steps}_train_test.pt'
results_path = 'Results/'

is_aug = True
stats_only = False
include_sent_emb = False #changed
temporal = True
attention_fusion = True
use_comm_signature = True

# -----------------------
#  Load Data
# -----------------------
data_final = torch.load(preprocessed_data_path)
sent_embeddings_final = torch.load(embeddings_path)
all_pid_final = list(data_final.keys())
print(f'Total number of communities: {len(all_pid_final)}')


data_all = data_final
sent_embeddings_all = sent_embeddings_final
all_pid = all_pid_final
print(f'Total number of train communities (only train): {len(all_pid)}')

results = []
for N in N_list:
    print(f'Running Grier {dataname} ...')
    print(f'training only for monocomm definition for N: {N}')

    # PINN input
    t = torch.linspace(1, time_steps, time_steps).unsqueeze(1).requires_grad_(True)




    # -------------------
    # Train Loop
    # -------------------
    stats_communities = {i: {"losses": [], "bcd_means": [], "r2_global_per_epoch": [], "pooled_r2_per_epoch": [], "comp_r2_per_epoch": []}
                            for i in range(len(all_pid))}

    print("Training...")


    for i, pid in enumerate(all_pid):
        # Slice data accroding to comm size
        community = data_all[pid][1][:N]
        otu_sig = get_comm_signature_grier_all_test(
                data_all[pid][0][:N],
                data_all[pid][1][:N].clone().detach().cpu().numpy()
            )
        sent_emb = sent_embeddings_all[pid][:N].to(torch.float32)

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

        for epoch in range(num_epochs):
            model.train()
            print(f'Epoch {epoch} ===========')

            optimizer.zero_grad()
            num_nodes = community.shape[0]
            edges = list(itertools.product(range(num_nodes), repeat=2))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            reconstructed_data, loss, bcd_mean, _, physics_loss, mse_loss, _, _, r2_global, sent_recon_loss, _, _, pooled_r2, comp_r2 = model(
                community, edge_index, t=t, sentence_embeddings=sent_emb, comm_signature=otu_sig)

            stats_communities[i]["losses"].append(loss.item())
            stats_communities[i]["bcd_means"].append(bcd_mean.item())
            stats_communities[i]["r2_global_per_epoch"].append(r2_global)
            stats_communities[i]["pooled_r2_per_epoch"].append(pooled_r2)
            stats_communities[i]["comp_r2_per_epoch"].append(comp_r2)


            loss.backward(retain_graph=True)
            optimizer.step()



    # Evaluate on train                
    #train_bcd = np.mean([1 - stats_communities[i]["bcd_means"][-1] for i in range(len(all_pid))])
    #train_r2 = np.mean([stats_communities[i]["r2_global_per_epoch"][-1] for i in range(len(all_pid))])
    #train_pooled_r2 = np.mean([stats_communities[i]["pooled_r2_per_epoch"][-1] for i in range(len(all_pid))])
    #train_comp_r2 = np.mean([stats_communities[i]["comp_r2_per_epoch"][-1] for i in range(len(all_pid))])

    # Collect values first
    train_bcd_vals = [1 - stats_communities[i]["bcd_means"][-1] for i in range(len(all_pid))]
    train_r2_vals = [stats_communities[i]["r2_global_per_epoch"][-1] for i in range(len(all_pid))]
    train_pooled_r2_vals = [stats_communities[i]["pooled_r2_per_epoch"][-1] for i in range(len(all_pid))]
    train_comp_r2_vals = [stats_communities[i]["comp_r2_per_epoch"][-1] for i in range(len(all_pid))]

    # Compute mean
    train_bcd = np.mean(train_bcd_vals)
    train_r2 = np.mean(train_r2_vals)
    train_pooled_r2 = np.mean(train_pooled_r2_vals)
    train_comp_r2 = np.mean(train_comp_r2_vals)

    # Compute std
    train_bcd_std = np.std(train_bcd_vals)
    train_r2_std = np.std(train_r2_vals)
    train_pooled_r2_std = np.std(train_pooled_r2_vals)
    train_comp_r2_std = np.std(train_comp_r2_vals)
    #print(f"For N: {N}, On average --> Train BCD={train_bcd:.4f}, Train R²={train_r2:.4f}, pooled r2:{train_pooled_r2}, comp r2:{train_comp_r2}")
    print(
        f"For N: {N}, On average --> "
        f"Train BCD={train_bcd:.4f} ± {train_bcd_std:.4f}, "
        f"Train R²={train_r2:.4f} ± {train_r2_std:.4f}, "
        f"Pooled R²={train_pooled_r2:.4f} ± {train_pooled_r2_std:.4f}, "
        f"Comp R²={train_comp_r2:.4f} ± {train_comp_r2_std:.4f}")
    
    for i, pid in enumerate(all_pid):
        results.append({
            "N": N,
            "participant": pid,
            "bcd": 1 - stats_communities[i]["bcd_means"][-1],
            "r2": stats_communities[i]["r2_global_per_epoch"][-1],
            "pooled_r2": stats_communities[i]["pooled_r2_per_epoch"][-1],
            "comp_r2": stats_communities[i]["comp_r2_per_epoch"][-1]
        })

df_results = pd.DataFrame(results)
df_results.to_csv(results_path+f"grier_results_per_participant_{comms_stats_save_name}.csv", index=False)