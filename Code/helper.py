import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import numpy as np

def plot_true_vs_predicted(x_real, x_reconstructed, r2, mse, info, results_path):
    x_real = x_real.cpu().numpy()
    x_reconstructed = x_reconstructed.cpu().numpy()
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_real, x_reconstructed, s=30, edgecolor='white', alpha=0.8)
    plt.plot([0, max(x_real.max(), x_reconstructed.max())],
            [0, max(x_real.max(), x_reconstructed.max())],
            'k--', linewidth=1)

    # Add text box with metrics
    plt.text(r2, mse, f'$R^2$ = {r2:.4f}\nMSE = {mse:.4f}',
            horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes,
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    # Style
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("OTU Abundance Reconstruction")
    plt.tight_layout()
    plt.savefig(results_path+f"OTU_true_vs_prediction_{info}.png")


def sample_communities(num_communities, otu_count, num_otus_all, used_subsets=None):
    communities = []
    used_subsets = used_subsets or set()

    while len(communities) < num_communities:
        otu_ids = tuple(sorted(random.sample(range(num_otus_all), otu_count)))
        if otu_ids not in used_subsets:
            communities.append(otu_ids)
            used_subsets.add(otu_ids)
    return communities, used_subsets

def sample_communities_from_clustering(
    otu_time_series,                 # shape: (num_otus, num_timesteps)
    num_communities,                # number of communities to return
    max_cluster_size=10,           # max OTUs per community
    initial_clusters=10,           # KMeans cluster count (should be >= num_communities)
    used_subsets=None,             # optional: track reused subsets
    seed=None                      # for reproducibility
):


    used_subsets = used_subsets or set()
    communities = []

    # Cluster OTUs
    kmeans = KMeans(n_clusters=initial_clusters)
    labels = kmeans.fit_predict(otu_time_series)

    # Group OTUs by cluster label
    clusters = defaultdict(list)
    for otu_idx, label in enumerate(labels):
        clusters[label].append(otu_idx)

    # Enforce max_cluster_size
    candidate_communities = []
    for cluster in clusters.values():
        if len(cluster) <= max_cluster_size:
            candidate_communities.append(tuple(sorted(cluster)))
        else:
            # Split large cluster into smaller sub-communities
            for i in range(0, len(cluster), max_cluster_size):
                chunk = cluster[i:i+max_cluster_size]
                candidate_communities.append(tuple(sorted(chunk)))

    # Sample from candidate communities, avoiding duplicates
    for comm in candidate_communities:
        if len(communities) >= num_communities:
            break
        if comm not in used_subsets:
            communities.append(comm)
            used_subsets.add(comm)

    return communities, used_subsets

def sample_predefined_communities(data, train_threshold, test_threshold, flip=False):
    train_data = []
    test_data = []
    used_comm = set()

    # Create train data
    for i, (k, v) in enumerate(data.items()):
        if v.shape[0] <= train_threshold:
            train_data.append(i)
            used_comm.add(i)
    # Create test data
    if train_threshold == test_threshold:
        N = int(len(train_data)*0.2)
        test_data = train_data[:N]
        train_data = train_data[N:]
    else:
        for j, (k, v) in enumerate(data.items()):
            if v.shape[0] <= test_threshold:
                if j not in used_comm:
                    test_data.append(j)
    if flip:
        return test_data, train_data
    else:
        return train_data, test_data