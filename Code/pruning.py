import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import itertools
import torch
import random

# Graph pruning methods

# Anchors
def abundance_based_anchors(num_otus, x, encoded, top_perc, mean_abundance):
    # Create anchors based on mean abundance
    k = int(num_otus * (top_perc / 100))
    print(f'Number of anchors: {k} out of {num_otus}')

    _, anchor_idx = torch.topk(mean_abundance, k=k, largest=True)
    anchors = anchor_idx.tolist()
    all_nodes = list(range(num_otus))
    non_anchors = [i for i in all_nodes if i not in anchors]

    return anchors, non_anchors

def correlation_based_anchors(num_otus, x, encoded, top_perc, eps=1e-8):
    # number of anchors
    k = int(num_otus * (top_perc / 100.0))
    k = max(1, min(k, num_otus))  # safety
    print(f'Number of anchors: {k} out of {num_otus}')

    # --- compute correlation matrix between OTUs ---
    # centre along features
    x_c = x - x.mean(dim=1, keepdim=True)      # [N, F]
    # covariance
    cov = x_c @ x_c.T                          # [N, N]
    # std dev
    var = (x_c ** 2).sum(dim=1)                # [N]
    std = torch.sqrt(var + eps)               # [N]
    # Pearson correlation
    corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + eps)  # [N, N]
    corr.fill_diagonal_(0.0)                  # ignore self-correlation
    # --- node "strength" = sum of absolute correlations to all others ---
    strength = corr.abs().sum(dim=1)          # [N]
    # pick top-k OTUs by strength
    _, anchor_idx = torch.topk(strength, k=k, largest=True)
    anchors = anchor_idx.tolist()
    all_nodes = list(range(num_otus))
    non_anchors = [i for i in all_nodes if i not in anchors]

    return anchors, non_anchors

def random_anchors(num_otus, x, encoded, top_perc):
    # Create anchors randomly
    if num_otus < 5:
        k = 1
    else:
        k = int(num_otus * (top_perc / 100))
    print(f'Number of anchors: {k} out of {num_otus}')

    perm = torch.randperm(num_otus)
    anchor_idx = perm[:k]   
    anchors = anchor_idx.tolist()
    all_nodes = list(range(num_otus))
    non_anchors = [i for i in all_nodes if i not in anchors]

    return anchors, non_anchors

def mlp_based_anchors():
    return

# Edge combining methods
def all_non_anchors_to_all_anchors(anchors, non_anchors):
    # Method 1: connecting all non-anchors to all anchors
    edges = []
    # 1) anchor–anchor edges (like a fully connected subgraph on anchors)
    # if you want to include self-loops, keep product(anchors, anchors) as is
    edges.extend(itertools.product(anchors, anchors))
    # 2) non-anchor → anchor edges
    edges.extend(itertools.product(non_anchors, anchors))
    # 3) anchor → non-anchor edges (for a symmetric directed graph as before)
    edges.extend(itertools.product(anchors, non_anchors))
    # Convert to tensor [2, E]
    pruned_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return pruned_edge_index

def top_r_abundance_anchors_to_non_anchors(anchors, non_anchors, mean_abundance, r):

    # Method 2: connecting non-anchors to top-r anchors
    k = anchors.numel()
    N_non = non_anchors.numel()
    # mean abundance of anchors and non-anchors
    mean_anchors = mean_abundance[anchors]       # [k]
    mean_non = mean_abundance[non_anchors]       # [N_non]
    # |mean_non[i] - mean_anchors[j]| for all i, j
    # shape: [N_non, k]
    diff = (mean_non.unsqueeze(1) - mean_anchors.unsqueeze(0)).abs()
    # for each non-anchor: indices of r closest anchors
    r = 1
    _, top_r_idx = diff.topk(r, dim=1, largest=False)  # smallest differences
    # map to actual node indices
    top_r_anchors = anchors[top_r_idx]  # [N_non, r]
    # ----- anchor–anchor edges -----
    aa_src = anchors.repeat_interleave(k)
    aa_dst = anchors.repeat(k)
    # ----- non-anchor -> top-r anchors -----
    na_src = non_anchors.unsqueeze(1).expand(-1, r).reshape(-1)  # [N_non * r]
    na_dst = top_r_anchors.reshape(-1)                           # [N_non * r]
    # undirected: add reverse edges
    src = torch.cat([aa_src, na_src, aa_dst, na_dst], dim=0)
    dst = torch.cat([aa_dst, na_dst, aa_src, na_src], dim=0)
    pruned_edge_index = torch.stack([src, dst], dim=0)
    pruned_edge_index = torch.unique(pruned_edge_index, dim=1)

    return

def top_r_correlation_anchors_to_non_anchors():
    return

# Main
def get_pruned_graph(edge_index, num_otus, x, encoded, prune_type='abundance', top_perc=20, r=1): 

    mean_abundance = x.mean(dim=1)
    # Get anchors
    anchors, non_anchors = abundance_based_anchors(num_otus, x, encoded, top_perc, mean_abundance)
    #anchors, non_anchors = random_anchors(num_otus, x, encoded, top_perc)
    print(f'anchors:{anchors}')
    print(f'non anchors:{non_anchors}')

    # Prune
    pruned_edge_index = all_non_anchors_to_all_anchors(anchors, non_anchors)
    #pruned_edge_index = top_r_abundance_anchors_to_non_anchors(anchors, non_anchors, mean_abundance, r)
    #pruned_edge_index = top_r_correlation_anchors_to_non_anchors(anchors, non_anchors, mean_abundance, r)
    print(f'pruned edge index shape:{pruned_edge_index.shape}')

    return pruned_edge_index


    
    


