import pandas as pd
import torch
import csv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import torch.nn as nn
import json
import pickle
from paper2.dataset import Dataset
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf
from Bio import Entrez
from time import sleep
import os

folder_path_train = "Data/Simulated_data/train/"
folder_path_test = "Data/Simulated_data/test/"

# Get all CSV filenames in the folder
df_train_list = [f for f in os.listdir(folder_path_train) if f.endswith(".csv")]
df_test_list = [f for f in os.listdir(folder_path_test) if f.endswith(".csv")]

df_train_dict = {}
df_test_dict = {}

for train_name in df_train_list:
    df = pd.read_csv(f'Data/Simulated_data/train/{train_name}')
    # preprocess
    tensor_list = []
    for exp, group in df.groupby("Experiments"):
        group = group.sort_values("Time")  # ensure time order
        species_data = group.drop(columns=["Experiments", "Time"]).values.T  # shape: (N_species x timepoints)
        tensor = torch.tensor(species_data, dtype=torch.float32)
        # Global normalization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        mod_tensor = torch.tensor(normalized_tensor, dtype=torch.float32)
        tensor_list.append(mod_tensor)

    for i, item in enumerate(tensor_list):
        df_train_dict[f'{train_name}_{i}'] = item

for test_name in df_test_list:
    df = pd.read_csv(f'Data/Simulated_data/test/{test_name}')
    # preprocess
    tensor_list = []
    for exp, group in df.groupby("Experiments"):
        group = group.sort_values("Time")  # ensure time order
        species_data = group.drop(columns=["Experiments", "Time"]).values.T  # shape: (N_species x timepoints)
        tensor = torch.tensor(species_data, dtype=torch.float32)
        # Global normalization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        mod_tensor = torch.tensor(normalized_tensor, dtype=torch.float32)
        tensor_list.append(mod_tensor)

    for i, item in enumerate(tensor_list):
        df_test_dict[f'{test_name}_{i}'] = item


community_tensors = {}
community_tensors['train'] = df_train_dict
print(f'train data: {df_train_dict}')
community_tensors['test'] = df_test_dict

torch.save(community_tensors, f'Embeddings/preprocesses_data_global_simulated.pt')



