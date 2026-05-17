from paper2.dataset import Dataset
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf
import torch
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import torch.nn as nn
from Bio import Entrez
from time import sleep

# Set Pandas options to display full content
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.max_rows', None)           # Show all rows (optional)
pd.set_option('display.max_colwidth', None)       # Show full content in each cell
pd.set_option('display.expand_frame_repr', False) # Prevent wrapping to multiple lines


# Load data and preprocess
name = 'NAS_L5' # NAS or REC or THR
mode = 'mean'
N = 30
#time_points = 100
df_raw = pd.read_csv(f'Data/raw_grier2018/{name}.csv', sep=",")
df_participant_info = pd.read_csv(f'Data/raw_grier2018/OTU_Table_Summary.csv', sep=',')
#print(df_participant_info)
print(df_participant_info.columns.tolist())
#print(df_raw)
print(df_raw.columns.tolist())

# Filter samples based on the participants
merged_df = (
    df_participant_info[["SampleID", "ParticipantID"]]  # only the join + group key
    .merge(df_raw, on="SampleID", how="inner")
)
# Select the participant with the highest samples
participant_counts = merged_df["ParticipantID"].value_counts()
top_participant = participant_counts.idxmax()
print(f'top_participant: {top_participant}')
time_points = participant_counts.max()
print(f'Number of time points: {time_points}')
top_participant_df = merged_df[merged_df["ParticipantID"] == top_participant]
print(f'top_participant_df:{top_participant_df}')


df_raw = top_participant_df

# Melt the dataframe to long format
df_melted = df_raw.melt(id_vars=["SampleID"], var_name="OTU ID", value_name="Abundance")

# Pivot the table to get OTUs as rows and SampleIDs as columns
df_long = df_melted.pivot(index="OTU ID", columns="SampleID", values="Abundance")

# Optional: reset index if you want OTUs as a column
df_long.reset_index(inplace=True)
#print(df_long.columns.tolist())
#df_long = df_long.drop('SampleID', axis=1)
cols = df_long.columns.tolist()         # Get current column names as a list
cols[1:] = range(1, len(cols))    # Rename all except first to 1, 2, 3, ...
df_long.columns = cols 
# Preview the result
df_long = df_long.set_index('OTU ID').apply(pd.to_numeric, errors='coerce')

print(df_long.head())

# Save
df_long.to_csv(f'Data/raw_grier2018/preprocessed_{name}.csv')

df2 = Dataset(df_long)

# Select higher abundance OTUs
df2 = df2.select_by_rank(count=N)
print(f'df2.df: {df2.df}')


# Select time interval
df3 = df2.time_interval(start=0, delta_t=time_points)
print(f'df3.df: {df3.df}')


#df3.visualise(f'abundance_{name}_{N}_{time_points}.jpg')
########################################################

# Convert data to tensors
data = torch.tensor(df3.df.values, dtype=torch.float32, requires_grad=True)
print(f'data: {data}')
print(f'data shape={data.shape}')

# Global normalization
data_min = torch.min(data)
data_max = torch.max(data)
data_norm = (data - data_min) / (data_max - data_min + 1e-8)

print(f'Global min: {data_min}, max: {data_max}')

print(f'Normalized data shape: {data_norm.shape}')

# Save
print(f'data_norm:{data_norm}, {data_norm.shape}')
torch.save(data_norm, f'Embeddings/preprocessed_data_global_{name}_{N}_{time_points}.pt')

