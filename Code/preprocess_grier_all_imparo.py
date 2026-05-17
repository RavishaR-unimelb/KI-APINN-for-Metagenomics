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
name = 'THR_L5' # NAS or REC or THR
mode = 'mean'
N = 25
#time_points = 100
df_raw = pd.read_csv(f'Data/raw_grier2018/{name}.csv', sep=",")
df_participant_info = pd.read_csv(f'Data/raw_grier2018/OTU_Table_Summary.csv', sep=',')
save_path = f'Data/Grier_IMPARO/THR/'

print(df_participant_info.columns.tolist())
#print(df_raw)
print(df_raw.columns.tolist())

# Filter samples based on the participants
merged_df = (
    df_participant_info[["SampleID", "ParticipantID"]]  # only the join + group key
    .merge(df_raw, on="SampleID", how="inner")
)

# Create a dictionary of participant-specific DataFrames
participant_dfs = {
    pid: group for pid, group in merged_df.groupby("ParticipantID")
}
participants_list = list(participant_dfs.keys())

# Example: access data for a specific participant
#print(participant_dfs[participants_list[0]].head())
lengths = []
for pid, df_pid in participant_dfs.items():
    lengths.append(len(df_pid))
print(f'lengths:{lengths}')
time_points = min(lengths)
print(f'time_points:{time_points}')
final_data = {}

# Example: save each participant DataFrame to CSV
for pid, df_pid in participant_dfs.items():
    print(f'For participant {pid}...')
    print(f'Number of samples: {len(df_pid)}')
    df_pid = df_pid.drop('ParticipantID', axis=1)

    # Melt the dataframe to long format
    df_melted = df_pid.melt(id_vars=["SampleID"], var_name="OTU ID", value_name="Abundance")

    # Pivot the table to get OTUs as rows and SampleIDs as columns
    df_long = df_melted.pivot(index="OTU ID", columns="SampleID", values="Abundance")

    # Optional: reset index if you want OTUs as a column
    df_long.reset_index(inplace=True)
    
    #df_long = df_long.drop('ParticipantID', axis=1)
    cols = df_long.columns.tolist()         # Get current column names as a list
    cols[1:] = range(1, len(cols))    # Rename all except first to 1, 2, 3, ...
    df_long.columns = cols 
    # Preview the result
    df_long = df_long.set_index('OTU ID').apply(pd.to_numeric, errors='coerce')
    print(f'df_long:{df_long}')

    #print(df_long.head())
    df2 = Dataset(df_long)
    # Select higher abundance OTUs
    df2 = df2.select_by_rank(count=N)
    #print(f'df2.df: {df2.df}')
    # Select time interval
    df3 = df2.time_interval(start=0, delta_t=time_points)
    df3.df.index.name = ""
    print(f'df3.df: {df3.df}')
    
    df3.df.to_csv(f'{save_path}{name}_{N}_{time_points}_{pid}.csv')
    torch.save(final_data, save_path+f'{name}_{N}_{time_points}.pt')

