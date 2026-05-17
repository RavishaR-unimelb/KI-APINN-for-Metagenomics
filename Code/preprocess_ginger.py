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
name = 'G_rhiz_CK' 
N = 150 
time_points = 7
df_raw = pd.read_csv(f'Data/ginger_data/{name}.csv', sep=",")
df_otu_info = pd.read_csv(f'Data/ginger_data/ginger_taxonomy.tsv', sep='\t')


# Filter samples based on the participants
merged_df = (
    df_otu_info[["Feature ID", "Taxon"]]  # only the join + group key
    .merge(df_raw, on="Feature ID", how="inner")
)
cols_to_keep = ['Taxon'] + [f"{name}_T{i}" for i in range(0, 7)] 
merged_df = merged_df[cols_to_keep]
merged_df = merged_df.rename(columns={'Taxon': 'OTU ID'})
merged_df['OTU ID'] = merged_df['OTU ID'].str.replace(r'D_\d__', lambda m: {
    'D_0__': 'd__', 'D_1__': 'p__', 'D_2__': 'c__',
    'D_3__': 'o__', 'D_4__': 'f__', 'D_5__': 'g__', 'D_6__': 's__'
    }[m.group(0)], regex=True)
print(merged_df.columns)

# Make L5
merged_df['OTU ID'] = merged_df['OTU ID'].str.extract(r'(.*f__[^;]*)')
merged_df = merged_df.dropna(subset=['OTU ID'])
family_level_df = (
    merged_df
    .groupby('OTU ID', as_index=False)
    .sum(numeric_only=True)
)
print(family_level_df.columns)
print(f'L5: {family_level_df}')

cols = family_level_df.columns.tolist()         # Get current column names as a list
cols[1:] = range(1, len(cols))    # Rename all except first to 1, 2, 3, ...
family_level_df.columns = cols 
# Preview the result
family_level_df = family_level_df.set_index('OTU ID').apply(pd.to_numeric, errors='coerce')



# Save - Already there
family_level_df.to_csv(f'Data/ginger_data/{name}_preprocessed.csv')

df2 = Dataset(family_level_df)

# Select higher abundance OTUs
df2 = df2.select_by_rank(count=N)
print(f'df2.df: {df2.df}')


# Select time interval
df3 = df2.time_interval(start=0, delta_t=time_points)
print(f'df3.df: {df3.df}')

df3.df.to_csv(f'Data/ginger_data/{name}_preprocessed_ranked_{N}.csv')



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
#torch.save(data_norm, f'preprocessed_data_local_{name}_{N}_{time_points}.pt')

#torch.save(data, f'preprocessed_data_raw_{name}_{N}_{time_points}.pt')
print(f'data_norm:{data_norm}, {data_norm.shape}')
torch.save(data_norm, f'Embeddings/preprocessed_data_global_{name}_{N}_{time_points}.pt')

