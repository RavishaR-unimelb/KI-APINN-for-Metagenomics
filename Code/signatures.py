from paper2.dataset import Dataset
import torch
import pandas as pd
import numpy as np
from itertools import combinations
import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# Set Pandas options to display full content
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.max_rows', None)           # Show all rows (optional)
pd.set_option('display.max_colwidth', None)       # Show full content in each cell
pd.set_option('display.expand_frame_repr', False) # Prevent wrapping to multiple lines

# Function to extract taxonomic levels from the OTU string
def extract_tax_levels(tax_str):
    #levels = {'k': 'Kingdom', 'p': 'Phylum', 'c': 'Class', 'o': 'Order', 'f': 'Family', 'g': 'Genus'}
    levels = {'k': 'Kingdom', 'p': 'Phylum', 'c': 'Class', 'o': 'Order', 'f': 'Family'}
    parts = tax_str.split(';')
    extracted = {lvl: '' for lvl in levels.values()}
    for part in parts:
        if '__' in part:
            prefix, name = part.split('__', 1)
            if prefix in levels:
                extracted[levels[prefix]] = name if name else 'unclassified'
    return pd.Series(extracted)

def get_community_df(name, N, time_points, ids):
    # Load data and preprocess
    df2 = Dataset.from_csv(f'Data/raw/{name}.txt')
    print(len(df2.df))

    # Select higher abundance OTUs
    df2 = df2.select_by_rank(count=N)
    print(f'df2.df: {df2.df}')
    # Select time interval
    df3 = df2.time_interval(start=0, delta_t=time_points)
    print(f'df3.df: {df3.df}')
    print(f'Shape of df3.df: {df3.df.shape}')

    # Filter by otu ids
    comm = df3.df.iloc[ids]

    return comm



def get_comm_signature_ginger(name, N, time_points, ids):
    # Load data and preprocess
    df2 = pd.read_csv(f'Data/ginger_data/{name}_preprocessed.csv', index_col="OTU ID", sep=',')
    df2 = Dataset(df2)
    print(len(df2.df))

    # Select higher abundance OTUs
    df2 = df2.select_by_rank(count=N)
    print(f'df2.df: {df2.df}')
    # Select time interval
    df3 = df2.time_interval(start=0, delta_t=time_points)
    print(f'df3.df: {df3.df}')
    print(f'Shape of df3.df: {df3.df.shape}')

    # Filter by otu ids
    df = df3.df.iloc[ids]
    print(f'Required ids: {ids}')

    # Apply the extraction function to index
    tax_df = df.index.to_series().apply(extract_tax_levels)

    # Concatenate the taxonomic levels with the abundance data
    df_full = pd.concat([df, tax_df], axis=1)

    # Optional: move taxonomic columns to the front
    tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    #tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    df_full = df_full[tax_columns + list(df.columns)]
    df = df_full

    # Display
    print(df.head())

    # Step 1: Group by 'Order' and sum abundance across OTUs
    order_abundance = df.groupby('Order').sum()
    # Step 2 (Optional): Normalize abundance per timestep (column-wise)
    col_sums = order_abundance.sum(axis=0)
    order_abundance_normalized = order_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)
    #order_abundance_normalized = order_abundance.div(order_abundance.sum(axis=0), axis=1)
    # Output
    print("Raw order-level abundance:")
    print(order_abundance)
    print("\nNormalized order-level abundance (relative):")
    print(order_abundance_normalized)

    community_signature = order_abundance_normalized.mean(axis=1).values
    print(f'community_signature: {community_signature}')
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)
    print(f'community_signature tensor: {community_signature.shape}, {community_signature}')


    '''
    #For pooling using adding - skip for mean
    community_signature_add = order_abundance.mean(axis=1).values
    print(f'community_signature_add: {community_signature_add}')
    community_signature_add = torch.tensor(community_signature_add, dtype=torch.float32, requires_grad=True)
    print(f'community_signature_add tensor: {community_signature_add.shape}, {community_signature_add}')
    pooled_signature = community_signature_add.sum().unsqueeze(0) #Try adding instead
    '''

    # Pooling
    pooled_signature = community_signature.mean().unsqueeze(0) #mean
    print(f'pooled_signature:{pooled_signature.shape}, {pooled_signature}')

    return pooled_signature



def get_comm_signature(name, N, time_points, ids):
    print(f'Required ids: {ids}')
    df = get_community_df(name, N, time_points, ids)
    # Apply the extraction function to index
    tax_df = df.index.to_series().apply(extract_tax_levels)

    # Concatenate the taxonomic levels with the abundance data
    df_full = pd.concat([df, tax_df], axis=1)

    # Optional: move taxonomic columns to the front
    #tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    df_full = df_full[tax_columns + list(df.columns)]
    df = df_full

    # Display
    print(df.head())

    # Step 1: Group by 'Order' and sum abundance across OTUs
    order_abundance = df.groupby('Order').sum()
    # Step 2 (Optional): Normalize abundance per timestep (column-wise)
    col_sums = order_abundance.sum(axis=0)
    order_abundance_normalized = order_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)
    #order_abundance_normalized = order_abundance.div(order_abundance.sum(axis=0), axis=1)
    # Output
    print("Raw order-level abundance:")
    print(order_abundance)
    print("\nNormalized order-level abundance (relative):")
    print(order_abundance_normalized)

    community_signature = order_abundance_normalized.mean(axis=1).values
    print(f'community_signature: {community_signature}')
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)
    print(f'community_signature tensor: {community_signature.shape}, {community_signature}')


    '''
    #For pooling using adding - skip for mean
    community_signature_add = order_abundance.mean(axis=1).values
    print(f'community_signature_add: {community_signature_add}')
    community_signature_add = torch.tensor(community_signature_add, dtype=torch.float32, requires_grad=True)
    print(f'community_signature_add tensor: {community_signature_add.shape}, {community_signature_add}')
    pooled_signature = community_signature_add.sum().unsqueeze(0) #Try adding instead
    '''

    # Pooling
    pooled_signature = community_signature.mean().unsqueeze(0) #mean
    print(f'pooled_signature:{pooled_signature.shape}, {pooled_signature}')

    return pooled_signature

def get_comm_signature_all_levels_old(name, N, time_points, ids):
    print(f'Required ids: {ids}')
    df = get_community_df(name, N, time_points, ids)

    # Extract taxonomy levels
    tax_df = df.index.to_series().apply(extract_tax_levels)
    df_full = pd.concat([df, tax_df], axis=1)

    # Define taxonomic levels to use
    tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    df_full = df_full[tax_columns + list(df.columns)]
    
    print(df_full.head())  # Check structure

    all_signatures = []

    for level in tax_columns:
        print(f"\nProcessing level: {level}")
        level_abundance = df_full.groupby(level).sum()

        # Optional normalization
        col_sums = level_abundance.sum(axis=0)
        level_abundance = level_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)

        # Average across timepoints per taxon at this level
        level_signature = level_abundance.mean(axis=1).values
        print(f"{level} signature: {level_signature}")
        all_signatures.append(np.array([level_signature.mean()]))

    # Concatenate all levels’ signatures
    community_signature = np.concatenate(all_signatures)
    print(f'Final community signature shape: {community_signature}')
    
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)

    # Pooling
    #pooled_signature = community_signature.mean().unsqueeze(0)
    pooled_signature = community_signature

    print(f'pooled_signature: {pooled_signature.shape}, {pooled_signature}')

    return pooled_signature

def taxonomic_distance(path1, path2):
    shared = 0
    for a, b in zip(path1, path2):
        if a == b:
            shared += 1
        else:
            break
    return (len(path1) - shared) + (len(path2) - shared)


def get_comm_signature_all_levels(name, N, time_points, ids):
    print(f'Required ids: {ids}')
    df = get_community_df(name, N, time_points, ids)

    # Extract taxonomy levels
    tax_df = df.index.to_series().apply(extract_tax_levels)
    df_full = pd.concat([df, tax_df], axis=1)

    # Define taxonomic levels to use
    tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    df_full = df_full[tax_columns + list(df.columns)]
    
    print(df_full.head())  # Check structure
    stats = get_tree_stats(df_full)
    text = get_otu_name_text_embeddings(df_full)
    print(f'stats:{stats}')
    print(f'text: {text}')

    all_signatures = []

    for level in tax_columns:
        print(f"\nProcessing level: {level}")
        level_abundance = df_full.groupby(level).sum()

        # Optional normalization
        #col_sums = level_abundance.sum(axis=0)
        #level_abundance = level_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)

        # Average across timepoints per taxon at this level
        level_signature = level_abundance.mean(axis=1).values
        print(f"{level} signature: {level_signature}")
        # Scale and capture the distribution
        level_signature =  (level_signature - level_signature.min()) / (level_signature.max() - level_signature.min() + 1e-8)
        all_signatures.append(np.array([level_signature.var()])) # variance - add more statistics

    # Concatenate all levels’ signatures
    community_signature = np.concatenate(all_signatures)
    print(f'Final community signature shape: {community_signature}')
    
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)

    # scale
    stats_scaled = (stats - stats.min()) / (stats.max() - stats.min() + 1e-8)
    # Pooling
    #pooled_signature = community_signature.mean().unsqueeze(0)
    #pooled_signature = torch.cat([community_signature, stats], dim=0)
    #pooled_signature = stats_scaled
    pooled_signature = text.mean(dim=0)

    print(f'pooled_signature: {pooled_signature.shape}, {pooled_signature}')

    return pooled_signature

def get_otu_name_text_embeddings(df):
    otu_paths = df.iloc[:, :5].values.tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = model.tokenizer
    transformer_model = model[0].auto_model
    transformer_model.config.output_attentions = True
    combined_descriptions = list(set(term for sublist in otu_paths for term in sublist))
    print(f'unique terms: {len(combined_descriptions)}')
    embeddings = model.encode(combined_descriptions, convert_to_tensor=True)
    print(f'embeddings shape = {embeddings.shape}')

    return embeddings

def get_tree_stats(df):
    otu_paths = df.iloc[:, :5].values.tolist()  # Only taxonomy levels (kingdom to family)
    print(f'otu_paths:{otu_paths}')

    # Compute pairwise distances
    distances = []
    for i, j in combinations(range(len(otu_paths)), 2):
        d = taxonomic_distance(otu_paths[i], otu_paths[j])
        distances.append(d)

    # Extract genus and family from paths
    genera = {path[5] for path in otu_paths if len(path) > 5}
    families = {path[4] for path in otu_paths if len(path) > 4}

    '''
    return {
        "n_otus": len(otu_paths),
        "avg_taxonomic_distance": np.mean(distances),
        "max_taxonomic_distance": np.max(distances),
        "min_taxonomic_distance": np.min(distances),
        "unique_genera": len(genera),
        "unique_families": len(families)
    }
    '''
    stats_tensor = torch.tensor([
        len(otu_paths),
        np.mean(distances),
        np.max(distances),
        np.min(distances),
        #len(genera),
        len(families)], dtype=torch.float32)
    
    return stats_tensor




def get_comm_signature_baranwal(comm_name, comm_values):
    abbr_to_full_name = {
        "PC": "Bacteroidetes__P. copri",
        "PJ": "Bacteroidetes__P. johnsonii",
        "BV": "Bacteroidetes__B. vulgatus",
        "BF": "Bacteroidetes__B. fragilis",
        "BO": "Bacteroidetes__B. ovatus",
        "BT": "Bacteroidetes__B. thetaiotaomicron",
        "BC": "Bacteroidetes__B. caccae",
        "BY": "Bacteroidetes__B. cellulosilyticus",
        "BU": "Bacteroidetes__B. uniformis",
        "DP": "Proteobacteria__D. piger",
        "BL": "Actinobacteria__B. longum",
        "BA": "Actinobacteria__B. adolescentis",
        "BP": "Actinobacteria__B. pseudocatenulatum",
        "CA": "Actinobacteria__C. aerofaciens",
        "EL": "Actinobacteria__E. lenta",
        "FP": "Firmicutes__F. prausnitzii",
        "CH": "Firmicutes__C. hiranonis",
        "AC": "Firmicutes__A. caccae",
        "BH": "Firmicutes__B. hydrogenotrophica",
        "CG": "Firmicutes__C. asparagiforme",
        "ER": "Firmicutes__E. rectale",
        "RI": "Firmicutes__R. intestinalis",
        "CC": "Firmicutes__C. comes",
        "DL": "Firmicutes__D. longicatena",
        "DF": "Firmicutes__D. formicigenerans",
    }
    otu_names = comm_name.split("-")
    print(f'comm_values shape:{comm_values.shape}')
    df = pd.DataFrame(comm_values.numpy(), columns=["0", "1", "2", "3"])

    otu_long_names = [abbr_to_full_name.get(short) for short in otu_names]
    phylum_names = [name.split("__")[0] if "__" in name else "" for name in otu_long_names]
    species_names = [name.split("__")[1] if " " in name else "" for name in otu_long_names]

    df.insert(0, "OTU", otu_long_names)
    df.insert(1, "Phylum", phylum_names)
    df.insert(2, "Species", species_names)
    print(f'df created for community: {df}')

    ###########################
    # Step 1: Group by 'Phylum' and sum abundance across OTUs
    phylum_abundance = df.groupby('Phylum').sum()

    # Step 2 (Optional): Normalize abundance per timestep (column-wise)
    col_sums = phylum_abundance.sum(axis=0)
    phylum_abundance_normalized = phylum_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)

    # Output
    print("Raw phylum-level abundance:")
    print(phylum_abundance)

    print("\nNormalized phylum-level abundance (relative):")
    print(phylum_abundance_normalized)

    community_signature = phylum_abundance_normalized.mean(axis=1).values
    print(f'community_signature: {community_signature}')
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)
    print(f'community_signature tensor: {community_signature.shape}, {community_signature}')

    # Pooling
    pooled_signature = community_signature.mean().unsqueeze(0)
    print(f'pooled_signature:{pooled_signature.shape}, {pooled_signature}')


    return pooled_signature



# Grier
def extract_tax_levels_grier(tax_str):
    #levels = {'k': 'Kingdom', 'p': 'Phylum', 'c': 'Class', 'o': 'Order', 'f': 'Family', 'g': 'Genus'}
    levels = {'k': 'Kingdom', 'p': 'Phylum', 'c': 'Class', 'o': 'Order', 'f': 'Family'}
    parts = tax_str.split('|')
    extracted = {lvl: '' for lvl in levels.values()}
    for part in parts:
        if '__' in part:
            prefix, name = part.split('__', 1)
            if prefix in levels:
                extracted[levels[prefix]] = name if name else 'unclassified'
    return pd.Series(extracted)

def get_community_df_grier(name, N, time_points, ids):
    # Load data and preprocess
    df_raw = pd.read_csv(f'Data/raw_grier2018/{name}.csv', sep=",")
    df_melted = df_raw.melt(id_vars=["SampleID"], var_name="OTU ID", value_name="Abundance")

    # Pivot the table to get OTUs as rows and SampleIDs as columns
    df_long = df_melted.pivot(index="OTU ID", columns="SampleID", values="Abundance")

    # Optional: reset index if you want OTUs as a column
    df_long.reset_index(inplace=True)
    cols = df_long.columns.tolist()         # Get current column names as a list
    cols[1:] = range(1, len(cols))    # Rename all except first to 1, 2, 3, ...
    df_long.columns = cols 
    # Preview the result
    df_long = df_long.set_index('OTU ID').apply(pd.to_numeric, errors='coerce')
    print(df_long.head())


    df2 = Dataset(df_long)
    # Select higher abundance OTUs
    df2 = df2.select_by_rank(count=N)
    df3 = df2.time_interval(start=0, delta_t=time_points)

    # Filter by otu ids
    comm = df3.df.iloc[ids]

    return comm


def get_comm_signature_grier(name, N, time_points, ids):
    print(f'Required ids: {ids}')
    df = get_community_df_grier(name, N, time_points, ids)
    # Apply the extraction function to index
    tax_df = df.index.to_series().apply(extract_tax_levels_grier)

    # Concatenate the taxonomic levels with the abundance data
    df_full = pd.concat([df, tax_df], axis=1)

    # Optional: move taxonomic columns to the front
    #tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    df_full = df_full[tax_columns + list(df.columns)]
    df = df_full

    # Step 1: Group by 'Order' and sum abundance across OTUs
    order_abundance = df.groupby('Order').sum()
    # Step 2 (Optional): Normalize abundance per timestep (column-wise)
    col_sums = order_abundance.sum(axis=0)
    order_abundance_normalized = order_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)
    #order_abundance_normalized = order_abundance.div(order_abundance.sum(axis=0), axis=1)
    # Output
    print("Raw order-level abundance:")
    print(order_abundance)
    print("\nNormalized order-level abundance (relative):")
    print(order_abundance_normalized)

    community_signature = order_abundance_normalized.mean(axis=1).values
    print(f'community_signature: {community_signature}')
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)
    print(f'community_signature tensor: {community_signature.shape}, {community_signature}')


    '''
    #For pooling using adding - skip for mean
    community_signature_add = order_abundance.mean(axis=1).values
    print(f'community_signature_add: {community_signature_add}')
    community_signature_add = torch.tensor(community_signature_add, dtype=torch.float32, requires_grad=True)
    print(f'community_signature_add tensor: {community_signature_add.shape}, {community_signature_add}')
    pooled_signature = community_signature_add.sum().unsqueeze(0) #Try adding instead
    '''

    # Pooling
    pooled_signature = community_signature.mean().unsqueeze(0) #mean
    print(f'pooled_signature:{pooled_signature.shape}, {pooled_signature}')

    return pooled_signature

def get_comm_signature_grier_all_test(otu_list, data):
    # Create df
    df = pd.DataFrame(data)
    df.insert(0, "OTU_ID", otu_list)
    print(f'df:{df}')
    # Apply the extraction function to index
    tax_df = df['OTU_ID'].apply(extract_tax_levels_grier)

    # Concatenate the taxonomic levels with the abundance data
    df_full = pd.concat([df, tax_df], axis=1)

    # Optional: move taxonomic columns to the front
    #tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    tax_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family']
    df_full = df_full[tax_columns + list(df.columns)]
    df = df_full

    # Step 1: Group by 'Order' and sum abundance across OTUs
    order_abundance = df.groupby('Order').sum()
    # Step 2 (Optional): Normalize abundance per timestep (column-wise)
    col_sums = order_abundance.sum(axis=0)
    order_abundance_normalized = order_abundance.div(col_sums.where(col_sums != 0, 1), axis=1)
    #order_abundance_normalized = order_abundance.div(order_abundance.sum(axis=0), axis=1)
    # Output
    print("Raw order-level abundance:")
    print(order_abundance)
    print("\nNormalized order-level abundance (relative):")
    print(order_abundance_normalized)

    community_signature = order_abundance_normalized.mean(axis=1).values
    print(f'community_signature: {community_signature}')
    community_signature = torch.tensor(community_signature, dtype=torch.float32, requires_grad=True)
    print(f'community_signature tensor: {community_signature.shape}, {community_signature}')


    '''
    #For pooling using adding - skip for mean
    community_signature_add = order_abundance.mean(axis=1).values
    print(f'community_signature_add: {community_signature_add}')
    community_signature_add = torch.tensor(community_signature_add, dtype=torch.float32, requires_grad=True)
    print(f'community_signature_add tensor: {community_signature_add.shape}, {community_signature_add}')
    pooled_signature = community_signature_add.sum().unsqueeze(0) #Try adding instead
    '''

    # Pooling
    pooled_signature = community_signature.mean().unsqueeze(0) #mean
    print(f'pooled_signature:{pooled_signature.shape}, {pooled_signature}')

    return pooled_signature