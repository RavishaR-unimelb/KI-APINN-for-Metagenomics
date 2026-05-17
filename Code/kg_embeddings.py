import pandas as pd
import re
import ast
from itertools import combinations
from paper2.dataset import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Set Pandas options to display full content
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.max_rows', None)           # Show all rows (optional)
pd.set_option('display.max_colwidth', None)       # Show full content in each cell
pd.set_option('display.expand_frame_repr', True) # Prevent wrapping to multiple lines


def extract_family(s):
    match = re.search(r'f__([\w\-]+)', s)
    return match.group(1) if match else None

def match_level_to_nodes(otu_levels, nodes_df, string_filter):
    matched = []
    matches = 0
    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        names_raw = row["all_names"]
        info = row["description"]

        try:
            names = ast.literal_eval(names_raw) if isinstance(names_raw, str) else names_raw
        except (ValueError, SyntaxError):
            names = []

        for name in names:
            if name is None:
                continue
            for l in otu_levels:
                if l is None:
                    continue
                if (l.lower() == name.lower()) and (string_filter in info):
                    print('-----------------------------')
                    print(f'l:{l.lower()}')
                    print(f'name:{name.lower()}')
                    print(info)
                    matches +=1
                    matched.append((l, node_id))
    print(f'node matches: {matches}')
    return matched

def preprocess_data(path):
    df2 = Dataset.from_csv(path)
    # Select higher abundance OTUs
    N = 30
    df2 = df2.select_by_rank(count=N)
    
    time_points = 100
    df3 = df2.time_interval(start=0, delta_t=time_points)
    new_df = pd.DataFrame({"OTU_name": df3.df.index})
    return new_df

def filter_otus(df, string_filter, df_nodes):
    df["otu_family"] = df["OTU_name"].apply(extract_family)
    otu_families = pd.unique(df[["otu_family"]].values.ravel())
    print("Unique OTU families:", otu_families)

    # match
    matches = match_level_to_nodes(otu_families, df_nodes, string_filter)
    matched_node_ids = [node_id for fam, node_id in matches]
    #print("Matched node IDs:", matched_node_ids)
    return matched_node_ids, df

def average_embeddings(embeddings):
    return torch.tensor(embeddings).mean(dim=0).numpy()

def get_kg_matches(df_edges, df_nodes, df_otus_path, string_filter):
    # filter otu data - customize
    df_otus = preprocess_data(df_otus_path)
    matched_node_ids, df_otus_mod = filter_otus(df_otus, string_filter, df_nodes)
    #print(f'df_otus_mod:{df_otus_mod}')

    # subset
    sub_edges = df_edges[
        df_edges["source_node"].isin(matched_node_ids)|
        df_edges["target_node"].isin(matched_node_ids)
    ]
    edge_filter = ['biolink:superclass_of', 'biolink:subclass_of']
    sub_edges = sub_edges[~sub_edges['predicate'].isin(edge_filter)]
    sub_edges['edge_key'] = sub_edges.apply(
        lambda row: tuple(sorted([row['source_node'], row['target_node']])), axis=1
    )
    sub_edges = sub_edges.drop_duplicates(subset=['edge_key', 'predicate'])
    sub_edges = sub_edges.drop(columns=['edge_key', 'knowledge_source', 'description'])


    # merge
    sub_edges = sub_edges.merge(df_nodes[['node_id', 'all_names']], left_on='source_node', right_on='node_id', how='left')
    sub_edges = sub_edges.rename(columns={'all_names': 'source_name'}).drop(columns='node_id')
    sub_edges = sub_edges.merge(df_nodes[['node_id', 'all_names']], left_on='target_node', right_on='node_id', how='left')
    sub_edges = sub_edges.rename(columns={'all_names': 'target_name'}).drop(columns='node_id')
    print(f'sub_edges:{sub_edges}')
    print(f'length:{len(sub_edges)}')

    # Ensure disease names are consistent (use first in list if multiple)
    sub_edges["disease"] = sub_edges["target_name"].apply(lambda x: eval(x)[0] if isinstance(x, str) and x.startswith("[") else x)

    # Build pairs of microbes per disease
    edges = []
    for disease, group in sub_edges.groupby("disease"):
        microbes = group["source_name"].apply(lambda x: eval(x)[0] if isinstance(x, str) and x.startswith("[") else x).unique()
        for m1, m2 in combinations(microbes, 2):
            edges.append({
                "source": m1,
                "target": m2,
                "predicate": disease
            })
    cooccurrence_df = pd.DataFrame(edges)

    #remove duplicate undirected edges
    cooccurrence_df = pd.DataFrame(
        { 
            "source": cooccurrence_df[["source", "target"]].min(axis=1),
            "target": cooccurrence_df[["source", "target"]].max(axis=1),
            "predicate": cooccurrence_df["predicate"]
        }
    ).drop_duplicates()

    #print(f'cooccurrence_df:{cooccurrence_df}')
    return cooccurrence_df

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1)
    #print(f'embedding size: {embedding.shape}')
    return embedding.squeeze().numpy()

def average_embeddings(embeddings):
    return np.mean(np.vstack(embeddings), axis=0)

def kg_to_text_embeddings(df, model, tokenizer):
    df["interaction_text"] = df["source"] + " " + df["predicate"] + " " + df["target"]
    to_duplicate = df.copy()
    # Swap source and target
    to_duplicate[["source", "target"]] = to_duplicate[["target", "source"]]
    # Optionally update text (simple example)
    to_duplicate["interaction_text"] = to_duplicate["source"] + " " + to_duplicate["predicate"] + " " + to_duplicate["target"]
    # Append duplicated rows to the original DataFrame
    df_bidirectional = pd.concat([df, to_duplicate], ignore_index=True)
    df_bidirectional["node_pair"] = df_bidirectional["source"] + "-" + df_bidirectional["target"]
    df_bidirectional["embedding"] = df_bidirectional["interaction_text"].apply(get_embedding, args=(model, tokenizer))
    '''
    node_embeddings = {}
    for node in pd.unique(df[["source", "target"]].values.ravel()):
        related_rows = df[(df["source"] == node) & (df["target"] == node)]
        combined_embedding = average_embeddings(list(related_rows["embedding"]))
        print(f'length of embedding: {combined_embedding.shape}')
        node_embeddings[node] = combined_embedding

    print(f'node_embeddings:{node_embeddings}')
    '''
    grouped = df_bidirectional.groupby("node_pair")["embedding"].apply(lambda emb_list: average_embeddings(list(emb_list)))
    node_pair_embeddings_df = grouped.reset_index().rename(columns={"embedding": "avg_embedding"})
    node_pair_embeddings_df[["source", "target"]] = node_pair_embeddings_df["node_pair"].str.split("-", expand=True)

    return node_pair_embeddings_df

def get_kg_based_text_embeddings(df_otus_path, otu_list, edge_index):
    df_edges = pd.read_csv("MetagenomicKG/KG_edges.tsv", sep="\t")
    df_nodes = pd.read_csv("MetagenomicKG/KG_nodes.tsv", sep='\t')
    #df_otus_path = f'/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/raw/F4_feces_L5.txt'
    string_filter = "('rank', 'family')"

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )


    cooccurrence_df = get_kg_matches(df_edges, df_nodes, df_otus_path, string_filter)
    embeddings_lookup = kg_to_text_embeddings(cooccurrence_df, model, tokenizer)
    print(f'embeddings_lookup df:{len(embeddings_lookup)}')

    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    edge_embeddings_list = []
    embed_dim = 768
    existing_edge_matches = 0
    no_matches = 0

    global_avg_embedding = np.mean(np.vstack(embeddings_lookup['avg_embedding']), axis=0)
    print(f'global_avg_embedding shape: {global_avg_embedding.shape}')

    for (src, dst) in zip(src_nodes, dst_nodes):
        src_short = extract_family(otu_list[src])
        dst_short = extract_family(otu_list[dst])

        print(f'For {src_short} and {dst_short}--->')

        src_dst_matches = embeddings_lookup[(embeddings_lookup['source'] == src_short) & 
                                            (embeddings_lookup['target'] == dst_short)]


        print(f'matches: {src_dst_matches}')
        print(f'Number of matches:{len(src_dst_matches)}')

        if len(src_dst_matches) > 0:
            edge_embeddings_list.append(src_dst_matches['avg_embedding'].iloc[0]) #imputing
            existing_edge_matches += 1
        else:
            #edge_embeddings_list.append(np.zeros(embed_dim)) 
            edge_embeddings_list.append(global_avg_embedding) #baseline values
            no_matches += 1

    edge_embeddings_tensor = torch.tensor(np.stack(edge_embeddings_list), dtype=torch.torch.float32)
    print(f'edge_embeddings_tensor: {edge_embeddings_tensor}')
    print(f'edge_embeddings_tensor shape: {edge_embeddings_tensor.shape}')
    print(f'existing_edge_matches: {existing_edge_matches}, no_matches:{no_matches}')

    return edge_embeddings_tensor
        


  

    