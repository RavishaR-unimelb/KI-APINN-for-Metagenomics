import pandas as pd
import re
import ast
from itertools import combinations

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


dataname = "M3_feces_L5"
folder = "M3_F4_final"
num = 25
timesteps = 332
df_edges = pd.read_csv("MetagenomicKG/KG_edges.tsv", sep="\t")
df_nodes = pd.read_csv("MetagenomicKG/KG_nodes.tsv", sep='\t')
df_otus = pd.read_csv(f"Results/{folder}/{dataname}_AnchorGNN_{num}_{timesteps}_sent_True_temp_True_ta_edge_importances.csv", sep=',')

#num_edges = int(num*num*0.2)
num_edges = 50
print(df_edges.head(num_edges))
print(df_edges["predicate"].unique())

print(df_nodes.head(num_edges))
print(df_nodes['node_type'].unique())

df_otus = df_otus.reindex(df_otus['importance'].abs().sort_values(ascending=False).index)
df_otus = df_otus.head(num_edges)
print(f'The df we are considering...')
print(df_otus)

# filter otu data
df_otus["source_family"] = df_otus["source"].apply(extract_family)
df_otus["target_family"] = df_otus["target"].apply(extract_family)
otu_families = pd.unique(df_otus[["source_family", "target_family"]].values.ravel())
print("Unique OTU families:", otu_families)

otu_pairs = set()
for _, row in df_otus.iterrows():
    fam1 = row["source_family"]
    fam2 = row["target_family"]
    if (fam1 is not None) and (fam2 is not None):
        otu_pairs.add(tuple(sorted([fam1, fam2])))

# match
string_filter = "('rank', 'family')"
matches = match_level_to_nodes(otu_families, df_nodes, string_filter)
matched_node_ids = [node_id for fam, node_id in matches]
print("Matched node IDs:", matched_node_ids)

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
print(f'sub_edges:{sub_edges}')
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
        if tuple(sorted([m1, m2])) in otu_pairs:
            edges.append({
                "source": m1,
                "target": m2,
                "predicate": disease
            })
cooccurrence_df = pd.DataFrame(edges)

# Optional: remove duplicate undirected edges
cooccurrence_df = pd.DataFrame(
    { 
        "source": cooccurrence_df[["source", "target"]].min(axis=1),
        "target": cooccurrence_df[["source", "target"]].max(axis=1),
        "predicate": cooccurrence_df["predicate"]
    }
).drop_duplicates()

print(cooccurrence_df)
cooccurrence_df.to_csv(f"Results/{folder}/{dataname}_{num}_kg_top{num_edges}.csv")


