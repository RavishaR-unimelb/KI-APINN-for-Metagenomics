import csv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import json
import pickle
import torch
from paper2.dataset import Dataset
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf
from Bio import Entrez
from time import sleep

# Config
Entrez.email = "ravisha98rupasinghe@gmail.com"
text_type = 'titles_and_abstracts' #options: titles_only, titles_and_abstracts
dataset = 'G_rhiz_CK'
N = 150 
time_points = 7
mode = 'mean'


def fetch_abstracts(otu, max_results=50):
    handle = Entrez.esearch(db="pubmed", term=f"{otu} AND microbiome", retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    if not ids:
        return ""
    
    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
    papers = Entrez.read(handle)
    
    texts = []
    for article in papers['PubmedArticle']:
        try:
            title = article['MedlineCitation']['Article']['ArticleTitle']
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            full_abstract = " ".join(str(section) for section in abstract)
            texts.append(f"{title}. {full_abstract}")
        except KeyError:
            continue
    #print(f'texts:{texts}')
    return "\n".join(texts)

def fetch_only_titles(otu, max_results=50):
    handle = Entrez.esearch(db="pubmed", term=f"{otu} AND microbiome", retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    if not ids:
        return ""
    
    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
    papers = Entrez.read(handle)
    
    texts = []
    for article in papers['PubmedArticle']:
        try:
            title = article['MedlineCitation']['Article']['ArticleTitle']
            texts.append(f"{title}")
        except KeyError:
            continue
    return "\n".join(texts)


# Preprocess
df_raw = pd.read_csv(f'Data/ginger_data/{dataset}_preprocessed.csv', index_col="OTU ID", sep=',')

df2 = Dataset(df_raw)

# Select higher abundance OTUs
df2 = df2.select_by_rank(count=N)
print(f'df2.df: {df2.df}')


# Select time interval
df3 = df2.time_interval(start=0, delta_t=time_points)
print(f'df3.df: {df3.df}')

# Create a list of terms from the data
data = df3.df
print(f'data: {data}')

# Get the keywords 
# Extract the index (OTU taxonomy strings)
otu_taxonomies = data.index.tolist()

# Create a set to store all unique keywords
otu_keywords = set()

for taxonomy in otu_taxonomies:
    terms = taxonomy.split(';')
    for term in terms:
        if '__' in term:
            keyword = term.split('__')[1].strip()
            if '[' in keyword:
                keyword = keyword.strip("[]")
            if keyword:  # skip empty strings
                otu_keywords.add(keyword)

# Convert set to sorted list
otu_keywords = sorted(otu_keywords)
print(f'otu_keywords={otu_keywords}, length={len(otu_keywords)}')


# Compile dataset
otu_dataset = {}
for key in otu_keywords:
    #print(f"Fetching for {key}...")
    try:
        if text_type == 'titles_only':
            otu_dataset[key] = fetch_only_titles(key, max_results=100)
        else:
            otu_dataset[key] = fetch_abstracts(key, max_results=100)
    except Exception as e:
        print(f"Error fetching for {key}: {e}")
    sleep(1)  # Be respectful to NCBI servers

'''
with open("otu_literature_dataset.json", "w") as f:
    json.dump(otu_dataset, f, indent=2)
'''

#########################################################################################

model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer
transformer_model = model[0].auto_model
transformer_model.config.output_attentions = True


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device={device}')

combined_descriptions = []
otu_keys = otu_dataset.keys()

for key, text in otu_dataset.items():
    combined_description = key + " " + text
    combined_descriptions.append(combined_description)

print(f'Number of keys: {len(otu_keys)}')
print(f'otu_keys={otu_keys}')

embeddings = model.encode(combined_descriptions, convert_to_numpy=True)
print(f'embeddings shape = {embeddings.shape}')

embedding_dict = {sid: emb for sid, emb in zip(otu_keys, embeddings)}
embedding_dict_serializable = {k: v.tolist() for k, v in embedding_dict.items()}

#print(f'embedding_dict_serializable={embedding_dict_serializable}')

'''
with open("embeddings.json", "w") as f:
    json.dump(embedding_dict_serializable, f)
'''
#########################################################################################


embedding_dict = {k: np.array(v) for k, v in embedding_dict_serializable.items()}
embedding_tensor_dict = {k: torch.tensor(v) for k, v in embedding_dict.items()}

otu_to_embedding = {}
for otu_id in df3.df.index:
    taxonomy_names = otu_id.split(';')  # Split the OTU string
    final_embeddings = []

    for item in taxonomy_names:
        if '__' in item:
            _, name = item.split('__', 1)  # Remove the prefix like "k__"
            if name in embedding_tensor_dict:
                final_embeddings.append(embedding_tensor_dict[name])
    
    if final_embeddings:
        if mode == 'mean':
            otu_embedding = torch.stack(final_embeddings, dim=0).mean(dim=0) #Agg using mean
        elif mode == 'sum':
            otu_embedding = torch.stack(final_embeddings, dim=0).sum(dim=0) #Agg using sum
        else:
            otu_embedding = torch.stack(final_embeddings, dim=0).mean(dim=0) #Agg using mean
        print(f'otu_embedding.shape={otu_embedding.shape}')
        otu_to_embedding[otu_id] = otu_embedding
    else:
        print(f"Warning: No matching embeddings for OTU: {otu_id}")

otu_embeddings_tensor = torch.stack(list(otu_to_embedding.values()), dim=0)
print(f'embeddings shape: {otu_embeddings_tensor.shape}')
torch.save(otu_embeddings_tensor, f"Embeddings/final_otu_embeddings_{text_type}_{dataset}_{N}.pt")  