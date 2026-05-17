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

preprocessed = True
text_type = 'titles_and_abstracts' # titles_and_abstracts or titles_only
is_q1 = True 
save_abundance = False

def create_journal_filter(journal_filter_file):
    print(journal_filter_file.head())
    q1_journals = journal_filter_file[journal_filter_file['SJR Best Quartile'] == 'Q1']
    print(f'Number of Q1 journals in the reference file: {len(q1_journals)}')
    return q1_journals

df = pd.read_csv('Data/BaranwalClark2022/2021_02_19_MultifunctionalDynamicData.csv')
journal_filter_file = pd.read_csv(f'scimagojr 2024.csv', sep=';')

print(df)

# Identify OTU abundance columns (those ending in '_OD')
otu_cols = [col for col in df.columns if '_OD' in col]

# Dictionary to hold tensors for each community
communities = df['Community'].unique()

# Get all OTU + OTU_OD pairs
otu_all = df.columns[df.columns.get_loc("PC"):df.columns.get_loc("DF_OD")+1]
otu_pairs = [(otu_all[i], otu_all[i+1]) for i in range(0, len(otu_all), 2)]

community_tensors = {}

if preprocessed:
    print(f'Loading preprocessed data...')
    community_tensors = torch.load(f'Embeddings/preprocesses_data_global_baranwalclark.pt')
else:
    for community in communities:
        df_comm = df[df['Community'] == community].sort_values(by='Time')

        selected_otu_od = []
        for otu, otu_od in otu_pairs:
            if (df_comm[otu] == 1).any():  # Include OTU_OD only if OTU is 1 in any timepoint
                selected_otu_od.append(otu_od)

        # Construct abundance tensor: rows = OTUs (actually OTU_ODs), cols = timepoints
        tensor = df_comm[selected_otu_od].to_numpy().T  # shape: (N_selected_otus, 4)
        # Global normalization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        community_tensors[community] = torch.tensor(normalized_tensor, dtype=torch.float32)
        #community_tensors[community] = tensor

    print(f'Number of communities: {len(community_tensors)}')
    for key, val in community_tensors.items():
        print(f'community tensor {key}: {val.shape}')
        print(f'{val}')

    # Save abundance data
    if save_abundance:
        torch.save(community_tensors, f'Embeddings/preprocesses_data_global_baranwalclark.pt')

######################################################

# Map for OTU names and get sentence embeddings for each
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

######################################################

def fetch_abstracts(otu, max_results=50, journal_filter=None):
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
            journal = article['MedlineCitation']['Article']['Journal']['Title']
            info = article['MedlineCitation']['Article']['Journal']['ISSN'].replace("-", "")
            full_abstract = " ".join(str(section) for section in abstract)
            print(f'Comparison-----------------')
            print(f'pubmed journal: {journal}, info:{info}')
            if journal_filter is not None:
                reference = journal_filter[journal_filter['Issn'].str.split(",\s*").apply(lambda x: info in x)]
                
                if not reference.empty:
                    print(f"reference:{reference['Title']}, {reference['Issn']}")
                    print(f'Reference added...')
                    texts.append(f"{title}. {full_abstract}")
                else:
                    print(f'No match for {journal}')
        except KeyError:
            continue
    return "\n".join(texts)

def fetch_only_titles(otu, max_results=50, journal_filter=None):
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
            journal = article['MedlineCitation']['Article']['Journal']['Title']
            info = article['MedlineCitation']['Article']['Journal']['ISSN'].replace("-", "")
            print(f'Comparison-----------------')
            print(f'pubmed journal: {journal}, info:{info}')
            if journal_filter is not None:
                reference = journal_filter[journal_filter['Issn'].str.split(",\s*").apply(lambda x: info in x)]
                
                if not reference.empty:
                    print(f"reference:{reference['Title']}, {reference['Issn']}")
                    print(f'Reference added...')
                    texts.append(f"{title}")
                else:
                    print(f'No match for {journal}')
        except KeyError:
            continue
    return "\n".join(texts)

######################################################

# Create sentence embedding communities
# Config
Entrez.email = "ravisha98rupasinghe@gmail.com"


otu_keywords = set()
for key, value in abbr_to_full_name.items():
    terms = value.split('__')
    for term in terms:
        otu_keywords.add(term)

otu_keywords = sorted(otu_keywords)
print(f'total terms: {len(otu_keywords)}')

#Filter
q1_journals = create_journal_filter(journal_filter_file)

# Get sentences
otu_text_data = {}
for key in otu_keywords:
    try:
        if text_type == 'titles_only':
            if is_q1:
                otu_text_data[key] = fetch_only_titles(key, max_results=100, journal_filter=q1_journals)
            else:
                otu_text_data[key] = fetch_only_titles(key, max_results=100)
        else:
            if is_q1:
                otu_text_data[key] = fetch_abstracts(key, max_results=100, journal_filter=q1_journals)
            else:
                otu_text_data[key] = fetch_abstracts(key, max_results=100)
    except Exception as e:
        print(f"Error fetching for {key}: {e}")
    sleep(1)  # Be respectful to NCBI servers


######################################################

model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer
transformer_model = model[0].auto_model
transformer_model.config.output_attentions = True


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device={device}')

combined_descriptions = []
otu_keys = otu_text_data.keys()

for key, text in otu_text_data.items():
    combined_description = key + " " + text
    combined_descriptions.append(combined_description)

print(f'Number of keys: {len(otu_keys)}')
print(f'otu_keys={otu_keys}')

embeddings = model.encode(combined_descriptions, convert_to_numpy=True)
print(f'embeddings shape = {embeddings.shape}')

embedding_dict = {sid: emb for sid, emb in zip(otu_keys, embeddings)}
embedding_dict_serializable = {k: v.tolist() for k, v in embedding_dict.items()}

######################################################

embedding_dict = {k: np.array(v) for k, v in embedding_dict_serializable.items()}
embedding_tensor_dict = {k: torch.tensor(v) for k, v in embedding_dict.items()}

community_tensors_text = {} # final
for key, value in community_tensors.items():
    otu_to_embedding = {}
    taxonomy_names = key.split("-") #add each abbreviated term ex: PC-OJ-PJ

    for item in taxonomy_names: # ex: for PC
        final_embeddings = []
        full_terms = abbr_to_full_name[item].split("__")
        for term in full_terms:
            if term in embedding_tensor_dict:
                final_embeddings.append(embedding_tensor_dict[term])

        if final_embeddings:
            otu_emb = torch.stack(final_embeddings, dim=0).mean(dim=0)
            otu_to_embedding[item] = otu_emb
        else:
            print(f"Warning: No matching embeddings for OTU: {item}")

    otu_embeddings_tensor = torch.stack(list(otu_to_embedding.values()), dim=0)
    community_tensors_text[key] = otu_embeddings_tensor

for key, value in community_tensors_text.items():
    print(f'For community {key}: text shaped {value.shape}')

if is_q1:
    torch.save(community_tensors_text, f'Embeddings/final_otu_embeddings_{text_type}_baranwalclark_Q1.pt')
else:
    torch.save(community_tensors_text, f'Embeddings/final_otu_embeddings_{text_type}_baranwalclark.pt')