import csv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
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
from urllib.error import HTTPError

# Set Pandas options to display full content
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.max_colwidth', None)       # Show full content in each cell
pd.set_option('display.expand_frame_repr', False) # Prevent wrapping to multiple lines

# Config
Entrez.email = "ravisha98rupasinghe@gmail.com"
text_type = 'titles_and_abstracts' #options: titles_only, titles_and_abstracts
dataset = 'M3_feces_L2' # 'M3_feces_L5' or 'F4_feces_L5'
df2 = Dataset.from_csv(f'Data/raw/{dataset}.txt')
journal_filter_file = pd.read_csv(f'scimagojr 2024.csv', sep=';')
N = 30
time_points = 100
mode = 'mean'
is_q1 = False
llm_bert = False

def create_journal_filter(journal_filter_file):
    print(journal_filter_file.head())
    q1_journals = journal_filter_file[journal_filter_file['SJR Best Quartile'] == 'Q1']
    print(f'Number of Q1 journals in the reference file: {len(q1_journals)}')
    return q1_journals



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
        except HTTPError as e:
            print(f"Error fetching for {article}: {e}")
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

            

        #except KeyError:
        except HTTPError as e:
            print(f"Error fetching for {article}: {e}")
            continue
    

    return "\n".join(texts)



# Select higher abundance OTUs
df2 = df2.select_by_rank(count=N)
df3 = df2.time_interval(start=0, delta_t=time_points)

# Create a list of terms from the data
print(df3.df)
data = df3.df

# Get the keywords 
# Extract the index (OTU taxonomy strings)
otu_taxonomies = data.index.tolist()

# Create a set to store all unique keywords
otu_keywords = set()

for taxonomy in otu_taxonomies:
    terms = taxonomy.split(';')  # split by ';'
    for term in terms:
        if '__' in term:
            keyword = term.split('__')[1].strip()
            if keyword:  # skip empty strings
                otu_keywords.add(keyword)

# Convert set to sorted list
otu_keywords = sorted(otu_keywords)
#print(f'otu_keywords={otu_keywords}, length={len(otu_keywords)}')

#Filter
q1_journals = create_journal_filter(journal_filter_file)

# Compile dataset
otu_dataset = {}
for key in otu_keywords:
    #print(f"Fetching for {key}...")
    try:
        if text_type == 'titles_only':
            if is_q1:
                otu_dataset[key] = fetch_only_titles(key, max_results=100, journal_filter=q1_journals)
            else:
                otu_dataset[key] = fetch_only_titles(key, max_results=100)
        else:
            if is_q1:
                otu_dataset[key] = fetch_abstracts(key, max_results=100, journal_filter=q1_journals)
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
if llm_bert:
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_model = AutoModel.from_pretrained(model_name)
    transformer_model.config.output_attentions = True

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model = transformer_model.to(device)
    print(f'device={device}')

    # Function for mean pooling
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Prepare descriptions
    combined_descriptions = []
    otu_keys = otu_dataset.keys()

    for key, text in otu_dataset.items():
        combined_description = key + " " + text
        combined_descriptions.append(combined_description)

    print(f'Number of keys: {len(otu_keys)}')
    print(f'otu_keys={otu_keys}')

    # Tokenize
    inputs = tokenizer(combined_descriptions, padding=True, truncation=True, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        model_output = transformer_model(**inputs)

    # Mean pooling to get sentence embeddings
    embeddings = mean_pooling(model_output, inputs['attention_mask'])

    # Convert to numpy
    embeddings = embeddings.cpu().numpy()
    print(f'embeddings shape = {embeddings.shape}')
else:
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
if is_q1:
    torch.save(otu_embeddings_tensor, f"Embeddings/final_otu_embeddings_{text_type}_{dataset}_{N}_Q1.pt")
elif llm_bert:
    torch.save(otu_embeddings_tensor, f"Embeddings/final_otu_embeddings_{text_type}_{dataset}_{N}_pubmedbert.pt")
else:
    torch.save(otu_embeddings_tensor, f"Embeddings/final_otu_embeddings_{text_type}_{dataset}_{N}.pt")  
