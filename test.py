import numpy as np
import requests
from tqdm import tqdm

data = np.load("data/dataset_k562_filtered.npz")
print(data)
gene_names = data["var_names"]

def retrieve_ensembl_gene_info(gene: str):
    base_url = "https://rest.ensembl.org"  # Ensembl REST API base URL
    endpoint = f"/lookup/id/{gene}"  # The specific gene ID
    complete_url = base_url + endpoint  # Combining for the full URL

    headers = {"Content-Type": "application/json"}  # Specify JSON response
    response = requests.get(complete_url, headers=headers)  # Send the GET request
    if response.ok:
        return response.json()
    else:
        return dict()
    
text = ""
for gene in tqdm(gene_names):
    info = retrieve_ensembl_gene_info(gene)
    text += str(info)

print(len(text))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
enc = tok(text)
print(len(enc["input_ids"]))