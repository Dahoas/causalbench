from typing import List, Tuple

import numpy as np
import re
import requests
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime

from causalscbench.utils import retrieve_ensembl_gene_info

from tqdm import tqdm

from gptquery import GPT


class LLMAgentSelection(AbstractInferenceModel):
    """Network inference model based on the LASSO model.
    For each variable, a LASSO regressor is fit and the most predictive variables are set as parents."""

    def __init__(self, model_path="casperhansen/llama-3-70b-instruct-awq", local=True) -> None:
        super().__init__()
        task_prompt_text = """\
You are tasked with identifying causal relationships between genes and their expression. \
Which of the following genes (if any) will have a causal effect on the expression of the target {target_info}:

Candidates: {candidates_info}

Before answering you should reason about the relationship between the target and candidates \
using any prior knowledge you have.
Enclose your final answer in the tags <parents>GENE_1,GENE_2,...</parents><DONE> where GENE_i is the ensembl identifier.
If none of the candidates will have an effect on the target print <parents></parents><DONE>.
"""
        #local = False
        #model_path = "azure/gpt-4-turbo"
        model_name = f"openai/{model_path}" if local else model_path
        model_endpoint = "http://GCRAZGDL3051:8000/v1" if local else None
        max_num_tokens = 1024
        keys = dict(
            AZURE_API_KEY="f8f587bf8d6346ac9bb3c9b2db4e1fb6",
            AZURE_API_BASE="https://gcraoai7sw1.openai.azure.com/",
            AZURE_API_VERSION="2024-02-01",
        )
        self.model = GPT(model_name=model_name,
                        model_endpoint=model_endpoint,
                        task_prompt_text=task_prompt_text,
                        max_num_tokens=max_num_tokens,
                        keys=keys,
                        logging_path="output/llama_3_70B.jsonl")

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        edges = set()
        genes_info = {gene: retrieve_ensembl_gene_info(gene) for gene in gene_names}
        display_to_ensembl = {info["display_name"]: gene for gene, info in genes_info.items() if info.get("display_name")}
        assembly_to_ensembl = {info["assembly_name"]: gene for gene, info in genes_info.items() if info.get("assembly_name")}
        batch_size = 8
        num_batches = (len(gene_names) + batch_size - 1) // batch_size
        offset = 88  # used when already computed partial graph
        for i in tqdm(range(offset, len(gene_names))):
            target = gene_names[i]
            target_info = {target: genes_info[target]}
            for k in range(num_batches):
                candidates_info = [{gene_name: genes_info[gene_name]} for j, gene_name in enumerate(gene_names[k*batch_size:(k+1)*batch_size]) if i != j + k*batch_size]
                input_dict = dict(
                    target_info=target_info,
                    candidates_info=candidates_info,
                )
                output = self.model([input_dict], is_complete_keywords=["<DONE>"], keep_keywords=True)[0]["response"]
                parents = re.findall(r"<parents>([\w\W]*)</parents>", output)
                if len(parents) > 0:
                    parents = [parent.strip() for parent in parents[0].split(",")]
                else:
                    parents = [target]
                for parent in parents:
                    if parent in gene_names:
                        edges.add((parent, target))
                    if display_to_ensembl.get(parent) in gene_names:
                        edges.add((display_to_ensembl.get(parent), target))
                    if assembly_to_ensembl.get(parent) in gene_names:
                        edges.add((assembly_to_ensembl.get(parent), target))
        return list(edges)