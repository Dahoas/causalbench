"""
Copyright (C) 2022  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Dict, List, Set, Tuple

import numpy as np
from causalscbench.models.training_regimes import TrainingRegime


class Evaluator(object):
    def __init__(self, ground_truth_subnetwork: Set[Tuple[str]]) -> None:
        """
        Evaluation module to biologically evaluate a network using ground-truth biological data.

        Args:
            ground_truth_subnetwork: list of know gene-gene interactions
        """
        self.ground_truth_subnetwork = ground_truth_subnetwork

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        edges = set()
        gene_names = set(gene_names)
        for i, j in self.ground_truth_subnetwork:
            if i in gene_names and j in gene_names and i != j:
                edges.add((i, j))
        return list(edges)
    
    def make_undirected(self, network: List[Tuple]) -> List[Tuple]:
        network_undirected = set()
        for i, j in network:
            network_undirected.add((i, j))
            network_undirected.add((j, i))
        return network_undirected

    def evaluate_network(self, network: List[Tuple], directed: bool = False) -> Dict:
        tp, fp, fn = 0, 0, 0
        ground_truth_subnetwork = self.ground_truth_subnetwork
        if not directed:
            network = self.make_undirected(network)
            ground_truth_subnetwork = self.make_undirected(ground_truth_subnetwork)
        for edge in network:
            if edge in ground_truth_subnetwork:
                tp += 1
            else:
                fp += 1
        for edge in ground_truth_subnetwork:
            if edge not in network:
                fn += 1
        tp = tp if directed else tp / 2
        fp = fp if directed else fp / 2
        fn = fn if directed else fn / 2
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
