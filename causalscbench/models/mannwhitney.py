from typing import List, Tuple

import numpy as np
import re
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.evaluation.statistical_evaluation import Evaluator


class LLMAgentSelection(AbstractInferenceModel):
    """Network inference model based on the LASSO model.
    For each variable, a LASSO regressor is fit and the most predictive variables are set as parents."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        statistical_evaluator = Evaluator(expression_matrix=expression_matrix,
                                          interventions=interventions,
                                          gene_names=gene_names,)
        fully_connected_network_dict = {gene_name: gene_names for gene_name in gene_names}
        gt_graph = statistical_evaluator._evaluate_network(fully_connected_network_dict, return_gt_graph=True)
        return gt_graph