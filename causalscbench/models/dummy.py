from typing import List, Tuple

import numpy as np
import re
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime

from gptquery import GPT

class DummyModel(AbstractInferenceModel):
    """Network inference model based on the LASSO model.
    For each variable, a LASSO regressor is fit and the most predictive variables are set as parents."""

    def __init__(self, model_path="casperhansen/llama-3-70b-instruct-awq", local=True) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        return [(gene_names[0], gene_names[1])]