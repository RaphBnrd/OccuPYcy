from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel
from occupycy.models.constant import ConstantBlock


class LinearOccuModel(OccuModel):
    """
    Linear occupancy model with species-specific outputs.

    Parameters
    ----------
    n_species : int
        Number of species.
    n_cov_psi : int, default=1
        Number of occupancy covariates.
    n_cov_d : int, default=1
        Number of detection covariates.
    """
    def __init__(self, n_species: int, n_cov_psi: int = 1, n_cov_d: int = 1):
        super().__init__(n_species=n_species)
        
        # Model occupancy logits
        if n_cov_psi < 0:
            raise ValueError("n_cov_psi must be non-negative.")
        elif n_cov_psi == 0:
            print("Warning: n_cov_psi=0 results in no covariates for occupancy. Using constant block instead of linear.")
            self.psi_model = ConstantBlock(n_species)
        else:
            self.psi_model = nn.Linear(n_cov_psi, n_species)
        
        # Model detection logits
        if n_cov_d < 0:
            raise ValueError("n_cov_d must be non-negative.")
        elif n_cov_d == 0:
            print("Warning: n_cov_d=0 results in no covariates for detection. Using constant block instead of linear.")
            self.d_model = ConstantBlock(n_species)
        else:
            self.d_model = nn.Linear(n_cov_d, n_species)
