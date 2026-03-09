from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel


class ConstantOccuModel(OccuModel):
    """
    Constant-per-species model. 
    """

    def __init__(self, n_species: int):
        super().__init__()
        self.n_species = n_species
        # parameters are species-level logits
        self.logit_psi_species = nn.Parameter(torch.zeros(n_species))
        self.logit_d_species = nn.Parameter(torch.zeros(n_species))

    def predict_logit_psi(self, X_psi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        If X_psi is provided, only its first dim is used to infer n_sites_sub.
        If X_psi is None, user must rely on trainer to know n_sites and use broadcasting after call.
        """
        if X_psi is None:
            # caller must expand result to desired n_sites
            return self.logit_psi_species.unsqueeze(0)  # shape: (1, n_species)
        n_sites_sub = X_psi.shape[0]
        return self.logit_psi_species.unsqueeze(0).expand(n_sites_sub, -1)  # (n_sites_sub, n_species)

    def predict_logit_d(self, X_d: Optional[torch.Tensor] = None) -> torch.Tensor:
        if X_d is None:
            return self.logit_d_species.unsqueeze(0)  # (1, n_species)
        n_visits_sub = X_d.shape[0]
        return self.logit_d_species.unsqueeze(0).expand(n_visits_sub, -1)  # (n_visits_sub, n_species)
