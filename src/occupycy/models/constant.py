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
        super().__init__(n_species=n_species)
        
        self.psi_model = ConstantBlock(n_species)
        self.d_model = ConstantBlock(n_species)

    def predict_logit_psi(self, X_psi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        If X_psi is provided, only its first dim is used to infer n_sites_sub.
        If X_psi is None, user must rely on trainer to know n_sites and use broadcasting after call.
        """
        return self.psi_model(X_psi)

    def predict_logit_d(self, X_d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        If X_d is provided, only its first dim is used to infer n_visits_sub.
        If X_d is None, user must rely on trainer to know n_visits and use broadcasting after call.
        """
        return self.d_model(X_d)

class ConstantBlock(nn.Module):
    """
    Block of constant logits.
    """
    def __init__(self, n_species: int):
        super().__init__()
        self.logit_species = nn.Parameter(torch.zeros(n_species))

    def forward(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        if X is None:
            return self.logit_species.unsqueeze(0)  # shape: (1, n_species)
        n_sub = X.shape[0]
        return self.logit_species.unsqueeze(0).expand(n_sub, -1)  # (n_sub, n_species)
