from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel
from occupycy.data import OccuData


class ConstantOccuModel(OccuModel):
    """
    Constant-per-species model. 
    """
    def __init__(self, n_species: int):
        super().__init__(n_species=n_species)
        
        self.psi_model = ConstantBlock(n_species)
        self.d_model = ConstantBlock(n_species)

    def initialize_rough_estimate(self, data: "OccuData") -> None:
        """
        Initialize the constant parameters to the rough estimates.
        """
        y = data.y
        indices_visit_site = data.indices_visit_site # shape: (n_visits,) with values in [0, n_sites-1]

        n_sites = data.n_sites
        n_visits = data.n_visits
        n_species = data.n_species

        n_D = y.sum(dim=0) # shape: (n_species,)
        S = n_sites # total number of sites
        K = n_visits / n_sites # average number of visits per site
        # Number sites with at least one detection for each species

        # Binary indicator of whether each species was detected at each site
        detected = torch.zeros((n_sites, n_species), device=y.device, dtype=y.dtype)
        detected.index_add_(0, indices_visit_site, y)
        detected = detected.bool() # shape: (n_sites, n_species)
        S_D = detected.sum(dim=0) # shape: (n_species,)
        
        # Solve the equations for the rough estimates of psi and d. 
        # The equations are:
        # hat_d = 1 - (1 - hat_d) ** K * n_D / (S_D * K)
        # hat_psi = S_D / (S * (1 - (1 - hat_d) ** K))
        
        # If we approximate 1 - (1 - hat_d) ** K by 1
        hat_d = (n_D / (S_D * K)).clamp(min=1e-6, max=1-1e-6)
        hat_psi = (S_D / (S * (1 - (1 - hat_d) ** K))).clamp(min=1e-6, max=1-1e-6)

        with torch.no_grad():
            self.psi_model.logit_species.copy_(torch.logit(hat_psi))
            self.d_model.logit_species.copy_(torch.logit(hat_d))

class ConstantBlock(nn.Module):
    """
    Block of constant logits.
    """
    def __init__(self, n_species: int):
        super().__init__()
        self.logit_species = nn.Parameter(torch.zeros(n_species))

    def forward(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        X is a tensor of shape (n_sub, n_cov), or None. Only the first dim (n_sub) is used to infer the number of samples to expand the constant logits to.
        """
        if X is None:
            return self.logit_species.unsqueeze(0)  # shape: (1, n_species)
        n_sub = X.shape[0]
        return self.logit_species.unsqueeze(0).expand(n_sub, -1)  # (n_sub, n_species)
