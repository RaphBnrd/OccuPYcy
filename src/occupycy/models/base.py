from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn


class OccuModel(ABC):
    """Model that maps covariates -> logits. Subclasses should implement component predictors."""
    
    def __init__(
        self,
        psi_model: Optional[nn.Module] = None,
        d_model: Optional[nn.Module] = None,
        n_species: int = 1,
    ):
        super().__init__()

        self.psi_model = psi_model
        self.d_model = d_model
        self.n_species = n_species

    @abstractmethod
    def predict_logit_psi(self, X_psi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return occupancy logits for the provided site covariates.
        - X_psi shape: (n_sites_sub, n_cov_psi) or None for no-covariate models.
        """
        raise NotImplementedError(
            "Subclasses must implement predict_logit_psi to return occupancy logits."
        )

    @abstractmethod
    def predict_logit_d(self, X_d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return detection logits for the provided visit covariates.
        - X_d shape: (n_visits_sub, n_cov_d) or None for no-covariate models.
        """
        raise NotImplementedError(
            "Subclasses must implement predict_logit_d to return detection logits."
        )
    
    def forward_logits(
        self, 
        *, 
        X_psi: Optional[torch.Tensor] = None,
        X_d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute occupancy and detection logits for the provided covariates.
        """
        logit_psi = self.predict_logit_psi(X_psi)
        logit_d = self.predict_logit_d(X_d)
        return logit_psi, logit_d
