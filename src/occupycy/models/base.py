from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn


class OccuModel(ABC):
    """
    Base occupancy model mapping covariates to occupancy and detection logits.

    Parameters
    ----------
    psi_model : torch.nn.Module or None, default=None
        Module used to predict occupancy logits from site covariates.
    d_model : torch.nn.Module or None, default=None
        Module used to predict detection logits from visit covariates.
    n_species : int, default=1
        Number of species modeled jointly.
    """
    
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

    def predict_logit_psi(self, X_psi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict occupancy logits.

        Parameters
        ----------
        X_psi : torch.Tensor or None, default=None
            Site covariates with shape ``(n_sites_sub, n_cov_psi)``.
            Can be ``None`` for intercept-only models.

        Returns
        -------
        torch.Tensor
            Occupancy logits with shape ``(n_sites_sub, n_species)`` or
            ``(1, n_species)`` for intercept-only models.

        Raises
        ------
        NotImplementedError
            If ``psi_model`` is not set.
        """
        if self.psi_model is None:
            raise NotImplementedError(
                "Subclasses must implement predict_logit_psi to return occupancy logits."
            )
        return self.psi_model(X_psi)

    def predict_logit_d(self, X_d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict detection logits.

        Parameters
        ----------
        X_d : torch.Tensor or None, default=None
            Visit covariates with shape ``(n_visits_sub, n_cov_d)``.
            Can be ``None`` for intercept-only models.

        Returns
        -------
        torch.Tensor
            Detection logits with shape ``(n_visits_sub, n_species)`` or
            ``(1, n_species)`` for intercept-only models.

        Raises
        ------
        NotImplementedError
            If ``d_model`` is not set.
        """
        if self.d_model is None:
            raise NotImplementedError(
                "Subclasses must implement predict_logit_d to return detection logits."
            )
        return self.d_model(X_d)
    
    def forward_logits(
        self, 
        *, 
        X_psi: Optional[torch.Tensor] = None,
        X_d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute occupancy and detection logits.

        Parameters
        ----------
        X_psi : torch.Tensor or None, default=None
            Site covariates.
        X_d : torch.Tensor or None, default=None
            Visit covariates.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple ``(logit_psi, logit_d)``.
        """
        logit_psi = self.predict_logit_psi(X_psi)
        logit_d = self.predict_logit_d(X_d)
        return logit_psi, logit_d
    
    def eval(self):
        """
        Set internal modules to evaluation mode.

        Returns
        -------
        None
        """
        if self.psi_model is not None:
            self.psi_model.eval()
        if self.d_model is not None:
            self.d_model.eval()
    
    def train(self):
        """
        Set internal modules to training mode.

        Returns
        -------
        None
        """
        if self.psi_model is not None:
            self.psi_model.train()
        if self.d_model is not None:
            self.d_model.train()
    
    def to(self, *args, **kwargs):
        """
        Move internal modules to a target device/dtype.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``torch.nn.Module.to``.
        **kwargs
            Keyword arguments forwarded to ``torch.nn.Module.to``.

        Returns
        -------
        OccuModel
            The same model instance.
        """
        if self.psi_model is not None:
            self.psi_model.to(*args, **kwargs)
        if self.d_model is not None:
            self.d_model.to(*args, **kwargs)
        return self
