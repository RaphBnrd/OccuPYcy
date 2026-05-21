from __future__ import annotations

from typing import Optional, Callable
import warnings

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel
from occupycy.models.constant import ConstantBlock
from occupycy.models.mlp import MLPBlock


class LinearMLPOccuModel(OccuModel):
    """
    Hybrid model combining linear and MLP components.

    Notes
    -----
    For each component (psi and d), the first ``n_cov_*_linear`` columns are
    sent to the linear branch and the next ``n_cov_*_mlp`` columns to the MLP
    branch. Outputs are summed.
    More details in the docstring of ``LinearMLPBlock``.
    """
    def __init__(
            self, 
            n_species: int, 
            n_cov_psi_linear: int = 0,
            n_cov_psi_mlp: int = 0,
            n_cov_d_linear: int = 0,
            n_cov_d_mlp: int = 0,
            hidden_dims_psi: Optional[list[int]] = [], 
            hidden_dims_d: Optional[list[int]] = [],
            activation_fn_psi: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            activation_fn_d: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            dropout_psi: Optional[float] = 0.0, 
            dropout_d: Optional[float] = 0.0
        ):
        super().__init__(n_species=n_species)

        self.psi_model = LinearMLPBlock(
            n_species=n_species, 
            n_cov_linear=n_cov_psi_linear, 
            n_cov_mlp=n_cov_psi_mlp, 
            hidden_dims_mlp=hidden_dims_psi,
            activation_fn_mlp=activation_fn_psi, 
            dropout_mlp=dropout_psi
        )
        self.d_model = LinearMLPBlock(
            n_species=n_species, 
            n_cov_linear=n_cov_d_linear, 
            n_cov_mlp=n_cov_d_mlp, 
            hidden_dims_mlp=hidden_dims_d,
            activation_fn_mlp=activation_fn_d, 
            dropout_mlp=dropout_d
        )

class LinearMLPBlock(nn.Module):
    """
    Hybrid block with a linear branch and an MLP branch.

    Parameters
    ----------
    n_species : int
        Number of species.
    n_cov_linear : int, default=1
        Number of covariates routed to the linear branch.
    n_cov_mlp : int, default=1
        Number of covariates routed to the MLP branch.
    hidden_dims_mlp : list[int], optional
        Hidden layer sizes for the MLP branch.
    activation_fn_mlp : callable, optional
        Activation constructor for the MLP branch.
    dropout_mlp : float, default=0.0
        Dropout probability for the MLP branch.
    """
    def __init__(
            self, 
            n_species: int, 
            n_cov_linear: int = 1, 
            n_cov_mlp: int = 1,
            hidden_dims_mlp: Optional[list[int]] = [],
            activation_fn_mlp: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            dropout_mlp: Optional[float] = 0.0
        ):
        super().__init__()

        self.n_cov_linear = n_cov_linear
        self.n_cov_mlp = n_cov_mlp

        if self.n_cov_linear < 0:
            raise ValueError("n_cov_linear must be non-negative.")
        elif self.n_cov_linear == 0:
            warnings.warn(
                "n_cov_linear=0 results in no covariates for linear component. Using null linear since mlp will include bias.", 
                UserWarning
            )
            self.linear = lambda _: torch.tensor(0.0)
        else:
            self.linear = nn.Linear(self.n_cov_linear, n_species, bias=False)
        
        if self.n_cov_mlp < 0:
            raise ValueError("n_cov_mlp must be non-negative.")
        elif self.n_cov_mlp == 0:
            warnings.warn(
                "n_cov_mlp=0 results in no covariates for mlp component. Using constant block instead of mlp.", 
                UserWarning
            )
            self.mlp = ConstantBlock(n_species)
        elif hidden_dims_mlp == []:
            warnings.warn(
                "hidden_dims_mlp is empty, resulting in no hidden layers for mlp component. Using linear layer instead of mlp.", 
                UserWarning
            )
            self.mlp = nn.Linear(self.n_cov_mlp, n_species)
        else:
            self.mlp = MLPBlock(
                input_dim=self.n_cov_mlp, output_dim=n_species, hidden_dims=hidden_dims_mlp, 
                activation_fn=activation_fn_mlp, dropout=dropout_mlp
            )

    def forward(self, X: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute logits from split covariates.

        Parameters
        ----------
        X : torch.Tensor or None
            Input tensor containing linear covariates first, then MLP covariates.

        Returns
        -------
        torch.Tensor
            Species logits of shape ``(n_samples, n_species)``.
        """
        X_linear = X[:, :self.n_cov_linear] if self.n_cov_linear > 0 else None
        X_mlp = X[:, self.n_cov_linear:self.n_cov_linear+self.n_cov_mlp] if self.n_cov_mlp > 0 else None
        logits_linear = self.linear(X_linear)
        logits_mlp = self.mlp(X_mlp)
        return logits_linear + logits_mlp