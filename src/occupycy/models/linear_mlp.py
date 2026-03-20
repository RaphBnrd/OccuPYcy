from __future__ import annotations

from typing import Optional, Callable

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel
from occupycy.models.constant import ConstantBlock
from occupycy.models.mlp import MLPBlock


class LinearMLPOccuModel(OccuModel):
    """
    Multi-layer perceptron model for occupancy. 
    When inferring, first n_cov_linear covariates are fed to linear layer, 
    and next n_cov_mlp covariates are fed to mlp, and outputs are summed.
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
    Block of linear + MLP layers.
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
            print("Warning: n_cov_linear=0 results in no covariates for linear component. Using null linear since mlp will include bias.")
            self.linear = lambda _: torch.tensor(0.0)
        else:
            self.linear = nn.Linear(self.n_cov_linear, n_species, bias=False)
        
        if self.n_cov_mlp < 0:
            raise ValueError("n_cov_mlp must be non-negative.")
        elif self.n_cov_mlp == 0:
            print("Warning: n_cov_mlp=0 results in no covariates for mlp component. Using constant block instead of mlp.")
            self.mlp = ConstantBlock(n_species)
        elif hidden_dims_mlp == []:
            print("Warning: hidden_dims_mlp is empty, resulting in no hidden layers for mlp component. Using linear layer instead of mlp.")
            self.mlp = nn.Linear(self.n_cov_mlp, n_species)
        else:
            self.mlp = MLPBlock(
                input_dim=self.n_cov_mlp, output_dim=n_species, hidden_dims=hidden_dims_mlp, 
                activation_fn=activation_fn_mlp, dropout=dropout_mlp
            )

    def forward(self, X: Optional[torch.Tensor]) -> torch.Tensor:
        X_linear = X[:, :self.n_cov_linear] if self.n_cov_linear > 0 else None
        X_mlp = X[:, self.n_cov_linear:self.n_cov_linear+self.n_cov_mlp] if self.n_cov_mlp > 0 else None
        logits_linear = self.linear(X_linear)
        logits_mlp = self.mlp(X_mlp)
        return logits_linear + logits_mlp