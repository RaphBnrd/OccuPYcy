from __future__ import annotations

from typing import Optional, Callable
import warnings

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel
from occupycy.models.constant import ConstantBlock


class MLPOccuModel(OccuModel):
    """
    MLP-based occupancy model for occupancy and detection components.

    Parameters
    ----------
    n_species : int
        Number of species.
    n_cov_psi : int, default=1
        Number of occupancy covariates.
    n_cov_d : int, default=1
        Number of detection covariates.
    hidden_dims_psi : list[int], optional
        Hidden layer sizes for occupancy MLP.
    hidden_dims_d : list[int], optional
        Hidden layer sizes for detection MLP.
    activation_fn_psi : callable, optional
        Activation constructor for occupancy MLP.
    activation_fn_d : callable, optional
        Activation constructor for detection MLP.
    dropout_psi : float, default=0.0
        Dropout probability for occupancy MLP.
    dropout_d : float, default=0.0
        Dropout probability for detection MLP.
    """
    def __init__(
            self, 
            n_species: int, 
            n_cov_psi: int = 1, 
            n_cov_d: int = 1,
            hidden_dims_psi: Optional[list[int]] = [], 
            hidden_dims_d: Optional[list[int]] = [],
            activation_fn_psi: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            activation_fn_d: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            dropout_psi: Optional[float] = 0.0, 
            dropout_d: Optional[float] = 0.0
        ):
        super().__init__(n_species=n_species)
        
        # Model occupancy logits
        if n_cov_psi < 0:
            raise ValueError("n_cov_psi must be non-negative.")
        elif n_cov_psi == 0:
            warnings.warn(
                "n_cov_psi=0 results in no covariates for occupancy. Using constant block instead of mlp.",
                UserWarning
            )
            self.psi_model = ConstantBlock(n_species)
        elif hidden_dims_psi == []:
            warnings.warn(
                "hidden_dims_psi is empty, resulting in no hidden layers for occupancy. Using linear layer instead of mlp.",
                UserWarning
            )
            self.psi_model = nn.Linear(n_cov_psi, n_species)
        else:
            self.psi_model = MLPBlock(
                input_dim=n_cov_psi, output_dim=n_species, hidden_dims=hidden_dims_psi, 
                activation_fn=activation_fn_psi, dropout=dropout_psi
            )
        
        # Model detection logits
        if n_cov_d < 0:
            raise ValueError("n_cov_d must be non-negative.")
        elif n_cov_d == 0:
            warnings.warn(
                "n_cov_d=0 results in no covariates for detection. Using constant block instead of mlp.",
                UserWarning
            )
            self.d_model = ConstantBlock(n_species)
        elif hidden_dims_d == []:
            warnings.warn(
                "hidden_dims_d is empty, resulting in no hidden layers for detection. Using linear layer instead of mlp.",
                UserWarning
            )
            self.d_model = nn.Linear(n_cov_d, n_species)
        else:
            self.d_model = MLPBlock(
                input_dim=n_cov_d, output_dim=n_species, hidden_dims=hidden_dims_d, 
                activation_fn=activation_fn_d, dropout=dropout_d
            )

class MLPBlock(nn.Module):
    """
    Fully connected MLP block.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    hidden_dims : list[int]
        Hidden layer sizes.
    activation_fn : callable
        Activation constructor.
    dropout : float, default=0.0
        Dropout probability applied after hidden activations.
    """
    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int],
            activation_fn: Callable[..., torch.nn.Module],
            dropout: Optional[float] = 0.0
        ):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``(n_samples, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(n_samples, output_dim)``.
        """
        return self.mlp(X)
