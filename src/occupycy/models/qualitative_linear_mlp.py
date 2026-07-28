from __future__ import annotations

from typing import Optional, Callable
import warnings

import torch
import torch.nn as nn

from occupycy.models.base import OccuModel
from occupycy.models.constant import ConstantBlock
from occupycy.models.mlp import MLPBlock


class QualiLinearMLPOccuModel(OccuModel):
    """
    Hybrid model combining qualitative intercept, linear and MLP components.

    Notes
    -----
    For each component (psi and d), the first ``len(n_classes_*_intercept)`` 
    columns are sent to the intercept branch, the next ``n_cov_*_linear`` 
    columns to the linear branch and the next ``n_cov_*_mlp`` columns to the
    MLP branch. Outputs are summed.
    """
    def __init__(
            self, 
            n_species: int, 
            n_classes_psi_intercept: list[int] = [],
            n_cov_psi_linear: int = 0,
            n_cov_psi_mlp: int = 0,
            n_classes_d_intercept: list[int] = [],
            n_cov_d_linear: int = 0,
            n_cov_d_mlp: int = 0,
            hidden_dims_psi: Optional[list[int]] = [], 
            hidden_dims_d: Optional[list[int]] = [],
            activation_fn_psi: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            activation_fn_d: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            dropout_psi: Optional[float] = 0.0, 
            dropout_d: Optional[float] = 0.0,
            n_fourier_features_psi: Optional[list[int]] = None,
            n_fourier_features_d: Optional[list[int]] = None
        ):
        super().__init__(n_species=n_species)

        self.psi_model = QualiLinearMLPBlock(
            n_species=n_species, 
            n_classes_intercept=n_classes_psi_intercept,
            n_cov_linear=n_cov_psi_linear, 
            n_cov_mlp=n_cov_psi_mlp, 
            hidden_dims_mlp=hidden_dims_psi,
            activation_fn_mlp=activation_fn_psi, 
            dropout_mlp=dropout_psi,
            n_fourier_features=n_fourier_features_psi
        )
        self.d_model = QualiLinearMLPBlock(
            n_species=n_species, 
            n_classes_intercept=n_classes_d_intercept,
            n_cov_linear=n_cov_d_linear, 
            n_cov_mlp=n_cov_d_mlp, 
            hidden_dims_mlp=hidden_dims_d,
            activation_fn_mlp=activation_fn_d, 
            dropout_mlp=dropout_d,
            n_fourier_features=n_fourier_features_d
        )

class QualiLinearMLPBlock(nn.Module):
    """
    Hybrid block with a qualitative intercept, a linear branch and an MLP branch.

    Parameters
    ----------
    n_species : int
        Number of species.
    n_classes_intercept : list[int], default=[]
        Number of classes for each categorical covariate routed to the intercept branch.
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
    n_fourier_features : list[int], optional
        Number of Fourier features to add for each input dimension in the MLP branch.
    """
    def __init__(
            self, 
            n_species: int, 
            n_classes_intercept: list[int] = [],
            n_cov_linear: int = 1, 
            n_cov_mlp: int = 1,
            hidden_dims_mlp: Optional[list[int]] = [],
            activation_fn_mlp: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
            dropout_mlp: Optional[float] = 0.0,
            n_fourier_features: Optional[list[int]] = None
        ):
        super().__init__()

        self.n_classes_intercept = n_classes_intercept
        self.n_cov_linear = n_cov_linear
        self.n_cov_mlp = n_cov_mlp

        if len(n_classes_intercept) == 0:
            warnings.warn(
                "n_classes_intercept is empty, resulting in no categorical covariates for intercept branch. Using constant block instead of qualitative intercept.",
                UserWarning
            )
            self.intercepts = ConstantBlock(n_species)
        else:
            self.intercepts = nn.ParameterList([
                nn.Parameter(torch.zeros(n_classes, n_species))
                for n_classes in n_classes_intercept
            ])

        if self.n_cov_linear < 0:
            raise ValueError("n_cov_linear must be non-negative.")
        elif self.n_cov_linear == 0:
            warnings.warn(
                "n_cov_linear=0 results in no covariates for linear component. Using null linear since intercept will include bias.",
                UserWarning
            )
            self.linear = lambda _: torch.tensor(0.0)
        else:
            self.linear = nn.Linear(self.n_cov_linear, n_species, bias=False)
        
        if self.n_cov_mlp < 0:
            raise ValueError("n_cov_mlp must be non-negative.")
        elif self.n_cov_mlp == 0:
            warnings.warn(
                "n_cov_mlp=0 results in no covariates for mlp component. Using null mlp since intercept will include bias.",
                UserWarning
            )
            self.mlp = lambda _: torch.tensor(0.0)
        elif hidden_dims_mlp == []:
            warnings.warn(
                "hidden_dims_mlp is empty, resulting in no hidden layers for mlp component. Using linear layer instead of mlp.",
                UserWarning
            )
            self.mlp = nn.Linear(self.n_cov_mlp, n_species, bias=False)
        else:
            self.mlp = MLPBlock(
                input_dim=self.n_cov_mlp, output_dim=n_species, hidden_dims=hidden_dims_mlp, 
                activation_fn=activation_fn_mlp, dropout=dropout_mlp,
                n_fourier_features=n_fourier_features
            )

    def forward(self, X: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute logits from split covariates.

        Parameters
        ----------
        X : torch.Tensor or None
            Input tensor containing intercept covariates first, 
            then linear covariates, then MLP covariates. 

        Returns
        -------
        torch.Tensor
            Species logits of shape ``(n_samples, n_species)``.
        """
        idx1 = len(self.n_classes_intercept)
        idx2 = idx1 + self.n_cov_linear
        idx3 = idx2 + self.n_cov_mlp
        X_intercept = X[:, :idx1] if len(self.n_classes_intercept) > 0 else None
        X_linear = X[:, idx1:idx2] if self.n_cov_linear > 0 else None
        X_mlp = X[:, idx2:idx3] if self.n_cov_mlp > 0 else None
        # > Intercept branch
        if X_intercept is None:
            logits_intercept = self.intercepts(X_intercept) # shape: (1, n_species)
            if X.shape[0] != 1:
                logits_intercept = logits_intercept.expand(X.shape[0], -1) # shape: (n_samples, n_species)
        else:
            logits_intercept = 0.0
            for i in range(len(self.n_classes_intercept)):
                # self.intercepts[i] has shape (n_classes_i, n_species)
                # X_intercept[:, i] has shape (n_samples,) (class indices)
                logits_intercept += self.intercepts[i][X_intercept[:, i].long()] # shape: (n_samples, n_species)
        # > Linear branch
        logits_linear = self.linear(X_linear) # shape: (n_samples, n_species)
        # > MLP branch
        logits_mlp = self.mlp(X_mlp) # shape: (n_samples, n_species)
        
        return logits_intercept + logits_linear + logits_mlp # shape: (n_samples, n_species)