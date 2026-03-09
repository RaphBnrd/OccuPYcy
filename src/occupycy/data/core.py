from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class OccuData:
    """
    Canonical data container for the single-season occupancy setup.
      - y: (n_visits, n_species) with 0/1
      - visit_site_idxs: (n_visits,) integer site index for each visit row
      - X_psi: (n_sites, n_cov_psi) optional site/state covariates
      - X_d: (n_visits, n_cov_d) optional visit/detection covariates
    """
    y: torch.Tensor
    visit_site_idxs: torch.Tensor
    X_psi: Optional[torch.Tensor] = None
    X_d: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        y = torch.as_tensor(self.y)
        visit_site_idxs = torch.as_tensor(self.visit_site_idxs)

        if y.ndim != 2:
            raise ValueError(f"y must be 2D (n_visits, n_species); got {tuple(y.shape)}")
        if visit_site_idxs.ndim != 1:
            raise ValueError(
                f"visit_site_idxs must be 1D (n_visits,); got {tuple(visit_site_idxs.shape)}"
            )
        if visit_site_idxs.shape[0] != y.shape[0]:
            raise ValueError(
                f"visit_site_idxs length {visit_site_idxs.shape[0]} must match y n_visits {y.shape[0]}"
            )
        if visit_site_idxs.dtype not in (torch.int64, torch.int32):
            raise ValueError("visit_site_idxs must be int32/int64.")

        if self.X_d is not None:
            X_d = torch.as_tensor(self.X_d)
            if X_d.ndim != 2:
                raise ValueError(f"X_d must be 2D (n_visits, n_cov_d); got {tuple(X_d.shape)}")
            if X_d.shape[0] != y.shape[0]:
                raise ValueError("X_d first dim must match y n_visits.")
        if self.X_psi is not None:
            X_psi = torch.as_tensor(self.X_psi)
            if X_psi.ndim != 2:
                raise ValueError(f"X_psi must be 2D (n_sites, n_cov_psi); got {tuple(X_psi.shape)}")
            min_idx = int(visit_site_idxs.min().item())
            max_idx = int(visit_site_idxs.max().item())
            n_idx = visit_site_idxs.unique().numel()
            if min_idx != 0 or max_idx != X_psi.shape[0] - 1 or n_idx != X_psi.shape[0]:
                raise ValueError(f"visit_site_idxs must be in [0, n_sites-1] with no missing indices, but got min {min_idx}, max {max_idx}, unique count {n_idx} for X_psi rows {X_psi.shape[0]}.")

        # store coerced tensors back (because dataclass is frozen)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "visit_site_idxs", visit_site_idxs)
        if self.X_psi is not None:
            object.__setattr__(self, "X_psi", torch.as_tensor(self.X_psi))
        if self.X_d is not None:
            object.__setattr__(self, "X_d", torch.as_tensor(self.X_d))

    @property
    def n_visits(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_species(self) -> int:
        return int(self.y.shape[1])

    @property
    def n_sites(self) -> int:
        return int(self.visit_site_idxs.unique().shape[0])

    def to(
        self, 
        *, 
        device: str | torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> "OccuData":
        def _to(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)

        # visit_site_idxs should stay integer dtype
        vsi = self.visit_site_idxs.to(device=device)

        return replace(
            self,
            y=_to(self.y),
            visit_site_idxs=vsi,
            X_psi=_to(self.X_psi),
            X_d=_to(self.X_d),
        )