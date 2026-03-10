from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------------
#  OccuData: canonical data container for occupancy modeling
# -------------------------------------------------------------

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

    site_ids: Optional[torch.Tensor] = None

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

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
        return int(self.visit_site_idxs.max().item()) + 1

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
    
    def with_dataloaders(
        self,
        *,
        batch_size: int = 32,
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        shuffle: bool = True,
        seed: Optional[int] = None,
        device: str | torch.device | None = None,
    ):
        """
        Returns a new OccuData with train/val/test DataLoaders created from the dataset.
        """
        assert 0.0 <= val_fraction < 1.0, "val_fraction must be in [0.0, 1.0)"
        assert 0.0 <= test_fraction < 1.0, "test_fraction must be in [0.0, 1.0)"
        assert val_fraction + test_fraction < 1.0, "val_fraction + test_fraction must be < 1.0"

        data = self.to(device=device) if device is not None else self
        dataset = OccuSiteDataset(data)

        n = len(dataset)
        n_val = int(n * val_fraction)
        n_test = int(n * test_fraction)
        n_train = n - n_val - n_test

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_occu_sites, generator=generator
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_occu_sites, generator=generator
        ) if n_val else None
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_occu_sites, generator=generator
        ) if n_test else None

        return replace(
            self,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

# -------------------------------------------------------------
# OccuSiteDataset and collate_occu_sites for DataLoader support
# -------------------------------------------------------------

class OccuSiteDataset(Dataset):
    """
    PyTorch Dataset that exposes site-level samples from an OccuData object.

    Each dataset item corresponds to a single site and returns all visits
    associated with that site. Visit-level observations and covariates are
    selected using the site's index in `visit_site_idxs`.

    Args:
        data (OccuData):
            Occupancy dataset containing visit-level detections (`y`),
            site indices (`visit_site_idxs`), and optional covariates
            (`X_psi`, `X_d`).

    Returns:
        dict:
            A dictionary containing tensors for a single site:

            - 'site_id' (torch.Tensor): Scalar tensor with the site index.
            - 'y' (torch.Tensor): Detection matrix for the site,
            shape (n_visits_site, n_species).
            - 'X_d' (torch.Tensor or None): Detection covariates for the site's
            visits, shape (n_visits_site, n_cov_d), or None if not provided.
            - 'X_psi' (torch.Tensor or None): Site-level occupancy covariates,
            shape (n_cov_psi,), or None if not provided.

    Notes:
        - The number of visits per site may vary.
        - Visit-level rows are selected dynamically using a boolean mask on
        `visit_site_idxs`.
        - This dataset is typically used with `collate_occu_sites` to form
        mini-batches that reconstruct a valid `OccuData` object.
    """
    def __init__(self, data: OccuData):
        self.data = data

        self.n_sites = data.n_sites
        self.site_ids = torch.arange(self.n_sites)
        # self.site_ids = torch.unique(data.visit_site_idxs) # unique site indices present in the data

    def __len__(self):
        return self.n_sites

    def __getitem__(self, idx):
        site = self.site_ids[idx] # scalar tensor

        mask = self.data.visit_site_idxs == site # shape (n_visits_sub,)

        y = self.data.y[mask] # shape (n_visits_sub, n_species)
        X_d = self.data.X_d[mask] if self.data.X_d is not None else None # shape (n_visits_sub, n_cov_d) or None

        X_psi = self.data.X_psi[site] if self.data.X_psi is not None else None # shape (n_cov_psi,) or None

        return {
            "site_id": site,
            "y": y,
            "X_d": X_d,
            "X_psi": X_psi,
        }
    
def collate_occu_sites(batch):
    """
    Collate function for batching site-level samples from OccuSiteDataset.

    This function merges multiple site samples into a single batched
    `OccuData` object by concatenating visit-level observations while
    rebuilding the `visit_site_idxs` mapping within the batch.

    Args:
        batch (list[dict]):
            List of samples returned by `OccuSiteDataset.__getitem__`.
            Each sample contains:

            - 'site_id' (torch.Tensor): Scalar site index.
            - 'y' (torch.Tensor): Detection matrix,
            shape (n_visits_site, n_species).
            - 'X_d' (torch.Tensor or None): Detection covariates,
            shape (n_visits_site, n_cov_d).
            - 'X_psi' (torch.Tensor or None): Site-level occupancy covariates,
            shape (n_cov_psi,).

    Returns:
        OccuData:
            A batched occupancy dataset with:

            - y: (total_n_visits, n_species)
            - visit_site_idxs: (total_n_visits,)
            - X_psi: (batch_size, n_cov_psi) or empty tensor if absent
            - X_d: (total_n_visits, n_cov_d) or empty tensor if absent

    Notes:
        - `visit_site_idxs` is rebuilt so that visit rows reference the
        corresponding site within the batch (range `[0, batch_size-1]`).
        - This preserves the canonical `OccuData` structure required by
        occupancy likelihood functions.
        - Supports variable numbers of visits per site without padding.
    """
    batch_size = len(batch)

    site_ids = torch.stack([b["site_id"] for b in batch]) # shape (batch_size,)

    # ---- state covariates
    if batch[0]["X_psi"] is not None:
        X_psi = torch.stack([b["X_psi"] for b in batch]) # shape (batch_size, n_cov_psi)
    else:
        X_psi = None

    visit_site_idxs = []
    y_list = []
    X_d_list = []
    site_ids_list = []
    
    for i, b in enumerate(batch):
        site_ids_list.append(b["site_id"])

        n_visits = b["y"].shape[0]

        visit_site_idxs.append(torch.full((n_visits,), i, dtype=torch.long))
        y_list.append(b["y"])

        if b["X_d"] is not None:
            X_d_list.append(b["X_d"])

    site_ids = torch.tensor(site_ids_list) # shape (batch_size,)
    visit_site_idxs = torch.cat(visit_site_idxs) # shape (total_n_visits,)
    y = torch.cat(y_list) # shape (total_n_visits, n_species)

    if b["X_d"] is not None:
        X_d = torch.cat(X_d_list) if X_d_list else None # shape (total_n_visits, n_cov_d)
    else:
        X_d = None

    return OccuData(
        y=y,
        visit_site_idxs=visit_site_idxs,
        X_psi=X_psi,
        X_d=X_d,
        site_ids=site_ids,
    )