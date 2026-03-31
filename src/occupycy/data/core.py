from __future__ import annotations

from typing import Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset


# -------------------------------------------------------------
#  OccuData: canonical data container for occupancy modeling
# -------------------------------------------------------------

class OccuData():
    """
    Canonical data container for the single-season occupancy setup.
      - y: (n_visits, n_species) with 0/1
      - visit_site_ids: list of site ids (e.g. names) in for each visit, length n_visits
      - X_psi: (n_sites, n_cov_psi) optional site/state covariates
      - X_d: (n_visits, n_cov_d) optional visit/detection covariates
      - site_ids_in_X_psi: list of site ids corresponding to rows in X_psi, length n_sites. 
    """
    def __init__(
        self,
        *,
        y: torch.Tensor,
        visit_site_ids: list[int] | list[str],
        X_psi: Optional[torch.Tensor] = None,
        X_d: Optional[torch.Tensor] = None,
        site_ids_in_X_psi: Optional[list[int] | list[str]] = None
    ):

        self.y = torch.as_tensor(y)
        self.X_psi = torch.as_tensor(X_psi) if X_psi is not None else None
        self.X_d = torch.as_tensor(X_d) if X_d is not None else None
        
        if not isinstance(visit_site_ids, list):
            raise ValueError(f"visit_site_ids must be a list of site ids (e.g. names), got {type(visit_site_ids)}")

        if not (all([isinstance(e, str) for e in visit_site_ids]) or all([isinstance(e, int) for e in visit_site_ids])):
            raise ValueError("visit_site_ids must be a list of all strings or all integers.")

        if self.y.ndim != 2:
            raise ValueError(f"y must be 2D (n_visits, n_species); got {tuple(self.y.shape)}")
        if len(visit_site_ids) != self.y.shape[0]:
            raise ValueError(
                f"visit_site_ids length {len(visit_site_ids)} must match y n_visits {self.y.shape[0]}"
            )
        if self.X_d is not None:
            if self.X_d.ndim != 2:
                raise ValueError(f"X_d must be 2D (n_visits, n_cov_d); got {tuple(self.X_d.shape)}")
            if self.X_d.shape[0] != self.y.shape[0]:
                raise ValueError(f"X_d first dim ({self.X_d.shape[0]}) must match y n_visits ({self.y.shape[0]}).")
        if self.X_psi is not None:
            if self.X_psi.ndim != 2:
                raise ValueError(f"X_psi must be 2D (n_sites, n_cov_psi); got {tuple(self.X_psi.shape)}")
            if self.X_psi.shape[0] != len(set(visit_site_ids)):
                message = f"X_psi first dim ({self.X_psi.shape[0]}) must match number of unique site ids in visit_site_ids ({len(set(visit_site_ids))})."
                if self.X_psi.shape[0] > len(set(visit_site_ids)):
                    message += "\nIf X_psi has more rows than unique site ids, consider dropping unused rows in X_psi."
                else:
                    message += "\nIf X_psi has fewer rows than unique site ids, consider adding missing rows to X_psi or dropping visits with site ids that don't have corresponding rows in X_psi."
                raise ValueError(message)
        if site_ids_in_X_psi is not None:
            if len(set(site_ids_in_X_psi)) != len(site_ids_in_X_psi):
                raise ValueError("site_ids_in_X_psi must contain unique site ids (no duplicates).")
            if set(site_ids_in_X_psi) != set(visit_site_ids):
                raise ValueError("site_ids_in_X_psi must contain the same unique site ids as visit_site_ids.")
            if self.X_psi is not None:
                if len(site_ids_in_X_psi) != self.X_psi.shape[0]:
                    raise ValueError(f"site_ids_in_X_psi length ({len(site_ids_in_X_psi)}) must match X_psi first dim ({self.X_psi.shape[0]}).")

        # Usefull dimensions    
        self.n_sites = len(set(visit_site_ids))
        self.n_visits = len(visit_site_ids)
        self.n_species = self.y.shape[1]
        self.n_cov_psi = self.X_psi.shape[1] if self.X_psi is not None else 0
        self.n_cov_d = self.X_d.shape[1] if self.X_d is not None else 0

        # Dictionnary mapping: 
        #  - site ids in visit_site_ids 
        # to 
        # - site indices in X_psi (i.e. 0 to n_sites-1) and the values in indices_visit_site
        
        # self.mapping_site_idx = {initial_id: idx for idx, initial_id in enumerate(sorted(set(visit_site_ids)))}
        # unique_site_ids = []
        # for site_id in visit_site_ids:
        #     if site_id not in unique_site_ids:
        #         unique_site_ids.append(site_id)
        if site_ids_in_X_psi is not None:
            self.site_ids_in_X_psi = site_ids_in_X_psi
        elif self.X_psi is None:
            self.site_ids_in_X_psi = list(dict.fromkeys(visit_site_ids)) # preserves order of first occurrence
        else: # If site_ids_in_X_psi is None and X_psi is not None
            raise ValueError("If X_psi is provided, site_ids_in_X_psi must also be provided to map site ids to rows in X_psi.")
        self.mapping_site_idx = {initial_id: idx for idx, initial_id in enumerate(self.site_ids_in_X_psi)}
        
        # Reindex visit_site_ids to site indices in X_psi (i.e. 0 to n_sites-1)
        # indices = [self.mapping_site_idx[site_id] for site_id in visit_site_ids]
        mapping = self.mapping_site_idx.__getitem__
        indices = [mapping(s) for s in visit_site_ids]
        self.indices_visit_site = torch.tensor(
            indices,
            dtype=torch.long,
            device=self.y.device
        )

        # DataLoaders for train/val/test splits.
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def to(
        self, 
        *, 
        device: str | torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> "OccuData":
        def _to(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if device is not None:
                tensor = tensor.to(device=device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor
        self.y = _to(self.y)
        self.X_psi = _to(self.X_psi)
        self.X_d = _to(self.X_d)
        if device is not None: # indices_visit_site type should remain long, so only move to device if device is specified, but don't change dtype
            self.indices_visit_site = self.indices_visit_site.to(device=device)

    def build_dataloaders(
        self,
        *,
        batch_size: int = 32,
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        splitting: Optional[Dict[str, list]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        device: str | torch.device | None = None,
    ):
        """
        Returns a new OccuData with train/val/test DataLoaders created from the dataset.
        """
        if splitting is None:
            assert 0.0 <= val_fraction < 1.0, "val_fraction must be in [0.0, 1.0)"
            assert 0.0 <= test_fraction < 1.0, "test_fraction must be in [0.0, 1.0)"
            assert val_fraction + test_fraction < 1.0, "val_fraction + test_fraction must be < 1.0"

        if device is not None:
            self.to(device=device)
        dataset = OccuSiteDataset(self)
        n = len(dataset)
        
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        if splitting is None:
            n_val = int(n * val_fraction)
            n_test = int(n * test_fraction)
            n_train = n - n_val - n_test
            train_ds, val_ds, test_ds = torch.utils.data.random_split(
                dataset, [n_train, n_val, n_test], generator=generator
            )
        else:
            assert set(splitting.keys()) == {"train", "val", "test"}, "splitting keys must be 'train', 'val', 'test'"
            all_idxs = set(range(len(dataset)))
            train_idxs = set(self.mapping_site_idx[id] for id in set(splitting["train"]))
            val_idxs = set(self.mapping_site_idx[id] for id in set(splitting["val"]))
            test_idxs = set(self.mapping_site_idx[id] for id in set(splitting["test"]))
            assert train_idxs.isdisjoint(val_idxs), "train and val indices must be disjoint"
            assert train_idxs.isdisjoint(test_idxs), "train and test indices must be disjoint"
            assert val_idxs.isdisjoint(test_idxs), "val and test indices must be disjoint"
            assert train_idxs.union(val_idxs).union(test_idxs) == all_idxs, "train, val, test indices must cover all samples"
            if train_idxs:
                train_ds = Subset(dataset, sorted(train_idxs))
                n_train = len(train_ds)
            else:
                train_ds, n_train = None, 0
            if val_idxs:
                val_ds = Subset(dataset, sorted(val_idxs))
                n_val = len(val_ds)
            else:
                val_ds, n_val = None, 0
            if test_idxs:
                test_ds = Subset(dataset, sorted(test_idxs))
                n_test = len(test_ds)
            else:
                test_ds, n_test = None, 0

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, 
            collate_fn=collate_occu_sites, generator=generator
        ) if n_train else None

        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, 
            collate_fn=collate_occu_sites, generator=generator
        ) if n_val else None

        self.test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, 
            collate_fn=collate_occu_sites, generator=generator
        ) if n_test else None
    
    def reset_dataloaders(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def filtered_sites(self, site_id_mask: list) -> "OccuData":

        if len(site_id_mask) != len(set(site_id_mask)):
            print("Warning: site_mask contains duplicate indices. Duplicates are ignored.")
            site_id_mask = list(set(site_id_mask))
        
        indices_site_mask = [self.site_ids_in_X_psi.index(site_id) for site_id in site_id_mask]

        X_psi_new = self.X_psi[indices_site_mask] if self.X_psi is not None else None
        site_ids_in_X_psi_new = site_id_mask if self.X_psi is not None else None

        mapping_idxs_to_ids = {idx: site_id for site_id, idx in self.mapping_site_idx.items()}
        visit_site_ids = [mapping_idxs_to_ids[idx.item()] for idx in self.indices_visit_site]

        mask_v = torch.isin(self.indices_visit_site, torch.tensor(indices_site_mask))
        y_new = self.y[mask_v]
        visit_site_ids_new = [visit_site_ids[i] for i in torch.where(mask_v)[0].tolist()]
        X_d_new = self.X_d[mask_v] if self.X_d is not None else None
            
        return OccuData(
            y=y_new,
            visit_site_ids=visit_site_ids_new,
            X_psi=X_psi_new,
            X_d=X_d_new,
            site_ids_in_X_psi=site_ids_in_X_psi_new
        )


# -------------------------------------------------------------
# OccuSiteDataset and collate_occu_sites for DataLoader support
# -------------------------------------------------------------

class OccuSiteDataset(Dataset):
    """
    PyTorch Dataset that exposes site-level samples from an OccuData object.

    Each dataset item corresponds to a single site and returns all visits
    associated with that site. Visit-level observations and covariates are
    selected using the site's index in `indices_visit_site`.

    Args:
        data (OccuData):
            Occupancy dataset containing visit-level detections (`y`),
            site indices (`indices_visit_site`), and optional covariates
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
        `indices_visit_site`.
        - This dataset is typically used with `collate_occu_sites` to form
        mini-batches that reconstruct a valid `OccuData` object.
    """
    def __init__(self, data: OccuData):
        
        self.data = data
        
        self.site_ids = list(data.mapping_site_idx.keys())

    def __len__(self):
        return self.data.n_sites

    def __getitem__(self, idx):

        site_id = self.site_ids[idx] # site_id is the original site id (name)
        idx_site = self.data.mapping_site_idx[site_id] # scalar index corresponding to site_id in X_psi

        mask = self.data.indices_visit_site == idx_site # shape (n_visits,) with n_visits_sub True values corresponding to visits for the site of interest

        y = self.data.y[mask] # shape (n_visits_sub, n_species)
        X_d = self.data.X_d[mask] if self.data.X_d is not None else None # shape (n_visits_sub, n_cov_d) or None

        X_psi = self.data.X_psi[idx_site] if self.data.X_psi is not None else None # shape (n_cov_psi,) or None

        return {
            "site_id": site_id,
            "y": y,
            "X_d": X_d,
            "X_psi": X_psi,
        }
    
def collate_occu_sites(batch):
    """
    Collate function for batching site-level samples from OccuSiteDataset.

    This function merges multiple site samples into a single batched
    `OccuData` object by concatenating visit-level observations while
    rebuilding the `indices_visit_site` mapping within the batch.

    Args:
        batch (list[dict]):
            List of samples returned by `OccuSiteDataset.__getitem__`.
            Each sample contains:

            - 'site_id' (list): Original site id (name) for the sample.
            - 'idx_site' (torch.Tensor): Scalar tensor with the site index.
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
            - visit_site_ids: (total_n_visits,)
            - X_psi: (batch_size, n_cov_psi) or empty tensor if absent
            - X_d: (total_n_visits, n_cov_d) or empty tensor if absent

    Notes:
        - Supports variable numbers of visits per site without padding.
    """

    # Get the order of the batches based on the sorted set of site ids in the batch, to ensure consistent ordering of X_psi and indices_visit_site

    # batch = sorted(batch, key=lambda b: b["site_id"])

    if batch[0]["X_psi"] is not None:
        X_psi = torch.stack([b["X_psi"] for b in batch]) # shape (batch_size, n_cov_psi)
    else:
        X_psi = None

    site_ids_in_X_psi = []
    visit_site_ids = []
    y_list = []
    X_d_list = []
    
    for i, b in enumerate(batch):
        site_ids_in_X_psi.append(b["site_id"]) # add the site_id to the list of site ids in X_psi
        visit_site_ids += [b["site_id"]] * b["y"].shape[0] # add the site_id for each visit of the site
        y_list.append(b["y"])
        if b["X_d"] is not None:
            X_d_list.append(b["X_d"])
    y = torch.cat(y_list) # shape (total_n_visits, n_species)

    X_d = torch.cat(X_d_list) if X_d_list else None # shape (total_n_visits, n_cov_d)
    
    return OccuData(
        y=y,
        visit_site_ids=visit_site_ids,
        X_psi=X_psi,
        X_d=X_d,
        site_ids_in_X_psi=site_ids_in_X_psi
    )