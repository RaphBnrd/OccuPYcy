from __future__ import annotations

import warnings

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# -------------------------------------------------------------
#  OccuData: canonical data container for occupancy modeling
# -------------------------------------------------------------

class OccuData:
    """
    Canonical data container for single-season occupancy modeling.
    The representation can be sparse (by defining y_visits and y_labels) or dense (by defining y).

    Parameters
    ----------
    y : torch.Tensor or None, default=None
        Visit-level detections, shape ``(n_visits, n_species)``.
    y_visits : torch.Tensor or None, default=None
        Visit label of each observation, shape ``(n_detections,)`` 
        representing the sparse representation of the detection matrix y. 
        Each entry is an integer in [0, n_visits-1] indicating which 
        visit the detection corresponds to.
    y_labels : torch.Tensor or None, default=None
        Species label of each observation, shape ``(n_detections,)``
        representing the sparse representation of the detection matrix y.
        Each entry is an integer in [0, n_species-1] indicating which
        species the detection corresponds to.
    species_ids_ordered : list[int] or list[str]
        Species identifiers corresponding to columns of y.
    visit_site_ids : list[int] or list[str]
        Site identifier per visit.
    X_psi : torch.Tensor or None, default=None
        Site covariates, shape ``(n_sites, n_cov_psi)``.
    X_d : torch.Tensor or None, default=None
        Visit covariates, shape ``(n_visits, n_cov_d)``.
    site_ids_in_X_psi : list[int] or list[str] or None, default=None
        Site IDs aligned with rows of ``X_psi``.
    """
    def __init__(
        self,
        *,
        y: torch.Tensor | None = None,
        y_visits: torch.Tensor | None = None,
        y_labels: torch.Tensor | None = None,
        visit_site_ids: list[int] | list[str],
        species_ids_ordered: list[int] | list[str] | None = None,
        X_psi: torch.Tensor | None = None,
        X_d: torch.Tensor | None = None,
        site_ids_in_X_psi: list[int] | list[str] | None = None,
        verbose: bool = False
    ):

        if verbose:
            print("Initializing OccuData...")
        
        if y is not None:
            if y_visits is not None or y_labels is not None:
                raise ValueError("If y is provided, y_visits and y_labels must be None.")
            self.y = torch.as_tensor(y) # shape (n_visits, n_species)
            # y should be in [0, 1] for binary occupancy modeling
            if not torch.all((self.y == 0) | (self.y == 1)):
                warnings.warn("y contains values other than 0 and 1. Ensure that y is binary for occupancy modeling.")
            self.y_visits = None
            self.y_labels = None
            self.sparse = False
        else:
            if y_visits is None or y_labels is None:
                raise ValueError("If y is None, both y_visits and y_labels must be provided.")
            self.y = None
            if len(y_visits) != len(y_labels):
                raise ValueError("y_visits and y_labels must have the same length.")
            self.y_visits = torch.as_tensor(y_visits) # shape (n_detections,), entries in [0, n_visits-1]
            self.y_labels = torch.as_tensor(y_labels) # shape (n_detections,), entries in [0, n_species-1]                
            # there should be no repetition in y_visits and y_labels for the same detection
            if len(set(zip(self.y_visits.tolist(), self.y_labels.tolist()))) != len(self.y_visits):
                warnings.warn("Duplicate (visit, species) pairs found in (y_visits, y_labels). Ensure that each detection is unique.")
            if species_ids_ordered is None:
                raise ValueError("species_ids_ordered must be provided when y is None.")
            self.sparse = True

        # --- Store inputs as tensors within attributes ---
        
        self.X_psi = torch.as_tensor(X_psi) if X_psi is not None else None # shape (n_sites, n_cov_psi)
        self.X_d = torch.as_tensor(X_d) if X_d is not None else None # shape (n_visits, n_cov_d)

        # --- Checks input types and dimensions ---

        # visit_site_ids
        if not isinstance(visit_site_ids, list):
            raise TypeError(
                f"visit_site_ids must be a list of site ids (e.g. names),"
                f" got {type(visit_site_ids)}"
            )
        if not (
            all(isinstance(e, str) for e in visit_site_ids) 
            or all(isinstance(e, int) for e in visit_site_ids)
        ):
            raise TypeError("visit_site_ids must be a list of all strings or all integers.")

        if not self.sparse:
            if self.y.ndim != 2:
                raise ValueError(f"y must be 2D (n_visits, n_species); got {tuple(self.y.shape)}")
            if len(visit_site_ids) != self.y.shape[0]:
                raise ValueError(
                    f"visit_site_ids length {len(visit_site_ids)} must match y n_visits {self.y.shape[0]}"
                )
        else:
            # y_visits
            if self.y_visits.ndim != 1:
                raise ValueError(f"y_visits must be 1D (n_detections,); got shape {tuple(self.y_visits.shape)}")
            if self.y_visits.numel() == 0:
                self.y_visits = torch.empty(0, dtype=torch.long, device=self.y_visits.device)
            elif self.y_visits.dtype not in (torch.int32, torch.int64, torch.long):
                raise TypeError(f"y_visits must be a tensor of integers; got {self.y_visits.dtype}")
            if self.y_visits.numel() > 0 and (
                self.y_visits.min() < 0 or self.y_visits.max() >= len(visit_site_ids)
            ):
                raise ValueError(
                    f"y_visits entries must be in [0, n_visits-1] where n_visits={len(visit_site_ids)}; "
                    f"got min {self.y_visits.min().item()}, max {self.y_visits.max().item()}"
                )
            
            # y_labels
            if self.y_labels.ndim != 1:
                raise ValueError(f"y_labels must be 1D (n_detections,); got shape {tuple(self.y_labels.shape)}")
            if self.y_labels.numel() == 0:
                self.y_labels = torch.empty(0, dtype=torch.long, device=self.y_labels.device)
            elif self.y_labels.dtype not in (torch.int32, torch.int64, torch.long):
                raise TypeError(f"y_labels must be a tensor of integers; got {self.y_labels.dtype}")
            if self.y_labels.numel() > 0 and (
                self.y_labels.min() < 0 or self.y_labels.max() >= len(species_ids_ordered)
            ):
                raise ValueError(
                    f"y_labels entries must be in [0, n_species-1] where n_species={len(species_ids_ordered)}; "
                    f"got min {self.y_labels.min().item()}, max {self.y_labels.max().item()}"
                )
            if len(set(species_ids_ordered)) != len(species_ids_ordered):
                raise ValueError("species_ids_ordered must contain unique species ids (no duplicates).")

        # X_d
        if self.X_d is not None:
            if self.X_d.ndim != 2:
                raise ValueError(f"X_d must be 2D (n_visits, n_cov_d); got {tuple(self.X_d.shape)}")
            if self.X_d.shape[0] != len(visit_site_ids):
                raise ValueError(f"X_d first dim ({self.X_d.shape[0]}) must match length of visit_site_ids ({len(visit_site_ids)}).")
        
        # X_psi
        if self.X_psi is not None:
            if self.X_psi.ndim != 2:
                raise ValueError(f"X_psi must be 2D (n_sites, n_cov_psi); got {tuple(self.X_psi.shape)}")
            if self.X_psi.shape[0] != len(set(visit_site_ids)):
                message = f"X_psi first dim ({self.X_psi.shape[0]}) must match number of unique site ids in visit_site_ids ({len(set(visit_site_ids))})."
                if self.X_psi.shape[0] > len(set(visit_site_ids)):
                    message += "\nX_psi has more rows than unique site ids, consider dropping unused rows in X_psi."
                else:
                    message += "\nX_psi has fewer rows than unique site ids, consider adding missing rows to X_psi or dropping visits with site ids that don't have corresponding rows in X_psi."
                raise ValueError(message)
        if site_ids_in_X_psi is not None:
            if len(set(site_ids_in_X_psi)) != len(site_ids_in_X_psi):
                raise ValueError("site_ids_in_X_psi must contain unique site ids (no duplicates).")
            if set(site_ids_in_X_psi) != set(visit_site_ids):
                raise ValueError("site_ids_in_X_psi must contain the same unique site ids as visit_site_ids.")
            if self.X_psi is not None and len(site_ids_in_X_psi) != self.X_psi.shape[0]:
                raise ValueError(f"site_ids_in_X_psi length ({len(site_ids_in_X_psi)}) must match X_psi first dim ({self.X_psi.shape[0]}).")


        # --- Store additional attributes ---

        # Initialize splitting with all sites in train by default (can be updated later with build_dataloaders)
        self.splitting = {
            "train": list(set(visit_site_ids)),
            "val": [],
            "test": []
        }

        # Usefull dimensions    
        self.n_sites = len(set(visit_site_ids))
        self.n_visits = len(visit_site_ids)
        if self.sparse: # non-sparse representation
            self.n_species = len(species_ids_ordered)
        else:
            self.n_species = self.y.shape[1]
        self.n_cov_psi = self.X_psi.shape[1] if self.X_psi is not None else 0
        self.n_cov_d = self.X_d.shape[1] if self.X_d is not None else 0

        # Dictionnary mapping:
        # - species ids in species_ids_ordered
        # to
        # - species indices in y (i.e. 0 to n_species-1)
        if species_ids_ordered is None:
            self.species_ids_ordered = list(range(self.n_species))
        else:
            self.species_ids_ordered = species_ids_ordered
        self.mapping_species_idx = {initial_id: idx for idx, initial_id in enumerate(self.species_ids_ordered)}

        # Dictionnary mapping: 
        #  - site ids in visit_site_ids 
        # to 
        # - site indices in X_psi (i.e. 0 to n_sites-1) and the values in indices_visit_site
        if site_ids_in_X_psi is not None:
            self.site_ids_in_X_psi = site_ids_in_X_psi
        elif self.X_psi is None: # and site_ids_in_X_psi is None
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
            device=self.y.device if not self.sparse else self.y_visits.device
        )
        # Init _masks to None if not used
        self._mask_visits_each_site = None
        self._mask_detections_each_site = None

        # DataLoaders for train/val/test splits.
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    @property
    def mask_visits_each_site(self):
        """
        Visit indices grouped by site index.

        Returns
        -------
        list[torch.Tensor]
            ``i``-th tensor contains visit indices belonging to site ``i``.
        """
        if self._mask_visits_each_site is None:
            # self._mask_visits_each_site = [
            #     torch.where(self.indices_visit_site == i)[0]
            #     for i in tqdm(range(self.n_sites), desc="Building mask_visits_each_site")
            # ]
            order = torch.argsort(self.indices_visit_site, stable=True)
            counts = torch.bincount(self.indices_visit_site, minlength=self.n_sites)
            self._mask_visits_each_site = list(torch.split(order, counts.tolist()))
        return self._mask_visits_each_site
    
    @property
    def mask_detections_each_site(self):
        """
        Detection indices grouped by site index.

        Returns
        -------
        list[torch.Tensor]
            ``i``-th tensor contains detection indices belonging to site ``i``.
        """
        if self._mask_detections_each_site is None:
            # # Get the visit indices for each site
            # mask_visits_each_site = self.mask_visits_each_site
            # # Then, for each site, get the detection indices corresponding to those visits
            # self._mask_detections_each_site = [
            #     torch.where(torch.isin(self.y_visits, mask_visits_each_site[i]))[0]
            #     for i in tqdm(range(self.n_sites), desc="Building mask_detections_each_site")
            # 
            site_of_visit = self.indices_visit_site[self.y_visits]
            self._mask_detections_each_site = [
                torch.where(site_of_visit == i)[0]
                for i in tqdm(range(self.n_sites), desc="Building mask_detections_each_site", position=1, leave=False)
            ]
        return self._mask_detections_each_site

    def to(
        self, 
        *, 
        device: str | torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Move internal tensors to a given device/dtype.
        Parameters
        ----------
        device : str or torch.device or None, default=None
            Device to move tensors to (e.g. "cpu", "cuda"). If None, no device move is performed.
        dtype : torch.dtype or None, default=None
            Dtype to move tensors to (e.g. torch.float32). If None, no dtype change is performed.
        Returns
        -------
        None
        """
        def _to(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if device is not None:
                tensor = tensor.to(device=device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor
        if not self.sparse:
            self.y = _to(self.y)
        self.X_psi = _to(self.X_psi)
        self.X_d = _to(self.X_d)
        if device is not None: # indices_visit_site type should remain long, so only move to device if device is specified, but don't change dtype
            self.indices_visit_site = self.indices_visit_site.to(device=device)
            if self.sparse:
                self.y_visits = self.y_visits.to(device=device)
                self.y_labels = self.y_labels.to(device=device)
        # Move the cached masks to the new device if they exist
        if self._mask_visits_each_site is not None:
            self._mask_visits_each_site = [
                mask.to(device=device) for mask in self._mask_visits_each_site
            ]
        if self._mask_detections_each_site is not None:
            self._mask_detections_each_site = [
                mask.to(device=device) for mask in self._mask_detections_each_site
            ]

    def build_dataloaders(
        self,
        *,
        batch_size: int = 32,
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        splitting: dict[str, list] | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Create train/validation/test dataloaders at site level.

        Parameters
        ----------
        batch_size : int, default=32
            Number of sites per batch.
        val_fraction : float, default=0.0
            Validation split fraction when ``splitting`` is not provided.
        test_fraction : float, default=0.0
            Test split fraction when ``splitting`` is not provided.
        splitting : dict[str, list] or None, default=None
            Explicit split with keys ``train``, ``val``, ``test``.
        shuffle : bool, default=True
            Whether to shuffle train batches.
        seed : int or None, default=None
            Random seed for split/shuffle.
        device : str or torch.device or None, default=None
            Optional device move before creating loaders.

        Returns
        -------
        None
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
            # Get the ids corresponding to each split
            splitting = {
                "train": [dataset[i]["site_id"] for i in train_ds.indices],
                "val": [dataset[i]["site_id"] for i in val_ds.indices],
                "test": [dataset[i]["site_id"] for i in test_ds.indices],
            }
        else:
            assert set(splitting.keys()) == {"train", "val", "test"}, "splitting keys must be 'train', 'val', 'test'"
            all_idxs = set(range(len(dataset)))
            train_idxs = {self.mapping_site_idx[id] for id in set(splitting["train"])}
            val_idxs = {self.mapping_site_idx[id] for id in set(splitting["val"])}
            test_idxs = {self.mapping_site_idx[id] for id in set(splitting["test"])}
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

        self.splitting = splitting

        self.train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=lambda batch: collate_occu_sites(batch, sparse=self.sparse), 
            generator=generator
        ) if n_train else None

        self.val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: collate_occu_sites(batch, sparse=self.sparse), 
            generator=generator
        ) if n_val else None

        self.test_loader = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: collate_occu_sites(batch, sparse=self.sparse), 
            generator=generator
        ) if n_test else None
    
    def reset_dataloaders(self):
        """
        Clear train/validation/test dataloaders.

        Returns
        -------
        None
        """
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def filtered_sites(self, site_id_mask: list) -> OccuData:
        """
        Return a new ``OccuData`` restricted to selected site IDs.

        Parameters
        ----------
        site_id_mask : list
            Site IDs to keep.

        Returns
        -------
        OccuData
            Filtered dataset.
        """

        if len(site_id_mask) != len(set(site_id_mask)):
            warnings.warn(
                "site_mask contains duplicate indices. Duplicates are ignored.",
                UserWarning
            )
            site_id_mask = list(set(site_id_mask))
        
        indices_site_mask = [self.site_ids_in_X_psi.index(site_id) for site_id in site_id_mask]

        X_psi_new = self.X_psi[indices_site_mask] if self.X_psi is not None else None
        site_ids_in_X_psi_new = site_id_mask if self.X_psi is not None else None

        mapping_idxs_to_ids = {idx: site_id for site_id, idx in self.mapping_site_idx.items()}
        visit_site_ids = [mapping_idxs_to_ids[idx.item()] for idx in self.indices_visit_site]

        mask_visits = torch.isin(
            self.indices_visit_site, 
            torch.tensor(indices_site_mask, dtype=torch.long, device=self.indices_visit_site.device)
        ) # shape (n_visits,) with True for visits corresponding to the selected sites
        visit_site_ids_new = [visit_site_ids[i] for i in torch.where(mask_visits)[0].tolist()]
        X_d_new = self.X_d[mask_visits] if self.X_d is not None else None

        if self.sparse:
            visits_ids_masked = torch.where(mask_visits)[0] # shape (n_visits_masked,) with indices of visits corresponding to the selected sites
            mask_detections = torch.where(torch.isin(self.y_visits, visits_ids_masked))[0] # shape (n_detections_masked,) with True for detections corresponding to the selected sites

            reindexing_visits = { # Visit reindexing for the sparse representation of y, 
                # to ensure that all visit indices are contiguous and start from 0,
                # including visits with no detections
                old_idx.item(): new_idx
                for new_idx, old_idx in enumerate(visits_ids_masked)
            }
            y_visits_new = torch.tensor(
                [reindexing_visits[old_idx.item()] for old_idx in self.y_visits[mask_detections]],
                dtype=torch.long,
                device=self.y_visits.device
            )
            y_labels_new = self.y_labels[mask_detections]

            return OccuData(
                y_visits=y_visits_new,
                y_labels=y_labels_new,
                visit_site_ids=visit_site_ids_new,
                X_psi=X_psi_new,
                X_d=X_d_new,
                site_ids_in_X_psi=site_ids_in_X_psi_new,
                species_ids_ordered=self.species_ids_ordered,
                verbose=False
            )
        
        else:
            y_new = self.y[mask_visits]
            
            return OccuData(
                y=y_new,
                visit_site_ids=visit_site_ids_new,
                X_psi=X_psi_new,
                X_d=X_d_new,
                site_ids_in_X_psi=site_ids_in_X_psi_new,
                species_ids_ordered=self.species_ids_ordered,
                verbose=False
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
            Occupancy dataset containing visit-level detections (`y`)
            or sparse representation (`y_visits`, `y_labels`),
            site indices (`indices_visit_site`), and optional covariates
            (`X_psi`, `X_d`).
        sparse (bool, optional):
            If True, the dataset will return sparse representation of the data.

    Returns:
        dict:
            A dictionary containing tensors for a single site:

            - 'site_id' (torch.Tensor): Scalar tensor with the site index.
            - 'y' or ('y_visits', 'y_labels') (torch.Tensor):
            Detection matrix for the site with shape (n_visits_site, n_species)
            if `sparse=False`, or sparse representation if `sparse=True`.
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
        """
        Number of sites.

        Returns
        -------
        int
        """
        return self.data.n_sites

    def __getitem__(self, idx):
        """
        Fetch one site sample with all associated visits.

        Parameters
        ----------
        idx : int
            Site position.

        Returns
        -------
        dict
            Dictionary with keys ``site_id``, ``y``, ``X_d``, ``X_psi``.
        """
        site_id = self.site_ids[idx] # site_id is the original site id (name)
        idx_site = self.data.mapping_site_idx[site_id] # scalar index corresponding to site_id in X_psi

        # mask = self.data.indices_visit_site == idx_site # shape (n_visits,) with n_visits_sub True values corresponding to visits for the site of interest
        mask_visits = self.data.mask_visits_each_site[idx_site]
        # shape (n_visits,) with n_visits_sub True values corresponding to visits for the site of interest
        
        if self.data.sparse:
            # shape (n_visits_sub,) indices of visits corresponding to the site of interest
            mask_detections = self.data.mask_detections_each_site[idx_site].to(device=self.data.y_visits.device) 
            # shape (n_detections,) with n_detections_sub True values corresponding to detections for the site of interest
            y_visits = self.data.y_visits[mask_detections] # shape (n_detections_sub,)
            y_labels = self.data.y_labels[mask_detections] # shape (n_detections_sub,)
            y = (
                y_visits, 
                y_labels, 
                mask_visits, 
                self.data.n_species
            ) # sparse representation of y for the site of interest
        else:
            y = self.data.y[mask_visits] # shape (n_visits_sub, n_species)

        X_d = self.data.X_d[mask_visits] if self.data.X_d is not None else None # shape (n_visits_sub, n_cov_d) or None

        X_psi = self.data.X_psi[idx_site] if self.data.X_psi is not None else None # shape (n_cov_psi,) or None

        return {
            "site_id": site_id,
            "y": y,
            "X_d": X_d,
            "X_psi": X_psi,
        }
    
def collate_occu_sites(batch, sparse: bool = False):
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
            shape (n_visits_site, n_species) if `sparse=False`, 
            or sparse representation if `sparse=True`.
            - 'X_d' (torch.Tensor or None): Detection covariates,
            shape (n_visits_site, n_cov_d).
            - 'X_psi' (torch.Tensor or None): Site-level occupancy covariates,
            shape (n_cov_psi,).

    Returns:
        Dict[str, torch.Tensor | list]:
            Batched data with keys:
            - y: (total_n_visits, n_species) or sparse representation (y_visits, y_labels)
            - indices_visit_site: (total_n_visits,) mapping each visit to its site index
            - X_psi: (batch_size, n_cov_psi) or empty tensor if absent
            - X_d: (total_n_visits, n_cov_d) or empty tensor if absent
            - site_ids_in_X_psi: list of site ids corresponding to rows of X_psi
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
    if sparse:
        y_visits_list = []
        y_labels_list = []
    else:
        y_list = []
    X_d_list = []
    indices_visit_site = [] # For each visit, the index of the site in the batch (0 to batch_size-1)

    for i, b in enumerate(batch):
        site_ids_in_X_psi.append(b["site_id"]) # add the site_id to the list of site ids in X_psi

        if sparse:
            reindexing_visits = { # Visit reindexing for the sparse representation of y, 
                # to ensure that all visit indices are contiguous and start from 0 for each batch,
                # including visits with no detections
                old_idx: new_idx + len(indices_visit_site) 
                for new_idx, old_idx in enumerate(b["y"][2].tolist())
            }
            visit_site_ids += [b["site_id"]] * len(b["y"][2]) # add the site_id for each visit of the site
            indices_visit_site += [i] * len(b["y"][2]) # add the site index for each visit of the site
            y_visits_list.append(torch.tensor(
                [reindexing_visits[old_idx.item()] for old_idx in b["y"][0]], 
                dtype=torch.long,
                device=b["y"][0].device
            ))
            y_labels_list.append(b["y"][1])
        else:
            visit_site_ids += [b["site_id"]] * b["y"].shape[0] # add the site_id for each visit of the site
            indices_visit_site += [i] * b["y"].shape[0] # add the site index for each visit of the site
            y_list.append(b["y"])

        if b["X_d"] is not None:
            X_d_list.append(b["X_d"])

    if sparse:
        y_visits = torch.cat(y_visits_list) # shape (total_n_detections,)
        y_labels = torch.cat(y_labels_list) # shape (total_n_detections,)
        y = (y_visits, y_labels) # sparse representation of y for the batch
    else:
        y = torch.cat(y_list) # shape (total_n_visits, n_species)

    X_d = torch.cat(X_d_list) if X_d_list else None # shape (total_n_visits, n_cov_d)
    
    indices_visit_site = torch.tensor(
        indices_visit_site, 
        dtype=torch.long, 
        device=y[0].device if sparse else y.device
    )

    return {
        "y": y,
        "indices_visit_site": indices_visit_site,
        "X_psi": X_psi,
        "X_d": X_d,
        "site_ids_in_X_psi": site_ids_in_X_psi
    }