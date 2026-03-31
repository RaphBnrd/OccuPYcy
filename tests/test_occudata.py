import pytest
import torch
from copy import deepcopy

from occupycy.data import OccuData


def _valid_payload():
    y = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]] # shape (3 visits, 2 species)
    visit_site_ids = ["A", "A", "B"] # shape (3 visits,)
    site_ids_in_X_psi = ["A", "B"] # shape (2 sites,)
    X_psi = [[1.0], [2.0]] # shape (2 sites, 1 cov_psi)
    X_d = [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]] # shape (3 visits, 2 cov_d)
    return y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d


def test_occudata_coerces_to_tensors_and_properties_are_correct():
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=y, 
        visit_site_ids=visit_site_ids, 
        X_psi=X_psi, 
        X_d=X_d,
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    assert isinstance(data.y, torch.Tensor)
    assert isinstance(data.indices_visit_site, torch.Tensor)
    assert isinstance(data.X_psi, torch.Tensor)
    assert isinstance(data.X_d, torch.Tensor)

    assert data.n_visits == 3
    assert data.n_species == 2
    assert data.n_sites == 2


def test_raises_if_y_is_not_2d():
    with pytest.raises(ValueError, match="y must be 2D"):
        OccuData(y=[0.0, 1.0], visit_site_ids=[0, 0])


def test_raises_if_visit_site_ids_is_not_1d():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="visit_site_ids must be a list of "):
        OccuData(y=y, visit_site_ids=[[0], [0]])


def test_raises_if_visit_site_length_mismatch():
    y = [[0.0], [1.0], [0.0]]
    with pytest.raises(ValueError, match="must match y n_visits"):
        OccuData(y=y, visit_site_ids=[0, 0])


def test_raises_if_x_d_not_2d():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="X_d must be 2D"):
        OccuData(y=y, visit_site_ids=[0, 0], X_d=[1.0, 2.0])


def test_raises_if_x_d_first_dim_mismatch():
    y = [[0.0], [1.0], [0.0]]
    with pytest.raises(ValueError, match="X_d first dim"):
        OccuData(y=y, visit_site_ids=[0, 1, 1], X_d=[[1.0], [2.0]])


def test_raises_if_x_psi_not_2d():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="X_psi must be 2D"):
        OccuData(y=y, visit_site_ids=[0, 1], X_psi=[1.0, 2.0], site_ids_in_X_psi=[0, 1])


def test_raises_if_x_psi_rows_less_than_unique_sites():
    y = [[0.0], [1.0], [0.0]]
    with pytest.raises(ValueError, match="X_psi first dim"):
        OccuData(y=y, visit_site_ids=[0, 1, 2], X_psi=[[1.0], [2.0]], site_ids_in_X_psi=[0, 1, 2])


def test_to_with_dtype_changes_float_tensors_but_preserves_visit_site_integer_dtype():
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=torch.tensor(y, dtype=torch.float64),
        visit_site_ids=visit_site_ids,
        X_psi=torch.tensor(X_psi, dtype=torch.float64),
        X_d=torch.tensor(X_d, dtype=torch.float64),
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    data.to(dtype=torch.float32)

    assert data.y.dtype == torch.float32
    assert data.X_psi is not None and data.X_psi.dtype == torch.float32
    assert data.X_d is not None and data.X_d.dtype == torch.float32
    assert data.indices_visit_site.dtype == torch.int64


def test_to_handles_optional_covariates_as_none():
    y = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    visit_site_ids = [0, 1]
    data = OccuData(y=y, visit_site_ids=visit_site_ids)

    data.to(dtype=torch.float32)

    assert data.X_psi is None
    assert data.X_d is None
    assert data.y.dtype == torch.float32
    assert data.indices_visit_site.dtype == torch.int64

def test_to_returns_new_object():
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=torch.tensor(y, dtype=torch.float64),
        visit_site_ids=visit_site_ids,
        X_psi=torch.tensor(X_psi, dtype=torch.float64),
        X_d=torch.tensor(X_d, dtype=torch.float64),
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    initial = deepcopy(data)
    
    data.to(dtype=torch.float32)
    
    assert data is not initial
    assert initial.y.dtype != data.y.dtype

def test_train_dataloader_returns_dict_batches():
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=y, 
        visit_site_ids=visit_site_ids, 
        X_psi=X_psi, 
        X_d=X_d, 
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    data.build_dataloaders(batch_size=2)

    batch = next(iter(data.train_loader))

    assert isinstance(batch, dict)
    assert batch['y'].ndim == 2
    assert batch['indices_visit_site'].ndim == 1
    assert batch['X_psi'] is None or batch['X_psi'].ndim == 2
    assert batch['X_d'] is None or batch['X_d'].ndim == 2

def test_collate_rebuilds_visit_site_ids_within_batch():
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=y,
        visit_site_ids=visit_site_ids,
        X_psi=X_psi,
        X_d=X_d,
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    data.build_dataloaders(batch_size=2)

    batch = next(iter(data.train_loader))

    assert set(batch['indices_visit_site'].unique().tolist()) == \
        set(range(torch.unique(batch['indices_visit_site']).numel()))

def test_build_dataloaders_split_sizes():
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=y,
        visit_site_ids=visit_site_ids,
        X_psi=X_psi,
        X_d=X_d,
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    data.build_dataloaders(val_fraction=0.5, test_fraction=0.0)

    assert data.train_loader is not None
    assert data.val_loader is not None
    assert data.test_loader is None

def test_build_dataloaders_seed_reproducibility():
    
    y = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], 
         [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], 
         [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]] # shape: (24, 1)
    visit_site_ids = [0, 0,  1,  2,  2,  2,  3,  2, 
                       4, 4,  4,  4,  5,  5,  6,  7,
                       8, 9, 10, 11, 12, 13, 14, 15] # shape: (24,)
    visit_site_ids = [f"site_{i}" for i in visit_site_ids]
    
    data1 = OccuData(y=y, visit_site_ids=visit_site_ids)
    data2 = OccuData(y=y, visit_site_ids=visit_site_ids)

    data1.build_dataloaders(seed=123, batch_size=4, val_fraction=0.4, test_fraction=0.0)
    data2.build_dataloaders(seed=123, batch_size=4, val_fraction=0.4, test_fraction=0.0)
    
    train_sites1 = []
    for b in data1.train_loader:
        train_sites1 += list(b['site_ids_in_X_psi'])
    train_sites2 = []
    for b in data2.train_loader:
        train_sites2 += list(b['site_ids_in_X_psi'])
    assert train_sites1 == train_sites2

    val_sites1 = []
    for b in data1.val_loader:
        val_sites1 += list(b['site_ids_in_X_psi'])
    val_sites2 = []
    for b in data2.val_loader:
        val_sites2 += list(b['site_ids_in_X_psi'])
    assert val_sites1 == val_sites2
