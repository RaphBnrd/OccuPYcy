import pytest
import torch

from occupycy.data import OccuData


def _valid_payload():
    y = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]] # shape (3 visits, 2 species)
    visit_site_idxs = [0, 0, 1] # shape (3 visits,)
    X_psi = [[1.0], [2.0]] # shape (2 sites, 1 cov_psi)
    X_d = [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]] # shape (3 visits, 2 cov_d)
    return y, visit_site_idxs, X_psi, X_d


def test_occudata_coerces_to_tensors_and_properties_are_correct():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs, X_psi=X_psi, X_d=X_d)

    assert isinstance(data.y, torch.Tensor)
    assert isinstance(data.visit_site_idxs, torch.Tensor)
    assert isinstance(data.X_psi, torch.Tensor)
    assert isinstance(data.X_d, torch.Tensor)

    assert data.n_visits == 3
    assert data.n_species == 2
    assert data.n_sites == 2


def test_raises_if_y_is_not_2d():
    with pytest.raises(ValueError, match="y must be 2D"):
        OccuData(y=[0.0, 1.0], visit_site_idxs=[0, 0])


def test_raises_if_visit_site_idxs_is_not_1d():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="visit_site_idxs must be 1D"):
        OccuData(y=y, visit_site_idxs=[[0], [0]])


def test_raises_if_visit_site_length_mismatch():
    y = [[0.0], [1.0], [0.0]]
    with pytest.raises(ValueError, match="must match y n_visits"):
        OccuData(y=y, visit_site_idxs=[0, 0])


def test_raises_if_visit_site_idxs_not_integer_dtype():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="visit_site_idxs must be int32/int64"):
        OccuData(y=y, visit_site_idxs=torch.tensor([0.0, 1.0], dtype=torch.float32))


def test_raises_if_x_d_not_2d():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="X_d must be 2D"):
        OccuData(y=y, visit_site_idxs=[0, 0], X_d=[1.0, 2.0])


def test_raises_if_x_d_first_dim_mismatch():
    y = [[0.0], [1.0], [0.0]]
    with pytest.raises(ValueError, match="X_d first dim must match y n_visits"):
        OccuData(y=y, visit_site_idxs=[0, 1, 1], X_d=[[1.0], [2.0]])


def test_raises_if_x_psi_not_2d():
    y = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="X_psi must be 2D"):
        OccuData(y=y, visit_site_idxs=[0, 1], X_psi=[1.0, 2.0])


def test_raises_if_x_psi_rows_less_than_unique_sites():
    y = [[0.0], [1.0], [0.0]]
    with pytest.raises(ValueError, match="visit_site_idxs must be in"):
        OccuData(y=y, visit_site_idxs=[0, 1, 2], X_psi=[[1.0], [2.0]])


def test_to_with_dtype_changes_float_tensors_but_preserves_visit_site_integer_dtype():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=torch.tensor(y, dtype=torch.float64),
        visit_site_idxs=torch.tensor(visit_site_idxs, dtype=torch.int64),
        X_psi=torch.tensor(X_psi, dtype=torch.float64),
        X_d=torch.tensor(X_d, dtype=torch.float64),
    )

    converted = data.to(dtype=torch.float32)

    assert converted.y.dtype == torch.float32
    assert converted.X_psi is not None and converted.X_psi.dtype == torch.float32
    assert converted.X_d is not None and converted.X_d.dtype == torch.float32
    assert converted.visit_site_idxs.dtype == torch.int64


def test_to_handles_optional_covariates_as_none():
    y = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    visit_site_idxs = torch.tensor([0, 1], dtype=torch.int32)
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs)

    converted = data.to(dtype=torch.float32)

    assert converted.X_psi is None
    assert converted.X_d is None
    assert converted.y.dtype == torch.float32
    assert converted.visit_site_idxs.dtype == torch.int32

def test_to_returns_new_object():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(
        y=torch.tensor(y, dtype=torch.float64),
        visit_site_idxs=torch.tensor(visit_site_idxs, dtype=torch.int64),
        X_psi=torch.tensor(X_psi, dtype=torch.float64),
        X_d=torch.tensor(X_d, dtype=torch.float64)
    )
    
    converted = data.to(dtype=torch.float32)
    
    assert converted is not data
    assert data.y.dtype != converted.y.dtype

def test_with_dataloaders_returns_new_object_and_preserves_data():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs, X_psi=X_psi, X_d=X_d)

    new_data = data.with_dataloaders(batch_size=1)

    assert new_data is not data
    assert torch.equal(new_data.y, data.y)
    assert torch.equal(new_data.visit_site_idxs, data.visit_site_idxs)

    assert new_data.train_loader is not None
    
def test_train_dataloader_returns_occudata_batches():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs, X_psi=X_psi, X_d=X_d)

    data = data.with_dataloaders(batch_size=2)

    batch = next(iter(data.train_loader))

    assert isinstance(batch, OccuData)
    assert batch.y.ndim == 2
    assert batch.visit_site_idxs.ndim == 1
    
def test_collate_rebuilds_visit_site_idxs_within_batch():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs, X_psi=X_psi, X_d=X_d)

    data = data.with_dataloaders(batch_size=2)

    batch = next(iter(data.train_loader))

    assert batch.visit_site_idxs.min() >= 0
    assert batch.visit_site_idxs.max() < batch.n_sites

def test_with_dataloaders_split_sizes():
    y, visit_site_idxs, X_psi, X_d = _valid_payload()
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs, X_psi=X_psi, X_d=X_d)

    data = data.with_dataloaders(val_fraction=0.5, test_fraction=0.0)

    assert data.train_loader is not None
    assert data.val_loader is not None
    assert data.test_loader is None

def test_with_dataloaders_seed_reproducibility():
    
    y = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], 
         [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], 
         [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]] # shape: (24, 1)
    visit_site_idxs = [0, 0,  1,  2,  2,  2,  3,  2, 
                       4, 4,  4,  4,  5,  5,  6,  7,
                       8, 9, 10, 11, 12, 13, 14, 15] # shape: (24,)
    
    data1 = OccuData(y=y, visit_site_idxs=visit_site_idxs)
    data2 = OccuData(y=y, visit_site_idxs=visit_site_idxs)

    data1 = data1.with_dataloaders(seed=123, batch_size=4, val_fraction=0.4, test_fraction=0.0)
    data2 = data2.with_dataloaders(seed=123, batch_size=4, val_fraction=0.4, test_fraction=0.0)
    
    train_sites1 = []
    for b in data1.train_loader:
        train_sites1 += b.site_ids.tolist()
    train_sites2 = []
    for b in data2.train_loader:
        train_sites2 += b.site_ids.tolist()
    assert train_sites1 == train_sites2

    val_sites1 = []
    for b in data1.val_loader:
        val_sites1 += b.site_ids.tolist()
    val_sites2 = []
    for b in data2.val_loader:
        val_sites2 += b.site_ids.tolist()
    assert val_sites1 == val_sites2
