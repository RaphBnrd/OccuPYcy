import math

import torch

from occupycy.data.core import OccuData
from occupycy.fit import fit_lbfgs_full_batch
from occupycy.models.constant import ConstantOccuModel


def simulate_constant(
    *,
    n_sites: int = 200,
    n_visits_per_site: int = 4,
    n_species: int = 3,
    psi: float = 0.6,
    d: float = 0.3,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)

    z = torch.bernoulli(torch.full((n_sites, n_species), psi), generator=g)
    visit_site_idxs = torch.arange(n_sites).repeat_interleave(n_visits_per_site)
    z_for_visit = z[visit_site_idxs]
    y = torch.bernoulli(torch.full((visit_site_idxs.shape[0], n_species), d), generator=g) * z_for_visit
    return y.float(), visit_site_idxs.long()


def test_fit_lbfgs():
    y, visit_site_idxs = simulate_constant(seed=123)
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs)

    model = ConstantOccuModel(n_species=data.n_species)
    res = fit_lbfgs_full_batch(data, model, max_iter=50)

    assert math.isfinite(res.nll)
    assert res.n_iter > 0