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
    indices_visit_site = torch.arange(n_sites).repeat_interleave(n_visits_per_site)
    z_for_visit = z[indices_visit_site]
    y = torch.bernoulli(torch.full((indices_visit_site.shape[0], n_species), d), generator=g) * z_for_visit
    return y.float(), indices_visit_site.long().tolist()


def test_fit_lbfgs():
    y, visit_site_ids = simulate_constant(seed=123)
    data = OccuData(y=y, visit_site_ids=visit_site_ids)

    model = ConstantOccuModel(n_species=data.n_species)
    res = fit_lbfgs_full_batch(data, model, max_iter=50)

    assert math.isfinite(res.final_train_loss)
    assert res.n_epochs > 0
