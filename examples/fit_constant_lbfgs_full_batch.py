import torch

from occupycy.data.core import OccuData
from occupycy.fit import fit_lbfgs_full_batch
from occupycy.models.constant import ConstantOccuModel


def simulate_constant(
        n_sites=300, 
        n_visits_per_site=10, 
        psi=torch.tensor([0.7, 0.5, 0.3, 0.1, 0.05]),
        d=torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]),
        seed=123
    ):
    g = torch.Generator().manual_seed(seed)
    n_species = psi.shape[0]
    z = torch.bernoulli(psi.expand(n_sites, n_species), generator=g)
    visit_site_idxs = torch.arange(n_sites).repeat_interleave(n_visits_per_site)
    z_for_visit = z[visit_site_idxs]
    y = torch.bernoulli(d.expand(visit_site_idxs.shape[0], n_species), generator=g) * z_for_visit

    print("=== Simulated data summary ===")
    print(f"  n_sites={n_sites}, n_visits_per_site={n_visits_per_site}, n_species={n_species}")
    print(f"  true psi={psi}, true d={d}")
    print(f"  observed psi (knowing true occupancy): {z.mean(dim=0)}")
    print(f"  observed d (knowing true occupancy):   {y.sum(dim=0) / z_for_visit.sum(dim=0)}")

    return y.float(), visit_site_idxs.long()


def main():
    y, visit_site_idxs = simulate_constant()
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs)

    model = ConstantOccuModel(n_species=data.n_species)
    res = fit_lbfgs_full_batch(data, model, max_iter=1000, lr=0.1)

    psi_hat = torch.sigmoid(model.logit_psi_species.detach())
    d_hat = torch.sigmoid(model.logit_d_species.detach())

    print("=== Fit results ===")
    print(f"  nll={res.nll:.4f}, n_iter={res.n_iter}, converged={res.converged}")
    print(f"  estimated psi={psi_hat}")
    print(f"  estimated   d={d_hat}")


if __name__ == "__main__":
    main()