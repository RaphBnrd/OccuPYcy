import torch

import matplotlib.pyplot as plt
import polars as pl
from copy import deepcopy

from occupycy.data.core import OccuData
from occupycy.fit import fit_lbfgs_full_batch, fit_sgd_like
from occupycy.likelihood import nll_probs, nll_logits
from occupycy.models import LinearOccuModel


def simulate_linear(
        n_sites=400, n_total_visits_max=None,
        f_effort=lambda x1: torch.exp(1.5*x1), 
        intercept_psi=torch.tensor([0.0]), # shape: (n_species,)
        coefs_psi=torch.tensor([[0.0]]), # shape: (n_species, n_cov_psi)
        intercept_d=torch.tensor([0.0]), # shape: (n_species,)
        coefs_d=torch.tensor([[0.0]]), # shape: (n_species, n_cov_d)
        seed=None
    ):
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    X_psi = torch.randn(n_sites, len(coefs_psi[0]), generator=g)  # shape: (n_sites, n_cov_psi)
    
    # Define the number of visits per site
    n_visits_per_site = f_effort(X_psi[:, 0])  # shape: (n_sites,)
    n_visits_per_site = n_visits_per_site.round().long().clamp(min=0)  # shape: (n_sites,)
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # axes[0].plot(X_psi[:, 0].cpu(), n_visits_per_site.cpu(), 'o', alpha=0.5)
    # axes[0].set_xlabel("Covariate for occupancy (X_psi[:, 0])")
    # axes[0].set_ylabel("Number of visits per site")
    # axes[1].hist(X_psi[:, 0].cpu(), bins=30, edgecolor='k')
    # axes[1].set_xlabel("Covariate for occupancy (X_psi[:, 0])")
    # axes[1].set_ylabel("Frequency")
    indices_visit_site = torch.arange(n_sites).repeat_interleave(n_visits_per_site)  # shape: (n_visits,)
    # Filter out sites with zero visits
    if n_total_visits_max is not None: # Randomly subsample visits to have at most n_total_visits_max
        if indices_visit_site.shape[0] > n_total_visits_max:
            print(f"Subsampling from {indices_visit_site.shape[0]} visits to {n_total_visits_max} visits.")
            perm = torch.randperm(indices_visit_site.shape[0], generator=g)
            indices_visit_site = indices_visit_site[perm[:n_total_visits_max]] # shape: (n_visits_sub,)
    n_visits_per_site = torch.bincount(indices_visit_site, minlength=n_sites)  # shape: (n_sites,) (for each site, how many visits it has)
    # Reassign site indices to be consecutive after filtering
    map_sites_to_sub = {}
    curr_idx = 0
    for i in range(n_sites):
        if n_visits_per_site[i] > 0:
            map_sites_to_sub[i] = curr_idx
            curr_idx += 1
    n_sites_sub = curr_idx
    indices_visit_site = torch.tensor([map_sites_to_sub[idx.item()] for idx in indices_visit_site], dtype=torch.long)  # shape: (n_visits_sub,)
    n_visits_sub = indices_visit_site.shape[0]
    X_psi = X_psi[torch.tensor(list(map_sites_to_sub.keys()))]  # shape: (n_sites_sub, n_cov_psi)

    X_d = torch.randn(n_visits_sub, len(coefs_d[0]), generator=g)  # shape: (n_visits_sub, n_cov_d)

    psi = torch.sigmoid(intercept_psi + X_psi @ coefs_psi.T) # shape: (n_sites_sub, n_species)
    z = torch.bernoulli(psi, generator=g)  # shape: (n_sites_sub, n_species)
    z_for_visit = z[indices_visit_site]  # shape: (n_visits_sub, n_species)
    
    d = torch.sigmoid(intercept_d + X_d @ coefs_d.T) # shape: (n_visits_sub, n_species)
    
    y = torch.bernoulli(d * z_for_visit, generator=g)  # shape: (n_visits_sub, n_species)

    # Plot summary of simulated data
    print("--- Simulated data summary ---")
    print(f"  n_sites={n_sites_sub}")
    print(f"  n_visits={n_visits_sub}")
    print(f"  observed average psi (knowing true occupancy): {z.mean(dim=0)}")
    print(f"  observed average d (knowing true occupancy):   {y.sum(dim=0) / z_for_visit.sum(dim=0)}")

    fig, axes = plt.subplots(3, intercept_psi.shape[0], figsize=(8, 6))
    gs = fig.add_gridspec(3, intercept_psi.shape[0])
    # Histogram of n_visits_per_site
    ax0 = fig.add_subplot(gs[0, :])
    # logbins = np.logspace(0, np.log10(n_visits_per_site.max().item()), 30)
    # ax0.hist(n_visits_per_site.cpu(), bins=logbins, edgecolor='k')
    # ax0.set_xscale("log")
    ax0.hist(n_visits_per_site.cpu(), bins=30, edgecolor='k')
    ax0.set_xlabel("Number of visits per site")
    ax0.set_ylabel("Frequency")
    ax0.set_title("Distribution of visits per site")
    ax0.grid(True)
    ax0.set_yscale("log")
    for i in range(intercept_psi.shape[0]):
        axes[0, i].axis('off')
    # Histogram of psi
    for species_idx in range(intercept_psi.shape[0]):
        axes[1, species_idx].hist(psi[:, species_idx].cpu(), bins=30, edgecolor='k')
        axes[1, species_idx].set_xlabel("Occupancy probability (psi)")
        axes[1, species_idx].set_ylabel("Frequency")
        axes[1, species_idx].grid(True)
        axes[1, species_idx].set_title(f"psi (species {species_idx})")
    # Histogram of d
    for species_idx in range(intercept_d.shape[0]):
        axes[2, species_idx].hist(d[:, species_idx].cpu(), bins=30, edgecolor='k')
        axes[2, species_idx].set_xlabel("Detection probability (d)")
        axes[2, species_idx].set_ylabel("Frequency")
        axes[2, species_idx].grid(True)
        axes[2, species_idx].set_title(f"d (species {species_idx})")
    fig.tight_layout()

    return y.float(), indices_visit_site.long().tolist(), X_psi.float(), X_d.float(), z.float()


def main():

    # --- Simulate data ---

    true_intercept_psi = torch.tensor([0.0, -1.0])                      # shape: (n_species,)
    true_coefs_psi = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.5, -0.5]]) # shape: (n_species, n_cov_psi)
    true_intercept_d = torch.tensor([-1.0, 0.5])                        # shape: (n_species,)
    true_coefs_d = torch.tensor([[0.0], [1.0]])                         # shape: (n_species, n_cov_d)

    y, visit_site_ids, X_psi, X_d, true_z = simulate_linear(
        n_sites=500, 
        n_total_visits_max=int(1e6),
        # f_effort=lambda x1: torch.exp(3*x1),
        # f_effort=lambda x1: 10 * x1 ** 2,
        f_effort=lambda x1: torch.ones_like(x1) * 50,
        intercept_psi=true_intercept_psi, 
        coefs_psi=true_coefs_psi, 
        intercept_d=true_intercept_d, 
        coefs_d=true_coefs_d, 
        seed=123
    )

    # --- Initialize results DataFrame ---
    
    true_logit_psi = true_intercept_psi + X_psi @ true_coefs_psi.T
    true_logit_d = true_intercept_d + X_d @ true_coefs_d.T
    nll_true = nll_logits(y, indices_visit_site=torch.tensor(visit_site_ids), logit_psi=true_logit_psi, logit_d=true_logit_d).item()
    # true_psi = torch.sigmoid(true_logit_psi)
    # true_d = torch.sigmoid(true_logit_d)
    # nll_true = nll_probs(y, indices_visit_site=torch.tensor(visit_site_ids), psi=true_psi, d=true_d).item()

    def polar_res(*, fit_method, n_epochs, converged, final_train_loss, 
                  intercept_psi, coefs_psi, intercept_d, coefs_d):
        df_results = {
            "fit_method": [fit_method],
            "n_epochs": [n_epochs],
            "converged": [converged],
            "final_train_loss": [final_train_loss]
        }
        for species_idx in range(len(intercept_psi)):
            df_results[f"sp{species_idx}_psi_int"] = [intercept_psi[species_idx].item()]
            for cov_idx in range(coefs_psi.shape[1]):
                df_results[f"sp{species_idx}_psi_c{cov_idx}"] = [coefs_psi[species_idx, cov_idx].item()]
            df_results[f"sp{species_idx}_d_int"] = [intercept_d[species_idx].item()]
            for cov_idx in range(coefs_d.shape[1]):
                df_results[f"sp{species_idx}_d_c{cov_idx}"] = [coefs_d[species_idx, cov_idx].item()]
        df_results = pl.DataFrame(df_results).with_columns(
            pl.col("final_train_loss").cast(pl.Float64)
        )
        return df_results
    
    df_results = polar_res(
        fit_method='true', n_epochs=0, converged=True, final_train_loss=nll_true,
        intercept_psi=true_intercept_psi, coefs_psi=true_coefs_psi, 
        intercept_d=true_intercept_d, coefs_d=true_coefs_d
    )
    
    # --- Build OccuData ---
    
    init_data = OccuData(y=y, visit_site_ids=visit_site_ids, X_psi=X_psi, X_d=X_d)

    plt.figure(figsize=(8, 5))
    for fit_method in ["lbfgs_full_batch", "lbfgs_FB", "lbfgs", "sgd", "adam"]:
        print(f"\n--- Fitting with {fit_method} ---")
        
        data = deepcopy(init_data)

        # --- Define model ---
        
        model = LinearOccuModel(
            n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1]
        )
    
        # --- Fit model ---

        if fit_method == "lbfgs_full_batch":
            res = fit_lbfgs_full_batch(
                data, model, max_iter=1000, 
                # lr_psi=0.1, lr_d=0.1, 
                lr=5e-3,
                device=torch.device("cpu"),
                tolerance_grad=1e-8, tolerance_change=1e-10, history_size=100, line_search_fn="strong_wolfe"
            )
        elif fit_method == "lbfgs_FB":
            res = fit_sgd_like(
                data, model, n_epochs=20, batch_size=data.n_sites, 
                lr_psi=1e-1, lr_d=1e-1,
                optimizer_psi_cls=torch.optim.LBFGS, optimizer_d_cls=torch.optim.LBFGS,
                device=torch.device("cpu")
            )
        elif fit_method == "lbfgs":
            res = fit_sgd_like(
                data, model, n_epochs=20, batch_size=100, 
                lr_psi=5e-2, lr_d=5e-2,
                optimizer_psi_cls=torch.optim.LBFGS, optimizer_d_cls=torch.optim.LBFGS,
                device=torch.device("cpu")
            )
        elif fit_method == "sgd":
            res = fit_sgd_like(
                data, model, n_epochs=100, batch_size=100, 
                lr_psi=1e-3, lr_d=1e-3,
                optimizer_psi_cls=torch.optim.SGD, optimizer_d_cls=torch.optim.SGD,
                device=torch.device("cpu")
            )
        elif fit_method == "adam":
            res = fit_sgd_like(
                data, model, n_epochs=100, batch_size=100, 
                lr_psi=1e-2, lr_d=1e-2,
                optimizer_psi_cls=torch.optim.Adam, optimizer_d_cls=torch.optim.Adam,
                device=torch.device("cpu")
            )

        # --- Store results ---

        df_results = df_results.vstack(polar_res(
            fit_method=fit_method, n_epochs=res.n_epochs, converged=res.converged, final_train_loss=res.final_train_loss,
            intercept_psi=model.psi_model.bias.detach(), coefs_psi=model.psi_model.weight.detach(),
            intercept_d=model.d_model.bias.detach(), coefs_d=model.d_model.weight.detach()
        ))
        
        plt.plot(res.loss_history["train_loss"], label=f"{fit_method}\n(final nll={res.final_train_loss:.2f})")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
    plt.axhline(nll_true, color="gray", linestyle="--", label=f"(true nll={nll_true:.2f})")

    # --- Show results ---

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(df_results)

    plt.title("Fitting constant OccuModel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()