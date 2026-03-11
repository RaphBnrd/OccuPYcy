import torch

import matplotlib.pyplot as plt
import polars as pl
import copy

from occupycy.data.core import OccuData
from occupycy.fit import fit_lbfgs_full_batch, fit_sgd_like
from occupycy.likelihood import nll_probs, nll_logits
from occupycy.models import MLPOccuModel
from occupycy.models.mlp import MLPBlock


def simulate_non_linear(
        n_sites=400, 
        n_total_visits_max=None,
        n_cov_psi=2,
        n_cov_d=1,
        f_effort=lambda x1: torch.exp(1.5*x1), 
        f_psi=lambda X_psi: torch.sigmoid(
            torch.tensor([-1.0, 0.5]) + X_psi @ torch.tensor([[2.0, -1.0], [-1.0, 1.0]]).T
        ), # from X_psi of shape (n_sites, n_cov_psi) to psi of shape (n_sites, n_species)
        f_d=lambda X_d: torch.sigmoid(
            torch.tensor([-0.5, 1.0]) + X_d @ torch.tensor([[1.0], [-2.0]]).T
        ), # from X_d of shape (n_visits, n_cov_d) to d of shape (n_visits, n_species)
        seed=None,
        plot_summary=True,
        verbose=True
    ):
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    X_psi = torch.randn(n_sites, n_cov_psi, generator=g)  # shape: (n_sites, n_cov_psi)
    
    # Define the number of visits per site
    n_visits_per_site = f_effort(X_psi[:, 0])  # shape: (n_sites,)
    n_visits_per_site = n_visits_per_site.round().long().clamp(min=0)  # shape: (n_sites,)
    visit_site_idxs = torch.arange(n_sites).repeat_interleave(n_visits_per_site)  # shape: (n_visits,)
    # Filter out sites with zero visits
    if n_total_visits_max is not None: # Randomly subsample visits to have at most n_total_visits_max
        if visit_site_idxs.shape[0] > n_total_visits_max:
            print(f"Subsampling from {visit_site_idxs.shape[0]} visits to {n_total_visits_max} visits.")
            perm = torch.randperm(visit_site_idxs.shape[0], generator=g)
            visit_site_idxs = visit_site_idxs[perm[:n_total_visits_max]] # shape: (n_visits_sub,)
    n_visits_per_site = torch.bincount(visit_site_idxs, minlength=n_sites)  # shape: (n_sites,) (for each site, how many visits it has)
    # Reassign site indices to be consecutive after filtering
    map_sites_to_sub = {}
    curr_idx = 0
    for i in range(n_sites):
        if n_visits_per_site[i] > 0:
            map_sites_to_sub[i] = curr_idx
            curr_idx += 1
    n_sites_sub = curr_idx
    visit_site_idxs = torch.tensor([map_sites_to_sub[idx.item()] for idx in visit_site_idxs], dtype=torch.long)  # shape: (n_visits_sub,)
    n_visits_sub = visit_site_idxs.shape[0]
    X_psi = X_psi[torch.tensor(list(map_sites_to_sub.keys()))]  # shape: (n_sites_sub, n_cov_psi)

    X_d = torch.randn(n_visits_sub, n_cov_d, generator=g)  # shape: (n_visits_sub, n_cov_d)

    psi = f_psi(X_psi)  # shape: (n_sites_sub, n_species)
    z = torch.bernoulli(psi, generator=g)  # shape: (n_sites_sub, n_species)
    z_for_visit = z[visit_site_idxs]  # shape: (n_visits_sub, n_species)
    
    d = f_d(X_d)  # shape: (n_visits_sub, n_species)
    
    y = torch.bernoulli(d * z_for_visit, generator=g)  # shape: (n_visits_sub, n_species)

    # Plot summary of simulated data
    if verbose:
        print("--- Simulated data summary ---")
        print(f"  n_sites={n_sites_sub}")
        print(f"  n_visits={n_visits_sub}")
        print(f"  observed average psi (knowing true occupancy): {z.mean(dim=0)}")
        print(f"  observed average d (knowing true occupancy):   {y.sum(dim=0) / z_for_visit.sum(dim=0)}")

    if plot_summary:
        fig, axes = plt.subplots(3, psi.shape[1], figsize=(8, 6))
        gs = fig.add_gridspec(3, psi.shape[1])
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
        for i in range(psi.shape[1]):
            axes[0, i].axis('off')
        # Histogram of psi
        for species_idx in range(psi.shape[1]):
            axes[1, species_idx].hist(psi[:, species_idx].cpu(), bins=30, edgecolor='k')
            axes[1, species_idx].set_xlabel("Occupancy probability (psi)")
            axes[1, species_idx].set_ylabel("Frequency")
            axes[1, species_idx].grid(True)
            axes[1, species_idx].set_title(f"psi (species {species_idx})")
        # Histogram of d
        for species_idx in range(d.shape[1]):
            axes[2, species_idx].hist(d[:, species_idx].cpu(), bins=30, edgecolor='k')
            axes[2, species_idx].set_xlabel("Detection probability (d)")
            axes[2, species_idx].set_ylabel("Frequency")
            axes[2, species_idx].grid(True)
            axes[2, species_idx].set_title(f"d (species {species_idx})")
        fig.tight_layout()

    return y.float(), visit_site_idxs.long(), X_psi.float(), X_d.float(), z.float()


def main():

    # --- Simulate data ---

    n_cov_psi = 2
    # def true_f_psi(X_psi):
    #     psi = torch.sigmoid(
    #         torch.tensor([0.0, -1.0])
    #         + X_psi @ torch.tensor([[0.0, 1.0], [-1.0, 0.5]]).T
    #     )
    #     return psi
    def true_f_psi(X_psi):
        x1 = X_psi[..., 0]
        x2 = X_psi[..., 1]
        mu = torch.tensor([[0.3, -0.2],
                           [-0.4, 0.5]], device=X_psi.device, dtype=X_psi.dtype)  # optima
        sigma = torch.tensor([[0.5, 0.4],
                              [0.3, 0.6]], device=X_psi.device, dtype=X_psi.dtype)  # niche width
        logit = torch.stack([
            -((x1 - mu[0,0])**2 / sigma[0,0]**2 + (x2 - mu[0,1])**2 / sigma[0,1]**2),
            -((x1 - mu[1,0])**2 / sigma[1,0]**2 + (x2 - mu[1,1])**2 / sigma[1,1]**2),
        ], dim=-1)
        eps = 1e-8
        logit = logit.clamp(torch.logit(torch.tensor(eps)), torch.logit(torch.tensor(1-eps)))
        return torch.sigmoid(logit)
    
    n_cov_d = 1
    # def true_f_d(X_d):
    #     d = torch.sigmoid(
    #         torch.tensor([-1.0, 0.5]) 
    #         + X_d @ torch.tensor([[-1.0], [-2.0]]).T
    #     )
    #     return d
    def true_f_d(X_d):
        x = X_d[..., 0]
        mu = torch.tensor([0.2, -0.4], device=X_d.device, dtype=X_d.dtype)
        sigma = torch.tensor([0.6, 0.3], device=X_d.device, dtype=X_d.dtype)
        logit = torch.stack([
            -((x - mu[0])**2 / sigma[0]**2),
            -((x - mu[1])**2 / sigma[1]**2),
        ], dim=-1)
        eps = 1e-8
        logit = logit.clamp(torch.logit(torch.tensor(eps)), torch.logit(torch.tensor(1-eps)))
        return torch.sigmoid(logit)

    y, visit_site_idxs, X_psi, X_d, true_z = simulate_non_linear(
        n_sites=1000, 
        n_total_visits_max=int(1e6),
        n_cov_psi=n_cov_psi,
        n_cov_d=n_cov_d,
        # f_effort=lambda x1: torch.exp(3*x1),
        # f_effort=lambda x1: 10 * x1 ** 2,
        f_effort=lambda x1: torch.ones_like(x1) * 100,
        f_psi=true_f_psi,
        f_d=true_f_d,
        seed=None,
        plot_summary=False
    )

    # --- Initialize results DataFrame ---
    
    eps = 1e-8
    true_logit_psi = torch.logit(true_f_psi(X_psi).clamp(eps, 1-eps))
    true_logit_d = torch.logit(true_f_d(X_d).clamp(eps, 1-eps))
    nll_true = nll_logits(y, visit_site_idxs=visit_site_idxs, logit_psi=true_logit_psi, logit_d=true_logit_d).item()

    df_results = pl.DataFrame({
        "model": ['true'],
        "fit_method": ['true'],
        "n_epochs": [0],
        "converged": [True],
        "final_train_loss": [nll_true]
    })
    
    # --- Build OccuData ---
    
    data = OccuData(y=y, visit_site_idxs=visit_site_idxs, X_psi=X_psi, X_d=X_d)

    # --- Define model ---
    
    all_models_lrs = {
        "mlp-psi_64_32_16-d_64_32_16": (MLPOccuModel(
            n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
            hidden_dims_psi=[64, 32, 16], hidden_dims_d=[64, 32, 16],
            activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
            dropout_psi=0.0, dropout_d=0.0
        ), (1e-4, 1e-4, 1e-3)),
        # "mlp-psi_16_16-d_16_16": (MLPOccuModel(
        #     n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
        #     hidden_dims_psi=[16, 16], hidden_dims_d=[16, 16],
        #     activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #     dropout_psi=0.0, dropout_d=0.0
        # ), (1e-4, 1e-4, 2e-4)),
        "linear-psi-d": (MLPOccuModel(
            n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
            hidden_dims_psi=[], hidden_dims_d=[],
            activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
            dropout_psi=0.0, dropout_d=0.0
        ), (1e-4, 5e-4, 1e-2)),
        # "constant-psi-d": (MLPOccuModel(
        #     n_species=y.shape[1], n_cov_psi=0, n_cov_d=0,
        #     hidden_dims_psi=[], hidden_dims_d=[],
        #     activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #     dropout_psi=0.0, dropout_d=0.0
        # ), (1e-3, 2e-3, 5e-3))
    }

    # all_fit_methods = ["lbfgs_full_batch", "sgd", "adam"]
    all_fit_methods = ["adam"]

    # --- Prepare plots ---

    n_species = y.shape[1]
    fig, axes = plt.subplots(n_species+1, 2, figsize=(8, 2.5 * (n_species+1)))
    gs = fig.add_gridspec(n_species+1, 2)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    ax0 = fig.add_subplot(gs[0, :])

    if n_cov_psi == 1:
        # x_psi = torch.linspace(X_psi[:, 0].min(), X_psi[:, 0].max(), 100).unsqueeze(1)  # shape: (100, 1)
        x_psi = X_psi  # shape: (n_sites_sub, 1)
        n_plots_psi = 1
    elif n_cov_psi == 2:
        x_psi = torch.stack(torch.meshgrid(
            torch.linspace(X_psi[:, 0].min(), X_psi[:, 0].max(), 20),
            torch.linspace(X_psi[:, 1].min(), X_psi[:, 1].max(), 20),
            indexing="ij"
        ), dim=-1) # shape: (20, 20, n_cov_psi)
        # x_psi = X_psi  # shape: (n_sites_sub, 2)
        n_plots_psi = len(all_models_lrs) * len(all_fit_methods) + 1
    if n_cov_d == 1:
        # x_d = torch.linspace(X_d[:, 0].min(), X_d[:, 0].max(), 100).unsqueeze(1)  # shape: (100, 1)
        x_d = X_d  # shape: (n_visits_sub, 1)
        n_plots_d = 1
    elif n_cov_d == 2:
        x_d = torch.stack(torch.meshgrid(
            torch.linspace(X_d[:, 0].min(), X_d[:, 0].max(), 20),
            torch.linspace(X_d[:, 1].min(), X_d[:, 1].max(), 20),
            indexing="ij"
        ), dim=-1) # shape: (20, 20, n_cov_d)
        # x_d = X_d  # shape: (n_visits_sub, 2)
        n_plots_d = len(all_models_lrs) * len(all_fit_methods) + 1
    
    if n_cov_psi in [1, 2]:
        fig_pred_psi, axes_pred_psi = plt.subplots(n_species, n_plots_psi, figsize=(max(2.5 * n_plots_psi, 6), 3 * n_species))
        fig_pred_psi.suptitle("logit_psi", fontsize=16)
        tmp_true_logit_psi = torch.logit(true_f_psi(x_psi)).cpu() # shape: (100, n_species) if n_cov_psi==1 else (20, 20, n_species)
        if n_cov_psi == 1:
            for i in range(n_species):
                ax = axes_pred_psi[i]
                order_points_psi = torch.argsort(x_psi[:, 0])
                ax.plot(x_psi.cpu()[order_points_psi], tmp_true_logit_psi[:, i].cpu()[order_points_psi], 
                        'o-', label="true f_psi", color="black", markersize=2)
                ax.set_xlabel("X_psi covariate")
                ax.set_ylabel("logit_psi")
                ax.set_title(f"species {i}")
                ax.grid(True)
        else:
            for i in range(n_species):
                ax = axes_pred_psi[i, 0]
                ax.contourf(x_psi[:, :, 0].cpu(), x_psi[:, :, 1].cpu(), tmp_true_logit_psi[:, :, i].cpu(), levels=20, cmap="viridis")
                # ax.scatter(x_psi[:, 0].cpu(), x_psi[:, 1].cpu(), c=tmp_true_logit_psi[:, i].cpu(), s=20, cmap="viridis", alpha=0.1)
                ax.set_xlabel("X_psi covariate 1")
                ax.set_ylabel("X_psi covariate 2")
                ax.set_title(f"true\nspecies {i})", fontsize=8)
                ax.grid(True)
                cbar = fig_pred_psi.colorbar(ax.collections[0], ax=ax, shrink=0.7)
                cbar.ax.tick_params(labelsize=6)
    if n_cov_d in [1, 2]:
        fig_pred_d, axes_pred_d = plt.subplots(n_species, n_plots_d, figsize=(max(2.5 * n_plots_d, 6), 3 * n_species))
        fig_pred_d.suptitle("logit_d", fontsize=16)
        tmp_true_logit_d = torch.logit(true_f_d(x_d)).cpu() # shape: (100, n_species) if n_cov_d==1 else (20, 20, n_species)
        if n_cov_d == 1:
            for i in range(n_species):
                ax = axes_pred_d[i]
                order_points_d = torch.argsort(x_d[:, 0])
                ax.plot(x_d.cpu()[order_points_d], tmp_true_logit_d[:, i].cpu()[order_points_d], 
                        'o-', label="true f_d", color="black", markersize=2)
                ax.set_xlabel("X_d covariate")
                ax.set_ylabel("logit_d")
                ax.set_title(f"species {i}")
                ax.grid(True)
        else:
            for i in range(n_species):
                ax = axes_pred_d[i, 0]
                ax.contourf(x_d[:, :, 0].cpu(), x_d[:, :, 1].cpu(), tmp_true_logit_d[:, :, i].cpu(), levels=20, cmap="viridis")
                # ax.scatter(x_d[:, 0].cpu(), x_d[:, 1].cpu(), c=tmp_true_logit_d[:, i].cpu(), s=20, cmap="viridis", alpha=0.1)
                ax.set_xlabel("X_d covariate 1")
                ax.set_ylabel("X_d covariate 2")
                ax.set_title(f"true\nspecies {i}", fontsize=8)
                ax.grid(True)
                cbar = fig_pred_d.colorbar(ax.collections[0], ax=ax, shrink=0.7)
                cbar.ax.tick_params(labelsize=6)
    idx_plot = 0

    for model_name, (this_model, lrs) in all_models_lrs.items():
        print(f"\n=== Fitting {model_name} =================================")

        for fit_method in all_fit_methods:
            idx_plot += 1

            model = copy.deepcopy(this_model)

            print(f"> {fit_method}")
        
            # --- Fit model ---

            if fit_method == "lbfgs_full_batch":
                if isinstance(model.psi_model, MLPBlock) or isinstance(model.d_model, MLPBlock):
                    print("Skipping lbfgs_full_batch for mlp model.")
                    continue
                res = fit_lbfgs_full_batch(
                    data, model, max_iter=1000, 
                    lr=lrs[0],
                    device=torch.device("cpu"),
                    tolerance_grad=1e-8, tolerance_change=1e-10, history_size=100, line_search_fn="strong_wolfe"
                )
            elif fit_method == "sgd":
                res = fit_sgd_like(
                    data, model, n_epochs=500, batch_size=100, 
                    lr_psi=lrs[1], lr_d=lrs[1],
                    optimizer_psi_cls=torch.optim.SGD, optimizer_d_cls=torch.optim.SGD,
                    device=torch.device("cpu")
                )
            elif fit_method == "adam":
                res = fit_sgd_like(
                    data, model, n_epochs=500, batch_size=100, 
                    lr_psi=lrs[2], lr_d=lrs[2],
                    optimizer_psi_cls=torch.optim.Adam, optimizer_d_cls=torch.optim.Adam,
                    device=torch.device("cpu")
                )

            # --- Store results ---

            df_results = df_results.vstack(pl.DataFrame({
                "model": [model_name],
                "fit_method": [fit_method],
                "n_epochs": [res.n_epochs],
                "converged": [res.converged],
                "final_train_loss": [res.final_train_loss]
            }))
            
            # Plot loss
            ax0.plot(res.loss_history["train_loss"], label=f"{model_name} | {fit_method}\n(final nll={res.final_train_loss:.2f})")
            ax0.set_yscale("log")
            ax0.grid(True, which="both", ls="--")
            ax0.set_xlabel("Epoch")
            ax0.set_ylabel("Train Loss")
            # Plot predicted vs true logit_psi
            model.psi_model.eval()
            logit_psi_hat = model.psi_model(X_psi).detach() # shape: (n_sites, n_species)
            for i in range(n_species):
                axes[i+1, 0].scatter(true_logit_psi[:, i].cpu(), logit_psi_hat[:, i].cpu(), 
                                    s=3, alpha=0.5, label=f"{model_name} | {fit_method}")
                min_range = torch.max(torch.min(true_logit_psi[:, i]), torch.min(logit_psi_hat[:, i])).item()
                max_range = torch.min(torch.max(true_logit_psi[:, i]), torch.max(logit_psi_hat[:, i])).item()
                axes[i+1, 0].plot([min_range, max_range], [min_range, max_range], color="gray", linestyle="--")
                axes[i+1, 0].set_xlabel("True logit_psi")
                axes[i+1, 0].set_ylabel("Predicted logit_psi")
                axes[i+1, 0].set_title(f"species {i}")
                axes[i+1, 0].grid(True)
            # Plot predicted vs true logit_d
            model.d_model.eval()
            logit_d_hat = model.d_model(X_d).detach() # shape: (n_visits, n_species)
            for i in range(n_species):
                axes[i+1, 1].scatter(true_logit_d[:, i].cpu(), logit_d_hat[:, i].cpu(), 
                                    s=3, alpha=0.1, label=f"{model_name} | {fit_method}")
                min_range = torch.max(torch.min(true_logit_d[:, i]), torch.min(logit_d_hat[:, i])).item()
                max_range = torch.min(torch.max(true_logit_d[:, i]), torch.max(logit_d_hat[:, i])).item()
                axes[i+1, 1].plot([min_range, max_range], [min_range, max_range], color="gray", linestyle="--")
                axes[i+1, 1].set_xlabel("True logit_d")
                axes[i+1, 1].set_ylabel("Predicted logit_d")
                axes[i+1, 1].set_title(f"species {i}")
                axes[i+1, 1].grid(True)
        
            # Plot predictions
            if n_cov_psi == 1:
                with torch.no_grad():
                    logit_psi_pred = model.psi_model(x_psi).cpu()
                for i in range(n_species):
                    ax = axes_pred_psi[i]
                    ax.plot(x_psi.cpu()[order_points_psi], logit_psi_pred[:, i].cpu()[order_points_psi], 
                            'o-', label=f"{model_name} | {fit_method}", markersize=2)
            elif n_cov_psi == 2:
                with torch.no_grad():
                    logit_psi_pred = model.psi_model(x_psi.view(-1, n_cov_psi)).cpu().view(x_psi.shape[0], x_psi.shape[1], -1)
                for i in range(n_species):
                    ax = axes_pred_psi[i, idx_plot]
                    ax.contourf(x_psi[:, :, 0].cpu(), x_psi[:, :, 1].cpu(), logit_psi_pred[:, :, i].cpu(), levels=20, cmap="viridis")
                    # ax.scatter(x_psi[:, 0].cpu(), x_psi[:, 1].cpu(), c=logit_psi_pred[:, i].cpu(), s=20, cmap="viridis", alpha=0.1)
                    ax.set_xlabel("X_psi covariate 1")
                    ax.set_ylabel("X_psi covariate 2")
                    ax.set_title(f"{model_name}\n{fit_method} | species {i}", fontsize=8)
                    ax.grid(True)
                    cbar = fig_pred_psi.colorbar(ax.collections[0], ax=ax, shrink=0.7)
                    cbar.ax.tick_params(labelsize=6)
            if n_cov_d == 1:
                with torch.no_grad():
                    logit_d_pred = model.d_model(x_d).cpu()
                for i in range(n_species):
                    ax = axes_pred_d[i]
                    ax.plot(x_d.cpu()[order_points_d], logit_d_pred[:, i].cpu()[order_points_d],
                            'o-', label=f"{model_name} | {fit_method}", markersize=2)
            elif n_cov_d == 2:
                with torch.no_grad():
                    logit_d_pred = model.d_model(x_d.view(-1, n_cov_d)).cpu().view(x_d.shape[0], x_d.shape[1], -1)
                for i in range(n_species):
                    ax = axes_pred_d[i, idx_plot]
                    ax.contourf(x_d[:, :, 0].cpu(), x_d[:, :, 1].cpu(), logit_d_pred[:, :, i].cpu(), levels=20, cmap="viridis")
                    # ax.scatter(x_d[:, 0].cpu(), x_d[:, 1].cpu(), c=logit_d_pred[:, i].cpu(), s=20, cmap="viridis", alpha=0.1)
                    ax.set_xlabel("X_d covariate 1")
                    ax.set_ylabel("X_d covariate 2")
                    ax.set_title(f"{model_name}\n{fit_method} | species {i}", fontsize=8)
                    ax.grid(True)
                    cbar = fig_pred_d.colorbar(ax.collections[0], ax=ax, shrink=0.7)
                    cbar.ax.tick_params(labelsize=6)

    ax0.axhline(nll_true, color="gray", linestyle="--", label=f"(true nll={nll_true:.2f})")
    ax0.set_title("Fitting constant OccuModel")
    ax0.legend(fontsize=5)
    for i in range(n_species):
        axes[i+1, 0].legend(fontsize=5)
        axes[i+1, 1].legend(fontsize=5)
    fig.tight_layout()

    if n_cov_psi == 1:
        for i in range(n_species):
            axes_pred_psi[i].legend(fontsize=5)
    fig_pred_psi.tight_layout()

    if n_cov_d == 1:
        for i in range(n_species):
            axes_pred_d[i].legend(fontsize=5)
    fig_pred_d.tight_layout()

    # --- Show results ---

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        # Sorted by final_train_loss
        print(df_results.sort("final_train_loss"))

    plt.show()


if __name__ == "__main__":
    main()