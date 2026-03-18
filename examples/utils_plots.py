import torch
import matplotlib.pyplot as plt



# -------------------------------------------------------------
# Plot for fit_nonlinear.py
# -------------------------------------------------------------

def non_lin_initialize_plots(
    *, y, X_psi, X_d,
    n_cov_psi, n_cov_d, true_f_psi, true_f_d,
    all_models
):
    """
    Initializes the plots for the nonlinear fit example.

    . fig is for predicted logit_psi and logit_d vs true logit_psi and logit_d.
    . fig_pred_psi is for predicted logit_psi depending on the covariates.
    . fig_pred_d is for predicted logit_d depending on the covariates.
    """

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
        n_plots_psi = sum([len(mod['methods']) for mod in all_models.values()]) + 1
        
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
        n_plots_d = sum([len(mod['methods']) for mod in all_models.values()]) + 1

    # Variables that may not be defined below
    overall_min_logit_psi, overall_max_logit_psi = None, None
    overall_min_logit_d, overall_max_logit_d = None, None
    order_points_psi, order_points_d = None, None
    contours_psi, contours_d = None, None
    
    if n_cov_psi in [1, 2]:
        fig_pred_psi, axes_pred_psi = plt.subplots(n_species, n_plots_psi, figsize=(max(2.5 * n_plots_psi, 6), 3 * n_species))
        fig_pred_psi.suptitle("logit_psi", fontsize=16)
        tmp_true_logit_psi = torch.logit(true_f_psi(x_psi)).cpu() # shape: (100, n_species) if n_cov_psi==1 else (20, 20, n_species)
        if n_cov_psi == 1:
            for i in range(n_species):
                ax = axes_pred_psi[i]
                order_points_psi = torch.argsort(x_psi[:, 0])
                ax.plot(x_psi.cpu()[order_points_psi], tmp_true_logit_psi[:, i][order_points_psi], 
                        'o-', alpha=0.1, label="true f_psi", color="black", markersize=2)
                ax.set_xlabel("X_psi covariate")
                ax.set_ylabel("logit_psi")
                ax.set_title(f"species {i}")
                ax.grid(True)
        else:
            overall_min_logit_psi = tmp_true_logit_psi.min().item()
            overall_max_logit_psi = tmp_true_logit_psi.max().item()
            contours_psi = []
            for i in range(n_species):
                ax = axes_pred_psi[i, 0]
                cf = ax.contourf(x_psi[:, :, 0].cpu(), x_psi[:, :, 1].cpu(), tmp_true_logit_psi[:, :, i], levels=20, cmap="viridis")
                contours_psi.append(cf)
                # ax.scatter(x_psi[:, 0].cpu(), x_psi[:, 1].cpu(), c=tmp_true_logit_psi[:, i], s=20, cmap="viridis", alpha=0.1)
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
                ax.plot(x_d.cpu()[order_points_d], tmp_true_logit_d[:, i][order_points_d], 
                        'o-', alpha=0.1, label="true f_d", color="black", markersize=2)
                ax.set_xlabel("X_d covariate")
                ax.set_ylabel("logit_d")
                ax.set_title(f"species {i}")
                ax.grid(True)
        else:
            overall_min_logit_d = tmp_true_logit_d.min().item()
            overall_max_logit_d = tmp_true_logit_d.max().item()
            contours_d = []
            for i in range(n_species):
                ax = axes_pred_d[i, 0]
                cf = ax.contourf(x_d[:, :, 0].cpu(), x_d[:, :, 1].cpu(), tmp_true_logit_d[:, :, i], levels=20, cmap="viridis")
                contours_d.append(cf)
                # ax.scatter(x_d[:, 0].cpu(), x_d[:, 1].cpu(), c=tmp_true_logit_d[:, i], s=20, cmap="viridis", alpha=0.1)
                ax.set_xlabel("X_d covariate 1")
                ax.set_ylabel("X_d covariate 2")
                ax.set_title(f"true\nspecies {i}", fontsize=8)
                ax.grid(True)
                cbar = fig_pred_d.colorbar(ax.collections[0], ax=ax, shrink=0.7)
                cbar.ax.tick_params(labelsize=6)

    return fig, axes, ax0, fig_pred_psi, axes_pred_psi, fig_pred_d, axes_pred_d, \
        x_psi, x_d, order_points_psi, order_points_d, contours_psi, contours_d, \
        overall_min_logit_psi, overall_max_logit_psi, overall_min_logit_d, overall_max_logit_d


def non_lin_plot_each_loop(
    *, res, model, label, idx_plot,
    axes, ax0, fig_pred_psi, axes_pred_psi, fig_pred_d, axes_pred_d, contours_psi, contours_d,
    y, X_psi, X_d, x_psi, x_d, n_cov_psi, n_cov_d, true_logit_psi, true_logit_d,
    order_points_psi, order_points_d,
    overall_min_logit_psi, overall_max_logit_psi, overall_min_logit_d, overall_max_logit_d
):
    n_species = y.shape[1]
    
    # Plot loss
    ax0.plot(res.history["train_loss"], label=f"{label}\n(final nll={res.final_train_loss:.2f})")
    ax0.set_yscale("log")
    ax0.grid(True, which="both", ls="--")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Train Loss")
    # Plot predicted vs true logit_psi
    model.psi_model.eval()
    logit_psi_hat = model.psi_model(X_psi).detach().cpu() # shape: (n_sites, n_species)
    for i in range(n_species):
        axes[i+1, 0].scatter(true_logit_psi[:, i].cpu(), logit_psi_hat[:, i].cpu(), 
                                s=3, alpha=0.1, label=label)
        min_range = torch.max(torch.min(true_logit_psi[:, i]), torch.min(logit_psi_hat[:, i])).item()
        max_range = torch.min(torch.max(true_logit_psi[:, i]), torch.max(logit_psi_hat[:, i])).item()
        axes[i+1, 0].plot([min_range, max_range], [min_range, max_range], color="gray", linestyle="--")
        axes[i+1, 0].set_xlabel("True logit_psi")
        axes[i+1, 0].set_ylabel("Predicted logit_psi")
        axes[i+1, 0].set_title(f"species {i}")
        axes[i+1, 0].grid(True)
    # Plot predicted vs true logit_d
    model.d_model.eval()
    logit_d_hat = model.d_model(X_d).detach().cpu() # shape: (n_visits, n_species)
    for i in range(n_species):
        axes[i+1, 1].scatter(true_logit_d[:, i].cpu(), logit_d_hat[:, i].cpu(), 
                                s=3, alpha=0.1, label=label)
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
            logit_psi_pred = model.psi_model(x_psi).detach().cpu()
        for i in range(n_species):
            ax = axes_pred_psi[i]
            ax.plot(x_psi.cpu()[order_points_psi], logit_psi_pred[:, i][order_points_psi], 
                    'o-', alpha=0.1, label=label, markersize=2)
    elif n_cov_psi == 2:
        with torch.no_grad():
            logit_psi_pred = model.psi_model(x_psi.view(-1, n_cov_psi)).detach().cpu().view(x_psi.shape[0], x_psi.shape[1], -1)
        overall_min_logit_psi = min(overall_min_logit_psi, logit_psi_pred.min().item())
        overall_max_logit_psi = max(overall_max_logit_psi, logit_psi_pred.max().item())
        for i in range(n_species):
            ax = axes_pred_psi[i, idx_plot]
            cf = ax.contourf(x_psi[:, :, 0].cpu(), x_psi[:, :, 1].cpu(), logit_psi_pred[:, :, i], levels=20, cmap="viridis")
            contours_psi.append(cf)
            # ax.scatter(x_psi[:, 0].cpu(), x_psi[:, 1].cpu(), c=logit_psi_pred[:, i], s=20, cmap="viridis", alpha=0.1)
            ax.set_xlabel("X_psi covariate 1")
            ax.set_ylabel("X_psi covariate 2")
            ax.set_title(f"{label}\nspecies {i}", fontsize=8)
            ax.grid(True)
            cbar = fig_pred_psi.colorbar(ax.collections[0], ax=ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)
    if n_cov_d == 1:
        with torch.no_grad():
            logit_d_pred = model.d_model(x_d).detach().cpu()
        for i in range(n_species):
            ax = axes_pred_d[i]
            ax.plot(x_d.cpu()[order_points_d], logit_d_pred[:, i][order_points_d],
                    'o-', alpha=0.1, label=label, markersize=2)
    elif n_cov_d == 2:
        with torch.no_grad():
            logit_d_pred = model.d_model(x_d.view(-1, n_cov_d)).detach().cpu().view(x_d.shape[0], x_d.shape[1], -1)
        overall_min_logit_d = min(overall_min_logit_d, logit_d_pred.min().item())
        overall_max_logit_d = max(overall_max_logit_d, logit_d_pred.max().item())
        for i in range(n_species):
            ax = axes_pred_d[i, idx_plot]
            cf = ax.contourf(x_d[:, :, 0].cpu(), x_d[:, :, 1].cpu(), logit_d_pred[:, :, i], levels=20, cmap="viridis")
            contours_d.append(cf)
            # ax.scatter(x_d[:, 0].cpu(), x_d[:, 1].cpu(), c=logit_d_pred[:, i], s=20, cmap="viridis", alpha=0.1)
            ax.set_xlabel("X_d covariate 1")
            ax.set_ylabel("X_d covariate 2")
            ax.set_title(f"{label}\nspecies {i}", fontsize=8)
            ax.grid(True)
            cbar = fig_pred_d.colorbar(ax.collections[0], ax=ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

def non_lin_finalize_plots(
    *, n_species, fig, axes, ax0, fig_pred_psi, axes_pred_psi, fig_pred_d, axes_pred_d, contours_psi, contours_d,
    n_cov_psi, n_cov_d, nll_true,
    overall_min_logit_psi, overall_max_logit_psi, overall_min_logit_d, overall_max_logit_d
):

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
    elif n_cov_psi == 2:
        for cf in contours_psi:
            cf.set_clim(overall_min_logit_psi, overall_max_logit_psi)
    if n_cov_psi in [1, 2]:
        fig_pred_psi.tight_layout()

    if n_cov_d == 1:
        for i in range(n_species):
            axes_pred_d[i].legend(fontsize=5)
    elif n_cov_d == 2:
        for cf in contours_d:
            cf.set_clim(overall_min_logit_d, overall_max_logit_d)
    if n_cov_d in [1, 2]:
        fig_pred_d.tight_layout()