import torch

import matplotlib.pyplot as plt
import polars as pl
import copy

from occupycy.data.simulate_data import simulate_non_linear
from utils_plots import non_lin_initialize_plots, non_lin_plot_each_loop, non_lin_finalize_plots

from occupycy.data import OccuData
from occupycy.fit import fit_lbfgs_full_batch, fit_sgd_like
from occupycy.likelihood import nll_probs, nll_logits
from occupycy.models import MLPOccuModel, LinearMLPOccuModel
from occupycy.models.mlp import MLPBlock


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

def main():

    # -------------------------------------------------------------
    # Simulate data
    # -------------------------------------------------------------

    # * * * Models for PSI * * *
    # You can try various functions for true_f_psi

    n_cov_psi = 2

    # def true_f_psi(X_psi):
    #     psi = torch.sigmoid(
    #         torch.tensor([0.0, -1.0]) # shape: (n_species,) intercept
    #         + X_psi @ torch.tensor([[0.0, 1.0], [-1.0, 0.5]]).T # shape: (n_sites, n_cov_psi) @ (n_cov_psi, n_species) -> (n_sites, n_species)
    #     ) # shape: from X_psi of shape (n_sites, n_cov_psi) to psi of shape (n_sites, n_species)
    #     return psi
    # def true_f_psi(X_psi):
    #     x1 = X_psi[..., 0]
    #     x2 = X_psi[..., 1]
    #     mu = torch.tensor([[0.3, -0.2],
    #                        [-0.4, 0.5]], device=X_psi.device, dtype=X_psi.dtype)  # optima
    #     sigma = torch.tensor([[0.5, 0.4],
    #                           [0.3, 0.6]], device=X_psi.device, dtype=X_psi.dtype)  # niche width
    #     logit = torch.stack([
    #         -((x1 - mu[0,0])**2 / sigma[0,0]**2 + (x2 - mu[0,1])**2 / sigma[0,1]**2),
    #         -((x1 - mu[1,0])**2 / sigma[1,0]**2 + (x2 - mu[1,1])**2 / sigma[1,1]**2),
    #     ], dim=-1)
    #     eps = 1e-8
    #     logit = logit.clamp(torch.logit(torch.tensor(eps)), torch.logit(torch.tensor(1-eps)))
    #     return torch.sigmoid(logit)
    def true_f_psi(X_psi):
        x1 = X_psi[..., :1] # shape: (n_sites, 1)
        logit1 = x1 @ torch.tensor([[-1.0], [0.0]], device=X_psi.device, dtype=X_psi.dtype).T # shape: (n_sites, n_species)
        x2 = X_psi[..., 1] # shape: (n_sites,)
        mu = torch.tensor([0.3, -0.2], device=X_psi.device, dtype=X_psi.dtype)  # optima
        sigma = torch.tensor([0.5, 0.4], device=X_psi.device, dtype=X_psi.dtype)  # niche width
        logit2 = torch.stack([
            -((x2 - mu[0])**2 / sigma[0]**2),
            -((x2 - mu[1])**2 / sigma[1]**2),
        ], dim=-1) # shape: (n_sites, n_species)
        eps = 1e-8
        logit2 = logit2.clamp(torch.logit(torch.tensor(eps)), torch.logit(torch.tensor(1-eps)))
        return torch.sigmoid(logit1 + logit2)
    

    # * * * Models for D * * *
    # You can try various functions for true_f_d

    n_cov_d = 1
    
    def true_f_d(X_d):
        d = torch.sigmoid(
            torch.tensor([-1.0, 0.5]) 
            + X_d @ torch.tensor([[-1.0], [-2.0]]).T
        )
        return d
    # def true_f_d(X_d):
    #     x = X_d[..., 0]
    #     mu = torch.tensor([0.2, -0.4], device=X_d.device, dtype=X_d.dtype)
    #     sigma = torch.tensor([0.6, 0.3], device=X_d.device, dtype=X_d.dtype)
    #     logit = torch.stack([
    #         -((x - mu[0])**2 / sigma[0]**2),
    #         -((x - mu[1])**2 / sigma[1]**2),
    #     ], dim=-1)
    #     eps = 1e-8
    #     logit = logit.clamp(torch.logit(torch.tensor(eps)), torch.logit(torch.tensor(1-eps)))
    #     return torch.sigmoid(logit)
    # def true_f_d(X_d):
    #     x1 = X_d[..., :1] # shape: (n_visits, 1)
    #     logit1 = x1 @ torch.tensor([[-1.0], [-2.0]], device=X_d.device, dtype=X_d.dtype).T # shape: (n_visits, n_species)
    #     x2 = X_d[..., 1] # shape: (n_visits, )
    #     mu = torch.tensor([0.2, -0.4], device=X_d.device, dtype=X_d.dtype)
    #     sigma = torch.tensor([0.6, 0.3], device=X_d.device, dtype=X_d.dtype)
    #     logit2 = torch.stack([
    #         -((x2 - mu[0])**2 / sigma[0]**2),
    #         -((x2 - mu[1])**2 / sigma[1]**2),
    #     ], dim=-1) # shape: (n_visits, n_species)
    #     eps = 1e-8
    #     logit2 = logit2.clamp(torch.logit(torch.tensor(eps)), torch.logit(torch.tensor(1-eps)))
    #     return torch.sigmoid(logit1 + logit2)

    
    # * * * Simulate data * * *

    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, true_z = simulate_non_linear(
        n_sites=1000, 
        n_total_visits_max=int(1e5),
        n_cov_psi=n_cov_psi,
        n_cov_d=n_cov_d,
        f_effort=lambda x1: torch.exp(3*x1),
        # f_effort=lambda x1: 10 * x1 ** 2,
        # f_effort=lambda x1: torch.ones_like(x1) * 100,
        f_psi=true_f_psi,
        f_d=true_f_d,
        seed=None,
        plot_summary=False,
        verbose=True
    )

    # -------------------------------------------------------------
    # Initialize results DataFrame
    # -------------------------------------------------------------

    eps = 1e-8
    true_logit_psi = torch.logit(true_f_psi(X_psi).clamp(eps, 1-eps))
    true_logit_d = torch.logit(true_f_d(X_d).clamp(eps, 1-eps))
    nll_true = nll_logits(y, indices_visit_site=torch.tensor(visit_site_ids), logit_psi=true_logit_psi, logit_d=true_logit_d).item()

    df_results = pl.DataFrame({
        "model": ['true'],
        "fit_method": ['true'],
        "penalization": ['None'],
        "n_epochs": [0],
        "converged": [True],
        "final_train_loss": [nll_true]
    })
    
    # -------------------------------------------------------------
    # Build OccuData
    # -------------------------------------------------------------
    
    # Give names to the site ids
    fmt = f"site_{{:0{len(str(max(visit_site_ids)))}d}}"
    visit_site_ids = [fmt.format(i) for i in visit_site_ids]
    site_ids_in_X_psi = [fmt.format(i) for i in site_ids_in_X_psi]
    
    data = OccuData(
        y=y, 
        visit_site_ids=visit_site_ids, 
        X_psi=X_psi, 
        X_d=X_d,
        site_ids_in_X_psi=site_ids_in_X_psi
    )

    # -------------------------------------------------------------
    # Define models, learning rates and fitting methods
    # -------------------------------------------------------------
    
    # The dictionnary all_models contains various parameters for the different runs we want to try. 
    # For each model setup, we can try different fitting methods, learning rates, and penalization schemes.

    all_models = {
        "linear_mlp-psi_16-d_16": {
            'model': LinearMLPOccuModel(
                n_species=y.shape[1], 
                n_cov_psi_linear=1, n_cov_psi_mlp=X_psi.shape[1]-1,
                n_cov_d_linear=1, n_cov_d_mlp=X_d.shape[1]-1,
                hidden_dims_psi=[16], hidden_dims_d=[16],
                activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
                dropout_psi=0.0, dropout_d=0.0
            ), 
            'methods': [
                'adam',
                'sgd',
                'lbfgs_full_batch', 
                'adam', 
                'adam', 
                # 'adam', 
                # 'adam', 
                'adam'#, 
                # 'adam', 
                # 'lbfgs_full_batch'
            ],
            'lrs': [
                1e-2, 
                5e-3,
                1e-2,
                1e-2,
                1e-2,
                # 1e-2,
                # 1e-2,
                5e-3,
                # 1e-2,
                # 1e-2
            ],
            'penalty': [
                (None, 0.0, 0.0), 
                ("l2_params", 1e-4, 0.1), 
                ("l2_params", 1e-4, 0.1), 
                ("l2_params", 1e-4, 0.1), 
                ("l1_params", 0.01, 0.01), 
                # ("l2_logits", 0.1, 0.1),
                # ("l1_logits", 0.1, 0.1),
                ("l2_logits_naive", 0.01, 0.01)#,
                # ("l1_logits_naive", 0.1, 0.1),
                # ("l1_logits_naive", 0.1, 0.1)
            ]
        },
        # "linear_mlp-psi_16-d_none": {
        #     'model': LinearMLPOccuModel(
        #         n_species=y.shape[1], 
        #         n_cov_psi_linear=1, n_cov_psi_mlp=1,
        #         n_cov_d_linear=1, n_cov_d_mlp=0,
        #         hidden_dims_psi=[16], hidden_dims_d=[],
        #         activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #         dropout_psi=0.0, dropout_d=0.0
        #     ), 
        #     'methods': ['adam'],
        #     'lrs': [1e-2]
        # },
        # "mlp-psi_64_32_16-d_64_32_16": {
        #     'model': MLPOccuModel(
        #         n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
        #         hidden_dims_psi=[64, 32, 16], hidden_dims_d=[64, 32, 16],
        #         activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #         dropout_psi=0.0, dropout_d=0.0
        #     ), 
        #     'methods': ['adam'],
        #     'lrs': [1e-3]
        # },
        # "mlp-psi_16-d_16": {
        #     'model': MLPOccuModel(
        #         n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
        #         hidden_dims_psi=[16], hidden_dims_d=[16],
        #         activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #         dropout_psi=0.0, dropout_d=0.0
        #     ), 
        #     'methods': ['adam', 'adam'],
        #     'lrs': [3e-3, 3e-3],
        #     'penalty': [(None, 0.0, 0.0), ("l2_params", 0.1, 0.1)]
        # },
        # "linear-psi-d (adam)": {
        #     'model': MLPOccuModel(
        #         n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
        #         hidden_dims_psi=[], hidden_dims_d=[],
        #         activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #         dropout_psi=0.0, dropout_d=0.0
        #     ), 
        #     'methods': ['adam'],
        #     'lrs': [1e-2]
        # },
        # "linear-psi-d (lbfgs)": {
        #     'model': MLPOccuModel(
        #         n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1],
        #         hidden_dims_psi=[], hidden_dims_d=[],
        #         activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #         dropout_psi=0.0, dropout_d=0.0
        #     ), 
        #     'methods': ['lbfgs_full_batch'],
        #     'lrs': [1e-4]
        # },
        # "constant-psi-d": {
        #     'model': MLPOccuModel(
        #         n_species=y.shape[1], n_cov_psi=0, n_cov_d=0,
        #         hidden_dims_psi=[], hidden_dims_d=[],
        #         activation_fn_psi=torch.nn.ReLU, activation_fn_d=torch.nn.ReLU,
        #         dropout_psi=0.0, dropout_d=0.0
        #     ), 
        #     'methods': ['lbfgs_full_batch'],
        #     'lrs': [1e-3]
        # },
    }


    # -------------------------------------------------------------
    # Prepare fitting loop
    # -------------------------------------------------------------
        
    # Compute naive estimators (just curiosity check)

    from occupycy.fit.naive_estimator import fit_naive_estimator
    center_logit_psi, center_logit_d, _, _ = fit_naive_estimator(data=data)
    print("Naive estimators:")
    print(f"  psi: {torch.sigmoid(center_logit_psi).cpu().numpy()}")
    print(f"  d:   {torch.sigmoid(center_logit_d).cpu().numpy()}")

    # Prepare plots

    fig, axes, ax0, fig_pred_psi, axes_pred_psi, fig_pred_d, axes_pred_d, \
        x_psi, x_d, order_points_psi, order_points_d, contours_psi, contours_d, \
        overall_min_logit_psi, overall_max_logit_psi, overall_min_logit_d, overall_max_logit_d = \
        non_lin_initialize_plots(
            y=y, X_psi=X_psi, X_d=X_d,
            n_cov_psi=n_cov_psi, n_cov_d=n_cov_d, true_f_psi=true_f_psi, true_f_d=true_f_d,
            all_models=all_models
        )
    
    idx_plot = 0


    # -------------------------------------------------------------
    # FITTING LOOP
    # -------------------------------------------------------------

    for model_name, this_dict in all_models.items(): # Loop over models

        print(f"\n=== Fitting {model_name} =================================")

        for idx_meth, fit_method in enumerate(this_dict['methods']): # Loop over fitting methods for this model
            
            idx_plot += 1
            label = f"{model_name} | {fit_method}"

            model = copy.deepcopy(this_dict['model'])
            data.reset_dataloaders()

            # In case penalty is not specified for this method, we set it to None with lambda=0 (i.e. no penalization)
            if 'penalty' in this_dict:
                label += f"\npenalty={this_dict['penalty'][idx_meth]}"
                print(f"> {fit_method} with penalty={this_dict['penalty'][idx_meth]}")
                penalization_type = this_dict['penalty'][idx_meth][0]
                penalization_lambda_psi = this_dict['penalty'][idx_meth][1]
                penalization_lambda_d = this_dict['penalty'][idx_meth][2]
            else:
                print(f"> {fit_method}")
                penalization_type = None
                penalization_lambda_psi = 0.0
                penalization_lambda_d = 0.0
        
            # * * * Fit model * * *

            if fit_method == "lbfgs_full_batch":

                if isinstance(model.psi_model, MLPBlock) or isinstance(model.d_model, MLPBlock):
                    print("Skipping lbfgs_full_batch for mlp model.")
                    continue
                res = fit_lbfgs_full_batch(
                    data, model, max_iter=1000, 
                    lr=this_dict['lrs'][idx_meth],
                    device=device,
                    tolerance_grad=1e-8, tolerance_change=1e-10, history_size=100, line_search_fn="strong_wolfe"
                )

            elif fit_method == "lbfgs":
                
                res = fit_sgd_like(
                    data, model, n_epochs=300, batch_size=data.n_sites, 
                    lr_psi=this_dict['lrs'][idx_meth], lr_d=this_dict['lrs'][idx_meth],
                    optimizer_psi_cls=torch.optim.LBFGS, optimizer_d_cls=torch.optim.LBFGS,
                    device=device,
                    penalization_type=penalization_type,
                    penalization_lambda_psi=penalization_lambda_psi,
                    penalization_lambda_d=penalization_lambda_d
                )

            elif fit_method == "sgd":
                
                res = fit_sgd_like(
                    data, model, n_epochs=300, batch_size=100, 
                    lr_psi=this_dict['lrs'][idx_meth], lr_d=this_dict['lrs'][idx_meth],
                    optimizer_psi_cls=torch.optim.SGD, optimizer_d_cls=torch.optim.SGD,
                    device=device,
                    penalization_type=penalization_type,
                    penalization_lambda_psi=penalization_lambda_psi,
                    penalization_lambda_d=penalization_lambda_d
                )

            elif fit_method == "adam":

                res = fit_sgd_like(
                    data, model, n_epochs=300, batch_size=100, 
                    lr_psi=this_dict['lrs'][idx_meth], lr_d=this_dict['lrs'][idx_meth],
                    optimizer_psi_cls=torch.optim.Adam, optimizer_d_cls=torch.optim.Adam,
                    device=device,
                    penalization_type=penalization_type,
                    penalization_lambda_psi=penalization_lambda_psi,
                    penalization_lambda_d=penalization_lambda_d
                )

            # * * * Store results * * *

            df_results = df_results.vstack(pl.DataFrame({
                "model": [model_name],
                "fit_method": [fit_method],
                "penalization": [str(this_dict['penalty'][idx_meth]) if 'penalty' in this_dict else 'None'],
                "n_epochs": [res.n_epochs],
                "converged": [res.converged],
                "final_train_loss": [res.final_train_loss]
            }))
            
            non_lin_plot_each_loop(
                res=res, model=model, label=label, idx_plot=idx_plot,
                axes=axes, ax0=ax0, fig_pred_psi=fig_pred_psi, axes_pred_psi=axes_pred_psi, fig_pred_d=fig_pred_d, 
                axes_pred_d=axes_pred_d, contours_psi=contours_psi, contours_d=contours_d,
                y=y, X_psi=X_psi, X_d=X_d, x_psi=x_psi, x_d=x_d, n_cov_psi=n_cov_psi, n_cov_d=n_cov_d, 
                true_logit_psi=true_logit_psi, true_logit_d=true_logit_d,
                order_points_psi=order_points_psi, order_points_d=order_points_d,
                overall_min_logit_psi=overall_min_logit_psi, overall_max_logit_psi=overall_max_logit_psi, 
                overall_min_logit_d=overall_min_logit_d, overall_max_logit_d=overall_max_logit_d
            )

    non_lin_finalize_plots(
        n_species=y.shape[1], fig=fig, axes=axes, ax0=ax0, fig_pred_psi=fig_pred_psi, axes_pred_psi=axes_pred_psi, 
        fig_pred_d=fig_pred_d, axes_pred_d=axes_pred_d, contours_psi=contours_psi, contours_d=contours_d,
        n_cov_psi=n_cov_psi, n_cov_d=n_cov_d, nll_true=nll_true,
        overall_min_logit_psi=overall_min_logit_psi, overall_max_logit_psi=overall_max_logit_psi, 
        overall_min_logit_d=overall_min_logit_d, overall_max_logit_d=overall_max_logit_d
    )

    # * * * Show results * * *

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        # Sorted by final_train_loss
        print(df_results.sort("final_train_loss"))

    plt.show()


if __name__ == "__main__":
    main()
