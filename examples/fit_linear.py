import torch

import matplotlib.pyplot as plt
import polars as pl
from copy import deepcopy

from simulate_data import simulate_linear

from occupycy.data.core import OccuData
from occupycy.fit import fit_lbfgs_full_batch, fit_sgd_like
from occupycy.likelihood import nll_probs, nll_logits
from occupycy.models import LinearOccuModel


# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")

def main():

    # -------------------------------------------------------------
    # Simulate data
    # -------------------------------------------------------------

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


    # -------------------------------------------------------------
    # Initialize results DataFrame
    # -------------------------------------------------------------
    
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
    

    # -------------------------------------------------------------
    # Build OccuData
    # -------------------------------------------------------------
    
    init_data = OccuData(y=y, visit_site_ids=visit_site_ids, X_psi=X_psi, X_d=X_d)

    
    # -------------------------------------------------------------
    # FITTING LOOP
    # -------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Loop over different fitting methods
    for fit_method in ["lbfgs_full_batch", "lbfgs_FB", "lbfgs", "sgd", "adam"]:
    
        print(f"\n--- Fitting with {fit_method} ---")
        
        data = deepcopy(init_data)

        # * * * Define model * * *
        
        model = LinearOccuModel(
            n_species=y.shape[1], n_cov_psi=X_psi.shape[1], n_cov_d=X_d.shape[1]
        )
    
        # * * * Fit model * * *

        if fit_method == "lbfgs_full_batch":

            res = fit_lbfgs_full_batch(
                data, model, max_iter=1000, 
                # lr_psi=0.1, lr_d=0.1, 
                lr=5e-3,
                device=device,
                tolerance_grad=1e-8, tolerance_change=1e-10, history_size=100, line_search_fn="strong_wolfe"
            )

        elif fit_method == "lbfgs_FB":
            
            res = fit_sgd_like(
                data, model, n_epochs=20, batch_size=data.n_sites, 
                lr_psi=1e-1, lr_d=1e-1,
                optimizer_psi_cls=torch.optim.LBFGS, optimizer_d_cls=torch.optim.LBFGS,
                device=device
            )

        elif fit_method == "lbfgs":
            
            res = fit_sgd_like(
                data, model, n_epochs=20, batch_size=100, 
                lr_psi=5e-2, lr_d=5e-2,
                optimizer_psi_cls=torch.optim.LBFGS, optimizer_d_cls=torch.optim.LBFGS,
                device=device
            )

        elif fit_method == "sgd":
            
            res = fit_sgd_like(
                data, model, n_epochs=100, batch_size=100, 
                lr_psi=1e-3, lr_d=1e-3,
                optimizer_psi_cls=torch.optim.SGD, optimizer_d_cls=torch.optim.SGD,
                device=device
            )

        elif fit_method == "adam":
            
            res = fit_sgd_like(
                data, model, n_epochs=100, batch_size=100, 
                lr_psi=1e-2, lr_d=1e-2,
                optimizer_psi_cls=torch.optim.Adam, optimizer_d_cls=torch.optim.Adam,
                device=device
            )

        # * * * Store results * * *

        df_results = df_results.vstack(polar_res(
            fit_method=fit_method, n_epochs=res.n_epochs, converged=res.converged, final_train_loss=res.final_train_loss,
            intercept_psi=model.psi_model.bias.detach(), coefs_psi=model.psi_model.weight.detach(),
            intercept_d=model.d_model.bias.detach(), coefs_d=model.d_model.weight.detach()
        ))
        
        plt.plot(res.history["train_loss"], label=f"{fit_method}\n(final nll={res.final_train_loss:.2f})")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
    plt.axhline(nll_true, color="gray", linestyle="--", label=f"(true nll={nll_true:.2f})")

    # * * * Show results * * *

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(df_results)

    plt.title("Fitting constant OccuModel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()