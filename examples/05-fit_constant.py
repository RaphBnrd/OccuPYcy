import torch

import matplotlib.pyplot as plt
import polars as pl

from occupycy.data.simulate_data import simulate_constant

from occupycy.data.core import OccuData
from occupycy.fit import fit_lbfgs_full_batch, fit_sgd_like
from occupycy.likelihood import nll_logits
from occupycy.models import ConstantOccuModel


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

    y, visit_site_ids, true_psi, true_d, observed_psi, observed_d = simulate_constant(
        n_sites=400, 
        n_visits_per_site=25, 
        psi=torch.tensor([0.7, 0.5, 0.1, 0.05]),
        d=torch.tensor([0.01, 0.1, 0.9, 0.99]),
        seed=None
    )


    # -------------------------------------------------------------
    # Initialize results DataFrame
    # -------------------------------------------------------------

    n_sites = len(set(visit_site_ids))
    n_visits = len(visit_site_ids)
    
    nll_true = nll_logits(
        y, indices_visit_site=torch.tensor(visit_site_ids), 
        logit_psi=torch.logit(true_psi.unsqueeze(0).expand(n_sites, -1)),
        logit_d=torch.logit(true_d.unsqueeze(0).expand(n_visits, -1))
    ).item()
    nll_observed = nll_logits(
        y, indices_visit_site=torch.tensor(visit_site_ids), 
        logit_psi=torch.logit(observed_psi.unsqueeze(0).expand(n_sites, -1)),
        logit_d=torch.logit(observed_d.unsqueeze(0).expand(n_visits, -1))
    ).item()

    all_results = pl.DataFrame({
        "fit_method": ['true'] * len(true_psi) + ['observed'] * len(observed_psi),
        "n_epochs": [0] * (len(true_psi) + len(observed_psi)),
        "converged": [True] * (len(true_psi) + len(observed_psi)),
        "final_train_loss": [nll_true] * len(true_psi) + [nll_observed] * len(observed_psi),
        "species_idx": list(range(len(true_psi))) + list(range(len(observed_psi))),
        "psi_hat": torch.cat([true_psi, observed_psi]).tolist(),
        "d_hat": torch.cat([true_d, observed_d]).tolist(),
    }).with_columns(
        pl.col("final_train_loss").cast(pl.Float64)
    )
    

    # -------------------------------------------------------------
    # Build OccuData
    # -------------------------------------------------------------
    
    # No covariates in this example, so no need for X_psi, X_d or site_ids_in_X_psi
    data = OccuData(y=y, visit_site_ids=visit_site_ids)

    
    # -------------------------------------------------------------
    # FITTING LOOP
    # -------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Loop over different fitting methods
    for fit_method in ["lbfgs_full_batch", "lbfgs", "sgd", "adam"]:

        print(f"\n--- Fitting with {fit_method} ---")
        
        # * * * Define model * * *
        
        model = ConstantOccuModel(n_species=data.n_species)
    
        # * * * Fit model * * *

        if fit_method == "lbfgs_full_batch":

            res = fit_lbfgs_full_batch(
                data, model, max_iter=1000, 
                lr=0.1,
                device=device,
                tolerance_grad=1e-8, tolerance_change=1e-10, history_size=100, line_search_fn="strong_wolfe"
            )

        elif fit_method == "lbfgs":
            
            res = fit_sgd_like(
                data, model, n_epochs=50, batch_size=data.n_sites, 
                lr_psi=0.1, lr_d=0.1,
                optimizer_psi_cls=torch.optim.LBFGS, optimizer_d_cls=torch.optim.LBFGS,
                device=device
            )
            
        elif fit_method == "sgd":
            
            res = fit_sgd_like(
                data, model, n_epochs=50, batch_size=100, 
                lr_psi=2e-2, lr_d=2e-2,
                optimizer_psi_cls=torch.optim.SGD, optimizer_d_cls=torch.optim.SGD,
                device=device
            )

        elif fit_method == "adam":
            
            res = fit_sgd_like(
                data, model, n_epochs=50, batch_size=100, 
                lr_psi=2e-1, lr_d=2e-1,
                optimizer_psi_cls=torch.optim.Adam, optimizer_d_cls=torch.optim.Adam,
                device=device
            )

        # * * * Store results * * *

        psi_hat = torch.sigmoid(model.psi_model.logit_species.detach())
        d_hat = torch.sigmoid(model.d_model.logit_species.detach())

        all_results = all_results.vstack(pl.DataFrame({
            "fit_method": [fit_method] * len(psi_hat),
            "n_epochs": [res.n_epochs] * len(psi_hat),
            "converged": [res.converged] * len(psi_hat),
            "final_train_loss": [res.final_train_loss] * len(psi_hat),
            "species_idx": list(range(len(psi_hat))),
            "psi_hat": psi_hat.tolist(),
            "d_hat": d_hat.tolist(),
        }))
        
        plt.plot(res.history["train_loss"], label=f"{fit_method}\n(final nll={res.final_train_loss:.2f})")
        # Plot horizontal lines for true and observed nll
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
    plt.axhline(nll_true, color="gray", linestyle="--", label=f"(true nll={nll_true:.2f})")
    plt.axhline(nll_observed, color="black", linestyle="--", label=f"(observed nll={nll_observed:.2f})")

    # * * * Show results * * *

    df_results = pl.DataFrame(all_results).pivot(
        index=["fit_method", "n_epochs", "converged", "final_train_loss"],
        on="species_idx",
        values=["psi_hat", "d_hat"],
    )
    column_order = ["fit_method", "n_epochs", "converged", "final_train_loss"]
    for species_idx in range(len(true_psi)):
        column_order += [f"psi_hat_{species_idx}", f"d_hat_{species_idx}"]
    df_results = df_results.select(column_order)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(df_results)

    plt.title("Fitting constant OccuModel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()