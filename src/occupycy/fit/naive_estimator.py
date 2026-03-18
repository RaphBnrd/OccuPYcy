from typing import Callable, Optional

import torch
from .result import FitResult
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel, ConstantOccuModel

def fit_naive_estimator(
    data: OccuData,
    *,
    loss_fn: Callable[..., torch.Tensor] = nll_logits,
    device: Optional[torch.device] = None,
) -> FitResult:
    """
    Naive estimator fitting for any OccuModel.
    """

    print("Fitting naive estimator...")

    if device is None:
        device = data.y.device

    data.to(device=device)

    model = ConstantOccuModel(
        n_species=data.n_species
    )
    model.to(device=device)

    # This sets the constant parameters to a rough estimate based on the data, which is a good starting point for LBFGS.
    model.initialize_rough_estimate(data)

    # We can use LBFGS to optimize the constant parameters, which is more efficient than SGD for this simple model.
    from .lbfgs import fit_lbfgs_full_batch
    
    fit_result = fit_lbfgs_full_batch(
        data=data,
        model=model,
        max_iter=100,
        lr=1.0,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        history_size=10,
        line_search_fn="strong_wolfe",
        site_reduction="sum",
        species_reduction="mean",
        loss_fn=loss_fn,
        device=device,
        penalization_type=None,
        penalization_lambda_psi=None,
        penalization_lambda_d=None
    )

    # Psi parameters are in model.psi_model.logit_species, d parameters are in model.d_model.logit_species
    logit_psi = model.psi_model.logit_species.detach().clone() # shape: (n_species,)
    logit_d = model.d_model.logit_species.detach().clone() # shape: (n_species,)

    return logit_psi, logit_d, model, fit_result
