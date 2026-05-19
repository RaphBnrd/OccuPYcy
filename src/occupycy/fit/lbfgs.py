from __future__ import annotations

from typing import Callable, Optional

import torch
from tqdm import tqdm
from copy import deepcopy

from .result import FitResult
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel
from occupycy.likelihood.penalty import loss_penalty
from .checkpoints import save_checkpoint_if_best


def fit_lbfgs_full_batch(
    data: OccuData,
    model: OccuModel,
    *,
    max_iter: int = 200,
    lr: float = 1.0,
    tolerance_grad: float | None = None,
    tolerance_change: float | None = None,
    history_size: int | None = None,
    line_search_fn: Optional[str] = None,
    site_reduction: str = "sum",
    species_reduction: str = "mean",
    loss_fn: Callable[..., torch.Tensor] = nll_logits,
    loss_fn_eval: Optional[Callable[..., torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    penalization_type: Optional[str] = None,
    penalization_lambda_psi: Optional[float] = None,
    penalization_lambda_d: Optional[float] = None,
    path_checkpoints: Optional[str] = None
) -> FitResult:
    """
    Fit an occupancy model with full-batch L-BFGS.

    Parameters
    ----------
    data : OccuData
        Input dataset.
    model : OccuModel
        Model to optimize (modified in place).
    max_iter : int, default=200
        Maximum L-BFGS iterations.
    lr : float, default=1.0
        L-BFGS learning rate.
    tolerance_grad : float or None, default=None
        L-BFGS convergence criterion on gradient norm.
    tolerance_change : float or None, default=None
        L-BFGS convergence criterion on loss change.
    history_size : int or None, default=None
        L-BFGS history size for approximating inverse Hessian.
    line_search_fn : str or None, default=None
        Line search method for L-BFGS ("strong_wolfe", "backtracking", etc.).
    site_reduction : str, default="sum"
        Reduction method for aggregating loss over sites ("sum", "mean", etc.).
    species_reduction : str, default="mean"
        Reduction method for aggregating loss over species ("sum", "mean", etc.).
    loss_fn : callable, default=nll_logits
        Loss function used by LBFGS.
    loss_fn_eval : callable or None, default=None
        Optional loss function for evaluation at each iteration (if different from loss_fn).
         If provided, eval loss is computed at each iteration and stored in history as 'val_loss'.
         If None, no eval loss is computed.
    device : torch.device or None, default=None
        Target device for computation. If None, uses model's parameter device.
    penalization_type : str or None, default=None
        Type of penalization to apply to the loss (e.g., "l2_logits", "l1_logits", "l2_logits_naive", "l1_logits_naive").
    penalization_lambda_psi : float or None, default=None
        Strength of penalization for occupancy logits (used if penalization_type is not None).
    penalization_lambda_d : float or None, default=None
        Strength of penalization for detection logits (used if penalization_type is not None).
    path_checkpoints : str or None, default=None
        Path where checkpoint is saved.

    Returns
    -------
    FitResult
        Fitting summary and train-loss history.

    Raises
    ------
    ValueError
        If inputs are empty, model outputs have invalid shapes, or loss is non-finite.

    Notes
    -----
    The model is modified in-place. 
    """
    if device is None:
        device = next(model.psi_model.parameters()).device

    eval_loss = False
    if loss_fn_eval is not None and loss_fn_eval != loss_fn:
        eval_loss = True
        print("Evaluation loss function provided; will compute eval loss at each iteration. Stored in history as 'val_loss'.")

    data.to(device=device)
    model.to(device=device)

    y = data.y
    indices_visit_site = data.indices_visit_site

    if indices_visit_site.numel() == 0:
        raise ValueError("Cannot fit with zero visits.")

    X_psi = data.X_psi
    X_d = data.X_d

    # Penalization setup
    if penalization_type in ["l2_logits_naive", "l1_logits_naive"]:
        from .naive_estimator import fit_naive_estimator
        center_logit_psi, center_logit_d, _, _ = fit_naive_estimator(
            data=data,
            loss_fn=loss_fn,
            device=device
        ) # naive logits of shape: (n_species,)
    elif penalization_type in ["l2_logits", "l1_logits"]:
        center_logit_psi = torch.zeros((model.n_species,), device=device)
        center_logit_d = torch.zeros((model.n_species,), device=device)
    else:
        center_logit_psi, center_logit_d = None, None

    # LBFGS optimizer setup
    lbfgs_kwargs = dict(max_iter=max_iter)

    if tolerance_grad is not None:
        lbfgs_kwargs["tolerance_grad"] = tolerance_grad
    if tolerance_change is not None:
        lbfgs_kwargs["tolerance_change"] = tolerance_change
    if history_size is not None:
        lbfgs_kwargs["history_size"] = history_size
    if line_search_fn is not None:
        lbfgs_kwargs["line_search_fn"] = line_search_fn
    
    
    parameters = list(model.psi_model.parameters()) + list(model.d_model.parameters())
    parameters = [p for p in parameters if p.requires_grad]
    if len(parameters) == 0:
        raise ValueError("No parameters to optimize. Ensure that model parameters have requires_grad=True.")
    opt = torch.optim.LBFGS(parameters, lr=lr, **lbfgs_kwargs)

    # Closure function for LBFGS optimization
    n_iter = 0
    last_loss = None
    train_loss_history = []
    eval_loss_history = []
    pbar = None

    def closure():

        nonlocal n_iter, last_loss, train_loss_history, eval_loss_history, \
            pbar, path_checkpoints, model, opt, lr

        
        if n_iter == 0:
            pbar = tqdm(total=max_iter, desc="LBFGS iterations", unit="iter")
        opt.zero_grad(set_to_none=True)

        logit_psi, logit_d = model.forward_logits(X_psi=X_psi, X_d=X_d)

        # Expand logits if X_psi or X_d were None (model would output shape (1, n_species) 
        # but we need to broadcast to all sites/visits for loss calculation)
        if X_psi is None:
            logit_psi = logit_psi.expand(data.n_sites, -1)
        if X_d is None:
            logit_d = logit_d.expand(data.n_visits, -1)

        # Basic shape validation (helps catch mistakes in custom models early)
        if logit_psi.ndim != 2 or logit_d.ndim != 2:
            raise ValueError("Model must return 2D tensors for (logit_psi, logit_d).")
        if logit_psi.shape[0] != data.n_sites:
            raise ValueError(
                f"Model returned logit_psi with n_sites={logit_psi.shape[0]}, expected {data.n_sites}."
            )
        if logit_d.shape[0] != data.n_visits:
            raise ValueError(
                f"Model returned logit_d with n_visits={logit_d.shape[0]}, expected {data.n_visits}."
            )
        if logit_psi.shape[1] != y.shape[1] or logit_d.shape[1] != y.shape[1]:
            raise ValueError("Model species dimension must match y.shape[1].")

        nll = loss_fn(
            y,
            logit_psi=logit_psi,
            logit_d=logit_d,
            indices_visit_site=indices_visit_site,
            site_reduction=site_reduction,
            species_reduction=species_reduction,
        )

        penalty = loss_penalty(
            penalization_type=penalization_type,
            penalization_lambda_psi=penalization_lambda_psi,
            penalization_lambda_d=penalization_lambda_d,
            model=model,
            logit_psi=logit_psi,
            logit_d=logit_d,
            center_logit_psi=center_logit_psi,
            center_logit_d=center_logit_d,
            site_reduction=site_reduction,
            species_reduction=species_reduction
        )
        loss = nll + penalty
        
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered: {loss.item()}")
        loss.backward()
        last_loss = loss.detach()
        n_iter += 1
        train_loss_history.append(float(last_loss.cpu().item()))
        
        if eval_loss:
            this_eval_loss = loss_fn_eval(
                y,
                logit_psi=logit_psi,
                logit_d=logit_d,
                indices_visit_site=indices_visit_site,
                site_reduction=site_reduction,
                species_reduction=species_reduction,
            ).detach()
            eval_loss_history.append(float(this_eval_loss.cpu().item()))

        pbar.set_postfix({"train_loss": last_loss.cpu().item()})
        pbar.update(1)

        # Save the checkpoint
        save_checkpoint_if_best(
            path_checkpoints=path_checkpoints,
            epoch=n_iter-1,
            last_train_loss=last_loss,
            last_val_loss=None,
            final_params_model_psi=deepcopy(model.psi_model.state_dict()),
            final_params_model_d=deepcopy(model.d_model.state_dict()),
            opt_psi=None,
            opt_d=None,
            opt=opt,
            optimizer_psi_cls=torch.optim.LBFGS,
            optimizer_d_cls=torch.optim.LBFGS,
            train_loss_history=train_loss_history,
            val_loss_history=eval_loss_history,
            lr_psi_history=[lr] * n_iter,
            lr_d_history=[lr] * n_iter,
            verbose=False
        )

        return loss

    # Fit the model using LBFGS optimization
    print(f"Starting LBFGS fitting on device {device}")
    loss = opt.step(closure)
    pbar.close()

    print(f"LBFGS finished after {n_iter} iterations with final train loss {last_loss.cpu().item():.4f}")

    final = last_loss if last_loss is not None else loss.detach()

    save_checkpoint_if_best(
        path_checkpoints=path_checkpoints,
        # splitting=data.splitting,
        epoch=n_iter-1,
        last_train_loss=last_loss,
        last_val_loss=None,
        final_params_model_psi=deepcopy(model.psi_model.state_dict()),
        final_params_model_d=deepcopy(model.d_model.state_dict()),
        opt_psi=None,
        opt_d=None,
        opt=opt,
        optimizer_psi_cls=torch.optim.LBFGS,
        optimizer_d_cls=torch.optim.LBFGS,
        train_loss_history=train_loss_history,
        val_loss_history=eval_loss_history,
        lr_psi_history=[lr] * n_iter,
        lr_d_history=[lr] * n_iter,
        verbose=False
    )

    # LBFGS doesn't expose a robust convergence flag; keep best-effort.
    return FitResult(
        final_train_loss=float(final.cpu().item()), 
        final_val_loss=None,
        n_epochs=n_iter, 
        converged=True,
        history={"train_loss": train_loss_history, "val_loss": eval_loss_history},
        lr_psi=lr,
        lr_d=lr,
        optimizer_psi="LBFGS",
        optimizer_d="LBFGS",
        device=device
    )