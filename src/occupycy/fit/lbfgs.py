from __future__ import annotations

from typing import Callable, Optional

import torch

from .result import FitResult
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel


def fit_lbfgs_full_batch(
    data: OccuData,
    model: OccuModel,
    *,
    max_iter: int = 200,
    # lr_psi: float = 1.0,
    # lr_d: float = 1.0,
    lr: float = 1.0,
    tolerance_grad: float | None = None,
    tolerance_change: float | None = None,
    history_size: int | None = None,
    line_search_fn: Optional[str] = None,
    site_reduction: str = "sum",
    species_reduction: str = "mean",
    loss_fn: Callable[..., torch.Tensor] = nll_logits,
    device: Optional[torch.device] = None,
) -> FitResult:
    """
    General full-batch LBFGS fitting for any OccuModel.

    Requirements on the model:
      - model.forward_logits(X_psi=..., X_d=...) returns (logit_psi, logit_d)
      - logit_psi shape: (n_sites, n_species)
      - logit_d shape: (n_visits, n_species)

    Requirements on the data:
      - y: (n_visits, n_species)
      - visit_site_idxs: (n_visits,)
      - X_psi optional: (n_sites, n_cov_psi)
      - X_d optional: (n_visits, n_cov_d)
    """

    if device is None:
        device = next(model.psi_model.parameters()).device
    data = data.to(device=device)

    y = data.y
    visit_site_idxs = data.visit_site_idxs

    if visit_site_idxs.numel() == 0:
        raise ValueError("Cannot fit with zero visits.")

    X_psi = data.X_psi
    X_d = data.X_d

    lbfgs_kwargs = dict(max_iter=max_iter)

    if tolerance_grad is not None:
        lbfgs_kwargs["tolerance_grad"] = tolerance_grad
    if tolerance_change is not None:
        lbfgs_kwargs["tolerance_change"] = tolerance_change
    if history_size is not None:
        lbfgs_kwargs["history_size"] = history_size
    if line_search_fn is not None:
        lbfgs_kwargs["line_search_fn"] = line_search_fn

    # opt_psi = torch.optim.LBFGS(model.psi_model.parameters(), lr=lr_psi, **lbfgs_kwargs)
    # opt_d = torch.optim.LBFGS(model.d_model.parameters(), lr=lr_d, **lbfgs_kwargs)
    opt = torch.optim.LBFGS(
        list(model.psi_model.parameters()) + list(model.d_model.parameters()),
        lr=lr, **lbfgs_kwargs
    )

    n_iter = 0
    last_loss = None
    train_loss_history = []

    def closure():
        nonlocal n_iter, last_loss
        # opt_psi.zero_grad(set_to_none=True)
        # opt_d.zero_grad(set_to_none=True)
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

        loss = loss_fn(
            y,
            logit_psi=logit_psi,
            logit_d=logit_d,
            visit_site_idxs=visit_site_idxs,
            site_reduction=site_reduction,
            species_reduction=species_reduction,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered: {loss.item()}")
        loss.backward()
        last_loss = loss.detach()
        n_iter += 1
        train_loss_history.append(float(last_loss.cpu().item()))
        return loss

    # loss = opt_psi.step(closure)
    # opt_d.step(closure)
    loss = opt.step(closure)

    final = last_loss if last_loss is not None else loss.detach()

    # LBFGS doesn't expose a robust convergence flag; keep best-effort.
    return FitResult(
        final_train_loss=float(final.cpu().item()), 
        n_epochs=n_iter, 
        converged=True,
        loss_history={"train_loss": train_loss_history},
        # lr_psi=lr_psi,
        # lr_d=lr_d,
        lr_psi=lr,
        lr_d=lr,
        optimizer_psi="LBFGS",
        optimizer_d="LBFGS",
        device=device
    )