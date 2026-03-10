from __future__ import annotations

from typing import Callable, Optional

import torch

from .result import FitResult
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel


def fit_sgd_like(
    data: OccuData,
    model: OccuModel,
    *,
    batch_size: int = 256,
    n_epochs: int = 50,
    lr_psi: float = 1e-3,
    lr_d: float = 1e-3,
    optimizer_psi_cls: Callable = torch.optim.Adam,
    optimizer_d_cls: Callable = torch.optim.Adam,
    site_reduction: str = "sum",
    species_reduction: str = "mean",
    loss_fn: Callable[..., torch.Tensor] = nll_logits,
    device: Optional[torch.device] = None,
    seed_dataloader: Optional[int] = None,
) -> FitResult:
    """
    Mini-batch SGD-style fitting for any OccuModel.
    """

    if device is None:
        device = next(model.psi_model.parameters()).device

    model.psi_model = model.psi_model.to(device)
    model.d_model = model.d_model.to(device)

    # Build dataloader (sites batched)
    data_with_dataloader = data.with_dataloaders(
        batch_size=batch_size,
        val_fraction=0.0,
        test_fraction=0.0,
        shuffle=True,
        seed=seed_dataloader,
        device=device
    )

    opt_psi = optimizer_psi_cls(model.psi_model.parameters(), lr=lr_psi)
    opt_d = optimizer_d_cls(model.d_model.parameters(), lr=lr_d)

    last_loss = None
    train_loss_history = []

    for epoch in range(n_epochs):
        
        print(f"\rEpoch {str(epoch + 1).zfill(len(str(n_epochs)))}/{n_epochs}", end="", flush=True)

        current_loss = []

        for batch in data_with_dataloader.train_loader:

            batch = batch.to(device=device)
            y = batch.y
            visit_site_idxs = batch.visit_site_idxs
            X_psi = batch.X_psi
            X_d = batch.X_d

            opt_psi.zero_grad()
            opt_d.zero_grad()

            logit_psi, logit_d = model.forward_logits(
                X_psi=X_psi,
                X_d=X_d,
            )

            # Broadcast if intercept-only
            if X_psi is None:
                logit_psi = logit_psi.expand(batch.n_sites, -1)
            if X_d is None:
                logit_d = logit_d.expand(batch.n_visits, -1)
            
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
            opt_psi.step()
            opt_d.step()
            current_loss.append(loss.detach().cpu().item())
        
        last_loss = torch.tensor(current_loss).sum()
        train_loss_history.append(float(last_loss))

    print("  -> fitting done")
    final = last_loss if last_loss is not None else torch.tensor(float("nan"))

    return FitResult(
        final_train_loss=float(final.cpu().item()), 
        n_epochs=epoch + 1,
        converged=True,
        loss_history={"train_loss": train_loss_history},
        lr_psi=lr_psi,
        optimizer_psi=optimizer_psi_cls.__name__,
        lr_d=lr_d,
        optimizer_d=optimizer_d_cls.__name__,
        device=device
    )