from __future__ import annotations

from typing import Callable, Optional, Dict, Tuple

import torch
from tqdm import tqdm

from .result import FitResult
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel


def forward_path_one_batch(
    batch,
    model: OccuModel,
    loss_fn: Callable[..., torch.Tensor],
    site_reduction: str,
    species_reduction: str,
    device: torch.device
):
    batch.to(device=device)
    y = batch.y
    indices_visit_site = batch.indices_visit_site
    X_psi = batch.X_psi
    X_d = batch.X_d

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
        indices_visit_site=indices_visit_site,
        site_reduction=site_reduction,
        species_reduction=species_reduction,
    )

    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss encountered: {loss.item()}")
    
    return loss


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
    val_fraction: float = 0.0,
    test_fraction: float = 0.0,
    splitting: Optional[Dict[str, list]] = None,
) -> Tuple[FitResult, OccuData]:
    """
    Mini-batch SGD-style fitting for any OccuModel.
    """

    if device is None:
        device = next(model.psi_model.parameters()).device

    model.psi_model = model.psi_model.to(device)
    model.d_model = model.d_model.to(device)

    # Build dataloader (sites batched)
    if data.train_loader is None:
        print("Building dataloader for training...")
        data.build_dataloaders(
            batch_size=batch_size,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            splitting=splitting,
            shuffle=True,
            seed=seed_dataloader,
            device=device
        )
    else:
        if next(iter(data.train_loader)).n_sites != batch_size:
            print(f"Warning: provided dataloader has batch size {next(iter(data.train_loader)).n_sites}, which does not match the specified batch_size {batch_size}. Using the provided dataloader's batch size.")

    # Optimizer
    if optimizer_psi_cls == torch.optim.LBFGS or optimizer_d_cls == torch.optim.LBFGS:
        if optimizer_psi_cls != optimizer_d_cls:
            raise ValueError("LBFGS optimizer must be used for both psi and d.")
        if lr_psi != lr_d:
            raise ValueError("LBFGS optimizer must use the same learning rate for psi and d.")
        opt = torch.optim.LBFGS(
            list(model.psi_model.parameters()) + list(model.d_model.parameters()),
            lr=lr_psi, max_iter=1
        )
    else:
        opt_psi = optimizer_psi_cls(model.psi_model.parameters(), lr=lr_psi)
        opt_d = optimizer_d_cls(model.d_model.parameters(), lr=lr_d)

    train_loss_history = []
    val_loss_history = []

    pbar = tqdm(range(n_epochs), desc="Epochs", unit="epoch")
    for epoch in range(n_epochs):
        
        last_train_loss = 0.0
        for batch in data.train_loader:

            if optimizer_psi_cls == torch.optim.LBFGS:
                def closure():
                    opt.zero_grad()
                    loss = forward_path_one_batch(
                        batch=batch,
                        model=model,
                        loss_fn=loss_fn,
                        site_reduction=site_reduction,
                        species_reduction=species_reduction,
                        device=device
                    )
                    loss.backward()
                    return loss
                opt.step(closure)
                last_train_loss += closure().cpu().item()
            
            else:
                opt_psi.zero_grad()
                opt_d.zero_grad()
                loss = forward_path_one_batch(
                    batch=batch,
                    model=model,
                    loss_fn=loss_fn,
                    site_reduction=site_reduction,
                    species_reduction=species_reduction,
                    device=device
                )
                loss.backward()
                opt_psi.step()
                opt_d.step()
                last_train_loss += loss.cpu().item()
        
        train_loss_history.append(float(last_train_loss))
        
        if data.val_loader is not None:
            last_val_loss = 0.0
            with torch.no_grad():
                for batch in data.val_loader:
                    val_loss = forward_path_one_batch(
                        batch=batch,
                        model=model,
                        loss_fn=loss_fn,
                        site_reduction=site_reduction,
                        species_reduction=species_reduction,
                        device=device
                    )
                    last_val_loss += val_loss.cpu().item()
            val_loss_history.append(float(last_val_loss))

        pbar.set_postfix({
            "train_loss": last_train_loss,
            "val_loss": last_val_loss if data.val_loader is not None else "N/A"
        })
        pbar.update(1)
    
    loss_history = {"train_loss": train_loss_history}
    if len(val_loss_history) > 0:
        loss_history["val_loss"] = val_loss_history
    
    print(f"SGD-like fitting finished after {n_epochs} epochs with final train loss {last_train_loss:.4f}" + 
          (f" and final val loss {last_val_loss:.4f}" if data.val_loader is not None else ""))

    return FitResult(
        final_train_loss=last_train_loss,  
        n_epochs=epoch + 1,
        converged=True,
        loss_history=loss_history,
        lr_psi=lr_psi,
        optimizer_psi=optimizer_psi_cls.__name__,
        lr_d=lr_d,
        optimizer_d=optimizer_d_cls.__name__,
        device=device
    )