from __future__ import annotations

from typing import Callable, Optional, Dict, Tuple

import torch
from tqdm import tqdm
from copy import deepcopy

from .result import FitResult
from .early_stopper import EarlyStopper
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel
from .naive_estimator import fit_naive_estimator
from occupycy.likelihood.penalty import loss_penalty


def forward_path_one_batch(
    batch: OccuData,
    model: OccuModel,
    loss_fn: Callable[..., torch.Tensor],
    site_reduction: str,
    species_reduction: str,
    device: torch.device
):
    batch.to(device=device)
    y = batch.y # shape: (n_visits, n_species)
    indices_visit_site = batch.indices_visit_site
    X_psi = batch.X_psi # shape: (n_sites, n_cov_psi) or None
    X_d = batch.X_d # shape: (n_visits, n_cov_d) or None

    logit_psi, logit_d = model.forward_logits(
        X_psi=X_psi,
        X_d=X_d,
    ) # logit_psi shape: (n_sites, n_species), logit_d shape: (n_visits, n_species)

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
    
    return loss, logit_psi, logit_d


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
    patience_early_stopping: Optional[int] = None,
    patience_reduceLROnPlateau: Optional[int] = None,
    penalization_type: Optional[str] = None,
    penalization_lambda_psi: Optional[float] = None,
    penalization_lambda_d: Optional[float] = None
) -> FitResult:
    """
    Mini-batch SGD-style fitting for any OccuModel.

    It modifies the model in-place.

    Penalty implementations:
    [None, "l2_params", "l1_params", "l2_logits", "l1_logits", "l2_logits_naive", "l1_logits_naive"]
    """

    if device is None:
        device = next(model.psi_model.parameters()).device

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

    # Penalization setup
    if penalization_type in ["l2_logits_naive", "l1_logits_naive"]:
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
        # NB: L2 regularization is implemented as weight decay in the optimizer
        opt_psi = optimizer_psi_cls(
            model.psi_model.parameters(), 
            lr=lr_psi, 
            weight_decay=penalization_lambda_psi if penalization_type == "l2_params" else 0.0
        )
        opt_d = optimizer_d_cls(
            model.d_model.parameters(), 
            lr=lr_d, 
            weight_decay=penalization_lambda_d if penalization_type == "l2_params" else 0.0
        )

    # Early stopping
    if patience_early_stopping is not None:
        if data.val_loader is None:
            raise ValueError("Early stopping requires a validation set. Please provide a val_loader in the data or set patience_early_stopping to None.")
        early_stopper = EarlyStopper(patience=patience_early_stopping, min_delta=0.0)

    # LR scheduler
    if patience_reduceLROnPlateau is not None:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        kwargs_scheduler = {
            "mode": "min",
            "factor": 0.5,
            "patience": patience_reduceLROnPlateau,
            "min_lr": 1e-5
        }
        if optimizer_psi_cls == torch.optim.LBFGS:
            scheduler = ReduceLROnPlateau(opt, **kwargs_scheduler)
        else:
            scheduler_state = ReduceLROnPlateau(opt_psi, **kwargs_scheduler)
            scheduler_det = ReduceLROnPlateau(opt_d, **kwargs_scheduler)

    # Training loop
    train_loss_history = []
    val_loss_history = []
    lr_psi_history, lr_d_history = [], []

    data.to(device=device)
    model.to(device=device)
    print(f"Starting SGD-like fitting on device {device}")

    final_params_model_psi = deepcopy(model.psi_model.state_dict())
    final_params_model_d = deepcopy(model.d_model.state_dict())
    final_train_loss, final_val_loss = float('inf'), float('inf')
    epoch_best_model = 0

    pbar = tqdm(range(n_epochs), desc="Epochs", unit="epoch")
    for epoch in range(n_epochs):

        # Training steps
        model.train()
        last_train_loss = 0.0
        for batch in data.train_loader:

            if optimizer_psi_cls == torch.optim.LBFGS:
                def closure():
                    opt.zero_grad()
                    nll, logit_psi, logit_d = forward_path_one_batch(
                        batch=batch,
                        model=model,
                        loss_fn=loss_fn,
                        site_reduction=site_reduction,
                        species_reduction=species_reduction,
                        device=device
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
                    loss.backward()
                    return loss
                opt.step(closure)
                last_train_loss += closure().cpu().item()
            
            else:
                opt_psi.zero_grad()
                opt_d.zero_grad()
                nll, logit_psi, logit_d = forward_path_one_batch(
                    batch=batch,
                    model=model,
                    loss_fn=loss_fn,
                    site_reduction=site_reduction,
                    species_reduction=species_reduction,
                    device=device
                )
                penalty = loss_penalty(
                    penalization_type=penalization_type if penalization_type != "l2_params" else None, # L2 on parameters is handled by weight decay in the optimizer
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
                loss.backward()
                opt_psi.step()
                opt_d.step()
                last_train_loss += float(loss.cpu().item())
    
        train_loss_history.append(last_train_loss)
        
        # Validation steps
        if data.val_loader is not None:
            model.eval()
            last_val_loss = 0.0
            with torch.no_grad():
                for batch in data.val_loader:
                    val_loss, _, _ = forward_path_one_batch(
                        batch=batch,
                        model=model,
                        loss_fn=loss_fn,
                        site_reduction=site_reduction,
                        species_reduction=species_reduction,
                        device=device
                    )
                    last_val_loss += float(val_loss.cpu().item())
            val_loss_history.append(last_val_loss)

        lr_psi_history.append(opt_psi.param_groups[0]['lr'] if optimizer_psi_cls != torch.optim.LBFGS else opt.param_groups[0]['lr'])
        lr_d_history.append(opt_d.param_groups[0]['lr'] if optimizer_d_cls != torch.optim.LBFGS else opt.param_groups[0]['lr'])

        # Store best (eventually final) model and losses
        if data.val_loader is not None:
            if last_val_loss < final_val_loss:
                final_train_loss = last_train_loss
                final_val_loss = last_val_loss
                final_params_model_psi = deepcopy(model.psi_model.state_dict())
                final_params_model_d = deepcopy(model.d_model.state_dict())
                epoch_best_model = epoch
        else:
            if last_train_loss < final_train_loss:
                final_train_loss = last_train_loss
                final_params_model_psi = deepcopy(model.psi_model.state_dict())
                final_params_model_d = deepcopy(model.d_model.state_dict())
                epoch_best_model = epoch
        # Update progress bar
        pbar.set_postfix({
            "train_loss": last_train_loss,
            "val_loss": last_val_loss if data.val_loader is not None else "N/A"
        })
        pbar.update(1)
        
        # Early stopping check
        if patience_early_stopping is not None:
            if early_stopper.early_stop(last_val_loss):
                print(f"Early stopping at epoch n°{epoch + 1}")
                break
        
        # LR scheduler step
        if patience_reduceLROnPlateau is not None:
            if optimizer_psi_cls == torch.optim.LBFGS:
                scheduler.step(last_val_loss)
            else:
                scheduler_state.step(last_val_loss)
                scheduler_det.step(last_val_loss)
    
    pbar.close()

    history = {"train_loss": train_loss_history, "lr_psi": lr_psi_history, "lr_d": lr_d_history}
    if len(val_loss_history) > 0:
        history["val_loss"] = val_loss_history
    
    final_val_loss = last_val_loss if data.val_loader is not None else None
    model.psi_model.load_state_dict(final_params_model_psi)
    model.d_model.load_state_dict(final_params_model_d)
    
    print(f"SGD-like fitting finished after {n_epochs} epochs.\n"
          f"Best model at epoch n°{epoch_best_model + 1} with final train loss {final_train_loss:.4f}" + 
          (f" and final val loss {final_val_loss:.4f}" if data.val_loader is not None else ""))

    return FitResult(
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        n_epochs=epoch + 1,
        idx_epoch_best_model=epoch_best_model,
        converged=True,
        history=history,
        lr_psi=lr_psi,
        optimizer_psi=optimizer_psi_cls.__name__,
        lr_d=lr_d,
        optimizer_d=optimizer_d_cls.__name__,
        device=device
    )