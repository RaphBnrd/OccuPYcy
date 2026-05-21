from __future__ import annotations

from typing import Callable, Optional, Dict, Tuple
import warnings

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
from .checkpoints import save_checkpoint_if_best


def forward_path_one_batch(
    batch: OccuData,
    model: OccuModel,
    loss_fn: Callable[..., torch.Tensor],
    site_reduction: str,
    species_reduction: str,
    device: torch.device
):
    """
    Compute forward pass and loss for one batch.

    Parameters
    ----------
    batch : dict
        Batched data from ``collate_occu_sites``.
    model : OccuModel
        Occupancy model.
    loss_fn : callable
        Loss function (typically ``nll_logits``).
    site_reduction : str
        Reduction over sites.
    species_reduction : str
        Reduction over species.
    device : torch.device
        Device used for computation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(loss, logit_psi, logit_d)``.

    Raises
    ------
    ValueError
        If computed loss is non-finite.
    """
    y = batch["y"].to(device=device) # shape: (n_visits, n_species)
    indices_visit_site = batch["indices_visit_site"].to(device=device)
    X_psi = batch["X_psi"] # shape: (n_sites, n_cov_psi) or None
    if X_psi is not None:
        X_psi = X_psi.to(device=device)
    X_d = batch["X_d"] # shape: (n_visits, n_cov_d) or None
    if X_d is not None:
        X_d = X_d.to(device=device)

    logit_psi, logit_d = model.forward_logits(
        X_psi=X_psi,
        X_d=X_d,
    ) # logit_psi shape: (n_sites, n_species), logit_d shape: (n_visits, n_species)

    # Broadcast if intercept-only
    if X_psi is None:
        n_sites = len(torch.unique(indices_visit_site))
        logit_psi = logit_psi.expand(n_sites, -1)
    if X_d is None:
        n_visits = y.shape[0]
        logit_d = logit_d.expand(n_visits, -1)
    
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
    loss_fn_eval: Optional[Callable[..., torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed_dataloader: Optional[int] = None,
    val_fraction: float = 0.0,
    test_fraction: float = 0.0,
    splitting: Optional[Dict[str, list]] = None,
    patience_early_stopping: Optional[int] = None,
    min_delta_early_stopping: float = 0.0,
    score_early_stopping: str = "val_loss",
    patience_reduceLROnPlateau: Optional[int] = None,
    factor_reduceLROnPlateau: float = 0.5,
    min_lr_reduceLROnPlateau: float = 1e-5,
    score_reduceLROnPlateau: str = "val_loss",
    penalization_type: Optional[str] = None,
    penalization_lambda_psi: Optional[float] = None,
    penalization_lambda_d: Optional[float] = None,
    path_checkpoints: Optional[str] = None
) -> FitResult:
    """
    Fit an occupancy model with mini-batch SGD-like optimization.

    Parameters
    ----------
    data : OccuData
        Input data.
    model : OccuModel
        Model to optimize (modified in place).
    batch_size : int, default=256
        Number of sites per batch.
    n_epochs : int, default=50
        Maximum number of epochs.
    lr_psi : float, default=1e-3
        Learning rate for occupancy parameters.
    lr_d : float, default=1e-3
        Learning rate for detection parameters.
    optimizer_psi_cls, optimizer_d_cls : callable, default=torch.optim.Adam
        Optimizer classes for occupancy and detection branches.
    site_reduction : str, default="sum"
        Reduction over sites for the loss ("sum", "mean", or "none").
    species_reduction : str, default="mean"
        Reduction over species for the loss ("sum", "mean", or "none").
    loss_fn : callable, default=nll_logits
        Loss function to optimize.
    loss_fn_eval : callable or None, default=None
        Loss function for validation. If None, uses loss_fn.
    device : torch.device or None, default=None
        Device for computation. If None, uses the device of the model parameters.
    seed_dataloader : int or None, default=None
        Random seed for dataloader shuffling.
    val_fraction : float, default=0.0
        Fraction of data to use for validation if no val_loader is provided.
    test_fraction : float, default=0.0
        Fraction of data to use for testing if no test_loader is provided.
    splitting : dict or None, default=None
        Explicit splitting of sites into "train", "val", and "test" lists. Overrides val_fraction and test_fraction if provided.
    patience_early_stopping : int or None, default=None
        Number of epochs with no improvement after which training will be stopped. If None, no early stopping is performed.
    min_delta_early_stopping : float, default=0.0
        Minimum change in the monitored score to qualify as an improvement for early stopping.
    score_early_stopping : str, default="val_loss"
        Score to monitor for early stopping ("val_loss" or "train_loss").
    patience_reduceLROnPlateau : int or None, default=None
        Number of epochs with no improvement after which learning rate will be reduced. If None, no learning rate scheduling is performed.
    factor_reduceLROnPlateau : float, default=0.5
        Factor by which the learning rate will be reduced when using ReduceLROnPlateau.
    min_lr_reduceLROnPlateau : float, default=1e-5
        Minimum learning rate to which the optimizer will be reduced.
    score_reduceLROnPlateau : str, default="val_loss"
        Score to monitor for ReduceLROnPlateau ("val_loss" or "train_loss").
    penalization_type : str or None, default=None
        Type of penalization to apply ("l2_params", "l1_params", "l2_logits", "l1_logits", "l2_logits_naive", "l1_logits_naive", or None).
    penalization_lambda_psi : float or None, default=None
        Strength of penalization for occupancy parameters (used if penalization_type is "l2_params" or "l1_params").
    penalization_lambda_d : float or None, default=None
        Strength of penalization for detection parameters (used if penalization_type is "l2_params" or "l1_params").
    path_checkpoints : str or None, default=None
        Path where best checkpoint is saved.

    Returns
    -------
    FitResult
        Training summary and history.

    Notes
    -----
    Supported penalties:
    ``None, l2_params, l1_params, l2_logits, l1_logits, l2_logits_naive, l1_logits_naive``.
    The model is modified in-place. 
    If using early stopping, the best model is loaded at the end. 
    Otherwise, the final model is returned.
    """

    if device is None:
        device = next(model.psi_model.parameters()).device

    if loss_fn_eval is None:
        loss_fn_eval = loss_fn

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
        n_sites_in_dataloader = len(set(next(iter(data.train_loader))["indices_visit_site"]))
        if n_sites_in_dataloader != batch_size:
            warnings.warn(
                f"Provided dataloader has batch size {n_sites_in_dataloader}, "
                f"which does not match the specified batch_size {batch_size}. "
                f"Using the provided dataloader's batch size.",
                UserWarning
            )

    # Penalization setup
    if penalization_type in ["l2_logits_naive", "l1_logits_naive"]:
        if splitting is None or "train" not in splitting:
            if data.train_loader is None:
                raise ValueError("Splitting information is required for naive logits penalization. Please provide a splitting dictionary with a 'train' key or ensure that the dataloader is built with a train split.")
            else:
                train_sites_indices = []
                for batch in data.train_loader:
                    batch_site_indices = batch["indices_visit_site"].cpu().numpy()
                    train_sites_indices.extend(batch_site_indices)
                train_sites_indices = sorted(set(int(x) for x in train_sites_indices))
                train_sites_ids = [data.site_ids_in_X_psi[i] for i in train_sites_indices]
        else:
            train_sites_ids = splitting["train"]
        center_logit_psi, center_logit_d, _, _ = fit_naive_estimator(
            data=data.filtered_sites(train_sites_ids),
            loss_fn=loss_fn,
            device=device
        ) # naive logits of shape: (n_species,)
    elif penalization_type in ["l2_logits", "l1_logits"]:
        center_logit_psi = torch.zeros((model.n_species,), device=device)
        center_logit_d = torch.zeros((model.n_species,), device=device)
    else:
        center_logit_psi, center_logit_d = None, None
    
    # Optimizer
    parameters_psi = [p for p in model.psi_model.parameters() if p.requires_grad]
    parameters_d = [p for p in model.d_model.parameters() if p.requires_grad]
    # if sum(p.numel() for p in parameters_psi) + sum(p.numel() for p in parameters_d) == 0:
    if len(parameters_psi) + len(parameters_d) == 0:
        raise ValueError("No parameters to optimize. Ensure that model parameters have requires_grad=True.")
    if optimizer_psi_cls == torch.optim.LBFGS or optimizer_d_cls == torch.optim.LBFGS:
        if optimizer_psi_cls != optimizer_d_cls:
            raise ValueError("LBFGS optimizer must be used for both psi and d.")
        if lr_psi != lr_d:
            raise ValueError("LBFGS optimizer must use the same learning rate for psi and d.")
        opt = torch.optim.LBFGS(parameters_psi + parameters_d, lr=lr_psi, max_iter=1)
        opt_psi, opt_d = None, None
    else:
        # NB: L2 regularization is implemented as weight decay in the optimizer
        opt_psi = optimizer_psi_cls(
            parameters_psi,
            lr=lr_psi, 
            weight_decay=penalization_lambda_psi if penalization_type == "l2_params" else 0.0
        )
        opt_d = optimizer_d_cls(
            parameters_d, 
            lr=lr_d, 
            weight_decay=penalization_lambda_d if penalization_type == "l2_params" else 0.0
        )
        opt = None

    # Early stopping
    if patience_early_stopping is not None:
        if data.val_loader is None and score_early_stopping == "val_loss":
            raise ValueError("Early stopping requires a validation set. Please provide a val_loader in the data or set patience_early_stopping to None.")
        elif score_early_stopping not in ["val_loss", "train_loss"]:
            raise ValueError(f"Invalid score_early_stopping: {score_early_stopping}. Must be 'val_loss' or 'train_loss'.")
        early_stopper = EarlyStopper(
            patience=patience_early_stopping, 
            min_delta=min_delta_early_stopping
        )

    # LR scheduler
    if patience_reduceLROnPlateau is not None:
        if data.val_loader is None and score_reduceLROnPlateau == "val_loss":
            raise ValueError("ReduceLROnPlateau requires a validation set if score_reduceLROnPlateau is set to 'val_loss'. Please provide a val_loader in the data or set patience_reduceLROnPlateau to None or score_reduceLROnPlateau to 'train_loss'.")
        elif score_reduceLROnPlateau not in ["val_loss", "train_loss"]:
            raise ValueError(f"Invalid score_reduceLROnPlateau: {score_reduceLROnPlateau}. Must be 'val_loss' or 'train_loss'.")
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        kwargs_scheduler = {
            "mode": "min",
            "factor": factor_reduceLROnPlateau,
            "patience": patience_reduceLROnPlateau,
            "min_lr": min_lr_reduceLROnPlateau
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
                        loss_fn=loss_fn_eval,
                        site_reduction=site_reduction,
                        species_reduction=species_reduction,
                        device=device
                    )
                    last_val_loss += float(val_loss.cpu().item())
            val_loss_history.append(last_val_loss)
        else:
            last_val_loss = None

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
                save_checkpoint_if_best(
                    path_checkpoints=path_checkpoints,
                    epoch=epoch,
                    last_train_loss=last_train_loss,
                    last_val_loss=last_val_loss,
                    final_params_model_psi=final_params_model_psi,
                    final_params_model_d=final_params_model_d,
                    opt_psi=opt_psi,
                    opt_d=opt_d,
                    opt=opt,
                    optimizer_psi_cls=optimizer_psi_cls,
                    optimizer_d_cls=optimizer_d_cls,
                    train_loss_history=train_loss_history,
                    val_loss_history=val_loss_history,
                    lr_psi_history=lr_psi_history,
                    lr_d_history=lr_d_history,
                    verbose=False
                )
        else:
            if last_train_loss < final_train_loss:
                final_train_loss = last_train_loss
                final_params_model_psi = deepcopy(model.psi_model.state_dict())
                final_params_model_d = deepcopy(model.d_model.state_dict())
                epoch_best_model = epoch
    
        save_checkpoint_if_best(
            path_checkpoints=path_checkpoints,
            epoch=epoch,
            last_train_loss=last_train_loss,
            last_val_loss=last_val_loss,
            final_params_model_psi=final_params_model_psi,
            final_params_model_d=final_params_model_d,
            opt_psi=opt_psi,
            opt_d=opt_d,
            opt=opt,
            optimizer_psi_cls=optimizer_psi_cls,
            optimizer_d_cls=optimizer_d_cls,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            lr_psi_history=lr_psi_history,
            lr_d_history=lr_d_history,
            verbose=False
        )
        
        # Update progress bar
        pbar.set_postfix({
            "train_loss": last_train_loss,
            "val_loss": last_val_loss if data.val_loader is not None else "N/A"
        })
        pbar.update(1)
        
        # Early stopping check
        if patience_early_stopping is not None:
            if score_early_stopping == "val_loss":
                score_ES = last_val_loss
            elif score_early_stopping == "train_loss":
                score_ES = last_train_loss
            if early_stopper.early_stop(score_ES):
                print(f"Early stopping at epoch n°{epoch + 1}")
                break
        
        # LR scheduler step
        if patience_reduceLROnPlateau is not None:
            if score_reduceLROnPlateau == "val_loss":
                score_RLROP = last_val_loss
            elif score_reduceLROnPlateau == "train_loss":
                score_RLROP = last_train_loss
            if optimizer_psi_cls == torch.optim.LBFGS:
                scheduler.step(score_RLROP)
            else:
                scheduler_state.step(score_RLROP)
                scheduler_det.step(score_RLROP)

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