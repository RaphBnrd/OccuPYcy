from typing import Callable, Optional, Dict, Tuple, TypeVar

from copy import deepcopy
import torch
import polars as pl

from .lbfgs import fit_lbfgs_full_batch
from .sgd_like import fit_sgd_like
from .result import FitResult
from occupycy.data import OccuData
from occupycy.likelihood import nll_logits
from occupycy.models import OccuModel


T = TypeVar("T")
Grid = T | list[T] | tuple[T, ...]

def _ensure_list(x):
    """
    Normalize a scalar or sequence to a list.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    list
        ``x`` as a list.
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _build_combinations(param_grid, grid_mode):
    """
    Build hyperparameter combinations from a grid.

    Parameters
    ----------
    param_grid : dict[str, list]
        Parameter grid.
    grid_mode : {"combine_all", "match_by_index"}
        Combination strategy.

    Returns
    -------
    list[dict]
        List of parameter combinations.
    """
    keys = list(param_grid.keys())
    
    if grid_mode == "combine_all":
        for k in param_grid.keys(): # Remove duplicates while preserving order
            seen = set()
            param_grid[k] = [x for x in param_grid[k] if not (x in seen or seen.add(x))]
        from itertools import product
        combinations = [
            dict(zip(keys, values))
            for values in product(*param_grid.values())
        ]
    
    elif grid_mode == "match_by_index":
        lengths = [len(v) for v in param_grid.values() if len(v) > 1]
        if len(set(lengths)) != 1:
            raise ValueError("All parameters with more than one value must have the same number of values when using 'match_by_index' mode.")
        combinations = []
        for i in range(lengths[0]):
            combination = {}
            for key, values in param_grid.items():
                combination[key] = values[i] if len(values) > 1 else values[0]
            combinations.append(combination)

    else:
        raise ValueError("Invalid grid_mode. Must be 'combine_all' or 'match_by_index'.")
    
    return combinations

def _prepare_splitting(
        data: OccuData,
        splitting: Optional[Dict[str, list]]
    ) -> Tuple[OccuData, OccuData]:
    """
    Validate split dictionary and create train/validation subsets.

    Returns
    -------
    tuple[OccuData, OccuData]
        ``(data_train, data_val)``.
    """    
    if not isinstance(splitting, dict) or "train" not in splitting or "val" not in splitting:
        raise ValueError("Splitting must be a dictionary with 'train' and 'val' keys containing lists of site ids.")
    if not isinstance(splitting["train"], list) or not isinstance(splitting["val"], list):
        raise ValueError("'train' and 'val' in splitting must be lists of site ids.")
    if not all(site_id in data.mapping_site_idx for site_id in splitting["train"]):
        raise ValueError("All site ids in 'train' splitting must be valid (present in data.mapping_site_idx).")
    if not all(site_id in data.mapping_site_idx for site_id in splitting["val"]):
        raise ValueError("All site ids in 'val' splitting must be valid (present in data.mapping_site_idx).")
    if set(splitting["train"]).intersection(set(splitting["val"])):
        raise ValueError("Train and validation site ids must not overlap.")
  
    data_train = data.filtered_sites(splitting["train"])
    data_val = data.filtered_sites(splitting["val"])

    return data_train, data_val

def compute_criterion(
    model: OccuModel,
    data_val: OccuData,
    criterion_for_best: str,
) -> float:
    """
    Compute model selection criterion on validation data.

    Parameters
    ----------
    model : OccuModel
        Fitted model.
    data_val : OccuData
        Validation dataset.
    criterion_for_best : str
        Criterion name.

    Returns
    -------
    float
        Criterion value.
    """
    model.eval()
    with torch.no_grad():
        logit_psi, logit_d = model.forward_logits(
            X_psi=data_val.X_psi,
            X_d=data_val.X_d,
        ) # logit_psi shape: (n_sites, n_species), logit_d shape: (n_visits, n_species)
    if data_val.X_psi is None: # Broadcast if intercept-only
        logit_psi = logit_psi.expand(data_val.n_sites, -1)
    if data_val.X_d is None: # Broadcast if intercept-only
        logit_d = logit_d.expand(data_val.n_visits, -1)

    if criterion_for_best == "val_nll_logits":
        val_loss = nll_logits(
            data_val.y,
            logit_psi=logit_psi,
            logit_d=logit_d,
            indices_visit_site=data_val.indices_visit_site,
            site_reduction="sum",
            species_reduction="mean"
        )
        criterion = val_loss.cpu().item()
    
    else:
        raise ValueError(f"Invalid criterion_for_best: {criterion_for_best}")
    
    return criterion

def fit_lbfgs_full_batch_cv(
    data: OccuData,
    splitting: Optional[Dict[str, list]],
    grid_mode: str, # "combine_all" or "match_by_index"
    # Hyperparameters to cross-validate
    model: Grid[OccuModel],
    *,
    max_iter: Grid[int] = 200,
    lr: Grid[float] = 1.0,
    tolerance_grad: Grid[float | None] = None,
    tolerance_change: Grid[float | None] = None,
    history_size: Grid[int | None] = None,
    line_search_fn: Grid[Optional[str]] = None,
    site_reduction: Grid[str] = "sum",
    species_reduction: Grid[str] = "mean",
    loss_fn: Grid[Callable[..., torch.Tensor]] = nll_logits,
    device: Grid[Optional[torch.device]] = None,
    penalization_type: Grid[Optional[str]] = None,
    penalization_lambda_psi: Grid[Optional[float]] = None,
    penalization_lambda_d: Grid[Optional[float]] = None,
    # Additional args
    criterion_for_best: str = "val_nll_logits",
    verbose: str = "all"
) -> Tuple[FitResult, OccuModel, pl.DataFrame, list, list]:
    """
    Run hyperparameter search for full-batch L-BFGS fitting.

    Parameters
    ----------
    data : OccuData
        Dataset to fit (contains observations, covariates and mappings).
    splitting : dict or None
        Dictionary with keys ``'train'`` and ``'val'`` mapping to lists of site ids
        used to create the training and validation splits.
    grid_mode : str
        Combination strategy for the hyperparameter grid. One of
        ``'combine_all'`` or ``'match_by_index'``.
    model : OccuModel or sequence of OccuModel
        Model instance or iterable of model instances to evaluate during CV.
    max_iter : int or sequence, optional
        Maximum number of L-BFGS iterations (or grid values).
    lr : float or sequence, optional
        Learning rate(s) for the optimizer.
    tolerance_grad : float or None or sequence, optional
        Gradient tolerance(s) for termination.
    tolerance_change : float or None or sequence, optional
        Change-in-parameter tolerance(s) for termination.
    history_size : int or None or sequence, optional
        L-BFGS history size(s).
    line_search_fn : str or None or sequence, optional
        Line search function name(s) passed to the L-BFGS solver.
    site_reduction : str or sequence, optional
        Reduction mode over sites (e.g. ``'sum'``).
    species_reduction : str or sequence, optional
        Reduction mode over species (e.g. ``'mean'``).
    loss_fn : callable or sequence, optional
        Loss function(s) to use for fitting (default: ``nll_logits``).
    device : torch.device or None or sequence, optional
        Device(s) to run fitting on (CPU or CUDA device).
    penalization_type : str or None or sequence, optional
        Penalization type(s) (e.g. ``'l2'``) or ``None`` for no penalization.
    penalization_lambda_psi : float or None or sequence, optional
        Penalization strength(s) applied to psi parameters.
    penalization_lambda_d : float or None or sequence, optional
        Penalization strength(s) applied to detection parameters.
    criterion_for_best : str, optional
        Criterion used to select the best hyperparameters on validation data
        (default: ``'val_nll_logits'``).
    verbose : str, optional
        Verbosity level, one of ``'all'``, ``'summary'`` or ``'none'``.

    Returns
    -------
    tuple[FitResult, OccuModel, polars.DataFrame, list, list]
        Best fit result, best model, results table, all fit results, all models.
    
    Notes
    -----
    Doesn't modify the model. 
    Splitting shout contain keys 'train' and 'val' with lists site ids.
    More details in the docstring of ``fit_lbfgs_full_batch``.
    """

    # Prepare hyperparameter grid
    from_modelname_to_model = {f"model_{i+1}": m for i, m in enumerate(_ensure_list(model))}
    param_grid = {
        "model_name": list(from_modelname_to_model.keys()),
        "max_iter": _ensure_list(max_iter),
        "lr": _ensure_list(lr),
        "tolerance_grad": _ensure_list(tolerance_grad),
        "tolerance_change": _ensure_list(tolerance_change),
        "history_size": _ensure_list(history_size),
        "line_search_fn": _ensure_list(line_search_fn),
        "site_reduction": _ensure_list(site_reduction),
        "species_reduction": _ensure_list(species_reduction),
        "loss_fn": _ensure_list(loss_fn),
        "device": _ensure_list(device),
        "penalization_type": _ensure_list(penalization_type),
        "penalization_lambda_psi": _ensure_list(penalization_lambda_psi),
        "penalization_lambda_d": _ensure_list(penalization_lambda_d),
    }

    combinations = _build_combinations(param_grid, grid_mode)

    # Prepare data splits
    data_train, data_val = _prepare_splitting(data, splitting)

    # Prepare results storage
    all_fit_results = []
    all_fitted_models = []
    df_results = []
    
    for idx_combo, combo in enumerate(combinations):
        
        this_model = deepcopy(from_modelname_to_model[combo["model_name"]])
        
        if verbose == "all":
            print(f"\n ==== Fitting n°{idx_combo+1}/{len(combinations)} ====")
            for k, v in combo.items():
                print(f"  {k}: {v}")
            print("PSI Model:")
            print(this_model.psi_model)
            print("D Model:")
            print(this_model.d_model)
            print()
    
        # Fit model on training data
        res = fit_lbfgs_full_batch(
            data_train, 
            this_model, 
            max_iter=combo["max_iter"], 
            lr=combo["lr"], 
            tolerance_grad=combo["tolerance_grad"],
            tolerance_change=combo["tolerance_change"], 
            history_size=combo["history_size"], 
            line_search_fn=combo["line_search_fn"], 
            site_reduction=combo["site_reduction"], 
            species_reduction=combo["species_reduction"],
            loss_fn=combo["loss_fn"],
            device=combo["device"],
            penalization_type=combo["penalization_type"],
            penalization_lambda_psi=combo["penalization_lambda_psi"],
            penalization_lambda_d=combo["penalization_lambda_d"]
        )

        # Evaluate on validation data
        criterion = compute_criterion(
            model=this_model,
            data_val=data_val,
            criterion_for_best=criterion_for_best
        )

        # Store in results
        all_fit_results.append(res)
        all_fitted_models.append(this_model)

        new_row = {k: v for k, v in combo.items() if k != "model"}
        new_row["train_loss"] = res.final_train_loss
        new_row[criterion_for_best] = criterion
        df_results.append(new_row)

        if verbose == "all":
            print(f"> {criterion_for_best}: {criterion:.4f}")

    df_results = pl.DataFrame(df_results)

    # Find best model
    best_idx = df_results.select(pl.col(criterion_for_best)).to_series().arg_min()
    best_fit_result = all_fit_results[best_idx]
    best_model = all_fitted_models[best_idx]

    if verbose == "all" or verbose == "summary":
        df_print = deepcopy(df_results)
        for col_func in ["loss_fn"]:
            df_print = df_print.with_columns(pl.Series(
                col_func, [getattr(f, "__name__", str(f)) for f in df_results[col_func]]
            ))
        cols_to_drop = []
        for col, dtype in zip(df_print.columns, df_print.dtypes):
            if col in ["train_loss", "val_nll_logits"] or dtype == pl.Object:
                continue
            if df_print[col].n_unique() == 1:
                cols_to_drop.append(col)
        df_print = df_print.drop(cols_to_drop)
        print(df_print)
        print(f"\nFitting n°{best_idx+1} selected as best with {criterion_for_best}={df_results[criterion_for_best][best_idx]:.4f}")
    
    return best_fit_result, best_model, df_results, all_fit_results, all_fitted_models

def fit_sgd_like_cv(
    data: OccuData,
    splitting: Optional[Dict[str, list]],
    grid_mode: str, # "combine_all" or "match_by_index"
    # Hyperparameters to cross-validate
    model: Grid[OccuModel],
    *,
    batch_size: Grid[int] = 256,
    n_epochs: Grid[int] = 50,
    lr_psi: Grid[float] = 1e-3,
    lr_d: Grid[float] = 1e-3,
    optimizer_psi_cls: Grid[Callable] = torch.optim.Adam,
    optimizer_d_cls: Grid[Callable] = torch.optim.Adam,
    site_reduction: Grid[str] = "sum",
    species_reduction: Grid[str] = "mean",
    loss_fn: Grid[Callable[..., torch.Tensor]] = nll_logits,
    device: Grid[Optional[torch.device]] = None,
    patience_early_stopping: Grid[Optional[int]] = None,
    min_delta_early_stopping: Grid[Optional[float]] = 0.0,
    score_early_stopping: Grid[str] = "val_loss",
    patience_reduceLROnPlateau: Grid[Optional[int]] = None,
    factor_reduceLROnPlateau: Grid[float] = 0.5,
    score_reduceLROnPlateau: Grid[str] = "val_loss",
    penalization_type: Grid[Optional[str]] = None,
    penalization_lambda_psi: Grid[Optional[float]] = None,
    penalization_lambda_d: Grid[Optional[float]] = None,
    # Additional args
    seed_dataloader: Optional[int] = None,
    criterion_for_best: str = "val_nll_logits",
    verbose: str = "all"
) -> Tuple[FitResult, OccuModel, pl.DataFrame, list, list]:
    """
    Run hyperparameter search for SGD-like mini-batch fitting.

    Parameters
    ----------
    data : OccuData
        Dataset to fit (contains observations, covariates and mappings).
    splitting : dict or None
        Dictionary with keys ``'train'`` and ``'val'`` mapping to lists of site ids
        used to create the training and validation splits.
    grid_mode : str
        Combination strategy for the hyperparameter grid. One of
        ``'combine_all'`` or ``'match_by_index'``.
    model : OccuModel or sequence of OccuModel
        Model instance or iterable of model instances to evaluate during CV.
    batch_size : int or sequence, optional
        Mini-batch size(s) used for training.
    n_epochs : int or sequence, optional
        Number of epochs (or grid values) for training.
    lr_psi : float or sequence, optional
        Learning rate(s) for the occupancy (`psi`) parameters.
    lr_d : float or sequence, optional
        Learning rate(s) for the detection (`d`) parameters.
    optimizer_psi_cls : callable or sequence, optional
        Optimizer class(es) for psi parameters (e.g. ``torch.optim.Adam``).
    optimizer_d_cls : callable or sequence, optional
        Optimizer class(es) for detection parameters.
    site_reduction : str or sequence, optional
        Reduction mode over sites (e.g. ``'sum'``).
    species_reduction : str or sequence, optional
        Reduction mode over species (e.g. ``'mean'``).
    loss_fn : callable or sequence, optional
        Loss function(s) to use for fitting (default: ``nll_logits``).
    device : torch.device or None or sequence, optional
        Device(s) to run fitting on (CPU or CUDA device).
    patience_early_stopping : int or None or sequence, optional
        Patience value(s) for early stopping (in epochs) or ``None`` to disable.
    min_delta_early_stopping : float or None or sequence, optional
        Minimum change to qualify as improvement for early stopping.
    score_early_stopping : str or sequence, optional
        Score name used to monitor early stopping (e.g. ``'val_loss'``).
    patience_reduceLROnPlateau : int or None or sequence, optional
        Patience for reducing the learning rate on plateau.
    factor_reduceLROnPlateau : float or sequence, optional
        Multiplicative factor for reducing the learning rate.
    score_reduceLROnPlateau : str or sequence, optional
        Score name used to monitor ReduceLROnPlateau.
    penalization_type : str or None or sequence, optional
        Penalization type(s) (e.g. ``'l2'``) or ``None`` for no penalization.
    penalization_lambda_psi : float or None or sequence, optional
        Penalization strength(s) applied to psi parameters.
    penalization_lambda_d : float or None or sequence, optional
        Penalization strength(s) applied to detection parameters.
    seed_dataloader : int or None, optional
        Seed used to initialise dataloaders for reproducible shuffling.
    criterion_for_best : str, optional
        Criterion used to select the best hyperparameters on validation data
        (default: ``'val_nll_logits'``).
    verbose : str, optional
        Verbosity level, one of ``'all'``, ``'summary'`` or ``'none'``.

    Returns
    -------
    tuple[FitResult, OccuModel, polars.DataFrame, list, list]
        Best fit result, best model, results table, all fit results, all models.
    
    Notes
    -----
    Doesn't modify the model. 
    Splitting shout contain keys 'train' and 'val' with lists site ids.
    More details in the docstring of ``fit_sgd_like``.
    """

    # Prepare hyperparameter grid
    from_modelname_to_model = {f"model_{i+1}": m for i, m in enumerate(_ensure_list(model))}
    param_grid = {
        "model_name": list(from_modelname_to_model.keys()),
        "batch_size": _ensure_list(batch_size),
        "n_epochs": _ensure_list(n_epochs),
        "lr_psi": _ensure_list(lr_psi),
        "lr_d": _ensure_list(lr_d),
        "optimizer_psi_cls": _ensure_list(optimizer_psi_cls),
        "optimizer_d_cls": _ensure_list(optimizer_d_cls),
        "site_reduction": _ensure_list(site_reduction),
        "species_reduction": _ensure_list(species_reduction),
        "loss_fn": _ensure_list(loss_fn),
        "device": _ensure_list(device),
        "patience_early_stopping": _ensure_list(patience_early_stopping),
        "min_delta_early_stopping": _ensure_list(min_delta_early_stopping),
        "score_early_stopping": _ensure_list(score_early_stopping),
        "patience_reduceLROnPlateau": _ensure_list(patience_reduceLROnPlateau),
        "factor_reduceLROnPlateau": _ensure_list(factor_reduceLROnPlateau),
        "penalization_type": _ensure_list(penalization_type),
        "penalization_lambda_psi": _ensure_list(penalization_lambda_psi),
        "penalization_lambda_d": _ensure_list(penalization_lambda_d),
        "score_reduceLROnPlateau": _ensure_list(score_reduceLROnPlateau)
    }

    combinations = _build_combinations(param_grid, grid_mode)

    # Prepare data splits
    _, data_val = _prepare_splitting(data, splitting)

    # Prepare results storage
    all_fit_results = []
    all_fitted_models = []
    df_results = []
    
    for idx_combo, combo in enumerate(combinations):
        
        this_model = deepcopy(from_modelname_to_model[combo["model_name"]])
        
        if verbose == "all":
            print(f"\n ==== Fitting n°{idx_combo+1}/{len(combinations)} ====")
            for k, v in combo.items():
                print(f"  {k}: {v}")
            print("PSI Model:")
            print(this_model.psi_model)
            print("D Model:")
            print(this_model.d_model)
            print()

        # Drop dataloaders from data
        data.train_loader, data.val_loader, data.test_loader = None, None, None
    
        # Fit model on training data
        res = fit_sgd_like(
            data,
            this_model,
            batch_size=combo["batch_size"],
            n_epochs=combo["n_epochs"],
            lr_psi=combo["lr_psi"],
            lr_d=combo["lr_d"],
            optimizer_psi_cls=combo["optimizer_psi_cls"],
            optimizer_d_cls=combo["optimizer_d_cls"],
            site_reduction=combo["site_reduction"],
            species_reduction=combo["species_reduction"],
            loss_fn=combo["loss_fn"],
            device=combo["device"],
            seed_dataloader=seed_dataloader,
            splitting=splitting,
            patience_early_stopping=combo["patience_early_stopping"],
            min_delta_early_stopping=combo["min_delta_early_stopping"],
            score_early_stopping=combo["score_early_stopping"],
            patience_reduceLROnPlateau=combo["patience_reduceLROnPlateau"],
            factor_reduceLROnPlateau=combo["factor_reduceLROnPlateau"],
            score_reduceLROnPlateau=combo["score_reduceLROnPlateau"],
            penalization_type=combo["penalization_type"],
            penalization_lambda_psi=combo["penalization_lambda_psi"],
            penalization_lambda_d=combo["penalization_lambda_d"]
        )

        # Evaluate on validation data
        criterion = compute_criterion(
            model=this_model,
            data_val=data_val,
            criterion_for_best=criterion_for_best
        )

        # Store in results
        all_fit_results.append(res)
        all_fitted_models.append(this_model)

        new_row = {k: v for k, v in combo.items() if k != "model"}
        new_row["train_loss"] = res.final_train_loss
        new_row[criterion_for_best] = criterion
        df_results.append(new_row)

        if verbose == "all":
            print(f"> {criterion_for_best}: {criterion:.4f}")

    df_results = pl.DataFrame(df_results)

    # Find best model
    best_idx = df_results.select(pl.col(criterion_for_best)).to_series().arg_min()
    best_fit_result = all_fit_results[best_idx]
    best_model = all_fitted_models[best_idx]

    if verbose == "all" or verbose == "summary":
        df_print = deepcopy(df_results)
        for col_func in ["loss_fn", "optimizer_psi_cls", "optimizer_d_cls"]:
            df_print = df_print.with_columns(pl.Series(
                col_func, [getattr(f, "__name__", str(f)) for f in df_results[col_func]]
            ))
        cols_to_drop = []
        for col, dtype in zip(df_print.columns, df_print.dtypes):
            if col in ["train_loss", "val_nll_logits"] or dtype == pl.Object:
                continue
            if df_print[col].n_unique() == 1:
                cols_to_drop.append(col)
        df_print = df_print.drop(cols_to_drop)
        print(df_print)
        print(f"\nFitting n°{best_idx+1} selected as best with {criterion_for_best}={df_results[criterion_for_best][best_idx]:.4f}")
    
    return best_fit_result, best_model, df_results, all_fit_results, all_fitted_models
