from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

SiteReduction = Literal["sum", "mean", "none"]
SpeciesReduction = Literal["mean", "sum", "none"]


def _validate_inputs(
    y: torch.Tensor | None = None,
    y_visits: torch.Tensor | None = None,
    y_labels: torch.Tensor | None = None,
    logit_psi: torch.Tensor | None = None,
    logit_d: torch.Tensor | None = None,
    indices_visit_site: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    """
    Validate tensor shapes, dtypes, and index consistency.

    Parameters
    ----------
    y : torch.Tensor
        Detection matrix of shape ``(n_visits, n_species)``.
    logit_psi : torch.Tensor
        Occupancy logits of shape ``(n_sites, n_species)``.
    logit_d : torch.Tensor
        Detection logits of shape ``(n_visits, n_species)``.
    indices_visit_site : torch.Tensor
        Site index per visit, shape ``(n_visits,)``.

    Returns
    -------
    tuple[int, int, int]
        ``(n_sites, n_visits, n_species)``.
    """

    if logit_psi is None or logit_d is None or indices_visit_site is None:
        raise ValueError("logit_psi, logit_d, and indices_visit_site must all be provided.")

    if y is not None:
        if y_visits is not None or y_labels is not None:
            raise ValueError("If y is provided, y_visits and y_labels must be None.")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (n_visits, n_species); got shape {tuple(y.shape)}")
        n_visits, n_species = y.shape
    else:
        if y_visits is None or y_labels is None:
            raise ValueError("If y is None, both y_visits and y_labels must be provided.")
        if y_visits.ndim != 1:
            raise ValueError(f"y_visits must be 1D (n_detections,); got shape {tuple(y_visits.shape)}")
        if y_labels.ndim != 1:
            raise ValueError(f"y_labels must be 1D (n_detections,); got shape {tuple(y_labels.shape)}")
        if y_visits.shape[0] != y_labels.shape[0]:
            raise ValueError(
                "y_visits and y_labels must have the same length."
            )
        if y_visits.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"y_visits must be a tensor of integers; got {y_visits.dtype}")
        if y_labels.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"y_labels must be a tensor of integers; got {y_labels.dtype}")
        n_visits = indices_visit_site.shape[0]
        n_species = logit_psi.shape[1]
    if logit_psi.ndim != 2:
        raise ValueError(f"logit_psi must be 2D (n_sites, n_species); got shape {tuple(logit_psi.shape)}")
    if logit_d.ndim != 2:
        raise ValueError(f"logit_d must be 2D (n_visits, n_species); got shape {tuple(logit_d.shape)}")
    if indices_visit_site.ndim != 1:
        raise ValueError(
            f"indices_visit_site must be 1D (n_visits,); got shape {tuple(indices_visit_site.shape)}"
        )

    n_sites, n_species_psi = logit_psi.shape
    n_visits_d, n_species_d = logit_d.shape

    if n_species_psi != n_species or n_species_d != n_species:
        raise ValueError(
            "Species dimension mismatch: "
            f"y has {n_species}, logit_psi has {n_species_psi}, logit_d has {n_species_d}."
        )
    if n_visits_d != n_visits:
        raise ValueError(f"Visit dimension mismatch: y has {n_visits}, logit_d has {n_visits_d}.")
    if indices_visit_site.shape[0] != n_visits:
        raise ValueError(
            f"indices_visit_site length mismatch: expected {n_visits}, got {indices_visit_site.shape[0]}."
        )

    if y is None and y_visits.numel() > 0:
        if y_visits.min() < 0 or y_visits.max() >= n_visits:
            raise ValueError(
                f"y_visits entries must be in [0, n_visits-1] where n_visits={n_visits}."
            )
        if y_labels.min() < 0 or y_labels.max() >= n_species:
            raise ValueError(
                f"y_labels entries must be in [0, n_species-1] where n_species={n_species}."
            )
        coordinates = torch.stack((y_visits, y_labels), dim=1)
        if torch.unique(coordinates, dim=0).shape[0] != coordinates.shape[0]:
            raise ValueError("Duplicate (visit, species) detections are not allowed.")

    if indices_visit_site.dtype not in (torch.int64, torch.int32):
        raise ValueError("indices_visit_site must be an int tensor (int64/int32).")

    if n_sites == 0 or n_visits == 0 or n_species == 0:
        raise ValueError("Empty inputs are not supported (n_sites, n_visits, n_species must be > 0).")

    min_idx = int(indices_visit_site.min().item())
    max_idx = int(indices_visit_site.max().item())
    n_idx = indices_visit_site.unique().numel()
    if min_idx != 0 or max_idx != n_sites - 1 or n_idx != n_sites:
        raise ValueError(f"indices_visit_site must be in [0, n_sites-1] with no missing indices, but got min {min_idx}, max {max_idx}, unique count {n_idx} for X_psi rows {n_sites}.")

    return n_sites, n_visits, n_species


def nll_logits(
    y: torch.Tensor,
    *,
    logit_psi: torch.Tensor,
    logit_d: torch.Tensor,
    indices_visit_site: torch.Tensor,
    site_reduction: SiteReduction = "sum",
    species_reduction: SpeciesReduction = "mean",
    validate_inputs: bool = False,
) -> torch.Tensor:
    """
    Negative log-likelihood in logit space for single-season multi-species occupancy.

    Parameters
    ----------
    y : torch.Tensor
        Visit-level detections, shape ``(n_visits, n_species)``.
    logit_psi : torch.Tensor
        Occupancy logits, shape ``(n_sites, n_species)``.
    logit_d : torch.Tensor
        Detection logits, shape ``(n_visits, n_species)``.
    indices_visit_site : torch.Tensor
        Site index per visit.
    site_reduction : {"sum", "mean", "none"}, default="sum"
        Site-level reduction.
    species_reduction : {"mean", "sum", "none"}, default="mean"
        Species-level reduction.
    validate_inputs : bool, default=False
        Whether to validate input shapes, dtypes, and index consistency. 
        Can be set to False for performance if inputs are guaranteed to be correct 
        (e.g., when called from a training loop).

    Returns
    -------
    torch.Tensor
        NLL tensor with shape determined by reductions.
    """
    y = torch.as_tensor(y) # shape: (n_visits, n_species)
    logit_psi = torch.as_tensor(logit_psi) # shape: (n_sites, n_species)
    logit_d = torch.as_tensor(logit_d) # shape: (n_visits, n_species)
    indices_visit_site = torch.as_tensor(indices_visit_site) # shape: (n_visits,)

    if validate_inputs:
        n_sites, _, n_species = _validate_inputs(
            y=y, 
            logit_psi=logit_psi, 
            logit_d=logit_d, 
            indices_visit_site=indices_visit_site
        )
    else:
        n_sites, n_species = logit_psi.shape

    device = y.device
    y_dtype = y.dtype
    
    # Binary indicator of whether each species was detected at each site
    detected = torch.zeros((n_sites, n_species), device=device, dtype=y_dtype)
    detected.index_add_(0, indices_visit_site, y)
    detected = (detected > 0).to(y_dtype) # shape: (n_sites, n_species)

    # NB: the following tricks are used for numerical stability (if using sigmoid loss function and logits):
    #  log(prob) = log(1 / (1 + exp(-logit))) = LogSigmoid(logit)
    #  log(1 - prob) = log(1 - 1 / (1 + exp(-logit))) = log( 1 / (1 + exp(logit)) ) = LogSigmoid(-logit)

    # Term for sites with detections: sum_s [ - log(psi_s) - sum_v ( y_v log(d_v) + (1 - y_v) log(1 - d_v) ) ] = sum_s( - log(psi_s) + sum_v BCE(d_v, y_v) )
    nll_det = -F.logsigmoid(logit_psi) # = -log(psi); shape: (n_sites, n_species)
    bce = F.binary_cross_entropy_with_logits(logit_d, y, reduction="none") # shape: (n_visits, n_species)
    nll_det.index_add_(0, indices_visit_site, bce) # Accumulate BCE terms of the species for each site; shape: (n_sites, n_species)

    # Term for sites with no detections: sum_s [ - log(1 - psi_s + psi_s * prod_v (1 - d_v) ) ]
    log_prod_1m_d = torch.zeros((n_sites, n_species), device=device, dtype=logit_d.dtype) # shape: (n_sites, n_species)
    log_prod_1m_d.index_add_(0, indices_visit_site, F.logsigmoid(-logit_d)) # Accumulate log(1 - d_v) terms of the species for each site; shape: (n_sites, n_species)

    log1mpsi = F.logsigmoid(-logit_psi) # = log(1 - psi); shape: (n_sites, n_species)
    logpsi = F.logsigmoid(logit_psi) # = log(psi); shape: (n_sites, n_species)
    nll_no_det = -torch.logaddexp(log1mpsi, logpsi + log_prod_1m_d) # = -log(1 - psi + psi * prod_v (1 - d_v)); shape: (n_sites, n_species)

    # Combine terms based on whether species was detected or not at each site
    nll_terms = detected * nll_det + (1.0 - detected) * nll_no_det  # (n_sites, n_species)

    # Reduce over species (dim=1)
    if species_reduction == "none":
        out = nll_terms # shape: (n_sites, n_species)
    elif species_reduction == "mean":
        out = nll_terms.mean(dim=1)  # shape: (n_sites,)
    elif species_reduction == "sum":
        out = nll_terms.sum(dim=1)   # shape: (n_sites,)
    else:
        raise ValueError("species_reduction must be 'none', 'mean', or 'sum'.")

    # Reduce over sites
    if site_reduction == "none":
        return out
    if site_reduction == "sum":
        return out.sum(dim=0)
    if site_reduction == "mean":
        return out.mean(dim=0)
    raise ValueError("site_reduction must be 'none', 'mean', or 'sum'.")

def nll_logits_sparse(
    y_visits: torch.Tensor,
    y_labels: torch.Tensor,
    *,
    logit_psi: torch.Tensor,
    logit_d: torch.Tensor,
    indices_visit_site: torch.Tensor,
    site_reduction: SiteReduction = "sum",
    species_reduction: SpeciesReduction = "mean",
    validate_inputs: bool = False,
) -> torch.Tensor:
    """
    Negative log-likelihood in logit space for single-season multi-species occupancy, using sparse visit-level detections.

    Parameters
    ----------
    y_visits : torch.Tensor
        Visit indices of detections, shape ``(n_detections,)``.
    y_labels : torch.Tensor
        Species labels of detections, shape ``(n_detections,)``.
    logit_psi : torch.Tensor
        Occupancy logits, shape ``(n_sites, n_species)``.
    logit_d : torch.Tensor
        Detection logits, shape ``(n_visits, n_species)``.
    indices_visit_site : torch.Tensor
        Site index per visit.
    site_reduction : {"sum", "mean", "none"}, default="sum"
        Site-level reduction.
    species_reduction : {"mean", "sum", "none"}, default="mean"
        Species-level reduction.
    validate_inputs : bool, default=False
        Whether to validate input shapes, dtypes, and index consistency. 
        Can be set to False for performance if inputs are guaranteed to be correct 
        (e.g., when called from a training loop).

    Returns
    -------
    torch.Tensor
        NLL tensor with shape determined by reductions.
    """
    y_visits = torch.as_tensor(y_visits) # shape: (n_detections,)
    y_labels = torch.as_tensor(y_labels) # shape: (n_detections,)
    logit_psi = torch.as_tensor(logit_psi) # shape: (n_sites, n_species)
    logit_d = torch.as_tensor(logit_d) # shape: (n_visits, n_species)
    indices_visit_site = torch.as_tensor(indices_visit_site) # shape: (n_visits,)

    if validate_inputs:
        tensors = (y_visits, y_labels, logit_psi, logit_d, indices_visit_site)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("All sparse likelihood tensors must be on the same device.")

    if validate_inputs:
        n_sites, _, n_species = _validate_inputs(
            y_visits=y_visits,
            y_labels=y_labels,
            logit_psi=logit_psi, 
            logit_d=logit_d, 
            indices_visit_site=indices_visit_site
        )
    else:
        n_sites, n_species = logit_psi.shape

    device = logit_psi.device
    
    # Binary indicator of whether each species was detected at each site
    detected = torch.zeros((n_sites, n_species), device=device, dtype=torch.bool)
    detected[indices_visit_site[y_visits], y_labels] = True # shape: (n_sites, n_species)

    # === Term for sites s with detections ===

    # - log(psi_s) - sum_v ( y_v log(d_v) + (1 - y_v) log(1 - d_v) ) ] = - log(psi_s) + sum_v BCE(d_v, y_v)
    nll_det = -F.logsigmoid(logit_psi) # = -log(psi); shape: (n_sites, n_species)

    # NB: BCE = -y * log(d) - (1 - y) * log(1 - d) where d = sigmoid(logit_d)
    # Negative-class contribution for every class: -log(1 - d) if y=0
    bce_term = torch.zeros_like(logit_psi) # shape: (n_sites, n_species)
    # Accumulate the negative-class contribution for every visit: -log(1 - d) if y=0
    bce_term.index_add_(0, indices_visit_site, F.softplus(logit_d)) # = log(1 + exp(logit_d)) = -log(1 - d); shape: (n_sites, n_species)
    
    # Positive-class contribution for every positive observation: -log(d) if y=1
    if y_visits.numel() > 0:
        # Starting from the negative-class contribution, we should
        # remove the previous negative contribution (i.e. add log(1 - d))
        # and add the positive contribution (i.e. subtract log(d))
        # -> log(1 - d) - log(d) = -log(d / (1 - d)) = -logits
        # The correction becomes: 
        # (for each y_visit associated with a y_label, add -logit_d[y_visit, y_label] to the corresponding site of y_visit and the species)
        correction_indices = (
            indices_visit_site[y_visits] * n_species + y_labels
        )
        bce_term.view(-1).index_add_(
            0,
            correction_indices,
            -logit_d[y_visits, y_labels],
        ) # shape: (n_sites, n_species)
    
    nll_det += bce_term # shape: (n_sites, n_species)

    # === Term for sites s with no detections ===

    # - log(1 - psi_s + psi_s * prod_v (1 - d_v) )
    log_prod_1m_d = torch.zeros_like(logit_psi) # shape: (n_sites, n_species)
    log_prod_1m_d.index_add_(0, indices_visit_site, F.logsigmoid(-logit_d)) # Accumulate log(1 - d_v) terms of the species for each site; shape: (n_sites, n_species)

    log1mpsi = F.logsigmoid(-logit_psi) # = log(1 - psi); shape: (n_sites, n_species)
    logpsi = F.logsigmoid(logit_psi) # = log(psi); shape: (n_sites, n_species)
    nll_no_det = -torch.logaddexp(log1mpsi, logpsi + log_prod_1m_d) # = -log(1 - psi + psi * prod_v (1 - d_v)); shape: (n_sites, n_species)

    nll_terms = torch.where(detected, nll_det, nll_no_det)  # (n_sites, n_species)

    # Reduce over species (dim=1)
    if species_reduction == "none":
        out = nll_terms # shape: (n_sites, n_species)
    elif species_reduction == "mean":
        out = nll_terms.mean(dim=1)  # shape: (n_sites,)
    elif species_reduction == "sum":
        out = nll_terms.sum(dim=1)   # shape: (n_sites,)
    else:
        raise ValueError("species_reduction must be 'none', 'mean', or 'sum'.")

    # Reduce over sites
    if site_reduction == "none":
        return out
    if site_reduction == "sum":
        return out.sum(dim=0)
    if site_reduction == "mean":
        return out.mean(dim=0)
    raise ValueError("site_reduction must be 'none', 'mean', or 'sum'.")

def nll_probs(
    y: torch.Tensor,
    *,
    psi: torch.Tensor,
    d: torch.Tensor,
    indices_visit_site: torch.Tensor,
    site_reduction: SiteReduction = "sum",
    species_reduction: SpeciesReduction = "mean",
    validate_inputs: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Probability-space wrapper around ``nll_logits``.

    Parameters
    ----------
    y : torch.Tensor
        Visit-level detections.
    psi : torch.Tensor
        Occupancy probabilities in ``(0, 1)``.
    d : torch.Tensor
        Detection probabilities in ``(0, 1)``.
    indices_visit_site : torch.Tensor
        Site index per visit.
    site_reduction : {"sum", "mean", "none"}, default="sum"
        Site-level reduction.
    species_reduction : {"mean", "sum", "none"}, default="mean"
        Species-level reduction.
    validate_inputs : bool, default=False
        Whether to validate input shapes, dtypes, and index consistency. 
        Can be set to False for performance if inputs are guaranteed to be correct 
        (e.g., when called from a training loop).
    eps : float, default=1e-8
        Clamping value for numeric stability.

    Returns
    -------
    torch.Tensor
        NLL tensor with shape determined by reductions.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    psi = torch.as_tensor(psi).clamp(min=eps, max=1 - eps)
    d = torch.as_tensor(d).clamp(min=eps, max=1 - eps)

    # logit = log(prob / (1 - prob)) = log(prob) - log(1 - prob)
    logit_psi = torch.log(psi) - torch.log1p(-psi)
    logit_d = torch.log(d) - torch.log1p(-d)

    return nll_logits(
        y,
        logit_psi=logit_psi,
        logit_d=logit_d,
        indices_visit_site=indices_visit_site,
        site_reduction=site_reduction,
        species_reduction=species_reduction,
        validate_inputs=validate_inputs
    )
