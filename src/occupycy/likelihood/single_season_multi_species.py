from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

SiteReduction = Literal["sum", "mean", "none"]
SpeciesReduction = Literal["mean", "sum", "none"]


def _validate_inputs(
    y: torch.Tensor,
    logit_psi: torch.Tensor,
    logit_d: torch.Tensor,
    visit_site_idxs: torch.Tensor,
) -> tuple[int, int, int]:
    """
    Validate shapes/dtypes of inputs for nll functions.

    - sites x species for occupancy logits
    - visits x species for detection logits and observation y
    - visit_site_idxs maps each visit row to a site index
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (n_visits, n_species); got shape {tuple(y.shape)}")
    if logit_psi.ndim != 2:
        raise ValueError(f"logit_psi must be 2D (n_sites, n_species); got shape {tuple(logit_psi.shape)}")
    if logit_d.ndim != 2:
        raise ValueError(f"logit_d must be 2D (n_visits, n_species); got shape {tuple(logit_d.shape)}")
    if visit_site_idxs.ndim != 1:
        raise ValueError(
            f"visit_site_idxs must be 1D (n_visits,); got shape {tuple(visit_site_idxs.shape)}"
        )

    n_visits, n_species = y.shape
    n_sites, n_species_psi = logit_psi.shape
    n_visits_d, n_species_d = logit_d.shape

    if n_species_psi != n_species or n_species_d != n_species:
        raise ValueError(
            "Species dimension mismatch: "
            f"y has {n_species}, logit_psi has {n_species_psi}, logit_d has {n_species_d}."
        )
    if n_visits_d != n_visits:
        raise ValueError(f"Visit dimension mismatch: y has {n_visits}, logit_d has {n_visits_d}.")
    if visit_site_idxs.shape[0] != n_visits:
        raise ValueError(
            f"visit_site_idxs length mismatch: expected {n_visits}, got {visit_site_idxs.shape[0]}."
        )

    if visit_site_idxs.dtype not in (torch.int64, torch.int32):
        raise ValueError("visit_site_idxs must be an int tensor (int64/int32).")

    if n_sites == 0 or n_visits == 0 or n_species == 0:
        raise ValueError("Empty inputs are not supported (n_sites, n_visits, n_species must be > 0).")

    min_idx = int(visit_site_idxs.min().item())
    max_idx = int(visit_site_idxs.max().item())
    n_idx = visit_site_idxs.unique().numel()
    if min_idx != 0 or max_idx != n_sites - 1 or n_idx != n_sites:
        raise ValueError(f"visit_site_idxs must be in [0, n_sites-1] with no missing indices, but got min {min_idx}, max {max_idx}, unique count {n_idx} for X_psi rows {n_sites}.")

    return n_sites, n_visits, n_species


def nll_logits(
    y: torch.Tensor,
    *,
    logit_psi: torch.Tensor,
    logit_d: torch.Tensor,
    visit_site_idxs: torch.Tensor,
    site_reduction: SiteReduction = "sum",
    species_reduction: SpeciesReduction = "mean",
) -> torch.Tensor:
    """
    Core NLL in logits space.

    Returns depending on reductions:
      - species_reduction="none", site_reduction="none" -> (n_sites, n_species)
      - species_reduction="mean|sum", site_reduction="none" -> (n_sites,)
      - species_reduction="none", site_reduction="sum|mean" -> (n_species,)
      - species_reduction="mean|sum", site_reduction="sum|mean" -> scalar
    """
    y = torch.as_tensor(y) # shape: (n_visits, n_species)
    logit_psi = torch.as_tensor(logit_psi) # shape: (n_sites, n_species)
    logit_d = torch.as_tensor(logit_d) # shape: (n_visits, n_species)
    visit_site_idxs = torch.as_tensor(visit_site_idxs) # shape: (n_visits,)

    n_sites, _, n_species = _validate_inputs(y, logit_psi, logit_d, visit_site_idxs)

    device = y.device
    y_dtype = y.dtype
    
    # Binary indicator of whether each species was detected at each site
    detected = torch.zeros((n_sites, n_species), device=device, dtype=y_dtype)
    detected.index_add_(0, visit_site_idxs, y)
    detected = (detected > 0).to(y_dtype) # shape: (n_sites, n_species)

    # NB: the following tricks are used for numerical stability (if using sigmoid loss function and logits):
    #  log(prob) = log(1 / (1 + exp(-logit))) = LogSigmoid(logit)
    #  log(1 - prob) = log(1 - 1 / (1 + exp(-logit))) = log( 1 / (1 + exp(logit)) ) = LogSigmoid(-logit)

    # Term for sites with detections: sum_s [ - log(psi_s) - sum_v ( y_v log(d_v) + (1 - y_v) log(1 - d_v) ) ] = sum_s( - log(psi_s) + sum_v BCE(d_v, y_v) )
    nll_det = -F.logsigmoid(logit_psi) # = -log(psi); shape: (n_sites, n_species)
    bce = F.binary_cross_entropy_with_logits(logit_d, y, reduction="none") # shape: (n_visits, n_species)
    nll_det.index_add_(0, visit_site_idxs, bce) # Accumulate BCE terms of the species for each site; shape: (n_sites, n_species)

    # Term for sites with no detections: sum_s [ - log(1 - psi_s + psi_s * prod_v (1 - d_v) ) ]
    log_prod_1m_d = torch.zeros((n_sites, n_species), device=device, dtype=logit_d.dtype) # shape: (n_sites, n_species)
    log_prod_1m_d.index_add_(0, visit_site_idxs, F.logsigmoid(-logit_d)) # Accumulate log(1 - d_v) terms of the species for each site; shape: (n_sites, n_species)

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


def nll_probs(
    y: torch.Tensor,
    *,
    psi: torch.Tensor,
    d: torch.Tensor,
    visit_site_idxs: torch.Tensor,
    site_reduction: SiteReduction = "sum",
    species_reduction: SpeciesReduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Probability-space wrapper around `nll_logits`."""
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
        visit_site_idxs=visit_site_idxs,
        site_reduction=site_reduction,
        species_reduction=species_reduction,
    )
