from __future__ import annotations

from typing import Optional

import torch
from occupycy.models import OccuModel


def loss_penalty(
        penalization_type: Optional[str],
        penalization_lambda_psi: Optional[float],
        penalization_lambda_d: Optional[float],
        model: OccuModel,
        logit_psi: Optional[torch.Tensor], # shape: (n_sites, n_species)
        logit_d: Optional[torch.Tensor], # shape: (n_visits, n_species)
        center_logit_psi: Optional[torch.Tensor], # shape: (n_species,) or None if not on logits
        center_logit_d: Optional[torch.Tensor], # shape: (n_species,) or None if not on logits
        site_reduction: str,
        species_reduction: str,
    ) -> torch.Tensor:

    if penalization_type is None:
        return 0.0
    
    elif penalization_type in [
        "l2_logits", "l1_logits",
        "l2_logits_naive", "l1_logits_naive"
    ]:
                
        psi_diff = logit_psi - center_logit_psi.unsqueeze(0) # shape: (n_sites, n_species)
        d_diff   = logit_d   - center_logit_d.unsqueeze(0)   # shape: (n_visits, n_species)
        if penalization_type in ["l2_logits", "l2_logits_naive"]:
            penalty_psi, penalty_d = psi_diff ** 2, d_diff ** 2
        else: # "l1_logits" or "l1_logits_naive"
            penalty_psi, penalty_d = psi_diff.abs(), d_diff.abs()
        penalty_psi = penalty_psi.sum()
        penalty_d   = penalty_d.sum()

        if site_reduction == "mean":
            n_sites = logit_psi.shape[0]
            penalty_psi /= n_sites
            penalty_d   /= n_sites
        elif site_reduction == "sum":
            pass
        else:
            raise ValueError(f"Invalid site_reduction: {site_reduction}")
        if species_reduction == "mean":
            n_species = logit_psi.shape[1]
            penalty_psi /= n_species
            penalty_d   /= n_species
        elif species_reduction == "sum":
            pass
        else:
            raise ValueError(f"Invalid species_reduction: {species_reduction}")
        
        return (penalization_lambda_psi * penalty_psi) + \
               (penalization_lambda_d * penalty_d)
    
    elif penalization_type in ["l2_params", "l1_params"]:
        
        params_psi = torch.cat([p.view(-1) for p in model.psi_model.parameters()]) # shape: (n_params_psi,)
        params_d = torch.cat([p.view(-1) for p in model.d_model.parameters()]) # shape: (n_params_d,)
        
        if penalization_type == "l2_params":
            penalty_psi = params_psi.pow(2).sum()
            penalty_d   = params_d.pow(2).sum()
        else: # "l1_params"
            penalty_psi = params_psi.abs().sum()
            penalty_d   = params_d.abs().sum()

        return (penalization_lambda_psi * penalty_psi) + \
               (penalization_lambda_d * penalty_d)

    else:
        raise ValueError(f"Unsupported penalization type: {penalization_type}.")
