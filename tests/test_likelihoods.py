import pytest
import torch
import numpy as np
from occupycy.likelihood import nll_logits, nll_probs


def numpy_nll(y, psi, d, indices_visit_site):
    expected_nll = np.zeros_like(psi) # shape (n_sites, n_species)
    for k in range(psi.shape[1]): # loop over species
        for s in range(psi.shape[0]): # loop over sites
            # If species is detected at least once in the visits for this site
            if np.any(y[indices_visit_site == s, k] == 1):
                # term = -log(psi) - sum_over_visits [ y*log(d) + (1-y)*log(1-d) ]
                expected_nll[s, k] = - np.log(psi[s, k]) - np.sum(
                    y[indices_visit_site == s, k] * np.log(d[indices_visit_site == s, k]) +
                    (1 - y[indices_visit_site == s, k]) * np.log(1 - d[indices_visit_site == s, k])
                )
            else:
                # term = -log(1 - psi + psi * product_over_visits (1-d) )
                expected_nll[s, k] = - np.log(1 - psi[s, k] + psi[s, k] * np.prod(1 - d[indices_visit_site == s, k]))
    return expected_nll

def test_raises_if_indices_visit_site_out_of_range():
    indices_visit_site = torch.tensor([0, 0, 2], dtype=torch.int64)
    y = torch.tensor([[0.0], [0.0], [1.0]])
    logit_psi = torch.zeros((2, 1), requires_grad=True)
    logit_d = torch.ones((3, 1), requires_grad=True)

    with pytest.raises(ValueError, match="indices_visit_site must be in"):
        nll_logits(
            y, logit_psi=logit_psi, logit_d=logit_d, indices_visit_site=indices_visit_site,
            species_reduction="none", site_reduction="none"
        )

def test_nll_probs(tol=1e-6):

    indices_visit_site = torch.tensor([0, 0, 1], dtype=torch.int64) # 3 visits: 2 at site 0, 1 at site 1
    y = torch.tensor([[0.0, 1.0, 0.0, 1.0], # visit 1 (site 0): species 2 and 4 detected
                      [0.0, 0.0, 0.0, 1.0], # visit 2 (site 0): species 4 detected
                      [1.0, 0.0, 0.0, 0.0]]) # visit 3 (site 1): species 1 detected; shape (3 visits, 4 species)
    psi = torch.ones((2, 4), requires_grad=True) / 4 # shape (2 sites, 4 species)
    d = torch.ones((3, 4), requires_grad=True) / 4 # shape (3 visits, 4 species)
    n_sites, _, n_species = 2, 3, 4

    # No reductions
    a = nll_probs(
        y, psi=psi, d=d, indices_visit_site=indices_visit_site,
        species_reduction="none", site_reduction="none"
    )
    assert a.shape == (n_sites, n_species)
    a_np = numpy_nll(
        y.detach().numpy(), 
        psi.detach().numpy(), 
        d.detach().numpy(), 
        indices_visit_site.detach().numpy()
    ) # shape (n_sites, n_species)
    assert np.allclose(a.detach().numpy(), a_np, atol=tol)

    # Species reduction only
    b = nll_probs(
        y, psi=psi, d=d, indices_visit_site=indices_visit_site,
        species_reduction="sum", site_reduction="none"
    )
    assert b.shape == (n_sites,)
    b_np = a_np.sum(axis=1) # sum over species
    assert np.allclose(b.detach().numpy(), b_np, atol=tol)

    # Site reduction only
    c = nll_probs(
        y, psi=psi, d=d, indices_visit_site=indices_visit_site,
        species_reduction="none", site_reduction="mean"
    )
    assert c.shape == (n_species,)
    c_np = a_np.mean(axis=0) # mean over sites
    assert np.allclose(c.detach().numpy(), c_np, atol=tol)


def test_nll_logits(tol=1e-6):

    indices_visit_site = torch.tensor([0, 0, 1], dtype=torch.int64) # 3 visits: 2 at site 0, 1 at site 1
    y = torch.tensor([[0.0, 1.0, 0.0, 1.0], # visit 1 (site 0): species 2 and 4 detected
                      [0.0, 0.0, 0.0, 1.0], # visit 2 (site 0): species 4 detected
                      [1.0, 0.0, 0.0, 0.0]]) # visit 3 (site 1): species 1 detected; shape (3 visits, 4 species)
    logit_psi = torch.zeros((2, 4), requires_grad=True) # shape (2 sites, 4 species)
    logit_d = torch.zeros((3, 4), requires_grad=True) # shape (3 visits, 4 species)
    n_sites, _, n_species = 2, 3, 4

    # No reductions
    a = nll_logits(
        y, logit_psi=logit_psi, logit_d=logit_d, indices_visit_site=indices_visit_site,
        species_reduction="none", site_reduction="none"
    )
    assert a.shape == (n_sites, n_species)
    a_np = numpy_nll(
        y.detach().numpy(), 
        torch.sigmoid(logit_psi).detach().numpy(), 
        torch.sigmoid(logit_d).detach().numpy(), 
        indices_visit_site.detach().numpy()
    ) # shape (n_sites, n_species)
    assert np.allclose(a.detach().numpy(), a_np, atol=tol)

    # Species reduction only
    b = nll_logits(
        y, logit_psi=logit_psi, logit_d=logit_d, indices_visit_site=indices_visit_site,
        species_reduction="mean", site_reduction="none"
    )
    assert b.shape == (n_sites,)
    b_np = a_np.mean(axis=1) # mean over species
    assert np.allclose(b.detach().numpy(), b_np, atol=tol)

    # Site reduction only
    c = nll_logits(
        y, logit_psi=logit_psi, logit_d=logit_d, indices_visit_site=indices_visit_site,
        species_reduction="none", site_reduction="sum"
    )
    assert c.shape == (n_species,)
    c_np = a_np.sum(axis=0) # sum over sites
    assert np.allclose(c.detach().numpy(), c_np, atol=tol)
    assert np.all(c.detach().numpy() == c_np)