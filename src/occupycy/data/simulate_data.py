import torch
import matplotlib.pyplot as plt
import polars as pl

def filter_subset_visits(
        indices_visit_site, n_sites, n_total_visits_max, 
        generator, keep_all_sites=False, verbose=True
    ):
    """
    Subsample visit-to-site indices with optional one-visit-per-site guarantee.

    Returns
    -------
    torch.Tensor
        Subsampled ``indices_visit_site``.
    """
    if verbose:
        print(f"Subsampling from {indices_visit_site.shape[0]} visits to {n_total_visits_max} visits.")

    if keep_all_sites:
        if verbose:
            print("Keeping all sites, even those with few visits.")
        assert n_total_visits_max >= n_sites, "If keep_all_sites is True, n_total_visits_max must be at least n_sites."
        assert (indices_visit_site[1:] >= indices_visit_site[:-1]).all(), "If keep_all_sites is True, indices_visit_site must be sorted by site."
        # e.g. indices_visit_site might look like: [0, 0, 0, 1, 2, 2, 2, 2, 3, ...] meaning site 0 has 3 visits, site 1 has 1 visits, site 2 has 4 visits, site 3 has 1 visit, etc.
        mandatory_indices = torch.cat([
            torch.tensor([0], device=indices_visit_site.device),
            (indices_visit_site[1:] != indices_visit_site[:-1]).nonzero(as_tuple=True)[0] + 1
        ])
        mandatory_visits = indices_visit_site[mandatory_indices]
        if verbose:
            print(f"Number of mandatory visits (one per site): {mandatory_visits.shape[0]}")
        remaining_mask = torch.ones_like(indices_visit_site, dtype=torch.bool)
        remaining_mask[mandatory_indices] = False
        remaining_visits = indices_visit_site[remaining_mask]
        n_remaining = n_total_visits_max - mandatory_visits.shape[0]
        if verbose:
            print(f"Number of remaining visits to sample: {n_remaining}")
        if n_remaining > 0 and remaining_visits.numel() > 0:
            n_sample = min(n_remaining, remaining_visits.shape[0])
            perm = torch.randperm(remaining_visits.shape[0], generator=generator)
            sampled_remaining = remaining_visits[perm[:n_sample]]
            if verbose:
                print(f"Number of sampled remaining visits: {sampled_remaining.shape[0]}")
            indices_visit_site = torch.cat([mandatory_visits, sampled_remaining], dim=0)
        else:
            indices_visit_site = mandatory_visits
    
    else:
        if verbose:
            print("Without necessarily keeping all sites.")
        perm = torch.randperm(indices_visit_site.shape[0], generator=generator)
        indices_visit_site = indices_visit_site[perm[:n_total_visits_max]] # shape: (n_visits_sub,)
    
    return indices_visit_site


def simulate_constant(
        n_sites=400, 
        n_visits_per_site=25, 
        psi=torch.tensor([0.7, 0.5, 0.1, 0.05]),
        d=torch.tensor([0.01, 0.1, 0.9, 0.99]),
        seed=None,
        verbose=True
    ):
    """
    Simulate intercept-only occupancy/detection data.

    Returns
    -------
    tuple
        ``(y, visit_site_ids, psi, d, psi_obs, d_obs)``.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    n_species = psi.shape[0]
    z = torch.bernoulli(psi.expand(n_sites, n_species), generator=g)
    indices_visit_site = torch.arange(n_sites).repeat_interleave(n_visits_per_site)
    z_for_visit = z[indices_visit_site]
    y = torch.bernoulli(d.expand(indices_visit_site.shape[0], n_species), generator=g) * z_for_visit

    # Plot summary of simulated data
    if verbose:
        detected = torch.zeros((n_sites, y.shape[1]), device=y.device, dtype=y.dtype)
        detected.index_add_(0, indices_visit_site, y)
        detected = (detected > 0) # shape: (n_sites, n_species)
        print("--- Simulated data summary ---")
        print(f"  n_sites={n_sites}, n_visits_per_site={n_visits_per_site}, n_species={n_species}")
        print(f"  number of detections (per species): {y.sum(dim=0)}")
        print(f"  number of occupied sites (per species): {detected.sum(dim=0)}")
        print(f"  range of psi * d: {(psi * d).min():3e} to { (psi * d).max():3e}")
        print(f"  observed average psi (knowing true occupancy): {z.mean(dim=0)}")
        print(f"  observed average d (knowing true occupancy):   {y.sum(dim=0) / z_for_visit.sum(dim=0)}")

    return y.float(), indices_visit_site.long().tolist(), psi, d, z.mean(dim=0), y.sum(dim=0) / z_for_visit.sum(dim=0)


def simulate_linear(
        n_sites=400, n_total_visits_max=None,
        f_effort=lambda x1: torch.exp(1.5*x1), 
        keep_all_sites=False,
        intercept_psi=torch.tensor([0.0]), # shape: (n_species,)
        coefs_psi=torch.tensor([[0.0]]), # shape: (n_species, n_cov_psi)
        intercept_d=torch.tensor([0.0]), # shape: (n_species,)
        coefs_d=torch.tensor([[0.0]]), # shape: (n_species, n_cov_d)
        seed=None, 
        verbose=True, 
        plot_summary=True
    ):
    """
    Simulate occupancy data from linear logits for psi and d.

    Returns
    -------
    tuple
        ``(y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, z)``.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    X_psi = torch.randn(n_sites, len(coefs_psi[0]), generator=g)  # shape: (n_sites, n_cov_psi)
    
    # Define the number of visits per site
    n_visits_per_site = f_effort(X_psi[:, 0]).round().long()  # shape: (n_sites,)
    n_visits_per_site = n_visits_per_site.clamp(min=int(keep_all_sites))  # shape: (n_sites,) (if keep_all_sites is True, ensure at least 1 visit per site)
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # axes[0].plot(X_psi[:, 0].cpu(), n_visits_per_site.cpu(), 'o', alpha=0.5)
    # axes[0].set_xlabel("Covariate for occupancy (X_psi[:, 0])")
    # axes[0].set_ylabel("Number of visits per site")
    # axes[1].hist(X_psi[:, 0].cpu(), bins=30, edgecolor='k')
    # axes[1].set_xlabel("Covariate for occupancy (X_psi[:, 0])")
    # axes[1].set_ylabel("Frequency")
    indices_visit_site = torch.arange(n_sites).repeat_interleave(n_visits_per_site)  # shape: (n_visits,)
    # Filter out sites with zero visits
    if n_total_visits_max is not None: # Randomly subsample visits to have at most n_total_visits_max
        if indices_visit_site.shape[0] > n_total_visits_max:
            indices_visit_site = filter_subset_visits(
                indices_visit_site, n_sites, n_total_visits_max, 
                generator=g, keep_all_sites=keep_all_sites,
                verbose=verbose
            )
    n_visits_per_site = torch.bincount(indices_visit_site, minlength=n_sites)  # shape: (n_sites,) (for each site, how many visits it has)
    # Reassign site indices to be consecutive after filtering
    map_sites_to_sub = {}
    curr_idx = 0
    for i in range(n_sites):
        if n_visits_per_site[i] > 0:
            map_sites_to_sub[i] = curr_idx
            curr_idx += 1

    n_sites_sub = curr_idx
    mapping = torch.full((n_sites,), -1, dtype=torch.long, device=indices_visit_site.device)
    keys = torch.tensor(list(map_sites_to_sub.keys()), device=mapping.device)
    values = torch.tensor(list(map_sites_to_sub.values()), device=mapping.device)
    mapping[keys] = values
    indices_visit_site = mapping[indices_visit_site]
    n_visits_sub = indices_visit_site.shape[0]
    X_psi = X_psi[keys]

    X_d = torch.randn(n_visits_sub, len(coefs_d[0]), generator=g)  # shape: (n_visits_sub, n_cov_d)

    psi = torch.sigmoid(intercept_psi + X_psi @ coefs_psi.T) # shape: (n_sites_sub, n_species)
    z = torch.bernoulli(psi, generator=g)  # shape: (n_sites_sub, n_species)
    z_for_visit = z[indices_visit_site]  # shape: (n_visits_sub, n_species)
    
    d = torch.sigmoid(intercept_d + X_d @ coefs_d.T) # shape: (n_visits_sub, n_species)
    
    y = torch.bernoulli(d * z_for_visit, generator=g)  # shape: (n_visits_sub, n_species)

    # Plot summary of simulated data
    if verbose:
        detected = torch.zeros((n_sites, y.shape[1]), device=y.device, dtype=y.dtype)
        detected.index_add_(0, indices_visit_site, y)
        detected = (detected > 0) # shape: (n_sites, n_species)
        print("--- Simulated data summary ---")
        print(f"  n_sites={n_sites_sub}")
        print(f"  n_visits={n_visits_sub}")
        print(f"  number of detections (per species): {y.sum(dim=0)}")
        print(f"  number of occupied sites (per species): {detected.sum(dim=0)}")
        print(f"  range of psi * d: {(psi[indices_visit_site] * d).min():3e} to { (psi[indices_visit_site] * d).max():3e}")
        print(f"  observed average psi (knowing true occupancy): {z.mean(dim=0)}")
        print(f"  observed average d (knowing true occupancy):   {y.sum(dim=0) / z_for_visit.sum(dim=0)}")

    if plot_summary:
        fig, axes = plt.subplots(3, intercept_psi.shape[0], figsize=(8, 6))
        gs = fig.add_gridspec(3, intercept_psi.shape[0])
        # Histogram of n_visits_per_site
        ax0 = fig.add_subplot(gs[0, :])
        # logbins = np.logspace(0, np.log10(n_visits_per_site.max().item()), 30)
        # ax0.hist(n_visits_per_site.cpu(), bins=logbins, edgecolor='k')
        # ax0.set_xscale("log")
        ax0.hist(n_visits_per_site.cpu(), bins=30, edgecolor='k')
        ax0.set_xlabel("Number of visits per site")
        ax0.set_ylabel("Frequency")
        ax0.set_title("Distribution of visits per site")
        ax0.grid(True)
        ax0.set_yscale("log")
        for i in range(intercept_psi.shape[0]):
            axes[0, i].axis('off')
        # Histogram of psi
        for species_idx in range(intercept_psi.shape[0]):
            axes[1, species_idx].hist(psi[:, species_idx].cpu(), bins=30, edgecolor='k')
            axes[1, species_idx].set_xlabel("Occupancy probability (psi)")
            axes[1, species_idx].set_ylabel("Frequency")
            axes[1, species_idx].grid(True)
            axes[1, species_idx].set_title(f"psi (species {species_idx})")
        # Histogram of d
        for species_idx in range(intercept_d.shape[0]):
            axes[2, species_idx].hist(d[:, species_idx].cpu(), bins=30, edgecolor='k')
            axes[2, species_idx].set_xlabel("Detection probability (d)")
            axes[2, species_idx].set_ylabel("Frequency")
            axes[2, species_idx].grid(True)
            axes[2, species_idx].set_title(f"d (species {species_idx})")
        fig.tight_layout()

    visit_site_ids = indices_visit_site.long().tolist()
    # unique site ids, preserving the order of first occurence in visit_site_ids
    site_ids_in_X_psi = list(dict.fromkeys(visit_site_ids))

    # Filter X_psi on sites present in visit_site_ids
    X_psi = X_psi[site_ids_in_X_psi]

    return y.float(), visit_site_ids, site_ids_in_X_psi, X_psi.float(), X_d.float(), z.float()


def simulate_non_linear(
        n_sites=400, 
        n_total_visits_max=None,
        n_cov_psi=2,
        n_cov_d=1,
        f_effort=lambda x1: torch.exp(1.5*x1), 
        keep_all_sites=False,
        f_psi=lambda X_psi: torch.sigmoid(
            torch.tensor([-1.0, 0.5]) + X_psi @ torch.tensor([[2.0, -1.0], [-1.0, 1.0]]).T
        ), # from X_psi of shape (n_sites, n_cov_psi) to psi of shape (n_sites, n_species)
        f_d=lambda X_d: torch.sigmoid(
            torch.tensor([-0.5, 1.0]) + X_d @ torch.tensor([[1.0], [-2.0]]).T
        ), # from X_d of shape (n_visits, n_cov_d) to d of shape (n_visits, n_species)
        seed=None,
        plot_summary=True,
        verbose=True
    ):
    """
    Simulate occupancy data using user-defined nonlinear ``f_psi`` and ``f_d``.

    Returns
    -------
    tuple
        ``(y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, z)``.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    X_psi = torch.randn(n_sites, n_cov_psi, generator=g)  # shape: (n_sites, n_cov_psi)
    
    # Define the number of visits per site
    n_visits_per_site = f_effort(X_psi[:, 0]).round().long()  # shape: (n_sites,)
    n_visits_per_site = n_visits_per_site.clamp(min=int(keep_all_sites))  # shape: (n_sites,) (if keep_all_sites is True, ensure at least 1 visit per site)
    indices_visit_site = torch.arange(n_sites).repeat_interleave(n_visits_per_site)  # shape: (n_visits,)
    # Filter out sites with zero visits
    if n_total_visits_max is not None: # Randomly subsample visits to have at most n_total_visits_max
        if indices_visit_site.shape[0] > n_total_visits_max:
            indices_visit_site = filter_subset_visits(
                indices_visit_site, n_sites, n_total_visits_max, 
                generator=g, keep_all_sites=keep_all_sites,
                verbose=verbose
            )
    n_visits_per_site = torch.bincount(indices_visit_site, minlength=n_sites)  # shape: (n_sites,) (for each site, how many visits it has)
    # Reassign site indices to be consecutive after filtering
    map_sites_to_sub = {}
    curr_idx = 0
    for i in range(n_sites):
        if n_visits_per_site[i] > 0:
            map_sites_to_sub[i] = curr_idx
            curr_idx += 1
    n_sites_sub = curr_idx
    mapping = torch.full((n_sites,), -1, dtype=torch.long, device=indices_visit_site.device)
    keys = torch.tensor(list(map_sites_to_sub.keys()), device=mapping.device)
    values = torch.tensor(list(map_sites_to_sub.values()), device=mapping.device)
    mapping[keys] = values
    indices_visit_site = mapping[indices_visit_site]
    n_visits_sub = indices_visit_site.shape[0]
    X_psi = X_psi[keys]

    X_d = torch.randn(n_visits_sub, n_cov_d, generator=g)  # shape: (n_visits_sub, n_cov_d)

    psi = f_psi(X_psi)  # shape: (n_sites_sub, n_species)
    z = torch.bernoulli(psi, generator=g)  # shape: (n_sites_sub, n_species)
    z_for_visit = z[indices_visit_site]  # shape: (n_visits_sub, n_species)

    d = f_d(X_d)  # shape: (n_visits_sub, n_species)

    y = torch.bernoulli(d * z_for_visit, generator=g)  # shape: (n_visits_sub, n_species)

    # Plot summary of simulated data
    if verbose:
        detected = torch.zeros((n_sites, y.shape[1]), device=y.device, dtype=y.dtype)
        detected.index_add_(0, indices_visit_site, y)
        detected = (detected > 0) # shape: (n_sites, n_species)
        print("--- Simulated data summary ---")
        print(f"  n_sites={n_sites_sub}")
        print(f"  n_visits={n_visits_sub}")
        print(f"  number of detections (per species): {y.sum(dim=0)}")
        print(f"  number of occupied sites (per species): {detected.sum(dim=0)}")
        print(f"  range of psi * d: {(psi[indices_visit_site] * d).min():3e} to { (psi[indices_visit_site] * d).max():3e}")
        print(f"  observed average psi (knowing true occupancy): {z.mean(dim=0)}")
        print(f"  observed average d (knowing true occupancy):   {y.sum(dim=0) / z_for_visit.sum(dim=0)}")

    if plot_summary:
        fig, axes = plt.subplots(psi.shape[1]+1, 3, figsize=(10, (psi.shape[1]+1)*2))
        gs = fig.add_gridspec(psi.shape[1]+1, 3)
        # Histogram of n_visits_per_site
        ax0 = fig.add_subplot(gs[0, :])
        logbins = torch.logspace(torch.log10(n_visits_per_site[n_visits_per_site>0].min()), torch.log10(n_visits_per_site.max()), 30)
        ax0.hist(n_visits_per_site.cpu(), bins=logbins, edgecolor='k')
        ax0.set_xscale("log")
        # ax0.hist(n_visits_per_site.cpu(), bins=50, edgecolor='k')
        ax0.set_xlabel("Number of visits per site")
        ax0.set_ylabel("Frequency")
        ax0.set_title(f"Distribution of visits per site ({(n_visits_per_site>0).sum()}/{n_sites} sites with >0 visits)")
        ax0.grid(True)
        ax0.set_yscale("log")
        for i in range(3):
            axes[0, i].axis('off')
        # Histogram of psi
        for species_idx in range(psi.shape[1]):
            tmp = psi[:, species_idx]
            logbins = torch.logspace(torch.log10(tmp.min()), torch.log10(tmp.max()), 30)
            axes[species_idx+1, 0].hist(tmp.cpu(), bins=logbins, edgecolor='k')
            axes[species_idx+1, 0].set_xlabel("Occupancy probability (psi)")
            axes[species_idx+1, 0].set_ylabel("Frequency")
            axes[species_idx+1, 0].grid(True)
            axes[species_idx+1, 0].set_title(f"psi (species {species_idx})")
            axes[species_idx+1, 0].set_xscale("log")
        # Histogram of d
        for species_idx in range(d.shape[1]):
            tmp = d[:, species_idx]
            logbins = torch.logspace(torch.log10(tmp.min()), torch.log10(tmp.max()), 30)
            axes[species_idx+1, 1].hist(tmp.cpu(), bins=logbins, edgecolor='k')
            axes[species_idx+1, 1].set_xlabel("Detection probability (d)")
            axes[species_idx+1, 1].set_ylabel("Frequency")
            axes[species_idx+1, 1].grid(True)
            axes[species_idx+1, 1].set_title(f"d (species {species_idx})")
            axes[species_idx+1, 1].set_xscale("log")
        # Histogram of psi*d
        for species_idx in range(psi.shape[1]):
            tmp = psi[indices_visit_site, species_idx] * d[:, species_idx]
            logbins = torch.logspace(torch.log10(tmp.min()), torch.log10(tmp.max()), 30)
            axes[species_idx+1, 2].hist(tmp.cpu(), bins=logbins, edgecolor='k')
            axes[species_idx+1, 2].set_xlabel("Effective detection probability (psi * d)")
            axes[species_idx+1, 2].set_ylabel("Frequency")
            axes[species_idx+1, 2].grid(True)
            axes[species_idx+1, 2].set_title(f"psi*d (species {species_idx})")
            axes[species_idx+1, 2].set_xscale("log")
        fig.tight_layout()

    visit_site_ids = indices_visit_site.long().tolist()
    # unique site ids, preserving the order of first occurence in visit_site_ids
    site_ids_in_X_psi = list(dict.fromkeys(visit_site_ids))

    # Filter X_psi on sites present in visit_site_ids
    X_psi = X_psi[site_ids_in_X_psi]

    return y.float(), visit_site_ids, site_ids_in_X_psi, X_psi.float(), X_d.float(), z.float()

def _from_res_to_dataframe(y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, z):
    """
    Convert simulation outputs to visit-level and site-level Polars dataframes.

    Returns
    -------
    tuple[polars.DataFrame, polars.DataFrame]
        ``(df_visit, df_site)``.
    """

    # * * * Visit dataframe * * *

    df_visit = pl.DataFrame()
    # Add site id column
    fmt = f"site_{{:0{len(str(max(visit_site_ids)))}d}}"
    visit_site_ids_str = [fmt.format(i) for i in visit_site_ids]
    df_visit = df_visit.with_columns(
        pl.Series("visit_site_id", visit_site_ids_str)
    )
    # Add species presence/absence columns
    for species_idx in range(y.shape[1]):
        df_visit = df_visit.with_columns(
            pl.Series(f"y_sp{species_idx}", y[:, species_idx])
        )
    # Add covariates for detection columns
    for cov_idx in range(X_d.shape[1]):
        df_visit = df_visit.with_columns(
            pl.Series(f"X_d_c{cov_idx}", X_d[:, cov_idx])
        )

    # * * * Site dataframe * * *

    df_site = pl.DataFrame()
    # Add site id column
    site_ids_in_X_psi_str = [fmt.format(i) for i in site_ids_in_X_psi]
    df_site = df_site.with_columns(
        pl.Series("site_id", site_ids_in_X_psi_str)
    )
    # Add species presence/absence columns (knowing true occupancy)
    for species_idx in range(z.shape[1]):
        df_site = df_site.with_columns(
            pl.Series(f"true_sp{species_idx}", z[:, species_idx])
        )
    # Add covariates for occupancy columns
    for cov_idx in range(X_psi.shape[1]):
        df_site = df_site.with_columns(
            pl.Series(f"X_psi_c{cov_idx}", X_psi[:, cov_idx])
        )

    return df_visit, df_site

def simulate_linear_dataframe(
        n_sites=400, n_total_visits_max=None,
        keep_all_sites=False,
        f_effort=lambda x1: torch.exp(1.5*x1), 
        intercept_psi=torch.tensor([0.0]), # shape: (n_species,)
        coefs_psi=torch.tensor([[0.0]]), # shape: (n_species, n_cov_psi)
        intercept_d=torch.tensor([0.0]), # shape: (n_species,)
        coefs_d=torch.tensor([[0.0]]), # shape: (n_species, n_cov_d)
        seed=None,
        verbose=True,
        plot_summary=True
    ):
    """
    Simulate linear occupancy data and return dataframes.

    Returns
    -------
    tuple[polars.DataFrame, polars.DataFrame]
        ``(df_visit, df_site)``.
    """
    
    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, true_z = simulate_linear(
        n_sites=n_sites, 
        n_total_visits_max=n_total_visits_max,
        keep_all_sites=keep_all_sites,
        f_effort=f_effort,
        intercept_psi=intercept_psi,
        coefs_psi=coefs_psi,
        intercept_d=intercept_d, 
        coefs_d=coefs_d, 
        seed=seed,
        verbose=verbose,
        plot_summary=plot_summary
    )

    df_visit, df_site = _from_res_to_dataframe(y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, true_z)

    return df_visit, df_site

def simulate_non_linear_dataframe(
        n_sites=400, 
        n_total_visits_max=None,
        n_cov_psi=2,
        n_cov_d=1,
        keep_all_sites=False,
        f_effort=lambda x1: torch.exp(1.5*x1), 
        f_psi=lambda X_psi: torch.sigmoid(
            torch.tensor([-1.0, 0.5]) + X_psi @ torch.tensor([[2.0, -1.0], [-1.0, 1.0]]).T
        ), # from X_psi of shape (n_sites, n_cov_psi) to psi of shape (n_sites, n_species)
        f_d=lambda X_d: torch.sigmoid(
            torch.tensor([-0.5, 1.0]) + X_d @ torch.tensor([[1.0], [-2.0]]).T
        ), # from X_d of shape (n_visits, n_cov_d) to d of shape (n_visits, n_species)
        seed=None,
        plot_summary=True,
        verbose=True
    ):
    """
    Simulate nonlinear occupancy data and return dataframes.

    Returns
    -------
    tuple[polars.DataFrame, polars.DataFrame]
        ``(df_visit, df_site)``.
    """

    y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, true_z = simulate_non_linear(
        n_sites=n_sites, 
        n_total_visits_max=n_total_visits_max,
        n_cov_psi=n_cov_psi,
        n_cov_d=n_cov_d,
        keep_all_sites=keep_all_sites,
        f_effort=f_effort,
        f_psi=f_psi,
        f_d=f_d,
        seed=seed,
        plot_summary=plot_summary,
        verbose=verbose
    )

    df_visit, df_site = _from_res_to_dataframe(y, visit_site_ids, site_ids_in_X_psi, X_psi, X_d, true_z)

    return df_visit, df_site
