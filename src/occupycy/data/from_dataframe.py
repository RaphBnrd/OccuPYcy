import torch
import pandas as pd
import polars as pl

from occupycy.data import OccuData

def data_from_dataframe(
        *,
        df_visit: pd.DataFrame | pl.DataFrame, 
        df_site: pd.DataFrame | pl.DataFrame | None = None,
        col_visit_site_ids: str = "visit_site_id",
        col_site_ids_in_X_psi: str | None = "site_id",
        cols_y: list[str] | None = None,
        cols_covars_d: list[str] | None = None,
        cols_covars_psi: list[str] | None = None,
        device: torch.device,
        verbose: bool = False
) -> dict:
    """
    Create the data dictionary needed to create an OccuData object from pandas or polars DataFrames containing the visit-level and site-level data.
    """
    # Convert to polars if input is pandas DataFrame
    if isinstance(df_visit, pd.DataFrame):
        df_visit = pl.from_pandas(df_visit)
    if df_site is not None and isinstance(df_site, pd.DataFrame):
        df_site = pl.from_pandas(df_site)

    # Define y
    y = torch.tensor(df_visit.select(cols_y).to_numpy(), dtype=torch.float32, device=device) # shape: (n_visits, n_species)
    
    # Define visit_site_ids
    visit_site_ids = df_visit[col_visit_site_ids].to_list() # length: n_visits

    # Define site_ids_in_X_psi (could be None) and check consistency with visit_site_ids
    if df_site is None:
        site_ids_in_X_psi = None
    elif col_site_ids_in_X_psi is None:
        if cols_covars_psi is None:
            site_ids_in_X_psi = None
        else:
            raise ValueError("If df_site and cols_covars_psi is provided, then col_site_ids_in_X_psi must also be provided to specify the column in df_site that contains the site IDs corresponding to the rows of X_psi")
    else:
        site_ids_in_X_psi = df_site[col_site_ids_in_X_psi].to_list() # length: n_sites_in_X_psi
    if site_ids_in_X_psi is not None:
        if not set(visit_site_ids).issubset(set(site_ids_in_X_psi)):
            raise ValueError("The set of site IDs in df_visit (visit_site_ids) must be a subset of the set of site IDs in df_site (site_ids_in_X_psi)")
    
    # Check if we need to downsample the site covariates to only include the sites that are present in visit_site_ids
    if site_ids_in_X_psi is not None and len(site_ids_in_X_psi) > len(set(visit_site_ids)):
        if verbose:
            print("Downsampling site covariates to only include sites that are present in visit_site_ids")
        mask = df_site[col_site_ids_in_X_psi].is_in(visit_site_ids)
        df_site = df_site.filter(mask)
        site_ids_in_X_psi = df_site[col_site_ids_in_X_psi].to_list() # Update site_ids_in_X_psi after filtering
    
    # Define X_psi
    if cols_covars_psi is None:
        X_psi = None
    else:
        X_psi = torch.tensor(df_site.select(cols_covars_psi).to_numpy(), dtype=torch.float32, device=device) # shape: (n_sites_in_X_psi, n_cov_psi)
    
    # Define X_d
    if cols_covars_d is None:
        X_d = None
    else:
        X_d = torch.tensor(df_visit.select(cols_covars_d).to_numpy(), dtype=torch.float32, device=device) # shape: (n_visits, n_cov_d)

    # Print shapes and lengths if needed
    if verbose:
        print(f"y shape (n_visits, n_species): {y.shape}")
        print(f"visit_site_ids length (n_visits): {len(visit_site_ids)}")
        print(f"site_ids_in_X_psi length (n_sites): {len(site_ids_in_X_psi)}")
        if X_psi is not None:
            print(f"X_psi shape (n_sites, n_cov_psi): {X_psi.shape}")
        else:
            print("X_psi is None")
        if X_d is not None:
            print(f"X_d shape (n_visits, n_cov_d): {X_d.shape}")
        else:
            print("X_d is None")

    return {
        "y": y,
        "visit_site_ids": visit_site_ids,
        "site_ids_in_X_psi": site_ids_in_X_psi,
        "X_psi": X_psi,
        "X_d": X_d
    }
