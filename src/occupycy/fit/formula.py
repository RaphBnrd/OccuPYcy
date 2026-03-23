import torch

def build_design_matrix_from_formula(X_ini, formula, covars, verbose=False):

    X_new = torch.empty(X_ini.shape[0], 0, dtype=X_ini.dtype, device=X_ini.device) # Start with an empty design matrix
    # Split the formula by ',' and strip whitespace to get the new column names
    colnames = [col.strip() for col in formula.split(",")]

    # Loop over the new columns specified in the formula
    for i, new_col in enumerate(colnames):
        # e.g. new_col = "X_psi_c0 ^ 3 * X_psi_c1 ^ 2 * X_psi_c0"

        # Each new column can be a product of multiple terms
        terms = new_col.split("*")
        if verbose:
            print(f".{new_col:>40} -> ", end=' ')
            first_term = True
        
        # Start with a column of ones for multiplication
        new_col_values = torch.ones(X_ini.shape[0], dtype=X_ini.dtype, device=X_ini.device)
        
        # Loop over the individual terms in the new column
        for term in terms:
            # e.g. term = " X_psi_c1 ^ 2 "
            term = term.strip()

            # Handle power if present
            if "^" in term:
                base, power = term.split("^")
                base = base.strip()
                power = float(power.strip())
            else:
                base = term
                power = None

            # Handle absolute value
            is_abs = False
            if base.startswith("|") and base.endswith("|"):
                is_abs = True
                base = base[1:-1].strip()

            # Check validity
            if base not in covars:
                raise ValueError(f"Base '{base}' not found in original covariates {covars}")
            if power is not None and power % 1 != 0 and not is_abs:
                raise ValueError("Fractional powers require absolute value: use |x|^p")

            values = X_ini[:, covars.index(base)]

            # Apply absolute value if needed
            if is_abs:
                values = torch.abs(values)

            # Apply power if needed
            if power is not None:
                values = values ** power

            # Multiply into the column
            new_col_values *= values

            if verbose:
                part = f"|{base}|" if is_abs else base
                if power is not None:
                    part = f"({part}^{power})"
                print(f"{'' if first_term else ' * '}{part}", end='')
                first_term = False
        
        if verbose:
            print()
        
        # Append the new column to the design matrix
        X_new = torch.cat((X_new, new_col_values.unsqueeze(1)), dim=1)

    if verbose:
        print("New design matrix created with columns:", colnames)
        print("Shape of the new design matrix:", X_new.shape)
    
    return X_new, colnames