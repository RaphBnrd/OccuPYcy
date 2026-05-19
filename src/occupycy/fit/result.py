from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass(frozen=True)
class FitResult:
    """
    Immutable container for fitting outputs.

    Attributes
    ----------
    converged : bool
        Whether fitting is considered converged.
    n_epochs : int or None
        Number of optimization epochs/iterations.
    final_train_loss : float or None
        Final training loss.
    final_val_loss : float or None
        Final validation loss.
    idx_epoch_best_model : int or None
        Index of best epoch.
    history : dict[str, list[float]] or None
        Training history curves.
    lr_psi, lr_d : float or None
        Learning rates used.
    optimizer_psi, optimizer_d : str or None
        Optimizer names.
    device : torch.device or None
        Device used for fitting.
    """
    converged: bool
    n_epochs: Optional[int] = None

    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    idx_epoch_best_model: Optional[int] = None

    history: Optional[dict[str, List[float]]] = None

    lr_psi: Optional[float] = None
    optimizer_psi: Optional[str] = None
    lr_d: Optional[float] = None
    optimizer_d: Optional[str] = None

    device: Optional[torch.device] = None