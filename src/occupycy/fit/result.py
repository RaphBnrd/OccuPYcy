from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass(frozen=True)
class FitResult:
    converged: bool
    n_epochs: Optional[int] = None

    final_train_loss: Optional[float] = None

    loss_history: Optional[dict[str, List[float]]] = None

    lr_psi: Optional[float] = None
    optimizer_psi: Optional[str] = None
    lr_d: Optional[float] = None
    optimizer_d: Optional[str] = None

    device: Optional[torch.device] = None