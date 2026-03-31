import os
import torch

def save_checkpoint_if_best(
    path_checkpoints: str,
    epoch: int,
    last_train_loss: float,
    last_val_loss: float,
    final_params_model_psi: dict,
    final_params_model_d: dict,
    opt_psi: torch.optim.Optimizer | None,
    opt_d: torch.optim.Optimizer | None,
    opt: torch.optim.Optimizer | None,
    optimizer_psi_cls: type,
    optimizer_d_cls: type,
    train_loss_history: list[float],
    val_loss_history: list[float],
    lr_psi_history: list[float],
    lr_d_history: list[float],
    verbose: bool,
):
                    
    if path_checkpoints is not None:
        
        os.makedirs(os.path.dirname(path_checkpoints), exist_ok=True)

        tmp_path = path_checkpoints.replace(".pt", ".tmp")

        checkpoint = {
            # Core training state
            "epoch": epoch,
            "train_loss": last_train_loss,
            "val_loss": last_val_loss,

            # Model weights
            "model_state_dict_psi": final_params_model_psi,
            "model_state_dict_d": final_params_model_d,

            # Optimizers (handle LBFGS separately)
            "optimizer_state_dict_psi": (
                opt_psi.state_dict() if optimizer_psi_cls != torch.optim.LBFGS else None
            ),
            "optimizer_state_dict_d": (
                opt_d.state_dict() if optimizer_d_cls != torch.optim.LBFGS else None
            ),
            "optimizer_state_dict": (
                opt.state_dict() if optimizer_psi_cls == torch.optim.LBFGS else None
            ),

            # Training history
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,

            # Learning rates
            "lr_psi": lr_psi_history[-1],
            "lr_d": lr_d_history[-1],

            # 🔹 Small but useful additions
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Safe save (avoids corrupted file if crash during write)
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path_checkpoints)

        if verbose:
            print(f"✔ New best model at epoch {epoch+1} (val_loss={last_val_loss:.4f}) saved to {path_checkpoints}")