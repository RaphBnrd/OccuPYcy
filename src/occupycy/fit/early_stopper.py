class EarlyStopper:
    """
    Utility implementing patience-based early stopping.

    Parameters
    ----------
    patience : int, default=5
        Number of consecutive non-improving evaluations allowed.
    min_delta : float, default=0.0
        Minimum required improvement.
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0
    ):
        """Initialize early stopping state."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(
        self,
        validation_loss: float
    ):
        """
        Update state and decide whether to stop.

        Parameters
        ----------
        validation_loss : float
            Current monitored loss.

        Returns
        -------
        bool
            ``True`` if training should stop, else ``False``.
        """
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False