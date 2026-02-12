"""Early stopping callback for training.

Monitors validation metrics and stops training when no improvement.
"""

from __future__ import annotations

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting.
    
    Monitors a validation metric and stops training when it stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/AP
        restore_best_weights: Whether to restore best model weights on stop
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0001,
        mode: str = 'max',
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value: float | None = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.should_stop: bool = False
        
        # For restoring best weights
        self.best_state_dict: dict | None = None
        
    def step(self, current_value: float, epoch: int, model_state_dict: dict | None = None) -> bool:
        """Check if should stop training.
        
        Args:
            current_value: Current validation metric value
            epoch: Current epoch number
            model_state_dict: Model state dict to save if best
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            # First epoch
            self.best_value = current_value
            self.best_epoch = epoch
            if model_state_dict is not None:
                self.best_state_dict = model_state_dict.copy()
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            # Improvement found
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            if model_state_dict is not None:
                self.best_state_dict = model_state_dict.copy()
            LOGGER.info(f\"EarlyStopping: New best {current_value:.4f} at epoch {epoch}")
        else:
            # No improvement
            self.wait += 1
            LOGGER.info(f\"EarlyStopping: No improvement for {self.wait}/{self.patience} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                LOGGER.info(f\"EarlyStopping: Stopping at epoch {epoch}. Best was {self.best_value:.4f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def get_best_state(self) -> dict | None:
        """Get best model state dict."""
        return self.best_state_dict


__all__ = ['EarlyStopping']
