"""Learning rate scheduling utilities for training.

Provides helper functions to create LR schedulers with warmup for optimization.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Cosine annealing with linear warmup.
    
    Learning rate schedule:
    - Warmup: Linear increase from 0 to max_lr over warmup_steps
    - Annealing: Cosine decay from max_lr to min_lr over remaining steps
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of steps for warmup phase
            total_steps: Total number of training steps
            min_lr: Minimum learning rate (default: 0)
            last_epoch: Index of last epoch for resumption
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            alpha = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def create_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine_warmup",
    total_steps: int = 10000,
    warmup_steps: int | None = None,
    warmup_ratio: float = 0.1,
    min_lr: float = 1e-6,
    **kwargs,
) -> _LRScheduler:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine_warmup', 'onecycle', 'cosine_restart')
        total_steps: Total training steps
        warmup_steps: Number of warmup steps (if None, computed from warmup_ratio)
        warmup_ratio: Ratio of total steps for warmup (default: 0.1 = 10%)
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        Learning rate scheduler
    """
    if warmup_steps is None:
        warmup_steps = int(total_steps * warmup_ratio)
    
    if scheduler_type == "cosine_warmup":
        return LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    
    elif scheduler_type == "onecycle":
        max_lr = optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
            **kwargs,
        )
    
    elif scheduler_type == "cosine_restart":
        T_0 = kwargs.get("T_0", 10)  # Restart every 10 epochs
        T_mult = kwargs.get("T_mult", 2)  # Double period after each restart
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr,
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


__all__ = [
    "LinearWarmupCosineAnnealingLR",
    "create_lr_scheduler",
]
