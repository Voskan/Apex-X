"""Checkpoint management utilities for training.

Provides functions to save, load, and manage model checkpoints during training.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


_STATE_DICT_CANDIDATE_KEYS: tuple[str, ...] = (
    "model_state_dict",
    "state_dict",
    "model",
    "teacher",
    "ema_model",
    "ema",
)


def safe_torch_load(
    source: Path | str | BinaryIO,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load checkpoint payload with secure defaults.

    Uses ``weights_only=True`` when supported by the current PyTorch version.
    Falls back to the legacy behavior for older versions.
    """
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(source, map_location=map_location)


def is_tensor_state_dict(candidate: Any) -> bool:
    """Return ``True`` when payload looks like a tensor state_dict."""
    return (
        isinstance(candidate, Mapping)
        and bool(candidate)
        and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in candidate.items())
    )


def extract_model_state_dict(
    payload: Any,
    *,
    candidate_keys: tuple[str, ...] = _STATE_DICT_CANDIDATE_KEYS,
) -> tuple[dict[str, torch.Tensor], str]:
    """Extract model state_dict from common checkpoint formats.

    Returns:
        ``(state_dict, checkpoint_format)`` where format is either
        ``raw_state_dict`` or one of the candidate key names.
    """
    if is_tensor_state_dict(payload):
        return cast(dict[str, torch.Tensor], payload), "raw_state_dict"

    if isinstance(payload, Mapping):
        for key in candidate_keys:
            candidate = payload.get(key)
            if is_tensor_state_dict(candidate):
                return cast(dict[str, torch.Tensor], candidate), key

    expected = ", ".join(candidate_keys)
    raise ValueError(
        "Unsupported checkpoint format. Expected a raw tensor state_dict or a mapping "
        f"containing one of: {expected}."
    )


def load_checkpoint_payload(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load checkpoint payload from filesystem with secure defaults."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return safe_torch_load(ckpt_path, map_location=map_location)


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    
    epoch: int
    step: int
    best_metric: float
    best_metric_name: str
    timestamp: str
    config: dict[str, Any]
    train_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-friendly dictionary."""
        payload = asdict(self)
        # `ema_state_dict` is attached dynamically in `load_checkpoint` when present.
        ema_state_dict = getattr(self, "ema_state_dict", None)
        if ema_state_dict is not None:
            payload["ema_state_dict"] = ema_state_dict
        return payload


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int = 0,
    scheduler: _LRScheduler | None = None,
    metrics: dict[str, float] | None = None,
    best_metric: float = 0.0,
    best_metric_name: str = "loss",
    config: dict[str, Any] | None = None,
    is_best: bool = False,
    ema_state_dict: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.
    
    Args:
        path: Path to save checkpoint (will create parent dirs)
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        step: Current training step
        scheduler: Optional LR scheduler to save
        metrics: Current training metrics
        best_metric: Best validation metric achieved so far
        best_metric_name: Name of the best metric (e.g., "mAP", "loss")
        config: Training configuration dict
        is_best: Whether this is the best checkpoint so far
        ema_state_dict: Optional EMA model state dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build metadata
    metadata = CheckpointMetadata(
        epoch=epoch,
        step=step,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        timestamp=datetime.now().isoformat(),
        config=config or {},
        train_metrics=metrics,
    )
    
    # Build checkpoint dict
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": asdict(metadata),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    # Also save metadata as JSON for easy inspection
    metadata_path = path.parent / f"{path.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    
    LOGGER.info(f"Saved checkpoint to {path} (epoch={epoch}, step={step})")
    
    # If this is the best checkpoint, create/update symlink
    if is_best:
        best_path = path.parent / "best.pt"
        if best_path.exists() and best_path.is_symlink():
            best_path.unlink()
        elif best_path.exists():
            # Remove old best file if it's not a symlink
            best_path.unlink()
        
        # Create relative symlink
        try:
            best_path.symlink_to(path.name)
            LOGGER.info(f"Updated best checkpoint symlink -> {path.name}")
        except OSError:
            # Fallback: copy file if symlink fails (e.g., on Windows)
            shutil.copy2(path, best_path)
            LOGGER.info(f"Copied best checkpoint to {best_path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointMetadata:
    """Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map checkpoint to
        strict: Whether to strictly enforce state_dict keys match
    
    Returns:
        CheckpointMetadata with training state info
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    LOGGER.info(f"Loading checkpoint from {path}")
    checkpoint = safe_torch_load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Extract metadata
    metadata_dict = checkpoint.get("metadata", {})
    metadata = CheckpointMetadata(
        epoch=checkpoint.get("epoch", 0),
        step=checkpoint.get("step", 0),
        best_metric=metadata_dict.get("best_metric", 0.0),
        best_metric_name=metadata_dict.get("best_metric_name", "loss"),
        timestamp=metadata_dict.get("timestamp", ""),
        config=metadata_dict.get("config", {}),
        train_metrics=metadata_dict.get("train_metrics"),
    )
    
    LOGGER.info(f"Loaded checkpoint: epoch={metadata.epoch}, step={metadata.step}")
    
    # Return EMA state dict if present (caller can handle it)
    if "ema_state_dict" in checkpoint:
        # Store it in metadata for caller to access
        metadata.__dict__["ema_state_dict"] = checkpoint["ema_state_dict"]
    
    return metadata


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_best: bool = True,
    keep_last_n: int = 3,
    checkpoint_prefix: str = "epoch_",
) -> None:
    """Remove old checkpoints to save disk space.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Whether to keep best.pt (and its target)
        keep_last_n: Number of most recent checkpoints to keep
        checkpoint_prefix: Prefix for checkpoint files to clean
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Find all checkpoint files
    checkpoint_files = sorted(
        checkpoint_dir.glob(f"{checkpoint_prefix}*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # Newest first
    )
    
    if len(checkpoint_files) <= keep_last_n:
        return  # Nothing to clean
    
    # Determine which checkpoint is currently "best"
    best_path = checkpoint_dir / "best.pt"
    best_target = None
    if keep_best and best_path.exists():
        best_target = best_path.resolve() if best_path.is_symlink() else best_path
    
    # Keep last N + best
    to_remove = checkpoint_files[keep_last_n:]
    
    for ckpt_path in to_remove:
        # Don't remove if it's the best checkpoint
        if best_target and ckpt_path.resolve() == best_target.resolve():
            continue
        
        # Remove checkpoint and metadata
        ckpt_path.unlink()
        metadata_path = ckpt_path.parent / f"{ckpt_path.stem}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        LOGGER.info(f"Removed old checkpoint: {ckpt_path.name}")


__all__ = [
    "CheckpointMetadata",
    "safe_torch_load",
    "is_tensor_state_dict",
    "extract_model_state_dict",
    "load_checkpoint_payload",
    "save_checkpoint",
    "load_checkpoint",
    "cleanup_old_checkpoints",
]
