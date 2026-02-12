from __future__ import annotations

import gc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


class MemoryManager:
    """Manages GPU memory and optimizes batch sizes."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()

    def clear(self) -> None:
        """Clear cache and garbage collect."""
        if self.enabled:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics in GB."""
        if not self.enabled:
            return {"allocated": 0.0, "reserved": 0.0, "max_reserved": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved": torch.cuda.memory_reserved(self.device) / 1e9,
            "max_reserved": torch.cuda.max_memory_reserved(self.device) / 1e9,
        }

    @contextmanager
    def catch_oom(self) -> Generator[bool, None, None]:
        """Context manager to catch OOM errors and clear memory."""
        try:
            yield False
        except torch.cuda.OutOfMemoryError:
            self.clear()
            LOGGER.warning("OutOfMemoryError caught! Clearing cache...")
            yield True
        except Exception as e:
            if "out of memory" in str(e).lower():
                self.clear()
                LOGGER.warning(f"OOM-like error caught: {e}. Clearing cache...")
                yield True
            else:
                raise e

    def optimize_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        max_batch_size: int = 64,
        start_batch_size: int = 1,
        mode: str = "train",
    ) -> int:
        """Find the maximum batch size that fits in memory.

        Args:
            model: The PyTorch model (will be put in mode for sizing).
            input_shape: Shape of a single input sample (C, H, W).
            max_batch_size: Upper limit for batch size.
            start_batch_size: Starting batch size for search.
            mode: 'train' or 'eval'. 'train' enables gradients.

        Returns:
            Optimal batch size.
        """
        if not self.enabled:
            LOGGER.info("CUDA not available, returning default batch size 1")
            return 1

        LOGGER.info(f"Auto-tuning batch size (max={max_batch_size})...")
        self.clear()
        
        # Binary search or exponential search
        low = start_batch_size
        high = max_batch_size
        optimal = start_batch_size
        
        model.to(self.device)
        if mode == "train":
            model.train()
        else:
            model.eval()

        # Try powers of 2 first, then refinemement? 
        # Actually binary search is robust.
        
        # But first, quick check if start_batch_size even works
        if not self._try_batch_size(model, input_shape, start_batch_size, mode=mode):
             LOGGER.warning("Even batch_size=1 failed! Check model/memory.")
             return 1

        # Exponential growth to find upper bound
        curr = start_batch_size
        while curr <= max_batch_size:
            if self._try_batch_size(model, input_shape, curr, mode=mode):
                optimal = curr
                if curr == max_batch_size:
                    break
                curr *= 2
            else:
                high = curr
                break
        
        # Binary search between optimal and high
        low = optimal
        while low < high - 1:
            mid = (low + high) // 2
            if self._try_batch_size(model, input_shape, mid, mode=mode):
                optimal = mid
                low = mid
            else:
                high = mid
        
        self.clear()
        LOGGER.info(f"Found optimal batch size: {optimal}")
        return optimal

    def _try_batch_size(
        self, 
        model: nn.Module, 
        input_shape: tuple[int, ...], 
        bsz: int,
        mode: str = "train"
    ) -> bool:
        try:
            dummy_input = torch.rand((bsz, *input_shape), device=self.device)
            with self.catch_oom() as oom:
                if oom: return False
                
                if mode == "train":
                    # Backward pass simulation
                    out = model(dummy_input)
                    if isinstance(out, dict):
                        # Assume scalar loss can be computed from arbitrary output component
                        # Just do a backward on a dummy scalar to checking grad/activation memory
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        for v in out.values():
                             if isinstance(v, Tensor) and v.requires_grad:
                                  loss = loss + v.mean() * 0.0
                        loss.backward()
                        
                        # Zero grad to release graph
                        for p in model.parameters():
                            p.grad = None
                    elif isinstance(out, Tensor):
                         loss = out.mean()
                         loss.backward()
                         for p in model.parameters():
                            p.grad = None
                else:
                    with torch.no_grad():
                        _ = model(dummy_input)
            
            return True
        except Exception:
            return False
        finally:
            self.clear()

