from __future__ import annotations

import gc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class OOMState:
    triggered: bool = False
    message: str | None = None

    def __bool__(self) -> bool:
        return bool(self.triggered)


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
    def catch_oom(self) -> Generator[OOMState, None, None]:
        """Context manager to catch OOM errors and clear memory."""
        state = OOMState()
        try:
            yield state
        except torch.cuda.OutOfMemoryError as exc:
            state.triggered = True
            state.message = str(exc)
            self.clear()
            LOGGER.warning("OutOfMemoryError caught! Clearing cache...")
        except Exception as e:
            message = str(e)
            if "out of memory" in message.lower():
                state.triggered = True
                state.message = message
                self.clear()
                LOGGER.warning(f"OOM-like error caught: {e}. Clearing cache...")
            else:
                raise

    def optimize_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        max_batch_size: int = 64,
        start_batch_size: int = 1,
        mode: str = "train",
    ) -> int:
        """Find the maximum batch size that fits in memory."""
        if not self.enabled:
            LOGGER.info("CUDA not available, returning default batch size 1")
            return 1

        LOGGER.info(f"Auto-tuning batch size (max={max_batch_size})...")
        self.clear()
        
        # Binary search
        low = start_batch_size
        high = max_batch_size
        optimal = start_batch_size
        
        model.to(self.device)
        if mode == "train":
            model.train()
        else:
            model.eval()

        if not self._try_batch_size(model, input_shape, start_batch_size, mode=mode):
             LOGGER.warning("Even batch_size=1 failed! Check model/memory.")
             return 1

        # Exponential growth
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
        
        # Binary search
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

    def downshift_batch_size(self, current_bsz: int, min_bsz: int = 1) -> int:
        """Reduce batch size in response to OOM."""
        new_bsz = max(min_bsz, current_bsz // 2)
        LOGGER.warning(f"Downshifting batch size: {current_bsz} -> {new_bsz}")
        self.clear()
        return new_bsz
        
    def guard(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function with OOM guarding.
        
        If OOM occurs, clears cache and re-raises to let caller handle retry settings.
        """
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            self.clear()
            LOGGER.error("OOM caught in guarded execution!")
            raise
        except Exception as e:
            if "out of memory" in str(e).lower():
                 self.clear()
                 LOGGER.error(f"OOM-like error caught: {e}")
                 # Re-raise as proper OOM for handling
                 raise torch.cuda.OutOfMemoryError(str(e))
            raise

    def _try_batch_size(
        self, 
        model: nn.Module, 
        input_shape: tuple[int, ...], 
        bsz: int,
        mode: str = "train"
    ) -> bool:
        try:
            dummy_input = torch.rand((bsz, *input_shape), device=self.device)
            with self.catch_oom() as oom_state:
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
            if oom_state.triggered:
                return False
            return True
        except Exception:
            return False
        finally:
            self.clear()
