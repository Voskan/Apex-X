from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

try:
    import tensorrt as trt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    trt = None


CalibrationBatch = Mapping[str, np.ndarray] | np.ndarray


class CalibrationDataLoader(Protocol):
    """Protocol for calibration dataset loaders consumed by TRT calibrator."""

    def __iter__(self) -> Iterator[CalibrationBatch]:
        ...


@dataclass(frozen=True, slots=True)
class CalibratorConfig:
    input_names: tuple[str, ...]
    cache_path: Path | None = None
    device: str = "cuda"
    force_float32: bool = True


def _normalize_batch(
    batch: CalibrationBatch,
    *,
    input_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if isinstance(batch, np.ndarray):
        if len(input_names) != 1:
            raise ValueError(
                "single-array calibration batch requires exactly one input name"
            )
        normalized = {input_names[0]: batch}
    elif isinstance(batch, Mapping):
        normalized = {}
        for name in input_names:
            if name not in batch:
                raise ValueError(f"missing calibration input: {name}")
            value = batch[name]
            if not isinstance(value, np.ndarray):
                raise ValueError(f"calibration input {name} must be a numpy array")
            normalized[name] = value
    else:
        raise ValueError("calibration batch must be numpy array or mapping of numpy arrays")

    for name, value in normalized.items():
        if value.ndim == 0:
            raise ValueError(f"calibration input {name} must have batch dimension")
        if value.shape[0] <= 0:
            raise ValueError(f"calibration input {name} has empty batch")
        if not np.isfinite(value).all():
            raise ValueError(f"calibration input {name} must be finite")
    return normalized


class _EntropyCalibratorBase:
    def __init__(
        self,
        loader: CalibrationDataLoader | Iterable[CalibrationBatch],
        *,
        config: CalibratorConfig,
    ) -> None:
        if not config.input_names:
            raise ValueError("input_names must be non-empty")
        self.config = config
        self._iterator = iter(loader)
        self._pending_batch: dict[str, np.ndarray] | None = None
        self._batch_size: int | None = None
        self._device_tensors: dict[str, torch.Tensor] = {}
        self._cache_path = None if config.cache_path is None else Path(config.cache_path)
        self._prime_first_batch()

    def _prime_first_batch(self) -> None:
        try:
            raw = next(self._iterator)
        except StopIteration as exc:
            raise ValueError("calibration loader must provide at least one batch") from exc
        normalized = _normalize_batch(raw, input_names=self.config.input_names)
        self._batch_size = int(next(iter(normalized.values())).shape[0])
        for name in self.config.input_names:
            if int(normalized[name].shape[0]) != self._batch_size:
                raise ValueError("all calibration inputs must share batch dimension")
        self._pending_batch = normalized

    def get_batch_size(self) -> int:
        assert self._batch_size is not None
        return self._batch_size

    def _to_device_tensor(self, array: np.ndarray) -> torch.Tensor:
        target_dtype = np.float32 if self.config.force_float32 else array.dtype
        casted = np.asarray(array, dtype=target_dtype, order="C")
        tensor = torch.from_numpy(casted)
        if self.config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for TensorRT INT8 calibration")
            tensor = tensor.to(device=self.config.device, non_blocking=False)
        return tensor.contiguous()

    def get_batch(self, names: list[str]) -> list[int] | None:
        if tuple(names) != self.config.input_names:
            missing = set(self.config.input_names).difference(names)
            extra = set(names).difference(self.config.input_names)
            raise ValueError(
                f"unexpected calibration names; missing={sorted(missing)}, extra={sorted(extra)}"
            )

        batch = self._pending_batch
        if batch is None:
            try:
                raw = next(self._iterator)
            except StopIteration:
                return None
            batch = _normalize_batch(raw, input_names=self.config.input_names)
            if int(next(iter(batch.values())).shape[0]) != self.get_batch_size():
                raise ValueError("calibration batch size must remain constant")

        ptrs: list[int] = []
        self._device_tensors.clear()
        for name in self.config.input_names:
            tensor = self._to_device_tensor(batch[name])
            self._device_tensors[name] = tensor
            ptrs.append(int(tensor.data_ptr()))
        self._pending_batch = None
        return ptrs

    def read_calibration_cache(self) -> bytes | None:
        if self._cache_path is None:
            return None
        if not self._cache_path.exists():
            return None
        return self._cache_path.read_bytes()

    def write_calibration_cache(self, cache: bytes) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_bytes(cache)


if trt is not None:

    class _TensorRTEntropyCalibratorTRT(trt.IInt8EntropyCalibrator2):
        """Entropy calibrator streaming batches from a Python loader."""

        def __init__(
            self,
            loader: CalibrationDataLoader | Iterable[CalibrationBatch],
            *,
            config: CalibratorConfig,
        ) -> None:
            super().__init__()
            self._impl = _EntropyCalibratorBase(loader=loader, config=config)

        def get_batch_size(self) -> int:
            return self._impl.get_batch_size()

        def get_batch(self, names: list[str]) -> list[int] | None:
            return self._impl.get_batch(names)

        def read_calibration_cache(self) -> bytes | None:
            return self._impl.read_calibration_cache()

        def write_calibration_cache(self, cache: bytes) -> None:
            self._impl.write_calibration_cache(cache)

else:

    class _TensorRTEntropyCalibratorUnavailable:
        """Import-guard placeholder when TensorRT Python is unavailable."""

        def __init__(
            self,
            loader: CalibrationDataLoader | Iterable[CalibrationBatch],
            *,
            config: CalibratorConfig,
        ) -> None:
            _ = (loader, config)
            raise RuntimeError(
                "tensorrt Python package is required for TensorRTEntropyCalibrator"
            )


TensorRTEntropyCalibrator: type[object]

if trt is not None:
    TensorRTEntropyCalibrator = _TensorRTEntropyCalibratorTRT
else:
    TensorRTEntropyCalibrator = _TensorRTEntropyCalibratorUnavailable


__all__ = [
    "CalibrationBatch",
    "CalibrationDataLoader",
    "CalibratorConfig",
    "TensorRTEntropyCalibrator",
]
