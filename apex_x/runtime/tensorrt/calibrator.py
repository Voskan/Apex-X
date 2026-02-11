from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
CALIBRATION_CACHE_MAGIC = "APEXX_CALIBRATION_CACHE_V1"


class CalibrationDataLoader(Protocol):
    """Protocol for calibration dataset loaders consumed by TRT calibrator."""

    def __iter__(self) -> Iterator[CalibrationBatch]: ...


@dataclass(frozen=True, slots=True)
class CalibratorConfig:
    input_names: tuple[str, ...]
    cache_path: Path | None = None
    cache_key: str | None = None
    device: str = "cuda"
    force_float32: bool = True


def build_calibration_dataset_digest(
    batches: Sequence[CalibrationBatch],
    *,
    max_batches: int = 32,
) -> str:
    if max_batches <= 0:
        raise ValueError("max_batches must be > 0")
    if not batches:
        raise ValueError("batches must be non-empty")

    sampled = min(len(batches), max_batches)
    hasher = hashlib.sha256()
    hasher.update(b"apexx_calibration_dataset_digest_v1")
    hasher.update(f"|total={len(batches)}|sampled={sampled}".encode())

    for idx in range(sampled):
        batch = batches[idx]
        hasher.update(f"|batch={idx}".encode())
        if isinstance(batch, np.ndarray):
            array = np.ascontiguousarray(batch)
            hasher.update(b"|array|")
            hasher.update(str(array.dtype).encode("utf-8"))
            hasher.update(str(tuple(array.shape)).encode("utf-8"))
            hasher.update(array.view(np.uint8).tobytes())
            continue
        if isinstance(batch, Mapping):
            hasher.update(b"|mapping|")
            for name in sorted(batch):
                value = batch[name]
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"calibration input {name} must be a numpy array")
                array = np.ascontiguousarray(value)
                hasher.update(name.encode("utf-8"))
                hasher.update(str(array.dtype).encode("utf-8"))
                hasher.update(str(tuple(array.shape)).encode("utf-8"))
                hasher.update(array.view(np.uint8).tobytes())
            continue
        raise ValueError("calibration batch must be numpy array or mapping of numpy arrays")
    return hasher.hexdigest()


def _normalize_batch(
    batch: CalibrationBatch,
    *,
    input_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if isinstance(batch, np.ndarray):
        if len(input_names) != 1:
            raise ValueError("single-array calibration batch requires exactly one input name")
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
        if config.cache_key is not None and not str(config.cache_key).strip():
            raise ValueError("cache_key must be non-empty when provided")
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
        blob = self._cache_path.read_bytes()
        parsed = _parse_calibration_cache_blob(blob)
        if parsed is None:
            # Legacy cache blobs are accepted only when no cache-key governance is required.
            if self.config.cache_key is None:
                return blob
            return None

        metadata, payload = parsed
        key_in_blob = metadata.get("cache_key")
        if self.config.cache_key is None:
            return payload
        if not isinstance(key_in_blob, str):
            return None
        if key_in_blob != self.config.cache_key:
            return None
        return payload

    def write_calibration_cache(self, cache: bytes) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.cache_key is None:
            self._cache_path.write_bytes(cache)
            return
        wrapped = _build_calibration_cache_blob(
            payload=cache,
            cache_key=self.config.cache_key,
        )
        self._cache_path.write_bytes(wrapped)


def _build_calibration_cache_blob(*, payload: bytes, cache_key: str) -> bytes:
    metadata = {
        "schema_version": 1,
        "cache_key": cache_key,
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return CALIBRATION_CACHE_MAGIC.encode("utf-8") + b"\n" + header + b"\n\n" + payload


def _parse_calibration_cache_blob(blob: bytes) -> tuple[dict[str, object], bytes] | None:
    prefix = CALIBRATION_CACHE_MAGIC.encode("utf-8") + b"\n"
    if not blob.startswith(prefix):
        return None
    separator = b"\n\n"
    split = blob.find(separator, len(prefix))
    if split < 0:
        return None
    header_bytes = blob[len(prefix) : split]
    payload = blob[split + len(separator) :]
    try:
        metadata = json.loads(header_bytes.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(metadata, dict):
        return None
    return metadata, payload


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
            raise RuntimeError("tensorrt Python package is required for TensorRTEntropyCalibrator")


TensorRTEntropyCalibrator: type[object]

if trt is not None:
    TensorRTEntropyCalibrator = _TensorRTEntropyCalibratorTRT
else:
    TensorRTEntropyCalibrator = _TensorRTEntropyCalibratorUnavailable


__all__ = [
    "CalibrationBatch",
    "CalibrationDataLoader",
    "CalibratorConfig",
    "CALIBRATION_CACHE_MAGIC",
    "build_calibration_dataset_digest",
    "TensorRTEntropyCalibrator",
]
