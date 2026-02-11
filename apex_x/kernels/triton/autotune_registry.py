from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class TritonAutotuneEntry:
    op_name: str
    kernel_name: str
    shape_bucket: str
    selected_config: dict[str, Any]
    selection_source: str
    launches: int
    cache_hits: int
    cache_misses: int


_REGISTRY_LOCK = Lock()
_TRITON_AUTOTUNE_REGISTRY: dict[tuple[str, str], TritonAutotuneEntry] = {}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return None
    return str(value)


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    normalized = {str(k): _normalize_value(v) for k, v in config.items()}
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def build_shape_bucket(**fields: Any) -> str:
    pairs = [
        f"{str(k)}={_normalize_value(v)}" for k, v in sorted(fields.items(), key=lambda x: x[0])
    ]
    return "|".join(pairs)


def clear_triton_autotune_registry() -> None:
    with _REGISTRY_LOCK:
        _TRITON_AUTOTUNE_REGISTRY.clear()


def get_cached_triton_config(*, op_name: str, shape_bucket: str) -> dict[str, Any] | None:
    with _REGISTRY_LOCK:
        entry = _TRITON_AUTOTUNE_REGISTRY.get((op_name, shape_bucket))
        if entry is None:
            return None
        return dict(entry.selected_config)


def extract_triton_best_config(kernel: Any) -> dict[str, Any] | None:
    config = getattr(kernel, "best_config", None)
    if config is None:
        return None

    if isinstance(config, dict):
        return _normalize_config(config)

    values: dict[str, Any] = {}
    kwargs = getattr(config, "kwargs", None)
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            values[str(key)] = _normalize_value(value)
    num_warps = getattr(config, "num_warps", None)
    if num_warps is not None:
        values["num_warps"] = int(num_warps)
    num_stages = getattr(config, "num_stages", None)
    if num_stages is not None:
        values["num_stages"] = int(num_stages)
    num_ctas = getattr(config, "num_ctas", None)
    if num_ctas is not None:
        values["num_ctas"] = int(num_ctas)
    if not values:
        return None
    return _normalize_config(values)


def resolve_triton_launch_config(
    *,
    kernel: Any,
    fallback_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    best = extract_triton_best_config(kernel)
    if best is not None:
        return best, "triton_best_config"
    return _normalize_config(fallback_config), "heuristic"


def record_triton_autotune_selection(
    *,
    op_name: str,
    kernel_name: str,
    shape_bucket: str,
    selected_config: dict[str, Any],
    selection_source: str,
) -> None:
    key = (op_name, shape_bucket)
    normalized_config = _normalize_config(selected_config)

    with _REGISTRY_LOCK:
        existing = _TRITON_AUTOTUNE_REGISTRY.get(key)
        if existing is None:
            _TRITON_AUTOTUNE_REGISTRY[key] = TritonAutotuneEntry(
                op_name=op_name,
                kernel_name=kernel_name,
                shape_bucket=shape_bucket,
                selected_config=normalized_config,
                selection_source=selection_source,
                launches=1,
                cache_hits=0,
                cache_misses=1,
            )
            return

        updated_config = existing.selected_config
        updated_source = existing.selection_source
        if selection_source == "triton_best_config" or not updated_config and normalized_config:
            updated_config = normalized_config
            updated_source = selection_source

        _TRITON_AUTOTUNE_REGISTRY[key] = TritonAutotuneEntry(
            op_name=existing.op_name,
            kernel_name=existing.kernel_name,
            shape_bucket=existing.shape_bucket,
            selected_config=updated_config,
            selection_source=updated_source,
            launches=existing.launches + 1,
            cache_hits=existing.cache_hits + 1,
            cache_misses=existing.cache_misses,
        )


def snapshot_triton_autotune_registry() -> dict[str, Any]:
    with _REGISTRY_LOCK:
        entries = sorted(
            _TRITON_AUTOTUNE_REGISTRY.values(),
            key=lambda item: (item.op_name, item.shape_bucket),
        )

    launches = sum(item.launches for item in entries)
    cache_hits = sum(item.cache_hits for item in entries)
    cache_misses = sum(item.cache_misses for item in entries)
    total_cache_events = cache_hits + cache_misses
    hit_rate = float(cache_hits / total_cache_events) if total_cache_events > 0 else None

    return {
        "summary": {
            "cache_entries": len(entries),
            "launches": launches,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": hit_rate,
        },
        "entries": [
            {
                "op_name": item.op_name,
                "kernel_name": item.kernel_name,
                "shape_bucket": item.shape_bucket,
                "selected_config": dict(item.selected_config),
                "selection_source": item.selection_source,
                "launches": item.launches,
                "cache_hits": item.cache_hits,
                "cache_misses": item.cache_misses,
            }
            for item in entries
        ],
    }


__all__ = [
    "TritonAutotuneEntry",
    "build_shape_bucket",
    "clear_triton_autotune_registry",
    "get_cached_triton_config",
    "extract_triton_best_config",
    "resolve_triton_launch_config",
    "record_triton_autotune_selection",
    "snapshot_triton_autotune_registry",
]
