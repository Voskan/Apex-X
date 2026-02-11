from __future__ import annotations

import ctypes
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from apex_x.runtime.caps import RuntimeCaps, detect_runtime_caps
from apex_x.utils import hash_file_sha256
from apex_x.utils.logging import get_logger

from .calibrator import (
    CalibrationBatch,
    CalibratorConfig,
    TensorRTEntropyCalibrator,
    build_calibration_dataset_digest,
)

try:
    import tensorrt as trt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    trt = None


NetworkBuilderFn = Callable[[Any, Any], None]

DEFAULT_PLUGIN_VERSION = "1"
DEFAULT_PLUGIN_NAMESPACE = "apexx"
DEFAULT_PLUGIN_FIELD_SIGNATURES: dict[str, tuple[str, ...]] = {
    "TilePack": ("tile_size",),
    "TileSSMScan": ("direction", "clamp_value"),
    "TileUnpackFusion": (),
    "DecodeNMS": (
        "max_detections",
        "pre_nms_topk",
        "score_threshold",
        "iou_threshold",
    ),
}


@dataclass(frozen=True, slots=True)
class PluginStatus:
    name: str
    required: bool
    found: bool
    expected_version: str | None = None
    discovered_version: str | None = None
    version_match: bool | None = None
    expected_namespace: str | None = None
    discovered_namespace: str | None = None
    namespace_match: bool | None = None
    expected_fields: tuple[str, ...] = ()
    discovered_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    field_signature_match: bool | None = None


@dataclass(frozen=True, slots=True)
class PluginContract:
    name: str
    required: bool = True
    expected_version: str | None = DEFAULT_PLUGIN_VERSION
    expected_namespace: str | None = DEFAULT_PLUGIN_NAMESPACE
    expected_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PluginContract.name must be non-empty")


@dataclass(frozen=True, slots=True)
class ExportManifestInfo:
    manifest_path: Path
    onnx_path: Path
    onnx_sha256: str | None
    shape_mode: str | None
    profile: str | None


@dataclass(frozen=True, slots=True)
class TensorRTEngineBuildConfig:
    max_workspace_size_bytes: int = 1 << 30
    enable_fp16: bool = True
    enable_int8: bool = False
    optimization_level: int | None = None
    router_fp16_layer_keywords: tuple[str, ...] = ("router", "kan")
    strict_plugin_check: bool = True
    strict_precision_constraints: bool = True
    include_optional_nms_plugin: bool = False
    expected_plugins: tuple[str, ...] = ("TilePack", "TileSSMScan", "TileUnpackFusion")
    expected_plugin_contracts: tuple[PluginContract, ...] = ()
    calibration_cache_path: Path | None = None
    calibration_cache_key_override: str | None = None
    calibration_dataset_version: str | None = None
    precision_profile: str = "edge"


@dataclass(frozen=True, slots=True)
class LayerPrecisionStatus:
    layer_name: str
    matched_keyword: str
    precision_applied: bool
    output_constraints_applied: int


@dataclass(frozen=True, slots=True)
class EngineBuildResult:
    engine_path: Path
    used_fp16: bool
    used_int8: bool
    plugin_status: tuple[PluginStatus, ...]
    calibration_cache_path: Path | None
    calibration_cache_key: str | None = None
    calibration_dataset_version: str | None = None
    layer_precision_status: tuple[LayerPrecisionStatus, ...] = ()


def build_calibration_cache_key(
    *,
    onnx_sha256: str,
    plugin_statuses: Sequence[PluginStatus],
    precision_profile: str,
    calibration_dataset_version: str,
) -> str:
    if not onnx_sha256.strip():
        raise ValueError("onnx_sha256 must be non-empty")
    if not precision_profile.strip():
        raise ValueError("precision_profile must be non-empty")
    if not calibration_dataset_version.strip():
        raise ValueError("calibration_dataset_version must be non-empty")

    plugin_meta: list[dict[str, str]] = []
    for status in sorted(plugin_statuses, key=lambda item: item.name):
        plugin_meta.append(
            {
                "name": status.name,
                "version": status.discovered_version or status.expected_version or "unknown",
                "namespace": status.discovered_namespace or status.expected_namespace or "",
            }
        )
    payload = {
        "schema_version": 1,
        "onnx_sha256": onnx_sha256,
        "precision_profile": precision_profile,
        "calibration_dataset_version": calibration_dataset_version,
        "plugins": plugin_meta,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_export_manifest(
    *,
    manifest_path: str | Path,
    verify_onnx_hash: bool = True,
) -> ExportManifestInfo:
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"export manifest not found: {manifest_file}")
    payload_raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise ValueError("export manifest must be a JSON object")
    payload = cast(dict[str, Any], payload_raw)

    if payload.get("format") != "onnx":
        raise ValueError("export manifest format must be 'onnx'")
    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, dict):
        raise ValueError("export manifest must include object field 'artifacts'")
    artifacts = cast(dict[str, Any], artifacts_raw)
    onnx_path_raw = artifacts.get("onnx_path")
    if not isinstance(onnx_path_raw, str) or not onnx_path_raw.strip():
        raise ValueError("export manifest artifacts.onnx_path must be a non-empty string")

    onnx_path = Path(onnx_path_raw).expanduser()
    if not onnx_path.is_absolute():
        onnx_path = (manifest_file.parent / onnx_path).resolve()
    else:
        onnx_path = onnx_path.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"export manifest ONNX file not found: {onnx_path}")

    onnx_sha = artifacts.get("onnx_sha256")
    if onnx_sha is not None and not isinstance(onnx_sha, str):
        raise ValueError("export manifest artifacts.onnx_sha256 must be a string when present")
    if verify_onnx_hash and isinstance(onnx_sha, str) and onnx_sha:
        actual = hash_file_sha256(onnx_path)
        if actual != onnx_sha:
            raise ValueError(
                "export manifest ONNX hash mismatch: " f"expected={onnx_sha} actual={actual}"
            )

    shape_mode = payload.get("shape_mode")
    profile = payload.get("profile")
    shape_mode_str = str(shape_mode) if shape_mode is not None else None
    profile_str = str(profile) if profile is not None else None
    return ExportManifestInfo(
        manifest_path=manifest_file,
        onnx_path=onnx_path,
        onnx_sha256=onnx_sha if isinstance(onnx_sha, str) else None,
        shape_mode=shape_mode_str,
        profile=profile_str,
    )


class TensorRTEngineBuilder:
    """TensorRT engine builder with Apex-X custom plugin registration hooks."""

    def __init__(
        self,
        *,
        plugin_library_paths: Sequence[str | Path] | None = None,
        runtime_caps: RuntimeCaps | None = None,
    ) -> None:
        self._logger = get_logger(__name__)
        self._caps = detect_runtime_caps() if runtime_caps is None else runtime_caps
        self._loaded_plugin_libraries: list[ctypes.CDLL] = []
        self._active_calibrator: Any | None = None
        self._plugin_library_paths: tuple[Path, ...] = tuple(
            Path(p).expanduser().resolve() for p in (plugin_library_paths or ())
        )
        if trt is None:
            self._trt_logger = None
        else:
            self._trt_logger = trt.Logger(trt.Logger.WARNING)

    @property
    def runtime_caps(self) -> RuntimeCaps:
        return self._caps

    def load_plugin_libraries(self) -> tuple[Path, ...]:
        dlopen_mode = int(getattr(ctypes, "RTLD_GLOBAL", 0))
        loaded: list[Path] = []
        for path in self._plugin_library_paths:
            if not path.exists():
                raise FileNotFoundError(f"plugin library not found: {path}")
            handle = ctypes.CDLL(str(path), mode=dlopen_mode)
            self._loaded_plugin_libraries.append(handle)
            loaded.append(path)
            self._logger.info("loaded_tensorrt_plugin_library path=%s", path)
        return tuple(loaded)

    def _require_trt(self) -> Any:
        if trt is None:
            raise RuntimeError("tensorrt Python package is not available")
        assert self._trt_logger is not None
        return trt

    def _plugin_statuses(
        self,
        *,
        plugin_contracts: Sequence[PluginContract],
    ) -> tuple[PluginStatus, ...]:
        trt_mod = self._require_trt()
        registry = trt_mod.get_plugin_registry()
        statuses: list[PluginStatus] = []
        for contract in plugin_contracts:
            creator = self._find_plugin_creator(registry=registry, contract=contract)
            if creator is None:
                statuses.append(
                    PluginStatus(
                        name=contract.name,
                        required=contract.required,
                        found=False,
                        expected_version=contract.expected_version,
                        expected_namespace=contract.expected_namespace,
                        expected_fields=contract.expected_fields,
                    )
                )
                continue

            discovered_version = self._normalize_optional_str(
                getattr(creator, "plugin_version", None)
            )
            discovered_namespace = self._normalize_optional_str(
                getattr(creator, "plugin_namespace", None)
            )
            discovered_fields = self._extract_creator_field_names(creator)
            missing_fields = tuple(
                name for name in contract.expected_fields if name not in set(discovered_fields)
            )
            version_match = (
                True
                if contract.expected_version is None
                else discovered_version == contract.expected_version
            )
            namespace_match = (
                True
                if contract.expected_namespace is None
                else discovered_namespace == contract.expected_namespace
            )
            statuses.append(
                PluginStatus(
                    name=contract.name,
                    required=contract.required,
                    found=True,
                    expected_version=contract.expected_version,
                    discovered_version=discovered_version,
                    version_match=version_match,
                    expected_namespace=contract.expected_namespace,
                    discovered_namespace=discovered_namespace,
                    namespace_match=namespace_match,
                    expected_fields=contract.expected_fields,
                    discovered_fields=discovered_fields,
                    missing_fields=missing_fields,
                    field_signature_match=len(missing_fields) == 0,
                )
            )
        return tuple(statuses)

    def _resolve_plugin_contracts(
        self,
        *,
        build: TensorRTEngineBuildConfig,
    ) -> tuple[PluginContract, ...]:
        if build.expected_plugin_contracts:
            contracts = list(build.expected_plugin_contracts)
        else:
            contracts = [
                self._default_plugin_contract(name=name, required=True)
                for name in build.expected_plugins
            ]

        if build.include_optional_nms_plugin and not any(
            contract.name == "DecodeNMS" for contract in contracts
        ):
            contracts.append(self._default_plugin_contract(name="DecodeNMS", required=False))
        return tuple(contracts)

    def _default_plugin_contract(self, *, name: str, required: bool) -> PluginContract:
        return PluginContract(
            name=name,
            required=required,
            expected_version=DEFAULT_PLUGIN_VERSION,
            expected_namespace=DEFAULT_PLUGIN_NAMESPACE,
            expected_fields=DEFAULT_PLUGIN_FIELD_SIGNATURES.get(name, ()),
        )

    def _find_plugin_creator(self, *, registry: Any, contract: PluginContract) -> Any | None:
        getter = getattr(registry, "get_plugin_creator", None)
        if (
            callable(getter)
            and contract.expected_version is not None
            and contract.expected_namespace is not None
        ):
            try:
                creator = getter(
                    contract.name,
                    contract.expected_version,
                    contract.expected_namespace,
                )
                if creator is not None:
                    return creator
            except Exception:
                pass

        for creator in getattr(registry, "plugin_creator_list", ()):
            if getattr(creator, "name", None) != contract.name:
                continue
            return creator
        return None

    def _extract_creator_field_names(self, creator: Any) -> tuple[str, ...]:
        raw_fields = getattr(creator, "field_names", None)
        if raw_fields is None and hasattr(creator, "get_field_names"):
            try:
                raw_fields = creator.get_field_names()
            except Exception:
                raw_fields = None
        if raw_fields is None:
            return ()

        names: list[str] = []
        if hasattr(raw_fields, "nb_fields") and hasattr(raw_fields, "fields"):
            nb_fields = int(raw_fields.nb_fields)
            fields = raw_fields.fields
            for idx in range(nb_fields):
                field = fields[idx]
                name = getattr(field, "name", None)
                if isinstance(name, str):
                    names.append(name)
        else:
            for field in raw_fields:
                if isinstance(field, str):
                    names.append(field)
                    continue
                name = getattr(field, "name", None)
                if isinstance(name, str):
                    names.append(name)

        deduped = sorted(set(names))
        return tuple(deduped)

    def _normalize_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        value_str = str(value)
        return value_str

    def _ensure_plugin_status(
        self,
        statuses: Sequence[PluginStatus],
        *,
        strict: bool,
    ) -> None:
        missing_required = [
            status.name for status in statuses if status.required and not status.found
        ]

        mismatched_required: list[str] = []
        mismatched_optional: list[str] = []
        for status in statuses:
            if not status.found:
                continue
            issues: list[str] = []
            if status.version_match is False:
                issues.append(
                    "version "
                    f"expected={status.expected_version!r} found={status.discovered_version!r}"
                )
            if status.namespace_match is False:
                issues.append(
                    "namespace "
                    f"expected={status.expected_namespace!r} found={status.discovered_namespace!r}"
                )
            if status.field_signature_match is False:
                issues.append(
                    "field_signature "
                    f"missing={list(status.missing_fields)!r} "
                    f"expected={list(status.expected_fields)!r} "
                    f"found={list(status.discovered_fields)!r}"
                )
            if not issues:
                continue
            line = f"{status.name}: " + "; ".join(issues)
            if status.required:
                mismatched_required.append(line)
            else:
                mismatched_optional.append(line)

        if strict and (missing_required or mismatched_required):
            parts: list[str] = []
            if missing_required:
                parts.append("missing required plugins: " + ", ".join(missing_required))
            if mismatched_required:
                parts.append(
                    "mismatched required plugin contracts: " + " | ".join(mismatched_required)
                )
            raise RuntimeError("TensorRT plugin contract validation failed: " + " ; ".join(parts))
        if missing_required:
            self._logger.warning("missing_required_tensorrt_plugins plugins=%s", missing_required)
        if mismatched_required:
            self._logger.warning(
                "mismatched_required_tensorrt_plugins details=%s",
                mismatched_required,
            )
        if mismatched_optional:
            self._logger.warning(
                "mismatched_optional_tensorrt_plugins details=%s",
                mismatched_optional,
            )

    def _set_memory_pool_limit(self, config_obj: Any, workspace_bytes: int) -> None:
        if workspace_bytes <= 0:
            raise ValueError("max_workspace_size_bytes must be > 0")
        if hasattr(config_obj, "set_memory_pool_limit"):
            config_obj.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
            return
        config_obj.max_workspace_size = int(workspace_bytes)

    def _mark_router_layers_fp16(
        self,
        network: Any,
        layer_keywords: Sequence[str],
        *,
        strict: bool,
    ) -> tuple[LayerPrecisionStatus, ...]:
        if not layer_keywords:
            return ()
        statuses: list[LayerPrecisionStatus] = []
        for index in range(int(network.num_layers)):
            layer = network.get_layer(index)
            name = (layer.name or "").lower()
            matched_keyword: str | None = None
            for keyword in layer_keywords:
                if keyword.lower() in name:
                    matched_keyword = keyword
                    break
            if matched_keyword is None:
                continue
            precision_applied = False
            output_constraints = 0
            if hasattr(layer, "precision"):
                layer.precision = trt.float16
                precision_applied = True
            if hasattr(layer, "set_output_type"):
                for output_idx in range(int(layer.num_outputs)):
                    layer.set_output_type(output_idx, trt.float16)
                    output_constraints += 1
            if strict and not precision_applied and output_constraints <= 0:
                raw_name = layer.name if layer.name else f"layer_{index}"
                raise RuntimeError(
                    "unable to enforce FP16 precision constraint for sensitive layer "
                    f"{raw_name!r}; TensorRT layer does not expose precision/output-type APIs"
                )
            statuses.append(
                LayerPrecisionStatus(
                    layer_name=layer.name if layer.name else f"layer_{index}",
                    matched_keyword=matched_keyword,
                    precision_applied=precision_applied,
                    output_constraints_applied=output_constraints,
                )
            )
        return tuple(statuses)

    def _resolve_calibration_dataset_version(
        self,
        *,
        build: TensorRTEngineBuildConfig,
        calibration_batches: Sequence[CalibrationBatch] | None,
    ) -> str:
        explicit_version = build.calibration_dataset_version
        if explicit_version is not None:
            normalized = explicit_version.strip()
            if not normalized:
                raise ValueError("calibration_dataset_version must be non-empty when provided")
            return normalized
        if calibration_batches is None:
            raise ValueError("INT8 build requires calibration_batches")
        return build_calibration_dataset_digest(calibration_batches)

    def _build_network_definition_signature(self, network: Any) -> str:
        layers: list[dict[str, object]] = []
        for index in range(int(network.num_layers)):
            layer = network.get_layer(index)
            output_shapes: list[tuple[int, ...] | None] = []
            if hasattr(layer, "get_output"):
                for output_idx in range(int(layer.num_outputs)):
                    output = layer.get_output(output_idx)
                    shape_raw = getattr(output, "shape", None)
                    if shape_raw is None:
                        output_shapes.append(None)
                        continue
                    shape = tuple(int(dim) for dim in shape_raw)
                    output_shapes.append(shape)
            layers.append(
                {
                    "index": index,
                    "name": layer.name if layer.name else f"layer_{index}",
                    "type": str(getattr(layer, "type", "")),
                    "num_inputs": int(getattr(layer, "num_inputs", 0)),
                    "num_outputs": int(getattr(layer, "num_outputs", 0)),
                    "output_shapes": output_shapes,
                }
            )
        encoded = json.dumps(layers, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_serialized_network(
        self,
        *,
        builder: Any,
        network: Any,
        config_obj: Any,
    ) -> bytes:
        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config_obj)
            if serialized is None:
                raise RuntimeError("TensorRT build_serialized_network returned None")
            return bytes(serialized)
        engine = builder.build_engine(network, config_obj)  # pragma: no cover - legacy TRT
        if engine is None:
            raise RuntimeError("TensorRT build_engine returned None")
        return bytes(engine.serialize())

    def _prepare_builder_config(
        self,
        *,
        builder: Any,
        network: Any,
        build: TensorRTEngineBuildConfig,
        calibration_batches: Sequence[CalibrationBatch] | None,
        calibration_cache_key: str | None = None,
    ) -> tuple[Any, bool, bool, tuple[LayerPrecisionStatus, ...]]:
        trt_mod = self._require_trt()
        config_obj = builder.create_builder_config()
        self._set_memory_pool_limit(config_obj, build.max_workspace_size_bytes)
        if build.optimization_level is not None and hasattr(
            config_obj, "builder_optimization_level"
        ):
            config_obj.builder_optimization_level = int(build.optimization_level)

        used_fp16 = False
        used_int8 = False
        layer_precision_status: tuple[LayerPrecisionStatus, ...] = ()
        if build.enable_fp16 and hasattr(trt_mod.BuilderFlag, "FP16"):
            config_obj.set_flag(trt_mod.BuilderFlag.FP16)
            used_fp16 = True

        if build.enable_int8:
            if not hasattr(trt_mod.BuilderFlag, "INT8"):
                raise RuntimeError("TensorRT INT8 builder flag is unavailable")
            if calibration_batches is None:
                raise ValueError("INT8 build requires calibration_batches")
            if not self._caps.cuda.available:
                raise RuntimeError("CUDA is required for TensorRT INT8 build")
            config_obj.set_flag(trt_mod.BuilderFlag.INT8)
            cache_path = build.calibration_cache_path
            calibrator_cls = cast(Any, TensorRTEntropyCalibrator)
            calibrator = calibrator_cls(
                calibration_batches,
                config=CalibratorConfig(
                    input_names=tuple(network.get_input(i).name for i in range(network.num_inputs)),
                    cache_path=cache_path,
                    cache_key=calibration_cache_key,
                ),
            )
            self._active_calibrator = calibrator
            config_obj.int8_calibrator = calibrator
            used_int8 = True
            layer_precision_status = self._mark_router_layers_fp16(
                network,
                build.router_fp16_layer_keywords,
                strict=build.strict_precision_constraints,
            )
            if hasattr(trt_mod.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config_obj.set_flag(trt_mod.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            self._active_calibrator = None

        return config_obj, used_fp16, used_int8, layer_precision_status

    def build_from_onnx(
        self,
        *,
        onnx_path: str | Path,
        engine_path: str | Path,
        build: TensorRTEngineBuildConfig | None = None,
        calibration_batches: Sequence[CalibrationBatch] | None = None,
    ) -> EngineBuildResult:
        trt_mod = self._require_trt()
        build_cfg = TensorRTEngineBuildConfig() if build is None else build
        self.load_plugin_libraries()
        plugin_contracts = self._resolve_plugin_contracts(build=build_cfg)
        statuses = self._plugin_statuses(
            plugin_contracts=plugin_contracts,
        )
        self._ensure_plugin_status(statuses, strict=build_cfg.strict_plugin_check)

        onnx_file = Path(onnx_path).expanduser().resolve()
        if not onnx_file.exists():
            raise FileNotFoundError(f"onnx file not found: {onnx_file}")

        builder = trt_mod.Builder(self._trt_logger)
        flags = 1 << int(trt_mod.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt_mod.OnnxParser(network, self._trt_logger)
        onnx_bytes = onnx_file.read_bytes()
        if not parser.parse(onnx_bytes):
            errors = [str(parser.get_error(i)) for i in range(int(parser.num_errors))]
            raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))

        calibration_dataset_version: str | None = None
        calibration_cache_key: str | None = None
        if build_cfg.enable_int8:
            calibration_dataset_version = self._resolve_calibration_dataset_version(
                build=build_cfg,
                calibration_batches=calibration_batches,
            )
        if build_cfg.enable_int8 and build_cfg.calibration_cache_path is not None:
            if build_cfg.calibration_cache_key_override is not None:
                calibration_cache_key = build_cfg.calibration_cache_key_override
            else:
                assert calibration_dataset_version is not None
                calibration_cache_key = build_calibration_cache_key(
                    onnx_sha256=hash_file_sha256(onnx_file),
                    plugin_statuses=statuses,
                    precision_profile=build_cfg.precision_profile,
                    calibration_dataset_version=calibration_dataset_version,
                )

        config_obj, used_fp16, used_int8, layer_precision_status = self._prepare_builder_config(
            builder=builder,
            network=network,
            build=build_cfg,
            calibration_batches=calibration_batches,
            calibration_cache_key=calibration_cache_key,
        )
        serialized = self._build_serialized_network(
            builder=builder,
            network=network,
            config_obj=config_obj,
        )
        out_path = Path(engine_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(serialized)
        return EngineBuildResult(
            engine_path=out_path,
            used_fp16=used_fp16,
            used_int8=used_int8,
            plugin_status=statuses,
            calibration_cache_path=build_cfg.calibration_cache_path,
            calibration_cache_key=calibration_cache_key,
            calibration_dataset_version=calibration_dataset_version,
            layer_precision_status=layer_precision_status,
        )

    def build_from_export_manifest(
        self,
        *,
        manifest_path: str | Path,
        engine_path: str | Path,
        build: TensorRTEngineBuildConfig | None = None,
        calibration_batches: Sequence[CalibrationBatch] | None = None,
        verify_onnx_hash: bool = True,
    ) -> EngineBuildResult:
        manifest = load_export_manifest(
            manifest_path=manifest_path,
            verify_onnx_hash=verify_onnx_hash,
        )
        return self.build_from_onnx(
            onnx_path=manifest.onnx_path,
            engine_path=engine_path,
            build=build,
            calibration_batches=calibration_batches,
        )

    def build_from_network(
        self,
        *,
        network_builder: NetworkBuilderFn,
        engine_path: str | Path,
        build: TensorRTEngineBuildConfig | None = None,
        calibration_batches: Sequence[CalibrationBatch] | None = None,
    ) -> EngineBuildResult:
        trt_mod = self._require_trt()
        build_cfg = TensorRTEngineBuildConfig() if build is None else build
        self.load_plugin_libraries()
        plugin_contracts = self._resolve_plugin_contracts(build=build_cfg)
        statuses = self._plugin_statuses(
            plugin_contracts=plugin_contracts,
        )
        self._ensure_plugin_status(statuses, strict=build_cfg.strict_plugin_check)

        builder = trt_mod.Builder(self._trt_logger)
        flags = 1 << int(trt_mod.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        network_builder(network, trt_mod)
        if int(network.num_outputs) <= 0:
            raise RuntimeError("network_builder did not mark any network outputs")

        calibration_dataset_version: str | None = None
        if build_cfg.enable_int8:
            calibration_dataset_version = self._resolve_calibration_dataset_version(
                build=build_cfg,
                calibration_batches=calibration_batches,
            )

        calibration_cache_key = build_cfg.calibration_cache_key_override
        if (
            build_cfg.enable_int8
            and build_cfg.calibration_cache_path is not None
            and calibration_cache_key is None
        ):
            assert calibration_dataset_version is not None
            calibration_cache_key = build_calibration_cache_key(
                onnx_sha256=self._build_network_definition_signature(network),
                plugin_statuses=statuses,
                precision_profile=build_cfg.precision_profile,
                calibration_dataset_version=calibration_dataset_version,
            )

        config_obj, used_fp16, used_int8, layer_precision_status = self._prepare_builder_config(
            builder=builder,
            network=network,
            build=build_cfg,
            calibration_batches=calibration_batches,
            calibration_cache_key=calibration_cache_key,
        )
        serialized = self._build_serialized_network(
            builder=builder,
            network=network,
            config_obj=config_obj,
        )
        out_path = Path(engine_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(serialized)
        return EngineBuildResult(
            engine_path=out_path,
            used_fp16=used_fp16,
            used_int8=used_int8,
            plugin_status=statuses,
            calibration_cache_path=build_cfg.calibration_cache_path,
            calibration_cache_key=calibration_cache_key,
            calibration_dataset_version=calibration_dataset_version,
            layer_precision_status=layer_precision_status,
        )


__all__ = [
    "ExportManifestInfo",
    "load_export_manifest",
    "NetworkBuilderFn",
    "PluginContract",
    "PluginStatus",
    "TensorRTEngineBuildConfig",
    "build_calibration_cache_key",
    "LayerPrecisionStatus",
    "EngineBuildResult",
    "TensorRTEngineBuilder",
]
