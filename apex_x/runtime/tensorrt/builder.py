from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from apex_x.runtime.caps import RuntimeCaps, detect_runtime_caps
from apex_x.utils.logging import get_logger

from .calibrator import CalibrationBatch, CalibratorConfig, TensorRTEntropyCalibrator

try:
    import tensorrt as trt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    trt = None


NetworkBuilderFn = Callable[[Any, Any], None]


@dataclass(frozen=True, slots=True)
class PluginStatus:
    name: str
    required: bool
    found: bool


@dataclass(frozen=True, slots=True)
class TensorRTEngineBuildConfig:
    max_workspace_size_bytes: int = 1 << 30
    enable_fp16: bool = True
    enable_int8: bool = False
    optimization_level: int | None = None
    router_fp16_layer_keywords: tuple[str, ...] = ("router", "kan")
    strict_plugin_check: bool = True
    include_optional_nms_plugin: bool = False
    expected_plugins: tuple[str, ...] = ("TilePack", "TileSSMScan", "TileUnpackFusion")
    calibration_cache_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EngineBuildResult:
    engine_path: Path
    used_fp16: bool
    used_int8: bool
    plugin_status: tuple[PluginStatus, ...]
    calibration_cache_path: Path | None


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
        include_optional_nms: bool,
        expected_plugins: Sequence[str],
    ) -> tuple[PluginStatus, ...]:
        trt_mod = self._require_trt()
        registry = trt_mod.get_plugin_registry()
        creator_names = {creator.name for creator in registry.plugin_creator_list}
        statuses: list[PluginStatus] = []
        for name in expected_plugins:
            statuses.append(
                PluginStatus(
                    name=name,
                    required=True,
                    found=name in creator_names,
                )
            )
        if include_optional_nms:
            statuses.append(
                PluginStatus(
                    name="DecodeNMS",
                    required=False,
                    found="DecodeNMS" in creator_names,
                )
            )
        return tuple(statuses)

    def _ensure_plugin_status(
        self,
        statuses: Sequence[PluginStatus],
        *,
        strict: bool,
    ) -> None:
        missing_required = [
            status.name for status in statuses if status.required and not status.found
        ]
        if missing_required and strict:
            raise RuntimeError(f"missing required TensorRT plugins: {', '.join(missing_required)}")
        if missing_required:
            self._logger.warning("missing_required_tensorrt_plugins plugins=%s", missing_required)

    def _set_memory_pool_limit(self, config_obj: Any, workspace_bytes: int) -> None:
        if workspace_bytes <= 0:
            raise ValueError("max_workspace_size_bytes must be > 0")
        if hasattr(config_obj, "set_memory_pool_limit"):
            config_obj.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
            return
        config_obj.max_workspace_size = int(workspace_bytes)

    def _mark_router_layers_fp16(self, network: Any, layer_keywords: Sequence[str]) -> None:
        if not layer_keywords:
            return
        for index in range(int(network.num_layers)):
            layer = network.get_layer(index)
            name = (layer.name or "").lower()
            if not any(keyword.lower() in name for keyword in layer_keywords):
                continue
            if hasattr(layer, "precision"):
                layer.precision = trt.float16
            if hasattr(layer, "set_output_type"):
                for output_idx in range(int(layer.num_outputs)):
                    layer.set_output_type(output_idx, trt.float16)

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
    ) -> tuple[Any, bool, bool]:
        trt_mod = self._require_trt()
        config_obj = builder.create_builder_config()
        self._set_memory_pool_limit(config_obj, build.max_workspace_size_bytes)
        if build.optimization_level is not None and hasattr(
            config_obj, "builder_optimization_level"
        ):
            config_obj.builder_optimization_level = int(build.optimization_level)

        used_fp16 = False
        used_int8 = False
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
                ),
            )
            self._active_calibrator = calibrator
            config_obj.int8_calibrator = calibrator
            used_int8 = True
            self._mark_router_layers_fp16(network, build.router_fp16_layer_keywords)
            if hasattr(trt_mod.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config_obj.set_flag(trt_mod.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            self._active_calibrator = None

        return config_obj, used_fp16, used_int8

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
        statuses = self._plugin_statuses(
            include_optional_nms=build_cfg.include_optional_nms_plugin,
            expected_plugins=build_cfg.expected_plugins,
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

        config_obj, used_fp16, used_int8 = self._prepare_builder_config(
            builder=builder,
            network=network,
            build=build_cfg,
            calibration_batches=calibration_batches,
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
        statuses = self._plugin_statuses(
            include_optional_nms=build_cfg.include_optional_nms_plugin,
            expected_plugins=build_cfg.expected_plugins,
        )
        self._ensure_plugin_status(statuses, strict=build_cfg.strict_plugin_check)

        builder = trt_mod.Builder(self._trt_logger)
        flags = 1 << int(trt_mod.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        network_builder(network, trt_mod)
        if int(network.num_outputs) <= 0:
            raise RuntimeError("network_builder did not mark any network outputs")

        config_obj, used_fp16, used_int8 = self._prepare_builder_config(
            builder=builder,
            network=network,
            build=build_cfg,
            calibration_batches=calibration_batches,
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
        )


__all__ = [
    "NetworkBuilderFn",
    "PluginStatus",
    "TensorRTEngineBuildConfig",
    "EngineBuildResult",
    "TensorRTEngineBuilder",
]
