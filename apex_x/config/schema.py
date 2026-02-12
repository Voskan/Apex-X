from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

PROFILE_PRESETS: dict[str, dict[str, tuple[int, ...]]] = {
    "nano": {
        "pv_channels": (64, 128, 192),
        "pv_strides": (8, 16, 32),
        "ff_channels": (96, 160),
        "ff_strides": (8, 4),
    },
    "small": {
        "pv_channels": (80, 160, 256),
        "pv_strides": (8, 16, 32),
        "ff_channels": (128, 224),
        "ff_strides": (8, 4),
    },
    "base": {
        "pv_channels": (96, 192, 320),
        "pv_strides": (8, 16, 32),
        "ff_channels": (160, 288),
        "ff_strides": (8, 4),
    },
    "large": {
        "pv_channels": (128, 256, 384),
        "pv_strides": (8, 16, 32),
        "ff_channels": (192, 352),
        "ff_strides": (8, 4),
    },
}


@dataclass(slots=True)
class ModelConfig:
    profile: str = "small"
    pv_channels: tuple[int, ...] = ()
    pv_strides: tuple[int, ...] = ()
    ff_channels: tuple[int, ...] = ()
    ff_strides: tuple[int, ...] = ()

    pv_stride: int = 16
    ff_primary_stride: int = 8

    tile_size_l0: int = 16
    tile_size_l1: int = 8
    tile_size_l2: int = 4

    kmax_l0: int = 32
    kmax_l1: int = 64
    kmax_l2: int = 0
    nesting_depth: int = 1
    force_dense_routing: bool = False
    disable_nesting: bool = False
    disable_ssm: bool = False

    input_height: int = 128
    input_width: int = 128

    def __post_init__(self) -> None:
        self.profile = self.profile.lower()
        preset = PROFILE_PRESETS.get(self.profile)
        if preset is None:
            valid = ", ".join(sorted(PROFILE_PRESETS.keys()))
            raise ValueError(f"model.profile must be one of: {valid}")

        if not self.pv_channels:
            self.pv_channels = preset["pv_channels"]
        if not self.pv_strides:
            self.pv_strides = preset["pv_strides"]
        if not self.ff_channels:
            self.ff_channels = preset["ff_channels"]
        if not self.ff_strides:
            self.ff_strides = preset["ff_strides"]

    def effective_nesting_depth(self) -> int:
        return 0 if self.disable_nesting else self.nesting_depth

    def router_enabled(self) -> bool:
        return not self.force_dense_routing

    def ssm_enabled(self) -> bool:
        return not self.disable_ssm

    def validate(self) -> None:
        if self.nesting_depth not in (0, 1, 2):
            raise ValueError("model.nesting_depth must be 0, 1, or 2")
        nesting_depth = self.effective_nesting_depth()

        if self.pv_stride <= 0 or self.ff_primary_stride <= 0:
            raise ValueError("model.pv_stride and model.ff_primary_stride must be positive")

        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("model.input_height and model.input_width must be positive")

        if self.input_height % self.ff_primary_stride != 0:
            raise ValueError("model.input_height must be divisible by model.ff_primary_stride")
        if self.input_width % self.ff_primary_stride != 0:
            raise ValueError("model.input_width must be divisible by model.ff_primary_stride")

        if self.tile_size_l0 <= 0 or self.tile_size_l1 <= 0 or self.tile_size_l2 <= 0:
            raise ValueError("model.tile_size_l0/l1/l2 must be positive")

        if nesting_depth >= 1 and self.tile_size_l0 != 2 * self.tile_size_l1:
            raise ValueError("model.tile_size_l1 must be exactly model.tile_size_l0/2")
        if nesting_depth >= 2 and self.tile_size_l1 != 2 * self.tile_size_l2:
            raise ValueError("model.tile_size_l2 must be exactly model.tile_size_l1/2")

        h_ff = self.input_height // self.ff_primary_stride
        w_ff = self.input_width // self.ff_primary_stride
        if h_ff % self.tile_size_l0 != 0 or w_ff % self.tile_size_l0 != 0:
            raise ValueError("FF map size must be divisible by model.tile_size_l0")
        if nesting_depth >= 1 and (h_ff % self.tile_size_l1 != 0 or w_ff % self.tile_size_l1 != 0):
            raise ValueError("FF map size must be divisible by model.tile_size_l1")
        if nesting_depth >= 2 and (h_ff % self.tile_size_l2 != 0 or w_ff % self.tile_size_l2 != 0):
            raise ValueError("FF map size must be divisible by model.tile_size_l2")

        if self.kmax_l0 <= 0:
            raise ValueError("model.kmax_l0 must be > 0")
        if (
            nesting_depth == 0
            and not self.disable_nesting
            and (self.kmax_l1 != 0 or self.kmax_l2 != 0)
        ):
            raise ValueError("model.kmax_l1 and model.kmax_l2 must be 0 when nesting_depth=0")
        if nesting_depth == 1:
            if self.kmax_l1 <= 0:
                raise ValueError("model.kmax_l1 must be > 0 when nesting_depth=1")
            if self.kmax_l2 != 0:
                raise ValueError("model.kmax_l2 must be 0 when nesting_depth=1")
        if nesting_depth == 2 and (self.kmax_l1 <= 0 or self.kmax_l2 <= 0):
            raise ValueError("model.kmax_l1 and model.kmax_l2 must be > 0 when nesting_depth=2")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        parsed = dict(data)
        for key in ("pv_channels", "pv_strides", "ff_channels", "ff_strides"):
            if key in parsed:
                parsed[key] = tuple(int(v) for v in parsed[key])
        return cls(**parsed)


@dataclass(slots=True)
class RoutingConfig:
    budget_total: float = 32.0
    budget_b1: float = 16.0
    budget_b2: float = 8.0
    budget_b3: float = 0.0

    cost_heavy: float = 1.0
    cost_cheap: float = 0.2

    theta_on: float = 0.6
    theta_off: float = 0.45
    theta_split: float = 0.5

    def validate(self) -> None:
        if self.budget_total <= 0:
            raise ValueError("routing.budget_total must be > 0")
        if self.budget_b1 < 0 or self.budget_b2 < 0 or self.budget_b3 < 0:
            raise ValueError("routing.budget_b1/b2/b3 must be >= 0")
        if self.budget_b1 + self.budget_b2 + self.budget_b3 > self.budget_total + 1e-9:
            raise ValueError("routing budgets b1+b2+b3 must be <= budget_total")

        if self.cost_heavy <= self.cost_cheap:
            raise ValueError("routing.cost_heavy must be > routing.cost_cheap")
        if self.cost_cheap < 0:
            raise ValueError("routing.cost_cheap must be >= 0")

        for name, value in (
            ("theta_on", self.theta_on),
            ("theta_off", self.theta_off),
            ("theta_split", self.theta_split),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"routing.{name} must be within [0, 1]")
        if self.theta_on <= self.theta_off:
            raise ValueError("routing.theta_on must be > routing.theta_off")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutingConfig:
        return cls(**dict(data))


@dataclass(slots=True)
class TrainConfig:
    curriculum_stages: tuple[str, ...] = ("det_stability", "seg_boundary")

    mu_init: float = 0.1
    mu_lr: float = 0.01
    mu_min: float = 0.0
    mu_max: float = 10.0
    dual_adaptive_lr: bool = False
    dual_lr_decay: float = 0.0
    dual_delta_clip: float | None = None
    dual_deadband_ratio: float = 0.0
    dual_error_ema_beta: float = 0.9
    dual_lr_min_scale: float = 0.5
    dual_lr_max_scale: float = 3.0

    distill_weight: float = 1.0
    seg_boundary_weight: float = 1.0

    use_pcgradpp: bool = True
    disable_distill: bool = False
    disable_pcgradpp: bool = False

    output_dir: str = "artifacts/train_output"
    save_interval: int = 1

    qat_enable: bool = False
    qat_int8: bool = False
    qat_fp8: bool = False

    # Loss configuration
    box_loss_type: str = "mpdiou"  # iou, giou, diou, ciou, mpdiou

    # Performance optimizations
    torch_compile: bool = False
    tf32_enabled: bool = True
    cudnn_benchmark: bool = True
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True
    enable_lora_finetune: bool = False
    
    # Advanced Training
    auto_batch_size: bool = False
    swa_enabled: bool = False
    swa_lr: float = 0.05
    swa_start_epoch: int = 5
    tta_enabled: bool = False

    def validate(self) -> None:
        if not self.curriculum_stages:
            raise ValueError("train.curriculum_stages must not be empty")

        if self.mu_lr <= 0:
            raise ValueError("train.mu_lr must be > 0")
        if self.mu_min < 0:
            raise ValueError("train.mu_min must be >= 0")
        if self.mu_min > self.mu_max:
            raise ValueError("train.mu_min must be <= train.mu_max")
        if not (self.mu_min <= self.mu_init <= self.mu_max):
            raise ValueError("train.mu_init must be in [mu_min, mu_max]")
        if self.dual_lr_decay < 0:
            raise ValueError("train.dual_lr_decay must be >= 0")
        if self.dual_delta_clip is not None and self.dual_delta_clip <= 0:
            raise ValueError("train.dual_delta_clip must be > 0 when provided")
        if not (0.0 <= self.dual_deadband_ratio < 1.0):
            raise ValueError("train.dual_deadband_ratio must be in [0, 1)")
        if not (0.0 <= self.dual_error_ema_beta < 1.0):
            raise ValueError("train.dual_error_ema_beta must be in [0, 1)")
        if self.dual_lr_min_scale <= 0:
            raise ValueError("train.dual_lr_min_scale must be > 0")
        if self.dual_lr_max_scale < self.dual_lr_min_scale:
            raise ValueError("train.dual_lr_max_scale must be >= train.dual_lr_min_scale")

        if self.distill_weight < 0 or self.seg_boundary_weight < 0:
            raise ValueError("train distill and seg_boundary weights must be >= 0")

        if self.save_interval <= 0:
            raise ValueError("train.save_interval must be > 0")

        if self.dataloader_num_workers < 0:
            raise ValueError("train.dataloader_num_workers must be >= 0")

        if (self.qat_int8 or self.qat_fp8) and not self.qat_enable:
            raise ValueError("train.qat_enable must be true when qat_int8 or qat_fp8 is enabled")

        if self.box_loss_type not in {"iou", "giou", "diou", "ciou", "mpdiou"}:
             raise ValueError("train.box_loss_type must be iou, giou, diou, ciou, or mpdiou")

        if self.swa_enabled:
            if self.swa_lr <= 0:
                raise ValueError("train.swa_lr must be > 0")
            if self.swa_start_epoch < 0:
                raise ValueError("train.swa_start_epoch must be >= 0")

    def distill_enabled(self) -> bool:
        return (not self.disable_distill) and self.distill_weight > 0.0

    def pcgradpp_enabled(self) -> bool:
        return self.use_pcgradpp and (not self.disable_pcgradpp)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        parsed = dict(data)
        if "curriculum_stages" in parsed:
            parsed["curriculum_stages"] = tuple(str(v) for v in parsed["curriculum_stages"])
        return cls(**parsed)


@dataclass(slots=True)
class DataConfig:
    coco_train_images: str = ""
    coco_train_annotations: str = ""
    coco_val_images: str = ""
    coco_val_annotations: str = ""

    flip_prob: float = 0.5
    hsv_prob: float = 0.5
    mosaic_prob: float = 0.0
    scale_min: float = 0.8
    scale_max: float = 1.2

    def validate(self) -> None:
        for name, value in (
            ("flip_prob", self.flip_prob),
            ("hsv_prob", self.hsv_prob),
            ("mosaic_prob", self.mosaic_prob),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"data.{name} must be within [0, 1]")

        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("data.scale_min and data.scale_max must be > 0")
        if self.scale_min > self.scale_max:
            raise ValueError("data.scale_min must be <= data.scale_max")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataConfig:
        return cls(**dict(data))


@dataclass(slots=True)
class RuntimeConfig:
    backend: str = "cpu"
    fallback_policy: str = "permissive"
    precision_profile: str = "balanced"
    enable_export: bool = True
    enable_runtime_plugins: bool = False
    export_format: str = "onnx"
    trt_enable: bool = False
    ort_enable: bool = True
    deterministic: bool = True

    def validate(self) -> None:
        if self.backend not in {"cpu", "torch", "triton", "tensorrt"}:
            raise ValueError("runtime.backend must be cpu, torch, triton, or tensorrt")
        if self.fallback_policy not in {"strict", "permissive"}:
            raise ValueError("runtime.fallback_policy must be strict or permissive")
        if self.precision_profile not in {"quality", "balanced", "edge"}:
            raise ValueError("runtime.precision_profile must be quality, balanced, or edge")
        if self.export_format not in {"onnx", "torchscript"}:
            raise ValueError("runtime.export_format must be onnx or torchscript")

        if self.trt_enable and not self.enable_runtime_plugins:
            raise ValueError("runtime.enable_runtime_plugins must be true when trt_enable is true")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeConfig:
        parsed = dict(data)
        if "backend" in parsed:
            parsed["backend"] = str(parsed["backend"]).lower()
        if "fallback_policy" in parsed:
            parsed["fallback_policy"] = str(parsed["fallback_policy"]).lower()
        if "precision_profile" in parsed:
            parsed["precision_profile"] = str(parsed["precision_profile"]).lower()
        if "export_format" in parsed:
            parsed["export_format"] = str(parsed["export_format"]).lower()
        return cls(**parsed)


@dataclass(slots=True)
class ApexXConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.model.validate()
        self.routing.validate()
        self.train.validate()
        self.data.validate()
        self.runtime.validate()

        if (
            not self.model.disable_nesting
            and self.model.effective_nesting_depth() < 2
            and self.routing.budget_b3 != 0
        ):
            raise ValueError("routing.budget_b3 must be 0 unless model.nesting_depth=2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ApexXConfig:
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            routing=RoutingConfig.from_dict(data.get("routing", {})),
            train=TrainConfig.from_dict(data.get("train", {})),
            data=DataConfig.from_dict(data.get("data", {})),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {})),
        )
