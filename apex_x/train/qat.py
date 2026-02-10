"""INT8 QAT/PTQ helpers for CPU-friendly Apex-X training paths."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as f

QuantMode = Literal["disabled", "qat_int8", "ptq_int8"]
ForwardFn = Callable[[nn.Module, Tensor], Any]


def _as_int8_range(*, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        return -127, 127
    return 0, 255


def _compute_affine_qparams(
    min_val: Tensor,
    max_val: Tensor,
    *,
    symmetric: bool,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, int, int]:
    qmin, qmax = _as_int8_range(symmetric=symmetric)
    min_c = torch.minimum(min_val, max_val)
    max_c = torch.maximum(min_val, max_val)

    if symmetric:
        max_abs = torch.maximum(min_c.abs(), max_c.abs())
        scale = (max_abs / float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale, dtype=torch.int64)
        return scale, zero_point, qmin, qmax

    scale = ((max_c - min_c) / float(qmax - qmin)).clamp_min(eps)
    zero_point = torch.round(torch.tensor(float(qmin), device=scale.device) - min_c / scale).to(
        dtype=torch.int64
    )
    zero_point = zero_point.clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


class ActivationObserver(nn.Module):
    """Simple running min/max observer for activation fake quantization."""

    def __init__(self, *, momentum: float = 0.95) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)
        self.register_buffer("running_min", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("running_max", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def observe(self, x: Tensor) -> None:
        if x.numel() == 0:
            return
        x_detached = x.detach().to(dtype=torch.float32)
        current_min = x_detached.amin()
        current_max = x_detached.amax()
        initialized = cast(Tensor, self.initialized)
        running_min = cast(Tensor, self.running_min)
        running_max = cast(Tensor, self.running_max)
        if not bool(initialized.item()):
            running_min.copy_(current_min)
            running_max.copy_(current_max)
            initialized.fill_(True)
            return

        keep = self.momentum
        update = 1.0 - keep
        running_min.mul_(keep).add_(current_min, alpha=update)
        running_max.mul_(keep).add_(current_max, alpha=update)

    def qparams(self) -> tuple[Tensor, Tensor, int, int]:
        running_min = cast(Tensor, self.running_min)
        running_max = cast(Tensor, self.running_max)
        return _compute_affine_qparams(
            running_min,
            running_max,
            symmetric=False,
        )

    def is_calibrated(self) -> bool:
        initialized = cast(Tensor, self.initialized)
        return bool(initialized.item())


class ActivationFakeQuant(nn.Module):
    """Per-tensor activation fake quant with observer state."""

    def __init__(self, *, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.observer = ActivationObserver(momentum=observer_momentum)
        self.observer_enabled = True
        self.fake_quant_enabled = True

    def set_quant_state(self, *, observer_enabled: bool, fake_quant_enabled: bool) -> None:
        self.observer_enabled = bool(observer_enabled)
        self.fake_quant_enabled = bool(fake_quant_enabled)

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.to(dtype=torch.float32)
        if self.observer_enabled:
            self.observer.observe(x_fp32)
        if (not self.fake_quant_enabled) or (not self.observer.is_calibrated()):
            return x_fp32.to(dtype=x.dtype)

        scale, zero_point, qmin, qmax = self.observer.qparams()
        quantized = torch.fake_quantize_per_tensor_affine(
            x_fp32,
            float(scale.item()),
            int(zero_point.item()),
            qmin,
            qmax,
        )
        return quantized.to(dtype=x.dtype)


class WeightPerChannelFakeQuant(nn.Module):
    """Symmetric per-channel weight fake quantization."""

    def __init__(self, *, channel_axis: int = 0) -> None:
        super().__init__()
        self.channel_axis = int(channel_axis)
        self.fake_quant_enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self.fake_quant_enabled = bool(enabled)

    def forward(self, weight: Tensor) -> Tensor:
        if (not self.fake_quant_enabled) or weight.numel() == 0:
            return weight
        axis = self.channel_axis
        if axis < 0 or axis >= weight.ndim:
            raise ValueError("channel_axis out of range for weight tensor")
        reduce_dims = [idx for idx in range(weight.ndim) if idx != axis]
        weight_fp32 = weight.to(dtype=torch.float32)
        weight_stats = weight_fp32.detach()
        min_vals = weight_stats.amin(dim=reduce_dims)
        max_vals = weight_stats.amax(dim=reduce_dims)
        scale, zero_point, qmin, qmax = _compute_affine_qparams(
            min_vals,
            max_vals,
            symmetric=True,
        )
        zero_point_int = zero_point.to(dtype=torch.int32)
        quantized = torch.fake_quantize_per_channel_affine(
            weight_fp32,
            scale=scale,
            zero_point=zero_point_int,
            axis=axis,
            quant_min=qmin,
            quant_max=qmax,
        )
        return quantized.to(dtype=weight.dtype)


class FakeQuantConv2d(nn.Module):
    """Conv2d wrapper with activation + per-channel weight fake quant."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        self.activation_fake_quant = ActivationFakeQuant()
        self.weight_fake_quant = WeightPerChannelFakeQuant(channel_axis=0)

    def set_quant_state(self, *, observer_enabled: bool, fake_quant_enabled: bool) -> None:
        self.activation_fake_quant.set_quant_state(
            observer_enabled=observer_enabled,
            fake_quant_enabled=fake_quant_enabled,
        )
        self.weight_fake_quant.set_enabled(fake_quant_enabled)

    @property
    def observer_enabled(self) -> bool:
        return self.activation_fake_quant.observer_enabled

    @property
    def fake_quant_enabled(self) -> bool:
        return self.activation_fake_quant.fake_quant_enabled

    def forward(self, x: Tensor) -> Tensor:
        x_q = self.activation_fake_quant(x)
        weight_q = self.weight_fake_quant(self.conv.weight)
        return f.conv2d(
            x_q,
            weight_q,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class FakeQuantLinear(nn.Module):
    """Linear wrapper with activation + per-channel weight fake quant."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear
        self.activation_fake_quant = ActivationFakeQuant()
        self.weight_fake_quant = WeightPerChannelFakeQuant(channel_axis=0)

    def set_quant_state(self, *, observer_enabled: bool, fake_quant_enabled: bool) -> None:
        self.activation_fake_quant.set_quant_state(
            observer_enabled=observer_enabled,
            fake_quant_enabled=fake_quant_enabled,
        )
        self.weight_fake_quant.set_enabled(fake_quant_enabled)

    @property
    def observer_enabled(self) -> bool:
        return self.activation_fake_quant.observer_enabled

    @property
    def fake_quant_enabled(self) -> bool:
        return self.activation_fake_quant.fake_quant_enabled

    def forward(self, x: Tensor) -> Tensor:
        x_q = self.activation_fake_quant(x)
        weight_q = self.weight_fake_quant(self.linear.weight)
        return f.linear(x_q, weight_q, self.linear.bias)


QATWrapper = FakeQuantConv2d | FakeQuantLinear


def iter_qat_wrappers(module: nn.Module) -> Iterator[QATWrapper]:
    for child in module.modules():
        if isinstance(child, (FakeQuantConv2d, FakeQuantLinear)):
            yield child


def set_qat_state(
    module: nn.Module,
    *,
    observer_enabled: bool,
    fake_quant_enabled: bool,
) -> None:
    for wrapped in iter_qat_wrappers(module):
        wrapped.set_quant_state(
            observer_enabled=observer_enabled,
            fake_quant_enabled=fake_quant_enabled,
        )


def _replace_modules_for_fake_quant(
    module: nn.Module,
    *,
    skip_name_contains: Sequence[str],
    prefix: str = "",
) -> tuple[int, tuple[str, ...]]:
    wrapped_paths: list[str] = []
    normalized_skip = tuple(token.lower() for token in skip_name_contains)

    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        lowered_path = path.lower()
        if any(token in lowered_path for token in normalized_skip):
            continue
        if isinstance(child, (FakeQuantConv2d, FakeQuantLinear)):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, FakeQuantConv2d(child))
            wrapped_paths.append(path)
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child))
            wrapped_paths.append(path)
            continue
        nested_count, nested_paths = _replace_modules_for_fake_quant(
            child,
            skip_name_contains=skip_name_contains,
            prefix=path,
        )
        if nested_count > 0:
            wrapped_paths.extend(nested_paths)
    return len(wrapped_paths), tuple(wrapped_paths)


@dataclass(frozen=True, slots=True)
class QuantizationSummary:
    mode: QuantMode
    wrapped_modules: int
    wrapped_paths: tuple[str, ...]
    calibration_batches: int
    router_gating_fp16: bool

    @classmethod
    def disabled(cls) -> QuantizationSummary:
        return cls(
            mode="disabled",
            wrapped_modules=0,
            wrapped_paths=(),
            calibration_batches=0,
            router_gating_fp16=True,
        )


def prepare_int8_qat(
    module: nn.Module,
    *,
    skip_name_contains: Sequence[str] = ("router", "gate", "gating", "kan"),
) -> QuantizationSummary:
    wrapped_count, wrapped_paths = _replace_modules_for_fake_quant(
        module,
        skip_name_contains=skip_name_contains,
    )
    set_qat_state(
        module,
        observer_enabled=True,
        fake_quant_enabled=True,
    )
    return QuantizationSummary(
        mode="qat_int8",
        wrapped_modules=wrapped_count,
        wrapped_paths=wrapped_paths,
        calibration_batches=0,
        router_gating_fp16=True,
    )


def calibrate_ptq(
    module: nn.Module,
    *,
    calibration_inputs: Iterable[Tensor],
    forward_fn: ForwardFn,
) -> int:
    set_qat_state(
        module,
        observer_enabled=True,
        fake_quant_enabled=False,
    )
    count = 0
    was_training = module.training
    module.eval()
    with torch.no_grad():
        for batch in calibration_inputs:
            forward_fn(module, batch)
            count += 1
    if was_training:
        module.train()
    set_qat_state(
        module,
        observer_enabled=False,
        fake_quant_enabled=True,
    )
    return count


def prepare_int8_ptq(
    module: nn.Module,
    *,
    calibration_inputs: Iterable[Tensor],
    forward_fn: ForwardFn,
    skip_name_contains: Sequence[str] = ("router", "gate", "gating", "kan"),
) -> QuantizationSummary:
    wrapped_count, wrapped_paths = _replace_modules_for_fake_quant(
        module,
        skip_name_contains=skip_name_contains,
    )
    calibration_batches = calibrate_ptq(
        module,
        calibration_inputs=calibration_inputs,
        forward_fn=forward_fn,
    )
    return QuantizationSummary(
        mode="ptq_int8",
        wrapped_modules=wrapped_count,
        wrapped_paths=wrapped_paths,
        calibration_batches=calibration_batches,
        router_gating_fp16=True,
    )


__all__ = [
    "QuantMode",
    "QuantizationSummary",
    "ActivationObserver",
    "ActivationFakeQuant",
    "WeightPerChannelFakeQuant",
    "FakeQuantConv2d",
    "FakeQuantLinear",
    "prepare_int8_qat",
    "calibrate_ptq",
    "prepare_int8_ptq",
    "iter_qat_wrappers",
    "set_qat_state",
]
