from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from apex_x.utils.repro import seed_all


def _fp8_dtype_candidates() -> tuple[torch.dtype, ...]:
    dtypes: list[torch.dtype] = []
    for attr_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dtype = getattr(torch, attr_name, None)
        if isinstance(dtype, torch.dtype):
            dtypes.append(dtype)
    return tuple(dtypes)


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return any(dtype is fp8_dtype for fp8_dtype in _fp8_dtype_candidates())


@dataclass(frozen=True, slots=True)
class NumericTolerance:
    atol: float
    rtol: float
    rel_eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.atol < 0.0 or self.rtol < 0.0 or self.rel_eps <= 0.0:
            raise ValueError("tolerances must satisfy atol>=0, rtol>=0, rel_eps>0")


@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    default: NumericTolerance = NumericTolerance(atol=1e-6, rtol=1e-5)
    fp16: NumericTolerance = NumericTolerance(atol=1e-3, rtol=1e-2)
    bf16: NumericTolerance = NumericTolerance(atol=2e-3, rtol=2e-2)
    int8: NumericTolerance = NumericTolerance(atol=0.0, rtol=0.0)
    fp8: NumericTolerance = NumericTolerance(atol=3e-3, rtol=3e-2)

    def for_dtype(self, dtype: torch.dtype) -> NumericTolerance:
        if dtype is torch.float16:
            return self.fp16
        if dtype is torch.bfloat16:
            return self.bf16
        if dtype is torch.int8:
            return self.int8
        if _is_fp8_dtype(dtype):
            return self.fp8
        return self.default


@dataclass(frozen=True, slots=True)
class ParityToleranceProfile:
    name: str
    op_tolerances: ToleranceConfig
    e2e_tolerances: ToleranceConfig
    mismatch_ratio_limit: float = 0.0

    def __post_init__(self) -> None:
        if self.mismatch_ratio_limit < 0.0 or self.mismatch_ratio_limit > 1.0:
            raise ValueError("mismatch_ratio_limit must be in [0, 1]")


_PARITY_TOLERANCE_PROFILES: dict[str, ParityToleranceProfile] = {
    "quality": ParityToleranceProfile(
        name="quality",
        op_tolerances=ToleranceConfig(
            default=NumericTolerance(atol=1e-6, rtol=1e-5),
            fp16=NumericTolerance(atol=1e-3, rtol=1e-2),
            bf16=NumericTolerance(atol=2e-3, rtol=2e-2),
            int8=NumericTolerance(atol=0.0, rtol=0.0),
            fp8=NumericTolerance(atol=3e-3, rtol=3e-2),
        ),
        e2e_tolerances=ToleranceConfig(
            default=NumericTolerance(atol=2e-5, rtol=2e-4),
            fp16=NumericTolerance(atol=2e-3, rtol=2e-2),
            bf16=NumericTolerance(atol=3e-3, rtol=3e-2),
            int8=NumericTolerance(atol=0.0, rtol=0.0),
            fp8=NumericTolerance(atol=4e-3, rtol=4e-2),
        ),
        mismatch_ratio_limit=0.0,
    ),
    "balanced": ParityToleranceProfile(
        name="balanced",
        op_tolerances=ToleranceConfig(
            default=NumericTolerance(atol=5e-6, rtol=5e-5),
            fp16=NumericTolerance(atol=2e-3, rtol=2e-2),
            bf16=NumericTolerance(atol=3e-3, rtol=3e-2),
            int8=NumericTolerance(atol=0.0, rtol=0.0),
            fp8=NumericTolerance(atol=5e-3, rtol=5e-2),
        ),
        e2e_tolerances=ToleranceConfig(
            default=NumericTolerance(atol=5e-5, rtol=5e-4),
            fp16=NumericTolerance(atol=3e-3, rtol=3e-2),
            bf16=NumericTolerance(atol=4e-3, rtol=4e-2),
            int8=NumericTolerance(atol=0.0, rtol=0.0),
            fp8=NumericTolerance(atol=6e-3, rtol=6e-2),
        ),
        mismatch_ratio_limit=1e-4,
    ),
    "edge": ParityToleranceProfile(
        name="edge",
        op_tolerances=ToleranceConfig(
            default=NumericTolerance(atol=1e-5, rtol=1e-4),
            fp16=NumericTolerance(atol=3e-3, rtol=3e-2),
            bf16=NumericTolerance(atol=4e-3, rtol=4e-2),
            int8=NumericTolerance(atol=0.0, rtol=0.0),
            fp8=NumericTolerance(atol=6e-3, rtol=6e-2),
        ),
        e2e_tolerances=ToleranceConfig(
            default=NumericTolerance(atol=8e-5, rtol=8e-4),
            fp16=NumericTolerance(atol=4e-3, rtol=4e-2),
            bf16=NumericTolerance(atol=6e-3, rtol=6e-2),
            int8=NumericTolerance(atol=0.0, rtol=0.0),
            fp8=NumericTolerance(atol=8e-3, rtol=8e-2),
        ),
        mismatch_ratio_limit=5e-4,
    ),
}


def list_parity_tolerance_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PARITY_TOLERANCE_PROFILES))


def get_parity_tolerance_profile(profile_name: str) -> ParityToleranceProfile:
    key = profile_name.strip().lower()
    profile = _PARITY_TOLERANCE_PROFILES.get(key)
    if profile is None:
        supported = ", ".join(list_parity_tolerance_profiles())
        raise ValueError(
            f"Unsupported parity tolerance profile: {profile_name!r}. Supported: {supported}"
        )
    return profile


@dataclass(frozen=True, slots=True)
class TensorParityStats:
    name: str
    reference_dtype: str
    candidate_dtype: str
    shape: tuple[int, ...]
    atol: float
    rtol: float
    max_abs_err: float
    mean_abs_err: float
    max_rel_err: float
    mean_rel_err: float
    mismatch_count: int
    total_count: int
    mismatch_ratio: float
    passed: bool

    def to_dict(self) -> dict[str, str | float | int | bool | list[int]]:
        return {
            "name": self.name,
            "reference_dtype": self.reference_dtype,
            "candidate_dtype": self.candidate_dtype,
            "shape": list(self.shape),
            "atol": self.atol,
            "rtol": self.rtol,
            "max_abs_err": self.max_abs_err,
            "mean_abs_err": self.mean_abs_err,
            "max_rel_err": self.max_rel_err,
            "mean_rel_err": self.mean_rel_err,
            "mismatch_count": self.mismatch_count,
            "total_count": self.total_count,
            "mismatch_ratio": self.mismatch_ratio,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ParityReport:
    case_name: str
    reference_backend: str
    candidate_backend: str
    seed: int
    deterministic: bool
    outputs: tuple[TensorParityStats, ...]
    mismatch_ratio_limit: float
    passed: bool

    @property
    def max_abs_err(self) -> float:
        return max((s.max_abs_err for s in self.outputs), default=0.0)

    @property
    def mean_abs_err(self) -> float:
        total = sum(s.mean_abs_err for s in self.outputs)
        return total / float(len(self.outputs)) if self.outputs else 0.0

    @property
    def max_rel_err(self) -> float:
        return max((s.max_rel_err for s in self.outputs), default=0.0)

    @property
    def mismatch_count(self) -> int:
        return sum(s.mismatch_count for s in self.outputs)

    @property
    def total_count(self) -> int:
        return sum(s.total_count for s in self.outputs)

    @property
    def mismatch_ratio(self) -> float:
        if self.total_count == 0:
            return 0.0
        return float(self.mismatch_count) / float(self.total_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "mismatch_ratio_limit": self.mismatch_ratio_limit,
            "passed": self.passed,
            "max_abs_err": self.max_abs_err,
            "mean_abs_err": self.mean_abs_err,
            "max_rel_err": self.max_rel_err,
            "mismatch_count": self.mismatch_count,
            "total_count": self.total_count,
            "mismatch_ratio": self.mismatch_ratio,
            "outputs": [s.to_dict() for s in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class ParityCase:
    name: str
    input_factory: Callable[[], Any]
    reference_fn: Callable[[Any], Any]
    candidate_fn: Callable[[Any], Any]
    reference_backend: str = "pytorch_ref"
    candidate_backend: str = "candidate"


@dataclass(frozen=True, slots=True)
class ParityMatrixCase:
    name: str
    input_factory: Callable[[], Any]
    backend_fns: Mapping[str, Callable[[Any], Any]]
    backend_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.backend_fns) < 2:
            raise ValueError("ParityMatrixCase requires at least two backends")


@dataclass(frozen=True, slots=True)
class ParitySweepReport:
    sweep_name: str
    profile_name: str
    reports: tuple[ParityReport, ...]

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

    @property
    def case_count(self) -> int:
        return len({report.case_name for report in self.reports})

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_name": self.sweep_name,
            "profile_name": self.profile_name,
            "passed": self.passed,
            "case_count": self.case_count,
            "report_count": len(self.reports),
            "reports": [report.to_dict() for report in self.reports],
        }


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    raise TypeError(f"Unsupported input type for cloning: {type(value)!r}")


def _flatten_tensor_tree(value: Any, *, prefix: str = "out") -> list[tuple[str, Tensor]]:
    if torch.is_tensor(value):
        return [(prefix, value)]
    if isinstance(value, list):
        tensors: list[tuple[str, Tensor]] = []
        for idx, item in enumerate(value):
            tensors.extend(_flatten_tensor_tree(item, prefix=f"{prefix}[{idx}]"))
        return tensors
    if isinstance(value, tuple):
        tensors = []
        for idx, item in enumerate(value):
            tensors.extend(_flatten_tensor_tree(item, prefix=f"{prefix}[{idx}]"))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for key in sorted(value):
            tensors.extend(_flatten_tensor_tree(value[key], prefix=f"{prefix}.{key}"))
        return tensors
    raise TypeError(f"Unsupported output type for parity comparison: {type(value)!r}")


def _resolve_effective_tolerance(
    ref: Tensor, cand: Tensor, tolerances: ToleranceConfig
) -> NumericTolerance:
    # Use candidate dtype by default because candidate precision typically drives expected error.
    tol = tolerances.for_dtype(cand.dtype)
    if tol is tolerances.default:
        tol = tolerances.for_dtype(ref.dtype)
    return tol


def _compute_tensor_stats(
    *,
    name: str,
    reference: Tensor,
    candidate: Tensor,
    tolerance: NumericTolerance,
    mismatch_ratio_limit: float,
) -> TensorParityStats:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch for {name}: reference {tuple(reference.shape)} "
            f"vs candidate {tuple(candidate.shape)}"
        )
    ref64 = reference.detach().to(dtype=torch.float64, device="cpu")
    cand64 = candidate.detach().to(dtype=torch.float64, device="cpu")

    abs_err = (cand64 - ref64).abs()
    rel_denom = ref64.abs().clamp_min(tolerance.rel_eps)
    rel_err = abs_err / rel_denom

    allowed = tolerance.atol + tolerance.rtol * ref64.abs()
    mismatch_mask = abs_err > allowed

    total_count = int(ref64.numel())
    mismatch_count = int(mismatch_mask.sum().item())
    mismatch_ratio = float(mismatch_count) / float(total_count) if total_count > 0 else 0.0
    passed = mismatch_ratio <= mismatch_ratio_limit

    return TensorParityStats(
        name=name,
        reference_dtype=str(reference.dtype).replace("torch.", ""),
        candidate_dtype=str(candidate.dtype).replace("torch.", ""),
        shape=tuple(int(dim) for dim in reference.shape),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
        max_abs_err=float(abs_err.max().item()) if total_count > 0 else 0.0,
        mean_abs_err=float(abs_err.mean().item()) if total_count > 0 else 0.0,
        max_rel_err=float(rel_err.max().item()) if total_count > 0 else 0.0,
        mean_rel_err=float(rel_err.mean().item()) if total_count > 0 else 0.0,
        mismatch_count=mismatch_count,
        total_count=total_count,
        mismatch_ratio=mismatch_ratio,
        passed=passed,
    )


def evaluate_parity_outputs(
    *,
    case_name: str,
    reference_backend: str,
    candidate_backend: str,
    reference_output: Any,
    candidate_output: Any,
    seed: int = 0,
    deterministic: bool = True,
    tolerances: ToleranceConfig | None = None,
    mismatch_ratio_limit: float = 0.0,
) -> ParityReport:
    if mismatch_ratio_limit < 0.0 or mismatch_ratio_limit > 1.0:
        raise ValueError("mismatch_ratio_limit must be in [0, 1]")
    tolerance_cfg = tolerances or ToleranceConfig()

    ref_tensors = _flatten_tensor_tree(reference_output, prefix="out")
    cand_tensors = _flatten_tensor_tree(candidate_output, prefix="out")
    if len(ref_tensors) != len(cand_tensors):
        raise ValueError(
            "Output tensor count mismatch: "
            f"reference={len(ref_tensors)} candidate={len(cand_tensors)}"
        )

    stats: list[TensorParityStats] = []
    for (ref_name, ref_tensor), (cand_name, cand_tensor) in zip(
        ref_tensors, cand_tensors, strict=True
    ):
        if ref_name != cand_name:
            raise ValueError(f"Output structure mismatch: {ref_name} != {cand_name}")
        tol = _resolve_effective_tolerance(ref_tensor, cand_tensor, tolerance_cfg)
        stat = _compute_tensor_stats(
            name=ref_name,
            reference=ref_tensor,
            candidate=cand_tensor,
            tolerance=tol,
            mismatch_ratio_limit=mismatch_ratio_limit,
        )
        stats.append(stat)

    outputs = tuple(stats)
    passed = all(stat.passed for stat in outputs)
    return ParityReport(
        case_name=case_name,
        reference_backend=reference_backend,
        candidate_backend=candidate_backend,
        seed=seed,
        deterministic=deterministic,
        outputs=outputs,
        mismatch_ratio_limit=mismatch_ratio_limit,
        passed=passed,
    )


def run_parity_case(
    case: ParityCase,
    *,
    seed: int = 0,
    deterministic: bool = True,
    tolerances: ToleranceConfig | None = None,
    mismatch_ratio_limit: float = 0.0,
) -> ParityReport:
    seed_all(seed=seed, deterministic=deterministic)
    inputs = case.input_factory()
    ref_out = case.reference_fn(_clone_tree(inputs))
    cand_out = case.candidate_fn(_clone_tree(inputs))
    return evaluate_parity_outputs(
        case_name=case.name,
        reference_backend=case.reference_backend,
        candidate_backend=case.candidate_backend,
        reference_output=ref_out,
        candidate_output=cand_out,
        seed=seed,
        deterministic=deterministic,
        tolerances=tolerances,
        mismatch_ratio_limit=mismatch_ratio_limit,
    )


def _resolve_backend_pairs(
    *,
    backend_names: tuple[str, ...],
    requested_pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    known = set(backend_names)
    if requested_pairs:
        resolved: list[tuple[str, str]] = []
        for reference_backend, candidate_backend in requested_pairs:
            if reference_backend == candidate_backend:
                raise ValueError("backend pairs must compare different backend names")
            if reference_backend not in known:
                raise ValueError(
                    f"Unknown backend in pair: {reference_backend!r}. Known: {sorted(known)!r}"
                )
            if candidate_backend not in known:
                raise ValueError(
                    f"Unknown backend in pair: {candidate_backend!r}. Known: {sorted(known)!r}"
                )
            resolved.append((reference_backend, candidate_backend))
        return tuple(resolved)

    auto_pairs: list[tuple[str, str]] = []
    for left_idx in range(len(backend_names)):
        for right_idx in range(left_idx + 1, len(backend_names)):
            auto_pairs.append((backend_names[left_idx], backend_names[right_idx]))
    return tuple(auto_pairs)


def run_parity_matrix_case(
    case: ParityMatrixCase,
    *,
    seed: int = 0,
    deterministic: bool = True,
    tolerances: ToleranceConfig | None = None,
    mismatch_ratio_limit: float = 0.0,
) -> tuple[ParityReport, ...]:
    backend_names = tuple(case.backend_fns.keys())
    backend_pairs = _resolve_backend_pairs(
        backend_names=backend_names,
        requested_pairs=case.backend_pairs,
    )

    seed_all(seed=seed, deterministic=deterministic)
    inputs = case.input_factory()
    backend_outputs: dict[str, Any] = {}
    for backend_name in backend_names:
        backend_outputs[backend_name] = case.backend_fns[backend_name](_clone_tree(inputs))

    reports: list[ParityReport] = []
    for reference_backend, candidate_backend in backend_pairs:
        reports.append(
            evaluate_parity_outputs(
                case_name=case.name,
                reference_backend=reference_backend,
                candidate_backend=candidate_backend,
                reference_output=backend_outputs[reference_backend],
                candidate_output=backend_outputs[candidate_backend],
                seed=seed,
                deterministic=deterministic,
                tolerances=tolerances,
                mismatch_ratio_limit=mismatch_ratio_limit,
            )
        )
    return tuple(reports)


def run_parity_sweep(
    *,
    sweep_name: str,
    cases: Sequence[ParityMatrixCase],
    profile_name: str = "balanced",
    use_e2e_tolerances: bool = True,
    seed: int = 0,
    deterministic: bool = True,
) -> ParitySweepReport:
    if not cases:
        raise ValueError("cases must be non-empty")

    profile = get_parity_tolerance_profile(profile_name)
    tolerances = profile.e2e_tolerances if use_e2e_tolerances else profile.op_tolerances
    mismatch_ratio_limit = profile.mismatch_ratio_limit

    reports: list[ParityReport] = []
    for case_idx, case in enumerate(cases):
        case_seed = seed + case_idx
        reports.extend(
            run_parity_matrix_case(
                case,
                seed=case_seed,
                deterministic=deterministic,
                tolerances=tolerances,
                mismatch_ratio_limit=mismatch_ratio_limit,
            )
        )

    return ParitySweepReport(
        sweep_name=sweep_name,
        profile_name=profile.name,
        reports=tuple(reports),
    )


def format_parity_report(report: ParityReport) -> str:
    lines = [
        (
            f"case={report.case_name} ref={report.reference_backend} "
            f"cand={report.candidate_backend} pass={report.passed} "
            f"mismatch={report.mismatch_count}/{report.total_count} "
            f"ratio={report.mismatch_ratio:.6f}"
        ),
    ]
    for stat in report.outputs:
        lines.append(
            f"  {stat.name}: pass={stat.passed} "
            f"max_abs={stat.max_abs_err:.6e} mean_abs={stat.mean_abs_err:.6e} "
            f"max_rel={stat.max_rel_err:.6e} mean_rel={stat.mean_rel_err:.6e} "
            f"mismatch={stat.mismatch_count}/{stat.total_count} "
            f"tol(atol={stat.atol:.3e},rtol={stat.rtol:.3e})"
        )
    return "\n".join(lines)


def format_parity_sweep_report(report: ParitySweepReport) -> str:
    lines = [
        (
            f"sweep={report.sweep_name} profile={report.profile_name} "
            f"pass={report.passed} cases={report.case_count} reports={len(report.reports)}"
        ),
    ]
    for item in report.reports:
        lines.append(format_parity_report(item))
    return "\n".join(lines)


__all__ = [
    "NumericTolerance",
    "ToleranceConfig",
    "ParityToleranceProfile",
    "list_parity_tolerance_profiles",
    "get_parity_tolerance_profile",
    "TensorParityStats",
    "ParityReport",
    "ParityCase",
    "ParityMatrixCase",
    "ParitySweepReport",
    "evaluate_parity_outputs",
    "run_parity_case",
    "run_parity_matrix_case",
    "run_parity_sweep",
    "format_parity_report",
    "format_parity_sweep_report",
]
