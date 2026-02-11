from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from apex_x.cli import app

runner = CliRunner()
FIXTURE_CFG = Path(__file__).parent / "fixtures" / "apex_x_config.yaml"
EVAL_FIXTURE = Path(__file__).parent / "fixtures" / "eval_tiny_fixture.json"


def test_train_with_overrides_parses() -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(FIXTURE_CFG),
            "--set",
            "model.profile=large",
            "--set",
            "routing.budget_b1=22",
        ],
    )
    assert result.exit_code == 0
    assert "train ok" in result.stdout
    assert "profile=large" in result.stdout
    assert "budget_b1=22" in result.stdout
    assert "stage_count=5" in result.stdout


def test_eval_command_parses() -> None:
    result = runner.invoke(app, ["eval", "--config", str(FIXTURE_CFG)])
    assert result.exit_code == 0
    assert "eval ok" in result.stdout


def test_eval_command_reports_backend_selection() -> None:
    result = runner.invoke(
        app,
        ["eval", "--config", str(FIXTURE_CFG), "--backend", "cpu", "--fallback-policy", "strict"],
    )
    assert result.exit_code == 0
    assert "eval ok" in result.stdout
    assert "backend=cpu" in result.stdout
    assert "selected_backend=cpu" in result.stdout
    assert "requested_backend=cpu" in result.stdout
    assert "fallback_reason=none" in result.stdout


def test_eval_command_panoptic_pq_hook_parses() -> None:
    result = runner.invoke(app, ["eval", "--config", str(FIXTURE_CFG), "--panoptic-pq"])
    assert result.exit_code == 0
    assert "eval ok" in result.stdout
    assert "panoptic_pq=" in result.stdout


def test_eval_command_emits_reports_from_fixture(tmp_path: Path) -> None:
    report_json = tmp_path / "eval_report.json"
    report_md = tmp_path / "eval_report.md"
    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(FIXTURE_CFG),
            "--fixture",
            str(EVAL_FIXTURE),
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ],
    )
    assert result.exit_code == 0
    assert "eval ok" in result.stdout
    assert "det_map=" in result.stdout
    assert "mask_map=" in result.stdout
    assert "miou=" in result.stdout
    assert "panoptic_pq=" in result.stdout
    assert report_json.exists()
    assert report_md.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert "runtime" in payload
    assert "execution_backend" in payload["runtime"]
    assert "latency_ms" in payload["runtime"]
    assert "total" in payload["runtime"]["latency_ms"]


def test_eval_command_dataset_npz_executes_model_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval_images.npz"
    images = np.random.RandomState(321).rand(3, 3, 256, 256).astype(np.float32)
    np.savez(dataset_path, images=images)

    report_json = tmp_path / "eval_report.json"
    report_md = tmp_path / "eval_report.md"
    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(FIXTURE_CFG),
            "--dataset-npz",
            str(dataset_path),
            "--max-samples",
            "2",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ],
    )
    assert result.exit_code == 0
    assert "eval ok" in result.stdout
    assert "model_eval_samples=2" in result.stdout
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert "model_eval" in payload
    assert payload["model_eval"]["num_samples"] == 2


def test_eval_command_dataset_npz_with_target_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eval_images_with_target.npz"
    images = np.random.RandomState(654).rand(3, 3, 256, 256).astype(np.float32)
    det_score_target = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    selected_tiles_target = np.asarray([2, 2, 1], dtype=np.int64)
    np.savez(
        dataset_path,
        images=images,
        det_score_target=det_score_target,
        selected_tiles_target=selected_tiles_target,
    )

    report_json = tmp_path / "eval_report_target.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(FIXTURE_CFG),
            "--dataset-npz",
            str(dataset_path),
            "--max-samples",
            "3",
            "--report-json",
            str(report_json),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert "model_eval" in payload
    assert "det_score_target" in payload["model_eval"]
    assert "mae" in payload["model_eval"]["det_score_target"]
    assert "selected_tiles_target" in payload["model_eval"]
    assert "exact_match_rate" in payload["model_eval"]["selected_tiles_target"]


def test_predict_command_parses() -> None:
    result = runner.invoke(app, ["predict", "--config", str(FIXTURE_CFG)])
    assert result.exit_code == 0
    assert "predict ok" in result.stdout


def test_predict_command_writes_json_report(tmp_path: Path) -> None:
    report_path = tmp_path / "predict_report.json"
    result = runner.invoke(
        app,
        [
            "predict",
            "--config",
            str(FIXTURE_CFG),
            "--report-json",
            str(report_path),
        ],
    )
    assert result.exit_code == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "runtime" in payload
    assert "latency_ms" in payload["runtime"]
    assert "total" in payload["runtime"]["latency_ms"]
    assert "selected_tiles" in payload
    assert "det_score" in payload
    assert "routing_diagnostics" in payload


def test_predict_command_backend_permissive_falls_back_to_cpu() -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--config",
            str(FIXTURE_CFG),
            "--backend",
            "tensorrt",
            "--fallback-policy",
            "permissive",
        ],
    )
    assert result.exit_code == 0
    assert "predict ok" in result.stdout
    assert "backend=cpu" in result.stdout
    assert "requested_backend=tensorrt" in result.stdout
    assert "fallback_reason=" in result.stdout
    assert ("fallback_reason=none" not in result.stdout) or (
        "exec_fallback=none" not in result.stdout
    )


def test_predict_command_backend_strict_fails_when_backend_unavailable() -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--config",
            str(FIXTURE_CFG),
            "--backend",
            "triton",
            "--fallback-policy",
            "strict",
        ],
    )
    assert result.exit_code != 0
    assert (
        "requested backend 'triton' is unavailable" in result.stdout
        or "execution path is not implemented in CLI runtime" in result.stdout
        or "triton backend is unavailable for execution" in result.stdout
    )


def test_predict_command_invalid_backend_fails() -> None:
    result = runner.invoke(
        app,
        ["predict", "--config", str(FIXTURE_CFG), "--backend", "metal"],
    )
    assert result.exit_code != 0
    assert "invalid backend" in result.stdout


def test_predict_command_tensorrt_backend_strict_fails() -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--config",
            str(FIXTURE_CFG),
            "--backend",
            "tensorrt",
            "--fallback-policy",
            "strict",
        ],
    )
    assert result.exit_code != 0
    assert (
        "requested backend 'tensorrt' is unavailable" in result.stdout
        or "tensorrt_preflight_failed" in result.stdout
        or "tensorrt_runtime_execution_failed" in result.stdout
    )


def test_bench_command_parses() -> None:
    result = runner.invoke(app, ["bench", "--config", str(FIXTURE_CFG), "--iters", "3"])
    assert result.exit_code == 0
    assert "bench ok" in result.stdout
    assert "iters=3" in result.stdout


def test_ablate_command_parses(tmp_path: Path) -> None:
    report_csv = tmp_path / "ablation.csv"
    report_md = tmp_path / "ablation.md"
    result = runner.invoke(
        app,
        [
            "ablate",
            "--config",
            str(FIXTURE_CFG),
            "--name",
            "split_ablation",
            "--max-experiments",
            "1",
            "--seed",
            "0",
            "--steps-per-stage",
            "1",
            "--output-csv",
            str(report_csv),
            "--output-md",
            str(report_md),
            "--set",
            "routing.theta_on=0.9",
        ],
    )
    assert result.exit_code == 0
    assert "ablate ok" in result.stdout
    assert "name=split_ablation" in result.stdout
    assert "experiments=1" in result.stdout
    assert report_csv.exists()
    assert report_md.exists()


def test_export_command_writes_artifact(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    out = tmp_path / "artifact.json"
    result = runner.invoke(
        app,
        [
            "export",
            "--config",
            str(FIXTURE_CFG),
            "--output",
            str(out),
            "--shape-mode",
            "dynamic",
            "--set",
            "runtime.precision_profile='quality'",
        ],
    )
    assert result.exit_code == 0
    assert "export ok" in result.stdout
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "onnx"
    assert payload["shape_mode"] == "dynamic"
    assert Path(payload["artifacts"]["onnx_path"]).exists()


def test_missing_config_fails_parse() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
