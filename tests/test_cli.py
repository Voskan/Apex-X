from __future__ import annotations

from pathlib import Path

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


def test_predict_command_parses() -> None:
    result = runner.invoke(app, ["predict", "--config", str(FIXTURE_CFG)])
    assert result.exit_code == 0
    assert "predict ok" in result.stdout


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
    out = tmp_path / "artifact.txt"
    result = runner.invoke(
        app,
        [
            "export",
            "--config",
            str(FIXTURE_CFG),
            "--output",
            str(out),
            "--set",
            "runtime.precision_profile='quality'",
        ],
    )
    assert result.exit_code == 0
    assert "export ok" in result.stdout
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Apex-X export placeholder" in content


def test_missing_config_fails_parse() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
