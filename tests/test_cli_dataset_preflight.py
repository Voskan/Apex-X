from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_x.cli import dataset_preflight_cmd


def test_cli_dataset_preflight_smoke_config_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "dataset_preflight.json"
    dataset_preflight_cmd(
        config="examples/smoke_cpu.yaml",
        output_json=str(output),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["dataset_type"] == "synthetic"


def test_cli_dataset_preflight_fails_when_dataset_contract_invalid(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  allow_synthetic_fallback: false\n"
        "data:\n"
        "  dataset_type: auto\n"
        "  dataset_root: \"\"\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        dataset_preflight_cmd(
            config=str(config_path),
            output_json=str(tmp_path / "dataset_preflight_fail.json"),
        )
