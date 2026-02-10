from __future__ import annotations

import subprocess
import sys


def test_train_stages_smoke_runs_quickly() -> None:
    cmd = [
        sys.executable,
        "examples/train_stages_smoke.py",
        "--config",
        "examples/smoke_cpu.yaml",
        "--steps-per-stage",
        "1",
        "--seed",
        "123",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "Apex-X staged train smoke run completed" in proc.stdout
    assert (
        "stages=baseline_warmup,teacher_full_compute,oracle_bootstrap,"
        "continuous_budgeting,deterministic_emulation"
    ) in proc.stdout
