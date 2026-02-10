from __future__ import annotations

import subprocess
import sys


def test_smoke_cpu_example_runs_quickly() -> None:
    cmd = [
        sys.executable,
        "examples/smoke_cpu.py",
        "--config",
        "examples/smoke_cpu.yaml",
        "--seed",
        "123",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "Apex-X CPU smoke run completed" in proc.stdout
    assert "selected_tiles=" in proc.stdout
