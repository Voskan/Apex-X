from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution via: python examples/smoke_cpu.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apex_x.config import load_yaml_config
from apex_x.model import ApexXModel
from apex_x.utils import seed_all


def run_smoke(
    config_path: Path,
    overrides: Sequence[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = load_yaml_config(path=config_path, overrides=overrides)
    seed_all(seed=seed, deterministic=cfg.runtime.deterministic)

    model = ApexXModel(config=cfg)
    rng = np.random.RandomState(seed)
    image = rng.rand(1, 3, cfg.model.input_height, cfg.model.input_width).astype(np.float32)
    out = model.forward(image)

    return {
        "selected_tiles": len(out["selected_tiles"]),
        "det_score": float(out["det"]["scores"][0]),
        "merged_shape": tuple(out["merged"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apex-X CPU smoke example")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("smoke_cpu.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config value with section.key=value",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    result = run_smoke(config_path=args.config, overrides=args.overrides, seed=args.seed)

    print("Apex-X CPU smoke run completed")
    print(f"selected_tiles={result['selected_tiles']}")
    print(f"det_score={result['det_score']:.4f}")
    print(f"merged_shape={result['merged_shape']}")


if __name__ == "__main__":
    main()
