from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct execution via: python examples/train_stages_smoke.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apex_x.config import load_yaml_config
from apex_x.train import ApexXTrainer
from apex_x.utils import seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Apex-X staged trainer CPU smoke run")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("smoke_cpu.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--steps-per-stage", type=int, default=1, help="Lightweight stage steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    cfg = load_yaml_config(path=args.config)
    seed_all(seed=args.seed, deterministic=cfg.runtime.deterministic)

    trainer = ApexXTrainer(config=cfg)
    result = trainer.run(steps_per_stage=args.steps_per_stage, seed=args.seed)

    stage_names = ",".join(stage.name for stage in result.stage_results)
    selected_ratio_l0 = float(result.routing_diagnostics["selected_ratios"].get("l0", 0.0))
    print("Apex-X staged train smoke run completed")
    print(f"stages={stage_names}")
    print(f"loss_proxy={result.loss_proxy:.6f}")
    print(f"selected_ratio_l0={selected_ratio_l0:.6f}")
    print(f"final_mu={result.final_mu:.6f}")


if __name__ == "__main__":
    main()
