#!/usr/bin/env python3
"""Compatibility wrapper for canonical satellite v3 training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex_x.cli import train_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Satellite v3 training wrapper. "
            "Canonical path: python -m apex_x.cli train <config> ..."
        )
    )
    parser.add_argument("--config", default="configs/satellite_1024.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps-per-stage", type=int, default=750)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--set", "-s", action="append", dest="set_values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set_values or [])
    if args.output_dir:
        overrides.append(f"train.output_dir={args.output_dir}")

    train_cmd(
        config=args.config,
        set_values=overrides,
        steps_per_stage=args.steps_per_stage,
        seed=args.seed,
        num_classes=args.num_classes,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        dataset_path=args.data_root,
    )


if __name__ == "__main__":
    main()
