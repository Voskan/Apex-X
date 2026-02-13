# Apex-X

Apex-X is a computer vision training/inference codebase with staged training, routing diagnostics,
and export/inference CLI flows.

This README intentionally contains only commands that are currently supported in this repository.

## Installation

```bash
git clone https://github.com/Voskan/Apex-X.git
cd Apex-X
pip install -e .
```

Optional worldclass dependencies for `TeacherModelV3` / DINOv2 paths:

```bash
pip install "apex-x[worldclass]"
# or
pip install transformers timm peft safetensors
```

Preflight check:

```bash
python -m apex_x.cli preflight --profile worldclass
```

## Quick CLI

### Train

```bash
python -m apex_x.cli train configs/coco_baseline.yaml \
  --steps-per-stage 10 \
  --seed 0 \
  --num-classes 3 \
  --set train.epochs=1 \
  --set train.output_dir=artifacts/train_output
```

Important:
- By default, synthetic fallback is disabled (`train.allow_synthetic_fallback=false`).
- Provide a valid dataset configuration/path for real training.
- Enable synthetic fallback explicitly only for smoke/debug runs:

```bash
python -m apex_x.cli train examples/smoke_cpu.yaml \
  --steps-per-stage 1 \
  --set train.allow_synthetic_fallback=true
```

### Eval

```bash
python -m apex_x.cli eval configs/coco_baseline.yaml \
  --report-json artifacts/eval_report.json \
  --report-md artifacts/eval_report.md
```

### Predict

```bash
python -m apex_x.cli predict configs/coco_baseline.yaml \
  --report-json artifacts/predict_report.json
```

### Export

```bash
python -m apex_x.cli export configs/coco_baseline.yaml \
  --output artifacts/apex_x_export_manifest.json
```

If you need to export from a checkpoint:

```bash
python -m apex_x.cli export configs/coco_baseline.yaml \
  --checkpoint artifacts/train_output/checkpoints/best.pt \
  --output artifacts/apex_x_export_manifest.json
```

## Checkpoints

The repository supports common checkpoint payload formats through a unified loader:
- raw tensor state_dict
- dict with `model_state_dict`
- dict with `state_dict`
- dict with `model` / `teacher` / `ema_model` / `ema`

Training artifacts are written under `train.output_dir` and checkpoint files under
`<train.output_dir>/checkpoints`.

Training report artifacts:
- `<train.output_dir>/train_report.json`
- `<train.output_dir>/train_report.md`

## Notebook inference

Use:

- `notebooks/checkpoint_image_inference.ipynb`

The notebook supports:
- checkpoint upload/path
- image upload/path
- model family auto-detection
- strict/non-strict loading
- secure checkpoint loading path (`weights_only` when available)

## Testing

```bash
pytest -q
```

For focused checks:

```bash
pytest -q tests/test_train_stages_smoke.py
pytest -q tests/test_routing_diagnostics.py
```

## Documentation

- `docs/TRAINING_GUIDE.md`
- `docs/index.md`

## Notes on metrics and claims

This repository should only publish performance claims backed by reproducible artifacts
(config, checkpoint lineage, commit hash, and evaluation reports).
