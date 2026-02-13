# Apex-X Training Guide

This guide documents the current, runnable training flow.

## 1. Prerequisites

- Python 3.11+
- PyTorch installed
- Optional GPU for accelerated training

Install project:

```bash
pip install -e .
```

Optional worldclass deps (required for `TeacherModelV3` / DINOv2 paths):

```bash
pip install "apex-x[worldclass]"
# or
pip install transformers timm peft safetensors
```

Optional dependency preflight:

```bash
python -m apex_x.cli preflight --profile worldclass
```

Worldclass startup checklist (run before any DINOv2/V3 flow):

1. Install optional deps: `pip install "apex-x[worldclass]"`
2. Run preflight: `python -m apex_x.cli preflight --profile worldclass`
3. Validate cache/network access to `facebook/dinov2-large` (first run downloads weights)
4. Start one of the supported entrypoints:
   - CLI: `python -m apex_x.cli train configs/worldclass.yaml ...`
   - Wrapper: `python scripts/train_worldclass_coco.py --config configs/worldclass.yaml ...`
   - Notebook: `notebooks/checkpoint_image_inference.ipynb`

## 2. Canonical train entrypoints

Equivalent entrypoints:

```bash
python -m apex_x.cli train <config> [options]
```

or wrapper:

```bash
python scripts/train.py --config <config> [options]
```

### Minimal example

```bash
python -m apex_x.cli train configs/coco_baseline.yaml \
  --steps-per-stage 10 \
  --seed 0 \
  --num-classes 3 \
  --set train.epochs=1 \
  --set train.output_dir=artifacts/train_output
```

## 3. Dataset policy (fail-fast)

Default behavior:
- `train.allow_synthetic_fallback = false`
- invalid/missing dataset configuration causes a hard error

This prevents silent synthetic training when real dataset loading fails.

Run explicit dataset contract preflight:

```bash
python -m apex_x.cli dataset-preflight configs/coco_baseline.yaml \
  --output-json artifacts/dataset_preflight.json
```

For explicit smoke/debug synthetic runs only:

```bash
python -m apex_x.cli train examples/smoke_cpu.yaml \
  --steps-per-stage 1 \
  --set train.allow_synthetic_fallback=true
```

Training report (`train_report.json`) now includes dataset mode:
- `real`
- `synthetic`

## 4. Checkpoints

Checkpoint utilities use a unified secure load path with `weights_only=True`
when supported by your PyTorch version.

Supported payloads:
- raw state_dict
- `{model_state_dict: ...}`
- `{state_dict: ...}`
- `{model: ...}`
- `{teacher: ...}`
- `{ema_model: ...}`
- `{ema: ...}`

Trainer checkpoints are saved in:

- `<train.output_dir>/checkpoints/epoch_XXXX.pt`
- `<train.output_dir>/checkpoints/best.pt`
- `<train.output_dir>/checkpoints/last.pt`

## 5. Validation and report artifacts

Training writes artifacts to `train.output_dir`:
- `train_report.json`
- `train_report.md`
- `config.json`
- checkpoint files in `checkpoints/`

`train_report.json` includes:
- selected checkpoint metric
- best metric value
- history by epoch
- dataset mode/backend
- loss diagnostics (component means, grad norm, NaN/Inf/OOM counters)
- per-stage metrics snapshot

Optional loss-stability knobs (`train.*`):
- `loss_boundary_warmup_epochs`
- `loss_boundary_max_scale`
- `loss_box_warmup_epochs`
- `loss_box_scale_start`
- `loss_box_scale_end`
- `loss_det_component_clip`
- `loss_boundary_component_clip`
- `loss_seg_component_clip`

## 6. Resume training

```bash
python -m apex_x.cli train configs/coco_baseline.yaml \
  --resume artifacts/train_output/checkpoints/last.pt
```

## 7. Common troubleshooting

### Missing worldclass dependencies

If `TeacherModelV3`/DINOv2 path fails with missing modules, install:

```bash
pip install "apex-x[worldclass]"
```

### Dataset path errors

If training fails with synthetic fallback disabled, either:
- fix dataset configuration/path
- or explicitly set `train.allow_synthetic_fallback=true` for a smoke run

### Checkpoint load warnings

The repository uses a centralized checkpoint loader that prefers secure loading behavior.
If you load checkpoints manually in custom code, use the same helper APIs from
`apex_x.train.checkpoint`.

## 8. Notebook Inference Smoke Artifact (CPU/CUDA)

Use this reproducible smoke runner before manual notebook checks:

```bash
python scripts/notebook_checkpoint_smoke.py \
  --checkpoint outputs/a100_v3_1024px/best_1024.pt \
  --image /path/to/image.png \
  --devices cpu,cuda \
  --output-json artifacts/notebook_smoke/report.json
```

Notes:
- If `cuda` is unavailable, the CUDA run is marked as `skip` in JSON.
- For DINOv2 token-grid reshape failures, the runner retries with square resize.
