# Apex-X Baseline Training Guide

**Quick reference for training Apex-X on COCO dataset**

---

## Prerequisites

- Python 3.11+
- NVIDIA GPU (16GB+ VRAM recommended)
- COCO dataset downloaded
- Dependencies installed: `pip install -r requirements.txt`

---

## Quick Start (One Command)

```bash
# Set dataset path
export COCO_ROOT=/data/coco

# Launch training
./scripts/train_quick_start.sh
```

That's it! The script will:
1. Check dataset exists
2. Detect GPU
3. Start training with default config
4. Save checkpoints to `./outputs/baseline`

---

## Manual Training

```bash
python scripts/train.py \
    --config configs/coco_baseline.yaml \
    --dataset-path /data/coco \
    --output-dir ./outputs/baseline \
    --num-classes 80 \
    --steps-per-stage 100
```

Canonical CLI equivalent:

```bash
python -m apex_x.cli train configs/coco_baseline.yaml \
    --dataset-path /data/coco \
    --set train.output_dir=./outputs/baseline \
    --num-classes 80 \
    --steps-per-stage 100
```

### Command-Line Options

- `--config`: Path to YAML config file (default: `configs/coco_baseline.yaml`)
- `--dataset-path`: Dataset root path used by staged trainer
- `--output-dir`: Where to save checkpoints and logs
- `--resume`: Resume from checkpoint path
- `--checkpoint-dir`: Explicit checkpoint directory override
- `--seed`: Random seed for deterministic run

---

## Configuration

### Default Config (`configs/coco_baseline.yaml`)

```yaml
# Training
epochs: 300
batch_size: 16
base_lr: 0.01
warmup_epochs: 5

# Augmentation
augmentation:
  mosaic: true       # 2x2 grid (+3-5% mAP)
  mixup: true        # Alpha blending (+1-2% mAP)

# Loss (SimOTA)
loss:
  progressive: true  # Dynamic weight scheduling
  simota:
    dynamic_topk: 10
    small_object_boost: 2.0
```

### Custom Config

Create `my_config.yaml`:

```yaml
epochs: 500
batch_size: 32  # If you have more VRAM
base_lr: 0.02   # Higher LR for larger batch

augmentation:
  mosaic_prob: 0.7  # More aggressive
```

Train with:
```bash
python scripts/train_baseline_coco.py --config my_config.yaml
```

---

## Training Process

### Epoch 0-5: Warmup
- Learning rate: 0.001 → 0.01 (linear)
- Loss: High, rapidly decreasing
- Focus: Finding objects (classification)

### Epoch 5-150: Early Training
- Learning rate: 0.01 (plateau)
- Loss: Steady decrease
- Box weight: 0.5 → 1.25 (progressive)
- mAP: 0 → 25

### Epoch 150-250: Mid Training
- Learning rate: 0.01 → 0.001 (cosine decay)
- Box weight: 1.25 → 1.75
- mAP: 25 → 40

### Epoch 250-300: Fine-tuning
- Learning rate: 0.001 → 0.0001
- Box weight: 1.75 → 2.0 (max refinement)
- mAP: 40 → 45+

---

## Monitoring

### Training Logs
```bash
tail -f outputs/baseline/train.log
```

Output:
```
[2026-02-12 08:00:00] INFO Epoch [1/300] Batch [50/7393] Loss: 15.234
[2026-02-12 08:01:23] INFO Epoch [1/300] Batch [100/7393] Loss: 12.456
...
```

### Validation Metrics
```
[2026-02-12 10:00:00] INFO Running validation...
[2026-02-12 10:05:12] INFO Val - mAP: 0.152, AP50: 0.289, AP75: 0.142
[2026-02-12 10:05:12] INFO       APs: 0.045, APm: 0.178, APl: 0.256
[2026-02-12 10:05:12] INFO ✓ Saved new best checkpoint (mAP: 0.152)
```

### Checkpoints
```bash
ls outputs/baseline/checkpoints/

epoch_0010.pt  # Periodic (every 10 epochs)
epoch_0020.pt
...
best.pt        # Best mAP so far
```

---

## Resume Training

If training interrupted:

```bash
python scripts/train.py \
    --config configs/coco_baseline.yaml \
    --resume outputs/baseline/checkpoints/last.pt
```

Or resume from specific epoch:

```bash
python scripts/train.py \
    --config configs/coco_baseline.yaml \
    --resume outputs/baseline/checkpoints/epoch_0150.pt
```

---

## Expected Results

### After 300 Epochs (~3-5 days on V100)

| Metric | Value | Quality |
|--------|-------|---------|
| **mAP** | ~45 | Baseline |
| **AP50** | ~65 | Good |
| **AP75** | ~48 | Good |
| **APs** | ~25 | Medium |
| **APm** | ~50 | Good |
| **APl** | ~60 | Good |

### Comparison to SOTA

- YOLO26: 53 mAP (+18% better)
- YOLOv8x: 51 mAP (+13% better)
- **Apex-X Baseline**: 45 mAP (solid start!)

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
batch_size: 8  # Instead of 16
```

Or use gradient accumulation (future feature).

### Slow Training

Increase workers:
```bash
python scripts/train_baseline_coco.py --num-workers 8
```

Enable mixed precision (future feature).

### Loss Not Decreasing

Check:
1. Dataset loaded correctly (no empty annotations)
2. Learning rate not too high/low
3. Gradients are finite (check logs for NaN warnings)

---

## Multi-GPU Training (Future)

Coming soon:

```bash
# DDP training on 4 GPUs
torchrun --nproc_per_node=4 scripts/train_distributed.py
```

Expected: 4x speedup with linear scaling.

---

## Next Steps

After baseline training:

1. **Evaluate Best Model**:
   ```bash
   python scripts/evaluate.py \
       --checkpoint outputs/baseline/checkpoints/best.pt
   ```

2. **Try DINOv2 Backbone** (+5-8% mAP):
   ```bash
   python scripts/train_dinov2.py  # Coming soon
   ```

3. **Hyperparameter Tuning**:
   - Learning rate sweep
   - Augmentation probability tuning
   - Loss weight optimization

## Fine-tuning & Active Learning (Phase 2)

Apex-X now supports a semi-automated fine-tuning pipeline to continuously improve accuracy on new satellite imagery.

### 1. Data Mining (Active Learning)
Identify "hard examples" where the model is uncertain using entropy and quality headers:
```bash
python scripts/mine_hard_examples.py \
    --model-path checkpoints/teacher_latest.pt \
    --image-path data/new_unlabeled_imagery/ \
    --output-dir data/mined_examples
```

### 2. Semi-supervised Training
The `PseudoLabeler` generates "silver" labels from a high-capacity teacher, filtered by the `MaskQualityHead` to ensure only reliable targets are used.

### 3. Unified Fine-tuning Pipeline
Run the entire orchestration flow with a single command:
```bash
bash scripts/apex_finetune_pipeline.sh \
    --config configs/satellite_v3_finetune.yaml \
    --data data/new_imagery \
    --model checkpoints/teacher_latest.pt
```

This pipeline leverages **LoRA fine-tuning** to adapt the DINOv2 backbone with minimal trainable parameters, ensuring fast and stable convergence.

---

## Support

- **Issues**: https://github.com/yourrepo/apex-x/issues
- **Documentation**: `docs/`
- **Examples**: `examples/`

**Happy Training!** 🚀
