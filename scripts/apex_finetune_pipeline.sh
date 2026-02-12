#!/bin/bash
# ==============================================================================
# Apex-X Unified Fine-tuning Pipeline
# ==============================================================================
# Orchestrates:
# 1. Mining (Active Learning)
# 2. Pseudo-labeling (Semi-supervised)
# 3. Fine-tuning (LoRA Adaptation)
# ==============================================================================

set -e

CONFIG="./configs/satellite_v3_finetune.yaml"
DATA_ROOT="data/new_imagery"
OUTPUT_DIR="./outputs/finetune_run_$(date +%Y%m%d)"
MODEL_PATH="checkpoints/teacher_v3_latest.pt"

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config) CONFIG="$2"; shift; shift ;;
    --data) DATA_ROOT="$2"; shift; shift ;;
    --model) MODEL_PATH="$2"; shift; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "🚀 Starting Apex-X Fine-tuning Pipeline..."
echo "----------------------------------------"
echo "Config: $CONFIG"
echo "Data:   $DATA_ROOT"
echo "Model:  $MODEL_PATH"
echo "----------------------------------------"

# 1. Mine Hard Examples (Active Learning)
echo "🔍 Step 1: Mining hard examples from new imagery..."
python scripts/mine_hard_examples.py \
    --model-path "$MODEL_PATH" \
    --image-path "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR/mined"

# 2. Semi-supervised Training (Future implementation: Integrate pseudo_label.py into training loop)
echo "🧬 Step 2: Preparing semi-supervised labels..."
# Placeholder for direct pseudo-label integration inside train script

# 3. Run Fine-tuning (LoRA)
echo "🏋️ Step 3: Running LoRA fine-tuning..."
python scripts/train_satellite_v3.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR/checkpoints" \
    --resume "$MODEL_PATH" \
    --epochs 50

echo "✅ Fine-tuning Pipeline Complete! 🎉"
echo "Checkpoints: $OUTPUT_DIR/checkpoints"
