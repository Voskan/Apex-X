#!/bin/bash
# Quick start script for baseline COCO training

echo "======================================"
echo "Apex-X Baseline Training Quick Start"
echo "======================================"
echo ""

# Check if COCO dataset exists
COCO_ROOT="${COCO_ROOT:-/data/coco}"

if [ ! -d "$COCO_ROOT/train2017" ] || [ ! -d "$COCO_ROOT/val2017" ]; then
    echo "❌ COCO dataset not found at $COCO_ROOT"
    echo ""
    echo "Please download COCO dataset:"
    echo "  1. Download train2017 images: http://images.cocodataset.org/zips/train2017.zip"
    echo "  2. Download val2017 images: http://images.cocodataset.org/zips/val2017.zip"
    echo "  3. Download annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    echo ""
    echo "Or set COCO_ROOT environment variable:"
    echo "  export COCO_ROOT=/path/to/coco"
    exit 1
fi

echo "✓ COCO dataset found at $COCO_ROOT"
echo "  - train2017: $(find $COCO_ROOT/train2017 -name '*.jpg' | wc -l) images"
echo "  - val2017: $(find $COCO_ROOT/val2017 -name '*.jpg' | wc -l) images"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  No GPU detected - training will be slow on CPU"
    echo ""
fi

# Start training
echo "Starting training..."
echo "Config: configs/coco_baseline.yaml"
echo "Output: ./outputs/baseline"
echo ""

python scripts/train.py \
    --config configs/coco_baseline.yaml \
    --dataset-path "$COCO_ROOT" \
    --output-dir ./outputs/baseline \
    --num-classes 80 \
    --steps-per-stage 100

echo ""
echo "======================================"
echo "Training complete!"
echo "Check results in: ./outputs/baseline"
echo "======================================"
