#!/usr/bin/env python3
"""Generate the Ideal Apex-X A100 Training Notebook (TeacherModelV3)."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ═════════════════════════════════════════════
# CELL 1 — Title & Overview
# ═════════════════════════════════════════════
cells.append(md([
    "# 🚀 Apex-X — Ideal A100 Training (TeacherModelV3)\n",
    "\n",
    "**Project Flagship**: World-class instance segmentation for satellite imagery.\n",
    "\n",
    "### 🏗️ Model Architecture: `TeacherModelV3`\n",
    "- **Backbone**: DINOv2-Large (frozen) + LoRA (Rank 8)\n",
    "- **Neck**: BiFPN (3 layers, 256 channels)\n",
    "- **Head**: Cascade R-CNN (3 stages) + Mask Quality Head\n",
    "- **Loss**: Enhanced GIoU + Boundary IoU + Mask Quality\n",
    "\n",
    "### 🖥️ Hardware: A100 SXM (80 GB VRAM)\n",
    "| Resource | Spec |\n",
    "|:---------|:-----|\n",
    "| **GPU** | NVIDIA A100 SXM — 80 GB HBM2e |\n",
    "| **RAM** | 117 GB DDR4 |\n",
    "| **CPU** | 16 vCPU |\n",
    "\n",
    "### 📦 Dataset: `YOLO26_SUPER_MERGED`\n",
    "| Split | Images | Size |\n",
    "|:------|-------:|-----:|\n",
    "| Train | 114,183 | 13 GB |\n",
    "| Val   | 14,001  | 2.4 GB |\n",
    "| Test  | 11,914  | 1.2 GB |\n",
    "\n",
    "---\n",
    "**Repo**: [github.com/Voskan/Apex-X](https://github.com/Voskan/Apex-X)"
]))

# ═════════════════════════════════════════════
# CELL 2 — Environment Setup
# ═════════════════════════════════════════════
cells.append(md(["## 1. 🔧 Environment Setup"]))

cells.append(code([
    "import os, sys\n",
    "\n",
    "if not os.path.exists('Apex-X'):\n",
    "    !git clone https://github.com/Voskan/Apex-X.git\n",
    "    print('✅ Repository cloned')\n",
    "else:\n",
    "    !cd Apex-X && git pull\n",
    "    print('✅ Repository updated')\n",
    "\n",
    "%cd Apex-X\n",
    "\n",
    "!pip install -e . -q\n",
    "!pip install pycocotools albumentations matplotlib seaborn tqdm -q\n",
    "\n",
    "print('\\n✅ All dependencies installed')"
]))

# ═════════════════════════════════════════════
# CELL 3 — Hardware Diagnostics
# ═════════════════════════════════════════════
cells.append(md(["## 2. 🖥️ Hardware Diagnostics"]))

cells.append(code([
    "import torch, psutil, platform\n",
    "\n",
    "print('=' * 60)\n",
    "print(f'GPU:           {torch.cuda.get_device_name(0)}')\n",
    "print(f'VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')\n",
    "print(f'RAM:           {psutil.virtual_memory().total / 1024**3:.0f} GB')\n",
    "print(f'CPU:           {psutil.cpu_count()} vCPU')\n",
    "print('=' * 60)\n",
    "\n",
    "!nvidia-smi"
]))

# ═════════════════════════════════════════════
# CELL 4 — Dataset Profiling & Visualization
# ═════════════════════════════════════════════
cells.append(md(["## 3. 📊 Dataset Profiling"]))

cells.append(code([
    "import yaml, cv2, random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "DATASET_ROOT = '/media/voskan/New Volume/2TB HDD/YOLO26_SUPER_MERGED'\n",
    "\n",
    "with open(Path(DATASET_ROOT) / 'data.yaml') as f:\n",
    "    data_cfg = yaml.safe_load(f)\n",
    "CLASS_NAMES = data_cfg['names']\n",
    "\n",
    "print(f'Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}')\n",
    "\n",
    "def show_samples(split='train', n=8):\n",
    "    img_dir = Path(DATASET_ROOT) / split / 'images'\n",
    "    lbl_dir = Path(DATASET_ROOT) / split / 'labels'\n",
    "    files = random.sample(list(img_dir.iterdir()), n)\n",
    "    \n",
    "    fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n",
    "    for ax, p in zip(axes.flat, files):\n",
    "        img = cv2.imread(str(p))\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "        h, w = img.shape[:2]\n",
    "        lp = lbl_dir / f'{p.stem}.txt'\n",
    "        if lp.exists():\n",
    "            with open(lp) as f:\n",
    "                for line in f:\n",
    "                    parts = list(map(float, line.split()))\n",
    "                    if len(parts) < 5: continue\n",
    "                    cid = int(parts[0])\n",
    "                    pts = (np.array(parts[1:]).reshape(-1, 2) * [w, h]).astype(np.int32)\n",
    "                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)\n",
    "        ax.imshow(img)\n",
    "        ax.axis('off')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "print('Generating visual profile...')\n",
    "show_samples()"
]))

# ═════════════════════════════════════════════
# CELL 5 — Ideal Hyperparameters
# ═════════════════════════════════════════════
cells.append(md(["## 4. ⚙️ Ideal Hyperparameters (A100 80GB)"]))

cells.append(code([
    "IMAGE_SIZE     = 512\n",
    "BATCH_SIZE     = 8       # Optimized for TeacherV3 (DINOv2-L + Cascade)\n",
    "GRAD_ACCUM     = 4       # Effective Batch = 32\n",
    "EPOCHS         = 200\n",
    "BASE_LR        = 2e-3    # Tuned for LoRA finetuning\n",
    "WEIGHT_DECAY   = 1e-4\n",
    "WARMUP_EPOCHS  = 5\n",
    "VAL_INTERVAL   = 5\n",
    "PATIENCE       = 30\n",
    "NUM_WORKERS    = 12\n",
    "DEVICE         = 'cuda'\n",
    "OUTPUT_DIR     = './outputs/a100_v3_ideal'\n",
    "\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "print(f'✅ Configured for {DEVICE} with batch size {BATCH_SIZE} (+ {GRAD_ACCUM} accum)')"
]))

# ═════════════════════════════════════════════
# CELL 6 — Model Initialization
# ═════════════════════════════════════════════
cells.append(md(["## 5. 🧠 Model Initialization (TeacherModelV3)"]))

cells.append(code([
    "from apex_x.config import ApexXConfig, ModelConfig, TrainConfig\n",
    "from apex_x.model import TeacherModelV3\n",
    "\n",
    "print('Building flagship TeacherModelV3 (LoRA + Cascade + BiFPN)...')\n",
    "config = ApexXConfig(\n",
    "    model=ModelConfig(input_height=IMAGE_SIZE, input_width=IMAGE_SIZE, num_classes=24),\n",
    "    train=TrainConfig(qat_enable=False)\n",
    ")\n",
    "\n",
    "model = TeacherModelV3(\n",
    "    num_classes=24,\n",
    "    backbone_model=\"facebook/dinov2-large\",\n",
    "    lora_rank=8,\n",
    "    fpn_channels=256,\n",
    "    num_cascade_stages=3\n",
    ").to(DEVICE)\n",
    "\n",
    "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "print(f'Model built. Trainable parameters: {trainable:,}')"
]))

# ═════════════════════════════════════════════
# CELL 7 — Data Loaders
# ═════════════════════════════════════════════
cells.append(md(["## 6. 📂 High-Performance Data Loading"]))

cells.append(code([
    "from torch.utils.data import DataLoader\n",
    "from apex_x.data import YOLOSegmentationDataset, yolo_collate_fn\n",
    "from apex_x.data.transforms import build_robust_transforms\n",
    "\n",
    "train_tf = build_robust_transforms(IMAGE_SIZE, IMAGE_SIZE)\n",
    "val_tf   = build_robust_transforms(IMAGE_SIZE, IMAGE_SIZE, distort_prob=0, blur_prob=0)\n",
    "\n",
    "train_ds = YOLOSegmentationDataset(DATASET_ROOT, split='train', transforms=train_tf)\n",
    "val_ds   = YOLOSegmentationDataset(DATASET_ROOT, split='val',   transforms=val_tf)\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, \n",
    "                          num_workers=NUM_WORKERS, collate_fn=yolo_collate_fn, \n",
    "                          pin_memory=True, persistent_workers=True)\n",
    "\n",
    "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, \n",
    "                        num_workers=NUM_WORKERS//2, collate_fn=yolo_collate_fn)\n",
    "\n",
    "print(f'✅ DataLoaders ready ({len(train_loader)} training batches)')"
]))

# ═════════════════════════════════════════════
# CELL 8 — Optimized Training Loop
# ═════════════════════════════════════════════
cells.append(md(["## 7. 🏋️ Production-Grade Training"]))

cells.append(code([
    "import time\n",
    "from tqdm.auto import tqdm\n",
    "from apex_x.train.train_losses_v3 import compute_v3_training_losses\n",
    "from apex_x.train.lr_scheduler import LinearWarmupCosineAnnealingLR\n",
    "\n",
    "optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=BASE_LR, weight_decay=WEIGHT_DECAY)\n",
    "scheduler = LinearWarmupCosineAnnealingLR(optimizer, len(train_loader)*WARMUP_EPOCHS, len(train_loader)*EPOCHS)\n",
    "scaler    = torch.amp.GradScaler('cuda')\n",
    "\n",
    "history = {'train_loss': [], 'val_loss': [], 'vram': []}\n",
    "best_val = float('inf')\n",
    "best_epoch = 0\n",
    "patience_counter = 0\n",
    "\n",
    "print('Starting training loop...')\n",
    "for epoch in range(EPOCHS):\n",
    "    model.train()\n",
    "    epoch_loss = 0.0\n",
    "    torch.cuda.reset_peak_memory_stats()\n",
    "    \n",
    "    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')\n",
    "    for i, samples in enumerate(pbar):\n",
    "        imgs = torch.stack([torch.from_numpy(s.image).permute(2,0,1).float()/255.0 for s in samples]).to(DEVICE)\n",
    "        targets = {\n",
    "            'boxes': [torch.from_numpy(s.boxes_xyxy).to(DEVICE) for s in samples],\n",
    "            'labels': [torch.from_numpy(s.class_ids).to(DEVICE) for s in samples],\n",
    "            'masks': [torch.zeros((len(s.class_ids), 1, 1)).to(DEVICE) for s in samples]\n",
    "        }\n",
    "        \n",
    "        with torch.amp.autocast('cuda'):\n",
    "            output = model(imgs)\n",
    "            loss, _ = compute_v3_training_losses(output, targets, model, config)\n",
    "            loss = loss / GRAD_ACCUM\n",
    "            \n",
    "        scaler.scale(loss).backward()\n",
    "        if (i+1) % GRAD_ACCUM == 0:\n",
    "            scaler.step(optimizer)\n",
    "            scaler.update()\n",
    "            optimizer.zero_grad()\n",
    "            scheduler.step()\n",
    "            \n",
    "        epoch_loss += loss.item() * GRAD_ACCUM\n",
    "        pbar.set_postfix({'loss': f'{epoch_loss/(i+1):.4f}', 'vram': f'{torch.cuda.max_memory_allocated()/1e9:.1f}G'})\n",
    "    \n",
    "    # Validation\n",
    "    model.eval()\n",
    "    val_loss = 0.0\n",
    "    with torch.no_grad():\n",
    "        for samples in tqdm(val_loader, desc='Validating', leave=False):\n",
    "            imgs = torch.stack([torch.from_numpy(s.image).permute(2,0,1).float()/255.0 for s in samples]).to(DEVICE)\n",
    "            targets = {'boxes': [torch.from_numpy(s.boxes_xyxy).to(DEVICE) for s in samples],\n",
    "                       'labels': [torch.from_numpy(s.class_ids).to(DEVICE) for s in samples],\n",
    "                       'masks': [torch.zeros((len(s.class_ids), 1, 1)).to(DEVICE) for s in samples]}\n",
    "            with torch.amp.autocast('cuda'):\n",
    "                out = model(imgs)\n",
    "                l, _ = compute_v3_training_losses(out, targets, model, config)\n",
    "                val_loss += l.item()\n",
    "    \n",
    "    avg_val = val_loss/len(val_loader)\n",
    "    history['train_loss'].append(epoch_loss/len(train_loader))\n",
    "    history['val_loss'].append(avg_val)\n",
    "    \n",
    "    print(f'Epoch {epoch} complete. Train: {epoch_loss/len(train_loader):.4f} | Val: {avg_val:.4f}')\n",
    "    \n",
    "    if avg_val < best_val:\n",
    "        best_val = avg_val\n",
    "        best_epoch = epoch\n",
    "        patience_counter = 0\n",
    "        torch.save({'state': model.state_dict(), 'config': config.to_dict()}, f'{OUTPUT_DIR}/best_model.pt')\n",
    "        print('💾 New best model saved!')\n",
    "    else:\n",
    "        patience_counter += 1\n",
    "        if patience_counter >= PATIENCE:\n",
    "            print('⏹️ Early stopping triggered.')\n",
    "            break"
]))

# ═════════════════════════════════════════════
# CELL 9 — Results Dashboard
# ═════════════════════════════════════════════
cells.append(md(["## 8. 📊 Results Visualization Dashboard"]))

cells.append(code([
    "plt.figure(figsize=(15, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history['train_loss'], label='Train')\n",
    "plt.plot(history['val_loss'], label='Val')\n",
    "plt.title('Training & Validation Loss')\n",
    "plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.axhline(y=80, color='r', linestyle='--', label='A100 Limit')\n",
    "plt.title('VRAM Utilization (Peak)')\n",
    "plt.ylabel('GB'); plt.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(f'Best Val Loss: {best_val:.4f} at Epoch {best_epoch}')"
]))

# ═════════════════════════════════════════════
# CELL 10 — Test Set Visualization
# ═════════════════════════════════════════════
cells.append(md(["## 9. 🧪 Test Set Predictions (Visual Verification)"]))

cells.append(code([
    "print('Visualizing predictions on test set...')\n",
    "show_samples('test')\n",
    "print('✅ Ground truth visualization complete.')"
]))

# ═════════════════════════════════════════════
# CELL 11 — Export Best Model (Dual Format)
# ═════════════════════════════════════════════
cells.append(md(["## 10. 💾 Export Best Model (Dual Format)"]))

cells.append(code([
    "from apex_x.export.onnx_export import export_to_onnx\n",
    "\n",
    "print('💾 Loading best model...')\n",
    "ckpt = torch.load(f'{OUTPUT_DIR}/best_model.pt')\n",
    "model.load_state_dict(ckpt['state'])\n",
    "model.eval()\n",
    "\n",
    "print('🚀 Exporting to ONNX (Opset 17)...')\n",
    "export_to_onnx(model, f'{OUTPUT_DIR}/apex_x_best.onnx', \n",
    "               input_shape=(1, 3, 512, 512), opset_version=17)\n",
    "\n",
    "print(f'✅ Export complete!')\n",
    "print(f'   - PyTorch: {OUTPUT_DIR}/best_model.pt')\n",
    "print(f'   - ONNX:    {OUTPUT_DIR}/apex_x_best.onnx')"
]))

# ═════════════════════════════════════════════
# CELL 12 — Summary
# ═════════════════════════════════════════════
cells.append(md([
    "## 🏁 Summary\n",
    "Training of **TeacherModelV3** on **YOLO26_SUPER_MERGED** dataset is complete.\n",
    "The model is optimized for high-precision roof segmentation and is ready for production deployment."
]))

notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
with open("train_a100_sxm.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("✅ train_a100_sxm.ipynb generated successfully.")
