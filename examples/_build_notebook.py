#!/usr/bin/env python3
"""Generate the Apex-X A100 training notebook with V3 optimizations."""
import json, sys, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ═══════════════════════════════════════════════════════════════
# CELL 1 — Title & Hardware Overview
# ═══════════════════════════════════════════════════════════════
cells.append(md([
    "# 🚀 Apex-X — A100 SXM Training (TeacherModelV3)\n",
    "\n",
    "**World-Class Satellite Roof Segmentation** optimized for the **YOLO26_SUPER_MERGED** dataset.\n",
    "\n",
    "### 🏗️ Model: `TeacherModelV3` (v2.0 FLAGSHIP)\n",
    "- **Backbone**: DINOv2-Large (frozen) + LoRA\n",
    "- **Neck**: BiFPN (Bidirectional Feature Pyramid Network)\n",
    "- **Head**: Cascade R-CNN (3 stages) + Mask Quality Head\n",
    "- **Loss**: Enhanced GIoU + Boundary IoU + Mask Quality Loss\n",
    "\n",
    "---\n",
    "\n",
    "### 🖥️ Server Specifications\n",
    "| Resource | Spec |\n",
    "|:---------|:-----|\n",
    "| **GPU** | 1× NVIDIA A100 SXM — 80 GB HBM2e |\n",
    "| **RAM** | 117 GB DDR4 |\n",
    "| **CPU** | 16 vCPU |\n",
    "\n",
    "---\n",
    "**Repository**: [github.com/Voskan/Apex-X](https://github.com/Voskan/Apex-X)"
]))

# ═══════════════════════════════════════════════════════════════
# CELL 2 — Environment Setup
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 1. 🔧 Environment Setup"]))

cells.append(code([
    "import os, subprocess, sys\n",
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

# ═══════════════════════════════════════════════════════════════
# CELL 3 — System Diagnostics
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 2. 🖥️ System & GPU Diagnostics"]))

cells.append(code([
    "import torch\n",
    "import platform\n",
    "import psutil\n",
    "\n",
    "print('=' * 60)\n",
    "print('SYSTEM DIAGNOSTICS')\n",
    "print('=' * 60)\n",
    "print(f'GPU:           {torch.cuda.get_device_name(0)}')\n",
    "print(f'VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')\n",
    "print(f'PyTorch:       {torch.__version__}')\n",
    "print(f'CPU:           {psutil.cpu_count()} vCPU')\n",
    "print(f'RAM:           {psutil.virtual_memory().total / 1024**3:.0f} GB')\n",
    "print('=' * 60)\n",
    "\n",
    "!nvidia-smi"
]))

# ═══════════════════════════════════════════════════════════════
# CELL 4 — Dataset Visualization
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 3. 📊 Dataset Visualization"]))

cells.append(code([
    "import yaml, cv2, random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "\n",
    "DATASET_ROOT = '/root/YOLO26_SUPER_MERGED'\n",
    "\n",
    "with open(Path(DATASET_ROOT) / 'data.yaml') as f:\n",
    "    data_cfg = yaml.safe_load(f)\n",
    "CLASS_NAMES = data_cfg['names']\n",
    "\n",
    "def draw_sample(img_path, lbl_path):\n",
    "    img = cv2.imread(str(img_path))\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    h, w = img.shape[:2]\n",
    "    if lbl_path.exists():\n",
    "        with open(lbl_path) as f:\n",
    "            for line in f:\n",
    "                parts = list(map(float, line.strip().split()))\n",
    "                if len(parts) < 5: continue\n",
    "                cls_id = int(parts[0])\n",
    "                pts = (np.array(parts[1:]).reshape(-1, 2) * [w, h]).astype(np.int32)\n",
    "                cv2.polylines(img, [pts], True, (255, 0, 0), 2)\n",
    "    return img\n",
    "\n",
    "plt.figure(figsize=(20, 10))\n",
    "img_files = random.sample(list((Path(DATASET_ROOT)/'train'/'images').iterdir()), 4)\n",
    "for i, im_p in enumerate(img_files):\n",
    "    plt.subplot(1, 4, i+1)\n",
    "    plt.imshow(draw_sample(im_p, Path(DATASET_ROOT)/'train'/'labels'/f'{im_p.stem}.txt'))\n",
    "    plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# ═══════════════════════════════════════════════════════════════
# CELL 5 — Hyperparameters
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 4. ⚙️ Hyperparameter Configuration"]))

cells.append(code([
    "# Optimized for 1× A100 80GB\n",
    "IMAGE_SIZE     = 512\n",
    "BATCH_SIZE     = 16      # Increased for A100 80GB\n",
    "GRAD_ACCUM     = 4       # Effective batch = 64\n",
    "EPOCHS         = 300\n",
    "BASE_LR        = 2e-3    # Finetuning LR\n",
    "WEIGHT_DECAY   = 1e-4\n",
    "WARMUP_EPOCHS  = 5\n",
    "VAL_INTERVAL   = 5\n",
    "NUM_WORKERS    = 12\n",
    "DEVICE         = 'cuda'\n",
    "OUTPUT_DIR     = './outputs/a100_v3_run'"
]))

# ═══════════════════════════════════════════════════════════════
# CELL 6 — Model Initialization
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 5. 🧠 Model Initialization (TeacherModelV3)"]))

cells.append(code([
    "from apex_x.config import ApexXConfig, ModelConfig, TrainConfig\n",
    "from apex_x.model import TeacherModelV3\n",
    "from apex_x.train.lr_scheduler import LinearWarmupCosineAnnealingLR\n",
    "\n",
    "config = ApexXConfig(\n",
    "    model=ModelConfig(input_height=IMAGE_SIZE, input_width=IMAGE_SIZE, num_classes=24),\n",
    "    train=TrainConfig(qat_enable=False)\n",
    ")\n",
    "\n",
    "print('Initializing TeacherModelV3 with DINOv2-Large + LoRA...')\n",
    "model = TeacherModelV3(\n",
    "    num_classes=24,\n",
    "    backbone_model=\"facebook/dinov2-large\",\n",
    "    lora_rank=8,\n",
    "    fpn_channels=256,\n",
    "    num_cascade_stages=3\n",
    ").to(DEVICE)\n",
    "\n",
    "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "print(f'Trainable parameters: {trainable_params:,}')\n",
    "\n",
    "optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=BASE_LR, weight_decay=WEIGHT_DECAY)\n",
    "scaler = torch.amp.GradScaler('cuda')"
]))

# ═══════════════════════════════════════════════════════════════
# CELL 7 — Training Loop
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 6. 🏋️ Training Loop (V3 Loss)"]))

cells.append(code([
    "import time\n",
    "from tqdm.auto import tqdm\n",
    "from apex_x.train.train_losses_v3 import compute_v3_training_losses\n",
    "from apex_x.data import YOLOSegmentationDataset, yolo_collate_fn\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "train_ds = YOLOSegmentationDataset(DATASET_ROOT, split='train', image_size=IMAGE_SIZE)\n",
    "val_ds   = YOLOSegmentationDataset(DATASET_ROOT, split='val',   image_size=IMAGE_SIZE)\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=yolo_collate_fn)\n",
    "val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS // 2, collate_fn=yolo_collate_fn)\n",
    "\n",
    "scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=len(train_loader)*WARMUP_EPOCHS, total_steps=len(train_loader)*EPOCHS)\n",
    "\n",
    "best_loss = float('inf')\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    model.train()\n",
    "    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')\n",
    "    for batch_idx, samples in enumerate(pbar):\n",
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
    "        \n",
    "        if (batch_idx + 1) % GRAD_ACCUM == 0:\n",
    "            scaler.step(optimizer)\n",
    "            scaler.update()\n",
    "            optimizer.zero_grad()\n",
    "            scheduler.step()\n",
    "        \n",
    "        pbar.set_postfix({'loss': loss.item()*GRAD_ACCUM})\n",
    "    \n",
    "    # Save latest\n",
    "    torch.save(model.state_dict(), f'{OUTPUT_DIR}/latest.pt')\n",
    "    print(f'Epoch {epoch} finished.')"
]))

# ═══════════════════════════════════════════════════════════════
# CELL 8 — Final Summary
# ═══════════════════════════════════════════════════════════════
cells.append(md(["## 7. ✅ Summary\n", "Training complete. Models saved to `outputs/a100_v3_run`."]))

notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
with open("train_a100_sxm.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("✅ train_a100_sxm.ipynb generated.")
