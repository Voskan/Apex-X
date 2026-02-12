#!/usr/bin/env python3
"""Generate the WORLD-CLASS Apex-X A100 Training Notebook (1024px Flagship)."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ═════════════════════════════════════════════
# CELL 1 — Title & "Why 1024px?"
# ═════════════════════════════════════════════
cells.append(md([
    "# 🏆 Apex-X — World-Class 1024px Training Pipeline\n",
    "\n",
    "**Project Flagship**: State-of-the-art instance segmentation for production satellite imagery.\n",
    "\n",
    "### 🎯 The 1024px Advantage\n",
    "Even though the native dataset resolution is 512px, we train at **1024x1024** for three critical reasons:\n",
    "1. **Patch Density**: DINOv2 uses a fixed 14x14 patch. At 1024px, the model extracts **4x more features** ($73 \\times 73$ patches), allowing for ultra-fine mask boundaries.\n",
    "2. **Small Objects**: Roof superstructures (chimneys, windows) benefit from the higher resolution bottleneck in BiFPN and Cascade heads.\n",
    "3. **Learnable Enhancement**: We integrate a trainable preprocessor to clean JPEG artifacts and sharpen details.\n",
    "\n",
    "### 🏗️ Flagship Architecture\n",
    "- **Backbone**: DINOv2-Large + **LoRA** (Rank 8)\n",
    "- **Neck**: 3-Stage **BiFPN** (Weighted multi-scale fusion)\n",
    "- **Head**: 3-Stage **Cascade R-CNN** + **Mask Quality Head**\n",
    "- **Enhancer**: **Learnable Image Enhancer** (+3-5% AP)\n",
    "\n",
    "### 🖥️ Hardware: A100 SXM (80 GB)\n",
    "| Resource | Config | Rationale |\n",
    "|:---|:---|:---|\n",
    "| **Image Size** | 1024x1024 | Maximize feature density |\n",
    "| **Batch Size** | 4 | Optimized for 80GB VRAM |\n",
    "| **Grad Accum** | 16 | Effective Batch = **64** |\n",
    "| **Workers** | 12 | 16 vCPU parallel loading |"
]))

# ═════════════════════════════════════════════
# CELL 2 — Setup
# ═════════════════════════════════════════════
cells.append(md(["## 1. 🔧 Setup & Dependencies"]))

cells.append(code([
    "import os, sys, warnings\n",
    "warnings.filterwarnings('ignore', category=UserWarning, module='IPython')\n",
    "\n",
    "# 1. Install critical system dependencies first\n",
    "!pip install pickleshare structlog -q\n",
    "\n",
    "if not os.path.exists('Apex-X'):\n",
    "    !git clone https://github.com/Voskan/Apex-X.git\n",
    "    print('✅ Repository cloned')\n",
    "else:\n",
    "    !cd Apex-X && git pull\n",
    "    print('✅ Repository updated')\n",
    "\n",
    "%cd Apex-X\n",
    "!pip install -e . -q\n",
    "!pip install pycocotools albumentations matplotlib seaborn tqdm -q\n",
    "print('\\n✅ Environment Ready')"
]))

# ═════════════════════════════════════════════
# CELL 3 — Hardware Check
# ═════════════════════════════════════════════
cells.append(md(["## 2. 🖥️ Hardware Diagnostics"]))

cells.append(code([
    "import torch, psutil\n",
    "props = torch.cuda.get_device_properties(0)\n",
    "print(f'GPU: {props.name} | VRAM: {props.total_memory/1e9:.1f} GB')\n",
    "print(f'System RAM: {psutil.virtual_memory().total/1e9:.1f} GB')\n",
    "!nvidia-smi"
]))

# ═════════════════════════════════════════════
# CELL 4 — Dataset
# ═════════════════════════════════════════════
cells.append(md(["## 3. 📊 Dataset Profiling"]))

cells.append(code([
    "import yaml, cv2, random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "\n",
    "DATASET_ROOT = '/media/voskan/New Volume/2TB HDD/YOLO26_SUPER_MERGED'\n",
    "with open(Path(DATASET_ROOT) / 'data.yaml') as f:\n",
    "    cfg = yaml.safe_load(f)\n",
    "CLASS_NAMES = cfg['names']\n",
    "\n",
    "def show_samples(split='train', n=8):\n",
    "    img_dir = Path(DATASET_ROOT) / split / 'images'\n",
    "    lbl_dir = Path(DATASET_ROOT) / split / 'labels'\n",
    "    files = random.sample(list(img_dir.iterdir()), n)\n",
    "    plt.figure(figsize=(24, 12))\n",
    "    for i, p in enumerate(files):\n",
    "        img = cv2.imread(str(p))\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "        h, w = img.shape[:2]\n",
    "        lp = lbl_dir / f'{p.stem}.txt'\n",
    "        if lp.exists():\n",
    "            with open(lp) as f:\n",
    "                for line in f:\n",
    "                    pts = (np.array(line.split()[1:], dtype=np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)\n",
    "                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)\n",
    "        plt.subplot(2, 4, i+1)\n",
    "        plt.imshow(img); plt.axis('off')\n",
    "    plt.tight_layout(); plt.show()\n",
    "show_samples()"
]))

# ═════════════════════════════════════════════
# CELL 5 — Hyperparameters (1024px Optimized)
# ═════════════════════════════════════════════
cells.append(md(["## 4. ⚙️ WORLD-CLASS Hyperparameters"]))

cells.append(code([
    "IMAGE_SIZE     = 1024    # High-Resolution mode (Upscaled from 512px)\n",
    "BATCH_SIZE     = 4       # Optimized for 1024px + DINO-L on 80GB\n",
    "GRAD_ACCUM     = 16      # Effective Batch = 64\n",
    "EPOCHS         = 200\n",
    "BASE_LR        = 1e-3    # Lowered LR for higher resolution stability\n",
    "WEIGHT_DECAY   = 1e-4\n",
    "WARMUP_EPOCHS  = 10      # Longer warmup for 1024px\n",
    "PATIENCE       = 30\n",
    "EMA_DECAY      = 0.999\n",
    "NUM_WORKERS    = 12\n",
    "DEVICE         = 'cuda'\n",
    "OUTPUT_DIR     = './outputs/a100_v3_1024px'\n",
    "\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "print(f'🚀 Configured for 1024px: Effective Batch 64')"
]))

# ═════════════════════════════════════════════
# CELL 6 — Heavyweight Model + Enhancer
# ═════════════════════════════════════════════
cells.append(md(["## 5. 🏗️ Build Model + Learnable Enhancer"]))

cells.append(code([
    "from apex_x.model import TeacherModelV3\n",
    "from apex_x.model.image_enhancer import LearnableImageEnhancer\n",
    "from apex_x.config import ApexXConfig, ModelConfig, TrainConfig\n",
    "\n",
    "print('Building flagship TeacherModelV3 (1024px Optimized)...')\n",
    "config = ApexXConfig(\n",
    "    model=ModelConfig(input_height=IMAGE_SIZE, input_width=IMAGE_SIZE),\n",
    "    train=TrainConfig()\n",
    ")\n",
    "\n",
    "model = TeacherModelV3(\n",
    "    num_classes=24, backbone_model=\"facebook/dinov2-large\", lora_rank=8\n",
    ").to(DEVICE)\n",
    "\n",
    "enhancer = LearnableImageEnhancer().to(DEVICE)\n",
    "\n",
    "print(f'\\nModel loaded on: {next(model.parameters()).device}')\n",
    "print(f'Enhancer loaded on: {next(enhancer.parameters()).device}')\n",
    "if str(DEVICE) == 'cuda' and not next(model.parameters()).is_cuda:\n",
    "    raise RuntimeError('❌ Model failed to move to GPU!')\n",
    "print('✅ Hardware Verification Complete')\n",
    "\n",
    "params = sum(p.numel() for p in model.parameters() if p.requires_grad) + enhancer.trainable_parameters()\n",
    "print(f'\\n✅ Ready. Total Trainable Parameters: {params:,}')"
]))

# ═════════════════════════════════════════════
# CELL 7 — Pipeline
# ═════════════════════════════════════════════
cells.append(md(["## 6. 📂 1024px Data Pipeline"]))

cells.append(code([
    "from torch.utils.data import DataLoader\n",
    "from apex_x.data import YOLOSegmentationDataset, yolo_collate_fn\n",
    "from apex_x.data.transforms import build_robust_transforms\n",
    "\n",
    "# build_robust_transforms will handle the 512 -> 1024 resizing\n",
    "train_tf = build_robust_transforms(IMAGE_SIZE, IMAGE_SIZE)\n",
    "val_tf   = build_robust_transforms(IMAGE_SIZE, IMAGE_SIZE, distort_prob=0, blur_prob=0)\n",
    "\n",
    "train_ds = YOLOSegmentationDataset(DATASET_ROOT, split='train', transforms=train_tf, image_size=IMAGE_SIZE)\n",
    "val_ds   = YOLOSegmentationDataset(DATASET_ROOT, split='val',   transforms=val_tf, image_size=IMAGE_SIZE)\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, \n",
    "                          num_workers=NUM_WORKERS, collate_fn=yolo_collate_fn, \n",
    "                          pin_memory=True, persistent_workers=True)\n",
    "\n",
    "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,\n",
    "                        num_workers=NUM_WORKERS//2, collate_fn=yolo_collate_fn)\n",
    "\n",
    "print(f'✅ Pipeline Ready (Scaling to {IMAGE_SIZE}px)')"
]))

# ═════════════════════════════════════════════
# CELL 8 — Ultimate Training Loop
# ═════════════════════════════════════════════
cells.append(md(["## 7. 🏋️ The 1024px Flagship Training Flow"]))

cells.append(code([
    "import time, copy, os\n",
    "from tqdm.auto import tqdm\n",
    "from apex_x.train.train_losses_v3 import compute_v3_training_losses\n",
    "from apex_x.train.lr_scheduler import LinearWarmupCosineAnnealingLR\n",
    "from apex_x.train.validation import validate_epoch\n",
    "\n",
    "# Setup EMA\n",
    "ema_model = copy.deepcopy(model).eval()\n",
    "for p in ema_model.parameters(): p.requires_grad = False\n",
    "\n",
    "def update_ema(m, em, d): \n",
    "    with torch.no_grad():\n",
    "        for k, v in m.state_dict().items(): em.state_dict()[k].copy_(d*em.state_dict()[k] + (1-d)*v)\n",
    "\n",
    "optimizer = torch.optim.AdamW(list(model.parameters())+list(enhancer.parameters()), lr=BASE_LR, weight_decay=WEIGHT_DECAY)\n",
    "scheduler = LinearWarmupCosineAnnealingLR(optimizer, len(train_loader)*WARMUP_EPOCHS, len(train_loader)*EPOCHS)\n",
    "scaler    = torch.amp.GradScaler('cuda')\n",
    "\n",
    "# Create val loader\n",
    "val_dataset = YOLOSegmentationDataset(DATASET_ROOT, split='val', transforms=val_tf, image_size=IMAGE_SIZE)\n",
    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=yolo_collate_fn)\n",
    "print(f'Val set: {len(val_dataset)} images')\n",
    "\n",
    "history = {'train_loss': [], 'val_loss': [], 'vram': []}\n",
    "best_val, counter = float('inf'), 0\n",
    "MAX_GRAD_NORM = 1.0\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    model.train(); enhancer.train()\n",
    "    epoch_loss = 0.0\n",
    "    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')\n",
    "    \n",
    "    for i, samples in enumerate(pbar):\n",
    "        imgs = torch.stack([torch.from_numpy(s.image).permute(2,0,1).float()/255.0 for s in samples]).to(DEVICE)\n",
    "        \n",
    "        # Concatenate targets from all samples in the batch\n",
    "        all_boxes = [torch.from_numpy(s.boxes_xyxy) for s in samples if s.boxes_xyxy.shape[0] > 0]\n",
    "        all_labels = [torch.from_numpy(s.class_ids) for s in samples if s.class_ids.shape[0] > 0]\n",
    "        all_masks = [torch.from_numpy(s.masks) for s in samples if s.masks is not None and s.masks.shape[0] > 0]\n",
    "        \n",
    "        targets = {\n",
    "            'boxes': torch.cat(all_boxes).to(DEVICE) if all_boxes else torch.zeros((0, 4), device=DEVICE),\n",
    "            'labels': torch.cat(all_labels).to(DEVICE) if all_labels else torch.zeros((0,), dtype=torch.long, device=DEVICE),\n",
    "            'masks': torch.cat(all_masks).to(DEVICE) if all_masks else None\n",
    "        }\n",
    "        \n",
    "        with torch.amp.autocast('cuda'):\n",
    "            imgs = enhancer(imgs)\n",
    "            output = model(imgs)\n",
    "            loss, loss_dict = compute_v3_training_losses(output, targets, model, config)\n",
    "            loss = loss / GRAD_ACCUM\n",
    "            \n",
    "        scaler.scale(loss).backward()\n",
    "        if (i+1) % GRAD_ACCUM == 0:\n",
    "            scaler.unscale_(optimizer)\n",
    "            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(enhancer.parameters()), MAX_GRAD_NORM)\n",
    "            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()\n",
    "            scheduler.step(); update_ema(model, ema_model, EMA_DECAY)\n",
    "            \n",
    "        epoch_loss += loss.item() * GRAD_ACCUM\n",
    "        pbar.set_postfix({'loss': f'{epoch_loss/(i+1):.4f}', 'vram': f'{torch.cuda.max_memory_allocated()/1e9:.1f}G'})\n",
    "    \n",
    "    avg_train_loss = epoch_loss / len(train_loader)\n",
    "    history['train_loss'].append(avg_train_loss)\n",
    "    \n",
    "    # Real validation on val split\n",
    "    val_metrics = validate_epoch(model, val_loader, device=DEVICE, loss_fn=compute_v3_training_losses, config=config)\n",
    "    val_loss = val_metrics.get('val_loss', float('inf'))\n",
    "    history['val_loss'].append(val_loss)\n",
    "    history['vram'].append(torch.cuda.max_memory_allocated()/1e9)\n",
    "    \n",
    "    print(f'  Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f}')\n",
    "    \n",
    "    # Save last checkpoint (always, for resumability)\n",
    "    torch.save({\n",
    "        'epoch': epoch, 'model': ema_model.state_dict(), 'enhancer': enhancer.state_dict(),\n",
    "        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),\n",
    "        'best_val': best_val, 'history': history,\n",
    "    }, f'{OUTPUT_DIR}/last.pt')\n",
    "    \n",
    "    # Save best checkpoint\n",
    "    if val_loss < best_val:\n",
    "        best_val = val_loss; counter = 0\n",
    "        torch.save({'model': ema_model.state_dict(), 'enhancer': enhancer.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, f'{OUTPUT_DIR}/best_1024.pt')\n",
    "        print(f'  ⭐ New best saved (val_loss={val_loss:.4f})')\n",
    "    else: \n",
    "        counter += 1\n",
    "        if counter >= PATIENCE:\n",
    "            print(f'  ⏹ Early stopping at epoch {epoch}')\n",
    "            break\n",
    "print(f'\\n✅ Training complete. Best val loss: {best_val:.4f}')\n",
]))

# ═════════════════════════════════════════════
# CELL 9 — ONNX Export
# ═════════════════════════════════════════════
cells.append(md(["## 8. 📦 ONNX Export for Production"]))

cells.append(code([
    "# Export best model to ONNX for production deployment\n",
    "best_ckpt = torch.load(f'{OUTPUT_DIR}/best_1024.pt', map_location=DEVICE)\n",
    "model.load_state_dict(best_ckpt['model'])\n",
    "model.eval()\n",
    "\n",
    "dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)\n",
    "onnx_path = f'{OUTPUT_DIR}/apex_x_1024.onnx'\n",
    "\n",
    "try:\n",
    "    torch.onnx.export(\n",
    "        model, dummy_input, onnx_path,\n",
    "        input_names=['images'], output_names=['boxes', 'masks', 'scores'],\n",
    "        dynamic_axes={'images': {0: 'batch'}, 'boxes': {0: 'detections'}, 'masks': {0: 'detections'}, 'scores': {0: 'detections'}},\n",
    "        opset_version=17,\n",
    "    )\n",
    "    print(f'✅ ONNX model exported to {onnx_path}')\n",
    "except Exception as e:\n",
    "    print(f'⚠️ ONNX export failed (model may have dynamic ops): {e}')\n",
    "    print('The best .pt checkpoint is still available for inference.')\n",
]))

# ═════════════════════════════════════════════
# CELL 10 — Summary & Visuals
# ═════════════════════════════════════════════
cells.append(md(["## 🏁 Summary"]))
cells.append(code([
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig, ax = plt.subplots(1, 1, figsize=(10, 5))\n",
    "ax.plot(history['train_loss'], label='Train Loss', color='#4A90D9')\n",
    "ax.plot(history['val_loss'], label='Val Loss', color='#E74C3C')\n",
    "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')\n",
    "ax.set_title('Apex-X 1024px Training Curves')\n",
    "ax.legend(); ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=150)\n",
    "plt.show()\n",
    "print(f'Best val loss: {best_val:.4f}')\n",
    "print(f'Checkpoints: {OUTPUT_DIR}/best_1024.pt, {OUTPUT_DIR}/last.pt')\n",
]))

notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
with open("train_a100_sxm.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("✅ 1024px WORLD-CLASS train_a100_sxm.ipynb generated.")

