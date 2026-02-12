"""Baseline training script for Apex-X on COCO dataset.

This script trains the Teacher model from scratch on COCO train2017
and validates on COCO val2017 to establish baseline performance.

Usage:
    python scripts/train_baseline_coco.py --config configs/coco_baseline.yaml
    
Expected results after 300 epochs:
    - mAP > 45 (competitive baseline)
    - Small object AP (APs) > 25
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add apex_x to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apex_x.model import TeacherModel
from apex_x.train import ApexXTrainer
from apex_x.data import (
    CocoDetectionDataset,
    coco_collate_fn,
    MosaicAugmentation,
    MixUpAugmentation,
    build_robust_transforms,
)
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Apex-X on COCO")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/coco_baseline.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/coco",
        help="Root directory for COCO dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/baseline",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_datasets(config: dict, data_root: str):
    """Build training and validation datasets with augmentations.
    
    Args:
        config: Configuration dict
        data_root: Root directory for COCO data
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_root = Path(data_root)
    
    # Training dataset
    train_dataset = CocoDetectionDataset(
        root=data_root / "train2017",
        ann_file=data_root / "annotations" / "instances_train2017.json",
        filter_crowd=True,
        remap_categories=True,
    )
    
    # Apply augmentations if enabled
    aug_config = config.get("augmentation", {})
    
    if aug_config.get("mosaic", False):
        LOGGER.info("Enabling Mosaic augmentation")
        mosaic = MosaicAugmentation(
            dataset=train_dataset,
            output_size=config.get("image_size", 640),
            mosaic_prob=aug_config.get("mosaic_prob", 0.5),
        )
        train_dataset.transforms = mosaic
    
    if aug_config.get("mixup", False):
        LOGGER.info("Enabling MixUp augmentation")
        mixup = MixUpAugmentation(
            dataset=train_dataset,
            alpha=aug_config.get("mixup_alpha", 0.5),
            mixup_prob=aug_config.get("mixup_prob", 0.15),
        )
        # Chain with mosaic if both enabled
        if hasattr(train_dataset, 'transforms'):
            original_transform = train_dataset.transforms
            train_dataset.transforms = lambda x: mixup(original_transform(x))
        else:
            train_dataset.transforms = mixup
    
    # Apply standard albumentations transforms
    if aug_config.get("albumentations", True):
        # Chain augmentations together
        from apex_x.data import Compose
        is_train = True # Assuming this is for training dataset
        transform_list = [
            build_robust_transforms(image_size=config.get("image_size", 640), is_training=is_train)
        ]
        
        if is_train:
            # Add advanced augmentations for training
            from apex_x.data import RandomErasing, GridMask
            transform_list.extend([
                RandomErasing(prob=0.5, min_area=0.02, max_area=0.4),
                GridMask(ratio=0.6, prob=0.3),
            ])
        
        albumentations_transforms = Compose(transform_list) if len(transform_list) > 1 else transform_list[0]
        
        # Chain with existing transforms (mosaic/mixup) if they exist
        if hasattr(train_dataset, 'transforms'):
            original_transform = train_dataset.transforms
            train_dataset.transforms = lambda x: albumentations_transforms(original_transform(x))
        else:
            train_dataset.transforms = albumentations_transforms
    
    # Validation dataset (no augmentations)
    val_dataset = CocoDetectionDataset(
        root=data_root / "val2017",
        ann_file=data_root / "annotations" / "instances_val2017.json",
        filter_crowd=True,
        remap_categories=True,
    )
    
    LOGGER.info(f"Training set: {len(train_dataset)} images")
    LOGGER.info(f"Validation set: {len(val_dataset)} images")
    
    return train_dataset, val_dataset


def build_model(config: dict, device: str) -> TeacherModel:
    """Build Teacher model.
    
    Args:
        config: Configuration dict
        device: Device to create model on
    
    Returns:
        Initialized TeacherModel
    """
    model_config = config.get("model", {})
    
    # Create ApexXConfig from dict (if config file exists)
    from apex_x.config import ApexXConfig
    
    apex_config = ApexXConfig(
        num_classes=80,  # COCO has 80 classes
        image_size=config.get("image_size", 640),
    )
    
    model = TeacherModel(apex_config)
    model = model.to(device)
    
    # Initialize from ImageNet pretrained if specified
    if model_config.get("pretrained", False):
        LOGGER.info("Loading ImageNet pretrained weights (if available)")
        # Load pretrained backbone weights if available
        if model_config.get('pretrained_weights'):
            weights_path = Path(model_config['pretrained_weights'])
            if weights_path.exists():
                LOGGER.info(f'Loading pretrained backbone from {weights_path}')
                # Load only backbone weights (partial loading)
                state_dict = torch.load(weights_path, map_location='cpu')
                # Filter backbone weights
                backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone' in k}
                if backbone_state:
                    # Assuming model has a pv_module attribute for the backbone
                    # This might need adjustment based on actual model structure
                    if hasattr(model, 'pv_module') and hasattr(model.pv_module, 'backbone'):
                        model.pv_module.backbone.load_state_dict(backbone_state, strict=False)
                        LOGGER.info(f'Loaded {len(backbone_state)} backbone parameters')
                    else:
                        LOGGER.warning("Model structure does not have 'pv_module.backbone' for loading pretrained weights.")
                else:
                    LOGGER.warning(f'No backbone weights found in {weights_path}')
            else:
                LOGGER.warning(f'Pretrained weights not found: {weights_path}')
    
    LOGGER.info(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    LOGGER.info(f"Loaded config from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build datasets
    train_dataset, val_dataset = build_datasets(config, args.data_root)
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=coco_collate_fn,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("val_batch_size", 16),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=coco_collate_fn,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    # Build model
    model = build_model(config, args.device)
    
    # Create trainer
    trainer = ApexXTrainer(
        teacher=model,
        device=args.device,
        checkpoint_dir=output_dir / "checkpoints",
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        LOGGER.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Training hyperparameters
    epochs = config.get("epochs", 300)
    base_lr = config.get("base_lr", 0.01)
    warmup_epochs = config.get("warmup_epochs", 5)
    weight_decay = config.get("weight_decay", 0.0005)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    
    # Setup learning rate scheduler
    from apex_x.train import LinearWarmupCosineAnnealingLR
    
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        eta_min=base_lr * 0.01,
    )
    
    # Training loop
    best_map = 0.0
    
    LOGGER.info("=" * 80)
    LOGGER.info("Starting training")
    LOGGER.info(f"  Total epochs: {epochs}")
    LOGGER.info(f"  Batch size: {config.get('batch_size', 16)}")
    LOGGER.info(f"  Base LR: {base_lr}")
    LOGGER.info(f"  Device: {args.device}")
    LOGGER.info("=" * 80)
    
    for epoch in range(epochs):
        LOGGER.info(f"\nEpoch {epoch + 1}/{epochs}")
        LOGGER.info("-" * 80)
        
        # Training
        train_metrics = trainer.train_epoch(
            train_loader,
            optimizer,
            epoch=epoch,
            total_epochs=epochs,
        )
        
        LOGGER.info(f"Train - Loss: {train_metrics.get('total_loss', 0.0):.4f}, "
                   f"Cls: {train_metrics.get('cls_loss', 0.0):.4f}, "
                   f"Box: {train_metrics.get('box_loss', 0.0):.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        LOGGER.info(f"Learning rate: {current_lr:.6f}")
        
        # Validation every N epochs
        val_interval = config.get("val_interval", 5)
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            LOGGER.info("Running validation...")
            val_metrics = trainer.validate(
                val_loader,
                device=args.device,
                compute_map=True,
            )
            
            current_map = val_metrics.get("mAP_bbox", 0.0)
            LOGGER.info(f"Val - mAP: {current_map:.4f}, "
                       f"AP50: {val_metrics.get('mAP_50_bbox', 0.0):.4f}, "
                       f"AP75: {val_metrics.get('mAP_75_bbox', 0.0):.4f}")
            LOGGER.info(f"      APs: {val_metrics.get('mAP_small_bbox', 0.0):.4f}, "
                       f"APm: {val_metrics.get('mAP_medium_bbox', 0.0):.4f}, "
                       f"APl: {val_metrics.get('mAP_large_bbox', 0.0):.4f}")
            
            # Save best checkpoint
            if current_map > best_map:
                best_map = current_map
                trainer.save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    epoch=epoch,
                    metrics=val_metrics,
                )
                LOGGER.info(f"✓ Saved new best checkpoint (mAP: {best_map:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch+1}.pt",
                epoch=epoch,
                metrics=train_metrics,
            )
    
    LOGGER.info("=" * 80)
    LOGGER.info("Training complete!")
    LOGGER.info(f"Best mAP: {best_map:.4f}")
    LOGGER.info(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
