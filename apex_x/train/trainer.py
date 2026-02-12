"""Staged training flow for Apex-X v4 CPU baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from apex_x.config import ApexXConfig
from apex_x.data import SatelliteDataset, build_robust_transforms
from apex_x.model import (
    ApexXModel,
    DetHead,
    DualPathFPN,
    PVModule,
    TeacherModel,
    TimmBackboneAdapter,
)
from apex_x.routing import (
    BudgetDualController,
    build_routing_diagnostics,
    compute_oracle_delta_targets,
    deterministic_greedy_selection,
    deterministic_two_stage_selection,
    sample_oracle_set,
    ste_gate_from_utilities,
    summarize_oracle_delta_targets,
    utility_oracle_loss,
)
from apex_x.runtime import heavy_ops_autocast_context, resolve_precision_policy
from apex_x.tiles import tile_grid_shape
from apex_x.utils import get_logger, log_event, seed_all

from apex_x.eval import COCOEvaluator, PYCOCOTOOLS_AVAILABLE

from .checkpoint import CheckpointMetadata, cleanup_old_checkpoints, load_checkpoint, save_checkpoint
from .lr_scheduler import create_lr_scheduler
from .pcgrad import apply_pcgradpp, diagnostics_to_dict
from .train_losses import compute_teacher_training_loss
from .qat import QuantizationSummary, prepare_int8_ptq, prepare_int8_qat
from .trainer_utils import add_train_epoch_method

LOGGER = get_logger(__name__)

StageMetricValue = float | int | bool | str | None


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_id: int
    name: str
    metrics: Mapping[str, StageMetricValue]


@dataclass(frozen=True, slots=True)
class StagedTrainResult:
    stage_results: tuple[StageResult, ...]
    routing_diagnostics: dict[str, Any]
    train_summary: dict[str, Any]
    loss_proxy: float
    final_mu: float


@add_train_epoch_method
class ApexXTrainer:
    """Implements stage-0..4 trainer flow defined in Apex-X engineering docs."""

    def __init__(
        self,
        config: ApexXConfig | None = None,
        *,
        num_classes: int = 3,
        backbone_type: str = "pv",
        backbone_name: str = "efficientnet_b5",
        pretrained_backbone: bool = True,
        checkpoint_dir: Path | str | None = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.config = config or ApexXConfig()
        self.config.validate()

        # Gradient accumulation for larger effective batch size
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        if self.gradient_accumulation_steps > 1:
            LOGGER.info(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = int(num_classes)
        self.backbone_type = backbone_type
        self.backbone_name = backbone_name
        self.pretrained_backbone = pretrained_backbone

        self.baseline_model = ApexXModel(config=self.config)
        self.teacher = self._build_teacher_model(num_classes=self.num_classes)
        self.teacher.train()

        self.dual_controller = BudgetDualController(
            budget=self.config.routing.budget_total,
            mu_init=self.config.train.mu_init,
            mu_lr=self.config.train.mu_lr,
            mu_min=self.config.train.mu_min,
            mu_max=self.config.train.mu_max,
            adaptive_lr=self.config.train.dual_adaptive_lr,
            lr_decay=self.config.train.dual_lr_decay,
            delta_clip=self.config.train.dual_delta_clip,
            deadband_ratio=self.config.train.dual_deadband_ratio,
            error_ema_beta=self.config.train.dual_error_ema_beta,
            adaptive_lr_min_scale=self.config.train.dual_lr_min_scale,
            adaptive_lr_max_scale=self.config.train.dual_lr_max_scale,
            logger_name="train.staged.dual",
        )
        self.mu_history: list[float] = [float(self.dual_controller.mu)]
        self.quantization_summary = QuantizationSummary.disabled()
        self._quantization_prepared = False
        self.precision_policy = resolve_precision_policy(self.config)
        
        # AMP (Automatic Mixed Precision) support
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            LOGGER.info("AMP enabled: 2-3x speedup, 50% memory reduction")
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.best_metric = float('-inf')  # Higher is better for AP
        self.best_metric_name = "mAP_segm"  # Track mask AP
        self.current_epoch = 0
        self.val_interval = 5  # Validate every 5 epochs
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def _build_teacher_model(self, *, num_classes: int) -> TeacherModel:
        # Keep the staged trainer CPU-fast by default, or use high-capacity backbone
        if self.backbone_type == "timm":
            pv_module = TimmBackboneAdapter(
                model_name=self.backbone_name,
                pretrained=self.pretrained_backbone,
                out_indices=(2, 3, 4), # P3, P4, P5
            )
            # Adapt output channels from the backbone
            p3_ch = pv_module.p3_channels
            p4_ch = pv_module.p4_channels
            p5_ch = pv_module.p5_channels
        else:
            pv_module = PVModule(
                in_channels=3,
                p3_channels=16,
                p4_channels=24,
                p5_channels=32,
                coarse_level="P4",
            )
            p3_ch = 16
            p4_ch = 24
            p5_ch = 32

        fpn = DualPathFPN(
            pv_p3_channels=p3_ch,
            pv_p4_channels=p4_ch,
            pv_p5_channels=p5_ch,
            ff_channels=16,
            out_channels=16,
        )
        det_head = DetHead(
            in_channels=16,
            num_classes=num_classes,
            hidden_channels=16,
            depth=1,
        )
        return TeacherModel(
            num_classes=num_classes,
            config=self.config,
            pv_module=pv_module,
            fpn=fpn,
            det_head=det_head,
            feature_layers=("P3", "P4"),
            use_ema=True,
            ema_decay=0.99,
            use_ema_for_forward=False,
        )

    def _l0_grid(self) -> tuple[int, int, int]:
        ff_h = self.config.model.input_height // self.config.model.ff_primary_stride
        ff_w = self.config.model.input_width // self.config.model.ff_primary_stride
        grid_h, grid_w = tile_grid_shape(ff_h, ff_w, self.config.model.tile_size_l0)
        return grid_h, grid_w, grid_h * grid_w

    def _stage0_baseline_warmup(self, rng: np.random.RandomState, *, steps: int) -> StageResult:
        det_score_last = 0.0
        selected_total = 0
        for _ in range(steps):
            image = rng.rand(
                1,
                3,
                self.config.model.input_height,
                self.config.model.input_width,
            ).astype(np.float32)
            out = self.baseline_model.forward(image)
            det_score_last = float(out["det"]["scores"][0])
            selected_total += len(out["selected_tiles"])

        mean_selected = selected_total / max(steps, 1)
        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "det_score_last": det_score_last,
            "selected_tiles_mean": float(mean_selected),
        }
        return StageResult(stage_id=0, name="baseline_warmup", metrics=metrics)


    def save_training_checkpoint(
        self,
        epoch: int,
        step: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> None:
        """Save a training checkpoint.
        
        Args:
            epoch: Current epoch number
            step: Current training step
            optimizer: Optimizer to save
            scheduler: Optional LR scheduler
            metrics: Training/validation metrics
            is_best: Whether this is the best checkpoint
        """
        if self.checkpoint_dir is None:
            LOGGER.warning("checkpoint_dir not set, skipping checkpoint save")
            return
        
        # Build checkpoint filename
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        
        # Get EMA state if available
        ema_state = None
        if self.teacher.use_ema:
            ema_state = {
                "ema_pv_module": self.teacher.ema_pv_module.state_dict() if self.teacher.ema_pv_module else None,
                "ema_fpn": self.teacher.ema_fpn.state_dict() if self.teacher.ema_fpn else None,
                "ema_det_head": self.teacher.ema_det_head.state_dict() if self.teacher.ema_det_head else None,
            }
        
        # Save checkpoint
        save_checkpoint(
            path=ckpt_path,
            model=self.teacher,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            scheduler=scheduler,
            metrics=metrics,
            best_metric=self.best_metric,
            best_metric_name=self.best_metric_name,
            config=self.config.to_dict(),
            is_best=is_best,
            ema_state_dict=ema_state,
        )
        
        # Cleanup old checkpoints (keep last 3 + best)
        cleanup_old_checkpoints(
            checkpoint_dir=self.checkpoint_dir,
            keep_best=True,
            keep_last_n=3,
        )
        
        # Update last checkpoint symlink
        last_path = self.checkpoint_dir / "last.pt"
        if last_path.exists() and last_path.is_symlink():
            last_path.unlink()
        elif last_path.exists():
            last_path.unlink()
        
        try:
            last_path.symlink_to(ckpt_path.name)
        except OSError:
            import shutil
            shutil.copy2(ckpt_path, last_path)
    
    def load_training_checkpoint(
        self,
        checkpoint_path: Path | str,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str = "cpu",
    ) -> CheckpointMetadata:
        """Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            device: Device to load checkpoint to
        
        Returns:
            CheckpointMetadata with training state
        """
        metadata = load_checkpoint(
            path=Path(checkpoint_path),
            model=self.teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        
        # Restore EMA state if present
        if hasattr(metadata, "ema_state_dict"):
            ema_dict = getattr(metadata, "ema_state_dict", None)
            if ema_dict and self.teacher.use_ema:
                if ema_dict.get("ema_pv_module") and self.teacher.ema_pv_module:
                    self.teacher.ema_pv_module.load_state_dict(ema_dict["ema_pv_module"])
                if ema_dict.get("ema_fpn") and self.teacher.ema_fpn:
                    self.teacher.ema_fpn.load_state_dict(ema_dict["ema_fpn"])
                if ema_dict.get("ema_det_head") and self.teacher.ema_det_head:
                    self.teacher.ema_det_head.load_state_dict(ema_dict["ema_det_head"])
                LOGGER.info("Restored EMA model state from checkpoint")
        
        # Restore best metric tracking
        self.best_metric = metadata.best_metric
        self.best_metric_name = metadata.best_metric_name
        
        LOGGER.info(f"Loaded checkpoint from epoch {metadata.epoch}, step {metadata.step}")
        return metadata



    def validate(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
        max_batches: int | None = None,
        compute_map: bool = True,
        conf_threshold: float = 0.001,
        nms_threshold: float = 0.65,
    ) -> dict[str, float]:
        """Run validation and compute metrics including mAP.
        
        Args:
            val_dataloader: DataLoader for validation dataset
            device: Device to run validation on
            max_batches: Optional limit on number of batches to evaluate
            compute_map: Whether to compute mAP (requires COCO annotations)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
        
        Returns:
            Dict of validation metrics including mAP if available
        """
        from apex_x.model import post_process_detections
        from apex_x.eval import COCOEvaluator
        
        self.teacher.eval()
        self.teacher.to(device)
        
        # Initialize COCO evaluator if computing mAP
        evaluator = None
        if compute_map:
            try:
                # Try to get COCO annotations from dataset
                if hasattr(val_dataloader.dataset, 'coco'):
                    evaluator = COCOEvaluator(val_dataloader.dataset.coco)
                    LOGGER.info("Initialized COCO evaluator for mAP computation")
            except Exception as e:
                LOGGER.warning(f"Could not initialize COCO evaluator: {e}")
                compute_map = False
        
        all_image_ids = []
        total_batches = 0
        total_loss = 0.0
        
        LOGGER.info("Starting validation...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Extract batch components
                if isinstance(batch_data, dict):
                    images = batch_data.get("images")
                    image_ids = batch_data.get("image_ids", [])
                else:
                    # Fallback for simple batch format
                    images = batch_data
                    image_ids = []
                
                if images is None:
                    continue
                
                # Move to device
                images = images.to(device)
                
                # Model forward pass
                output = self.teacher(images)
                
                # Compute validation loss (simplified)
                if hasattr(output, 'logits'):
                    val_loss = output.logits.pow(2).mean()
                    if hasattr(output, 'boundaries'):
                        val_loss = val_loss + 0.05 * output.boundaries.abs().mean()
                    total_loss += float(val_loss.item())
                
                # Post-process detections for mAP
                if compute_map and evaluator is not None:
                    # Apply NMS and get final detections
                    detections = post_process_detections(
                        cls_logits_by_level=output.logits_by_level,
                        box_reg_by_level=output.boxes_by_level,
                        quality_by_level=output.quality_by_level,
                        conf_threshold=conf_threshold,
                        nms_threshold=nms_threshold,
                        box_format="distance",  # or "direct" depending on head implementation
                    )
                    
                    # Convert to COCO format and update evaluator
                    for det_idx, detection in enumerate(detections):
                        if det_idx < len(image_ids):
                            img_id = image_ids[det_idx]
                            
                            # Convert to COCO format
                            coco_dets = []
                            boxes = detection['boxes'].cpu().numpy()
                            scores = detection['scores'].cpu().numpy()
                            classes = detection['classes'].cpu().numpy()
                            
                            for box, score, cls_id in zip(boxes, scores, classes):
                                x1, y1, x2, y2 = box
                                coco_dets.append({
                                    'image_id': int(img_id),
                                    'category_id': int(cls_id),
                                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                    'score': float(score),
                                })
                            
                            evaluator.update(coco_dets)
                            all_image_ids.append(img_id)
                
                total_batches += 1
                
                if (batch_idx + 1) % 100 == 0:
                    LOGGER.info(f"Processed {batch_idx + 1} batches...")
        
        self.teacher.train()
        
        # Compute metrics
        avg_loss = total_loss / max(1, total_batches)
        
        metrics = {
            "val_batches": total_batches,
            "val_loss": avg_loss,
        }
        
        # Compute mAP if evaluator available
        if evaluator is not None and len(all_image_ids) > 0:
            LOGGER.info("Computing COCO metrics...")
            try:
                coco_metrics = evaluator.compute()
                metrics.update(coco_metrics)
                LOGGER.info(f"mAP: {coco_metrics.get('mAP', 0.0):.4f}")
            except Exception as e:
                LOGGER.error(f"Failed to compute COCO metrics: {e}")
        
        LOGGER.info(f"Validation complete: {total_batches} batches, avg_loss={avg_loss:.4f}")
        return metrics


    def _stage1_teacher_training(
        self,
        generator: torch.Generator,
        *,
        steps: int,
        dataset_path: str | Path | None = None,
    ) -> tuple[StageResult, float]:
        optimizer = torch.optim.AdamW(self.teacher.parameters(), lr=1e-3)
        
        # Create LR scheduler with warmup
        scheduler = create_lr_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine_warmup",
            total_steps=steps,
            warmup_ratio=0.1,  # 10% warmup
            min_lr=1e-6,
        )
        train_h = min(self.config.model.input_height, 512)
        train_w = min(self.config.model.input_width, 512)
        
        dataloader = None
        if dataset_path:
            dataset = SatelliteDataset(
                root_dir=dataset_path,
                tile_size=max(train_h, 256),
                stride=max(train_h // 2, 128),
            )
            if len(dataset) > 0:
                # Basic PyTorch DataLoader
                dataloader = torch.utils.data.DataLoader(
                     dataset, 
                     batch_size=1, # Single image for now to avoid collage logic complexity here
                     shuffle=True,
                     collate_fn=lambda x: x, # Return list of samples
                )
                
        transforms = build_robust_transforms(
            height=train_h, 
            width=train_w,
            blur_prob=0.3,
            noise_prob=0.3,
            distort_prob=0.3
        )
        rng = np.random.RandomState(42)

        loss_last = 0.0
        
        # If dataloader exists, we iterate it. Else random loop.
        iterator = iter(dataloader) if dataloader else None
        
        for step_idx in range(steps):
            if iterator:
                try:
                    samples = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    samples = next(iterator)
                
                # Apply transforms
                batch_images = []
                for s in samples:
                    # s is TransformSample. 
                    # Augment
                    aug = transforms(s, rng=rng)
                    # Convert to tensor [C, H, W]
                    img_t = torch.from_numpy(aug.image).permute(2, 0, 1).float() / 255.0
                    batch_images.append(img_t)
                
                image = torch.stack(batch_images, dim=0)

            else:
                image = torch.rand((1, 3, train_h, train_w), generator=generator, dtype=torch.float32)

            samples_for_loss = samples if iterator else None
            
            # Only zero gradients at start of accumulation cycle
            is_accumulation_step = (step_idx % self.gradient_accumulation_steps) != 0
            should_step = (step_idx + 1) % self.gradient_accumulation_steps == 0

            if self.use_amp and self.scaler is not None:
                # Zero gradients only at start of cycle
                if not is_accumulation_step:
                    optimizer.zero_grad(set_to_none=True)
                    
                with torch.cuda.amp.autocast():
                    # Forward pass with AMP
                    out = self.teacher(image, use_ema=False)
                    # Compute training loss with real detection outputs
                    total_loss, _ = compute_teacher_training_loss(
                        output=out,
                        samples=samples_for_loss,
                        det_weight=1.0,
                        boundary_weight=0.05,
                    )
                    
                    # Scale loss by accumulation steps
                    total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward with scaler (accumulate gradients)
                self.scaler.scale(total_loss).backward()
                
                # Only step optimizer after accumulating enough gradients
                if should_step:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=10.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                # Zero gradients only at start of cycle
                if not is_accumulation_step:
                    optimizer.zero_grad(set_to_none=True)
                    
                with heavy_ops_autocast_context(self.precision_policy):
                    out = self.teacher(image, use_ema=False)
                    # Compute training loss with real detection outputs
                    total_loss, _ = compute_teacher_training_loss(
                        output=out,
                        samples=samples_for_loss,
                        det_weight=1.0,
                        boundary_weight=0.05,
                    )
                    
                    # Scale loss by accumulation steps
                    total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward (accumulate gradients)
                total_loss.backward()
                
                # Only step optimizer after accumulating enough gradients
                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=10.0)
                    optimizer.step()
            
            # Update scheduler and EMA only when we actually step
            if should_step:
                scheduler.step()
                self.teacher.update_ema()
                
            loss_last = float(total_loss.detach().item() * self.gradient_accumulation_steps)  # Unscale for logging
            
            # Log current LR (only do this periodically to avoid spam)
            if step_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                LOGGER.debug(f"Step {step_idx}/{steps}: loss={loss_last:.4f}, lr={current_lr:.6f}")

        # Get final LR
        final_lr = scheduler.get_last_lr()[0]
        
        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "teacher_loss_last": loss_last,
            "final_lr": float(final_lr),
            "full_compute_mode": bool(self.teacher.full_compute_mode),
            "qat_wrapped_modules": self.quantization_summary.wrapped_modules,
            "heavy_ops_dtype": self.precision_policy.to_dict()["heavy_ops_dtype"],
            "fp8_enabled": self.precision_policy.fp8_enabled,
        }
        return StageResult(stage_id=1, name="teacher_full_compute", metrics=metrics), loss_last

    def _build_calibration_inputs(
        self,
        generator: torch.Generator,
        *,
        batches: int,
    ) -> list[Tensor]:
        if batches <= 0:
            raise ValueError("batches must be > 0")
        train_h = min(self.config.model.input_height, 64)
        train_w = min(self.config.model.input_width, 64)
        return [
            torch.rand((1, 3, train_h, train_w), generator=generator, dtype=torch.float32)
            for _ in range(batches)
        ]

    def _prepare_quantization(self, generator: torch.Generator) -> None:
        if self._quantization_prepared:
            return
        cfg = self.config
        if cfg.train.qat_enable and cfg.train.qat_int8:
            self.quantization_summary = prepare_int8_qat(self.teacher)
        elif cfg.runtime.precision_profile == "edge":
            calibration_inputs = self._build_calibration_inputs(generator, batches=4)
            self.quantization_summary = prepare_int8_ptq(
                self.teacher,
                calibration_inputs=calibration_inputs,
                forward_fn=lambda module, batch: cast(TeacherModel, module)(
                    batch,
                    use_ema=False,
                ),
            )
        else:
            self.quantization_summary = QuantizationSummary.disabled()
        self._quantization_prepared = True
        log_event(
            LOGGER,
            "quantization_prepare",
            level="DEBUG",
            fields={
                "mode": self.quantization_summary.mode,
                "wrapped_modules": self.quantization_summary.wrapped_modules,
                "calibration_batches": self.quantization_summary.calibration_batches,
                "router_gating_fp16": self.quantization_summary.router_gating_fp16,
                "precision_policy": self.precision_policy.to_dict(),
            },
        )

    def _stage2_oracle_bootstrapping(
        self,
        rng: np.random.RandomState,
        generator: torch.Generator,
        *,
        steps: int,
    ) -> tuple[StageResult, float]:
        _, _, k_tiles = self._l0_grid()
        if k_tiles <= 0:
            raise ValueError("k_tiles must be > 0")

        loss_last = 0.0
        sampled_count_last = 0
        sampled_random_last = 0
        sampled_uncertainty_last = 0
        sampled_long_tail_last = 0
        oracle_delta_mean_last = 0.0
        oracle_delta_std_last = 0.0
        oracle_delta_abs_p95_last = 0.0
        oracle_delta_clipped_ratio_last = 0.0
        oracle_delta_abs_p95_sum = 0.0
        oracle_delta_clipped_ratio_sum = 0.0
        clamp_abs = 2.0
        for step in range(steps):
            uncertainty = np.abs(rng.standard_normal(k_tiles)).tolist()

            cheap = 0.4 + torch.rand((1, k_tiles), generator=generator, dtype=torch.float32)
            heavy_noise = 0.2 * torch.rand(
                (1, k_tiles),
                generator=generator,
                dtype=torch.float32,
            )
            heavy = (cheap - heavy_noise).clamp(min=0.0)
            long_tail_scores = [
                float(value) for value in (cheap - heavy).detach().abs().reshape(-1).cpu().tolist()
            ]

            sampled = sample_oracle_set(
                uncertainty,
                random_fraction=0.2,
                uncertainty_fraction=0.2,
                long_tail_fraction=0.1,
                long_tail_scores=long_tail_scores,
                seed=int(rng.randint(0, 10_000_000)),
            )
            sampled_count_last = len(sampled.indices)
            sampled_random_last = len(sampled.random_indices)
            sampled_uncertainty_last = len(sampled.uncertainty_indices)
            sampled_long_tail_last = len(sampled.long_tail_indices)
            sampled_idx = torch.as_tensor(sampled.indices, dtype=torch.int64).unsqueeze(0)
            raw_sampled_delta = torch.gather(cheap - heavy, dim=1, index=sampled_idx)
            oracle_targets = compute_oracle_delta_targets(
                cheap_distill_loss=cheap,
                heavy_distill_loss=heavy,
                sampled_tile_indices=sampled_idx,
                clamp_abs=clamp_abs,
            )
            delta_stats = summarize_oracle_delta_targets(
                oracle_targets.delta_targets,
                raw_delta_targets=raw_sampled_delta,
                clamp_abs=clamp_abs,
            )
            oracle_delta_mean_last = delta_stats.mean
            oracle_delta_std_last = delta_stats.std
            oracle_delta_abs_p95_last = delta_stats.abs_p95
            oracle_delta_clipped_ratio_last = delta_stats.clipped_ratio
            oracle_delta_abs_p95_sum += delta_stats.abs_p95
            oracle_delta_clipped_ratio_sum += delta_stats.clipped_ratio

            utility_logits = torch.randn((1, k_tiles), generator=generator, dtype=torch.float32)
            utility_logits.requires_grad_(True)
            oracle_loss = utility_oracle_loss(
                utility_logits=utility_logits,
                targets=oracle_targets,
                regression_weight=1.0,
                ranking_weight=0.25,
                regression_type="smooth_l1",
            )
            oracle_loss.total_loss.backward()
            loss_last = float(oracle_loss.total_loss.detach().item())

            log_event(
                LOGGER,
                "stage2_oracle_step",
                level="DEBUG",
                fields={
                    "step": step,
                    "sampled_tiles": sampled_count_last,
                    "sampled_random": sampled_random_last,
                    "sampled_uncertainty": sampled_uncertainty_last,
                    "sampled_long_tail": sampled_long_tail_last,
                    "oracle_loss": round(loss_last, 6),
                    "oracle_delta_abs_p95": round(oracle_delta_abs_p95_last, 6),
                    "oracle_delta_clipped_ratio": round(oracle_delta_clipped_ratio_last, 6),
                },
            )

        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "oracle_loss_last": loss_last,
            "sampled_tiles_last": int(sampled_count_last),
            "sampled_random_last": int(sampled_random_last),
            "sampled_uncertainty_last": int(sampled_uncertainty_last),
            "sampled_long_tail_last": int(sampled_long_tail_last),
            "oracle_delta_mean_last": float(oracle_delta_mean_last),
            "oracle_delta_std_last": float(oracle_delta_std_last),
            "oracle_delta_abs_p95_last": float(oracle_delta_abs_p95_last),
            "oracle_delta_clipped_ratio_last": float(oracle_delta_clipped_ratio_last),
            "oracle_delta_abs_p95_avg": float(oracle_delta_abs_p95_sum / max(steps, 1)),
            "oracle_delta_clipped_ratio_avg": float(oracle_delta_clipped_ratio_sum / max(steps, 1)),
        }
        return StageResult(stage_id=2, name="oracle_bootstrap", metrics=metrics), loss_last

    def _stage3_continuous_budgeting(
        self,
        generator: torch.Generator,
        *,
        steps: int,
    ) -> tuple[StageResult, float, list[float]]:
        _, _, k_tiles = self._l0_grid()
        mu_before = float(self.dual_controller.mu)
        expected_cost_last = 0.0
        budget_loss_last = 0.0
        utility_snapshot: list[float] = []

        for _ in range(steps):
            utilities = torch.randn((1, k_tiles), generator=generator, dtype=torch.float32)
            utilities.requires_grad_(True)
            utilities_fp16 = utilities.to(dtype=torch.float16)
            probabilities, _gates = ste_gate_from_utilities(
                utilities_fp16,
                threshold=self.config.routing.theta_on,
                mode="threshold",
            )
            expected_cost = self.dual_controller.expected_cost(
                probabilities=probabilities.to(dtype=torch.float32),
                c_heavy=self.config.routing.cost_heavy,
                c_cheap=self.config.routing.cost_cheap,
            )
            if not isinstance(expected_cost, Tensor):
                raise TypeError("expected_cost must be Tensor for tensor probabilities")
            budget_loss = self.dual_controller.budget_loss(
                expected_cost=expected_cost,
                budget=float(self.config.routing.budget_total),
            )
            if not isinstance(budget_loss, Tensor):
                raise TypeError("budget_loss must be Tensor for tensor expected_cost")

            total_loss = utilities.square().mean() + budget_loss
            total_loss.backward()

            expected_cost_last = float(expected_cost.detach().item())
            budget_loss_last = float(budget_loss.detach().item())
            utility_snapshot = [float(value) for value in utilities.detach().reshape(-1).tolist()]
            self.dual_controller.update_mu(
                expected_cost=expected_cost_last,
                budget=float(self.config.routing.budget_total),
            )
            self.mu_history.append(float(self.dual_controller.mu))

        metrics: dict[str, StageMetricValue] = {
            "steps": int(steps),
            "expected_cost_last": expected_cost_last,
            "budget_loss_last": budget_loss_last,
            "mu_before": mu_before,
            "mu_after": float(self.dual_controller.mu),
            "dual_effective_lr_last": float(self.dual_controller.last_effective_lr),
            "dual_error_ema_last": float(self.dual_controller.error_ema),
            "dual_update_count": int(self.dual_controller.update_count),
        }
        return (
            StageResult(stage_id=3, name="continuous_budgeting", metrics=metrics),
            budget_loss_last,
            utility_snapshot,
        )

    def _pcgrad_monitoring_snapshot(self) -> dict[str, Any]:
        if not self.config.train.pcgradpp_enabled():
            snapshot = {
                "enabled": False,
                "group_names": [],
                "projected_pairs": 0,
                "conflicting_pairs": 0,
                "conflicting_pairs_after": 0,
                "total_pairs": 0,
                "conflict_rate_before": 0.0,
                "conflict_rate_after": 0.0,
                "shared_param_count": 0,
                "head_param_count": 0,
                "shared_grad_norm": 0.0,
                "head_grad_norm": 0.0,
            }
            log_event(
                LOGGER,
                "pcgrad_monitoring",
                level="DEBUG",
                fields=snapshot,
            )
            return snapshot

        shared = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))
        det_head_w = torch.nn.Parameter(torch.ones((1, 1), dtype=torch.float32))
        det_head_b = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        seg_head_w = torch.nn.Parameter(torch.ones((1, 1), dtype=torch.float32))
        seg_head_b = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))

        x = torch.ones((1, 1), dtype=torch.float32)
        trunk = x @ shared.t()
        det_pred = trunk @ det_head_w.t() + det_head_b
        seg_pred = trunk @ seg_head_w.t() + seg_head_b

        losses = {
            "det_cls": (det_pred - 2.0).pow(2).mean(),
            "det_box": det_pred.pow(2).mean() * 0.0,
            "seg_mask": (seg_pred + 1.0).pow(2).mean(),
            "seg_boundary": seg_pred.pow(2).mean() * 0.0,
        }
        diagnostics = apply_pcgradpp(
            loss_terms=losses,
            shared_params=[shared],
            head_params=[det_head_w, det_head_b, seg_head_w, seg_head_b],
        )
        snapshot = diagnostics_to_dict(diagnostics)
        snapshot["enabled"] = True
        snapshot["shared_grad_norm"] = (
            float(shared.grad.detach().norm().item()) if shared.grad is not None else 0.0
        )
        head_grad_norm = 0.0
        for parameter in (det_head_w, det_head_b, seg_head_w, seg_head_b):
            if parameter.grad is not None:
                head_grad_norm += float(parameter.grad.detach().norm().item())
        snapshot["head_grad_norm"] = float(head_grad_norm)

        log_event(
            LOGGER,
            "pcgrad_monitoring",
            level="DEBUG",
            fields=snapshot,
        )
        return snapshot

    def _stage4_deterministic_emulation(
        self,
        utility_snapshot: list[float],
    ) -> tuple[StageResult, dict[str, Any]]:
        grid_h, grid_w, k_tiles = self._l0_grid()
        if len(utility_snapshot) != k_tiles:
            raise ValueError("utility_snapshot length must match tile count")

        delta_cost = max(self.config.routing.cost_heavy - self.config.routing.cost_cheap, 1e-9)
        delta_costs = [float(delta_cost) for _ in range(k_tiles)]

        if self.config.model.effective_nesting_depth() >= 1:
            split_utilities = [max(float(value), 0.0) + 1e-4 for value in utility_snapshot]
            split_overheads = [1.0 for _ in range(k_tiles)]
            two_stage = deterministic_two_stage_selection(
                l0_utilities=utility_snapshot,
                l0_delta_costs=delta_costs,
                split_utilities=split_utilities,
                split_overheads=split_overheads,
                budget_b1=float(self.config.routing.budget_b1),
                budget_b2=float(self.config.routing.budget_b2),
                kmax_l0=int(self.config.model.kmax_l0),
                kmax_l1=int(self.config.model.kmax_l1),
                l0_grid_h=grid_h,
                l0_grid_w=grid_w,
                l1_order_mode="hilbert",
            )
            l0_selected = two_stage.l0.selected_indices
            l1_selected = two_stage.l1_ordered_indices
            spent_b1 = float(two_stage.l0.spent_budget)
            spent_b2 = float(two_stage.split_spent_budget)
        else:
            greedy = deterministic_greedy_selection(
                utilities=utility_snapshot,
                delta_costs=delta_costs,
                budget=float(self.config.routing.budget_b1),
                kmax=int(self.config.model.kmax_l0),
            )
            l0_selected = greedy.selected_indices
            l1_selected = []
            spent_b1 = float(greedy.spent_budget)
            spent_b2 = 0.0

        l1_total = k_tiles * 4 if self.config.model.effective_nesting_depth() >= 1 else 0
        routing_diag = build_routing_diagnostics(
            utilities_by_level={"l0": utility_snapshot, "l1": []},
            selected_counts={"l0": len(l0_selected), "l1": len(l1_selected), "l2": 0},
            total_counts={"l0": k_tiles, "l1": l1_total, "l2": 0},
            budget_used={
                "b1": spent_b1,
                "b2": spent_b2,
                "b3": 0.0,
                "total": spent_b1 + spent_b2,
            },
            budget_total={
                "b1": float(self.config.routing.budget_b1),
                "b2": float(self.config.routing.budget_b2),
                "b3": float(self.config.routing.budget_b3),
                "total": float(self.config.routing.budget_total),
            },
            mu_history=self.mu_history,
        )

        metrics: dict[str, StageMetricValue] = {
            "selected_l0": int(len(l0_selected)),
            "selected_l1": int(len(l1_selected)),
            "spent_b1": spent_b1,
            "spent_b2": spent_b2,
        }
        return (
            StageResult(stage_id=4, name="deterministic_emulation", metrics=metrics),
            routing_diag,
        )

    def run(
        self,
        *,
        steps_per_stage: int = 1,
        seed: int = 0,
        enable_budgeting: bool = True,
        dataset_path: str | Path | None = None,
    ) -> StagedTrainResult:
        if steps_per_stage <= 0:
            raise ValueError("steps_per_stage must be > 0")

        seed_all(seed=seed, deterministic=self.config.runtime.deterministic)
        rng = np.random.RandomState(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self._prepare_quantization(generator)

        stage0 = self._stage0_baseline_warmup(rng, steps=steps_per_stage)
        stage1, stage1_loss = self._stage1_teacher_training(
            generator, 
            steps=steps_per_stage, 
            dataset_path=dataset_path
        )
        stage2, stage2_loss = self._stage2_oracle_bootstrapping(
            rng,
            generator,
            steps=steps_per_stage,
        )
        if enable_budgeting:
            stage3, stage3_loss, utility_snapshot = self._stage3_continuous_budgeting(
                generator,
                steps=steps_per_stage,
            )
        else:
            _, _, k_tiles = self._l0_grid()
            utility_snapshot = [float(v) for v in rng.standard_normal(k_tiles).tolist()]
            stage3 = StageResult(
                stage_id=3,
                name="continuous_budgeting",
                metrics={
                    "steps": int(steps_per_stage),
                    "expected_cost_last": 0.0,
                    "budget_loss_last": 0.0,
                    "mu_before": float(self.dual_controller.mu),
                    "mu_after": float(self.dual_controller.mu),
                    "budgeting_enabled": False,
                },
            )
            stage3_loss = 0.0
        stage4, routing_diag = self._stage4_deterministic_emulation(utility_snapshot)
        pcgrad_snapshot = self._pcgrad_monitoring_snapshot()

        train_summary = {
            "routing_diagnostics": routing_diag,
            "training_toggles": {
                "distill_enabled": self.config.train.distill_enabled(),
                "pcgradpp_enabled": self.config.train.pcgradpp_enabled(),
            },
            "pcgrad": pcgrad_snapshot,
            "quantization": {
                "mode": self.quantization_summary.mode,
                "wrapped_modules": self.quantization_summary.wrapped_modules,
                "calibration_batches": self.quantization_summary.calibration_batches,
                "router_gating_fp16": self.quantization_summary.router_gating_fp16,
            },
            "precision": self.precision_policy.to_dict(),
        }
        loss_proxy = float(stage1_loss + stage2_loss + abs(stage3_loss))
        result = StagedTrainResult(
            stage_results=(stage0, stage1, stage2, stage3, stage4),
            routing_diagnostics=routing_diag,
            train_summary=train_summary,
            loss_proxy=loss_proxy,
            final_mu=float(self.dual_controller.mu),
        )

        log_event(
            LOGGER,
            "staged_training_complete",
            fields={
                "stages": len(result.stage_results),
                "loss_proxy": round(result.loss_proxy, 6),
                "final_mu": round(result.final_mu, 6),
            },
        )

        # Save checkpoint and config
        output_dir = Path(self.config.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / "teacher_checkpoint.pt"
        torch.save(self.teacher.state_dict(), checkpoint_path)

        # config_path = output_dir / "config.yaml"
        # Simple dict dump for now; proper YAML requires extra deps

        # We'll use json for config to avoid extra deps if needed, but the plan said config.yaml
        # Let's try to just write the dict as json for simplicity and robustness in this env
        import json

        with open(output_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        log_event(
            LOGGER,
            "training_artifacts_saved",
            fields={
                "checkpoint": str(checkpoint_path),
                "config": str(output_dir / "config.json"),
            },
        )

        return result


__all__ = [
    "StageResult",
    "StagedTrainResult",
    "ApexXTrainer",
]
