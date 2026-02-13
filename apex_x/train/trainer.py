"""Staged training flow for Apex-X v4 CPU baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from apex_x.config import ApexXConfig
from apex_x.data import (
    CocoDetectionDataset,
    SatelliteDataset,
    YOLOSegmentationDataset,
    build_robust_transforms,
    infer_dataset_type as infer_dataset_type_preflight,
    run_dataset_preflight,
    write_dataset_preflight_report,
    standard_collate_fn,
)
from apex_x.model import (
    ApexXModel,
    DetHead,
    DualPathFPN,
    PVModule,
    TeacherModel,
    TeacherModelV3,
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


from .checkpoint import CheckpointMetadata, cleanup_old_checkpoints, load_checkpoint, save_checkpoint
from .checkpoint import update_checkpoint_manifest
from .lr_scheduler import create_lr_scheduler
from .pcgrad import apply_pcgradpp, diagnostics_to_dict
from .train_losses import compute_teacher_training_loss
from .train_losses_v3 import compute_v3_training_losses
from .qat import QuantizationSummary, prepare_int8_ptq, prepare_int8_qat
from .trainer_utils import add_train_epoch_method
from .memory_manager import MemoryManager
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

LOGGER = get_logger(__name__)

StageMetricValue = float | int | bool | str | None


def _to_json_compatible_metric(value: StageMetricValue) -> float | int | bool | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    return None


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "n/a"
    return str(value)


def _build_train_report_markdown(report_payload: Mapping[str, Any]) -> str:
    dataset = cast(Mapping[str, Any], report_payload.get("dataset", {}))
    final = cast(Mapping[str, Any], report_payload.get("final", {}))
    loss_diag = cast(Mapping[str, Any], final.get("loss_diagnostics", {}))
    history = cast(list[Mapping[str, Any]], report_payload.get("history", []))
    stages = cast(list[Mapping[str, Any]], report_payload.get("stages", []))

    lines: list[str] = [
        "# Apex-X Train Report",
        "",
        "## Summary",
        f"- epochs: {report_payload.get('epochs', 'n/a')}",
        f"- best_metric_name: {report_payload.get('best_metric_name', 'n/a')}",
        f"- best_metric_value: {_format_metric(report_payload.get('best_metric_value'))}",
        "",
        "## Dataset",
        f"- mode: {dataset.get('mode', 'unknown')}",
        f"- backend: {dataset.get('backend', 'unknown')}",
        (
            "- allow_synthetic_fallback: "
            f"{_format_metric(dataset.get('allow_synthetic_fallback'))}"
        ),
        "",
        "## Final",
        f"- loss_proxy: {_format_metric(final.get('loss_proxy'))}",
        f"- final_mu: {_format_metric(final.get('final_mu'))}",
        "",
        "## Loss Diagnostics",
    ]
    if loss_diag:
        for key in sorted(loss_diag):
            lines.append(f"- {key}: {_format_metric(loss_diag[key])}")
    else:
        lines.append("- n/a")

    lines.extend(["", "## Stage Metrics"])
    if stages:
        lines.extend(
            [
                "| stage_id | name | metrics |",
                "|---|---|---|",
            ]
        )
        for stage in stages:
            stage_id = stage.get("stage_id", "n/a")
            stage_name = stage.get("name", "unknown")
            metrics = cast(Mapping[str, Any], stage.get("metrics", {}))
            metric_parts = [f"{key}={_format_metric(metrics[key])}" for key in sorted(metrics)]
            lines.append(f"| {stage_id} | {stage_name} | {'; '.join(metric_parts)} |")
    else:
        lines.append("- n/a")

    lines.extend(["", "## Epoch History"])
    if history:
        lines.extend(
            [
                "| epoch | loss_proxy | selected_metric_name | selected_metric_value | is_best |",
                "|---|---|---|---|---|",
            ]
        )
        for row in history:
            lines.append(
                "| "
                f"{row.get('epoch', 'n/a')} | "
                f"{_format_metric(row.get('loss_proxy'))} | "
                f"{row.get('selected_metric_name', 'n/a')} | "
                f"{_format_metric(row.get('selected_metric_value'))} | "
                f"{_format_metric(row.get('is_best'))} |"
            )
    else:
        lines.append("- n/a")

    return "\n".join(lines) + "\n"


def _identity_collate(batch: list[Any]) -> list[Any]:
    """Deprecated: using standard_collate_fn instead."""
    return batch


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

        # TF32 support for Ampere+ GPUs
        if self.config.train.tf32_enabled:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            LOGGER.info("TF32 enabled: Faster matmul on Ampere+ GPUs")

        # CuDNN benchmark
        if self.config.train.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            LOGGER.info("CuDNN benchmark enabled: Optimized kernel selection")

        # torch.compile (PyTorch 2.0+)
        if self.config.train.torch_compile:
            if hasattr(torch, "compile"):
                LOGGER.info("Wrapping teacher model with torch.compile...")
                self.teacher = torch.compile(self.teacher) # type: ignore
            else:
                LOGGER.warning("torch.compile requested but not available in this PyTorch version")
        
        # Checkpoint management
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path(self.config.train.output_dir) / "checkpoints"
        self.best_metric = float("-inf")
        self.best_metric_name = str(self.config.train.primary_metric)
        self.current_epoch = 0
        self.val_interval = max(1, int(self.config.train.val_interval))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
        self.memory_manager = MemoryManager()
        self.swa_model = AveragedModel(self.teacher) if self.config.train.swa_enabled else None
        if self.swa_model:
             LOGGER.info("SWA (Stochastic Weight Averaging) enabled.")

    def _build_teacher_model(self, *, num_classes: int) -> TeacherModel | TeacherModelV3:
        # Check for V3 World-Class profile
        if self.config.model.profile == "worldclass":
             LOGGER.info("Building TeacherModelV3 (World-Class Profile)...")
             return TeacherModelV3(
                 num_classes=num_classes,
                 backbone_model=self.config.model.backbone_model,
                 lora_rank=self.config.model.lora_rank,
                 fpn_channels=self.config.model.fpn_channels,
                 # num_cascade_stages=3 # Default in V3
             )

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
            ff_channels = p3_ch
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
            ff_channels = 16

        fpn = DualPathFPN(
            pv_p3_channels=p3_ch,
            pv_p4_channels=p4_ch,
            pv_p5_channels=p5_ch,
            ff_channels=ff_channels,
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
        selection_metric_name: str | None = None,
        selection_metric_value: float | None = None,
    ) -> None:
        """Save a training checkpoint.
        
        Args:
            epoch: Current epoch number
            step: Current training step
            optimizer: Optimizer to save
            scheduler: Optional LR scheduler
            metrics: Training/validation metrics
            is_best: Whether this is the best checkpoint
            selection_metric_name: Name of checkpoint-selection metric
            selection_metric_value: Value of checkpoint-selection metric
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

        metric_name = selection_metric_name or str(self.best_metric_name)
        metric_value = selection_metric_value
        if metric_value is None and metrics is not None:
            maybe_value = metrics.get("selection_metric_value")
            if isinstance(maybe_value, (int, float)) and np.isfinite(float(maybe_value)):
                metric_value = float(maybe_value)
        update_checkpoint_manifest(
            self.checkpoint_dir,
            checkpoint_path=ckpt_path,
            epoch=epoch,
            step=step,
            metric_name=str(metric_name),
            metric_value=metric_value,
            is_best=bool(is_best),
            has_ema=bool(ema_state),
            model_family=type(self.teacher).__name__,
            config=self.config.to_dict(),
        )
    
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
                    evaluator = COCOEvaluator(val_dataloader.dataset.coco, iou_types=["bbox", "segm"])
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
                
                if not isinstance(batch_data, dict):
                    continue
                    
                images = batch_data.get("images")
                image_ids = batch_data.get("image_ids", [])
                
                if images is None:
                    continue
                
                # Move to device
                images = images.to(device)
                
                # Model forward pass
                output = self.teacher(images, use_ema=True)
                
                # Compute real validation loss
                if isinstance(self.teacher, TeacherModelV3):
                     val_loss_tens, _ = compute_v3_training_losses(
                         outputs=output,
                         targets=batch_data,
                         model=self.teacher,
                         config=self.config,
                     )
                else:
                    val_loss_tens, _ = compute_teacher_training_loss(
                        output=output,
                        samples=batch_data,
                        det_weight=1.0,
                        boundary_weight=1.0,
                        epoch=999,
                        total_epochs=999,
                        box_loss_type=self.config.train.box_loss_type,
                        boundary_warmup_epochs=0,
                        boundary_max_scale=1.0,
                        box_warmup_epochs=0,
                        box_scale_start=1.0,
                        box_scale_end=1.0,
                    )
                total_loss += float(val_loss_tens.item())
                
                # Post-process detections for mAP
                if compute_map and evaluator is not None:
                    results = {}
                    
                    if isinstance(self.teacher, TeacherModelV3):
                        # V3 Output Processing
                        boxes_all = output["boxes"] # [N_total, 4]
                        scores_all = output["scores"] # [N_total, C]
                        masks_all = output.get("masks") # [N_total, 1, 28, 28] or None
                        batch_indices = output["batch_indices"] # [N_total]
                        
                        for i, img_id in enumerate(image_ids):
                            # Ensure image_id is int for COCO
                            if isinstance(img_id, torch.Tensor):
                                img_id_val = int(img_id.item())
                            else:
                                img_id_val = int(img_id)
                                
                            mask_i = (batch_indices == i)
                            if not mask_i.any():
                                continue
                            
                            boxes_i = boxes_all[mask_i]
                            scores_i = scores_all[mask_i]
                            masks_i = masks_all[mask_i] if masks_all is not None else None
                            
                            # Filter low confidence
                            max_scores, labels = scores_i.max(dim=1)
                            keep = max_scores > conf_threshold
                            
                            boxes_i = boxes_i[keep]
                            labels = labels[keep] + 1 # 1-based class capability (bg is 0?) - Check COCO mapping
                            scores_i = max_scores[keep]
                            masks_i = masks_i[keep] if masks_i is not None else None
                            
                            # NMS (if not already done sufficiently by model)
                            # TeacherV3 likely returns many boxes.
                            from torchvision.ops import nms
                            keep_nms = nms(boxes_i, scores_i, nms_threshold)
                            
                            results[img_id_val] = {
                                "boxes": boxes_i[keep_nms],
                                "scores": scores_i[keep_nms],
                                "labels": labels[keep_nms],
                                "masks": masks_i[keep_nms] if masks_i is not None else None,
                            }

                    else:
                        # Legacy Output Processing
                        from apex_x.model import post_process_detections
                        detections = post_process_detections(
                            cls_logits_by_level=output.logits_by_level,
                            box_reg_by_level=output.boxes_by_level,
                            quality_by_level=output.quality_by_level,
                            conf_threshold=conf_threshold,
                            nms_threshold=nms_threshold,
                            box_format="distance",
                        )
                        
                        for det_idx, detection in enumerate(detections):
                            if det_idx < len(image_ids):
                                img_id = image_ids[det_idx]
                                
                                pred_masks = None
                                if output.masks is not None and output.masks.ndim == 4 and det_idx < output.masks.shape[0]:
                                    pred_masks = output.masks[det_idx]
                                
                                coco_dets = []
                                boxes = detection['boxes'].cpu().numpy()
                                scores = detection['scores'].cpu().numpy()
                                classes = detection['classes'].cpu().numpy()
                                image_h, image_w = images.shape[2:]
                                
                                for inst_idx, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                                    x1, y1, x2, y2 = box
                                    payload = {
                                        'image_id': int(img_id),
                                        'category_id': int(cls_id),
                                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                        'score': float(score),
                                    }

                                    mask_np = None
                                    if pred_masks is not None and inst_idx < pred_masks.shape[0]:
                                        candidate = pred_masks[inst_idx].detach().cpu().numpy()
                                        mask_np = (candidate > 0.5).astype(np.uint8)
                                    else:
                                        x1i, y1i = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
                                        x2i, y2i = int(min(image_w, np.ceil(x2))), int(min(image_h, np.ceil(y2)))
                                        mask_np = np.zeros((image_h, image_w), dtype=np.uint8)
                                        if x2i > x1i and y2i > y1i:
                                            mask_np[y1i:y2i, x1i:x2i] = 1

                                    if mask_np is not None:
                                        try:
                                            from pycocotools import mask as mask_utils
                                            rle = mask_utils.encode(np.asfortranarray(mask_np))
                                            rle["counts"] = rle["counts"].decode("utf-8")
                                            payload["segmentation"] = rle
                                        except ImportError:
                                            pass

                                    coco_dets.append(payload)
                                
                                evaluator.update(coco_dets)
                                all_image_ids.append(img_id)

                    if results:
                        evaluator.update(results)
                
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

    def _infer_dataset_type(self, dataset_path: str | Path | None) -> str:
        return infer_dataset_type_preflight(self.config, dataset_path)

    def _handle_synthetic_fallback(
        self,
        *,
        reason: str,
        dataset_type: str,
        steps_per_epoch: int,
    ) -> tuple[torch.utils.data.DataLoader | None, int, str]:
        if not self.config.train.allow_synthetic_fallback:
            # STRICT FAIL-FAST
            LOGGER.error(f"CRITICAL: Data bootstrap failed and fallback is DISABLED. Reason: {reason}")
            raise RuntimeError(
                f"Dataset bootstrap failed for dataset_type={dataset_type}. "
                f"Reason: {reason}. "
                "Synthetic fallback is disabled (train.allow_synthetic_fallback=false). "
                "Please fix your dataset path or configuration."
            )
        
        LOGGER.warning(
            "synthetic_dataset_fallback_enabled",
            dataset_type=dataset_type,
            reason=reason,
        )
        return None, steps_per_epoch, "synthetic"

    def _build_training_dataloader(
        self,
        *,
        train_h: int,
        train_w: int,
        dataset_path: str | Path | None,
    ) -> tuple[torch.utils.data.DataLoader | None, int, str]:
        dataset_type = self._infer_dataset_type(dataset_path)
        steps_per_epoch = 1000
        if dataset_type == "synthetic":
            return self._handle_synthetic_fallback(
                reason="dataset_type resolved to synthetic",
                dataset_type=dataset_type,
                steps_per_epoch=steps_per_epoch,
            )

        dataset: Any | None = None
        try:
            if dataset_type == "satellite":
                root_text = str(dataset_path) if dataset_path is not None else str(self.config.data.dataset_root)
                if not root_text:
                    return self._handle_synthetic_fallback(
                        reason="empty dataset root for satellite dataset",
                        dataset_type=dataset_type,
                        steps_per_epoch=steps_per_epoch,
                    )
                dataset = SatelliteDataset(
                    root_dir=root_text,
                    tile_size=max(train_h, 256),
                    stride=max(train_h // 2, 128),
                )
            elif dataset_type == "coco":
                if not self.config.data.coco_train_images or not self.config.data.coco_train_annotations:
                    return self._handle_synthetic_fallback(
                        reason="COCO train image/annotation paths are missing",
                        dataset_type=dataset_type,
                        steps_per_epoch=steps_per_epoch,
                    )
                # 1. Initialize Raw Dataset (No Transforms)
                raw_dataset = CocoDetectionDataset(
                    root=self.config.data.coco_train_images,
                    ann_file=self.config.data.coco_train_annotations,
                    transforms=None, # Raw samples for CopyPaste/Mosaic source
                )
                
                # 2. Build Advanced Transforms relying on the dataset
                from apex_x.data.augmentations import CopyPasteAugmentation, MosaicAugmentation, MixUpAugmentation
                from apex_x.data.transforms import TransformPipeline, MosaicV2
                from apex_x.data.dataset_wrappers import AugmentedDataset
                
                # Base robust transforms (Resize, Color, Flip)
                base_transform = build_robust_transforms(
                    height=train_h,
                    width=train_w,
                    blur_prob=0.3,
                    noise_prob=0.3,
                    distort_prob=0.5,
                )
                
                # Advanced Augmentations
                # We define a custom pipeline list
                aug_list = []
                
                # Mosaic (4-image) - Puts 4 images together, usually applied first
                if self.config.data.mosaic_prob > 0:
                    # MosaicV2 handles the 4-image logic internally if passed a list, 
                    # OR we can use the MosaicAugmentation from augmentations.py which wraps the sampling logic.
                    # augmentations.py MosaicAugmentation takes 'dataset' and does the sampling.
                    aug_list.append(MosaicAugmentation(
                        dataset=raw_dataset,
                        output_size=max(train_h, train_w),
                        mosaic_prob=self.config.data.mosaic_prob
                    ))
                
                # CopyPaste - Pastes objects from other images
                if self.config.data.copypaste_prob > 0:
                    aug_list.append(CopyPasteAugmentation(
                        dataset=raw_dataset,
                        paste_prob=self.config.data.copypaste_prob,
                        max_paste=5,
                        blend_alpha=1.0
                    ))
                
                # MixUp - Blends 2 images
                if self.config.data.mixup_prob > 0:
                    aug_list.append(MixUpAugmentation(
                        dataset=raw_dataset,
                        alpha=1.0,
                        mixup_prob=self.config.data.mixup_prob
                    ))
                    
                # Add base transforms (Resize, Geometry, Color)
                # Note: Mosaic output is already resized to output_size, so base_transform resize is redundant but safe.
                # If Mosaic didn't run, we need base_transform to resize.
                
                # Wrapping base_transform (AlbumentationsAdapter) to be call-compatible if needed,
                # but TransformPipeline handles it.
                
                class CompositePipeline:
                    def __init__(self, augs, base):
                        self.augs = augs
                        self.base = base
                        
                    def __call__(self, sample, rng=None):
                        # Apply advanced augs (Mosaic/CopyPaste)
                        for aug in self.augs:
                            sample = aug(sample) # These don't take rng in signature in augmentations.py, they use random module
                            
                        # Apply base (Albumentations) - requires rng
                        sample = self.base(sample, rng=rng)
                        return sample

                full_transform = CompositePipeline(aug_list, base_transform)
                
                # 3. Wrap with AugmentedDataset
                dataset = AugmentedDataset(
                    dataset=raw_dataset,
                    transform=full_transform
                )
                
                LOGGER.info(f"Initialized COCO Dataset with SOTA Augmentations (Mosaic={self.config.data.mosaic_prob}, CopyPaste={self.config.data.copypaste_prob})")

            elif dataset_type == "yolo":

                root_text = str(dataset_path) if dataset_path is not None else str(self.config.data.dataset_root)
                if not root_text:
                    return self._handle_synthetic_fallback(
                        reason="empty dataset root for YOLO dataset",
                        dataset_type=dataset_type,
                        steps_per_epoch=steps_per_epoch,
                    )
                root = Path(root_text).expanduser()
                if not (root / "data.yaml").exists():
                    return self._handle_synthetic_fallback(
                        reason=f"missing data.yaml at {root}",
                        dataset_type=dataset_type,
                        steps_per_epoch=steps_per_epoch,
                    )
                dataset = YOLOSegmentationDataset(
                    root=root,
                    split="train",
                    image_size=max(train_h, train_w),
                )
            else:
                raise ValueError(f"unsupported dataset type: {dataset_type}")
        except Exception as exc:
            return self._handle_synthetic_fallback(
                reason=f"failed to build train dataloader: {exc}",
                dataset_type=dataset_type,
                steps_per_epoch=steps_per_epoch,
            )

        if dataset is None or len(dataset) == 0:
            return self._handle_synthetic_fallback(
                reason=f"empty training dataset for dataset_type={dataset_type}",
                dataset_type=dataset_type,
                steps_per_epoch=steps_per_epoch,
            )

        batch_size = 1
        if self.config.train.auto_batch_size:
            try:
                batch_size = self.memory_manager.optimize_batch_size(
                    self.teacher,
                    (3, train_h, train_w),
                    max_batch_size=32,
                    start_batch_size=2,
                    mode="train",
                )
                LOGGER.info(f"Using auto-tuned batch size: {batch_size}")
            except Exception as exc:
                LOGGER.warning(f"Auto batch-size tuning failed, fallback to batch_size=1: {exc}")
                batch_size = 1

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.train.dataloader_num_workers,
            pin_memory=self.config.train.dataloader_pin_memory,
            collate_fn=standard_collate_fn,
        )
        steps_per_epoch = len(dataloader)
        return dataloader, steps_per_epoch, dataset_type

    def _build_validation_dataloader(
        self,
        *,
        train_h: int,
        train_w: int,
        dataset_path: str | Path | None,
    ) -> torch.utils.data.DataLoader | None:
        dataset_type = self._infer_dataset_type(dataset_path)
        dataset: Any | None = None
        try:
            if dataset_type == "satellite":
                root_text = str(dataset_path) if dataset_path is not None else str(self.config.data.dataset_root)
                if root_text:
                    dataset = SatelliteDataset(
                        root_dir=root_text,
                        tile_size=max(train_h, 256),
                        stride=max(train_h // 2, 128),
                    )
            elif dataset_type == "coco":
                if self.config.data.coco_val_images and self.config.data.coco_val_annotations:
                    dataset = CocoDetectionDataset(
                        root=self.config.data.coco_val_images,
                        ann_file=self.config.data.coco_val_annotations,
                        transforms=None,
                    )
            elif dataset_type == "yolo":
                root_text = str(dataset_path) if dataset_path is not None else str(self.config.data.dataset_root)
                if root_text:
                    root = Path(root_text).expanduser()
                    if (root / "data.yaml").exists():
                        dataset = YOLOSegmentationDataset(
                            root=root,
                            split="val",
                            image_size=max(train_h, train_w),
                        )
        except Exception as exc:
            LOGGER.warning(f"Failed to build val dataloader for dataset_type={dataset_type}: {exc}")
            return None

        if dataset is None or len(dataset) == 0:
            return None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.train.dataloader_num_workers,
            pin_memory=self.config.train.dataloader_pin_memory,
            collate_fn=standard_collate_fn,
        )

    @staticmethod
    def _metric_score(metric_name: str, value: float) -> float:
        lower = metric_name.lower()
        return -value if "loss" in lower else value

    def _select_checkpoint_metric(
        self,
        *,
        val_metrics: Mapping[str, float] | None,
        loss_proxy: float,
    ) -> tuple[float, str, float]:
        metrics = val_metrics or {}
        preferred = str(self.config.train.primary_metric).strip()
        ordered = [preferred, "mAP_segm", "mAP_bbox", "mAP", "val_loss", "loss_proxy"]
        seen: set[str] = set()
        for name in ordered:
            key = str(name).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            if key == "loss_proxy":
                value = float(loss_proxy)
            elif key in metrics:
                value = float(metrics[key])
            else:
                continue
            return self._metric_score(key, value), key, value
        value = float(loss_proxy)
        return self._metric_score("loss_proxy", value), "loss_proxy", value


    def _stage1_teacher_training(
        self,
        generator: torch.Generator,
        *,
        steps: int,
        epoch: int,
        total_epochs: int,
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
        
        swa_scheduler = None
        if self.swa_model:
            swa_scheduler = SWALR(optimizer, swa_lr=self.config.train.swa_lr)
            LOGGER.info(f"SWA scheduler initialized (start epoch: {self.config.train.swa_start_epoch})")
        train_h = min(self.config.model.input_height, 512)
        train_w = min(self.config.model.input_width, 512)
        
        dataloader, steps_per_epoch, dataset_type = self._build_training_dataloader(
            train_h=train_h,
            train_w=train_w,
            dataset_path=dataset_path,
        )
        LOGGER.info(f"Stage1 dataset backend: {dataset_type}")
        dataset_mode = "synthetic" if dataloader is None else "real"
        LOGGER.info(f"Stage1 dataset mode: {dataset_mode}")
                
        transforms = build_robust_transforms(
            height=train_h, 
            width=train_w,
            blur_prob=0.3,
            noise_prob=0.3,
            distort_prob=0.3
        )
        rng = np.random.RandomState(42)

        loss_last = 0.0
        loss_component_sums: dict[str, float] = {}
        loss_component_updates = 0
        nan_or_inf_grad_steps = 0
        skipped_non_diff_steps = 0
        oom_skipped_steps = 0
        grad_norm_last = 0.0
        grad_norm_max = 0.0

        def _update_loss_component_sums(loss_components: Mapping[str, float]) -> None:
            nonlocal loss_component_updates
            for name, value in loss_components.items():
                value_f = float(value)
                if not np.isfinite(value_f):
                    continue
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value_f
            loss_component_updates += 1
        
        # If dataloader exists, we iterate it. Else random loop.
        iterator = iter(dataloader) if dataloader else None
        
        for step_idx in range(steps):
            samples = None
            if iterator:
                try:
                    samples = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    samples = next(iterator)

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

            image = image.to(next(self.teacher.parameters()).device)
            samples_for_loss = samples if iterator else None

            # OOM Recovery Wrapper
            with self.memory_manager.catch_oom() as oom_state:


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
                        # Compute training loss with real detection outputs
                        if isinstance(self.teacher, TeacherModelV3):
                             total_loss, loss_components = compute_v3_training_losses(
                                 outputs=out,
                                 targets=samples_for_loss,
                                 model=self.teacher,
                                 config=self.config,
                             )
                        else:
                             total_loss, loss_components = compute_teacher_training_loss(
                                 output=out,
                                 samples=samples_for_loss,
                                 det_weight=1.0,
                                 boundary_weight=0.05,
                                 epoch=epoch,
                                 total_epochs=total_epochs,
                                 box_loss_type=self.config.train.box_loss_type,
                                 boundary_warmup_epochs=self.config.train.loss_boundary_warmup_epochs,
                                 boundary_max_scale=self.config.train.loss_boundary_max_scale,
                                 box_warmup_epochs=self.config.train.loss_box_warmup_epochs,
                                 box_scale_start=self.config.train.loss_box_scale_start,
                                 box_scale_end=self.config.train.loss_box_scale_end,
                                 max_det_component=self.config.train.loss_det_component_clip,
                                 max_boundary_component=self.config.train.loss_boundary_component_clip,
                                 max_seg_component=self.config.train.loss_seg_component_clip,
                             )
                        _update_loss_component_sums(loss_components)
                        
                        # Scale loss by accumulation steps
                        total_loss = total_loss / self.gradient_accumulation_steps
                    if not total_loss.requires_grad:
                        skipped_non_diff_steps += 1
                        LOGGER.warning(
                            f"Non-differentiable loss at step {step_idx}; skipping batch."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # Backward with scaler (accumulate gradients)
                    try:
                        self.scaler.scale(total_loss).backward()
                    except RuntimeError as exc:
                        if "inplace operation" in str(exc).lower():
                            LOGGER.warning(
                                f"Autograd inplace-modification error at step {step_idx}; skipping batch."
                            )
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        raise
                    
                    # Only step optimizer after accumulating enough gradients
                    if should_step:
                        self.scaler.unscale_(optimizer)
                        
                        # Gradient clip for stability
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.teacher.parameters(),
                            max_norm=10.0,
                        )
                        grad_norm_last = float(grad_norm.item())
                        grad_norm_max = max(grad_norm_max, grad_norm_last)
                        
                        # Verify gradients are finite before stepping
                        grads_finite = True
                        for param in self.teacher.parameters():
                            if param.grad is not None and not torch.isfinite(param.grad).all():
                                grads_finite = False
                                break
                        
                        if grads_finite:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            nan_or_inf_grad_steps += 1
                            LOGGER.warning(f"NaN/Inf detected in gradients at step {step_idx}. Skipping step.")
                            optimizer.zero_grad(set_to_none=True)
                else:
                    # Zero gradients only at start of cycle
                    if not is_accumulation_step:
                        optimizer.zero_grad(set_to_none=True)
                        
                    with heavy_ops_autocast_context(self.precision_policy):
                        out = self.teacher(image, use_ema=False)
                        # Compute training loss with real detection outputs
                        if isinstance(self.teacher, TeacherModelV3):
                             total_loss, loss_components = compute_v3_training_losses(
                                 outputs=out,
                                 samples=samples_for_loss,
                                 model=self.teacher,
                                 config=self.config,
                             )
                        else:
                             total_loss, loss_components = compute_teacher_training_loss(
                                 output=out,
                                 samples=samples_for_loss,
                                 det_weight=1.0,
                                 boundary_weight=0.05,
                                 epoch=epoch,
                                 total_epochs=total_epochs,
                                 box_loss_type=self.config.train.box_loss_type,
                                 boundary_warmup_epochs=self.config.train.loss_boundary_warmup_epochs,
                                 boundary_max_scale=self.config.train.loss_boundary_max_scale,
                                 box_warmup_epochs=self.config.train.loss_box_warmup_epochs,
                                 box_scale_start=self.config.train.loss_box_scale_start,
                                 box_scale_end=self.config.train.loss_box_scale_end,
                                 max_det_component=self.config.train.loss_det_component_clip,
                                 max_boundary_component=self.config.train.loss_boundary_component_clip,
                                 max_seg_component=self.config.train.loss_seg_component_clip,
                             )
                        _update_loss_component_sums(loss_components)
                        
                        # Scale loss by accumulation steps
                        total_loss = total_loss / self.gradient_accumulation_steps
                    if not total_loss.requires_grad:
                        skipped_non_diff_steps += 1
                        LOGGER.warning(
                            f"Non-differentiable loss at step {step_idx}; skipping batch."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # Backward (accumulate gradients)
                    try:
                        total_loss.backward()
                    except RuntimeError as exc:
                        if "inplace operation" in str(exc).lower():
                            LOGGER.warning(
                                f"Autograd inplace-modification error at step {step_idx}; skipping batch."
                            )
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        raise
                    
                    # Only step optimizer after accumulating enough gradients
                    if should_step:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.teacher.parameters(),
                            max_norm=10.0,
                        )
                        grad_norm_last = float(grad_norm.item())
                        grad_norm_max = max(grad_norm_max, grad_norm_last)
                        
                        # Verify gradients are finite before stepping
                        grads_finite = True
                        for param in self.teacher.parameters():
                            if param.grad is not None and not torch.isfinite(param.grad).all():
                                grads_finite = False
                                break
                                
                        if grads_finite:
                            optimizer.step()
                        else:
                            nan_or_inf_grad_steps += 1
                            LOGGER.warning(f"NaN/Inf detected in gradients at step {step_idx}. Skipping step.")
                            optimizer.zero_grad(set_to_none=True)
                
                # Update scheduler and EMA only when we actually step
                if should_step:
                    # SWA Logic
                    use_swa = False
                    if self.swa_model and swa_scheduler:
                        current_epoch = step_idx // steps_per_epoch
                        if current_epoch >= self.config.train.swa_start_epoch:
                            use_swa = True
                            self.swa_model.update_parameters(self.teacher)
                            swa_scheduler.step()
                    
                    if not use_swa:
                        scheduler.step()
                        
                    self.teacher.update_ema()
                    
                loss_last = float(total_loss.detach().item() * self.gradient_accumulation_steps)  # Unscale for logging
                
                # Log current LR (only do this periodically to avoid spam)
                if step_idx % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    LOGGER.debug(f"Step {step_idx}/{steps}: loss={loss_last:.4f}, lr={current_lr:.6f}")

            if oom_state.triggered:
                oom_skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue


        # Finalize SWA
        if self.swa_model and swa_scheduler and dataloader:
            LOGGER.info("Updating SWA batch norm statistics...")
            update_bn(dataloader, self.swa_model, device=next(self.teacher.parameters()).device)
            self.teacher.load_state_dict(self.swa_model.module.state_dict())
            LOGGER.info("SWA weights loaded into teacher model.")

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
            "dataset_backend": dataset_type,
            "dataset_mode": dataset_mode,
            "loss_component_updates": int(loss_component_updates),
            "nan_or_inf_grad_steps": int(nan_or_inf_grad_steps),
            "skipped_non_diff_steps": int(skipped_non_diff_steps),
            "oom_skipped_steps": int(oom_skipped_steps),
            "grad_norm_last": float(grad_norm_last),
            "grad_norm_max": float(grad_norm_max),
        }
        if loss_component_updates > 0:
            for name, value in sorted(loss_component_sums.items()):
                metrics[f"loss_{name}_mean"] = float(value / float(loss_component_updates))
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

    def _stage_finetune_lora(
        self,
        generator: torch.Generator,
        *,
        steps: int,
        epoch: int,
        total_epochs: int,
        dataset_path: str | Path | None = None,
        lr: float = 1e-4,
    ) -> tuple[StageResult, float]:
        """Phase 2: High-precision LoRA fine-tuning stage.
        
        Focuses on adapting the frozen DINOv2 backbone (via LoRA) to 
        domain-specific satellite imagery with a lower learning rate.
        """
        LOGGER.info(f"Starting LoRA Fine-tuning with learning_rate={lr}...")
        
        # Filter for parameters that actually require gradients (LoRA + Heads)
        trainable_params = [p for p in self.teacher.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        # Use existing training utilities if possible, or a simplified loop
        # For this implementation, we replicate the high-perf loop from stage 1
        # but specialized for the finetuning use case.
        
        scheduler = create_lr_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine_warmup",
            total_steps=steps,
            warmup_ratio=0.1,
            min_lr=lr * 0.01,
        )

        train_h = min(self.config.model.input_height, 512)
        train_w = min(self.config.model.input_width, 512)
        dataloader, _, dataset_type = self._build_training_dataloader(
            train_h=train_h,
            train_w=train_w,
            dataset_path=dataset_path,
        )
        LOGGER.info(f"LoRA finetune dataset backend: {dataset_type}")
        dataset_mode = "synthetic" if dataloader is None else "real"
        LOGGER.info(f"LoRA finetune dataset mode: {dataset_mode}")

        transforms = build_robust_transforms(
            height=train_h,
            width=train_w,
            blur_prob=0.1,  # Lower augmentation for fine-tuning
            noise_prob=0.1,
            distort_prob=0.1
        )
        rng = np.random.RandomState(42)
        loss_last = 0.0
        iterator = iter(dataloader) if dataloader else None

        for step_idx in range(steps):
            if iterator:
                try:
                    samples = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    samples = next(iterator)

                batch_images = []
                for s in samples:
                    aug = transforms(s, rng=rng)
                    img_t = torch.from_numpy(aug.image).permute(2, 0, 1).float() / 255.0
                    batch_images.append(img_t)
                image = torch.stack(batch_images, dim=0).to(next(self.teacher.parameters()).device)
            else:
                image = torch.rand((1, 3, train_h, train_w), generator=generator, dtype=torch.float32).to(next(self.teacher.parameters()).device)

            samples_for_loss = samples if iterator else None
            is_accumulation_step = (step_idx % self.gradient_accumulation_steps) != 0
            should_step = (step_idx + 1) % self.gradient_accumulation_steps == 0

            if self.use_amp and self.scaler is not None:
                if not is_accumulation_step:
                    optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    out = self.teacher(image, use_ema=False)
                    if isinstance(self.teacher, TeacherModelV3):
                        total_loss, _ = compute_v3_training_losses(
                            outputs=out,
                            targets=samples_for_loss,
                            model=self.teacher,
                            config=self.config,
                         )
                    else:
                        total_loss, _ = compute_teacher_training_loss(
                            output=out,
                            samples=samples_for_loss,
                            det_weight=1.0,
                            boundary_weight=0.1,  # Higher boundary focus in finetuning
                            epoch=epoch,
                            total_epochs=total_epochs,
                            box_loss_type=self.config.train.box_loss_type,
                            boundary_warmup_epochs=self.config.train.loss_boundary_warmup_epochs,
                            boundary_max_scale=self.config.train.loss_boundary_max_scale,
                            box_warmup_epochs=self.config.train.loss_box_warmup_epochs,
                            box_scale_start=self.config.train.loss_box_scale_start,
                            box_scale_end=self.config.train.loss_box_scale_end,
                            max_det_component=self.config.train.loss_det_component_clip,
                            max_boundary_component=self.config.train.loss_boundary_component_clip,
                            max_seg_component=self.config.train.loss_seg_component_clip,
                        )
                    total_loss = total_loss / self.gradient_accumulation_steps
                if not total_loss.requires_grad:
                    LOGGER.warning(
                        f"Non-differentiable finetune loss at step {step_idx}; skipping batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.scale(total_loss).backward()
                if should_step:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                    
                    grads_finite = True
                    for param in trainable_params:
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grads_finite = False
                            break
                    if grads_finite:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        LOGGER.warning(f"NaN in gradients at finetune step {step_idx}. Skipping.")
                        optimizer.zero_grad(set_to_none=True)
            else:
                if not is_accumulation_step:
                    optimizer.zero_grad(set_to_none=True)
                with heavy_ops_autocast_context(self.precision_policy):
                    out = self.teacher(image, use_ema=False)
                    if isinstance(self.teacher, TeacherModelV3):
                        total_loss, _ = compute_v3_training_losses(
                            outputs=out,
                            targets=samples_for_loss,
                            model=self.teacher,
                            config=self.config,
                         )
                    else:
                        total_loss, _ = compute_teacher_training_loss(
                            output=out,
                            samples=samples_for_loss,
                            det_weight=1.0,
                            boundary_weight=0.1,
                            epoch=epoch,
                            total_epochs=total_epochs,
                            box_loss_type=self.config.train.box_loss_type,
                            boundary_warmup_epochs=self.config.train.loss_boundary_warmup_epochs,
                            boundary_max_scale=self.config.train.loss_boundary_max_scale,
                            box_warmup_epochs=self.config.train.loss_box_warmup_epochs,
                            box_scale_start=self.config.train.loss_box_scale_start,
                            box_scale_end=self.config.train.loss_box_scale_end,
                            max_det_component=self.config.train.loss_det_component_clip,
                            max_boundary_component=self.config.train.loss_boundary_component_clip,
                            max_seg_component=self.config.train.loss_seg_component_clip,
                        )
                    total_loss = total_loss / self.gradient_accumulation_steps
                if not total_loss.requires_grad:
                    LOGGER.warning(
                        f"Non-differentiable finetune loss at step {step_idx}; skipping batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                total_loss.backward()
                if should_step:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                    grads_finite = True
                    for param in trainable_params:
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grads_finite = False
                            break
                    if grads_finite:
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)

            if should_step:
                scheduler.step()
                self.teacher.update_ema()
            loss_last = float(total_loss.detach().item() * self.gradient_accumulation_steps)

        final_lr = scheduler.get_last_lr()[0]
        metrics = {
            "steps": int(steps),
            "finetune_loss_last": loss_last,
            "final_lr": float(final_lr),
            "num_lora_params": len(trainable_params),
            "dataset_backend": dataset_type,
            "dataset_mode": dataset_mode,
        }
        return StageResult(stage_id=5, name="lora_finetune", metrics=metrics), loss_last

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

        num_epochs = max(1, int(self.config.train.epochs))
        save_interval = max(1, int(self.config.train.save_interval))
        seed_all(seed=seed, deterministic=self.config.runtime.deterministic)
        rng = np.random.RandomState(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self._prepare_quantization(generator)

        output_dir = Path(self.config.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_preflight = run_dataset_preflight(self.config, dataset_path=dataset_path)
        dataset_preflight_path = write_dataset_preflight_report(
            dataset_preflight,
            path=output_dir / "dataset_preflight.json",
        )
        if not dataset_preflight.passed:
            raise RuntimeError(
                "Dataset preflight failed. "
                f"See report: {dataset_preflight_path}"
            )

        train_h = min(self.config.model.input_height, 512)
        train_w = min(self.config.model.input_width, 512)
        val_loader = self._build_validation_dataloader(
            train_h=train_h,
            train_w=train_w,
            dataset_path=dataset_path,
        )
        if val_loader is None:
            LOGGER.info("Validation dataloader not available; checkpoint selection will fallback to loss proxy")

        trainable_params = [p for p in self.teacher.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(self.teacher.parameters())
        checkpoint_optimizer = torch.optim.SGD(trainable_params, lr=0.0, momentum=0.0)

        history: list[dict[str, float | int | str]] = []
        last_result: StagedTrainResult | None = None
        final_val_metrics: dict[str, float] = {}

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            stage0 = self._stage0_baseline_warmup(rng, steps=steps_per_stage)
            stage1, stage1_loss = self._stage1_teacher_training(
                generator,
                steps=steps_per_stage,
                epoch=epoch,
                total_epochs=num_epochs,
                dataset_path=dataset_path,
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
            stage_results: list[StageResult] = [stage0, stage1, stage2, stage3, stage4]
            stage5_loss = 0.0
            if self.config.train.enable_lora_finetune:
                stage5, stage5_loss = self._stage_finetune_lora(
                    generator,
                    steps=steps_per_stage,
                    epoch=epoch,
                    total_epochs=num_epochs,
                    dataset_path=dataset_path,
                )
                stage_results.append(stage5)

            pcgrad_snapshot = self._pcgrad_monitoring_snapshot()
            loss_proxy = float(stage1_loss + stage2_loss + abs(stage3_loss) + stage5_loss)
            val_metrics: dict[str, float] = {}
            if val_loader is not None and (
                ((epoch + 1) % self.val_interval == 0) or (epoch == num_epochs - 1)
            ):
                val_metrics = self.validate(
                    val_dataloader=val_loader,
                    device=str(next(self.teacher.parameters()).device),
                    compute_map=True,
                    max_batches=64,
                )
                final_val_metrics = dict(val_metrics)

            metric_score, metric_name, metric_value = self._select_checkpoint_metric(
                val_metrics=val_metrics,
                loss_proxy=loss_proxy,
            )
            if np.isfinite(float(self.best_metric)):
                current_best_score = self._metric_score(self.best_metric_name, float(self.best_metric))
            else:
                current_best_score = float("-inf")
            is_best = metric_score > current_best_score
            if is_best:
                self.best_metric = float(metric_value)
                self.best_metric_name = str(metric_name)

            train_summary = {
                "epoch": int(epoch),
                "epochs": int(num_epochs),
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
                "selection_metric": {
                    "name": str(metric_name),
                    "value": float(metric_value),
                    "score": float(metric_score),
                },
            }
            if val_metrics:
                train_summary["val_metrics"] = dict(val_metrics)

            result = StagedTrainResult(
                stage_results=tuple(stage_results),
                routing_diagnostics=routing_diag,
                train_summary=train_summary,
                loss_proxy=loss_proxy,
                final_mu=float(self.dual_controller.mu),
            )
            last_result = result
            history.append(
                {
                    "epoch": int(epoch),
                    "loss_proxy": float(loss_proxy),
                    "selected_metric_name": str(metric_name),
                    "selected_metric_value": float(metric_value),
                    "is_best": bool(is_best),
                }
            )

            should_save = ((epoch + 1) % save_interval == 0) or (epoch == num_epochs - 1)
            if should_save:
                metrics_payload: dict[str, float] = {
                    "loss_proxy": float(loss_proxy),
                    "final_mu": float(result.final_mu),
                    "selection_metric_value": float(metric_value),
                }
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(float(value)):
                        metrics_payload[str(key)] = float(value)
                self.save_training_checkpoint(
                    epoch=epoch,
                    step=0,
                    optimizer=checkpoint_optimizer,
                    metrics=metrics_payload,
                    is_best=is_best,
                    selection_metric_name=str(metric_name),
                    selection_metric_value=float(metric_value),
                )

            log_event(
                LOGGER,
                "staged_training_epoch_complete",
                fields={
                    "epoch": int(epoch),
                    "epochs": int(num_epochs),
                    "stages": len(result.stage_results),
                    "loss_proxy": round(result.loss_proxy, 6),
                    "final_mu": round(result.final_mu, 6),
                    "best_metric_name": self.best_metric_name,
                    "best_metric_value": float(self.best_metric),
                },
            )

        if last_result is None:
            raise RuntimeError("training completed without producing a result")

        log_event(
            LOGGER,
            "staged_training_complete",
            fields={
                "epochs": int(num_epochs),
                "stages": len(last_result.stage_results),
                "loss_proxy": round(last_result.loss_proxy, 6),
                "final_mu": round(last_result.final_mu, 6),
                "best_metric_name": self.best_metric_name,
                "best_metric": float(self.best_metric),
            },
        )

        legacy_checkpoint_path = output_dir / "teacher_checkpoint.pt"
        torch.save(self.teacher.state_dict(), legacy_checkpoint_path)

        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        stage1_metrics = next(
            (stage.metrics for stage in last_result.stage_results if stage.name == "teacher_full_compute"),
            {},
        )
        report_payload = {
            "epochs": int(num_epochs),
            "best_metric_name": self.best_metric_name,
            "best_metric_value": float(self.best_metric),
            "dataset": {
                "mode": str(stage1_metrics.get("dataset_mode", "unknown")),
                "backend": str(stage1_metrics.get("dataset_backend", "unknown")),
                "allow_synthetic_fallback": bool(self.config.train.allow_synthetic_fallback),
                "preflight_report": str(dataset_preflight_path),
                "preflight_passed": bool(dataset_preflight.passed),
                "preflight_errors": list(dataset_preflight.errors),
                "preflight_warnings": list(dataset_preflight.warnings),
            },
            "history": history,
            "final": {
                "loss_proxy": float(last_result.loss_proxy),
                "final_mu": float(last_result.final_mu),
                "val_metrics": final_val_metrics,
                "loss_diagnostics": {
                    str(k): float(v)
                    for k, v in stage1_metrics.items()
                    if isinstance(v, (int, float))
                    and (
                        str(k).startswith("loss_")
                        or str(k)
                        in {
                            "nan_or_inf_grad_steps",
                            "skipped_non_diff_steps",
                            "oom_skipped_steps",
                            "grad_norm_last",
                            "grad_norm_max",
                            "teacher_loss_last",
                        }
                    )
                },
            },
            "stages": [
                {
                    "stage_id": int(stage.stage_id),
                    "name": str(stage.name),
                    "metrics": {
                        str(key): _to_json_compatible_metric(value)
                        for key, value in stage.metrics.items()
                    },
                }
                for stage in last_result.stage_results
            ],
        }
        report_path = output_dir / "train_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        report_md_path = output_dir / "train_report.md"
        report_md_path.write_text(
            _build_train_report_markdown(report_payload),
            encoding="utf-8",
        )

        log_event(
            LOGGER,
            "training_artifacts_saved",
            fields={
                "checkpoint_legacy": str(legacy_checkpoint_path),
                "checkpoint_best": str(self.checkpoint_dir / "best.pt"),
                "checkpoint_last": str(self.checkpoint_dir / "last.pt"),
                "config": str(config_path),
                "report": str(report_path),
                "report_markdown": str(report_md_path),
            },
        )

        return last_result


__all__ = [
    "StageResult",
    "StagedTrainResult",
    "ApexXTrainer",
]
