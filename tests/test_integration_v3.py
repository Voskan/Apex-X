"""Integration tests for TeacherModelV3 and v2.0 features.

Tests:
 - Model forward pass shape correctness
 - Cascade intermediate outputs
 - All v2.0 loss functions
 - Quality filter pipeline
 - End-to-end backward pass (gradient flow)
"""

import pytest
import torch
import numpy as np

from apex_x.model import TeacherModelV3
from apex_x.train.train_losses_v3 import compute_v3_training_losses
from apex_x.data.quality_filter import ImageQualityFilter
from apex_x.losses.seg_loss import boundary_iou_loss


# ======================================================================
# Model tests
# ======================================================================

class TestTeacherModelV3:
    """Forward-pass tests (DINOv2 may or may not be installed)."""

    def _make_model(self, num_classes: int = 3) -> TeacherModelV3:
        return TeacherModelV3(num_classes=num_classes)

    def test_forward_produces_expected_keys(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(1, 3, 518, 518)          # DINOv2 likes multiples of 14
        with torch.no_grad():
            out = model(x)

        for key in ("boxes", "scores", "predicted_quality", "fpn_features",
                     "all_boxes", "all_masks", "all_scores"):
            assert key in out, f"Missing key {key!r}"

    def test_box_and_quality_shapes(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = model(x)

        assert out["boxes"].ndim == 2 and out["boxes"].shape[1] == 4
        assert out["predicted_quality"].ndim == 1
        assert out["boxes"].shape[0] == out["predicted_quality"].shape[0]

    def test_cascade_stages_count(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = model(x)

        # initial + 3 stages = 4 box sets
        assert len(out["all_boxes"]) == 4
        # 3 score and mask sets
        assert len(out["all_scores"]) == 3
        assert len(out["all_masks"]) == 3


# ======================================================================
# Loss tests
# ======================================================================

class TestV3Losses:

    def test_boundary_iou_loss_basic(self):
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = boundary_iou_loss(logits, targets, boundary_width=3)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_compute_v3_losses_all_components(self):
        from types import SimpleNamespace

        model = TeacherModelV3(num_classes=3)
        model.eval()

        x = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            outputs = model(x)

        n_boxes = outputs["boxes"].shape[0]
        targets = {
            "labels": torch.randint(0, 3, (n_boxes,)),
            "boxes":  torch.rand(n_boxes, 4) * 518,
            "masks":  torch.randint(0, 2, outputs["masks"].shape).float()
                      if outputs["masks"] is not None
                      else torch.randint(0, 2, (n_boxes, 1, 28, 28)).float(),
        }
        config = SimpleNamespace(loss=SimpleNamespace(
            multi_scale_supervision=False,
            boundary_weight=0.5,
            quality_weight=1.0,
        ))

        total_loss, loss_dict = compute_v3_training_losses(
            outputs, targets, model, config,
        )
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert len(loss_dict) >= 1


# ======================================================================
# Data quality
# ======================================================================

class TestDataQualityPipeline:

    def test_high_entropy_passes(self):
        filt = ImageQualityFilter(
            min_entropy=4.0,
            enable_entropy=True,
            enable_sharpness=False,
            enable_cloud=False,
        )
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        ok, metrics = filt.filter_image(img)
        assert "entropy" in metrics
        assert ok, f"Random image should pass; entropy={metrics['entropy']:.2f}"

    def test_uniform_image_fails(self):
        filt = ImageQualityFilter(
            min_entropy=6.0,     # intentionally high threshold
            enable_entropy=True,
            enable_sharpness=False,
            enable_cloud=False,
        )
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        ok, _ = filt.filter_image(img)
        assert not ok, "Uniform image must be rejected"


# ======================================================================
# End-to-end backward
# ======================================================================

class TestEndToEnd:

    @pytest.mark.slow
    def test_gradient_flow(self):
        """One full forward+backward; verify at least one param has grad."""
        model = TeacherModelV3(num_classes=3)
        model.train()

        x = torch.randn(1, 3, 518, 518)
        outputs = model(x)

        n_boxes = outputs["boxes"].shape[0]
        targets = {
            "labels": torch.randint(0, 3, (n_boxes,)),
            "boxes":  torch.rand(n_boxes, 4) * 518,
            "masks":  torch.randint(0, 2, outputs["masks"].shape).float()
                      if outputs["masks"] is not None
                      else torch.randint(0, 2, (n_boxes, 1, 28, 28)).float(),
        }
        from types import SimpleNamespace
        cfg = SimpleNamespace(loss=SimpleNamespace(
            multi_scale_supervision=False,
            boundary_weight=0.5,
        ))

        total_loss, _ = compute_v3_training_losses(outputs, targets, model, cfg)
        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Model must have gradients after backward"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
