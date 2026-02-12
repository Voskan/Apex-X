from __future__ import annotations

from torch import Tensor, nn

try:
    import timm
except ImportError:
    timm = None


class TimmBackboneAdapter(nn.Module):
    """Adapts a TIMM model to provide P3/P4/P5 features for Apex-X."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        out_indices: tuple[int, int, int] | None = None,
        norm_groups: int = 1, # Kept for API compatibility, though timm handles norm internally
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for TimmBackboneAdapter")
        
        # Determine indices for P3 (stride 8), P4 (stride 16), P5 (stride 32)
        # Standard ResNet/EfficientNet features usually:
        # 0: stride 2
        # 1: stride 4
        # 2: stride 8 (P3)
        # 3: stride 16 (P4)
        # 4: stride 32 (P5)
        self.out_indices = out_indices if out_indices is not None else (2, 3, 4)
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            in_chans=in_channels,
        )
        
        # Get feature info to populate channel counts
        feature_info = self.backbone.feature_info
        if hasattr(feature_info, "channels"):
            self.feature_channels = list(feature_info.channels())
        else:
            self.feature_channels = [info["num_chs"] for info in feature_info]
        
        self.p3_channels = self.feature_channels[0]
        self.p4_channels = self.feature_channels[1]
        self.p5_channels = self.feature_channels[2]
        
        # Optional: Add FPN or lateral connections here if strict channel sizing is needed?
        # Apex-X architecture usually expects specific channel counts for heads.
        # However, the PVBackbone defines p3_channels etc. as outputs. 
        # The downstream heads likely project these. 
        # Let's check `ApexXModel` usages. It matches channel output to head input.
        # So we expose what we have, and the config should probably match or we need projection layers.
        # For an adapter, it's safer to just return raw features and let the head adaptation happen elsewhere
        # OR we add 1x1 convs here to match requested channels?
        # The default PVBackbone constructor takes output channel args.
        # So this adapter should probably ideally project to those if possible.
        # But `trainer.py` constructs heads based on `pv_module.p3_channels`.
        # So as long as we expose `self.p3_channels` correctly, the heads will be built to match US.
        # Good.

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.backbone(x)
        
        # features is a list of tensors corresponding to out_indices
        return {
            "P3": features[0],
            "P4": features[1],
            "P5": features[2],
        }

__all__ = ["TimmBackboneAdapter"]
