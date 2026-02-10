from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class DetHeadOutput:
    """Anchor-free DET outputs per pyramid level."""

    cls_logits: dict[str, Tensor]  # [B, C_cls, H, W]
    box_reg: dict[str, Tensor]  # [B, 4, H, W]
    quality: dict[str, Tensor]  # [B, 1, H, W]
    features: dict[str, Tensor]  # normalized P3..P7 features used by the head


class DetHead(nn.Module):
    """Detection head over P3..P7 producing class, box, and quality maps."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        hidden_channels: int = 160,
        depth: int = 2,
        norm_groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if depth <= 0:
            raise ValueError("depth must be > 0")
        if norm_groups <= 0:
            raise ValueError("norm_groups must be > 0")
        if hidden_channels % norm_groups != 0:
            raise ValueError("norm_groups must divide hidden_channels")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)

        def _make_tower() -> nn.Sequential:
            layers: list[nn.Module] = []
            in_c = self.in_channels
            for _ in range(depth):
                layers.extend(
                    [
                        nn.Conv2d(
                            in_c,
                            self.hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.GroupNorm(num_groups=norm_groups, num_channels=self.hidden_channels),
                        nn.SiLU(inplace=False),
                    ]
                )
                in_c = self.hidden_channels
            return nn.Sequential(*layers)

        self.cls_tower = _make_tower()
        self.box_tower = _make_tower()
        self.quality_tower = _make_tower()

        self.cls_pred = nn.Conv2d(self.hidden_channels, self.num_classes, kernel_size=1, bias=True)
        self.box_pred = nn.Conv2d(self.hidden_channels, 4, kernel_size=1, bias=True)
        self.quality_pred = nn.Conv2d(self.hidden_channels, 1, kernel_size=1, bias=True)

        self.p6_from_p5 = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.p7_from_p6 = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def _validate_feature(self, name: str, feature: Tensor, *, batch_size: int | None) -> None:
        if feature.ndim != 4:
            raise ValueError(f"{name} must be [B,C,H,W]")
        if feature.shape[1] != self.in_channels:
            raise ValueError(f"{name} channel dim must be {self.in_channels}")
        if batch_size is not None and feature.shape[0] != batch_size:
            raise ValueError("all pyramid levels must share batch size")

    def _prepare_levels(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        if "P3" not in features or "P4" not in features or "P5" not in features:
            raise ValueError("features must contain P3, P4, and P5")

        p3 = features["P3"]
        p4 = features["P4"]
        p5 = features["P5"]
        self._validate_feature("P3", p3, batch_size=None)
        self._validate_feature("P4", p4, batch_size=p3.shape[0])
        self._validate_feature("P5", p5, batch_size=p3.shape[0])

        p6 = features.get("P6")
        if p6 is None:
            p6 = self.p6_from_p5(p5)
        else:
            self._validate_feature("P6", p6, batch_size=p3.shape[0])

        p7 = features.get("P7")
        if p7 is None:
            p7 = self.p7_from_p6(p6)
        else:
            self._validate_feature("P7", p7, batch_size=p3.shape[0])

        return {"P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7}

    def forward(self, features: dict[str, Tensor]) -> DetHeadOutput:
        levels = self._prepare_levels(features)
        ordered_levels = ("P3", "P4", "P5", "P6", "P7")

        cls_logits: dict[str, Tensor] = {}
        box_reg: dict[str, Tensor] = {}
        quality: dict[str, Tensor] = {}

        for name in ordered_levels:
            x = levels[name]
            cls_logits[name] = self.cls_pred(self.cls_tower(x))
            box_reg[name] = self.box_pred(self.box_tower(x))
            quality[name] = self.quality_pred(self.quality_tower(x))

        return DetHeadOutput(
            cls_logits=cls_logits,
            box_reg=box_reg,
            quality=quality,
            features=levels,
        )
