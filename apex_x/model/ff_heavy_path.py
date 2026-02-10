from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as f

from apex_x.kernels.triton import BidirectionalMergeMode, ScanDirection, tilessm_scan_dispatch
from apex_x.tiles import OverlapMode, TilePackTorch, TileUnpackTorch
from apex_x.utils import StableBidirectionalStateSpaceScan, StableStateSpaceScan

from .film import TileFiLM
from .fusion_gate import FusionGate
from .tile_refine_block import TileRefineBlock


@dataclass(frozen=True)
class FFHeavyPathOutput:
    """Heavy FF path outputs aligned to dense feature shape."""

    heavy_features: Tensor  # [B,C,H,W] final fused heavy output
    detail_map: Tensor  # [B,C,H,W] additive detail contribution vs dense input
    alpha: Tensor  # [B,1,H,W] fusion gate map
    tokens: Tensor  # [B,K,C]
    mixed_tokens: Tensor  # [B,K,C]
    gamma: Tensor  # [B,K,C]
    beta: Tensor  # [B,K,C]
    state: Tensor  # [B,S,C], S=1 forward or S=2 bidirectional


class FFHeavyPath(nn.Module):
    """Tile heavy path: pack -> tokenize -> scan -> FiLM -> refine -> unpack -> fuse."""

    def __init__(
        self,
        channels: int,
        tile_size: int,
        *,
        order_mode: str = "hilbert",
        scan_mode: str = "bidirectional",
        overlap_mode: OverlapMode = "override",
        blend_alpha: float = 0.5,
        use_refine: bool = True,
        use_fusion_gate: bool = True,
        use_triton_inference_scan: bool = False,
        gamma_limit: float = 1.0,
        refine_norm_groups: int = 1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if scan_mode not in {"forward", "bidirectional"}:
            raise ValueError("scan_mode must be 'forward' or 'bidirectional'")
        if not (0.0 <= blend_alpha <= 1.0):
            raise ValueError("blend_alpha must be within [0, 1]")

        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.order_mode = order_mode
        self.scan_mode = scan_mode
        self.overlap_mode = overlap_mode
        self.blend_alpha = float(blend_alpha)
        self.use_fusion_gate = bool(use_fusion_gate)
        self.use_triton_inference_scan = bool(use_triton_inference_scan)

        self.packer = TilePackTorch()
        self.unpacker = TileUnpackTorch()
        self.scan_forward = StableStateSpaceScan(channels=self.channels)
        self.scan_bidirectional = StableBidirectionalStateSpaceScan(channels=self.channels)
        self.film = TileFiLM(
            token_dim=self.channels,
            tile_channels=self.channels,
            gamma_limit=gamma_limit,
        )
        self.refine: nn.Module
        if use_refine:
            self.refine = TileRefineBlock(
                in_channels=self.channels,
                out_channels=self.channels,
                use_residual=True,
                norm_groups=refine_norm_groups,
            )
        else:
            self.refine = nn.Identity()
        self.fusion_gate = FusionGate() if self.use_fusion_gate else None

    def _align_proxy(self, proxy: Tensor | None, *, like: Tensor, fallback: float) -> Tensor:
        bsz, _, height, width = like.shape
        if proxy is None:
            return torch.full(
                (bsz, 1, height, width),
                float(fallback),
                dtype=like.dtype,
                device=like.device,
            )

        if proxy.ndim == 3:
            proxy = proxy.unsqueeze(1)
        if proxy.ndim != 4:
            raise ValueError("proxy must be [B,1,H,W] or [B,H,W]")
        if proxy.shape[0] != bsz or proxy.shape[1] != 1:
            raise ValueError("proxy must have batch B and single channel")
        proxy_clean = torch.nan_to_num(
            proxy.to(dtype=like.dtype, device=like.device),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if proxy_clean.shape[2:] == (height, width):
            return proxy_clean
        return f.interpolate(
            proxy_clean,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def _scan(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if self.use_triton_inference_scan and not self.training:
            if self.scan_mode == "bidirectional":
                mixed_f, state_f = self._scan_dispatch_direction(
                    tokens,
                    self.scan_bidirectional.forward_scan,
                    direction="forward",
                )
                mixed_b, state_b = self._scan_dispatch_direction(
                    tokens,
                    self.scan_bidirectional.backward_scan,
                    direction="backward",
                )
                gate = self.scan_bidirectional.merge_gate().to(
                    dtype=tokens.dtype,
                    device=tokens.device,
                ).reshape(1, 1, -1)
                mixed = gate * mixed_f + (1.0 - gate) * mixed_b
                states = torch.stack((state_f, state_b), dim=1)
                return mixed, states

            mixed, state_f = self._scan_dispatch_direction(
                tokens,
                self.scan_forward,
                direction="forward",
            )
            return mixed, state_f.unsqueeze(1)

        if self.scan_mode == "bidirectional":
            mixed, state_f, state_b = self.scan_bidirectional(tokens)
            state = torch.stack((state_f, state_b), dim=1)
            return mixed, state

        mixed, state_f = self.scan_forward(tokens)
        state = state_f.unsqueeze(1)
        return mixed, state

    def _scan_dispatch_direction(
        self,
        tokens: Tensor,
        scan_module: StableStateSpaceScan,
        *,
        direction: ScanDirection,
        merge_mode: BidirectionalMergeMode = "avg",
        merge_gate: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        dispatch = tilessm_scan_dispatch(
            tokens,
            decay=scan_module.constrained_decay()
            .to(dtype=tokens.dtype, device=tokens.device)
            .detach(),
            input_gain=scan_module.constrained_input_gain().to(
                dtype=tokens.dtype, device=tokens.device
            ).detach(),
            output_gain=scan_module.constrained_output_gain().to(
                dtype=tokens.dtype, device=tokens.device
            ).detach(),
            state_bias=scan_module.state_bias.to(dtype=tokens.dtype, device=tokens.device).detach(),
            direction=direction,
            merge_mode=merge_mode,
            merge_gate=None if merge_gate is None else merge_gate.detach(),
            prefer_triton=True,
            allow_fallback=True,
            inference_only=True,
        )
        return dispatch.y, dispatch.final_state

    def forward(
        self,
        dense_features: Tensor,
        tile_indices: Tensor,
        *,
        boundary_proxy: Tensor | None = None,
        uncertainty_proxy: Tensor | None = None,
    ) -> FFHeavyPathOutput:
        if dense_features.ndim != 4:
            raise ValueError("dense_features must be [B,C,H,W]")
        if tile_indices.ndim != 2:
            raise ValueError("tile_indices must be [B,K]")
        if dense_features.shape[0] != tile_indices.shape[0]:
            raise ValueError("dense_features and tile_indices batch dimensions must match")
        if dense_features.shape[1] != self.channels:
            raise ValueError("dense_features channel dimension does not match channels")
        if tile_indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("tile_indices must be an integer tensor")

        packed, meta = self.packer.pack(
            feature_map=dense_features,
            indices=tile_indices,
            tile_size=self.tile_size,
            order_mode=self.order_mode,
        )

        tokens = packed.mean(dim=(-2, -1))
        if packed.shape[1] == 0:
            mixed = tokens
            gamma = tokens
            beta = tokens
            refined = packed
            state_slots = 2 if self.scan_mode == "bidirectional" else 1
            state = dense_features.new_zeros((dense_features.shape[0], state_slots, self.channels))
            heavy_unfused = dense_features.clone()
        else:
            mixed, state = self._scan(tokens)
            modulated, gamma, beta = self.film(mixed, packed)
            refined = self.refine(modulated)
            if refined.shape != packed.shape:
                raise ValueError("refine block must preserve packed tile shape")
            heavy_unfused, _ = self.unpacker.unpack(
                base_map=dense_features,
                packed_out=refined,
                meta=meta,
                level_priority=1,
                overlap_mode=self.overlap_mode,
                blend_alpha=self.blend_alpha,
            )

        if self.fusion_gate is None:
            alpha = torch.ones(
                (dense_features.shape[0], 1, dense_features.shape[2], dense_features.shape[3]),
                dtype=dense_features.dtype,
                device=dense_features.device,
            )
            heavy_features = heavy_unfused
        else:
            boundary = self._align_proxy(boundary_proxy, like=dense_features, fallback=1.0)
            uncertainty = self._align_proxy(uncertainty_proxy, like=dense_features, fallback=1.0)
            heavy_features, alpha = self.fusion_gate(
                base_features=dense_features,
                heavy_features=heavy_unfused,
                boundary_proxy=boundary,
                uncertainty_proxy=uncertainty,
            )

        detail_map = heavy_features - dense_features
        return FFHeavyPathOutput(
            heavy_features=heavy_features,
            detail_map=detail_map,
            alpha=alpha,
            tokens=tokens,
            mixed_tokens=mixed,
            gamma=gamma,
            beta=beta,
            state=state,
        )
