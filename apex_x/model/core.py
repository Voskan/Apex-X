from __future__ import annotations

from typing import Any, cast

import numpy as np

from apex_x.config import ApexXConfig
from apex_x.routing import (
    BudgetControllerProtocol,
    BudgetDualController,
    GreedyBudgetController,
    IdentityRouter,
    RouterProtocol,
    build_routing_diagnostics,
    expected_cost,
    hysteresis_update,
)
from apex_x.tiles import NumpyTileCodec, TileCodecProtocol, tile_grid_shape
from apex_x.utils import get_logger, log_event, tile_ssm_scan

LOGGER = get_logger(__name__)


def _downsample_mean(x: np.ndarray, stride: int) -> np.ndarray:
    b, c, h, w = x.shape
    hs = h // stride
    ws = w // stride
    x_crop = x[:, :, : hs * stride, : ws * stride]
    x_view = x_crop.reshape(b, c, hs, stride, ws, stride)
    return cast(np.ndarray, x_view.mean(axis=(3, 5)))


class ApexXModel:
    """CPU-only baseline model implementing Apex-X routing contracts."""

    def __init__(
        self,
        config: ApexXConfig | None = None,
        router: RouterProtocol | None = None,
        budget_controller: BudgetControllerProtocol | None = None,
        tile_codec: TileCodecProtocol | None = None,
    ):
        self.config = config or ApexXConfig()
        self.router = router or IdentityRouter()
        self.budget_controller = budget_controller or GreedyBudgetController()
        self.tile_codec = tile_codec or NumpyTileCodec()
        self.prev_mask: list[int] | None = None
        self.dual_controller = BudgetDualController(
            budget=self.config.routing.budget_total,
            mu_init=self.config.train.mu_init,
            mu_lr=self.config.train.mu_lr,
            mu_min=self.config.train.mu_min,
            mu_max=self.config.train.mu_max,
            logger_name="model.dual",
        )
        self.mu_history: list[float] = [self.dual_controller.mu]

    def _tile_signals(self, ff8: np.ndarray, tile_size: int) -> tuple[list[float], tuple[int, int]]:
        _, _, h, w = ff8.shape
        gh, gw = tile_grid_shape(h, w, tile_size)
        signals: list[float] = []

        for ty in range(gh):
            for tx in range(gw):
                y = ty * tile_size
                x = tx * tile_size
                tile = ff8[0, :, y : y + tile_size, x : x + tile_size]
                signal = float(np.var(tile) + np.mean(np.abs(tile)))
                signals.append(signal)
        return signals, (gh, gw)

    def forward(self, image: np.ndarray, *, update_dual: bool = False) -> dict[str, Any]:
        """Run one frame through the CPU baseline.

        image: [B,3,H,W] float32
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be [B,3,H,W]")
        if image.shape[0] != 1:
            raise ValueError("baseline supports batch size 1 for now")

        cfg = self.config
        model_cfg = cfg.model
        router_enabled = model_cfg.router_enabled()
        ssm_enabled = model_cfg.ssm_enabled()
        nesting_depth = model_cfg.effective_nesting_depth()
        pv16 = _downsample_mean(image, cfg.model.pv_stride)
        ff8 = _downsample_mean(image, cfg.model.ff_primary_stride)

        tile_size = cfg.model.tile_size_l0
        signals, (grid_h, grid_w) = self._tile_signals(ff8, tile_size)
        utilities = self.router.predict_utilities(signals) if router_enabled else list(signals)
        costs = [1.0] * len(utilities)
        if router_enabled:
            selected, spent_budget_b1 = self.budget_controller.select(
                utilities=utilities,
                costs=costs,
                budget=cfg.routing.budget_b1,
                kmax=cfg.model.kmax_l0,
            )
        else:
            utilities = list(signals)
            selected = list(range(len(utilities)))
            spent_budget_b1 = float(sum(costs))

        if self.prev_mask is None:
            self.prev_mask = [0] * len(utilities)
        if router_enabled:
            hyst_mask = hysteresis_update(
                utilities,
                self.prev_mask,
                cfg.routing.theta_on,
                cfg.routing.theta_off,
            )
        else:
            hyst_mask = [1] * len(utilities)
        self.prev_mask = hyst_mask

        if update_dual:
            if router_enabled:
                exp_cost = expected_cost(
                    utilities=utilities,
                    c_heavy=cfg.routing.cost_heavy,
                    c_cheap=cfg.routing.cost_cheap,
                )
            else:
                exp_cost = float(len(utilities)) * cfg.routing.cost_heavy
            self.dual_controller.update_mu(exp_cost, budget=cfg.routing.budget_total)
            self.mu_history.append(self.dual_controller.mu)

        total_l0 = grid_h * grid_w
        total_l1 = total_l0 * 4 if nesting_depth >= 1 else 0
        total_l2 = total_l1 * 4 if nesting_depth >= 2 else 0
        routing_diagnostics = build_routing_diagnostics(
            utilities_by_level={"l0": utilities},
            selected_counts={
                "l0": len(selected),
                "l1": 0,
                "l2": 0,
            },
            total_counts={
                "l0": total_l0,
                "l1": total_l1,
                "l2": total_l2,
            },
            budget_used={
                "b1": spent_budget_b1,
                "b2": 0.0,
                "b3": 0.0,
                "total": spent_budget_b1,
            },
            budget_total={
                "b1": cfg.routing.budget_b1,
                "b2": cfg.routing.budget_b2,
                "b3": cfg.routing.budget_b3,
                "total": cfg.routing.budget_total,
            },
            mu_history=self.mu_history,
        )

        selected_arr = np.asarray(selected, dtype=np.int64)
        if selected_arr.size == 0:
            selected_arr = np.asarray([0], dtype=np.int64)
        k = selected_arr.size if not router_enabled else min(cfg.model.kmax_l0, selected_arr.size)
        idx = selected_arr[:k][None, :]

        packed, meta = self.tile_codec.pack(ff8, idx, tile_size, order_mode="hilbert")
        tokens = packed.mean(axis=(3, 4))
        if ssm_enabled:
            mixed, state = tile_ssm_scan(tokens)
        else:
            mixed = np.zeros_like(tokens)
            state = np.zeros((tokens.shape[0], tokens.shape[2]), dtype=tokens.dtype)

        gamma = np.tanh(mixed).astype(packed.dtype, copy=False)
        beta = mixed.astype(packed.dtype, copy=False)
        gamma_5d = gamma[:, :, :, None, None]
        beta_5d = beta[:, :, :, None, None]
        packed_out = packed * (np.ones_like(gamma_5d) + gamma_5d) + beta_5d

        merged, _ = self.tile_codec.unpack(ff8, packed_out, meta, level_priority=1)

        det_score = float(np.clip(np.max(merged), 0.0, 1.0))
        det_box = np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32)
        log_event(
            LOGGER,
            "model_forward",
            level="DEBUG",
            fields={
                "selected_tiles": len(selected[:k]),
                "selected_ratio_l0": round(routing_diagnostics["selected_ratios"]["l0"], 6),
                "budget_b1_ratio": round(routing_diagnostics["budget_usage"]["b1"]["ratio"], 6),
                "mu_last": round(self.mu_history[-1], 6),
                "router_enabled": router_enabled,
                "ssm_enabled": ssm_enabled,
                "nesting_depth": nesting_depth,
                "det_score": round(det_score, 6),
            },
        )

        return {
            "pv16": pv16,
            "ff8": ff8,
            "merged": merged,
            "selected_tiles": selected[:k],
            "hysteresis_mask": hyst_mask,
            "routing_diagnostics": routing_diagnostics,
            "feature_toggles": {
                "router_off": not router_enabled,
                "no_nesting": bool(model_cfg.disable_nesting),
                "no_ssm": not ssm_enabled,
                "no_distill": bool(cfg.train.disable_distill),
                "no_pcgradpp": not cfg.train.pcgradpp_enabled(),
            },
            "ssm_state": state,
            "det": {
                "boxes": det_box,
                "scores": np.asarray([det_score], dtype=np.float32),
                "class_ids": np.asarray([0], dtype=np.int64),
            },
        }
