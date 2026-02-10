from __future__ import annotations

import pytest
import torch

import apex_x.model.ff_heavy_path as ff_heavy_path_module
from apex_x.kernels.triton.tilessm_scan import TileSSMScanDispatchResult, tilessm_scan_reference
from apex_x.model import FFHeavyPath


def test_ff_heavy_path_uses_tilessm_dispatch_only_in_eval_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_dispatch(
        tokens: torch.Tensor,
        *,
        decay: torch.Tensor,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        state_bias: torch.Tensor,
        **_: object,
    ) -> TileSSMScanDispatchResult:
        calls["count"] += 1
        y, state = tilessm_scan_reference(
            tokens,
            decay=decay,
            input_gain=input_gain,
            output_gain=output_gain,
            state_bias=state_bias,
        )
        return TileSSMScanDispatchResult(
            y=y,
            final_state=state,
            backend="reference",
            fallback_reason="fake",
        )

    monkeypatch.setattr(ff_heavy_path_module, "tilessm_scan_dispatch", _fake_dispatch)

    path = FFHeavyPath(
        channels=8,
        tile_size=4,
        scan_mode="forward",
        use_refine=False,
        use_fusion_gate=False,
        use_triton_inference_scan=True,
    ).cpu()
    dense = torch.randn((1, 8, 16, 16), dtype=torch.float32)
    idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    path.eval()
    _ = path(dense, idx)
    assert calls["count"] == 1

    path.train()
    _ = path(dense, idx)
    assert calls["count"] == 1
