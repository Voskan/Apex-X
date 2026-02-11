from __future__ import annotations

import torch
from torch import nn

from apex_x.train import apply_pcgradpp, group_loss_terms


class TinyConflictNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(1, 1, bias=False)
        self.det_head = nn.Linear(1, 1, bias=True)
        self.seg_head = nn.Linear(1, 1, bias=True)

        with torch.no_grad():
            self.shared.weight.fill_(0.0)
            self.det_head.weight.fill_(1.0)
            self.det_head.bias.fill_(0.0)
            self.seg_head.weight.fill_(1.0)
            self.seg_head.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.shared(x)
        det = self.det_head(trunk)
        seg = self.seg_head(trunk)
        return det, seg


def test_group_loss_terms_uses_canonical_order_then_sorted_extras() -> None:
    losses = {
        "seg_boundary": torch.tensor(1.0),
        "det_box": torch.tensor(2.0),
        "other_loss": torch.tensor(3.0),
        "det_cls": torch.tensor(4.0),
        "seg_mask": torch.tensor(5.0),
    }
    groups = group_loss_terms(losses)
    assert tuple(group.name for group in groups) == (
        "det_cls",
        "det_box",
        "seg_mask",
        "seg_boundary",
        "other_loss",
    )


def test_pcgradpp_projects_conflicts_only_for_shared_params() -> None:
    torch.manual_seed(0)
    net = TinyConflictNet()
    x = torch.ones((1, 1), dtype=torch.float32)
    det_pred, seg_pred = net(x)

    det_cls = (det_pred - 2.0).pow(2).mean()
    det_box = det_pred.pow(2).mean() * 0.0
    seg_mask = (seg_pred + 1.0).pow(2).mean()
    seg_boundary = seg_pred.pow(2).mean() * 0.0
    losses = {
        "det_cls": det_cls,
        "det_box": det_box,
        "seg_mask": seg_mask,
        "seg_boundary": seg_boundary,
    }

    total_loss = det_cls + det_box + seg_mask + seg_boundary
    naive_shared_grad = torch.autograd.grad(total_loss, [net.shared.weight], retain_graph=True)[0]
    head_params = [
        net.det_head.weight,
        net.det_head.bias,
        net.seg_head.weight,
        net.seg_head.bias,
    ]
    expected_head_grads = torch.autograd.grad(total_loss, head_params, retain_graph=True)

    diag = apply_pcgradpp(
        loss_terms=losses,
        shared_params=[net.shared.weight],
        head_params=head_params,
    )

    assert diag.group_names == ("det_cls", "det_box", "seg_mask", "seg_boundary")
    assert diag.conflicting_pairs > 0
    assert diag.projected_pairs > 0
    assert diag.total_pairs > 0
    assert diag.conflicting_pairs_after <= diag.conflicting_pairs
    assert 0.0 <= diag.conflict_rate_after <= diag.conflict_rate_before <= 1.0

    assert net.shared.weight.grad is not None
    assert abs(float(naive_shared_grad.item())) > 1e-6
    assert abs(float(net.shared.weight.grad.item())) < 1e-6

    for parameter, expected in zip(head_params, expected_head_grads, strict=True):
        assert parameter.grad is not None
        assert torch.allclose(parameter.grad, expected, atol=1e-6, rtol=1e-6)


def test_pcgradpp_with_no_shared_params_returns_zero_conflict_rates() -> None:
    x = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    losses = {"det_cls": x.square()}

    diag = apply_pcgradpp(loss_terms=losses, shared_params=(), head_params=())

    assert diag.shared_param_count == 0
    assert diag.total_pairs == 0
    assert diag.conflicting_pairs == 0
    assert diag.conflicting_pairs_after == 0
    assert diag.conflict_rate_before == 0.0
    assert diag.conflict_rate_after == 0.0
