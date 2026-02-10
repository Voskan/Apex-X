from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_x.infer import (
    evaluate_fixture_file,
    evaluate_fixture_payload,
    tiny_eval_fixture_payload,
    write_eval_reports,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eval_tiny_fixture.json"


def test_eval_fixture_metrics_tiny_perfect_case() -> None:
    summary = evaluate_fixture_file(FIXTURE)
    assert summary.det_map == pytest.approx(1.0, abs=1e-9)
    assert summary.det_ap50 == pytest.approx(1.0, abs=1e-9)
    assert summary.det_ap75 == pytest.approx(1.0, abs=1e-9)
    assert summary.mask_map == pytest.approx(1.0, abs=1e-9)
    assert summary.mask_ap50 == pytest.approx(1.0, abs=1e-9)
    assert summary.mask_ap75 == pytest.approx(1.0, abs=1e-9)
    assert summary.semantic_miou == pytest.approx(1.0, abs=1e-9)
    assert summary.panoptic_pq == pytest.approx(1.0, abs=1e-9)
    assert summary.panoptic_source == "fallback"


def test_eval_reports_emit_json_and_markdown(tmp_path: Path) -> None:
    summary = evaluate_fixture_file(FIXTURE)
    json_path, md_path = write_eval_reports(
        summary,
        json_path=tmp_path / "eval.json",
        markdown_path=tmp_path / "eval.md",
    )
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "generated_at_utc" in payload
    assert payload["summary"]["det"]["map"] == pytest.approx(1.0, abs=1e-9)
    assert payload["summary"]["inst_seg"]["map"] == pytest.approx(1.0, abs=1e-9)
    assert payload["summary"]["semantic"]["miou"] == pytest.approx(1.0, abs=1e-9)
    assert payload["summary"]["panoptic"]["pq"] == pytest.approx(1.0, abs=1e-9)

    md = md_path.read_text(encoding="utf-8")
    assert "COCO mAP (det)" in md
    assert "COCO mAP (inst-seg)" in md
    assert "mIoU (semantic)" in md
    assert "PQ (panoptic)" in md


def test_builtin_tiny_fixture_payload_evaluates() -> None:
    payload = tiny_eval_fixture_payload()
    summary = evaluate_fixture_payload(payload)
    assert summary.det_map == pytest.approx(1.0, abs=1e-9)
