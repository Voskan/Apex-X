from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from apex_x.infer import OfficialPQPaths, PQMetrics, evaluate_panoptic_quality

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / name
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _run_fixture(name: str) -> tuple[dict[str, Any], PQMetrics]:
    payload = _load_fixture(name)
    pred_map = torch.tensor(payload["pred_map"], dtype=torch.int64).unsqueeze(0)
    gt_map = torch.tensor(payload["gt_map"], dtype=torch.int64).unsqueeze(0)
    pred_segments = [cast(list[dict[str, Any]], payload["pred_segments"])]
    gt_segments = [cast(list[dict[str, Any]], payload["gt_segments"])]
    thing_class_ids = {
        int(v) for v in cast(list[int], payload["thing_class_ids"])
    }
    metrics = evaluate_panoptic_quality(
        pred_panoptic_map=pred_map,
        pred_segments_info=pred_segments,
        gt_panoptic_map=gt_map,
        gt_segments_info=gt_segments,
        thing_class_ids=thing_class_ids,
    )
    return payload, metrics


def test_pq_fixture_perfect_case() -> None:
    payload, metrics = _run_fixture("pq_case_perfect.json")
    expected = cast(dict[str, float], payload["expected"])

    assert metrics.used_official_api is False
    assert metrics.source == "fallback"
    assert metrics.all_pq == expected["all_pq"]
    assert metrics.all_sq == expected["all_sq"]
    assert metrics.all_rq == expected["all_rq"]
    assert metrics.things_pq == expected["things_pq"]
    assert metrics.stuff_pq == expected["stuff_pq"]
    assert set(metrics.per_class.keys()) == {1, 2}


def test_pq_fixture_partial_overlap_case() -> None:
    payload, metrics = _run_fixture("pq_case_partial.json")
    expected = cast(dict[str, float], payload["expected"])

    assert metrics.used_official_api is False
    assert metrics.source == "fallback"
    assert metrics.all_pq == pytest.approx(expected["all_pq"], abs=1e-6)
    assert metrics.all_sq == pytest.approx(expected["all_sq"], abs=1e-6)
    assert metrics.all_rq == pytest.approx(expected["all_rq"], abs=1e-6)
    assert metrics.things_pq == pytest.approx(expected["things_pq"], abs=1e-6)
    assert metrics.stuff_pq == pytest.approx(expected["stuff_pq"], abs=1e-6)

    class1 = metrics.per_class[1]
    class2 = metrics.per_class[2]
    assert (class1.tp, class1.fp, class1.fn) == (0, 1, 1)
    assert (class2.tp, class2.fp, class2.fn) == (1, 0, 0)


def test_pq_official_path_falls_back_to_minimal_when_unavailable() -> None:
    payload = _load_fixture("pq_case_perfect.json")
    pred_map = torch.tensor(payload["pred_map"], dtype=torch.int64).unsqueeze(0)
    gt_map = torch.tensor(payload["gt_map"], dtype=torch.int64).unsqueeze(0)
    pred_segments = [cast(list[dict[str, Any]], payload["pred_segments"])]
    gt_segments = [cast(list[dict[str, Any]], payload["gt_segments"])]

    metrics = evaluate_panoptic_quality(
        pred_panoptic_map=pred_map,
        pred_segments_info=pred_segments,
        gt_panoptic_map=gt_map,
        gt_segments_info=gt_segments,
        thing_class_ids={1},
        official_paths=OfficialPQPaths(
            gt_json=FIXTURES_DIR / "missing_gt.json",
            pred_json=FIXTURES_DIR / "missing_pred.json",
        ),
        use_official_if_available=True,
    )

    assert metrics.used_official_api is False
    assert metrics.source == "fallback"
    assert metrics.all_pq == 1.0
