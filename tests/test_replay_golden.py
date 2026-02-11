from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apex_x.routing import deterministic_greedy_selection, deterministic_two_stage_selection
from apex_x.tiles import TileSelection, TileSelectionTrace
from apex_x.utils import build_replay_manifest, hash_json_sha256, seed_all

_FIXTURE_NAMES = ("replay_golden_small.json", "replay_golden_medium.json")


def _fixture_path(name: str) -> Path:
    return Path("tests/fixtures") / name


def _load_fixture(name: str) -> dict[str, Any]:
    payload = json.loads(_fixture_path(name).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture {name} must contain an object payload.")
    return payload


def _run_replay_fixture(fixture: dict[str, Any]) -> tuple[dict[str, Any], TileSelectionTrace]:
    name = str(fixture["name"])
    seed = int(fixture["seed"])
    mode = str(fixture["mode"])
    inputs = fixture["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("Fixture inputs must be an object.")

    seed_all(seed=seed, deterministic=True)

    if mode == "greedy":
        result = deterministic_greedy_selection(
            utilities=[float(v) for v in inputs["utilities"]],
            delta_costs=[float(v) for v in inputs["delta_costs"]],
            budget=float(inputs["budget"]),
            kmax=int(inputs["kmax"]),
        )
        summary = {
            "selected_indices": result.selected_indices,
            "ordered_candidates": result.ordered_candidates,
            "kmax_buffer": result.kmax_buffer,
            "valid_count": result.valid_count,
            "spent_budget": result.spent_budget,
        }
        trace = TileSelectionTrace(
            selections=[
                TileSelection(
                    level="l0",
                    indices=result.selected_indices,
                    ordered_indices=result.selected_indices,
                    meta={
                        "ordered_candidates": result.ordered_candidates,
                        "kmax_buffer": result.kmax_buffer,
                        "scores": result.scores,
                    },
                    budgets_used={"b1": result.spent_budget},
                )
            ],
            run_meta={"fixture": name, "seed": seed, "mode": mode},
        )
        return summary, trace

    if mode == "two_stage":
        result = deterministic_two_stage_selection(
            l0_utilities=[float(v) for v in inputs["l0_utilities"]],
            l0_delta_costs=[float(v) for v in inputs["l0_delta_costs"]],
            split_utilities=[float(v) for v in inputs["split_utilities"]],
            split_overheads=[float(v) for v in inputs["split_overheads"]],
            budget_b1=float(inputs["budget_b1"]),
            budget_b2=float(inputs["budget_b2"]),
            kmax_l0=int(inputs["kmax_l0"]),
            kmax_l1=int(inputs["kmax_l1"]),
            l0_grid_h=int(inputs["l0_grid_h"]),
            l0_grid_w=int(inputs["l0_grid_w"]),
            l1_order_mode=str(inputs["l1_order_mode"]),
        )
        summary = {
            "l0_selected_indices": result.l0.selected_indices,
            "l0_ordered_candidates": result.l0.ordered_candidates,
            "split_parent_indices": result.split_parent_indices,
            "split_parent_order": result.split_parent_order,
            "l1_indices": result.l1_indices,
            "l1_ordered_indices": result.l1_ordered_indices,
            "l1_kmax_buffer": result.l1_kmax_buffer,
            "l1_valid_count": result.l1_valid_count,
            "spent_budget_b1": result.l0.spent_budget,
            "spent_budget_b2": result.split_spent_budget,
        }
        trace = TileSelectionTrace(
            selections=[
                TileSelection(
                    level="l0",
                    indices=result.l0.selected_indices,
                    ordered_indices=result.l0.selected_indices,
                    meta={
                        "ordered_candidates": result.l0.ordered_candidates,
                        "scores": result.l0.scores,
                        "kmax_buffer": result.l0.kmax_buffer,
                    },
                    budgets_used={"b1": result.l0.spent_budget},
                ),
                TileSelection(
                    level="l1",
                    indices=result.l1_indices,
                    ordered_indices=result.l1_ordered_indices,
                    meta={
                        "split_parent_indices": result.split_parent_indices,
                        "split_parent_order": result.split_parent_order,
                        "l1_kmax_buffer": result.l1_kmax_buffer,
                    },
                    budgets_used={"b2": result.split_spent_budget},
                ),
            ],
            run_meta={"fixture": name, "seed": seed, "mode": mode},
        )
        return summary, trace

    raise ValueError(f"Unsupported replay fixture mode: {mode}")


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_replay_fixture_matches_expected_selection_and_hashes(
    fixture_name: str, tmp_path: Path
) -> None:
    fixture = _load_fixture(fixture_name)
    expected = fixture.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("Fixture expected must be an object.")

    summary, trace = _run_replay_fixture(fixture)
    assert summary == expected["summary"]

    trace_hash = hash_json_sha256(trace.to_dict())
    assert trace_hash == expected["trace_sha256"]

    trace_path_a = tmp_path / f"{fixture['name']}_a.json"
    trace_path_b = tmp_path / "nested" / f"{fixture['name']}_b.json"
    trace.save_json(trace_path_a)
    trace.save_json(trace_path_b)

    manifest_a = build_replay_manifest(
        seed=int(fixture["seed"]),
        config=dict(fixture["config"]),
        artifact_paths={"trace": trace_path_a},
        include_determinism_state=False,
    )
    manifest_b = build_replay_manifest(
        seed=int(fixture["seed"]),
        config=dict(fixture["config"]),
        artifact_paths={"trace": trace_path_b},
        include_determinism_state=False,
    )
    assert manifest_a["manifest_sha256"] == expected["manifest_sha256"]
    assert manifest_b["manifest_sha256"] == expected["manifest_sha256"]
