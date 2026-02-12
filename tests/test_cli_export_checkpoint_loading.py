from __future__ import annotations

from pathlib import Path

import torch

from apex_x import cli


class _DummyExportModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.loaded_state_dict: dict[str, torch.Tensor] | None = None
        self.strict_load_calls: list[bool] = []
        self._expected_state = {
            "det_head.cls_pred.weight": torch.zeros((self.num_classes, 16, 1, 1)),
            "det_head.cls_pred.bias": torch.zeros((self.num_classes,)),
        }

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ):
        self.loaded_state_dict = state_dict
        self.strict_load_calls.append(bool(strict))
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return dict(self._expected_state)


def test_export_cmd_loads_structured_checkpoint_and_infers_num_classes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_models: list[_DummyExportModel] = []

    def _fake_build_teacher_for_export(config, num_classes: int) -> _DummyExportModel:
        model = _DummyExportModel(num_classes=num_classes)
        created_models.append(model)
        return model

    class _FakeExporter:
        def __init__(self, shape_mode: str = "static") -> None:
            self.shape_mode = shape_mode

        def export(self, model: _DummyExportModel, output_path: str) -> str:
            output = Path(output_path)
            output.write_text("{\"ok\": true}", encoding="utf-8")
            return str(output)

    monkeypatch.setattr(cli, "_build_teacher_for_export", _fake_build_teacher_for_export)
    monkeypatch.setattr(cli, "ApexXExporter", _FakeExporter)

    checkpoint = tmp_path / "structured.pt"
    state_dict = {
        "det_head.cls_pred.weight": torch.randn(5, 16, 1, 1),
        "det_head.cls_pred.bias": torch.randn(5),
    }
    torch.save({"model_state_dict": state_dict, "epoch": 1}, checkpoint)

    output_manifest = tmp_path / "manifest.json"
    cli.export_cmd(
        config="configs/satellite_1024.yaml",
        checkpoint=str(checkpoint),
        output=str(output_manifest),
        num_classes=3,
    )

    # First build uses requested classes, second rebuild follows checkpoint classes.
    assert len(created_models) == 2
    assert created_models[0].num_classes == 3
    assert created_models[1].num_classes == 5
    assert created_models[1].loaded_state_dict is not None
    assert created_models[1].loaded_state_dict["det_head.cls_pred.weight"].shape[0] == 5
    assert output_manifest.exists()


def test_export_cmd_loads_raw_state_dict_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_models: list[_DummyExportModel] = []

    def _fake_build_teacher_for_export(config, num_classes: int) -> _DummyExportModel:
        model = _DummyExportModel(num_classes=num_classes)
        created_models.append(model)
        return model

    class _FakeExporter:
        def __init__(self, shape_mode: str = "static") -> None:
            self.shape_mode = shape_mode

        def export(self, model: _DummyExportModel, output_path: str) -> str:
            output = Path(output_path)
            output.write_text("{\"ok\": true}", encoding="utf-8")
            return str(output)

    monkeypatch.setattr(cli, "_build_teacher_for_export", _fake_build_teacher_for_export)
    monkeypatch.setattr(cli, "ApexXExporter", _FakeExporter)

    checkpoint = tmp_path / "raw_state.pt"
    raw_state_dict = {
        "det_head.cls_pred.weight": torch.randn(3, 16, 1, 1),
        "det_head.cls_pred.bias": torch.randn(3),
    }
    torch.save(raw_state_dict, checkpoint)

    output_manifest = tmp_path / "manifest.json"
    cli.export_cmd(
        config="configs/satellite_1024.yaml",
        checkpoint=str(checkpoint),
        output=str(output_manifest),
        num_classes=3,
    )

    assert len(created_models) == 1
    assert created_models[0].loaded_state_dict is not None
    assert set(created_models[0].loaded_state_dict.keys()) == set(raw_state_dict.keys())
    assert created_models[0].strict_load_calls == [True]
    assert output_manifest.exists()


def test_export_cmd_supports_non_strict_checkpoint_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_models: list[_DummyExportModel] = []

    def _fake_build_teacher_for_export(config, num_classes: int) -> _DummyExportModel:
        model = _DummyExportModel(num_classes=num_classes)
        created_models.append(model)
        return model

    class _FakeExporter:
        def __init__(self, shape_mode: str = "static") -> None:
            self.shape_mode = shape_mode

        def export(self, model: _DummyExportModel, output_path: str) -> str:
            output = Path(output_path)
            output.write_text("{\"ok\": true}", encoding="utf-8")
            return str(output)

    monkeypatch.setattr(cli, "_build_teacher_for_export", _fake_build_teacher_for_export)
    monkeypatch.setattr(cli, "ApexXExporter", _FakeExporter)

    checkpoint = tmp_path / "structured.pt"
    state_dict = {"det_head.cls_pred.weight": torch.randn(3, 16, 1, 1)}
    torch.save({"model_state_dict": state_dict, "epoch": 1}, checkpoint)

    output_manifest = tmp_path / "manifest.json"
    cli.export_cmd(
        config="configs/satellite_1024.yaml",
        checkpoint=str(checkpoint),
        output=str(output_manifest),
        num_classes=3,
        strict_load=False,
    )

    assert len(created_models) == 1
    assert created_models[0].strict_load_calls == [False]
    assert output_manifest.exists()


def test_export_cmd_non_strict_skips_shape_mismatch_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_models: list[_DummyExportModel] = []

    def _fake_build_teacher_for_export(config, num_classes: int) -> _DummyExportModel:
        model = _DummyExportModel(num_classes=num_classes)
        created_models.append(model)
        return model

    class _FakeExporter:
        def __init__(self, shape_mode: str = "static") -> None:
            self.shape_mode = shape_mode

        def export(self, model: _DummyExportModel, output_path: str) -> str:
            output = Path(output_path)
            output.write_text("{\"ok\": true}", encoding="utf-8")
            return str(output)

    monkeypatch.setattr(cli, "_build_teacher_for_export", _fake_build_teacher_for_export)
    monkeypatch.setattr(cli, "ApexXExporter", _FakeExporter)

    checkpoint = tmp_path / "structured.pt"
    state_dict = {
        "det_head.cls_pred.weight": torch.randn(7, 16, 1, 1),
        # Deliberate mismatch (expected [7] after num_classes inference).
        "det_head.cls_pred.bias": torch.randn(3),
    }
    torch.save({"model_state_dict": state_dict, "epoch": 1}, checkpoint)

    output_manifest = tmp_path / "manifest.json"
    cli.export_cmd(
        config="configs/satellite_1024.yaml",
        checkpoint=str(checkpoint),
        output=str(output_manifest),
        num_classes=3,
        strict_load=False,
    )

    assert len(created_models) == 2
    loaded = created_models[-1].loaded_state_dict
    assert loaded is not None
    assert "det_head.cls_pred.weight" in loaded
    assert "det_head.cls_pred.bias" not in loaded
