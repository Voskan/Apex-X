from __future__ import annotations

import json
from pathlib import Path


def test_checkpoint_notebook_uses_secure_loader_and_retry_gate() -> None:
    notebook_path = Path("notebooks/checkpoint_image_inference.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    combined_source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    assert "safe_torch_load" in combined_source
    assert "extract_model_state_dict" in combined_source
    assert "align_mode = widgets.Dropdown" in combined_source
    assert "Retrying with square resize to stabilize token grid" in combined_source
    assert "Cannot reshape" in combined_source
