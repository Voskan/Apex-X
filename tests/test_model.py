import numpy as np

from apex_x import ApexXModel


def test_cpu_baseline_forward_runs():
    model = ApexXModel()
    image = np.random.RandomState(0).rand(1, 3, 128, 128).astype(np.float32)
    out = model.forward(image)

    assert "det" in out
    assert out["pv16"].shape[2:] == (8, 8)
    assert len(out["selected_tiles"]) >= 1
