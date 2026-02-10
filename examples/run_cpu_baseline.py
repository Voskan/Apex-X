from __future__ import annotations

import numpy as np

from apex_x import ApexXModel


def main() -> None:
    model = ApexXModel()
    image = np.random.RandomState(42).rand(1, 3, 128, 128).astype(np.float32)
    out = model.forward(image)

    print("Apex-X CPU baseline run completed")
    print(f"Selected tiles: {len(out['selected_tiles'])}")
    print(f"DET score: {out['det']['scores'][0]:.4f}")


if __name__ == "__main__":
    main()
