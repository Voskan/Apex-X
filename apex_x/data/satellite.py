from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

try:
    import rasterio  # type: ignore
except ImportError:
    rasterio = None

from apex_x.data.transforms import TransformSample


@dataclass(frozen=True, slots=True)
class SatelliteTile:
    image_path: Path
    mask_path: Path | None
    x_off: int
    y_off: int
    width: int
    height: int


class SatelliteDataset:
    """Dataset for training on tiled large-format satellite imagery."""

    def __init__(
        self,
        root_dir: str | Path,
        tile_size: int = 512,
        stride: int = 256,
        image_glob: str = "*.tif",
        mask_glob: str = "*.mask.tif",
        require_mask: bool = True,
        limit_tiles: int | None = None,
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.tiles: list[SatelliteTile] = []

        if not self.root_dir.exists():
             # Fail gracefully if directory doesn't exist yet, to allow test setup
             return

        images = sorted(list(self.root_dir.glob(image_glob)))
        for img_path in images:
            # Simple heuristic for mask pair: image.tif -> image.mask.tif
            # Or assume mask_glob pattern relative to root
            # Here we try to find a matching mask by replacement or specific pattern
            # For robustness, we assume the mask has the same stem + suffix diff
            
            # Common pattern: image_01.tif -> image_01_mask.tif
            # We'll use a lenient discovery: check if a file exists with the mask pattern 
            # constructed from the image name. 
            # Actually, standardizing on a simple convention is better:
            # valid_mask = img_path.parent / img_path.name.replace(".tif", "_mask.tif")
            # But let's stick to the glob provided in args if possible. 
            
            # Implementation: Scan dir for masks, match by stem
            # This is slow for huge dirs, but safe.
            # Faster approach: just check existence of expected mask path.
            
            mask_candidate = img_path.parent / img_path.name.replace(Path(image_glob).suffix, "_mask" + Path(image_glob).suffix)
            mask_path = mask_candidate if mask_candidate.exists() else None
            
            if require_mask and mask_path is None:
                continue

            self._slice_image(img_path, mask_path)
            
            if limit_tiles and len(self.tiles) >= limit_tiles:
                break
                
    def _slice_image(self, img_path: Path, mask_path: Path | None) -> None:
        # Use rasterio if available for speed/metadata, else PIL
        width, height = self._get_dims(img_path)
        
        y_steps = (height - self.tile_size) // self.stride + 1
        x_steps = (width - self.tile_size) // self.stride + 1
        
        if y_steps < 1: y_steps = 1
        if x_steps < 1: x_steps = 1
        
        for y in range(y_steps):
            for x in range(x_steps):
                y_off = y * self.stride
                x_off = x * self.stride
                
                # Check boundaries
                if y_off + self.tile_size > height:
                    y_off = height - self.tile_size
                if x_off + self.tile_size > width:
                    x_off = width - self.tile_size
                    
                self.tiles.append(
                    SatelliteTile(
                        image_path=img_path,
                        mask_path=mask_path,
                        x_off=max(0, y_off),
                        y_off=max(0, x_off),
                        width=self.tile_size,
                        height=self.tile_size,
                    )
                )

    def _get_dims(self, path: Path) -> tuple[int, int]:
        if rasterio:
            with rasterio.open(path) as src:
                return src.width, src.height
        with Image.open(path) as img:
            return img.width, img.height

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> TransformSample:
        tile = self.tiles[idx]
        
        # Load Raw Data
        image_np = self._load_patch(tile.image_path, tile)
        if tile.mask_path:
            mask_np = self._load_patch(tile.mask_path, tile, is_mask=True)
            # Binary mask to instance masks? 
            # For now, assume semantic mask (H,W) -> instance mask (N,H,W) via connected components
            # or simply one class.
            # Let's do simple semantic-to-instance if needed, or just return as is.
            # TransformSample expects masks as [N, H, W] bool.
            # We'll treat all non-zero pixels as one object for single-class segmentation, 
            # or connected components for instance.
            import cv2
            num_labels, labels = cv2.connectedComponents(mask_np.astype(np.uint8))
            
            masks = []
            boxes = []
            class_ids = []
            
            for i in range(1, num_labels):
                component_mask = (labels == i)
                if component_mask.sum() < 10: # Min area
                    continue
                    
                masks.append(component_mask)
                
                # Bounding box
                y_indices, x_indices = np.where(component_mask)
                y1, y2 = y_indices.min(), y_indices.max() + 1
                x1, x2 = x_indices.min(), x_indices.max() + 1
                boxes.append([x1, y1, x2, y2])
                class_ids.append(1) # Default class 1
                
            if masks:
                final_masks = np.stack(masks, axis=0)
                final_boxes = np.array(boxes, dtype=np.float32)
                final_classes = np.array(class_ids, dtype=np.int64)
            else:
                final_masks = np.zeros((0, self.tile_size, self.tile_size), dtype=bool)
                final_boxes = np.zeros((0, 4), dtype=np.float32)
                final_classes = np.array([], dtype=np.int64)

        else:
             image_np = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
             final_masks = None
             final_boxes = np.zeros((0, 4), dtype=np.float32)
             final_classes = np.array([], dtype=np.int64)

        return TransformSample(
            image=image_np,
            boxes_xyxy=final_boxes,
            class_ids=final_classes,
            masks=final_masks,
        )

    def _load_patch(self, path: Path, tile: SatelliteTile, is_mask: bool = False) -> np.ndarray:
        if rasterio:
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(tile.x_off, tile.y_off, tile.width, tile.height)
                # rasterio reads (C, H, W)
                data = src.read(window=window)
                if is_mask:
                     # (1, H, W) -> (H, W)
                    return cast(np.ndarray, data[0])
                # (3, H, W) -> (H, W, 3)
                return cast(np.ndarray, np.moveaxis(data, 0, -1))
        else:
            with Image.open(path) as img:
                crop = img.crop((tile.x_off, tile.y_off, tile.x_off + tile.width, tile.y_off + tile.height))
                arr = np.array(crop)
                if is_mask and arr.ndim == 3:
                    return arr[:,:,0] # Take first channel of mask
                return arr

__all__ = ["SatelliteDataset", "SatelliteTile"]
