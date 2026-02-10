from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

AllowedTopLevelKey = Literal["images", "annotations", "categories", "info", "licenses"]
_ALLOWED_TOP_LEVEL_KEYS: set[AllowedTopLevelKey] = {
    "images",
    "annotations",
    "categories",
    "info",
    "licenses",
}
_REQUIRED_TOP_LEVEL_KEYS: set[str] = {"images", "annotations", "categories"}

_ALLOWED_IMAGE_KEYS: set[str] = {
    "id",
    "file_name",
    "width",
    "height",
    "license",
    "flickr_url",
    "coco_url",
    "date_captured",
}
_REQUIRED_IMAGE_KEYS: set[str] = {"id", "file_name", "width", "height"}

_ALLOWED_CATEGORY_KEYS: set[str] = {"id", "name", "supercategory"}
_REQUIRED_CATEGORY_KEYS: set[str] = {"id", "name"}

_ALLOWED_ANNOTATION_KEYS: set[str] = {
    "id",
    "image_id",
    "category_id",
    "bbox",
    "area",
    "iscrowd",
    "segmentation",
}
_REQUIRED_ANNOTATION_KEYS: set[str] = {"id", "image_id", "category_id", "bbox", "area", "iscrowd"}


@dataclass(frozen=True, slots=True)
class CocoBBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class CocoPolygon:
    points: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CocoRLE:
    size: tuple[int, int]  # (height, width)
    counts: tuple[int, ...]
    compressed: bool


@dataclass(frozen=True, slots=True)
class CocoSegmentation:
    kind: Literal["polygon", "rle"]
    polygons: tuple[CocoPolygon, ...] = ()
    rle: CocoRLE | None = None


@dataclass(frozen=True, slots=True)
class CocoCategory:
    category_id: int
    name: str
    supercategory: str | None = None


@dataclass(frozen=True, slots=True)
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CocoAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    bbox: CocoBBox
    area: float
    iscrowd: int
    segmentation: CocoSegmentation | None


@dataclass(frozen=True, slots=True)
class CocoCategoryMapping:
    original_to_contiguous: dict[int, int]
    contiguous_to_original: dict[int, int]
    names_by_original: dict[int, str]
    names_by_contiguous: dict[int, str]


@dataclass(slots=True)
class CocoDataset:
    images_by_id: dict[int, CocoImage]
    categories_by_id: dict[int, CocoCategory]
    annotations: tuple[CocoAnnotation, ...]
    annotations_by_image_id: dict[int, tuple[CocoAnnotation, ...]]
    _category_mapping_cache: CocoCategoryMapping | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def image_count(self) -> int:
        return len(self.images_by_id)

    @property
    def category_count(self) -> int:
        return len(self.categories_by_id)

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    def annotations_for_image(self, image_id: int) -> tuple[CocoAnnotation, ...]:
        return self.annotations_by_image_id.get(image_id, ())

    def category_mapping(self) -> CocoCategoryMapping:
        if self._category_mapping_cache is not None:
            return self._category_mapping_cache

        sorted_category_ids = sorted(self.categories_by_id)
        original_to_contiguous = {
            category_id: idx for idx, category_id in enumerate(sorted_category_ids)
        }
        contiguous_to_original = {
            idx: category_id for category_id, idx in original_to_contiguous.items()
        }
        names_by_original = {
            category_id: self.categories_by_id[category_id].name
            for category_id in sorted_category_ids
        }
        names_by_contiguous = {
            idx: names_by_original[category_id]
            for idx, category_id in contiguous_to_original.items()
        }
        mapping = CocoCategoryMapping(
            original_to_contiguous=original_to_contiguous,
            contiguous_to_original=contiguous_to_original,
            names_by_original=names_by_original,
            names_by_contiguous=names_by_contiguous,
        )
        self._category_mapping_cache = mapping
        return mapping


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value: object, *, field_name: str, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return int(value)


def _as_float(value: object, *, field_name: str, min_value: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    if min_value is not None and out < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return out


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value == "":
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _validate_record_keys(
    record: dict[str, object],
    *,
    required: set[str],
    allowed: set[str],
    record_name: str,
    strict: bool,
) -> None:
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{record_name} missing required keys: {missing}")
    if strict:
        unknown = sorted(set(record) - allowed)
        if unknown:
            raise ValueError(f"{record_name} contains unknown keys: {unknown}")


def _decode_compressed_rle_counts(encoded_counts: str) -> tuple[int, ...]:
    if encoded_counts == "":
        raise ValueError("compressed RLE counts must not be empty")

    counts: list[int] = []
    position = 0
    encoded = encoded_counts.encode("ascii")
    while position < len(encoded):
        value = 0
        shift = 0
        continuation = True
        sign_bit = 0
        while continuation:
            if position >= len(encoded):
                raise ValueError("invalid compressed RLE encoding")
            current = int(encoded[position]) - 48
            if current < 0:
                raise ValueError("compressed RLE contains invalid characters")
            position += 1
            continuation = (current & 0x20) != 0
            value |= (current & 0x1F) << (5 * shift)
            sign_bit = current & 0x10
            shift += 1

        if sign_bit != 0:
            value |= -1 << (5 * shift)
        if len(counts) > 1:
            value += counts[-2]
        if value < 0:
            raise ValueError("decoded RLE counts must be non-negative")
        counts.append(value)

    return tuple(int(v) for v in counts)


def _decode_rle_to_mask(rle: CocoRLE) -> np.ndarray:
    height, width = rle.size
    total = height * width
    flat = np.zeros((total,), dtype=np.uint8)
    write_pos = 0
    bit = 0
    for count in rle.counts:
        if count < 0:
            raise ValueError("RLE counts must be non-negative")
        if write_pos + count > total:
            raise ValueError("RLE counts exceed mask size")
        if bit == 1 and count > 0:
            flat[write_pos : write_pos + count] = 1
        write_pos += count
        bit = 1 - bit

    if write_pos != total:
        raise ValueError("RLE counts do not cover full mask size")
    return flat.reshape((height, width), order="F").astype(bool, copy=False)


def _rasterize_polygon(polygon: CocoPolygon, *, height: int, width: int) -> np.ndarray:
    points = np.asarray(polygon.points, dtype=np.float64)
    xs = points[0::2]
    ys = points[1::2]
    if xs.shape[0] < 3:
        raise ValueError("polygon must have at least 3 points")

    x_coords = np.arange(width, dtype=np.float64) + 0.5
    y_coords = np.arange(height, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    inside = np.zeros((height, width), dtype=bool)

    n = xs.shape[0]
    for idx in range(n):
        prev = (idx - 1) % n
        x1 = xs[idx]
        y1 = ys[idx]
        x2 = xs[prev]
        y2 = ys[prev]
        y_span = y2 - y1
        denom = y_span if abs(y_span) > 1e-12 else 1e-12
        crosses = ((y1 > grid_y) != (y2 > grid_y)) & (
            grid_x < ((x2 - x1) * (grid_y - y1) / denom + x1)
        )
        inside ^= crosses

    return inside


def segmentation_to_mask(
    segmentation: CocoSegmentation,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be > 0")

    if segmentation.kind == "polygon":
        mask = np.zeros((image_height, image_width), dtype=bool)
        for polygon in segmentation.polygons:
            mask |= _rasterize_polygon(polygon, height=image_height, width=image_width)
        return mask

    if segmentation.rle is None:
        raise ValueError("RLE segmentation must include rle payload")
    rle_height, rle_width = segmentation.rle.size
    if rle_height != image_height or rle_width != image_width:
        raise ValueError("RLE size must match image dimensions")
    return _decode_rle_to_mask(segmentation.rle)


def _parse_bbox(raw_bbox: object, *, field_name: str) -> CocoBBox:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError(f"{field_name} must be a list of length 4")
    x = _as_float(raw_bbox[0], field_name=f"{field_name}[0]")
    y = _as_float(raw_bbox[1], field_name=f"{field_name}[1]")
    width = _as_float(raw_bbox[2], field_name=f"{field_name}[2]", min_value=0.0)
    height = _as_float(raw_bbox[3], field_name=f"{field_name}[3]", min_value=0.0)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{field_name} width and height must be > 0")
    return CocoBBox(x=x, y=y, width=width, height=height)


def _parse_polygon_segmentation(raw: object, *, field_name: str) -> CocoSegmentation:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list for polygon segmentation")

    polygons: list[CocoPolygon] = []
    for poly_idx, polygon_raw in enumerate(raw):
        if not isinstance(polygon_raw, list):
            raise ValueError(f"{field_name}[{poly_idx}] must be a list")
        if len(polygon_raw) < 6 or len(polygon_raw) % 2 != 0:
            raise ValueError(f"{field_name}[{poly_idx}] must have even length >= 6")
        points: list[float] = []
        for coord_idx, coord in enumerate(polygon_raw):
            points.append(
                _as_float(coord, field_name=f"{field_name}[{poly_idx}][{coord_idx}]"),
            )
        polygons.append(CocoPolygon(points=tuple(points)))

    if not polygons:
        raise ValueError(f"{field_name} must contain at least one polygon")
    return CocoSegmentation(kind="polygon", polygons=tuple(polygons))


def _parse_rle_segmentation(
    raw: object,
    *,
    field_name: str,
    image_height: int,
    image_width: int,
) -> CocoSegmentation:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a dict for RLE segmentation")
    if "size" not in raw or "counts" not in raw:
        raise ValueError(f"{field_name} RLE must include size and counts")

    size_raw = raw["size"]
    if not isinstance(size_raw, list) or len(size_raw) != 2:
        raise ValueError(f"{field_name}.size must be [height, width]")
    rle_height = _as_int(size_raw[0], field_name=f"{field_name}.size[0]", min_value=1)
    rle_width = _as_int(size_raw[1], field_name=f"{field_name}.size[1]", min_value=1)
    if rle_height != image_height or rle_width != image_width:
        raise ValueError(f"{field_name}.size must match image dimensions")

    counts_raw = raw["counts"]
    compressed = isinstance(counts_raw, str)
    if compressed:
        counts = _decode_compressed_rle_counts(counts_raw)
    else:
        if not isinstance(counts_raw, list) or not counts_raw:
            raise ValueError(f"{field_name}.counts must be a non-empty list or string")
        counts_list: list[int] = []
        for idx, count in enumerate(counts_raw):
            parsed_count = _as_int(
                count,
                field_name=f"{field_name}.counts[{idx}]",
                min_value=0,
            )
            counts_list.append(parsed_count)
        counts = tuple(counts_list)

    rle = CocoRLE(size=(rle_height, rle_width), counts=counts, compressed=compressed)
    # Decode once during parse so invalid lengths/values fail at load-time.
    _decode_rle_to_mask(rle)
    return CocoSegmentation(kind="rle", rle=rle)


def _parse_segmentation(
    raw_segmentation: object,
    *,
    field_name: str,
    image_height: int,
    image_width: int,
    iscrowd: int,
) -> CocoSegmentation | None:
    if raw_segmentation is None:
        return None
    if isinstance(raw_segmentation, list):
        if iscrowd == 1:
            raise ValueError(f"{field_name} polygon segmentation is invalid when iscrowd=1")
        return _parse_polygon_segmentation(raw_segmentation, field_name=field_name)
    if isinstance(raw_segmentation, dict):
        return _parse_rle_segmentation(
            raw_segmentation,
            field_name=field_name,
            image_height=image_height,
            image_width=image_width,
        )
    raise ValueError(f"{field_name} must be polygon-list, RLE-dict, or null")


def _parse_category(record: dict[str, object], *, strict: bool) -> CocoCategory:
    _validate_record_keys(
        record,
        required=_REQUIRED_CATEGORY_KEYS,
        allowed=_ALLOWED_CATEGORY_KEYS,
        record_name="category",
        strict=strict,
    )
    category_id = _as_int(record["id"], field_name="category.id", min_value=1)
    name = _as_str(record["name"], field_name="category.name")
    supercategory_raw = record.get("supercategory")
    supercategory = (
        None
        if supercategory_raw is None
        else _as_str(supercategory_raw, field_name="category.supercategory")
    )
    return CocoCategory(category_id=category_id, name=name, supercategory=supercategory)


def _parse_image(record: dict[str, object], *, strict: bool) -> CocoImage:
    _validate_record_keys(
        record,
        required=_REQUIRED_IMAGE_KEYS,
        allowed=_ALLOWED_IMAGE_KEYS,
        record_name="image",
        strict=strict,
    )
    image_id = _as_int(record["id"], field_name="image.id", min_value=1)
    file_name = _as_str(record["file_name"], field_name="image.file_name")
    width = _as_int(record["width"], field_name="image.width", min_value=1)
    height = _as_int(record["height"], field_name="image.height", min_value=1)
    return CocoImage(image_id=image_id, file_name=file_name, width=width, height=height)


def _parse_annotation(
    record: dict[str, object],
    *,
    images_by_id: dict[int, CocoImage],
    categories_by_id: dict[int, CocoCategory],
    strict: bool,
) -> CocoAnnotation:
    _validate_record_keys(
        record,
        required=_REQUIRED_ANNOTATION_KEYS,
        allowed=_ALLOWED_ANNOTATION_KEYS,
        record_name="annotation",
        strict=strict,
    )
    annotation_id = _as_int(record["id"], field_name="annotation.id", min_value=1)
    image_id = _as_int(record["image_id"], field_name="annotation.image_id", min_value=1)
    category_id = _as_int(record["category_id"], field_name="annotation.category_id", min_value=1)
    if image_id not in images_by_id:
        raise ValueError(f"annotation.image_id={image_id} does not reference an existing image")
    if category_id not in categories_by_id:
        raise ValueError(f"annotation.category_id={category_id} does not reference a category")

    area = _as_float(record["area"], field_name="annotation.area", min_value=0.0)
    if area <= 0.0:
        raise ValueError("annotation.area must be > 0")
    iscrowd = _as_int(record["iscrowd"], field_name="annotation.iscrowd", min_value=0)
    if iscrowd not in (0, 1):
        raise ValueError("annotation.iscrowd must be 0 or 1")

    bbox = _parse_bbox(record["bbox"], field_name="annotation.bbox")
    image = images_by_id[image_id]
    segmentation = _parse_segmentation(
        record.get("segmentation"),
        field_name="annotation.segmentation",
        image_height=image.height,
        image_width=image.width,
        iscrowd=iscrowd,
    )
    return CocoAnnotation(
        annotation_id=annotation_id,
        image_id=image_id,
        category_id=category_id,
        bbox=bbox,
        area=area,
        iscrowd=iscrowd,
        segmentation=segmentation,
    )


def _load_coco_dataset_uncached(path: Path, *, strict: bool) -> CocoDataset:
    with path.open("r", encoding="utf-8") as f:
        payload_raw = json.load(f)
    if not isinstance(payload_raw, dict):
        raise ValueError("COCO JSON root must be an object")

    payload: dict[str, object] = payload_raw
    top_keys = set(payload)
    missing_top = sorted(_REQUIRED_TOP_LEVEL_KEYS - top_keys)
    if missing_top:
        raise ValueError(f"COCO JSON missing required top-level keys: {missing_top}")
    if strict:
        unknown_top = sorted(top_keys - _ALLOWED_TOP_LEVEL_KEYS)
        if unknown_top:
            raise ValueError(f"COCO JSON contains unknown top-level keys: {unknown_top}")

    images_raw = payload["images"]
    categories_raw = payload["categories"]
    annotations_raw = payload["annotations"]
    if not isinstance(images_raw, list):
        raise ValueError("COCO JSON 'images' must be a list")
    if not isinstance(categories_raw, list):
        raise ValueError("COCO JSON 'categories' must be a list")
    if not isinstance(annotations_raw, list):
        raise ValueError("COCO JSON 'annotations' must be a list")

    categories_by_id: dict[int, CocoCategory] = {}
    for idx, category_raw in enumerate(categories_raw):
        if not isinstance(category_raw, dict):
            raise ValueError(f"categories[{idx}] must be an object")
        category = _parse_category(category_raw, strict=strict)
        if category.category_id in categories_by_id:
            raise ValueError(f"duplicate category id: {category.category_id}")
        categories_by_id[category.category_id] = category

    images_by_id: dict[int, CocoImage] = {}
    for idx, image_raw in enumerate(images_raw):
        if not isinstance(image_raw, dict):
            raise ValueError(f"images[{idx}] must be an object")
        image = _parse_image(image_raw, strict=strict)
        if image.image_id in images_by_id:
            raise ValueError(f"duplicate image id: {image.image_id}")
        images_by_id[image.image_id] = image

    annotations_by_image: dict[int, list[CocoAnnotation]] = defaultdict(list)
    annotations: list[CocoAnnotation] = []
    annotation_ids: set[int] = set()
    for idx, annotation_raw in enumerate(annotations_raw):
        if not isinstance(annotation_raw, dict):
            raise ValueError(f"annotations[{idx}] must be an object")
        annotation = _parse_annotation(
            annotation_raw,
            images_by_id=images_by_id,
            categories_by_id=categories_by_id,
            strict=strict,
        )
        if annotation.annotation_id in annotation_ids:
            raise ValueError(f"duplicate annotation id: {annotation.annotation_id}")
        annotation_ids.add(annotation.annotation_id)
        annotations.append(annotation)
        annotations_by_image[annotation.image_id].append(annotation)

    annotations_by_image_id: dict[int, tuple[CocoAnnotation, ...]] = {
        image_id: tuple(annotations_by_image.get(image_id, [])) for image_id in images_by_id
    }

    return CocoDataset(
        images_by_id=images_by_id,
        categories_by_id=categories_by_id,
        annotations=tuple(annotations),
        annotations_by_image_id=annotations_by_image_id,
    )


@lru_cache(maxsize=8)
def _load_coco_dataset_cached(
    resolved_path: str,
    mtime_ns: int,
    file_size: int,
    strict: bool,
) -> CocoDataset:
    del mtime_ns, file_size  # cache-key only
    return _load_coco_dataset_uncached(Path(resolved_path), strict=strict)


def load_coco_dataset(
    path: str | Path,
    *,
    strict: bool = True,
    use_cache: bool = True,
) -> CocoDataset:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    if use_cache:
        return _load_coco_dataset_cached(
            str(resolved),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            bool(strict),
        )
    return _load_coco_dataset_uncached(resolved, strict=strict)


def clear_coco_dataset_cache() -> None:
    _load_coco_dataset_cached.cache_clear()


__all__ = [
    "CocoBBox",
    "CocoPolygon",
    "CocoRLE",
    "CocoSegmentation",
    "CocoCategory",
    "CocoImage",
    "CocoAnnotation",
    "CocoCategoryMapping",
    "CocoDataset",
    "segmentation_to_mask",
    "load_coco_dataset",
    "clear_coco_dataset_cache",
]
