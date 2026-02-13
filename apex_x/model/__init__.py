from .cheap_block import CheapBlock
from .core import ApexXModel
from .det_head import DetHead, DetHeadOutput
from .ff_heavy_path import FFHeavyPath, FFHeavyPathOutput
from .ff_module import FFInferOutput, FFModule, FFTrainOutput
from .film import TileFiLM, apply_film
from .fpn import DualPathFPN, DualPathFPNOutput
from .fusion_gate import FusionGate
from .inst_seg_head import (
    FFTileRefinementHook,
    InstanceSegOutput,
    PrototypeInstanceSegHead,
    assemble_mask_logits_from_prototypes,
    rasterize_box_masks,
)
from .pv_backbone import PVBackbone
from .pv_coarse_heads import PVCoarseHeads, PVCoarseOutput
from .pv_module import PVModule, PVModuleOutput
from .pv_dinov2 import PVModuleDINOv2, LoRAAdapter, DINOV2_AVAILABLE
from .teacher import TeacherDistillOutput, TeacherModel, flatten_logits_for_distill
from .teacher_v3 import TeacherModelV3  # NEW: World-class v2.0 model
from .teacher_v5 import TeacherModelV5  # FLAGSHIP: Ascension V5
from .post_process import (
    compute_anchor_centers,
    decode_boxes_distance,
    decode_boxes_direct,
    post_process_detections,
    post_process_detections_per_class,
)
from .tile_refine_block import TileRefineBlock
from .timm_backbone import TimmBackboneAdapter
from .track_head import TrackEmbeddingHead, TrackEmbeddingOutput

__all__ = [
    "ApexXModel",
    "CheapBlock",
    "DetHead",
    "DetHeadOutput",
    "FFHeavyPath",
    "FFHeavyPathOutput",
    "FFModule",
    "FFTrainOutput",
    "FFInferOutput",
    "TileFiLM",
    "apply_film",
    "DualPathFPN",
    "DualPathFPNOutput",
    "FusionGate",
    "FFTileRefinementHook",
    "InstanceSegOutput",
    "PrototypeInstanceSegHead",
    "assemble_mask_logits_from_prototypes",
    "rasterize_box_masks",
    "PVBackbone",
    "PVCoarseHeads",
    "PVCoarseOutput",
    "PVModule",
    "PVModuleOutput",
    "PVModuleDINOv2",
    "LoRAAdapter",
    "DINOV2_AVAILABLE",
    "TileRefineBlock",
    "TeacherDistillOutput",
    "TeacherModel",
    "TeacherModelV3",
    "TeacherModelV5",
    "flatten_logits_for_distill",
    "TimmBackboneAdapter",
    "TrackEmbeddingHead",
    "TrackEmbeddingOutput",
    "compute_anchor_centers",
    "decode_boxes_distance",
    "decode_boxes_direct",
    "post_process_detections",
    "post_process_detections_per_class",
]

