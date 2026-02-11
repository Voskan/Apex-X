# Apex-X Evaluation Report

| Metric | Value |
|---|---:|
| COCO mAP (det) | 1.000000 |
| AP50 (det) | 1.000000 |
| AP75 (det) | 1.000000 |
| COCO mAP (inst-seg) | 1.000000 |
| AP50 (inst-seg) | 1.000000 |
| AP75 (inst-seg) | 1.000000 |
| mIoU (semantic) | 1.000000 |
| PQ (panoptic) | 1.000000 |

Panoptic source: `fallback`


## Model Dataset Eval

- source: `model_dataset_npz`
- samples: `3`
- det_score: `mean=0.998487`, `std=0.002139`, `min=0.995462`, `max=1.000000`
- selected_tiles: `mean=4.000`, `p95=4.000`
- det_score_target: `mae=0.698487`, `rmse=0.703032`, `bias=0.698487`, `r2=-73.138010`, `pearson_corr=0.866025`
- selected_tiles_target: `mae=2.333333`, `rmse=2.380476`, `bias=2.333333`, `exact_match_rate=0.000000`
- execution_backend: `cpu`
- precision_profile: `balanced`
