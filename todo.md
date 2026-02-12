# Apex-X Master TODO (Execution Plan)

Last update: 2026-02-12
Owner: Apex-X engineering track
Status legend:
- `[ ]` not started
- `[~]` in progress / partial
- `[x]` done

## 0) Scope and quality bar

This file is the practical execution plan from current repo state to:
- stable training and inference on CPU and GPU
- reproducible DET/INST-SEG evaluation
- reliable checkpoint/export/runtime lifecycle
- measurable improvement track toward SOTA under fair benchmarking

Important realism:
- "100% ideal segmentation" on real noisy data is not physically guaranteed.
- Target is: maximize quality with strict reproducibility and verified evidence.

## 1) P0 Critical blockers (must be fixed first)

### [x] T0.1 Restore `compute_v3_training_losses` API and module structure
Problem:
- `apex_x/train/train_losses_v3.py` exports `compute_v3_training_losses` but function is missing.
- `pytest` fails at collection with ImportError.

Reshenie (kak dolzhno byt):
- Add explicit `def compute_v3_training_losses(...)` with stable contract:
  - finite-safe sanitization for logits/boxes/masks
  - returns `(total_loss: Tensor, loss_dict: dict[str, Tensor])`
  - backward-compatible keys for tests
- Ensure helper `_finite_or_zero` is small and not swallowing full file body.

Files:
- `apex_x/train/train_losses_v3.py`
- `tests/test_integration_v3.py`
- `tests/test_v3_nan_stability.py`

Definition of done:
- import works
- `pytest` collection succeeds
- v3 tests pass

Validation:
- `pytest -q tests/test_integration_v3.py tests/test_v3_nan_stability.py`
- `pytest -q`

---

### [x] T0.2 Fix dataset->transform mask dtype crash in real training
Problem:
- Training with `dataset_path` can crash in OpenCV/albumentations path due to bool masks.

Reshenie (kak dolzhno byt):
- Normalize mask dtype before augmentation and after augmentation consistently.
- Keep internal semantic contract explicit:
  - augment pipeline input mask dtype: `uint8`/`float32`
  - model/loss path mask dtype: binary float/bool as required
- Add regression test with tiny synthetic satellite dataset.

Files:
- `apex_x/data/satellite.py`
- `apex_x/data/transforms.py`
- new test in `tests/`

Definition of done:
- one-step train run with `dataset_path` completes without dtype errors.

Validation:
- `pytest -q tests/test_train_stages_smoke.py tests/test_trainer_stages.py`
- local tiny dataset smoke command (documented in test/notes)

---

### [x] T0.3 Fix `MemoryManager.catch_oom` context manager
Problem:
- current implementation yields twice on exception path and can raise `generator didn't stop`.

Reshenie (kak dolzhno byt):
- Refactor API to robust OOM handling, e.g.:
  - either return context that never double-yields
  - or use helper function wrapper around step execution
- Keep behavior deterministic:
  - clear cache
  - log reason code
  - allow caller to skip current step

Files:
- `apex_x/train/memory_manager.py`
- `apex_x/train/trainer.py` (usage path)
- tests: `tests/verify_sota_upgrade.py` + new focused unit test

Definition of done:
- OOM-like error no longer crashes with generator runtime error.

Validation:
- `pytest -q tests/verify_sota_upgrade.py`
- targeted unit test for OOM-like exception

---

### [x] T0.4 Fix Stage-1 training loop correctness
Problem:
- `batch_images` usage before initialization in stage1 path.
- high risk of hidden runtime bugs in dataset-backed branch.

Reshenie (kak dolzhno byt):
- initialize `batch_images` per step before append
- ensure `samples` lifecycle safe for both iterator and random path
- ensure image tensors moved to model device consistently in both branches
- remove placeholder comment fragments and keep explicit code path

Files:
- `apex_x/train/trainer.py`

Definition of done:
- stage1 runs with dataset and without dataset on CPU/GPU.

Validation:
- `pytest -q tests/test_trainer_stages.py tests/test_train_stages_smoke.py`

---

### [x] T0.5 Fix export checkpoint loading contract
Problem:
- `export_cmd --checkpoint` tries to load full checkpoint dict as pure model state_dict.

Reshenie (kak dolzhno byt):
- support both formats:
  - raw state_dict file
  - structured checkpoint with `model_state_dict`
- optional strict/non-strict mode with explicit log of missing/unexpected keys

Files:
- `apex_x/cli.py`
- `apex_x/train/checkpoint.py` (if shared helper needed)
- tests: `tests/test_export.py` (extend)

Definition of done:
- export works from `best.pt` and from legacy pure state_dict artifacts.

Validation:
- `pytest -q tests/test_export.py tests/test_tensorrt_export_manifest.py`

---

### [x] T0.6 Fix checkpoint metadata compatibility wrapper
Problem:
- `trainer_utils.load_checkpoint` expects `metadata.to_dict()` but dataclass has no method.

Reshenie (kak dolzhno byt):
- convert metadata via `dataclasses.asdict(metadata)` or add canonical serializer in metadata class.

Files:
- `apex_x/train/trainer_utils.py`
- optionally `apex_x/train/checkpoint.py`

Definition of done:
- wrapper returns real metadata payload.

Validation:
- add/update unit test for wrapper return structure

---

### [x] T0.7 P0 stabilization gate
Problem:
- no formal gate after critical fixes.

Reshenie (kak dolzhno byt):
- add mini gate checklist after P0:
  - import smoke
  - trainer smoke
  - export smoke
  - v3 loss tests

Definition of done:
- all P0 gate checks green.

Validation:
- `pytest -q tests/test_import_smoke.py tests/test_trainer_stages.py tests/test_train_stages_smoke.py tests/test_export.py tests/test_integration_v3.py tests/test_v3_nan_stability.py`

---

## 2) P1 Training system completion (train/val/test as real pipeline)

### [~] T1.1 Unify real dataloaders (COCO + satellite)
Problem:
- CLI/docs mention COCO, but trainer path currently hard-wired mainly to satellite dataset flow.

Reshenie (kak dolzhno byt):
- configurable dataset backend:
  - `train.dataset_type: coco|satellite|yolo`
  - explicit annotation/image paths from config
- proper collate functions per dataset

Files:
- `apex_x/train/trainer.py`
- `apex_x/config/schema.py`
- `configs/*.yaml`

Definition of done:
- same trainer entrypoint can run at least COCO and satellite.

Validation:
- dataset-specific smoke tests + integration CLI tests

---

### [x] T1.2 Epoch-based lifecycle and checkpoint policy
Problem:
- `current_epoch` not progressing meaningfully in staged run; `epoch_0000` only.

Reshenie (kak dolzhno byt):
- explicit epoch loop with:
  - train epoch
  - validate by interval
  - checkpoint every `save_interval`
  - `last.pt` always latest, `best.pt` by selected validation metric

Files:
- `apex_x/train/trainer.py`
- `apex_x/train/checkpoint.py`

Definition of done:
- checkpoint folder contains proper epoch sequence + best/last semantics.

Validation:
- checkpoint lifecycle tests

---

### [x] T1.3 Proper val metric selection for segmentation
Problem:
- current best tracking uses `loss_proxy`, not final segmentation quality target.

Reshenie (kak dolzhno byt):
- configurable primary metric:
  - default: `mAP_segm` for INST-SEG track
  - fallback hierarchy if metric unavailable
- early stopping on same target metric

Files:
- `apex_x/train/trainer.py`
- `apex_x/train/early_stopping.py`

Definition of done:
- best checkpoint selection tied to validation quality metric.

Validation:
- tests for metric-based best selection

---

### [~] T1.4 Integrate test split evaluation and final report artifact
Reshenie (kak dolzhno byt):
- after training completion:
  - evaluate on held-out test split
  - write structured report JSON/MD with key quality metrics

Files:
- `apex_x/train/trainer.py`
- `apex_x/infer/runner.py` (shared reporting blocks)

Definition of done:
- reproducible final train/val/test artifact package.

---

## 3) P2 CPU/GPU execution and memory orchestration

### [ ] T2.1 Unified device policy
Reshenie (kak dolzhno byt):
- single deterministic device resolver:
  - cpu
  - cuda single-gpu
  - optional ddp launcher path
- all train stages must use same source of truth for device.

Files:
- `apex_x/train/trainer.py`
- `apex_x/train/cpu_support.py`
- `apex_x/train/ddp.py`

---

### [ ] T2.2 Adaptive memory governor
Reshenie (kak dolzhno byt):
- automatic batch-size search + runtime adjustment on OOM
- gradient accumulation fallback
- optional activation checkpointing for heavy blocks
- telemetry:
  - allocated/reserved/max memory
  - chosen batch size
  - OOM recovery count

Files:
- `apex_x/train/memory_manager.py`
- `apex_x/train/trainer.py`
- `apex_x/train/gradient_checkpoint.py`

---

### [ ] T2.3 CPU reliability mode
Reshenie (kak dolzhno byt):
- clean CPU path:
  - AMP disabled
  - deterministic knobs valid
  - no CUDA-only assumptions

Validation:
- dedicated CPU training smoke in CI-compatible test.

---

## 4) P3 Metrics, benchmarking, and fair comparisons

### [ ] T3.1 Replace proxy-only quality claims with strict evaluation protocol
Reshenie (kak dolzhno byt):
- canonical metrics:
  - COCO AP/AP50/AP75 for bbox+segm
  - PQ for panoptic when enabled
- report latency p50/p95, throughput, peak memory

Files:
- `apex_x/infer/runner.py`
- `scripts/sota_benchmark.py` (upgrade to label-based AP mode)
- docs updates

---

### [ ] T3.2 Benchmark fairness contract
Reshenie (kak dolzhno byt):
- all compared models use:
  - same dataset split
  - same image size policy
  - same hardware
  - same postproc constraints
- produce artifact bundle for every claim.

---

## 5) P4 SOTA research track (segmentation-first)

### [ ] T4.1 Establish strong public baselines in-repo
Reshenie (kak dolzhno byt):
- reproducible baselines:
  - YOLO-seg baseline
  - MaskDINO/DETR-like baseline (if feasible in this repo setup)
- training/eval scripts and fixed configs

---

### [ ] T4.2 Low-data and noisy-data strategy
Reshenie (kak dolzhno byt):
- semi-supervised pipeline:
  - pseudo-label filtering by confidence+quality
  - active hard-example mining
  - curriculum for noisy labels

Files:
- `apex_x/train/pseudo_label.py`
- `apex_x/data/miner.py`
- `scripts/apex_finetune_pipeline.sh` (remove step2 placeholder)

---

### [ ] T4.3 Loss/math ablation for segmentation quality
Reshenie (kak dolzhno byt):
- run strict ablations for:
  - BCE/Dice/Lovasz/boundary/quality/multiscale weights
  - box loss variants (IoU/GIoU/DIoU/CIoU/MPDIoU)
  - router/budget effects on quality-latency frontier
- keep only statistically validated gains.

---

## 6) P5 Runtime/deploy closure (existing backlog alignment)

### [ ] T5.1 Close open TODO phases in `docs/TODO.md`
Target closures:
- `P3-05` TRT E2E parity harness on production engine
- `P4-07` FP8 measurable value on sm90+
- `P6-01` mandatory GPU PR gate enforced in branch protection
- `P6-02` unified perf policy fully validated on deployment runners

Reshenie (kak dolzhno byt):
- close each item only with artifact evidence and reproducible command logs.

---

## 7) P6 Cleanup and documentation truthfulness

### [ ] T6.1 Remove or replace placeholder APIs from critical user paths
Targets:
- `bench_placeholder`
- `train_step_placeholder`
- `infer_placeholder` alias naming debt
- `loss_placeholder` export debt

Reshenie (kak dolzhno byt):
- no critical command should call no-op placeholder.

---

### [ ] T6.2 Sync README and training docs with real implementation
Reshenie (kak dolzhno byt):
- remove unsupported claims/AP numbers without evidence
- remove non-existing API examples
- update all commands to current canonical CLI

Files:
- `README.md`
- `docs/TRAINING_GUIDE.md`
- `docs/index.md`

---

## 8) Execution order

1. P0 (T0.1 -> T0.7)
2. P1 (T1.1 -> T1.4)
3. P2 (T2.1 -> T2.3)
4. P3 (T3.1 -> T3.2)
5. P4 (T4.1 -> T4.3)
6. P5 (T5.1)
7. P6 (T6.1 -> T6.2)

## 9) Start criteria for implementation

Implementation starts only after:
- this file approved
- priority confirmed: default start from `T0.1`

Recommended first implementation batch:
1. `T0.1` v3 losses restore
2. `T0.3` OOM context fix
3. `T0.4` stage1 loop fix
4. `T0.5` export checkpoint load fix
