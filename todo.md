# Apex-X Active TODO (Open/Partial Only)

Last update: 2026-02-13
Owner: Apex-X engineering track

Status legend:
- `[ ]` not started
- `[~]` in progress / partial

Priority legend:
- `P0` critical blocker
- `P1` high impact
- `P2` medium impact
- `P3` research/optimization

---

## 1) Execution order (strict)

1. Close all `P0` tasks.
2. Close all `P1` tasks.
3. Close all `P2` tasks.
4. Close `P3` tasks with reproducible evidence.

No task is done without:
- code changes
- tests
- docs updates
- reproducible artifacts

---

## 2) Active queue by priority

| ID | Task | Priority | Status |
|---|---|---|---|
| T0.8 | Dependency hardening for DINOv2/V3 paths | P0 | [ ] |
| T0.9 | Remove silent synthetic fallback in train path | P0 | [ ] |
| T0.10 | Unified secure checkpoint loading (`weights_only`) | P0 | [ ] |
| T0.11 | Remove placeholder/no-op APIs from critical user paths | P0 | [ ] |
| T0.12 | README/TRAINING docs truth sync | P0 | [ ] |
| T0.13 | Lint/type debt closure (`ruff/black/mypy`) | P0 | [ ] |
| T0.14 | Notebook checkpoint/image robustness gate | P0 | [ ] |
| T0.15 | Loss explosion diagnostics and stabilization | P0 | [ ] |
| T1.1 | Unify real dataloaders (COCO + satellite + YOLO) | P1 | [~] |
| T1.4 | Integrate test split evaluation + final report artifact | P1 | [~] |
| T1.5 | Dataset contract validation (fail-fast preflight) | P1 | [ ] |
| T1.6 | Checkpoint manifest governance (best/last/ema/export) | P1 | [ ] |
| T5.1 | Close deployment-open tasks from `docs/TODO.md` | P1 | [ ] |
| T2.1 | Unified device policy | P2 | [ ] |
| T2.2 | Adaptive memory governor | P2 | [ ] |
| T2.3 | CPU reliability mode | P2 | [ ] |
| T2.4 | DDP/FSDP production path | P2 | [ ] |
| T2.5 | Runtime parity gates (cpu/torch/triton/trt) | P2 | [ ] |
| T3.1 | Replace proxy-only quality claims with strict protocol | P2 | [ ] |
| T3.2 | Benchmark fairness contract | P2 | [ ] |
| T3.3 | Reproducible AP benchmark harness + artifacts | P2 | [ ] |
| T3.4 | Statistical significance for ablations | P2 | [ ] |
| T6.3 | Warning hygiene in tests/export paths | P2 | [ ] |
| T4.1 | Establish strong public baselines in-repo | P3 | [ ] |
| T4.2 | Low-data + noisy-data strategy | P3 | [ ] |
| T4.3 | Loss/math ablation for segmentation quality | P3 | [ ] |
| T4.4 | Foundation-assisted pseudo-labeling (SAM2.1/SAM3 track) | P3 | [ ] |
| T4.5 | Backbone track (DINOv3 and alternatives) | P3 | [ ] |
| T4.6 | New architecture experiments (seg-first) | P3 | [ ] |

---

## 3) Detailed active tasks

### [ ] T0.8 Dependency hardening for DINOv2/V3 paths
Priority: `P0`

Problem:
- `TeacherModelV3` fails when optional deps are absent (`transformers`, `timm`, `peft`, `safetensors`).

Implementation:
1. Add optional extras profile `worldclass` in packaging.
2. Add explicit preflight checker for missing deps in CLI/notebook startup.
3. Keep baseline paths usable without V3 deps.

Files:
- `pyproject.toml`
- `requirements.txt`
- `apex_x/model/pv_dinov2.py`
- `apex_x/model/teacher_v3.py`
- `docs/TRAINING_GUIDE.md`

Done criteria:
- clean missing-dependency error with exact install command
- v3 path starts successfully when extras are installed

Validation:
- smoke with deps missing
- smoke with deps installed

---

### [ ] T0.9 Remove silent synthetic fallback in train path
Priority: `P0`

Problem:
- training can silently switch to synthetic data and produce misleading loss.

Implementation:
1. Add `train.allow_synthetic_fallback` (default `false`).
2. If dataset build fails and fallback is disabled: fail-fast.
3. Always write dataset mode (`real|synthetic`) into train report.

Files:
- `apex_x/config/schema.py`
- `apex_x/train/trainer.py`
- `configs/*.yaml`
- `tests/test_train_stages_smoke.py`

Done criteria:
- no silent fallback by default
- explicit warning + report flag if fallback is enabled

Validation:
- broken dataset path -> hard error
- fallback enabled -> controlled behavior

---

### [ ] T0.10 Unified secure checkpoint loading (`weights_only`)
Priority: `P0`

Problem:
- checkpoint loading is inconsistent and security warning appears in multiple paths.

Implementation:
1. Add shared helper for secure `torch.load` with `weights_only=True` when supported.
2. Unify checkpoint extraction (`model_state_dict`, `state_dict`, raw).
3. Replace direct `torch.load(...)` in CLI/scripts/train/notebooks.

Files:
- `apex_x/train/checkpoint.py`
- `apex_x/cli.py`
- `scripts/mine_hard_examples.py`
- `notebooks/checkpoint_image_inference.ipynb`

Done criteria:
- one canonical checkpoint load path everywhere
- common checkpoint formats load consistently

Validation:
- load raw state_dict
- load structured checkpoint
- load checkpoint with EMA branch

---

### [ ] T0.11 Remove placeholder/no-op APIs from critical user paths
Priority: `P0`

Problem:
- placeholder/no-op APIs still exposed in critical paths.

Implementation:
1. Remove placeholder exports from user-facing modules.
2. Replace with real implementation or remove from public API.
3. Update tests to use real runtime/train APIs.

Files:
- `apex_x/bench/__init__.py`
- `apex_x/train/__init__.py`
- `apex_x/losses/__init__.py`
- `apex_x/infer/__init__.py`
- `apex_x/cli.py`
- `tests/test_feature_toggle_smoke.py`
- `tests/test_routing_diagnostics.py`

Done criteria:
- no placeholder API in critical user surface

Validation:
- `rg -n "placeholder" apex_x` has no critical-path hits

---

### [ ] T0.12 README/TRAINING docs truth sync
Priority: `P0`

Problem:
- docs contain unsupported commands/APIs and unverified claims.

Implementation:
1. Remove unsupported API examples and missing file references.
2. Keep only runnable commands aligned with current CLI.
3. Mark benchmark claims as artifact-backed only.

Files:
- `README.md`
- `docs/TRAINING_GUIDE.md`
- `docs/index.md`
- `scripts/apex_finetune_pipeline.sh`

Done criteria:
- all documented commands are executable in current repo
- no unverified quality claims

Validation:
- command smoke checklist
- `mkdocs build --strict`

---

### [ ] T0.13 Lint/type debt closure (`ruff/black/mypy`)
Priority: `P0`

Problem:
- lint/type gates are not clean.

Implementation:
1. Fix import order, line length, typing issues, mutable defaults.
2. Keep consistent style across `apex_x`, `scripts`, `tests`.
3. Make local checks match CI checks.

Files:
- `apex_x/**`
- `scripts/**`
- `tests/**`

Done criteria:
- `ruff check .` pass
- `black --check .` pass
- `mypy` pass for configured targets

Validation:
- run CI-equivalent commands from `.github/workflows/ci.yml`

---

### [ ] T0.14 Notebook checkpoint/image robustness gate
Priority: `P0`

Problem:
- notebook must reliably support upload checkpoint + image and run inference.

Implementation:
1. Keep `FileUpload` compatibility for ipywidgets 7/8.
2. Add model-family auto-detection and num_classes inference.
3. Add strict/non-strict loading toggle + mismatch diagnostics.
4. Add dependency/device preflight cell.

Files:
- `notebooks/checkpoint_image_inference.ipynb`
- optional helper in `apex_x/infer/`

Done criteria:
- user can upload `.pt` and image and get result consistently

Validation:
- manual notebook smoke on CPU and CUDA

---

### [ ] T0.15 Loss explosion diagnostics and stabilization
Priority: `P0`

Problem:
- very large early losses are not fully explained by current logs.

Implementation:
1. Log detailed per-component loss and assignment stats.
2. Add gradient norm / NaN counters into train report.
3. Add warmup normalization knobs for unstable components.

Files:
- `apex_x/train/trainer.py`
- `apex_x/train/train_losses.py`
- `apex_x/losses/det_loss.py`

Done criteria:
- loss spikes are diagnosable from logs/report

Validation:
- train smoke with report assertions

---

### [~] T1.1 Unify real dataloaders (COCO + satellite + YOLO)
Priority: `P1`

Current state:
- partial integration exists; batch contracts still need full unification.

Implementation:
1. Standardize batch schema (`images/boxes/labels/masks/image_ids`).
2. Enforce dataset-specific collate and validation.
3. Align augmentation/data typing across backends.

Files:
- `apex_x/train/trainer.py`
- `apex_x/data/coco_dataset.py`
- `apex_x/data/satellite.py`
- `apex_x/data/yolo.py`

Done criteria:
- one trainer entrypoint works for all 3 backend types

Validation:
- backend-specific integration smoke tests

---

### [~] T1.4 Integrate test split evaluation + final report artifact
Priority: `P1`

Current state:
- val path exists; test-split lifecycle is incomplete.

Implementation:
1. Add explicit test dataloader builder.
2. Run held-out test after training completion.
3. Write consolidated `train_val_test_report.json/.md`.

Files:
- `apex_x/train/trainer.py`
- `apex_x/train/validation.py`
- `apex_x/infer/runner.py`

Done criteria:
- final report always includes train+val+test metrics

Validation:
- end-to-end tiny dataset run with report checks

---

### [ ] T1.5 Dataset contract validation (fail-fast preflight)
Priority: `P1`

Implementation:
1. Add preflight checks for paths/annotations/class bounds/mask validity.
2. Save preflight artifact before training.
3. Abort training on contract failure.

Files:
- `apex_x/data/**`
- `apex_x/cli.py`
- optional preflight script in `scripts/`

Done criteria:
- invalid dataset cannot silently enter training

Validation:
- malformed dataset negative tests

---

### [ ] T1.6 Checkpoint manifest governance (best/last/ema/export)
Priority: `P1`

Implementation:
1. Write manifest per run (hash, metric, config, model family).
2. Explicitly track `best`, `last`, `ema_best`, `ema_last`.
3. Integrate manifest usage into export/infer flows.

Files:
- `apex_x/train/checkpoint.py`
- `apex_x/train/trainer.py`
- `apex_x/export/**`
- `apex_x/cli.py`

Done criteria:
- checkpoint lineage is fully traceable

Validation:
- manifest consistency unit/integration tests

---

### [ ] T5.1 Close deployment-open tasks from `docs/TODO.md`
Priority: `P1`

Open closures required:
1. TRT end-to-end parity closure on production engine.
2. FP8 evidence on `sm90+` hardware.
3. Mandatory GPU PR gate in protected branch.
4. Unified perf policy validation on deployment runners.
5. Weekly GPU trend artifacts.
6. Go TRT bridge validation on deployment engine.

Files:
- `docs/TODO.md`
- `.github/workflows/perf_gpu.yml`
- `.github/workflows/perf_trend_weekly.yml`
- `runtime/go/**`

Done criteria:
- open deployment blockers become evidence-backed done items

Validation:
- attach command logs + artifacts for each closure

---

### [ ] T2.1 Unified device policy
Priority: `P2`

Implementation:
1. Single resolver for `cpu|cuda|ddp` used by all stages.
2. Remove divergent per-module device assumptions.

Files:
- `apex_x/train/trainer.py`
- `apex_x/train/cpu_support.py`
- `apex_x/train/ddp.py`

Done criteria:
- one source of truth for device selection

---

### [ ] T2.2 Adaptive memory governor
Priority: `P2`

Implementation:
1. auto-batch search + runtime downshift on OOM.
2. gradient accumulation fallback.
3. activation checkpoint policy.
4. memory telemetry in training reports.

Files:
- `apex_x/train/memory_manager.py`
- `apex_x/train/trainer.py`
- `apex_x/train/gradient_checkpoint.py`

Done criteria:
- stable OOM recovery with telemetry

---

### [ ] T2.3 CPU reliability mode
Priority: `P2`

Implementation:
1. force CPU-safe settings (no AMP/CUDA assumptions).
2. deterministic and reproducibility guardrails.
3. add CPU train smoke gate in CI.

Files:
- `apex_x/train/cpu_support.py`
- `apex_x/train/trainer.py`
- `.github/workflows/ci.yml`

Done criteria:
- CPU train path is stable and reproducible

---

### [ ] T2.4 DDP/FSDP production path
Priority: `P2`

Implementation:
1. integrate DDP launch in canonical train flow.
2. add optional FSDP profile for large models.
3. ensure checkpoint compatibility with single-GPU inference.

Files:
- `apex_x/train/ddp.py`
- `apex_x/train/trainer.py`
- `docs/TRAINING_GUIDE.md`

Done criteria:
- multi-GPU training path documented and smoke-tested

---

### [ ] T2.5 Runtime parity gates (cpu/torch/triton/trt)
Priority: `P2`

Implementation:
1. backend pairwise parity matrix with tolerances.
2. CI artifact for parity summary.
3. fail gate on parity regressions.

Files:
- `apex_x/runtime/parity.py`
- `tests/test_runtime_parity.py`
- `.github/workflows/perf_gpu.yml`

Done criteria:
- backend parity is measurable and enforced

---

### [ ] T3.1 Replace proxy-only quality claims with strict protocol
Priority: `P2`

Implementation:
1. canonical metrics: bbox/segm AP/AP50/AP75.
2. report p50/p95 latency, throughput, peak memory.
3. ensure label-based evaluation path is default for claims.

Files:
- `apex_x/infer/runner.py`
- `apex_x/eval/coco_evaluator.py`
- `scripts/sota_benchmark.py`

Done criteria:
- no claim without strict metric artifacts

---

### [ ] T3.2 Benchmark fairness contract
Priority: `P2`

Implementation:
1. enforce same split/hardware/input/postproc across compared models.
2. store benchmark protocol manifest.

Files:
- `scripts/sota_benchmark.py`
- `docs/benchmarks.md`

Done criteria:
- benchmark rows are protocol-comparable

---

### [ ] T3.3 Reproducible AP benchmark harness + artifacts
Priority: `P2`

Implementation:
1. add canonical benchmark command for labeled datasets.
2. export JSON/MD + config + commit hash artifacts.
3. support baseline comparison mode.

Files:
- `scripts/sota_benchmark.py`
- `apex_x/infer/eval_metrics.py`

Done criteria:
- one command creates complete benchmark bundle

---

### [ ] T3.4 Statistical significance for ablations
Priority: `P2`

Implementation:
1. run >=3 seeds per ablation.
2. report mean/std/CI and acceptance threshold.

Files:
- `apex_x/train/ablation.py`
- reporting scripts

Done criteria:
- gains are accepted only with statistical evidence

---

### [ ] T6.3 Warning hygiene in tests/export paths
Priority: `P2`

Implementation:
1. register custom pytest marks (e.g., `slow`).
2. reduce ONNX export warning noise (`dynamic_shapes` migration where possible).
3. fix warning-prone tests (return-not-none etc.).

Files:
- `pyproject.toml`
- `tests/**`
- `apex_x/export/pipeline.py`

Done criteria:
- warning volume controlled and intentional

---

### [ ] T4.1 Establish strong public baselines in-repo
Priority: `P3`

Implementation:
1. reproducible baseline configs (YOLO-seg + DETR-like baseline track).
2. same protocol as Apex-X comparisons.

Files:
- `configs/**`
- `scripts/sota_benchmark.py`
- docs

Done criteria:
- baseline training/eval is reproducible in-repo

---

### [ ] T4.2 Low-data + noisy-data strategy
Priority: `P3`

Implementation:
1. pseudo-label filtering by confidence/quality.
2. hard-example mining and curriculum schedule.
3. remove placeholder step from fine-tune pipeline.

Files:
- `apex_x/train/pseudo_label.py`
- `apex_x/data/miner.py`
- `scripts/apex_finetune_pipeline.sh`

Done criteria:
- measured gain in low-data/noisy-label setups

---

### [ ] T4.3 Loss/math ablation for segmentation quality
Priority: `P3`

Implementation:
1. ablate BCE/Dice/Lovasz/boundary/quality weights.
2. ablate IoU-family loss variants.
3. keep only statistically validated improvements.

Files:
- `apex_x/train/train_losses.py`
- `apex_x/train/train_losses_v3.py`
- `apex_x/losses/**`

Done criteria:
- evidence-backed final loss configuration

---

### [ ] T4.4 Foundation-assisted pseudo-labeling (SAM2.1/SAM3 track)
Priority: `P3`

Implementation:
1. optional foundation-teacher pseudo-label adapter.
2. quality/boundary filtering for generated masks.
3. integrate into semi-supervised train loop.

Files:
- `apex_x/train/foundation_teacher.py` (new)
- `apex_x/train/pseudo_label.py`

Done criteria:
- reproducible gain in sparse-label regime

---

### [ ] T4.5 Backbone track (DINOv3 and alternatives)
Priority: `P3`

Implementation:
1. pluggable backbone registry.
2. DINOv3 profile integration (if compatible).
3. fair benchmark against current backbone.

Files:
- `apex_x/model/**`
- `configs/worldclass.yaml`

Done criteria:
- backbone switch is config-driven and benchmarked fairly

---

### [ ] T4.6 New architecture experiments (seg-first)
Priority: `P3`

Implementation:
1. experiment branch for new seg-first head/matcher.
2. strict ablation/rollback criteria.
3. promote only with consistent multi-seed gain.

Files:
- `apex_x/model/**`
- `apex_x/train/**`

Done criteria:
- architecture upgrades are evidence-backed

---

## 4) First implementation batch

Batch A (`P0`):
1. `T0.8`
2. `T0.9`
3. `T0.10`
4. `T0.11`
5. `T0.12`
6. `T0.13`
7. `T0.14`
8. `T0.15`

Batch B (`P1`):
1. `T1.1`
2. `T1.4`
3. `T1.5`
4. `T1.6`
5. `T5.1`

Batch C (`P2/P3`):
- `T2.*`, `T3.*`, `T4.*`, `T6.3`
