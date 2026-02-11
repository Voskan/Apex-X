# Changelog

All notable project changes are documented in this file.

## Unreleased (as of February 11, 2026)

### Added
- GPU PR gate wiring for GPU-critical paths in `.github/workflows/perf_gpu.yml`.
- Unified CPU/GPU perf trend artifacts (`--trend-output`) in:
  - `scripts/perf_regression.py`
  - `scripts/perf_regression_gpu.py`
- Weekly perf trend workflow:
  - `.github/workflows/perf_trend_weekly.yml`
- Release evidence attestation generator:
  - `scripts/release_attestation.py`
  - test coverage in `tests/test_release_attestation.py`

### Changed
- CI perf jobs now publish normalized trend artifacts and release evidence drafts.
- Release checklist docs now include automated evidence generation guidance.
- Performance docs updated with CPU/GPU trend and attestation outputs.

### Documentation
- Added migration guide:
  - `docs/release/MIGRATION.md`
- Updated context and TODO tracking for completed release-attestation wiring.
