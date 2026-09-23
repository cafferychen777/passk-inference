# Changelog

## 0.3.1 — 2026-09-23

- Publish on PyPI with GitHub OIDC Trusted Publishing, restricted to release tags.
- Gate releases on reusable CI, distribution inventory and metadata checks, then
  verify PyPI file hashes and installation before creating the GitHub release.
- Add Python 3.14 to CI and pin third-party Actions to immutable commits.
- Document direct PyPI installation and render README links on both registries.

## 0.3.0 — 2026-09-22

- Add reusable JSON/CSV/PNG reports through `--output` and `write_report`; the
  paper example now uses the same reporting implementation.
- Add `compare_files`, exact input-byte hashes, software version and a versioned
  JSON result contract; keep JSON-only CLI use free of plotting dependencies.
- Add a separate synthetic CPU example for whole-grid coverage, crossing power,
  false detection and Monte Carlo uncertainty.
- Preserve the v0.2.0 real-data numerical reference and the [11,61] paper result.

## 0.2.0 — 2026-09-22

- Add a real, CPU-only DeepScaleR 32k example that recomputes the paper's [11,61]
  dense-grid first-loss confidence set from 1,060 paired count records.
- Return base and RL pass@k curves and inconclusive budgets alongside the band.
- Add provenance hashes, a numerical reference, a rendered example, MIT license,
  standalone CI and a reviewed source-release allowlist.
- Preserve the strict ID alignment, count validation and sparse-grid limitations.

- Reject non-integer JSON count types before numerical conversion.
- Check setuptools source distributions against the reviewed release allowlist.
- Link the paper and example provenance to arXiv:2609.22547v1.
