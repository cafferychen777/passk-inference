# Changelog

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
