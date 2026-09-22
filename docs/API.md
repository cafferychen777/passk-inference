# API and assumptions

Import supported functions from `passk_inference`.

| Function | Inputs and result |
|---|---|
| `pass_at_k(c, n, k)` | Success-count vector, scalar/vector generation budget, positive integer k; returns unbiased per-prompt estimates. |
| `compare(c_base, n_base, c_rl, n_rl, *, ks=None, bootstrap=4000, alpha=.05, seed=0)` | Aligned counts; returns a JSON-compatible report. Default ks is the dense grid through the smallest budget. |
| `simultaneous_band(rows, *, bootstrap=4000, alpha=.05, seed=0)` | Matrix of independent prompt contributions; returns means, SEs, bands, critical value and degenerate indices. |
| `certify(ks, lower, upper)` | Valid bands on a strictly increasing grid; returns ordered signs and dense-grid first-loss interval. The caller is responsible for the bands' validity. |
| `compare_files(base_path, rl_path, *, ks=None, bootstrap=4000, alpha=.05, seed=0)` | Strict JSONL comparison with hashes of the exact bytes analysed. |
| `write_report(result, output, *, base_label="Base", rl_label="RL")` | Render JSON, CSV and PNG from a result; requires the report extra. Returns a mapping of filenames to Paths. |
| `load_counts(path)` / `load_pair(base, rl)` | Strict JSONL; the pair loader returns the four aligned vectors accepted by compare. |
| `fit_kernel(c, n, d, nprime, *, Lp=80, Lq=80, iters=4000, tol=1e-9)` | Joint grid NPMLE; returns weights W, grids pg/qg, likelihood, iterations and convergence flag. |
| `kernel_curve(fit, ks)` | Model-implied RL-minus-base pass@k, including requested extrapolation. |

JSONL fields `c` and `n` must be JSON integers; booleans, numeric strings,
and floating-point literals are rejected before conversion. The numerical Python
API also accepts integer-valued floating-point arrays for NumPy workflows.

## What the guarantees depend on

The pass@k estimator is the probability of at least one success in a uniformly
chosen k-subset of the n generated answers. It is unbiased for model pass@k
under independent identically distributed generations from the declared protocol.

The multiplier band treats prompts as independent sampling units and preserves
across-budget dependence within each prompt. It is asymptotic, conditional on
the checkpoints. It does not cover training-seed uncertainty, dependent prompt
clusters, adaptive prompt selection or budgets omitted from the grid.

Zero empirical variance does not identify zero population variance. The generic
`simultaneous_band` returns diagnostic values on these columns; `compare`
replaces their bounds by the known [-1, 1] support before certification.

The kernel is a working distribution across prompts. Convergence is not proof
of model fit or unrestricted mixing-law identification. Its q=0 binomial endpoint
uses the historical clipped likelihood (1e-12). The reported log likelihood is
evaluated at the returned weights. Confidence intervals and held-out validation
are separate tasks, not implicit outputs of fitting.

### Additional comparison fields (0.2.0)

`base_pass_at_k` and `rl_pass_at_k` contain the prompt-average unbiased estimates
on `ks`. `mean` is RL minus base. `inconclusive_budgets` contains exactly those
budgets where the simultaneous band includes zero. It is not an equivalence
claim. `first_loss_interval` is returned only for the full integer prefix grid;
a null upper endpoint means a finite upper bound was not established.


## Results and provenance

`compare` and `compare_files` use the same statistical implementation and return
the same estimates, band, evidence classification and first-loss set. Each result
has `package_version` and `result_format_version` (currently `"1.0"`). The latter
versions the JSON structure independently of the software. Existing numerical
fields remain at the top level; consumers should ignore unknown additive fields.
A change to existing field meanings requires a new major result format version.

`compare_files` reads each file once, then parses and hashes that snapshot:

```json
{
  "input_provenance": {
    "kind": "jsonl_files",
    "hash_algorithm": "sha256",
    "files": {
      "base": {"sha256": "<64 hexadecimal characters>"},
      "rl": {"sha256": "<64 hexadecimal characters>"}
    }
  }
}
```

Hashes identify exact bytes, including whitespace and row order; they are not
claims about upstream model or benchmark authenticity. Reordering rows preserves
seeded statistical results but changes file hashes. Paths, filenames and raw
records are not copied into the report. Preserve the input files alongside the
result if someone else needs to rerun the comparison. Direct array calls use
`input_provenance: {"kind": "aligned_arrays"}` and do not claim file provenance.

```python
from passk_inference import compare_files, write_report

result = compare_files("base.jsonl", "rl.jsonl", bootstrap=4000, seed=0)
paths = write_report(result, "output/comparison", rl_label="Checkpoint B")
```

`write_report` renders the supplied result without resampling or changing the
band. JSON, CSV and PNG share the same arrays and sign rule. It supports sparse
grids (no integer first-loss set), unbounded first-loss sets and arbitrary alpha.
Sparse-grid figures show bounds only at the supplied budgets rather than shading
unsampled budgets. Display labels only affect the plot. Existing report files are replaced.
Matplotlib is imported only when rendering; library inference and JSON-only CLI
use require only the core dependencies. The `example` extra remains a compatible
alias for installing the plotting dependency.
