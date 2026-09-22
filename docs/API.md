# API and assumptions

Import supported functions from `passk_inference`.

| Function | Inputs and result |
|---|---|
| `pass_at_k(c, n, k)` | Success-count vector, scalar/vector generation budget, positive integer k; returns unbiased per-prompt estimates. |
| `compare(c_base, n_base, c_rl, n_rl, *, ks=None, bootstrap=4000, alpha=.05, seed=0)` | Aligned counts; returns a JSON-compatible report. Default ks is the dense grid through the smallest budget. |
| `simultaneous_band(rows, *, bootstrap=4000, alpha=.05, seed=0)` | Matrix of independent prompt contributions; returns means, SEs, bands, critical value and degenerate indices. |
| `certify(ks, lower, upper)` | Valid bands on a strictly increasing grid; returns ordered signs and dense-grid first-loss interval. The caller is responsible for the bands' validity. |
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
