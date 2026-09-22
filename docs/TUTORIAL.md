# Tutorial: from counts to a defensible comparison

Run the commands below from the package directory after installing it.
The examples in this tutorial are synthetic and CPU-only. For the real paper
result, see [DeepScaleR 32k](../examples/deepscaler32k/README.md).

## 1. Run a complete example

```bash
python -m passk_inference --base examples/base.jsonl --rl examples/rl.jsonl \
  --bootstrap 1000 --seed 0 > comparison.json
```

Inspect the decision and the four estimated differences:

```python
import json

with open("comparison.json") as handle:
    result = json.load(handle)
print(result["ks"])
print(result["mean"])
print(result["certified_crossing"])
```

The budgets are `[1, 2, 3, 4]`; mean differences are approximately
`[0.09375, 0.02083, -0.0625, -0.125]`; certification is `False`.
Positive means RL is ahead. The simultaneous bands do not establish both
an early positive and a later negative difference, despite the point curve.

| Output | Meaning |
|---|---|
| `lower`, `upper` | Simultaneous bands at the listed budgets. |
| `certified_crossing` | Some earlier budget is certified positive and some later budget negative. |
| `first_loss_interval` | Conservative first-negative integer-budget interval on a dense grid; a null upper endpoint is unbounded. |
| `first_loss_scope` | Whether every integer from 1 through K was tested. |
| `degenerate_columns` | Zero-based column indices where the empirical bootstrap collapses; `compare` uses the safe support bounds [-1, 1]. |

`False` means insufficient evidence for the ordered two-sign claim. It does
not establish that the curves never cross. A first-loss interval can be
uninformative even when the sample curve crosses.

## 2. Prepare your own data

Each model file contains one JSON object per line with the same prompt IDs:

```json
{"id": "prompt-1", "c": 7, "n": 32}
{"id": "prompt-2", "c": 0, "n": 32}
```

`c` is the integer number of successful generations; `n` is the integer
number attempted. Require `0 <= c <= n` and `n >= 1`. At least two paired
prompts are needed for inference. Use enough prompts for the asymptotic
approximation to be credible; two is a validation minimum, not a design recommendation.

Duplicate or unmatched IDs fail explicitly. Establish the intended population
and exclusion rules before calling the tool. Canonical ID order makes a fixed
seed invariant to file row order. Generation failures must follow your declared
scoring protocol; the package cannot reconstruct missing attempts.

Use `--base your-base.jsonl --rl your-rl.jsonl` in the same command. To test
only selected budgets, add `--ks 1 2 4`. Every k must fit every prompt's budget.
A sparse grid provides no claim about the first loss at unobserved integers,
so its `first_loss_interval` is null.

## 3. Use Python for model-free inference

```python
from passk_inference import compare

base = [0, 1, 2, 3, 4, 2, 1, 0]
rl = [0, 2, 3, 4, 4, 3, 0, 0]
result = compare(base, 4, rl, 4, bootstrap=1000, seed=0)
assert result["certified_crossing"] is False
```

Array inputs are positional: you must align prompts yourself. Scalar 4 means
four generations for every prompt. Vector budgets are supported too.

## 4. Fit a response kernel only if prediction is your question

```python
from passk_inference import fit_kernel, kernel_curve

fit = fit_kernel(base, 4, rl, 4, Lp=20, Lq=20, iters=1000)
print("Converged:", fit["converged"])
print("Model-implied differences:", kernel_curve(fit, [1, 2, 4, 8]))
```

This small example illustrates the interface; it does not validate a fitted
kernel. Inspect convergence, grid sensitivity and held-out performance before
using predictions. k=8 is extrapolation beyond the four observed generations.
A kernel prediction is not a confidence certificate or a causal explanation
of how training changed capabilities.
