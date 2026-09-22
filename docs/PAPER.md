# Paper result and provenance

The package accompanies [*RLVR is a Kernel, Not a Function: Statistical Inference
for pass@k Crossovers*](https://arxiv.org/abs/2609.22547v1), by Chen Yang,
Xianyang Zhang, and Jun Chen. This release maps to **arXiv:2609.22547v1**, submitted
on 2026-09-18: Table 1 (`tab:ladder`), the **32k fresh** row, and the appendix
paragraph **Post hoc grid densification**. The versioned link fixes the paper
version for this example; [the unversioned page](https://arxiv.org/abs/2609.22547)
will show later revisions if any. Source SHA-256 hashes are in
[`metadata.json`](../examples/deepscaler32k/metadata.json).

| Paper quantity | Reproduction |
|---|---|
| First-loss confidence set [11,61] | `python examples/deepscaler32k/reproduce.py` |
| Positive lower bound through k=10 | `result.json`: `certified_positive_budgets` |
| Negative upper bound from k=61 to 128 | `result.json`: `certified_negative_budgets` |
| Underlying curve and band | `curves.csv` and `comparison.png` |

The preview is a new rendering of this paper result, not a reproduction of the
paper's multi-experiment figure. The original sparse-grid dominance p=.0027
is not recomputed by this command. Densification is post hoc and must not be
presented as the pre-specified grid. The 95% simultaneous band is asymptotic,
over the declared 128 integer budgets, using 4,000 Gaussian prompt-multiplier
replicates and seed 0. The first-loss set is an uncertainty set, not an exact
crossing or a threshold for other tasks. There is no uncertainty over training
seeds here, and no claim about every individual prompt.

## Data and generation settings

Base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
RL: `agentica-org/DeepScaleR-1.5B-Preview`.
The realized sample contains OlympiadBench (633), Gaokao-2023-EN (358),
AMC23 (39), and AIME24 (30): 1,060 prompts, 128 generations per model per prompt.
The paper describes exact-normalized and token-Jaccard >= .7 deduplication
against training and other evaluation prompts. Counts alone cannot revalidate
that step. Decoding uses vLLM, temperature .6, top-p .95 and a 32,768-token
per-response limit; model chat templates plus a boxed-answer instruction;
math-verify symbolic equivalence with a five-second timeout scored as failure.

The available count artifacts do not pin immutable model revisions, generation
seeds or complete runtime versions. These remain unknown, rather than inferred
from today's upstream models. The supported reproduction starts from the frozen
counts; exact regeneration of model responses is not claimed. Counts are original
experimental measurements. Benchmark texts, answers, third-party code, raw model
responses and model weights are excluded from this repository and its license.

Export keeps only `id, source, c, n` from the original count records and sorts by
ID. It removes the truncation diagnostic, which is unused by this analysis.
Source and export checksums make this transformation traceable without exposing
local machine paths or cluster accounts.
