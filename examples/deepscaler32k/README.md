# DeepScaleR at a 32k-token limit: real paired counts

Run from the repository root:

```bash
python -m pip install '.[example]'
python examples/deepscaler32k/reproduce.py
```

Expected: 1,060 prompts; gains at k=1–10; losses at k=61–128;
inconclusive signs at k=11–60; first-loss confidence set [11,61].
The result describes the pooled population and these fixed checkpoints.

`base.jsonl` and `rl.jsonl` contain one row per prompt. `id` preserves the
pairing, `source` identifies the benchmark, `c` is the success count, and `n`
is the number of generations (128). Prompt IDs match the original count files.
There are no question texts, reference answers, generated responses, or weights.
These counts suffice for this analysis, not for rescoring or deduplication.

`metadata.json` records the settings, source and export hashes, transformation,
and manuscript hashes. Missing generation revisions are explicitly recorded as
unknown. `expected.json` is a checked numerical reference; `comparison.png` is a
preview generated from these counts. The script always recomputes its result.
See [paper mapping](../../docs/PAPER.md) for scope and provenance.

The example uses the public `compare_files` and `write_report` APIs. The same
report is available for other data through `passk-inference --output DIR`.
`expected.json` retains the v0.2.0 numerical reference; newly generated JSON
records the currently installed version, format version and input hashes.
