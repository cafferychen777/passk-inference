# Repository boundary and release procedure

The directory containing `pyproject.toml` is the entire public repository.
Do not initialize or push its parent research workspace. Maintain this package
in place; export a reviewed copy only for release or initial GitHub publication.

```text
passk-inference/
  passk_inference/           # Supported Python API and CLI
  examples/
    base.jsonl, rl.jsonl     # Synthetic teaching data
    deepscaler32k/           # Real counts, provenance, script, checked result, image
    coverage/               # Synthetic repeated-sampling experiment; outside CI
  docs/                     # API, tutorial, paper mapping, release policy
  tests/                    # Local and CI correctness checks
  scripts/export_release.py # Exact allowlist export and privacy checks
  scripts/check_sdist.py     # Reject unexpected, missing or altered sdist files
  .github/workflows/ci.yml   # Tests, real example, distributions, isolated wheel
  pyproject.toml
  LICENSE
  README.md
  CHANGELOG.md
  release-files.txt         # Reviewed source-release inventory
```

## Upload decisions

Include the package source, metadata, documentation, license, CI configuration,
reviewed example data, checked numerical reference, preview figure and export
script. The portable release archive includes tests so recipients can run them.

Exclude the parent workspace's `paper/`, `paper-arxiv/`, submission receipts,
anonymous supplements, review records, proposals, literature PDFs, history,
`pilot/`, `passk-train/`, raw `data_cache/`, cluster scripts and configs.
Only the explicitly sanitized pair of count files crosses this boundary.
Also exclude model weights, raw prompts/responses, caches, environments,
credentials, editor/agent state, logs, backups, generated output and local paths.
Do not copy the entire parent directory and hope ignore patterns catch it.

`.gitignore` prevents ordinary accidental additions, but cannot remove already
tracked files. `release-files.txt` is the exact allowlist used for the portable
source archive; adding a new public file requires editing that list. The export
rejects symlinks, missing paths, traversal and common private-path/token patterns.
It is a focused safeguard, not a guarantee that arbitrary files are safe.

The setuptools sdist is checked against the same allowlist with
`python scripts/check_sdist.py dist/passk_inference-0.3.0.tar.gz`. Only an explicit set of generated
packaging metadata files may be added. Missing, unexpected, duplicate, linked or
altered source files fail the check. CI runs this check after building; run it
before uploading any locally built sdist as well.

## Validate and publish a fixed version

1. Run `python -m pytest` and the real example. Inspect the generated figure.
2. Run `python -m build`, `python scripts/check_sdist.py dist/passk_inference-0.3.0.tar.gz`, and
   `python scripts/export_release.py`. The latter writes
   a deterministic versioned source archive and per-file SHA-256 manifest.
3. Test from the extracted archive outside the research workspace; install the
   wheel into a fresh environment and run the example there too.
4. The example is pinned to [arXiv:2609.22547v1](https://arxiv.org/abs/2609.22547v1).
   Keep this versioned mapping when linking the code from a CV or later release.
5. Before any commit, inspect `git status`, add individually reviewed files by
   exact path (never `git add .`), and inspect `git diff --cached`.
6. When publishing is authorized,
   create the remote, push the reviewed tree, and create annotated tag for the tested version (currently `v0.3.0`)
   on that tested commit. Attach the source archive and checksums to the release.
   Verify the public links and GitHub CI before using them in a CV.

Published versions are listed on the [GitHub releases page](https://github.com/cafferychen777/passk-inference/releases).
The original `v0.2.0` tag remains the initial paper release. Later releases retain
the paper-result regression and document software changes in the changelog.
