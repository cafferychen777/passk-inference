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
  scripts/verify_pypi.py     # Published distribution identity check
  .github/workflows/        # Shared CI and gated PyPI/GitHub releases
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
`python scripts/check_sdist.py dist/*.tar.gz`. Only an explicit set of generated
packaging metadata files may be added. Missing, unexpected, duplicate, linked or
altered source files fail the check. CI runs this check after building; run it
before uploading any locally built sdist as well.

## Automatic PyPI and GitHub releases

The `release.yml` workflow runs on `v*` tag pushes. It calls the same CI workflow
used for pull requests and main, covering Python 3.10, 3.12, 3.13 and 3.14. Only
matching stable `vMAJOR.MINOR.PATCH` tags proceed to publication.

1. Update `passk_inference/_version.py` and the changelog. Review the allowlist.
2. Run the tests and paper example, then build, check the sdist inventory, and
   run `python -m twine check --strict dist/*.whl dist/*.tar.gz` in a clean build.
3. Commit the reviewed changes and push main. Do not include generated output,
   environments or credentials. Inspect `git status` and `git diff --cached`.
4. Once main CI passes, create an annotated tag matching the package version and
   push that tag. No local upload command or manual GitHub release is required.
5. The release workflow tests, builds once, checks the distributions and stores
   their hashes. A separate `pypi` environment job uploads that same wheel and
   sdist using PyPI OIDC Trusted Publishing, without a stored API token.
6. Post-publication checks compare PyPI SHA-256 hashes with the built artifacts,
   install the pinned public version in a fresh environment, and reproduce
   [11,61]. Only then does the workflow create the GitHub release and attach the
   same distributions, reviewed source ZIP and `SHA256SUMS.txt`.

The PyPI publisher is limited to owner `cafferychen777`, repository
`passk-inference`, workflow `release.yml`, environment `pypi`. GitHub restricts
that environment to `v*` tags. Pull requests and ordinary branch pushes cannot
publish. Third-party Actions are pinned to immutable commits. Only the publishing
job receives `id-token: write`; only the GitHub release job can write releases.

Failed runs can be retried on the same tag. Existing PyPI files are skipped but
must still pass exact hash verification. If rebuilding produces different bytes,
reuse the original build artifact and rerun only the failed jobs; do not replace
an existing version or move a published tag. A new build needs a new version.

The `example` and `report` extras install plotting dependencies. The PyPI wheel
contains the library and CLI; example scripts and frozen counts are in the
repository and source release. Historical tags, including v0.2.0 and v0.3.0,
remain unchanged. The first PyPI version is v0.3.1.
