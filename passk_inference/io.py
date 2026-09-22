"""Strict paired count loading and provenance for the exact bytes analysed."""

import hashlib
import json
from pathlib import Path

from .core import _counts, compare


def _parse_counts(raw, source):
    rows = {}
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("each record must be a JSON object")
            key = row["id"]
            if isinstance(key, bool) or not isinstance(key, (str, int)) or key == "":
                raise ValueError("id must be a nonempty string or integer")
            if key in rows:
                raise ValueError(f"duplicate prompt id {key!r}")
            for field in ("c", "n"):
                if type(row[field]) is not int:
                    raise ValueError(f"{field} must be a JSON integer (not a boolean, string, or float)")
            _counts([row["c"]], [row["n"]])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{source}:{number}: {exc}") from exc
        rows[key] = row
    if not rows:
        raise ValueError(f"{source}: no count records")
    return rows


def load_counts(path):
    return _parse_counts(Path(path).read_bytes(), path)


def _align_pair(base, rl):
    if base.keys() != rl.keys():
        raise ValueError("prompt IDs differ; explicitly construct the intended paired population")
    # Canonical order makes a seeded bootstrap invariant to JSONL row order.
    ids = sorted(base, key=lambda key: (isinstance(key, str), key))
    return ([base[i]["c"] for i in ids], [base[i]["n"] for i in ids],
            [rl[i]["c"] for i in ids], [rl[i]["n"] for i in ids])


def load_pair(base_path, rl_path):
    return _align_pair(load_counts(base_path), load_counts(rl_path))


def compare_files(base_path, rl_path, *, ks=None, bootstrap=4000, alpha=0.05, seed=0):
    """Compare paired JSONL snapshots and hash the exact input bytes consumed.

    Only roles and hashes are recorded, not local paths or prompt text. Hashes
    change with byte-level formatting or row order even when estimates do not.
    """
    raw = {"base": Path(base_path).read_bytes(), "rl": Path(rl_path).read_bytes()}
    counts = _align_pair(_parse_counts(raw['base'], base_path), _parse_counts(raw['rl'], rl_path))
    result = compare(*counts, ks=ks, bootstrap=bootstrap, alpha=alpha, seed=seed)
    result['input_provenance'] = {
        'kind': 'jsonl_files',
        'hash_algorithm': 'sha256',
        'files': {role: {'sha256': hashlib.sha256(content).hexdigest()}
                  for role, content in raw.items()},
    }
    return result
