"""Strict JSONL count loading: alignment is by prompt ID, never file order."""

import json
from pathlib import Path

from .core import _counts


def load_counts(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
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
                raise ValueError(f"{path}:{number}: {exc}") from exc
            rows[key] = row
    if not rows:
        raise ValueError(f"{path}: no count records")
    return rows


def load_pair(base_path, rl_path):
    base, rl = load_counts(base_path), load_counts(rl_path)
    if base.keys() != rl.keys():
        raise ValueError("prompt IDs differ; explicitly construct the intended paired population")
    # Canonical order makes a seeded bootstrap invariant to JSONL row order.
    ids = sorted(base, key=lambda key: (isinstance(key, str), key))
    return ([base[i]["c"] for i in ids], [base[i]["n"] for i in ids],
            [rl[i]["c"] for i in ids], [rl[i]["n"] for i in ids])
