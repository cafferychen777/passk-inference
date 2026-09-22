"""Portable count-file comparison CLI."""

import argparse
import json

from . import __version__
from .core import compare
from .io import load_pair


def main(argv=None):
    parser = argparse.ArgumentParser(prog="passk-inference")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--base", required=True, help="base JSONL with id,c,n")
    parser.add_argument("--rl", required=True, help="RL JSONL with identical prompt IDs")
    parser.add_argument("--ks", type=int, nargs="+", help="default: every integer up to minimum budget")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = compare(*load_pair(args.base, args.rl), ks=args.ks,
                         bootstrap=args.bootstrap, alpha=args.alpha, seed=args.seed)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
