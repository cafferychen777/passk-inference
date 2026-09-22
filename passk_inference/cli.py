"""Portable count-file comparison CLI."""

import argparse
import json

from . import __version__
from .io import compare_files
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="passk-inference")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--base", required=True, help="base JSONL with id,c,n")
    parser.add_argument("--rl", required=True, help="RL JSONL with identical prompt IDs")
    parser.add_argument("--ks", type=int, nargs="+", help="default: every integer up to minimum budget")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="directory for result.json, curves.csv and comparison.png")
    parser.add_argument("--base-label", default="Base", help="base model display name in the figure")
    parser.add_argument("--rl-label", default="RL", help="RL model display name in the figure")
    args = parser.parse_args(argv)
    try:
        result = compare_files(args.base, args.rl, ks=args.ks,
                               bootstrap=args.bootstrap, alpha=args.alpha, seed=args.seed)
        if args.output:
            write_report(result, args.output, base_label=args.base_label, rl_label=args.rl_label)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
