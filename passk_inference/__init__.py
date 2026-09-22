"""Paired pass@k inference and response-kernel working models."""

from ._version import __version__
from .report import write_report
from .core import certify, compare, pass_at_k, simultaneous_band
from .io import compare_files, load_counts, load_pair
from .kernel import fit_kernel, kernel_curve

__all__ = ["compare_files", "write_report", "certify", "compare", "pass_at_k", "simultaneous_band",
           "load_counts", "load_pair", "fit_kernel", "kernel_curve"]
