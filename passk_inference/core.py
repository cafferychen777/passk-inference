"""Paired prompt-level inference; no model fitting or filesystem side effects."""

from numbers import Integral

import numpy as np


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _counts(c, n):
    """Validate a nonempty vector of success counts and scalar/vector budgets."""
    c = np.asarray(c, dtype=float)
    n = np.asarray(n, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("success counts must be a nonempty one-dimensional vector")
    if n.ndim == 0:
        n = np.full(c.shape, n)
    if n.shape != c.shape:
        raise ValueError("success and generation counts must have matching shapes")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(n)):
        raise ValueError("counts must be finite")
    if np.any(c != np.floor(c)) or np.any(n != np.floor(n)):
        raise ValueError("counts must be integers")
    if np.any(n < 1) or np.any(c < 0) or np.any(c > n):
        raise ValueError("counts require n >= 1 and 0 <= c <= n")
    return c, n


def _budgets(ks):
    values = list(ks)
    if not values:
        raise ValueError("ks must be nonempty")
    values = np.array([_positive_int(k, "k") for k in values], dtype=int)
    if np.any(np.diff(values) <= 0):
        raise ValueError("ks must be strictly increasing without duplicates")
    return values


def pass_at_k(c, n, k):
    """Unbiased per-prompt pass@k; requires k <= every observed budget."""
    c, n = _counts(c, n)
    k = _positive_int(k, "k")
    if np.any(n < k):
        raise ValueError("k exceeds an observed generation budget")
    return _pass_at_k(c, n, k)


def _pass_at_k(c, n, k):
    # Two equal combinatorial ratios let us use min(c, k) factors. Summing
    # log1p factors avoids subtracting nearly equal log-gamma values.
    out = np.ones_like(c)
    for i in np.flatnonzero(n - c >= k):
        factors = np.arange(min(int(c[i]), k), dtype=float)
        log_failure = np.log1p(-max(c[i], k) / (n[i] - factors)).sum()
        out[i] = -np.expm1(log_failure)
    return out


def simultaneous_band(rows, *, bootstrap=4000, alpha=0.05, seed=0):
    """Gaussian prompt-multiplier sup-t band on the supplied finite grid.

    This is an asymptotic procedure for independent prompt rows. It does not
    include training-seed uncertainty. Constant columns have zero empirical
    variance; their returned bands are computational diagnostics only.
    They cannot support population inference without additional assumptions.
    compare() substitutes the known [-1, 1] support on such columns.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] == 0:
        raise ValueError("rows must have shape (at least 2 prompts, at least 1 budget)")
    if not np.all(np.isfinite(rows)):
        raise ValueError("rows must be finite")
    bootstrap = _positive_int(bootstrap, "bootstrap")
    if bootstrap < 2 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("require bootstrap >= 2 and 0 < alpha < 1")
    m = len(rows)
    mean = rows.mean(axis=0)
    se = rows.std(axis=0, ddof=1) / np.sqrt(m)
    centered = rows - mean
    scale = np.maximum(se, 1e-12)
    rng = np.random.default_rng(seed)
    maxima = np.empty(bootstrap)
    # Bound memory without changing the RNG draw order across batches.
    for start in range(0, bootstrap, 128):
        stop = min(start + 128, bootstrap)
        e = rng.standard_normal((stop - start, m))
        maxima[start:stop] = np.max(np.abs(e @ centered) / (m * scale), axis=1)
    critical = float(np.quantile(maxima, 1 - alpha))
    return {
        "mean": mean, "se": se, "lower": mean - critical * se,
        "upper": mean + critical * se, "critical_value": critical,
        "degenerate_columns": np.flatnonzero(se < 1e-12).tolist(),
    }


def certify(ks, lower, upper):
    """Certify an early gain followed by a later loss on a declared grid.

    A sparse grid does not bound the first loss at unobserved integers. A
    first-integer-loss interval is therefore emitted only on 1, ..., K.
    Null upper endpoints mean no finite upper bound was established.
    """
    ks = _budgets(ks)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != ks.shape or upper.shape != ks.shape:
        raise ValueError("bounds must match ks")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    positive = ks[lower > 0]
    negative = ks[upper < 0]
    crossing = bool(positive.size and negative.size and positive.min() < negative.max())
    first_upper = int(negative[0]) if negative.size else None
    last_positive_prefix = 0
    for k, lo in zip(ks, lower):
        if lo <= 0:
            break
        last_positive_prefix = int(k)
    dense = np.array_equal(ks, np.arange(1, len(ks) + 1))
    return {
        "certified_crossing": crossing,
        "first_loss_interval": [last_positive_prefix + 1, first_upper] if dense else None,
        "first_loss_scope": "all_integers_1_to_K" if dense else "sparse_grid_no_integer_interval",
        "certified_positive_budgets": positive.tolist(),
        "certified_negative_budgets": negative.tolist(),
    }


def compare(c_base, n_base, c_rl, n_rl, *, ks=None, bootstrap=4000, alpha=0.05, seed=0):
    """Compare aligned independent prompts; positive differences favor RL."""
    c_base, n_base = _counts(c_base, n_base)
    c_rl, n_rl = _counts(c_rl, n_rl)
    if c_base.shape != c_rl.shape:
        raise ValueError("base and RL must have the same aligned prompts")
    maximum = int(min(n_base.min(), n_rl.min()))
    ks = _budgets(range(1, maximum + 1) if ks is None else ks)
    if ks[-1] > maximum:
        raise ValueError("k exceeds an observed generation budget")
    base_rows = np.column_stack([_pass_at_k(c_base, n_base, k) for k in ks])
    rl_rows = np.column_stack([_pass_at_k(c_rl, n_rl, k) for k in ks])
    rows = rl_rows - base_rows
    band = simultaneous_band(rows, bootstrap=bootstrap, alpha=alpha, seed=seed)
    # A collapsed empirical bootstrap does not establish population certainty.
    # Each difference lies in [-1, 1], giving a safe band without inventing SEs.
    degenerate = band["degenerate_columns"]
    band["lower"][degenerate] = -1.0
    band["upper"][degenerate] = 1.0
    return {
        "ks": ks.tolist(), "prompts": len(c_base), "alpha": float(alpha),
        "bootstrap": int(bootstrap), "seed": seed,
        "difference_direction": "rl_minus_base",
        "base_pass_at_k": base_rows.mean(axis=0).tolist(),
        "rl_pass_at_k": rl_rows.mean(axis=0).tolist(),
        "inconclusive_budgets": ks[(band["lower"] <= 0) & (band["upper"] >= 0)].tolist(),
        **{key: value.tolist() if isinstance(value, np.ndarray) else value
           for key, value in band.items()},
        **certify(ks, band["lower"], band["upper"]),
    }
