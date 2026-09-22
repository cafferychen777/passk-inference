"""Grid NPMLE adapted from pilot/heterogeneity.py; explicit convergence metadata."""

import numpy as np
from scipy.special import expit, gammaln

from .core import _budgets, _counts, _positive_int


def _log_binomial(c, n, grid):
    # Preserve the pilot's endpoint convention for numerical comparability.
    p = np.clip(grid[None, :], 1e-12, 1 - 1e-12)
    c, n = c[:, None], n[:, None]
    return (gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1)
            + c * np.log(p) + (n - c) * np.log1p(-p))


def fit_kernel(c, n, d, nprime, *, Lp=80, Lq=80, iters=4000, tol=1e-9):
    """Fit the joint law of base/RL success probabilities on a fixed grid.

    This is a working model, not a mechanism of training or an uncertainty
    interval. A returned fit may reach the iteration limit before convergence.
    """
    c, n = _counts(c, n)
    d, nprime = _counts(d, nprime)
    if c.shape != d.shape:
        raise ValueError("base and RL must have matching aligned prompts")
    Lp, Lq = _positive_int(Lp, "Lp"), _positive_int(Lq, "Lq")
    iters = _positive_int(iters, "iters")
    if Lp < 2 or Lq < 2 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("require Lp >= 2, Lq >= 2, and finite tol > 0")
    pg = expit(np.linspace(-14, 9, Lp))
    qg = np.concatenate([[0.0], expit(np.linspace(-14, 9, Lq - 1))])
    lc, ld = _log_binomial(c, n, pg), _log_binomial(d, nprime, qg)
    mc, md = lc.max(axis=1, keepdims=True), ld.max(axis=1, keepdims=True)
    Lc, Ld = np.exp(lc - mc), np.exp(ld - md)
    W = np.full((Lp, Lq), 1.0 / (Lp * Lq))
    previous, converged = -np.inf, False
    for iteration in range(iters):
        den = ((Lc @ W) * Ld).sum(axis=1)
        if np.any(den <= 0) or not np.all(np.isfinite(den)):
            raise FloatingPointError("kernel likelihood underflow; use a better resolved grid")
        ll = float(np.log(den).sum() + mc.sum() + md.sum())
        W *= Lc.T @ (Ld / den[:, None])
        W /= W.sum()
        if iteration % 20 == 0:
            if abs(ll - previous) < tol * (1 + abs(ll)):
                converged = True
                break
            previous = ll
    # Report likelihood for the returned weights, not the preceding E-step.
    den = ((Lc @ W) * Ld).sum(axis=1)
    return {"W": W, "pg": pg, "qg": qg,
            "loglik": float(np.log(den).sum() + mc.sum() + md.sum()),
            "iters": iteration + 1, "converged": converged}


def kernel_curve(fit, ks):
    """Model-implied RL-minus-base pass@k, including extrapolation if requested."""
    ks = _budgets(ks)
    W = np.asarray(fit["W"], float)
    pg, qg = np.asarray(fit["pg"], float), np.asarray(fit["qg"], float)
    if pg.ndim != 1 or qg.ndim != 1 or W.shape != (len(pg), len(qg)):
        raise ValueError("kernel weights must match the probability grids")
    if (not np.all(np.isfinite(W)) or np.any(W < 0)
            or not np.isclose(W.sum(), 1.0)
            or not np.all(np.isfinite(pg)) or not np.all(np.isfinite(qg))
            or np.any((pg < 0) | (pg > 1)) or np.any((qg < 0) | (qg > 1))):
        raise ValueError("require normalized nonnegative weights and probabilities in [0, 1]")
    base_weights, rl_weights = W.sum(axis=1), W.sum(axis=0)
    return np.array([(base_weights * (1 - pg) ** k).sum()
                     - (rl_weights * (1 - qg) ** k).sum() for k in ks])
