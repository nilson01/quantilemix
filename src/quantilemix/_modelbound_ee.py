# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal-curve bounds for the restricted Gaussian EE fit."""

from __future__ import annotations

from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from scipy import stats

from ._core import _readonly
from ._ee import GaussianEEResult
from ._modelbound_boot import ModelBoundResult


def _from_coefficient_draws(
    result: GaussianEEResult, draws: np.ndarray, *, alpha: float, pwonly: bool,
) -> ModelBoundResult:
    """Calculate the table from supplied draws, including installed-R draws."""
    levels = np.asarray(result.intervention_levels, dtype=float)
    q = len(levels)
    if not np.array_equal(levels, np.arange(q, dtype=float)):
        raise ValueError("model bounds require consecutive joint levels starting at zero")
    beta = np.asarray(result.msm_coefficients, dtype=float)
    covariance = np.asarray(result.msm_covariance, dtype=float)
    sampled = np.asarray(draws, dtype=float)
    if (beta.shape != (2,) or covariance.shape != (2, 2)
            or sampled.shape != (5000, 2) or not np.all(np.isfinite(sampled))
            or not np.all(np.isfinite(beta)) or not np.all(np.isfinite(covariance))):
        raise ValueError("Gaussian EE requires finite 5000-by-2 coefficient draws")
    design = np.column_stack((np.ones(q), levels))
    fitted = design @ beta
    profiles = sampled @ design.T
    se = np.std(profiles, axis=0, ddof=1)
    critical = float(stats.norm.isf(alpha / 2))
    if pwonly:
        ll = np.full(q, np.nan)
        ul = np.full(q, np.nan)
    else:
        try:
            deviations = sampled - beta
            distances = np.sum(deviations * np.linalg.solve(covariance, deviations.T).T,
                               axis=1)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Gaussian EE marginal covariance is singular") from exc
        accepted = distances < float(stats.chi2.isf(alpha, 2))
        if not np.any(accepted):
            raise ValueError("no sampled coefficients lie inside the simultaneous set")
        ll = np.min(profiles[accepted], axis=0)
        ul = np.max(profiles[accepted], axis=0)
    columns = ("quantile", "quantile.midpoint", "linpred", "m", "se.pw",
               "ll.pw", "ul.pw", "ll.simul", "ul.simul")
    values = (levels, (levels + 0.5) / q, fitted, fitted, se,
              fitted - critical * se, fitted + critical * se, ll, ul)
    if not all(np.all(np.isfinite(value)) for value in values[:7]):
        raise ValueError("pointwise model bounds exceed the finite numeric range")
    if not pwonly and not all(np.all(np.isfinite(value)) for value in values[7:]):
        raise ValueError("simultaneous model bounds exceed the finite numeric range")
    table = MappingProxyType(dict(zip(columns, (_readonly(value) for value in values))))
    return ModelBoundResult(columns, table, float(alpha), "gaussian", "identity", pwonly)


def modelbound_ee(
    result: GaussianEEResult, *, alpha: float = 0.05, pwonly: bool = False,
    seed: int | None = None,
) -> ModelBoundResult:
    """Return pointwise and R-method simultaneous bounds for Gaussian EE.

    Like R qgcomp 2.19.6, this samples 5,000 marginal coefficient vectors
    from the fitted normal approximation. ``seed`` makes Python's draws
    reproducible; the same integer does not give the same draws in R.
    These bounds are sampling-based and do not establish repeated-data coverage.
    """
    if not isinstance(result, GaussianEEResult):
        raise TypeError("result must be a Gaussian estimating-equation fit")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if not isinstance(pwonly, (bool, np.bool_)):
        raise ValueError("pwonly must be a Boolean")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    covariance = np.asarray(result.msm_covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("Gaussian EE marginal covariance must be finite and 2-by-2")
    eigenvalues, basis = np.linalg.eigh((covariance + covariance.T) / 2)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Gaussian EE marginal covariance must be positive definite")
    rng = np.random.default_rng(seed)
    draws = (rng.standard_normal((5000, 2)) * np.sqrt(eigenvalues)) @ basis.T
    draws += result.msm_coefficients
    return _from_coefficient_draws(result, draws, alpha=float(alpha), pwonly=bool(pwonly))
