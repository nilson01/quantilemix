# SPDX-License-Identifier: GPL-2.0-or-later
"""Pointwise and simultaneous marginal-curve bounds for row-bootstrap fits."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import special, stats

from ._bootstrap import GaussianBootstrapResult
from ._core import _readonly
from ._poisson_bootstrap import PoissonBootstrapResult


@dataclass(frozen=True, eq=False)
class ModelBoundResult:
    """Read-only columns named as in R modelbound.boot.

    Quantile is the zero-based joint score. Linpred is the fitted marginal
    curve on the link scale. With pwonly=True, simultaneous columns contain
    missing values because no simultaneous set was requested.
    """

    columns: tuple[str, ...]
    table: Mapping[str, np.ndarray]
    alpha: float
    family: str
    link: str
    pwonly: bool

    def __getitem__(self, name: str) -> np.ndarray:
        """Return a read-only table column."""
        return self.table[name]


def modelbound_boot(
    result: GaussianBootstrapResult | PoissonBootstrapResult,
    *, alpha: float = 0.05, pwonly: bool = False,
) -> ModelBoundResult:
    """Bound a fitted additive marginal curve at every joint score.

    Pointwise limits use the bootstrap standard deviation at each score.
    Simultaneous limits follow R qgcomp 2.19.6: retain resampled marginal
    coefficient vectors inside a chi-square ellipsoid around the original
    coefficients, then take the minimum and maximum fitted curve among those
    retained vectors at each score. These are reference-method bounds, not
    a guaranteed finite-sample or causal coverage statement.
    """
    if not isinstance(result, (GaussianBootstrapResult, PoissonBootstrapResult)):
        raise TypeError("result must be a Gaussian or Poisson row-bootstrap fit")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if not isinstance(pwonly, (bool, np.bool_)):
        raise ValueError("pwonly must be a Boolean")
    q = result.point_fit.q
    if q is None or q < 2:
        raise ValueError("model bounds require a non-null q of at least two")
    levels = np.arange(q, dtype=float)
    if (not np.array_equal(result.intervention_levels, levels)
            or result.intervention_means.shape != (q,)
            or result.bootstrap_means.shape != (result.B, q)
            or result.B < 2):
        raise ValueError("bootstrap profiles must cover each fitted quantile level")
    log_link = isinstance(result, PoissonBootstrapResult)
    means = np.asarray(result.intervention_means, dtype=float)
    draws = np.asarray(result.bootstrap_means, dtype=float)
    if (not np.all(np.isfinite(means)) or not np.all(np.isfinite(draws))
            or (log_link and (np.any(means <= 0) or np.any(draws <= 0)))):
        raise ValueError("bootstrap marginal profiles must be finite and positive on a log link")
    original = np.log(means) if log_link else means
    profiles = np.log(draws) if log_link else draws
    se = np.std(profiles, axis=0, ddof=1)
    critical = float(-special.ndtri_exp(np.log(float(alpha)) - np.log(2.0)))
    lower = original - critical*se
    upper = original + critical*se

    if pwonly:
        simultaneous_lower = np.full(q, np.nan)
        simultaneous_upper = np.full(q, np.nan)
    else:
        # Supported additive degree-one marginal profiles are intercept plus
        # joint score times a common slope.
        design = np.column_stack((np.ones(q), levels))
        coefficients = np.linalg.lstsq(design, profiles.T, rcond=None)[0].T
        original_coef = np.linalg.lstsq(design, original, rcond=None)[0]
        if (not np.allclose(coefficients @ design.T, profiles, rtol=1e-9, atol=1e-10)
                or not np.allclose(design @ original_coef, original, rtol=1e-9, atol=1e-10)):
            raise ValueError("simultaneous bounds require degree-one marginal profiles")
        covariance = np.cov(coefficients, rowvar=False)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("bootstrap marginal-coefficient covariance is nonfinite")
        try:
            delta = coefficients - original_coef
            distances = np.sum(delta * np.linalg.solve(covariance, delta.T).T, axis=1)
        except np.linalg.LinAlgError as exc:
            raise ValueError("bootstrap marginal-coefficient covariance is singular") from exc
        accepted = distances < float(stats.chi2.isf(float(alpha), 2))
        if not np.any(accepted):
            raise ValueError("no bootstrap marginal coefficients lie inside the simultaneous set")
        accepted_profiles = coefficients[accepted] @ design.T
        simultaneous_lower = np.min(accepted_profiles, axis=0)
        simultaneous_upper = np.max(accepted_profiles, axis=0)

    if log_link:
        link, family = "log", "poisson"
        columns = ("quantile", "quantile.midpoint", "linpred", "r", "se.pw",
                   "ll.pw", "ul.pw", "ll.simul", "ul.simul")
        values = (levels, (levels+0.5)/q, original, means, se, np.exp(lower),
                  np.exp(upper), np.exp(simultaneous_lower), np.exp(simultaneous_upper))
    else:
        link, family = "identity", "gaussian"
        columns = ("quantile", "quantile.midpoint", "linpred", "m", "se.pw",
                   "ll.pw", "ul.pw", "ll.simul", "ul.simul")
        values = (levels, (levels+0.5)/q, original, original, se, lower, upper,
                  simultaneous_lower, simultaneous_upper)
    if not all(np.all(np.isfinite(value)) for value in values[:7]):
        raise ValueError("pointwise model bounds exceed the finite numeric range")
    if not pwonly and not all(np.all(np.isfinite(value)) for value in values[7:]):
        raise ValueError("simultaneous model bounds exceed the finite numeric range")
    table = MappingProxyType(dict(zip(columns, (_readonly(value) for value in values))))
    return ModelBoundResult(columns, table, float(alpha), family, link, bool(pwonly))
