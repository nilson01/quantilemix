# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal zero-inflated negative-binomial fixed-cut row bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy import special

from ._core import _matrix, _readonly, _real_array
from ._zero_inflated_bootstrap import ZeroInflatedPoissonBootstrapResult, _levels
from ._zero_inflated_negbin import (ZeroInflatedNegativeBinomialResult,
                                    fit_zero_inflated_negative_binomial)


@dataclass(frozen=True, eq=False)
class ZeroInflatedNegativeBinomialBootstrapResult(ZeroInflatedPoissonBootstrapResult):
    """Marginal NB count and structural-zero slopes with bootstrap intervals."""

    family: str = "zero_inflated_negative_binomial"

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: marginal zero-inflated negative-binomial bootstrap\n"
                f"Count-component log mean slope: {fmt(self.psi_count)}; "
                f"bootstrap SE {fmt(self.se[0])}; normal {(1-self.alpha)*100:g}% CI "
                f"({fmt(self.ci[0, 0])}, {fmt(self.ci[0, 1])}).\n"
                f"Structural-zero log odds slope: {fmt(self.psi_zero)}; "
                f"bootstrap SE {fmt(self.se[1])}; normal {(1-self.alpha)*100:g}% CI "
                f"({fmt(self.ci[1, 0])}, {fmt(self.ci[1, 1])}).\n"
                f"Count dispersion in point fit: {fmt(self.point_fit.theta)}.\n"
                f"{self.B} row resamples; {self.MCsize} simulated adjustment profiles per score.\n"
                "These component slopes are not an observed-count mean ratio. "
                "Cutpoints are fixed and rows are assumed independent.")


def _negbin_marginal_stage(point: ZeroInflatedNegativeBinomialResult,
                           z: np.ndarray, values: np.ndarray, MCsize: int,
                           rng: np.random.Generator):
    d = point.scores.shape[1]
    p = 1+d+z.shape[1]
    selected = rng.integers(0, len(z), size=MCsize, dtype=np.intp)
    adjustment = z[selected]
    outcomes = np.empty((len(values), MCsize), dtype=np.float64)
    means = np.empty(len(values))
    for j, level in enumerate(values):
        design = np.column_stack((np.ones(MCsize), np.full((MCsize, d), level), adjustment))
        count_eta = design @ point.coefficients[:p]
        if np.max(count_eta) > 20 or np.min(count_eta) < -25:
            raise ValueError("intervention count mean exceeds the supported numerical range")
        mu = np.exp(count_eta)
        pi = special.expit(design @ point.coefficients[p:])
        means[j] = float(np.mean((1-pi)*mu))
        outcomes[j] = (rng.binomial(1, 1-pi)
                       * rng.negative_binomial(point.theta, point.theta/(point.theta+mu)))
    joint_score = np.repeat(values, MCsize)
    marginal = fit_zero_inflated_negative_binomial(
        outcomes.ravel(), joint_score, q=None, alpha=point.alpha)
    return marginal, means


def fit_zero_inflated_negative_binomial_bootstrap(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    B: int = 200, MCsize: int = 10000, seed: int | None = None,
) -> ZeroInflatedNegativeBinomialBootstrapResult:
    """Fit degree-one marginal NB2 count and structural-zero slopes.

    Sample adjustment profiles under each joint exposure score. Draw a
    structural-zero indicator and an NB2 count using the fitted size. Fit a
    marginal NB2 mixture, then repeat both stages on independent-row
    resamples with the original exposure cuts. Failed replicates are named.
    The Python and R random streams differ. A finite MCsize adds simulation
    noise that the row-bootstrap interval does not separately remove.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if isinstance(MCsize, (bool, np.bool_)) or not isinstance(MCsize, Integral) or MCsize < 2:
        raise ValueError("MCsize must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    point = fit_zero_inflated_negative_binomial(y, exposures, covariates,
                                                q=q, breaks=breaks, alpha=alpha)
    values = _levels(point)
    outcome = _real_array(y, "y")
    n = len(outcome)
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
    rng = np.random.default_rng(seed)
    try:
        marginal, means = _negbin_marginal_stage(point, z, values, int(MCsize), rng)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"marginal point fit failed: {exc}") from exc
    effects = np.empty((int(B), 2))
    bootstrap_means = np.empty((int(B), len(values)))
    for b in range(int(B)):
        rows = rng.integers(0, n, size=n, dtype=np.intp)
        try:
            replica = fit_zero_inflated_negative_binomial(
                outcome[rows], point.scores[rows], z[rows] if z.shape[1] else None,
                q=None, alpha=alpha)
            stage, bootstrap_means[b] = _negbin_marginal_stage(
                replica, z[rows], values, int(MCsize), rng)
            effects[b] = (stage.psi_count, stage.psi_zero)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b+1} failed: {exc}") from exc
    covariance = np.atleast_2d(np.cov(effects, rowvar=False, ddof=1))
    se = np.sqrt(np.diag(covariance))
    psi = np.array((marginal.psi_count, marginal.psi_zero))
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2)))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_value = np.where(se > 0, 2*special.ndtr(-np.abs(psi/se)), np.nan)
    if not all(np.all(np.isfinite(v)) for v in (effects, means, bootstrap_means,
                                               covariance, ci)):
        raise ValueError("marginal bootstrap produced nonfinite estimates")
    return ZeroInflatedNegativeBinomialBootstrapResult(
        point, marginal, float(psi[0]), float(psi[1]), _readonly(covariance),
        _readonly(se), _readonly(ci), _readonly(p_value), values,
        _readonly(means), _readonly(effects), _readonly(bootstrap_means),
        float(alpha), int(B), int(MCsize), None if seed is None else int(seed),
    )
