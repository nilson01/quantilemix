# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal zero-inflated Poisson effects with fixed-cut row bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy import special

from ._core import _matrix, _readonly, _real_array
from ._zero_inflated import ZeroInflatedPoissonResult, fit_zero_inflated_poisson


@dataclass(frozen=True, eq=False)
class ZeroInflatedPoissonBootstrapResult:
    """Marginal count and structural-zero slopes with bootstrap uncertainty."""

    point_fit: ZeroInflatedPoissonResult = field(repr=False)
    marginal_fit: ZeroInflatedPoissonResult = field(repr=False)
    psi_count: float
    psi_zero: float
    covariance: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    intervention_levels: np.ndarray = field(repr=False)
    intervention_means: np.ndarray = field(repr=False)
    bootstrap_effects: np.ndarray = field(repr=False)
    bootstrap_means: np.ndarray = field(repr=False)
    alpha: float
    B: int
    MCsize: int
    seed: int | None
    family: str = "zero_inflated_poisson"
    method: str = "simulated_marginal_fixed_cut_row_bootstrap"
    reference_version: str = "2.19.6"

    def summary(self, digits: int = 6) -> str:
        """Display marginal component effects and bootstrap intervals."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: marginal zero-inflated Poisson bootstrap\n"
                f"Count-component log mean slope: {fmt(self.psi_count)}; "
                f"bootstrap SE {fmt(self.se[0])}; normal {(1-self.alpha)*100:g}% CI "
                f"({fmt(self.ci[0, 0])}, {fmt(self.ci[0, 1])}).\n"
                f"Structural-zero log odds slope: {fmt(self.psi_zero)}; "
                f"bootstrap SE {fmt(self.se[1])}; normal {(1-self.alpha)*100:g}% CI "
                f"({fmt(self.ci[1, 0])}, {fmt(self.ci[1, 1])}).\n"
                f"{self.B} row resamples; {self.MCsize} simulated adjustment profiles per score.\n"
                "These component slopes are not an observed-count mean ratio. "
                "Cutpoints are fixed and rows are assumed independent.")


def _levels(point: ZeroInflatedPoissonResult) -> np.ndarray:
    quantization = point._quantization
    if quantization.mode == "prequantized":
        values = np.unique(point.scores)
        if not 2 <= len(values) < 10:
            raise ValueError("prequantized bootstrap requires 2 through 9 pooled score levels")
    elif quantization.mode == "explicit":
        if len(set(quantization.category_counts)) != 1:
            raise ValueError("joint interventions require equal exposure category counts")
        values = np.arange(quantization.category_counts[0], dtype=float)
    else:
        values = np.arange(quantization.q, dtype=float)
    if len(values) < 2:
        raise ValueError("at least two joint intervention levels are required")
    return _readonly(values)


def _marginal_stage(point: ZeroInflatedPoissonResult, z: np.ndarray,
                    values: np.ndarray, MCsize: int,
                    rng: np.random.Generator) -> tuple[ZeroInflatedPoissonResult, np.ndarray]:
    d = point.scores.shape[1]
    p = 1+d+z.shape[1]
    selected = rng.integers(0, len(z), size=MCsize, dtype=np.intp)
    adjustment = z[selected]
    outcomes = np.empty((len(values), MCsize), dtype=np.float64)
    means = np.empty(len(values))
    for j, level in enumerate(values):
        design = np.column_stack((np.ones(MCsize), np.full((MCsize, d), level), adjustment))
        count_eta = design @ point.coefficients[:p]
        if np.max(count_eta) > 20:
            raise ValueError("intervention Poisson mean exceeds the supported numerical range")
        mu = np.exp(count_eta)
        pi = special.expit(design @ point.coefficients[p:])
        means[j] = np.mean((1-pi)*mu)
        outcomes[j] = rng.binomial(1, 1-pi) * rng.poisson(mu)
    joint_score = np.repeat(values, MCsize)
    marginal = fit_zero_inflated_poisson(outcomes.ravel(), joint_score,
                                         q=None, alpha=point.alpha)
    return marginal, means


def fit_zero_inflated_poisson_bootstrap(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    B: int = 200, MCsize: int = 10000, seed: int | None = None,
) -> ZeroInflatedPoissonBootstrapResult:
    """Fit a degree-one marginal zero-inflated Poisson model.

    Parameters
    ----------
    y : array-like, shape (n,)
        Complete nonnegative integer counts.
    exposures : array-like, shape (n, d)
        Complete numeric exposures. A vector means one exposure.
    covariates : array-like, shape (n, c), optional
        Complete numeric adjustment columns used in both components.
    q, breaks, alpha : optional
        Quantization controls and two-sided error level, as in the conditional
        :func:`fit_zero_inflated_poisson` model.
    B : int, default 200
        Number of independent-row bootstrap samples; at least two.
    MCsize : int, default 10000
        Number of sampled adjustment profiles per joint intervention. This is
        R qgcomp 2.19.6's default. Small values increase simulation error.
    seed : int or None, default None
        Local random-generator seed, leaving global NumPy state unchanged.

    Returns
    -------
    ZeroInflatedPoissonBootstrapResult
        Degree-one marginal log mean and structural-zero log odds slopes,
        normal intervals using bootstrap standard deviations, and mean
        conditional expected counts at each intervention score.

    Notes
    -----
    Under each joint score, adjustment rows are sampled with replacement.
    A Poisson draw is multiplied by a nonstructural-zero Bernoulli draw.
    A zero-inflated Poisson model with intercept and joint score in both
    components is fitted to the simulated counts. Both stages are refitted
    for every row-bootstrap sample, holding exposure cutpoints fixed. This
    follows R's default component-simulation stage; Python random draws and
    optimizer paths differ. No R runtime is needed. A failed sample reports
    its one-based position and is not replaced. Finite simulation size adds
    noise that the bootstrap interval does not separately remove.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if isinstance(MCsize, (bool, np.bool_)) or not isinstance(MCsize, Integral) or MCsize < 2:
        raise ValueError("MCsize must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    point = fit_zero_inflated_poisson(y, exposures, covariates,
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
        marginal, means = _marginal_stage(point, z, values, int(MCsize), rng)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"marginal point fit failed: {exc}") from exc
    effects = np.empty((int(B), 2))
    bootstrap_means = np.empty((int(B), len(values)))
    for b in range(int(B)):
        rows = rng.integers(0, n, size=n, dtype=np.intp)
        try:
            replica = fit_zero_inflated_poisson(outcome[rows], point.scores[rows],
                                                 z[rows] if z.shape[1] else None,
                                                 q=None, alpha=alpha)
            stage, bootstrap_means[b] = _marginal_stage(replica, z[rows], values,
                                                        int(MCsize), rng)
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
    return ZeroInflatedPoissonBootstrapResult(
        point, marginal, float(psi[0]), float(psi[1]), _readonly(covariance),
        _readonly(se), _readonly(ci), _readonly(p_value), values,
        _readonly(means), _readonly(effects), _readonly(bootstrap_means),
        float(alpha), int(B), int(MCsize), None if seed is None else int(seed),
    )
