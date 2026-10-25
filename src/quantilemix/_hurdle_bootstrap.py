# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal Poisson hurdle effects with fixed-cut row bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy import special, stats

from ._core import _matrix, _readonly, _real_array
from ._hurdle import HurdlePoissonResult, fit_hurdle_poisson


@dataclass(frozen=True, eq=False)
class HurdlePoissonBootstrapResult:
    """Marginal positive-count and nonzero slopes with bootstrap uncertainty."""

    point_fit: HurdlePoissonResult = field(repr=False)
    marginal_fit: HurdlePoissonResult = field(repr=False)
    psi_count: float
    psi_nonzero: float
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
    family: str = "hurdle_poisson"
    method: str = "simulated_marginal_fixed_cut_row_bootstrap"
    reference_version: str = "2.19.6"

    def summary(self, digits: int = 6) -> str:
        """Display marginal component effects and bootstrap intervals."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: marginal Poisson hurdle bootstrap\n"
                f"Positive-count log parameter slope: {fmt(self.psi_count)}; "
                f"bootstrap SE {fmt(self.se[0])}; normal {(1-self.alpha)*100:g}% CI "
                f"({fmt(self.ci[0, 0])}, {fmt(self.ci[0, 1])}).\n"
                f"Nonzero log odds slope: {fmt(self.psi_nonzero)}; "
                f"bootstrap SE {fmt(self.se[1])}; normal {(1-self.alpha)*100:g}% CI "
                f"({fmt(self.ci[1, 0])}, {fmt(self.ci[1, 1])}).\n"
                f"{self.B} row resamples; {self.MCsize} simulated adjustment profiles per score.\n"
                "These component slopes are not an observed-count mean ratio. "
                "Cutpoints are fixed and rows are assumed independent.")


def _levels(point: HurdlePoissonResult) -> np.ndarray:
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


def _marginal_stage(point: HurdlePoissonResult, z: np.ndarray,
                    values: np.ndarray, MCsize: int,
                    rng: np.random.Generator) -> tuple[HurdlePoissonResult, np.ndarray]:
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
            raise ValueError("intervention Poisson parameter exceeds the supported numerical range")
        mu = np.exp(count_eta)
        p_nonzero = special.expit(design @ point.coefficients[p:])
        normalizer = -np.expm1(-mu)
        means[j] = np.mean(p_nonzero*mu/normalizer)
        # Inverse CDF of Poisson conditional on being positive; clamp rounding
        # at the upper endpoint and keep the positive support explicit.
        positive = rng.binomial(1, p_nonzero)
        u = rng.random(MCsize)
        cdf = np.minimum(np.exp(-mu)+u*normalizer, np.nextafter(1.0, 0.0))
        counts = np.maximum(1, stats.poisson.ppf(cdf, mu))
        outcomes[j] = positive*counts
    joint_score = np.repeat(values, MCsize)
    marginal = fit_hurdle_poisson(outcomes.ravel(), joint_score,
                                  q=None, alpha=point.alpha)
    return marginal, means


def fit_hurdle_poisson_bootstrap(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    B: int = 200, MCsize: int = 10000, seed: int | None = None,
) -> HurdlePoissonBootstrapResult:
    """Fit a degree-one marginal Poisson hurdle model.

    Complete numeric additive inputs are required. Each joint intervention
    samples adjustment profiles, then simulates nonzero status and a positive
    zero-truncated Poisson count. A marginal hurdle model is fitted to those
    outcomes. Independent rows are resampled with the original exposure cuts
    fixed. Failed replicates are reported by position, not silently replaced.

    R qgcomp 2.19.6 calls its hurdle bootstrap experimental and defaults to
    sampling from a finite fitted category-probability matrix. This Python
    subset samples from the full zero-truncated Poisson support instead;
    finite Monte Carlo draws and optimizers also differ. No R runtime is used.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if isinstance(MCsize, (bool, np.bool_)) or not isinstance(MCsize, Integral) or MCsize < 2:
        raise ValueError("MCsize must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    point = fit_hurdle_poisson(y, exposures, covariates,
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
            replica = fit_hurdle_poisson(outcome[rows], point.scores[rows],
                                          z[rows] if z.shape[1] else None,
                                          q=None, alpha=alpha)
            stage, bootstrap_means[b] = _marginal_stage(replica, z[rows], values,
                                                        int(MCsize), rng)
            effects[b] = (stage.psi_count, stage.psi_nonzero)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b+1} failed: {exc}") from exc
    covariance = np.atleast_2d(np.cov(effects, rowvar=False, ddof=1))
    se = np.sqrt(np.diag(covariance))
    psi = np.array((marginal.psi_count, marginal.psi_nonzero))
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2)))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_value = np.where(se > 0, 2*special.ndtr(-np.abs(psi/se)), np.nan)
    if not all(np.all(np.isfinite(v)) for v in (effects, means, bootstrap_means,
                                               covariance, ci)):
        raise ValueError("marginal bootstrap produced nonfinite estimates")
    return HurdlePoissonBootstrapResult(
        point, marginal, float(psi[0]), float(psi[1]), _readonly(covariance),
        _readonly(se), _readonly(ci), _readonly(p_value), values,
        _readonly(means), _readonly(effects), _readonly(bootstrap_means),
        float(alpha), int(B), int(MCsize), None if seed is None else int(seed),
    )
