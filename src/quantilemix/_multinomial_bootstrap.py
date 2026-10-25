# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal multinomial joint effects with fixed-cut row bootstrap.

The method follows the simulation and marginal-model stages of R qgcomp
2.19.6. Random streams and optimizer details are Python-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

import numpy as np
from scipy import special

from ._core import _matrix, _readonly
from ._multinomial import MultinomialResult, _probabilities, _solve_multinomial, fit_multinomial


@dataclass(frozen=True, eq=False)
class MultinomialBootstrapResult:
    """Category-specific marginal log odds slopes and bootstrap uncertainty."""

    point_fit: MultinomialResult = field(repr=False)
    psi: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    intervention_levels: np.ndarray = field(repr=False)
    intervention_probabilities: np.ndarray = field(repr=False)
    bootstrap_effects: np.ndarray = field(repr=False)
    alpha: float
    B: int
    seed: int | None
    family: str = "multinomial"
    method: str = "simulated_marginal_fixed_cut_row_bootstrap"
    reference_version: str = "2.19.6"

    @property
    def levels(self) -> tuple[str, ...]:
        return self.point_fit.levels

    @property
    def odds_ratios(self) -> np.ndarray:
        return _readonly(np.exp(self.psi))

    def summary(self, digits: int = 6) -> str:
        """Display each category against the first outcome category."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        lines = ["Quantile g-computation: marginal multinomial row bootstrap",
                 f"Reference outcome: {self.levels[0]!r}; {self.B} resamples."]
        for j, label in enumerate(self.levels[1:]):
            lines.append(f"{label!r} versus {self.levels[0]!r}: log odds slope {fmt(self.psi[j])}; "
                         f"bootstrap SE {fmt(self.se[j])}; normal {(1-self.alpha)*100:g}% CI "
                         f"({fmt(self.ci[j, 0])}, {fmt(self.ci[j, 1])}); "
                         f"z p-value {fmt(self.p_value[j])}.")
        lines.append("Each slope describes a one-score joint increase in the fitted marginal model.")
        lines.append("Independent rows, fixed cutpoints and a linear marginal log odds curve are assumed.")
        return "\n".join(lines)


def _intervention_levels(point: MultinomialResult) -> np.ndarray:
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


def _marginal_stage(point: MultinomialResult, covariates: np.ndarray,
                    values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, d = point.scores.shape
    mean_probabilities = np.empty((len(values), len(point.levels)))
    sampled = np.empty((len(values), n), dtype=np.intp)
    for j, value in enumerate(values):
        design = np.column_stack((np.ones(n), np.full((n, d), value), covariates))
        probabilities = _probabilities(design, point.coefficients)
        mean_probabilities[j] = np.mean(probabilities, axis=0)
        cumulative = np.cumsum(probabilities, axis=1)
        sampled[j] = np.minimum(np.sum(rng.random(n)[:, None] > cumulative, axis=1),
                                len(point.levels)-1)
    marginal_design = np.column_stack((np.ones(len(values)*n), np.repeat(values, n)))
    beta, _, _ = _solve_multinomial(marginal_design, sampled.ravel(), len(point.levels))
    return beta[:, 1], mean_probabilities


def fit_multinomial_bootstrap(
    y: Any, exposures: Any, covariates: Any = None, *,
    levels: Sequence[str], q: int | None = 4, breaks: Any = None,
    alpha: float = 0.05, B: int = 200, seed: int | None = None,
) -> MultinomialBootstrapResult:
    """Fit a degree-one marginal multinomial model after joint interventions.

    Parameters
    ----------
    y : array-like of str, shape (n,)
        Observed outcome categories. Every value must occur in ``levels``.
    exposures : array-like, shape (n, d)
        Complete numeric exposures. One-dimensional input means one exposure.
    covariates : array-like, shape (n, c), optional
        Complete numeric adjustment covariates.
    levels : sequence of str
        Outcome category order. The first category is the reference.
    q, breaks, alpha : optional
        Quantization controls and two-sided error level, as in
        :func:`fit_multinomial`.
    B : int, default 200
        Number of row-bootstrap samples; at least two.
    seed : int or None, default None
        Local random-generator seed. The global NumPy generator is unchanged.

    Returns
    -------
    MultinomialBootstrapResult
        ``psi[j]`` is the marginal log odds slope for category ``j+1`` versus
        the first category. Each score is set jointly for all exposures.
        ``intervention_probabilities`` has one row per joint score and one
        column per outcome category in ``levels`` order.

    Raises
    ------
    ValueError
        Invalid input, failed point fit or failed numbered bootstrap sample.

    Notes
    -----
    The fitted conditional model gives category probabilities at every joint
    score. One outcome is drawn per row and score, then a multinomial logit
    with an intercept and joint score is fitted to the simulated outcomes.
    Each bootstrap sample refits both stages on sampled rows. Cutpoints remain
    fixed. A failed sample is not replaced. This follows the R qgcomp 2.19.6
    stage structure but uses a separate Python random stream and optimizer.
    Normal intervals use the bootstrap standard deviation; they do not include
    a separate correction for finite simulation size. R is not needed at runtime.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    point = fit_multinomial(y, exposures, covariates, levels=levels, q=q,
                            breaks=breaks, alpha=alpha)
    values = _intervention_levels(point)
    n = len(point.scores)
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
    labels = np.asarray(y, dtype=object)
    rng = np.random.default_rng(seed)
    try:
        psi, probabilities = _marginal_stage(point, z, values, rng)
    except ValueError as exc:
        raise ValueError(f"marginal point fit failed: {exc}") from exc
    effects = np.empty((int(B), len(point.levels)-1))
    for b in range(int(B)):
        rows = rng.integers(0, n, size=n, dtype=np.intp)
        try:
            replica = fit_multinomial(labels[rows], point.scores[rows], z[rows] if z.shape[1] else None,
                                      levels=point.levels, q=None, alpha=alpha)
            effects[b], _ = _marginal_stage(replica, z[rows], values, rng)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b+1} failed: {exc}") from exc
    covariance = np.cov(effects, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    se = np.sqrt(np.diag(covariance))
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2)))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_value = np.where(se > 0, 2*special.ndtr(-np.abs(psi/se)), np.nan)
    if not all(np.all(np.isfinite(item)) for item in (psi, probabilities, effects, covariance, ci)):
        raise ValueError("marginal bootstrap produced nonfinite estimates")
    return MultinomialBootstrapResult(point, _readonly(psi), _readonly(covariance),
                                      _readonly(se), _readonly(ci), _readonly(p_value),
                                      values, _readonly(probabilities), _readonly(effects),
                                      float(alpha), int(B), None if seed is None else int(seed))
