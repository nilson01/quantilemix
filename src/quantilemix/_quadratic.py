# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted quadratic Gaussian quantile g-computation with row bootstrap.

Method: Keil et al. (2020), doi:10.1289/EHP5838.
Reference conventions: R qgcomp 2.19.6, GPL (>= 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable
import warnings

import numpy as np
from scipy import special, stats

from ._core import (QGCompWarning, QuantizationResult, _matrix, _readonly,
                    _real_array, fit, quantize)


@dataclass(frozen=True, eq=False)
class QuadraticGaussianBootstrapResult:
    """Joint response as a quadratic curve in the common exposure score."""

    conditional_coefficients: np.ndarray = field(repr=False)
    conditional_covariance: np.ndarray = field(repr=False)
    fitted_values: np.ndarray = field(repr=False)
    intervention_levels: np.ndarray = field(repr=False)
    intervention_means: np.ndarray = field(repr=False)
    msm_coefficients: np.ndarray = field(repr=False)
    psi: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci: np.ndarray = field(repr=False)
    test_statistic: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    p_value_df: int
    bootstrap_coefficients: np.ndarray = field(repr=False)
    B: int
    seed: int | None
    alpha: float
    pairwise_interactions: bool
    covariate_interactions: bool
    _quantization: QuantizationResult = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "gaussian"
    method: str = "fixed_cut_quadratic_row_bootstrap"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def var_psi(self) -> np.ndarray:
        """Bootstrap variances of the linear and quadratic score terms."""
        return _readonly(np.diag(self.covariance))

    def transform(self, exposures: Any) -> QuantizationResult:
        return self._quantization.transform(exposures)

    def effect_at(self, level: int) -> float:
        """Mean difference from joint level ``level`` to ``level + 1``."""
        if (isinstance(level, (bool, np.bool_)) or not isinstance(level, Integral)
                or level not in self.intervention_levels[:-1]):
            raise ValueError("level must identify two consecutive intervention levels")
        return float(self.psi[0] + self.psi[1] * (2 * int(level) + 1))

    def summary(self, digits: int = 6) -> str:
        """Show both curve coefficients and their row-bootstrap uncertainty."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        terms = ("squares and pairwise exposure products" if self.pairwise_interactions
                 else "exposure squares")
        if self.covariate_interactions:
            terms += " and exposure-covariate products"
        return ("Quantile g-computation: quadratic Gaussian joint response\n"
                f"Conditional exposure terms: {terms}.\n"
                f"Linear score coefficient: {fmt(self.psi[0])}; bootstrap SE: {fmt(self.se[0])}\n"
                f"Quadratic score coefficient: {fmt(self.psi[1])}; bootstrap SE: {fmt(self.se[1])}\n"
                f"Normal {(1-self.alpha)*100:g}% CIs: "
                f"({fmt(self.ci[0, 0])}, {fmt(self.ci[0, 1])}); "
                f"({fmt(self.ci[1, 0])}, {fmt(self.ci[1, 1])})\n"
                f"Two-sided Student t p-values (df {self.p_value_df}): "
                f"{fmt(self.p_value[0])}; {fmt(self.p_value[1])}\n"
                f"Resamples: {self.B}; cutpoints held fixed; model-based polynomial means.\n"
                "The mean difference depends on the starting joint score; use effect_at(level).")


def _conditional(y: np.ndarray, scores: np.ndarray, z: np.ndarray,
                 alpha: float, pairwise_interactions: bool,
                 covariate_interactions: bool):
    products = (scores[:, i] * scores[:, j]
                for i in range(scores.shape[1]) for j in range(i + 1, scores.shape[1]))
    terms = [scores * scores]
    if pairwise_interactions:
        terms.extend(column[:, None] for column in products)
    if covariate_interactions:
        terms.extend((scores[:, i]*z[:, j])[:, None]
                     for i in range(scores.shape[1]) for j in range(z.shape[1]))
    squares_and_adjustment = np.column_stack((*terms, z))
    return fit(y, scores, squares_and_adjustment, q=None, alpha=alpha)


def _levels(transformation: QuantizationResult) -> np.ndarray:
    if transformation.mode == "prequantized":
        levels = np.unique(transformation.scores)
        if len(levels) >= 10:
            raise ValueError("prequantized quadratic response requires fewer than 10 pooled levels")
    elif transformation.mode == "explicit":
        if len(set(transformation.category_counts)) != 1:
            raise ValueError("joint interventions require equal exposure category counts")
        levels = np.arange(transformation.category_counts[0])
    else:
        levels = np.arange(transformation.q)
    if len(levels) < 3 or not np.array_equal(levels, np.arange(len(levels))):
        raise ValueError("quadratic response requires at least three consecutive levels starting at zero")
    return _readonly(np.asarray(levels, dtype=float))


def _from_rows(y: np.ndarray, transformation: QuantizationResult, z: np.ndarray,
               rows: Iterable[np.ndarray], B: int, seed: int | None,
               alpha: float, pairwise_interactions: bool,
               covariate_interactions: bool) -> QuadraticGaussianBootstrapResult:
    scores = transformation.scores
    n, d = scores.shape
    if pairwise_interactions and d < 2:
        raise ValueError("pairwise interactions require at least two exposures")
    c = z.shape[1]
    if covariate_interactions and c == 0:
        raise ValueError("exposure-covariate interactions require covariates")
    products = d*(d-1)//2 if pairwise_interactions else 0
    interaction_terms = d*c if covariate_interactions else 0
    df = n - 2*d - products - interaction_terms - c - 3
    if df <= 0:
        raise ValueError("quadratic bootstrap Student t test requires positive reference degrees of freedom")
    levels = _levels(transformation)
    point = _conditional(y, scores, z, alpha, pairwise_interactions,
                         covariate_interactions)
    beta = point.coefficients
    interaction_start = 1+2*d+products
    adjustment_start = interaction_start+interaction_terms
    linear = float(np.sum(beta[1:1+d]))
    if covariate_interactions:
        linear += float(np.sum(beta[interaction_start:adjustment_start].reshape(d, c)
                               * np.mean(z, axis=0)))
    curved = float(np.sum(beta[1+d:1+2*d+products]))
    intercept = float(beta[0] + np.mean(z @ beta[adjustment_start:]))
    msm = np.array([intercept, linear, curved])
    means = msm[0] + msm[1]*levels + msm[2]*levels**2
    effects = np.empty((B, 2))
    for b, sampled in enumerate(rows):
        try:
            replica = _conditional(y[sampled], scores[sampled], z[sampled],
                                   alpha, pairwise_interactions,
                                   covariate_interactions)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b + 1} failed: {exc}") from exc
        effects[b, 0] = np.sum(replica.coefficients[1:1+d])
        if covariate_interactions:
            effects[b, 0] += np.sum(
                replica.coefficients[interaction_start:adjustment_start].reshape(d, c)
                * np.mean(z[sampled], axis=0))
        effects[b, 1] = np.sum(replica.coefficients[1+d:1+2*d+products])
    covariance = np.cov(effects, rowvar=False, ddof=1)
    se = np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(se)):
        raise ValueError("quadratic bootstrap covariance is nonfinite")
    psi = msm[1:]
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = np.column_stack((psi - critical*se, psi + critical*se))
    statistic = np.divide(psi, se, out=np.full(2, np.nan), where=se > 0)
    p = 2*stats.t.sf(np.abs(statistic), df=df)
    if np.any(se == 0):
        warnings.warn("A quadratic bootstrap coefficient has zero variation; its test is undefined.",
                      QGCompWarning, stacklevel=2)
    return QuadraticGaussianBootstrapResult(
        conditional_coefficients=_readonly(beta),
        conditional_covariance=_readonly(point.covariance),
        fitted_values=_readonly(point.fitted_values),
        intervention_levels=levels, intervention_means=_readonly(means),
        msm_coefficients=_readonly(msm), psi=_readonly(psi),
        covariance=_readonly(covariance), se=_readonly(se), ci=_readonly(ci),
        test_statistic=_readonly(statistic), p_value=_readonly(p), p_value_df=df,
        bootstrap_coefficients=_readonly(effects), B=B, seed=seed,
        alpha=alpha, pairwise_interactions=pairwise_interactions,
        covariate_interactions=covariate_interactions,
        _quantization=transformation,
    )


def fit_quadratic_bootstrap(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    B: int = 200, seed: int | None = None,
    pairwise_interactions: bool = False,
    covariate_interactions: bool = False,
) -> QuadraticGaussianBootstrapResult:
    """Fit all exposure squares and a degree-two marginal joint response.

    Complete numeric rows only. Cutpoints are fixed before independent-row
    bootstrap resampling. Set ``pairwise_interactions=True`` to include every
    distinct exposure-score product. Set ``covariate_interactions=True`` to
    include every exposure score by numeric covariate product. Marginalization
    then averages over the observed covariates in each original or resampled
    dataset. Clusters and weights remain unsupported.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    if not isinstance(pairwise_interactions, (bool, np.bool_)):
        raise ValueError("pairwise_interactions must be a Boolean")
    if not isinstance(covariate_interactions, (bool, np.bool_)):
        raise ValueError("covariate_interactions must be a Boolean")
    transformation = quantize(exposures, q=q, breaks=breaks)
    _levels(transformation)
    y_array = _real_array(y, "y")
    n = len(transformation.scores)
    if y_array.ndim != 1 or len(y_array) != n:
        raise ValueError("y must have one value per exposure row")
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    generator = np.random.default_rng(seed)
    rows = (generator.integers(0, n, size=n, dtype=np.intp) for _ in range(int(B)))
    return _from_rows(y_array, transformation, z, rows, int(B),
                      None if seed is None else int(seed), alpha,
                      bool(pairwise_interactions), bool(covariate_interactions))


def _fit_quadratic_with_indices(
    y: Any, exposures: Any, indices: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    pairwise_interactions: bool = False,
    covariate_interactions: bool = False,
) -> QuadraticGaussianBootstrapResult:
    """Use an external fixed row schedule only for reference verification."""
    transformation = quantize(exposures, q=q, breaks=breaks)
    y_array = _real_array(y, "y")
    n = len(transformation.scores)
    if y_array.ndim != 1 or len(y_array) != n:
        raise ValueError("y must have one value per exposure row")
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    idx = _real_array(indices, "resample indices")
    if (idx.ndim != 2 or idx.shape[1] != n or idx.shape[0] < 2
            or np.any(idx < 0) or np.any(idx >= n)
            or np.any(idx != np.floor(idx))):
        raise ValueError("resample indices must be a B-by-n matrix of zero-based row positions")
    if not isinstance(pairwise_interactions, (bool, np.bool_)):
        raise ValueError("pairwise_interactions must be a Boolean")
    if not isinstance(covariate_interactions, (bool, np.bool_)):
        raise ValueError("covariate_interactions must be a Boolean")
    return _from_rows(y_array, transformation, z, idx.astype(np.intp),
                      len(idx), None, alpha, bool(pairwise_interactions),
                      bool(covariate_interactions))
