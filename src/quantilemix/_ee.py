# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted clustered Gaussian quantile g-computation estimating equations.

Reference: R qgcomp 2.19.6, GPL (>= 2). The robust covariance sums
estimating functions within each cluster before forming the sandwich meat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
from scipy import special, stats

from ._core import (QGCompWarning, QuantizationResult, _matrix, _readonly,
                    _real_array, fit, quantize)


@dataclass(frozen=True, eq=False)
class GaussianEEResult:
    """Additive joint mean difference with cluster-robust uncertainty."""

    conditional_coefficients: np.ndarray = field(repr=False)
    conditional_covariance: np.ndarray = field(repr=False)
    msm_coefficients: np.ndarray = field(repr=False)
    msm_covariance: np.ndarray = field(repr=False)
    full_covariance: np.ndarray = field(repr=False)
    bread: np.ndarray = field(repr=False)
    meat: np.ndarray = field(repr=False)
    fitted_values: np.ndarray = field(repr=False)
    intervention_levels: np.ndarray = field(repr=False)
    intervention_means: np.ndarray = field(repr=False)
    cluster_ids: np.ndarray = field(repr=False)
    case_weights: np.ndarray | None = field(repr=False)
    cluster_count: int
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    p_value_df: int
    alpha: float
    _quantization: QuantizationResult = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "gaussian"
    method: str = "clustered_gaussian_estimating_equations"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    def transform(self, exposures: Any) -> QuantizationResult:
        return self._quantization.transform(exposures)

    def summary(self, digits: int = 6) -> str:
        """Display the saved joint contrast and cluster-robust uncertainty."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: Gaussian estimating equations\n"
                f"Joint mean difference: {fmt(self.psi)}; cluster-robust SE: {fmt(self.se)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI: ({fmt(self.ci[0])}, {fmt(self.ci[1])})\n"
                f"Two-sided Student t p-value: {fmt(self.p_value)}; df: {self.p_value_df}\n"
                f"Clusters: {self.cluster_count}; exposure cuts held fixed.\n"
                "Rows with the same cluster ID contribute together to robust uncertainty.\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'none'}; "
                "they do not change exposure cuts.\n"
                "The interval and t test use different reference distributions.")


def _levels(transformation: QuantizationResult, family: str = "Gaussian") -> np.ndarray:
    if transformation.mode == "prequantized":
        levels = np.unique(transformation.scores)
        if not 2 <= len(levels) < 10:
            raise ValueError(f"prequantized {family} EE requires 2 through 9 pooled score levels")
    elif transformation.mode == "explicit":
        if len(set(transformation.category_counts)) != 1:
            raise ValueError("joint interventions require equal exposure category counts")
        levels = np.arange(transformation.category_counts[0])
    else:
        levels = np.arange(transformation.q)
    if len(levels) < 2 or not np.array_equal(levels, np.arange(len(levels))):
        raise ValueError(f"{family} EE requires consecutive joint levels starting at zero")
    return _readonly(np.asarray(levels, dtype=float))


def _ee_weights(weights: Any, n: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Return saved positive case weights and the estimating-equation weights."""
    if weights is None:
        return None, np.ones(n)
    values = _real_array(weights, "weights")
    if values.ndim != 1 or values.size != n:
        raise ValueError("weights must be a one-dimensional vector with one value per row")
    if np.any(values <= 0):
        raise ValueError("weights must be strictly positive")
    return _readonly(values), values


def fit_gaussian_ee(
    y: Any, exposures: Any, cluster_ids: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None,
) -> GaussianEEResult:
    """Fit an additive Gaussian joint response with a cluster sandwich.

    Use complete numeric rows and integer cluster IDs. Exposure cuts are
    estimated once; no nonlinear terms or time structure are
    inferred. Positive ``weights`` affect both estimating equations and
    cluster scores, but not exposure cuts. At least two clusters and two
    joint levels are required.
    """
    transformation = quantize(exposures, q=q, breaks=breaks)
    levels = _levels(transformation)
    scores = transformation.scores
    n, d = scores.shape
    case_weights, w = _ee_weights(weights, n)
    outcome = _real_array(y, "y")
    if outcome.ndim != 1 or len(outcome) != n:
        raise ValueError("y must have one value per exposure row")
    ids = _real_array(cluster_ids, "cluster_ids")
    if (ids.ndim != 1 or len(ids) != n or np.any(ids != np.floor(ids))
            or np.any(np.abs(ids) >= 2**53)):
        raise ValueError("cluster_ids must contain one finite exact integer ID per row")
    _, inverse = np.unique(ids, return_inverse=True)
    cluster_count = int(np.max(inverse) + 1)
    if cluster_count < 2:
        raise ValueError("Gaussian EE requires at least two clusters")
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    c = z.shape[1]
    df = n - d - c - 2
    if df <= 0:
        raise ValueError("Gaussian EE Student t test requires positive reference degrees of freedom")

    point = fit(outcome, scores, None if c == 0 else z, q=None, alpha=alpha,
                weights=case_weights)
    x = np.column_stack((np.ones(n), scores, z))
    beta = point.coefficients
    p = x.shape[1]
    x_int = np.concatenate(
        [np.column_stack((np.ones(n), np.full((n, d), a), z)) for a in levels],
        axis=0)
    m = np.concatenate([np.tile([1.0, a], (n, 1)) for a in levels], axis=0)
    int_weights = np.tile(w, len(levels))
    theta = np.linalg.lstsq(m * np.sqrt(int_weights)[:, None],
                            (x_int @ beta) * np.sqrt(int_weights), rcond=None)[0]
    residuals = outcome - x @ beta
    marginal_residuals = x_int @ beta - m @ theta
    bread = np.zeros((p+2, p+2))
    bread[:p, :p] = -(x.T @ (w[:, None] * x))
    bread[p:, :p] = m.T @ (int_weights[:, None] * x_int)
    bread[p:, p:] = -(m.T @ (int_weights[:, None] * m))
    cluster_scores = np.empty((cluster_count, p+2))
    marginal_by_level = marginal_residuals.reshape((len(levels), n))
    for group in range(cluster_count):
        selected = inverse == group
        cluster_scores[group, :p] = x[selected].T @ (w[selected] * residuals[selected])
        cluster_scores[group, p:] = sum(
            m[a*n:(a+1)*n][selected].T @ (w[selected] * marginal_by_level[a, selected])
            for a in range(len(levels)))
    meat = cluster_scores.T @ cluster_scores
    try:
        inverse_bread = np.linalg.solve(bread, np.eye(p+2))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Gaussian EE estimating-equation bread is singular") from exc
    covariance = inverse_bread @ meat @ inverse_bread.T
    covariance = (covariance + covariance.T) / 2
    if not all(np.all(np.isfinite(value)) for value in
               (theta, bread, meat, covariance)):
        raise ValueError("Gaussian EE result exceeds finite double-precision arithmetic")
    msm_covariance = covariance[p:, p:]
    variance = float(msm_covariance[1, 1])
    if variance < 0:
        raise ValueError("Gaussian EE robust variance is negative")
    se = float(np.sqrt(variance))
    psi = float(theta[1])
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = (float(psi-critical*se), float(psi+critical*se))
    if se == 0:
        warnings.warn("The Gaussian EE joint contrast has zero robust variation; its test is undefined.",
                      QGCompWarning, stacklevel=2)
        statistic, p_value = float("nan"), float("nan")
    else:
        statistic = psi/se
        p_value = float(2*stats.t.sf(abs(statistic), df))
    means = np.average((x_int @ beta).reshape((len(levels), n)), axis=1, weights=w)
    return GaussianEEResult(
        conditional_coefficients=_readonly(beta),
        conditional_covariance=_readonly(covariance[:p, :p]),
        msm_coefficients=_readonly(theta), msm_covariance=_readonly(msm_covariance),
        full_covariance=_readonly(covariance), bread=_readonly(bread), meat=_readonly(meat),
        fitted_values=_readonly(x @ beta), intervention_levels=levels,
        intervention_means=_readonly(means), cluster_ids=_readonly(ids),
        case_weights=case_weights,
        cluster_count=cluster_count, psi=psi, var_psi=variance, se=se, ci=ci,
        test_statistic=statistic, p_value=p_value, p_value_df=df,
        alpha=alpha, _quantization=transformation,
    )
