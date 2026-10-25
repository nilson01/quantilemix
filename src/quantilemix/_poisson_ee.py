# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted clustered Poisson quantile g-computation estimating equations.

Reference: R qgcomp 2.19.6, GPL (>= 2). The cluster sandwich sums the
conditional and marginal estimating functions within each cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special

from ._core import QuantizationResult, _matrix, _readonly, _real_array, quantize
from ._ee import _ee_weights, _levels
from ._poisson import _positive_time, fit_poisson


@dataclass(frozen=True, eq=False)
class PoissonEEResult:
    """Marginal log mean ratio with cluster-robust uncertainty."""

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
    exposure_time: np.ndarray | None = field(repr=False)
    cluster_count: int
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    alpha: float
    _quantization: QuantizationResult = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "poisson"
    method: str = "clustered_poisson_estimating_equations"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def mean_ratio(self) -> float:
        """Marginal mean or rate ratio for one joint score increase."""
        return float(np.exp(self.psi))

    @property
    def rate_ratio(self) -> float:
        """Rate ratio when observation times are supplied and held fixed."""
        return self.mean_ratio

    @property
    def mean_ratio_ci(self) -> tuple[float, float]:
        return (float(np.exp(self.ci[0])), float(np.exp(self.ci[1])))

    def transform(self, exposures: Any) -> QuantizationResult:
        return self._quantization.transform(exposures)

    def summary(self, digits: int = 6) -> str:
        """Display the marginal log mean ratio and cluster-robust uncertainty."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        offset_note = (
            "Observation-time offset: supplied; its values are held fixed "
            "at each joint intervention.\n"
            "This is a rate ratio for the observed time distribution; "
            "time-varying effects and lag structures are not supported."
            if self.exposure_time is not None else
            "This is an expected-count ratio for equal observation periods; "
            "time-varying effects and lag structures are not supported."
        )
        return ("Quantile g-computation: Poisson estimating equations\n"
                f"Marginal log mean ratio: {fmt(self.psi)}; cluster-robust SE: {fmt(self.se)}\n"
                f"{'Rate' if self.exposure_time is not None else 'Mean'} ratio: "
                f"{fmt(self.mean_ratio)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI on log scale: "
                f"({fmt(self.ci[0])}, {fmt(self.ci[1])})\n"
                f"Two-sided normal z p-value: {fmt(self.p_value)}\n"
                f"Clusters: {self.cluster_count}; exposure cuts held fixed.\n"
                "Rows sharing a cluster ID contribute together to robust uncertainty.\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'none'}; "
                "they do not change exposure cuts.\n"
                + offset_note)


def fit_poisson_ee(
    y: Any, exposures: Any, cluster_ids: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None, exposure_time: Any = None,
) -> PoissonEEResult:
    """Fit an additive Poisson-log marginal curve with a cluster sandwich.

    Inputs must be complete numeric rows, nonnegative integer counts and
    exact integer cluster IDs. Positive ``weights`` affect both estimating
    equations and cluster scores but not exposure cuts. Positive
    ``exposure_time`` supplies a log offset for unequal observation periods,
    held fixed under interventions. Nonlinear terms and lag structures are
    outside this subset.
    """
    transformation = quantize(exposures, q=q, breaks=breaks)
    levels = _levels(transformation, "Poisson")
    scores = transformation.scores
    n, d = scores.shape
    case_weights, w = _ee_weights(weights, n)
    time = None if exposure_time is None else _positive_time(exposure_time, n)
    fixed_time = np.ones(n) if time is None else time
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
        raise ValueError("Poisson EE requires at least two clusters")
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    point = fit_poisson(outcome, scores, None if z.shape[1] == 0 else z,
                        q=None, alpha=alpha, weights=case_weights,
                        exposure_time=time)
    x = np.column_stack((np.ones(n), scores, z))
    beta = np.asarray(point.coefficients, dtype=float)
    p = x.shape[1]
    x_int = np.concatenate(
        [np.column_stack((np.ones(n), np.full((n, d), a), z)) for a in levels],
        axis=0)
    m = np.concatenate([np.tile([1.0, a], (n, 1)) for a in levels], axis=0)
    int_weights = np.tile(w, len(levels))
    int_time = np.tile(fixed_time, len(levels))
    with np.errstate(over="ignore", invalid="ignore"):
        predicted_int = int_time * np.exp(x_int @ beta)
    if not np.all(np.isfinite(predicted_int)):
        raise ValueError("Poisson EE intervention means exceed the finite range")
    means = np.average(predicted_int.reshape((len(levels), n)), axis=1, weights=w)
    if np.any(means <= 0) or not np.all(np.isfinite(means)):
        raise ValueError("Poisson EE marginal means are nonpositive or nonfinite")
    weighted_time = float(np.average(fixed_time, weights=w))
    theta = np.array([np.log(means[0] / weighted_time),
                      float(np.sum(beta[1:d+1]))])
    with np.errstate(over="ignore", invalid="ignore"):
        marginal_fit = int_time * np.exp(m @ theta)
    if not np.all(np.isfinite(marginal_fit)):
        raise ValueError("Poisson EE fitted marginal means exceed the finite range")
    residuals = outcome - point.fitted_values
    marginal_residuals = predicted_int - marginal_fit
    bread = np.zeros((p+2, p+2))
    bread[:p, :p] = -(x.T @ ((w*point.fitted_values)[:, None] * x))
    bread[p:, :p] = m.T @ ((int_weights*predicted_int)[:, None] * x_int)
    bread[p:, p:] = -(m.T @ ((int_weights*marginal_fit)[:, None] * m))
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
        raise ValueError("Poisson EE estimating-equation bread is singular") from exc
    covariance = inverse_bread @ meat @ inverse_bread.T
    covariance = (covariance + covariance.T) / 2
    if not all(np.all(np.isfinite(value)) for value in
               (theta, bread, meat, covariance)):
        raise ValueError("Poisson EE result exceeds finite double-precision arithmetic")
    msm_covariance = covariance[p:, p:]
    variance = float(msm_covariance[1, 1])
    if variance <= 0:
        raise ValueError("Poisson EE robust joint variance must be positive")
    se = float(np.sqrt(variance))
    psi = float(theta[1])
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = (float(psi - critical*se), float(psi + critical*se))
    if not np.all(np.isfinite(ci)):
        raise ValueError("Poisson EE confidence limits are nonfinite")
    statistic = psi / se
    p_value = float(2 * special.ndtr(-abs(statistic)))
    return PoissonEEResult(
        conditional_coefficients=_readonly(beta),
        conditional_covariance=_readonly(covariance[:p, :p]),
        msm_coefficients=_readonly(theta), msm_covariance=_readonly(msm_covariance),
        full_covariance=_readonly(covariance), bread=_readonly(bread), meat=_readonly(meat),
        fitted_values=_readonly(point.fitted_values), intervention_levels=levels,
        intervention_means=_readonly(means), cluster_ids=_readonly(ids),
        case_weights=case_weights,
        exposure_time=None if time is None else _readonly(time),
        cluster_count=cluster_count, psi=psi, var_psi=variance, se=se, ci=ci,
        test_statistic=statistic, p_value=p_value, alpha=alpha,
        _quantization=transformation,
    )
