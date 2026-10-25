# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted clustered binary quantile g-computation estimating equations.

Reference: R qgcomp 2.19.6, GPL (>= 2). The marginal odds and risk links
have different estimating functions and cluster-sandwich derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special

from ._binomial import _solve_logit, fit_binomial
from ._core import QuantizationResult, _matrix, _readonly, _real_array, quantize
from ._ee import _ee_weights, _levels
from ._risk_ratio import _log_binomial_msm


@dataclass(frozen=True, eq=False)
class BinomialEEResult:
    """Marginal binary ratio with cluster-robust uncertainty."""

    conditional_coefficients: np.ndarray = field(repr=False)
    conditional_covariance: np.ndarray = field(repr=False)
    msm_coefficients: np.ndarray = field(repr=False)
    msm_covariance: np.ndarray = field(repr=False)
    full_covariance: np.ndarray = field(repr=False)
    bread: np.ndarray = field(repr=False)
    meat: np.ndarray = field(repr=False)
    fitted_values: np.ndarray = field(repr=False)
    intervention_levels: np.ndarray = field(repr=False)
    intervention_risks: np.ndarray = field(repr=False)
    msm_fitted_risks: np.ndarray = field(repr=False)
    cluster_ids: np.ndarray = field(repr=False)
    case_weights: np.ndarray | None = field(repr=False)
    cluster_count: int
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    rr: bool
    alpha: float
    _quantization: QuantizationResult = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "binomial"
    method: str = "clustered_binomial_estimating_equations"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def link(self) -> str:
        return "log" if self.rr else "logit"

    @property
    def effect_ratio(self) -> float:
        """Marginal risk ratio or odds ratio for one joint score increase."""
        return float(np.exp(self.psi))

    @property
    def effect_ratio_ci(self) -> tuple[float, float]:
        return (float(np.exp(self.ci[0])), float(np.exp(self.ci[1])))

    def transform(self, exposures: Any) -> QuantizationResult:
        return self._quantization.transform(exposures)

    def summary(self, digits: int = 6) -> str:
        """Display the marginal ratio and cluster-robust uncertainty."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        name = "risk" if self.rr else "odds"
        return ("Quantile g-computation: binomial estimating equations\n"
                f"Marginal log {name} ratio: {fmt(self.psi)}; cluster-robust SE: {fmt(self.se)}\n"
                f"{name.capitalize()} ratio: {fmt(self.effect_ratio)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI on log scale: "
                f"({fmt(self.ci[0])}, {fmt(self.ci[1])})\n"
                f"Two-sided normal z p-value: {fmt(self.p_value)}\n"
                f"Clusters: {self.cluster_count}; exposure cuts held fixed.\n"
                "Rows sharing a cluster ID contribute together to robust uncertainty.\n"
                "With more than two levels, the ratio is a fitted marginal-model slope.\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'none'}. "
                "Offsets and time structure are not supported.")


def fit_binomial_ee(
    y: Any, exposures: Any, cluster_ids: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05, rr: bool = True,
    weights: Any = None,
) -> BinomialEEResult:
    """Fit a marginal binary ratio with a cluster-summed sandwich covariance.

    ``rr=True`` uses a log-risk marginal model; ``rr=False`` uses a logit
    marginal model. The conditional fit is logistic in both cases. Inputs
    must be complete numeric rows with binary outcomes and integer IDs.
    Positive ``weights`` affect both estimating equations and cluster scores
    but not exposure cuts.
    """
    if not isinstance(rr, (bool, np.bool_)):
        raise ValueError("rr must be a Boolean")
    transformation = quantize(exposures, q=q, breaks=breaks)
    levels = _levels(transformation, "Binomial")
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
        raise ValueError("Binomial EE requires at least two clusters")
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    point = fit_binomial(outcome, scores, None if z.shape[1] == 0 else z,
                         q=None, alpha=alpha, weights=case_weights)
    x = np.column_stack((np.ones(n), scores, z))
    beta = np.asarray(point.coefficients, dtype=float)
    p = x.shape[1]
    x_int = np.concatenate(
        [np.column_stack((np.ones(n), np.full((n, d), a), z)) for a in levels],
        axis=0)
    m = np.concatenate([np.tile([1.0, a], (n, 1)) for a in levels], axis=0)
    int_weights = np.tile(w, len(levels))
    predicted_int = special.expit(x_int @ beta)
    risks = np.average(predicted_int.reshape((len(levels), n)), axis=1, weights=w)
    if not np.all((risks > 0) & (risks < 1)):
        raise ValueError("Binomial EE standardized risks must lie strictly between zero and one")
    if rr:
        theta = _log_binomial_msm(risks, levels)
        marginal_fit = np.exp(m @ theta)
        if not np.all((marginal_fit > 0) & (marginal_fit < 1)):
            raise ValueError("Binomial EE log-risk fitted values must lie strictly between zero and one")
        marginal_residuals = (predicted_int - marginal_fit) / (1-marginal_fit)
        lower_left_weights = predicted_int*(1-predicted_int)/(1-marginal_fit)
        lower_right_weights = marginal_fit*(1-predicted_int)/(1-marginal_fit)**2
    else:
        theta, _, _ = _solve_logit(m, predicted_int, int_weights)
        marginal_fit = special.expit(m @ theta)
        marginal_residuals = predicted_int - marginal_fit
        lower_left_weights = predicted_int*(1-predicted_int)
        lower_right_weights = marginal_fit*(1-marginal_fit)
    bread = np.zeros((p+2, p+2))
    conditional_weight = point.fitted_values*(1-point.fitted_values)
    bread[:p, :p] = -(x.T @ ((w*conditional_weight)[:, None] * x))
    bread[p:, :p] = m.T @ ((int_weights*lower_left_weights)[:, None] * x_int)
    bread[p:, p:] = -(m.T @ ((int_weights*lower_right_weights)[:, None] * m))
    cluster_scores = np.empty((cluster_count, p+2))
    marginal_by_level = marginal_residuals.reshape((len(levels), n))
    residuals = outcome - point.fitted_values
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
        raise ValueError("Binomial EE estimating-equation bread is singular") from exc
    covariance = inverse_bread @ meat @ inverse_bread.T
    covariance = (covariance + covariance.T) / 2
    if not all(np.all(np.isfinite(value)) for value in
               (theta, bread, meat, covariance)):
        raise ValueError("Binomial EE result exceeds finite double-precision arithmetic")
    msm_covariance = covariance[p:, p:]
    variance = float(msm_covariance[1, 1])
    if variance <= 0:
        raise ValueError("Binomial EE robust joint variance must be positive")
    se = float(np.sqrt(variance))
    psi = float(theta[1])
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = (float(psi - critical*se), float(psi + critical*se))
    if not np.all(np.isfinite(ci)):
        raise ValueError("Binomial EE confidence limits are nonfinite")
    statistic = psi/se
    p_value = float(2 * special.ndtr(-abs(statistic)))
    return BinomialEEResult(
        conditional_coefficients=_readonly(beta),
        conditional_covariance=_readonly(covariance[:p, :p]),
        msm_coefficients=_readonly(theta), msm_covariance=_readonly(msm_covariance),
        full_covariance=_readonly(covariance), bread=_readonly(bread), meat=_readonly(meat),
        fitted_values=_readonly(point.fitted_values), intervention_levels=levels,
        intervention_risks=_readonly(risks),
        msm_fitted_risks=_readonly(marginal_fit.reshape((len(levels), n))[:, 0]),
        cluster_ids=_readonly(ids), case_weights=case_weights,
        cluster_count=cluster_count, psi=psi,
        var_psi=variance, se=se, ci=ci, test_statistic=statistic,
        p_value=p_value, rr=bool(rr), alpha=alpha, _quantization=transformation,
    )
