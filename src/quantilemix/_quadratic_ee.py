# SPDX-License-Identifier: GPL-2.0-or-later
"""Degree-two Gaussian quantile g-computation with a cluster sandwich."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any
import warnings

import numpy as np
from scipy import special, stats

from ._core import QGCompWarning, QuantizationResult, _matrix, _readonly, _real_array, fit, quantize
from ._ee import _ee_weights
from ._quadratic import _levels


@dataclass(frozen=True, eq=False)
class QuadraticGaussianEEResult:
    """Polynomial joint-response coefficients with cluster-robust uncertainty."""

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
    psi: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci: np.ndarray = field(repr=False)
    test_statistic: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    p_value_df: int
    alpha: float
    pairwise_interactions: bool
    covariate_interactions: bool
    _quantization: QuantizationResult = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "gaussian"
    method: str = "clustered_quadratic_gaussian_estimating_equations"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    def transform(self, exposures: Any) -> QuantizationResult:
        return self._quantization.transform(exposures)

    def effect_at(self, level: int) -> float:
        """Fitted mean change when every score moves from level to level + 1."""
        if (isinstance(level, (bool, np.bool_)) or not isinstance(level, Integral)
                or level not in self.intervention_levels[:-1]):
            raise ValueError("level must identify two consecutive intervention levels")
        return float(self.psi[0] + self.psi[1] * (2 * int(level) + 1))

    def summary(self, digits: int = 6) -> str:
        """Display the two curve terms and their cluster-robust uncertainty."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: quadratic Gaussian estimating equations\n"
                f"Linear score coefficient: {fmt(self.psi[0])}; cluster-robust SE: {fmt(self.se[0])}\n"
                f"Quadratic score coefficient: {fmt(self.psi[1])}; cluster-robust SE: {fmt(self.se[1])}\n"
                f"Normal {(1-self.alpha)*100:g}% CIs: "
                f"({fmt(self.ci[0, 0])}, {fmt(self.ci[0, 1])}); "
                f"({fmt(self.ci[1, 0])}, {fmt(self.ci[1, 1])})\n"
                f"Two-sided Student t p-values (df {self.p_value_df}): "
                f"{fmt(self.p_value[0])}; {fmt(self.p_value[1])}\n"
                f"Clusters: {self.cluster_count}; exposure cuts held fixed.\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'none'}.\n"
                "The mean change depends on the starting score; use effect_at(level).")


def fit_quadratic_gaussian_ee(
    y: Any, exposures: Any, cluster_ids: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None, pairwise_interactions: bool = False,
    covariate_interactions: bool = False,
) -> QuadraticGaussianEEResult:
    """Fit a degree-two Gaussian joint-response curve with clustered uncertainty.

    Rows must be complete and numeric. Each exposure has a linear and squared
    score term. Optional products add every exposure pair or every exposure by
    numeric covariate. ``cluster_ids`` must contain exact integer IDs, with at
    least two clusters. Positive case weights enter both estimating equations
    and the cluster sandwich, but exposure cutpoints stay unweighted.

    The two entries of ``psi`` multiply the joint score and its square. Thus
    ``effect_at(0)`` gives the change from joint level zero to one; the change
    varies by starting level when the quadratic coefficient is nonzero.

    Parameters
    ----------
    y : array-like, shape (n_rows,)
        Continuous outcome.
    exposures : array-like, shape (n_rows, n_exposures)
        Exposure measurements or, with ``q=None``, integer scores.
    cluster_ids : array-like, shape (n_rows,)
        Exact integer cluster IDs. At least two clusters are required.
    covariates : array-like, optional
        Numeric adjustment columns with one row per outcome.
    q, breaks : optional
        Quantile count or explicit category boundaries. ``q=None`` accepts
        already scored exposures. At least three joint levels are required.
    alpha : float, default 0.05
        Error probability for two-sided confidence intervals.
    weights : array-like, optional
        Positive case weights. Exposure cutpoints remain unweighted.
    pairwise_interactions, covariate_interactions : bool, default False
        Add every exposure-pair or exposure-by-covariate product, respectively.

    Returns
    -------
    QuadraticGaussianEEResult
        ``psi`` contains the linear and quadratic marginal curve coefficients.

    Raises
    ------
    ValueError
        If input shapes, scores, cluster IDs, weights or design rank are invalid.

    Notes
    -----
    R qgcomp 2.19.6 agrees on the fixed point fits but may report a different
    nonlinear cluster covariance because its cluster-meat calculation reuses
    precomputed formula columns after changing exposure scores. This function
    rebuilds every derived column under each intervention. See the nonlinear
    estimating-equation validation report for the fixed comparison.
    """
    if not isinstance(pairwise_interactions, (bool, np.bool_)):
        raise ValueError("pairwise_interactions must be a Boolean")
    if not isinstance(covariate_interactions, (bool, np.bool_)):
        raise ValueError("covariate_interactions must be a Boolean")
    transformation = quantize(exposures, q=q, breaks=breaks)
    levels = _levels(transformation)
    scores = transformation.scores
    n, d = scores.shape
    if pairwise_interactions and d < 2:
        raise ValueError("pairwise interactions require at least two exposures")
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
        raise ValueError("quadratic Gaussian EE requires at least two clusters")
    if covariates is None:
        z = np.empty((n, 0))
    else:
        z, _ = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    if covariate_interactions and z.shape[1] == 0:
        raise ValueError("exposure-covariate interactions require covariates")
    case_weights, w = _ee_weights(weights, n)
    pairwise = [scores[:, i] * scores[:, j]
                for i in range(d) for j in range(i + 1, d)] if pairwise_interactions else []
    exposure_covariate = [scores[:, i] * z[:, j]
                          for i in range(d) for j in range(z.shape[1])]
    if not covariate_interactions:
        exposure_covariate = []
    extras = np.column_stack((scores * scores,
                              *[term[:, None] for term in pairwise],
                              *[term[:, None] for term in exposure_covariate], z))
    conditional = fit(outcome, scores, extras, q=None, alpha=alpha,
                      weights=case_weights)
    x = np.column_stack((np.ones(n), scores, extras))
    beta = conditional.coefficients
    p = len(beta)
    df = n-p-2
    if df <= 0:
        raise ValueError("quadratic Gaussian EE Student t test requires positive reference degrees of freedom")
    # Each block sets every exposure score to the same joint intervention level.
    blocks = []
    for level in levels:
        fixed = np.full_like(scores, level)
        fixed_pairwise = [fixed[:, i] * fixed[:, j]
                          for i in range(d) for j in range(i + 1, d)] if pairwise_interactions else []
        fixed_covariate = [fixed[:, i] * z[:, j]
                           for i in range(d) for j in range(z.shape[1])]
        if not covariate_interactions:
            fixed_covariate = []
        blocks.append(np.column_stack((np.ones(n), fixed, fixed * fixed,
                                       *[term[:, None] for term in fixed_pairwise],
                                       *[term[:, None] for term in fixed_covariate], z)))
    x_int = np.concatenate(blocks)
    m = np.concatenate([np.tile([1.0, level, level * level], (n, 1))
                        for level in levels])
    int_weights = np.tile(w, len(levels))
    predicted = x_int @ beta
    theta = np.linalg.lstsq(m * np.sqrt(int_weights)[:, None],
                            predicted * np.sqrt(int_weights), rcond=None)[0]
    residuals = outcome-x @ beta
    marginal_residuals = (predicted-m @ theta).reshape((len(levels), n))
    bread = np.zeros((p+3, p+3))
    bread[:p, :p] = -(x.T @ (w[:, None] * x))
    bread[p:, :p] = m.T @ (int_weights[:, None] * x_int)
    bread[p:, p:] = -(m.T @ (int_weights[:, None] * m))
    cluster_scores = np.empty((cluster_count, p+3))
    for group in range(cluster_count):
        selected = inverse == group
        cluster_scores[group, :p] = x[selected].T @ (w[selected] * residuals[selected])
        cluster_scores[group, p:] = sum(
            m[a*n:(a+1)*n][selected].T @ (w[selected] * marginal_residuals[a, selected])
            for a in range(len(levels)))
    meat = cluster_scores.T @ cluster_scores
    try:
        inverse_bread = np.linalg.solve(bread, np.eye(p+3))
    except np.linalg.LinAlgError as exc:
        raise ValueError("quadratic Gaussian EE estimating-equation bread is singular") from exc
    full_covariance = inverse_bread @ meat @ inverse_bread.T
    full_covariance = (full_covariance+full_covariance.T)/2
    if not all(np.all(np.isfinite(value)) for value in
               (theta, bread, meat, full_covariance)):
        raise ValueError("quadratic Gaussian EE result exceeds finite double-precision arithmetic")
    msm_covariance = full_covariance[p:, p:]
    covariance = msm_covariance[1:, 1:]
    variance = np.diag(covariance)
    if np.any(variance < 0):
        raise ValueError("quadratic Gaussian EE robust variance is negative")
    se = np.sqrt(variance)
    psi = theta[1:]
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2.0)))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    statistic = np.divide(psi, se, out=np.full(2, np.nan), where=se > 0)
    p_value = 2*stats.t.sf(np.abs(statistic), df)
    if np.any(se == 0):
        warnings.warn("A quadratic Gaussian EE curve term has zero robust variation; its test is undefined.",
                      QGCompWarning, stacklevel=2)
    means = np.average(predicted.reshape((len(levels), n)), axis=1, weights=w)
    return QuadraticGaussianEEResult(
        conditional_coefficients=_readonly(beta),
        conditional_covariance=_readonly(full_covariance[:p, :p]),
        msm_coefficients=_readonly(theta), msm_covariance=_readonly(msm_covariance),
        full_covariance=_readonly(full_covariance), bread=_readonly(bread),
        meat=_readonly(meat), fitted_values=_readonly(x @ beta),
        intervention_levels=levels, intervention_means=_readonly(means),
        cluster_ids=_readonly(ids), case_weights=case_weights,
        cluster_count=cluster_count, psi=_readonly(psi),
        covariance=_readonly(covariance), se=_readonly(se), ci=_readonly(ci),
        test_statistic=_readonly(statistic), p_value=_readonly(p_value),
        p_value_df=df, alpha=alpha,
        pairwise_interactions=bool(pairwise_interactions),
        covariate_interactions=bool(covariate_interactions),
        _quantization=transformation,
    )
