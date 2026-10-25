# SPDX-License-Identifier: GPL-2.0-or-later
"""Conditional negative-binomial hurdle with a logistic nonzero component."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from scipy import optimize, special

from ._binomial import _solve_logit
from ._core import _column_names, _matrix, _readonly, _real_array, quantize
from ._hurdle import HurdlePoissonResult


def _positive_negbin_likelihood_score(design: np.ndarray, y: np.ndarray,
                                      parameters: np.ndarray) -> tuple[float, np.ndarray]:
    """Zero-truncated NB2 log likelihood and score for beta, log(theta)."""
    beta, rho = parameters[:-1], parameters[-1]
    eta = design @ beta
    if np.max(eta) > 20 or np.min(eta) < -25 or not -12 < rho < 12:
        return -np.inf, np.full(len(parameters), np.nan)
    mu, theta = np.exp(eta), np.exp(rho)
    log_denom = np.log(theta+mu)
    log_p0 = theta*(rho-log_denom)
    positive_probability = -np.expm1(log_p0)
    if np.any(positive_probability <= 0):
        return -np.inf, np.full(len(parameters), np.nan)
    log_pmf = (special.gammaln(y+theta)-special.gammaln(theta)
               -special.gammaln(y+1)+theta*(rho-log_denom)
               +y*(eta-log_denom))
    likelihood = float(np.sum(log_pmf-np.log(positive_probability)))
    ratio = np.exp(log_p0)/positive_probability
    score_eta = (y-mu*(y+theta)/(mu+theta)
                 -ratio*theta*mu/(mu+theta))
    score_rho = theta*np.sum(
        special.digamma(y+theta)-special.digamma(theta)
        +rho-log_denom+1-(y+theta)/(mu+theta)
        +ratio*(rho-log_denom+1-theta/(mu+theta)))
    return likelihood, np.r_[design.T @ score_eta, score_rho]


def _observed_information(design: np.ndarray, y: np.ndarray,
                          parameters: np.ndarray) -> np.ndarray:
    """Central derivative of the analytic score, including dispersion."""
    k = len(parameters)
    information = np.empty((k, k))
    for j in range(k):
        step = 1e-5*max(1.0, abs(parameters[j]))
        plus, minus = parameters.copy(), parameters.copy()
        plus[j] += step
        minus[j] -= step
        information[:, j] = -(_positive_negbin_likelihood_score(design, y, plus)[1]
                               -_positive_negbin_likelihood_score(design, y, minus)[1])/(2*step)
    return (information+information.T)/2


@dataclass(frozen=True, eq=False)
class HurdleNegativeBinomialResult(HurdlePoissonResult):
    """Conditional hurdle effects with an estimated NB2 count dispersion."""

    theta: float = float("nan")
    se_logtheta: float = float("nan")
    count_covariance_full: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    family: str = "hurdle_negative_binomial"

    def predict(self, exposures: Any, covariates: Any = None, *,
                zero_covariates: Any = "same") -> np.ndarray:
        """Expected observed count after negative-binomial zero truncation."""
        mu, p_nonzero = self.predict_components(exposures, covariates,
                                                zero_covariates=zero_covariates)
        log_p0 = -self.theta*np.log1p(mu/self.theta)
        return _readonly(p_nonzero*mu/(-np.expm1(log_p0)))

    def summary(self, digits: int = 6) -> str:
        """Display separate count and nonzero effects with NB dispersion."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        count_line = (
            f"Positive-count component: log underlying-mean ratio {fmt(self.psi_count)}; "
            f"SE {fmt(self.se_count)}; normal {(1-self.alpha)*100:g}% CI "
            f"({fmt(self.ci_count[0])}, {fmt(self.ci_count[1])}); z p-value {fmt(self.p_value_count)}.\n"
            if self.psi_count is not None else "Positive-count component: no joint exposure effect fitted.\n")
        zero_line = (
            f"Nonzero hurdle component: log odds ratio {fmt(self.psi_nonzero)}; "
            f"SE {fmt(self.se_nonzero)}; normal {(1-self.alpha)*100:g}% CI "
            f"({fmt(self.ci_nonzero[0])}, {fmt(self.ci_nonzero[1])}); z p-value {fmt(self.p_value_nonzero)}.\n"
            if self.psi_nonzero is not None else "Nonzero hurdle component: no joint exposure effect fitted.\n")
        return ("Quantile g-computation: conditional negative-binomial hurdle\n"
                "Joint score change raises every included exposure by one; covariates fixed.\n"
                + count_line + zero_line
                + f"Count dispersion theta: {fmt(self.theta)}; SE of log theta {fmt(self.se_logtheta)}.\n"
                "The component slopes are not an observed-count ratio.")


def fit_hurdle_negative_binomial(
    y: Any, exposures: Any, covariates: Any = None, *,
    zero_covariates: Any = "same", exposure_component: str = "both",
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
) -> HurdleNegativeBinomialResult:
    """Fit zero-truncated NB2 counts and a separate nonzero logit.

    The count size ``theta`` is estimated jointly with its regression terms;
    variance before truncation is ``mu + mu**2/theta``. All exposure scores
    enter the selected count, nonzero, or both components. ``None`` in
    ``zero_covariates`` removes nonzero adjustment; omission reuses count
    adjustment. Inputs must be complete numeric arrays with independent rows.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if exposure_component not in ("both", "count", "zero"):
        raise ValueError("exposure_component must be 'both', 'count', or 'zero'")
    outcome = _real_array(y, "y")
    if outcome.ndim != 1 or outcome.size == 0 or np.any(outcome < 0) or np.any(outcome != np.floor(outcome)):
        raise ValueError("y must be a nonempty vector of nonnegative integer counts")
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if len(outcome) != n:
        raise ValueError("y and exposures must have the same rows")
    positive = outcome > 0
    if not np.any(~positive) or not np.any(positive):
        raise ValueError("hurdle fit needs zero and positive outcomes")
    if covariates is None:
        z, columns = np.empty((n, 0)), None
    else:
        z, columns = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same rows")
    names = _column_names(columns, z.shape[1], "z")
    if isinstance(zero_covariates, str):
        if zero_covariates != "same":
            raise ValueError("zero_covariates must be 'same', None, or complete numeric columns")
        zz, zero_columns, shared = z, columns, True
    elif zero_covariates is None:
        zz, zero_columns, shared = np.empty((n, 0)), None, False
    else:
        zz, zero_columns = _matrix(zero_covariates, "zero_covariates")
        if len(zz) != n:
            raise ValueError("zero_covariates and exposures must have the same rows")
        shared = False
    zero_names = _column_names(zero_columns, zz.shape[1], "zz")
    count_exposure_names = transformation.feature_names if exposure_component != "zero" else ()
    zero_exposure_names = transformation.feature_names if exposure_component != "count" else ()
    coefficient_names = ("(Intercept)",) + count_exposure_names + names
    zero_coefficient_names = ("(Intercept)",) + zero_exposure_names + zero_names
    if (len(set(coefficient_names)) != len(coefficient_names)
            or len(set(zero_coefficient_names)) != len(zero_coefficient_names)):
        raise ValueError("coefficient identifiers must be distinct")
    count_scores = transformation.scores if exposure_component != "zero" else np.empty((n, 0))
    zero_scores = transformation.scores if exposure_component != "count" else np.empty((n, 0))
    count_design = np.column_stack((np.ones(n), count_scores, z))
    zero_design = np.column_stack((np.ones(n), zero_scores, zz))
    p, p_zero = count_design.shape[1], zero_design.shape[1]
    if (np.sum(positive) <= p+1 or np.linalg.matrix_rank(count_design[positive]) != p
            or np.linalg.matrix_rank(zero_design) != p_zero):
        raise ValueError("hurdle component designs need full rank and enough positive rows")
    zero_beta, zero_covariance, zero_iterations = _solve_logit(
        zero_design, positive.astype(float), np.ones(n))
    count_positive, positive_y = count_design[positive], outcome[positive]
    start = np.zeros(p+1)
    start[0] = np.log(float(np.mean(positive_y)))
    def objective(parameters: np.ndarray):
        likelihood, score = _positive_negbin_likelihood_score(count_positive, positive_y, parameters)
        if not np.isfinite(likelihood):
            return 1e100, np.zeros_like(parameters)
        return -likelihood, -score
    fit = optimize.minimize(objective, start, jac=True, method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 1000})
    count_parameters = fit.x
    likelihood, score = _positive_negbin_likelihood_score(count_positive, positive_y, count_parameters)
    if (np.all(np.isfinite(count_parameters)) and np.isfinite(likelihood)
            and np.all(np.isfinite(score)) and np.max(np.abs(score)) > 1e-5):
        def score_equations(candidate: np.ndarray):
            return _positive_negbin_likelihood_score(count_positive, positive_y, candidate)[1]
        def score_jacobian(candidate: np.ndarray):
            return -_observed_information(count_positive, positive_y, candidate)
        refined = optimize.root(score_equations, count_parameters,
                                jac=score_jacobian, method="hybr")
        if np.all(np.isfinite(refined.x)):
            refined_likelihood, refined_score = _positive_negbin_likelihood_score(
                count_positive, positive_y, refined.x)
            if (np.isfinite(refined_likelihood) and refined_likelihood >= likelihood-1e-7
                    and np.all(np.isfinite(refined_score))
                    and np.max(np.abs(refined_score)) < np.max(np.abs(score))):
                count_parameters, likelihood, score = refined.x, refined_likelihood, refined_score
    if not np.isfinite(likelihood) or not np.all(np.isfinite(score)) or np.max(np.abs(score)) > 1e-5:
        raise ValueError("hurdle negative-binomial count fit did not converge")
    information = _observed_information(count_positive, positive_y, count_parameters)
    if not np.all(np.isfinite(information)) or np.linalg.eigvalsh(information)[0] <= 1e-8:
        raise ValueError("hurdle negative-binomial information is singular")
    count_covariance_full = np.linalg.solve(information, np.eye(p+1))
    count_covariance_full = (count_covariance_full+count_covariance_full.T)/2
    covariance = np.zeros((p+p_zero, p+p_zero))
    covariance[:p, :p] = count_covariance_full[:p, :p]
    covariance[p:, p:] = zero_covariance
    beta = np.r_[count_parameters[:p], zero_beta]
    contrast = np.zeros((2, p+p_zero))
    if exposure_component != "zero":
        contrast[0, 1:1+d] = 1
    if exposure_component != "count":
        contrast[1, p+1:p+1+d] = 1
    psi = contrast @ beta
    se = np.sqrt(np.diag(contrast @ covariance @ contrast.T))
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2)))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    z_statistic = np.divide(psi, se, out=np.zeros_like(psi), where=se > 0)
    p_value = 2*special.ndtr(-np.abs(z_statistic))
    def component_value(index: int, absent: bool):
        if absent:
            return None, None, None, None
        return float(psi[index]), float(se[index]), tuple(ci[index]), float(p_value[index])
    psi_c, se_c, ci_c, pv_c = component_value(0, exposure_component == "zero")
    psi_z, se_z, ci_z, pv_z = component_value(1, exposure_component == "count")
    return HurdleNegativeBinomialResult(
        _readonly(beta), _readonly(covariance), coefficient_names,
        transformation.feature_names, names, zero_names, zero_coefficient_names,
        psi_c, psi_z, se_c, se_z, ci_c, ci_z, pv_c, pv_z,
        float(alpha), int(fit.nit+zero_iterations), transformation,
        columns, zero_columns, shared, exposure_component,
        theta=float(np.exp(count_parameters[-1])),
        se_logtheta=float(np.sqrt(count_covariance_full[-1, -1])),
        count_covariance_full=_readonly(count_covariance_full),
    )
