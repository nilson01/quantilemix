# SPDX-License-Identifier: GPL-2.0-or-later
"""Conditional zero-inflated NB2 mixture with estimated dispersion."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from scipy import optimize, special

from ._core import _column_names, _matrix, _readonly, _real_array, quantize
from ._zero_inflated import ZeroInflatedPoissonResult


def _negbin_mixture_likelihood_score(count: np.ndarray, zero: np.ndarray,
                                     y: np.ndarray, parameters: np.ndarray):
    """Log likelihood and analytic score for count, zero, then log size."""
    p = count.shape[1]
    eta_c, eta_z, rho = count @ parameters[:p], zero @ parameters[p:-1], parameters[-1]
    if np.max(eta_c) > 20 or np.min(eta_c) < -25 or not -12 < rho < 12:
        return -np.inf, np.full(len(parameters), np.nan)
    mu, theta = np.exp(eta_c), np.exp(rho)
    log_denom = np.log(theta+mu)
    log_p0 = theta*(rho-log_denom)
    log_pi, log_one_minus_pi = special.log_expit(eta_z), special.log_expit(-eta_z)
    log_zero_mix = np.logaddexp(log_pi, log_one_minus_pi+log_p0)
    log_count = (special.gammaln(y+theta)-special.gammaln(theta)
                 -special.gammaln(y+1)+theta*(rho-log_denom)
                 +y*(eta_c-log_denom))
    positive = y > 0
    likelihood = float(np.sum(np.where(positive, log_one_minus_pi+log_count,
                                       log_zero_mix)))
    pi = special.expit(eta_z)
    structural_posterior = np.exp(log_pi-log_zero_mix)
    count_posterior = np.exp(log_one_minus_pi+log_p0-log_zero_mix)
    score_count = np.where(positive, y-mu*(y+theta)/(mu+theta),
                           -count_posterior*theta*mu/(mu+theta))
    score_zero = np.where(positive, -pi, structural_posterior-pi)
    score_rho_positive = theta*(
        special.digamma(y+theta)-special.digamma(theta)
        +rho-log_denom+1-(y+theta)/(mu+theta))
    score_rho_zero = count_posterior*theta*(rho-log_denom+1-theta/(mu+theta))
    score_rho = float(np.sum(np.where(positive, score_rho_positive, score_rho_zero)))
    return likelihood, np.r_[count.T @ score_count, zero.T @ score_zero, score_rho]


def _negbin_mixture_information(count: np.ndarray, zero: np.ndarray,
                                y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    k = len(parameters)
    information = np.empty((k, k))
    for j in range(k):
        step = 1e-5*max(1.0, abs(parameters[j]))
        plus, minus = parameters.copy(), parameters.copy()
        plus[j] += step
        minus[j] -= step
        information[:, j] = -(_negbin_mixture_likelihood_score(count, zero, y, plus)[1]
                               -_negbin_mixture_likelihood_score(count, zero, y, minus)[1])/(2*step)
    return (information+information.T)/2


@dataclass(frozen=True, eq=False)
class ZeroInflatedNegativeBinomialResult(ZeroInflatedPoissonResult):
    """Conditional NB2 count and structural-zero component effects."""

    theta: float = float("nan")
    se_logtheta: float = float("nan")
    covariance_full: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    family: str = "zero_inflated_negative_binomial"

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        count_line = (
            f"Negative-binomial count component: log mean ratio {fmt(self.psi_count)}; "
            f"SE {fmt(self.se_count)}; normal {(1-self.alpha)*100:g}% CI "
            f"({fmt(self.ci_count[0])}, {fmt(self.ci_count[1])}); z p-value {fmt(self.p_value_count)}.\n"
            if self.psi_count is not None else "Negative-binomial count component: no joint exposure effect fitted.\n")
        zero_line = (
            f"Structural-zero component: log odds ratio {fmt(self.psi_zero)}; "
            f"SE {fmt(self.se_zero)}; normal {(1-self.alpha)*100:g}% CI "
            f"({fmt(self.ci_zero[0])}, {fmt(self.ci_zero[1])}); z p-value {fmt(self.p_value_zero)}.\n"
            if self.psi_zero is not None else "Structural-zero component: no joint exposure effect fitted.\n")
        return ("Quantile g-computation: conditional zero-inflated negative binomial\n"
                "Joint score change raises every included exposure by one; covariates fixed.\n"
                + count_line + zero_line
                + f"Count dispersion theta: {fmt(self.theta)}; SE of log theta {fmt(self.se_logtheta)}.\n"
                "The component slopes are not an observed-count ratio.")


def fit_zero_inflated_negative_binomial(
    y: Any, exposures: Any, covariates: Any = None, *,
    zero_covariates: Any = "same", exposure_component: str = "both",
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
) -> ZeroInflatedNegativeBinomialResult:
    """Fit an NB2 count process plus separate structural-zero logit.

    Inputs are complete numeric arrays of independent rows. The count size
    is estimated jointly with both regression components. The full exposure
    set enters either or both components; absent effects are returned as
    ``None``. Omitted zero adjustment reuses count adjustment.
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
    if not np.any(outcome == 0) or not np.any(outcome > 0):
        raise ValueError("zero-inflated fit needs zero and positive outcomes")
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
    if (n <= p+p_zero+1 or np.linalg.matrix_rank(count_design) != p
            or np.linalg.matrix_rank(zero_design) != p_zero):
        raise ValueError("zero-inflated component designs need full rank and more rows than parameters")
    start = np.zeros(p+p_zero+1)
    start[0] = np.log(float(np.mean(outcome[outcome > 0])))
    start[p] = -1.0
    def objective(parameters: np.ndarray):
        likelihood, score = _negbin_mixture_likelihood_score(
            count_design, zero_design, outcome, parameters)
        if not np.isfinite(likelihood):
            return 1e100, np.zeros_like(parameters)
        return -likelihood, -score
    fit = optimize.minimize(objective, start, jac=True, method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 1000})
    parameters = fit.x
    likelihood, score = _negbin_mixture_likelihood_score(count_design, zero_design, outcome, parameters)
    if not np.isfinite(likelihood) or not np.all(np.isfinite(score)) or np.max(np.abs(score)) > 1e-5:
        raise ValueError("zero-inflated negative-binomial fit did not converge")
    information = _negbin_mixture_information(count_design, zero_design, outcome, parameters)
    if not np.all(np.isfinite(information)) or np.linalg.eigvalsh(information)[0] <= 1e-8:
        raise ValueError("zero-inflated negative-binomial information is singular")
    covariance_full = np.linalg.solve(information, np.eye(p+p_zero+1))
    covariance_full = (covariance_full+covariance_full.T)/2
    covariance = covariance_full[:-1, :-1]
    beta = parameters[:-1]
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
    return ZeroInflatedNegativeBinomialResult(
        _readonly(beta), _readonly(covariance), coefficient_names,
        transformation.feature_names, names, zero_names, zero_coefficient_names,
        psi_c, psi_z, se_c, se_z, ci_c, ci_z, pv_c, pv_z,
        float(alpha), int(fit.nit), transformation,
        columns, zero_columns, shared, exposure_component,
        theta=float(np.exp(parameters[-1])),
        se_logtheta=float(np.sqrt(covariance_full[-1, -1])),
        covariance_full=_readonly(covariance_full),
    )
