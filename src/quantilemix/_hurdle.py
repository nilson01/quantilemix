# SPDX-License-Identifier: GPL-2.0-or-later
"""Conditional additive Poisson hurdle component effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any
import warnings

import numpy as np
from scipy import optimize, special

from ._binomial import _solve_logit
from ._core import QGCompWarning, QuantizationResult, _check_columns, _column_names, _matrix, _readonly, _real_array, quantize


def _positive_likelihood_score_information(design: np.ndarray, y: np.ndarray,
                                           beta: np.ndarray):
    eta = design @ beta
    if np.max(eta) > 20 or np.min(eta) < -25:
        return -np.inf, np.full(len(beta), np.nan), None
    mu = np.exp(eta)
    normalizer = -np.expm1(-mu)  # Pr(Poisson count > 0)
    trunc_mean = mu/normalizer
    log_likelihood = float(np.sum(y*eta-mu-special.gammaln(y+1)-np.log(normalizer)))
    score = design.T @ (y-trunc_mean)
    variance = trunc_mean - mu*mu*np.exp(-mu)/(normalizer*normalizer)
    small = mu < 1e-4
    variance[small] = mu[small]/2 + mu[small]**2/6
    information = design.T @ (variance[:, None]*design)
    return log_likelihood, score, (information+information.T)/2


@dataclass(frozen=True, eq=False)
class HurdlePoissonResult:
    """Truncated-count log parameter and nonzero log-odds joint effects."""

    coefficients: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    exposure_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    zero_covariate_names: tuple[str, ...]
    zero_coefficient_names: tuple[str, ...]
    psi_count: float | None
    psi_nonzero: float | None
    se_count: float | None
    se_nonzero: float | None
    ci_count: tuple[float, float] | None
    ci_nonzero: tuple[float, float] | None
    p_value_count: float | None
    p_value_nonzero: float | None
    alpha: float
    iterations: int
    _quantization: QuantizationResult = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    _zero_covariate_columns: tuple[str, ...] | None = field(repr=False)
    _shared_zero_covariates: bool = field(repr=False)
    exposure_component: str = "both"
    reference_version: str = "2.19.6"
    family: str = "hurdle_poisson"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def count_parameter_ratio(self) -> float | None:
        """Ratio of underlying count means, not truncated positive means."""
        return None if self.psi_count is None else float(np.exp(self.psi_count))

    @property
    def nonzero_odds_ratio(self) -> float | None:
        return None if self.psi_nonzero is None else float(np.exp(self.psi_nonzero))

    def predict_components(self, exposures: Any, covariates: Any = None, *,
                           zero_covariates: Any = "same") -> tuple[np.ndarray, np.ndarray]:
        """Return underlying count mean and chance of a positive count."""
        transformed = self._quantization.transform(exposures)
        n = len(transformed.scores)
        if self.covariate_names:
            if covariates is None:
                raise ValueError("covariates are required for this fitted model")
            z, names = _matrix(covariates, "covariates")
            if z.shape != (n, len(self.covariate_names)):
                raise ValueError("covariates must have the fitted rows and columns")
            _check_columns(names, self._covariate_columns, "covariates")
        else:
            if covariates is not None:
                raise ValueError("this model was fitted without covariates")
            z = np.empty((n, 0))
        if isinstance(zero_covariates, str) and zero_covariates == "same":
            if not self._shared_zero_covariates:
                raise ValueError("zero_covariates must be supplied for this fitted model")
            zz = z
        elif zero_covariates is None:
            zz = np.empty((n, 0))
        else:
            if isinstance(zero_covariates, str):
                raise ValueError("zero_covariates must be 'same', None, or complete numeric columns")
            zz, zero_names = _matrix(zero_covariates, "zero_covariates")
            _check_columns(zero_names, self._zero_covariate_columns, "zero_covariates")
        if zz.shape != (n, len(self.zero_covariate_names)):
            raise ValueError("zero_covariates must have the fitted rows and columns")
        count_scores = transformed.scores if self.exposure_component != "zero" else np.empty((n, 0))
        zero_scores = transformed.scores if self.exposure_component != "count" else np.empty((n, 0))
        count_design = np.column_stack((np.ones(n), count_scores, z))
        zero_design = np.column_stack((np.ones(n), zero_scores, zz))
        p = count_design.shape[1]
        eta = count_design @ self.coefficients[:p]
        if np.max(eta) > 20 or np.min(eta) < -25:
            raise ValueError("predicted count means exceed the supported numerical range")
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn("Some exposures are outside their training ranges; fitted cutpoints are reused.",
                          QGCompWarning, stacklevel=2)
        return _readonly(np.exp(eta)), _readonly(special.expit(zero_design @ self.coefficients[p:]))

    def predict(self, exposures: Any, covariates: Any = None, *,
                zero_covariates: Any = "same") -> np.ndarray:
        """Expected observed count from the nonzero chance and truncated Poisson mean."""
        mu, p_nonzero = self.predict_components(exposures, covariates,
                                                zero_covariates=zero_covariates)
        return _readonly(p_nonzero*mu/(-np.expm1(-mu)))

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(v: float) -> str:
            return format(v, f".{digits}g")
        count_line = (
            f"Positive-count component: log Poisson-parameter ratio {fmt(self.psi_count)}; "
            f"SE {fmt(self.se_count)}; normal {(1-self.alpha)*100:g}% CI "
            f"({fmt(self.ci_count[0])}, {fmt(self.ci_count[1])}); z p-value {fmt(self.p_value_count)}.\n"
            if self.psi_count is not None else "Positive-count component: no joint exposure effect fitted.\n")
        zero_line = (
            f"Nonzero hurdle component: log odds ratio {fmt(self.psi_nonzero)}; "
            f"SE {fmt(self.se_nonzero)}; normal {(1-self.alpha)*100:g}% CI "
            f"({fmt(self.ci_nonzero[0])}, {fmt(self.ci_nonzero[1])}); z p-value {fmt(self.p_value_nonzero)}.\n"
            if self.psi_nonzero is not None else "Nonzero hurdle component: no joint exposure effect fitted.\n")
        return ("Quantile g-computation: conditional Poisson hurdle\n"
                "Joint score change raises every exposure by one; covariates fixed.\n"
                + count_line + zero_line +
                "These are separate component effects; neither is an observed-count ratio.")


def fit_hurdle_poisson(y: Any, exposures: Any, covariates: Any = None, *,
                       zero_covariates: Any = "same",
                       exposure_component: str = "both",
                       q: int | None = 4, breaks: Any = None,
                       alpha: float = 0.05) -> HurdlePoissonResult:
    """Fit truncated-positive Poisson counts and a separate nonzero logit.

    Omitted ``zero_covariates`` reuses ``covariates`` in both components.
    Pass a separate numeric matrix or explicit ``None`` to change the
    nonzero component. ``exposure_component`` is ``"both"``, ``"count"``,
    or ``"zero"``; an omitted component effect is returned as ``None``. This subset
    has no weights, clusters, formulas, offsets or negative-binomial option.
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
    if (np.sum(positive) <= p or np.linalg.matrix_rank(count_design) != p
            or np.linalg.matrix_rank(count_design[positive]) != p
            or np.linalg.matrix_rank(zero_design) != p_zero):
        raise ValueError("hurdle count and nonzero designs need full rank and enough positive rows")
    zero_beta, zero_covariance, zero_iterations = _solve_logit(zero_design, positive.astype(float), np.ones(n))
    count_design, positive_y = count_design[positive], outcome[positive]
    start = np.zeros(p)
    start[0] = np.log(float(np.mean(positive_y)))
    def objective(beta: np.ndarray):
        likelihood, score, _ = _positive_likelihood_score_information(count_design, positive_y, beta)
        if not np.isfinite(likelihood):
            return 1e100, np.zeros_like(beta)
        return -likelihood, -score
    fit = optimize.minimize(objective, start, jac=True, method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 1000})
    count_beta = fit.x
    likelihood, score, information = _positive_likelihood_score_information(count_design, positive_y, count_beta)
    if np.all(np.isfinite(count_beta)) and np.all(np.isfinite(score)) and np.max(np.abs(score)) > 1e-5:
        def score_equations(candidate: np.ndarray):
            _, gradient, _ = _positive_likelihood_score_information(count_design, positive_y, candidate)
            return gradient
        def score_jacobian(candidate: np.ndarray):
            _, _, observed_information = _positive_likelihood_score_information(
                count_design, positive_y, candidate)
            return -observed_information if observed_information is not None else np.full((p, p), np.nan)
        refined = optimize.root(score_equations, count_beta, jac=score_jacobian, method="hybr")
        if np.all(np.isfinite(refined.x)):
            refined_likelihood, refined_score, refined_information = _positive_likelihood_score_information(
                count_design, positive_y, refined.x)
            if (np.isfinite(refined_likelihood) and refined_likelihood >= likelihood-1e-7
                    and np.all(np.isfinite(refined_score))
                    and np.max(np.abs(refined_score)) < np.max(np.abs(score))):
                count_beta, score, information = refined.x, refined_score, refined_information
    if not np.all(np.isfinite(count_beta)) or np.max(np.abs(score)) > 1e-5:
        raise ValueError("hurdle positive-count fit did not converge")
    if information is None or np.linalg.eigvalsh(information)[0] <= 1e-9:
        raise ValueError("hurdle positive-count information is singular")
    count_covariance = np.linalg.solve(information, np.eye(p))
    covariance = np.zeros((p+p_zero, p+p_zero))
    covariance[:p, :p] = count_covariance
    covariance[p:, p:] = zero_covariance
    beta = np.r_[count_beta, zero_beta]
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
    return HurdlePoissonResult(_readonly(beta), _readonly(covariance), coefficient_names,
                               transformation.feature_names, names, zero_names, zero_coefficient_names,
                               psi_c, psi_z, se_c, se_z, ci_c, ci_z,
                               pv_c, pv_z, float(alpha),
                               int(fit.nit+zero_iterations), transformation, columns,
                               zero_columns, shared, exposure_component)
