# SPDX-License-Identifier: GPL-2.0-or-later
"""Conditional additive zero-inflated Poisson mixture effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any
import warnings

import numpy as np
from scipy import optimize, special

from ._core import QGCompWarning, QuantizationResult, _check_columns, _column_names, _matrix, _readonly, _real_array, quantize


def _likelihood_score_information(count: np.ndarray, zero: np.ndarray,
                                  y: np.ndarray, beta: np.ndarray):
    p = count.shape[1]
    eta_c, eta_z = count @ beta[:p], zero @ beta[p:]
    if np.max(eta_c) > 20:
        return -np.inf, np.full(len(beta), np.nan), None
    mu = np.exp(eta_c)
    pi = special.expit(eta_z)
    positive = y > 0
    s = special.expit(eta_z + mu)  # Pr(structural zero | observed zero)
    r = 1 - s
    # log(pi) = log_expit(eta_z); log(1-pi) = log_expit(-eta_z)
    log_p_zero = np.logaddexp(special.log_expit(eta_z),
                              special.log_expit(-eta_z) - mu)
    log_p_positive = special.log_expit(-eta_z) + y*eta_c - mu - special.gammaln(y+1)
    likelihood = float(np.sum(np.where(positive, log_p_positive, log_p_zero)))
    sc_c = np.where(positive, y-mu, -r*mu)
    sc_z = np.where(positive, -pi, s-pi)
    score = np.r_[count.T @ sc_c, zero.T @ sc_z]
    w_cc = np.where(positive, mu, r*mu-r*s*mu**2)
    w_zz = np.where(positive, pi*(1-pi), pi*(1-pi)-r*s)
    w_cz = np.where(positive, 0.0, -r*s*mu)
    cc = count.T @ (w_cc[:, None]*count)
    zz = zero.T @ (w_zz[:, None]*zero)
    cz = count.T @ (w_cz[:, None]*zero)
    info = np.block([[cc, cz], [cz.T, zz]])
    return likelihood, score, (info+info.T)/2


@dataclass(frozen=True, eq=False)
class ZeroInflatedPoissonResult:
    """Count log-mean and structural-zero log-odds joint effects."""

    coefficients: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    exposure_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    zero_covariate_names: tuple[str, ...]
    zero_coefficient_names: tuple[str, ...]
    psi_count: float | None
    psi_zero: float | None
    se_count: float | None
    se_zero: float | None
    ci_count: tuple[float, float] | None
    ci_zero: tuple[float, float] | None
    p_value_count: float | None
    p_value_zero: float | None
    alpha: float
    iterations: int
    _quantization: QuantizationResult = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    _zero_covariate_columns: tuple[str, ...] | None = field(repr=False)
    _shared_zero_covariates: bool = field(repr=False)
    exposure_component: str = "both"
    reference_version: str = "2.19.6"
    family: str = "zero_inflated_poisson"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def count_ratio(self) -> float | None:
        return None if self.psi_count is None else float(np.exp(self.psi_count))

    @property
    def structural_zero_odds_ratio(self) -> float | None:
        return None if self.psi_zero is None else float(np.exp(self.psi_zero))

    def predict_components(self, exposures: Any, covariates: Any = None, *,
                           zero_covariates: Any = "same") -> tuple[np.ndarray, np.ndarray]:
        """Return underlying count mean and structural-zero probability."""
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
        eta_c = count_design @ self.coefficients[:p]
        if np.max(eta_c) > 20:
            raise ValueError("predicted count mean exceeds the supported numerical range")
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn("Some exposures are outside their training ranges; fitted cutpoints are reused.",
                          QGCompWarning, stacklevel=2)
        return _readonly(np.exp(eta_c)), _readonly(special.expit(zero_design @ self.coefficients[p:]))

    def predict(self, exposures: Any, covariates: Any = None, *,
                zero_covariates: Any = "same") -> np.ndarray:
        """Expected count: (1 - structural-zero probability) times count mean."""
        mu, pi = self.predict_components(exposures, covariates, zero_covariates=zero_covariates)
        return _readonly((1-pi)*mu)

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(v: float) -> str:
            return format(v, f".{digits}g")
        count_line = (
            f"Poisson count component: log mean ratio {fmt(self.psi_count)}; SE {fmt(self.se_count)}; "
            f"normal {(1-self.alpha)*100:g}% CI ({fmt(self.ci_count[0])}, {fmt(self.ci_count[1])}); "
            f"z p-value {fmt(self.p_value_count)}.\n"
            if self.psi_count is not None else "Poisson count component: no joint exposure effect fitted.\n")
        zero_line = (
            f"Structural-zero component: log odds ratio {fmt(self.psi_zero)}; SE {fmt(self.se_zero)}; "
            f"normal {(1-self.alpha)*100:g}% CI ({fmt(self.ci_zero[0])}, {fmt(self.ci_zero[1])}); "
            f"z p-value {fmt(self.p_value_zero)}.\n"
            if self.psi_zero is not None else "Structural-zero component: no joint exposure effect fitted.\n")
        return ("Quantile g-computation: conditional zero-inflated Poisson\n"
                "Joint score change raises every exposure by one; covariates fixed.\n"
                + count_line + zero_line +
                "These are separate component effects; neither is an observed-count ratio.")


def fit_zero_inflated_poisson(y: Any, exposures: Any, covariates: Any = None, *,
                              zero_covariates: Any = "same",
                              exposure_component: str = "both",
                              q: int | None = 4, breaks: Any = None,
                              alpha: float = 0.05) -> ZeroInflatedPoissonResult:
    """Fit an additive ZIP model with the same exposures in both components.

    Omitted ``zero_covariates`` reuses ``covariates`` in both components.
    Pass a separate numeric matrix or explicit ``None`` to change the zero
    component. ``exposure_component`` is ``"both"``, ``"count"``, or
    ``"zero"``; an omitted component effect is returned as ``None``. This subset requires
    complete numeric data and has no weights, clusters, formulas, offsets or
    negative-binomial component.
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
    if (n <= p+p_zero or np.linalg.matrix_rank(count_design) != p
            or np.linalg.matrix_rank(zero_design) != p_zero):
        raise ValueError("zero-inflated component designs need full rank and more rows than parameters")
    start = np.zeros(p+p_zero)
    start[0] = np.log(float(np.mean(outcome[outcome > 0])))
    start[p] = -1.0
    def objective(beta: np.ndarray):
        likelihood, score, _ = _likelihood_score_information(count_design, zero_design, outcome, beta)
        if not np.isfinite(likelihood):
            return 1e100, np.zeros_like(beta)
        return -likelihood, -score
    fit = optimize.minimize(objective, start, jac=True, method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 1000})
    beta = fit.x
    likelihood, score, information = _likelihood_score_information(count_design, zero_design, outcome, beta)
    if np.all(np.isfinite(beta)) and np.all(np.isfinite(score)) and np.max(np.abs(score)) > 1e-5:
        # BFGS may stop at a precision-loss warning while the analytic score
        # still has a small correctable residual. Refine the score equations
        # using their analytic Jacobian; retain the full-rank check below.
        def score_equations(candidate: np.ndarray):
            _, gradient, _ = _likelihood_score_information(count_design, zero_design, outcome, candidate)
            return gradient
        def score_jacobian(candidate: np.ndarray):
            _, _, observed_information = _likelihood_score_information(
                count_design, zero_design, outcome, candidate)
            return -observed_information if observed_information is not None else np.full((p+p_zero, p+p_zero), np.nan)
        refined = optimize.root(score_equations, beta, jac=score_jacobian, method="hybr")
        if np.all(np.isfinite(refined.x)):
            refined_likelihood, refined_score, refined_information = _likelihood_score_information(
                count_design, zero_design, outcome, refined.x)
            if (np.isfinite(refined_likelihood) and refined_likelihood >= likelihood-1e-7
                    and np.all(np.isfinite(refined_score))
                    and np.max(np.abs(refined_score)) < np.max(np.abs(score))):
                beta, score, information = refined.x, refined_score, refined_information
    if not np.all(np.isfinite(beta)) or np.max(np.abs(score)) > 1e-5:
        raise ValueError("zero-inflated Poisson fit did not converge; check identifiability")
    if information is None or np.linalg.eigvalsh(information)[0] <= 1e-9:
        raise ValueError("zero-inflated Poisson information is singular")
    covariance = np.linalg.solve(information, np.eye(p+p_zero))
    covariance = (covariance+covariance.T)/2
    contrast = np.zeros((2, p+p_zero))
    if exposure_component != "zero":
        contrast[0, 1:1+d] = 1
    if exposure_component != "count":
        contrast[1, p+1:p+1+d] = 1
    psi = contrast @ beta
    var = np.diag(contrast @ covariance @ contrast.T)
    se = np.sqrt(var)
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
    return ZeroInflatedPoissonResult(_readonly(beta), _readonly(covariance), coefficient_names,
                                     transformation.feature_names, names, zero_names, zero_coefficient_names,
                                     psi_c, psi_z, se_c, se_z, ci_c, ci_z,
                                     pv_c, pv_z, float(alpha), int(fit.nit),
                                     transformation, columns, zero_columns, shared, exposure_component)
