# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted Gaussian Tobit model for a censored continuous outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from scipy import special

from ._core import QuantizationResult, _column_names, _matrix, _readonly, _real_array, quantize


_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _likelihood_score_information(design: np.ndarray, y: np.ndarray,
                                  left: float | None, right: float | None,
                                  theta: np.ndarray):
    """Gaussian censoring log likelihood, analytic score and information."""
    beta, log_sigma = theta[:-1], theta[-1]
    sigma = np.exp(log_sigma)
    mu = design @ beta
    k = design.shape[1]
    score = np.zeros(k + 1)
    information = np.zeros((k + 1, k + 1))
    loglik = 0.0
    lower = np.zeros(len(y), dtype=bool) if left is None else y <= left
    upper = np.zeros(len(y), dtype=bool) if right is None else y >= right
    exact = ~(lower | upper)
    if np.any(exact):
        x, r = design[exact], (y[exact] - mu[exact]) / sigma
        loglik += float(np.sum(-_LOG_SQRT_2PI - log_sigma - 0.5*r*r))
        score[:k] += x.T @ (r/sigma)
        score[-1] += float(np.sum(r*r-1))
        information[:k, :k] += x.T @ x / sigma**2
        information[:k, -1] += x.T @ (2*r/sigma)
        information[-1, -1] += float(np.sum(2*r*r))
    if np.any(lower):
        x, a = design[lower], (left - mu[lower]) / sigma
        logcdf = special.log_ndtr(a)
        mills = np.exp(-_LOG_SQRT_2PI - 0.5*a*a - logcdf)
        derivative = -mills*(a + mills)
        loglik += float(np.sum(logcdf))
        score[:k] -= x.T @ (mills/sigma)
        score[-1] -= float(np.sum(mills*a))
        information[:k, :k] += x.T @ ((-derivative/sigma**2)[:, None]*x)
        information[:k, -1] -= x.T @ ((mills+a*derivative)/sigma)
        information[-1, -1] -= float(np.sum(derivative*a*a+mills*a))
    if np.any(upper):
        x, b = design[upper], (right - mu[upper]) / sigma
        logsf = special.log_ndtr(-b)
        mills = np.exp(-_LOG_SQRT_2PI - 0.5*b*b - logsf)
        derivative = mills*(mills-b)
        loglik += float(np.sum(logsf))
        score[:k] += x.T @ (mills/sigma)
        score[-1] += float(np.sum(mills*b))
        information[:k, :k] += x.T @ ((derivative/sigma**2)[:, None]*x)
        information[:k, -1] += x.T @ ((mills+b*derivative)/sigma)
        information[-1, -1] += float(np.sum(derivative*b*b+mills*b))
    information[-1, :k] = information[:k, -1]
    return loglik, score, information


def _solve_tobit(design: np.ndarray, y: np.ndarray,
                 left: float | None, right: float | None):
    exact = np.ones(len(y), dtype=bool)
    if left is not None:
        exact &= y > left
    if right is not None:
        exact &= y < right
    k = design.shape[1]
    if np.count_nonzero(exact) <= k:
        raise ValueError("Tobit fit needs more uncensored rows than coefficients")
    beta = np.linalg.lstsq(design[exact], y[exact], rcond=None)[0]
    residual = y[exact] - design[exact] @ beta
    sigma = float(np.std(residual))
    if sigma <= 1e-8 or not np.isfinite(sigma):
        raise ValueError("uncensored residual scale is at a numerical boundary")
    theta = np.r_[beta, np.log(sigma)]
    previous, _, _ = _likelihood_score_information(design, y, left, right, theta)
    for iteration in range(1, 101):
        _, score, info = _likelihood_score_information(design, y, left, right, theta)
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Tobit information is singular") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("Tobit update is nonfinite")
        scale = 1.0
        for _ in range(45):
            candidate = theta + scale*step
            if abs(candidate[-1]) > 20 or np.max(np.abs(candidate[:-1])) > 1e6:
                scale /= 2
                continue
            likelihood, _, _ = _likelihood_score_information(design, y, left, right, candidate)
            if np.isfinite(likelihood) and likelihood >= previous - 1e-11:
                break
            scale /= 2
        else:
            raise ValueError("Tobit line search failed")
        theta = candidate
        if np.max(np.abs(scale*step)) <= 1e-10*(1+np.max(np.abs(theta))):
            break
        previous = likelihood
    else:
        raise ValueError("Tobit fit did not converge in 100 iterations")
    _, score, information = _likelihood_score_information(design, y, left, right, theta)
    if np.max(np.abs(score)) > 1e-7:
        raise ValueError("Tobit score did not converge")
    try:
        covariance = np.linalg.solve(information, np.eye(k+1))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Tobit covariance is singular") from exc
    covariance = (covariance + covariance.T)/2
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError("Tobit covariance is not positive definite")
    return theta, covariance, iteration


@dataclass(frozen=True, eq=False)
class TobitResult:
    """Censored Gaussian fit; joint effect on the latent mean scale."""

    coefficients: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray = field(repr=False)
    full_covariance: np.ndarray = field(repr=False)
    sigma: float
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    alpha: float
    left: float | None
    right: float | None
    left_count: int
    right_count: int
    iterations: int
    _quantization: QuantizationResult = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "gaussian_tobit"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def f(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: censored Gaussian Tobit\n"
                f"Left-censored: {self.left_count}; right-censored: {self.right_count}.\n"
                "Joint effect: one score category in every exposure; covariates fixed.\n"
                f"Latent-mean difference: {f(self.psi)}; SE: {f(self.se)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI: ({f(self.ci[0])}, {f(self.ci[1])})\n"
                f"Two-sided normal z p-value: {f(self.p_value)}\n"
                f"Latent residual SD: {f(self.sigma)}\n"
                "The latent mean is not the mean of the observed censored outcome.")


def fit_tobit(y: Any, exposures: Any, covariates: Any = None, *,
              left: float | None = 0.0, right: float | None = None,
              q: int | None = 4, breaks: Any = None, alpha: float = 0.05) -> TobitResult:
    """Fit Gaussian Tobit and sum exposure effects on the latent mean scale.

    The supplied outcome must already be clipped to any censoring limits.
    Inputs are complete numeric arrays. This subset has no weights, clusters,
    formula, alternative latent distribution or observed-mean g-computation.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    for name, limit in (("left", left), ("right", right)):
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, Real)
                                  or not np.isfinite(limit)):
            raise ValueError(f"{name} must be a finite real number or None")
    if left is None and right is None:
        raise ValueError("at least one censoring limit is required")
    if left is not None and right is not None and not left < right:
        raise ValueError("left must be below right")
    outcome = _real_array(y, "y")
    if outcome.ndim != 1 or outcome.size == 0:
        raise ValueError("y must be a nonempty vector")
    if left is not None and np.any(outcome < left):
        raise ValueError("y must already be clipped at the left limit")
    if right is not None and np.any(outcome > right):
        raise ValueError("y must already be clipped at the right limit")
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if len(outcome) != n:
        raise ValueError("y and exposures must have the same rows")
    if covariates is None:
        z, names = np.empty((n, 0)), ()
    else:
        z, labels = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same rows")
        names = _column_names(labels, z.shape[1], "z")
    coefficient_names = ("(Intercept)",) + transformation.feature_names + names
    if len(set(coefficient_names)) != len(coefficient_names):
        raise ValueError("coefficient identifiers must be distinct")
    design = np.column_stack((np.ones(n), transformation.scores, z))
    if n <= design.shape[1]+1 or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Tobit design needs full rank and more rows than parameters")
    theta, full_covariance, iterations = _solve_tobit(design, outcome, left, right)
    k = design.shape[1]
    contrast = np.zeros(k)
    contrast[1:1+d] = 1
    psi = float(contrast @ theta[:k])
    covariance = full_covariance[:k, :k]
    variance = float(contrast @ covariance @ contrast)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("joint Tobit variance is not positive and finite")
    se = float(np.sqrt(variance))
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2)))
    statistic = psi/se
    return TobitResult(_readonly(theta[:k]), coefficient_names, _readonly(covariance),
                       _readonly(full_covariance), float(np.exp(theta[-1])), psi,
                       variance, se, (float(psi-critical*se), float(psi+critical*se)),
                       statistic, float(2*special.ndtr(-abs(statistic))), float(alpha),
                       left, right, int(np.count_nonzero(outcome <= left)) if left is not None else 0,
                       int(np.count_nonzero(outcome >= right)) if right is not None else 0,
                       iterations, transformation)
