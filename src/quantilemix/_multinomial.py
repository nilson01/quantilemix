# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted conditional multinomial-logit quantile g-computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Sequence
import warnings

import numpy as np
from scipy import special
from scipy.stats import chi2

from ._core import QGCompWarning, QuantizationResult, _check_columns, _column_names, _matrix, _readonly, quantize


def _probabilities(design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = np.column_stack((np.zeros(len(design)), design @ beta.T))
    return np.exp(eta - special.logsumexp(eta, axis=1, keepdims=True))


def _likelihood_score_information(design: np.ndarray, outcome: np.ndarray,
                                  beta: np.ndarray):
    n, p = design.shape
    m = beta.shape[0]
    eta = np.column_stack((np.zeros(n), design @ beta.T))
    normalizer = special.logsumexp(eta, axis=1)
    probabilities = np.exp(eta - normalizer[:, None])
    likelihood = float(np.sum(eta[np.arange(n), outcome] - normalizer))
    targets = (outcome[:, None] == np.arange(1, m+1)[None, :]).astype(float)
    score = (targets - probabilities[:, 1:]).T @ design
    information = np.empty((m*p, m*p))
    for a in range(m):
        pa = probabilities[:, a+1]
        for b in range(m):
            pb = probabilities[:, b+1]
            weight = pa * ((1.0 if a == b else 0.0) - pb)
            information[a*p:(a+1)*p, b*p:(b+1)*p] = design.T @ (weight[:, None] * design)
    return likelihood, score.ravel(), (information + information.T)/2


def _solve_multinomial(design: np.ndarray, outcome: np.ndarray, class_count: int):
    n, p = design.shape
    beta = np.zeros((class_count-1, p))
    previous, _, _ = _likelihood_score_information(design, outcome, beta)
    for iteration in range(1, 101):
        _, score, info = _likelihood_score_information(design, outcome, beta)
        try:
            step = np.linalg.solve(info, score).reshape(beta.shape)
        except np.linalg.LinAlgError as exc:
            raise ValueError("multinomial information is singular") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("multinomial update is nonfinite")
        scale = 1.0
        for _ in range(45):
            candidate = beta + scale*step
            if np.max(np.abs(candidate)) > 30:
                scale /= 2
                continue
            likelihood, _, _ = _likelihood_score_information(design, outcome, candidate)
            if np.isfinite(likelihood) and likelihood >= previous - 1e-11:
                break
            scale /= 2
        else:
            raise ValueError("multinomial line search failed; check for separation")
        beta = candidate
        if np.max(np.abs(scale*step)) <= 1e-10*(1+np.max(np.abs(beta))):
            break
        previous = likelihood
    else:
        raise ValueError("multinomial fit did not converge in 100 iterations")
    _, score, information = _likelihood_score_information(design, outcome, beta)
    if np.max(np.abs(score)) > 1e-7:
        raise ValueError("multinomial score did not converge")
    try:
        covariance = np.linalg.solve(information, np.eye((class_count-1)*p))
    except np.linalg.LinAlgError as exc:
        raise ValueError("multinomial covariance is singular") from exc
    covariance = (covariance + covariance.T)/2
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError("multinomial covariance is not positive definite")
    return beta, covariance, iteration


@dataclass(frozen=True, eq=False)
class MultinomialResult:
    """Conditional category-vs-reference joint log odds effects."""

    coefficients: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray = field(repr=False)
    levels: tuple[str, ...]
    exposure_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    psi: np.ndarray = field(repr=False)
    psi_covariance: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    fitted_probabilities: np.ndarray = field(repr=False)
    alpha: float
    iterations: int
    _quantization: QuantizationResult = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "multinomial"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def odds_ratios(self) -> np.ndarray:
        return _readonly(np.exp(self.psi))

    def joint_test(self):
        """Chi-square test that every category-vs-reference joint effect is zero."""
        statistic = float(self.psi @ np.linalg.solve(self.psi_covariance, self.psi))
        df = len(self.psi)
        return MappingProxyType({"chi_square": statistic, "df": df,
                                 "p_value": float(chi2.sf(statistic, df))})

    def homogeneity_test(self):
        """Chi-square test that category-vs-reference joint effects are equal."""
        m = len(self.psi)
        contrast = np.column_stack((-np.ones(m-1), np.eye(m-1)))
        difference = contrast @ self.psi
        covariance = contrast @ self.psi_covariance @ contrast.T
        statistic = float(difference @ np.linalg.solve(covariance, difference))
        df = m-1
        return MappingProxyType({"chi_square": statistic, "df": df,
                                 "p_value": float(chi2.sf(statistic, df))})

    def predict_proba(self, exposures: Any, covariates: Any = None) -> np.ndarray:
        """Conditional outcome probabilities in saved level order."""
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
        design = np.column_stack((np.ones(n), transformed.scores, z))
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn("Some exposures are outside their training ranges; fitted cutpoints are reused.",
                          QGCompWarning, stacklevel=2)
        return _readonly(_probabilities(design, self.coefficients))

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def f(value: float) -> str:
            return format(value, f".{digits}g")
        lines = ["Quantile g-computation: conditional multinomial logit",
                 f"Reference outcome: {self.levels[0]!r}; {self.iterations} fitting iterations.",
                 "Each joint effect raises every exposure score by one; covariates fixed."]
        for index, label in enumerate(self.levels[1:]):
            lines.append(f"{label!r} versus {self.levels[0]!r}: log odds ratio {f(self.psi[index])}; "
                         f"SE {f(self.se[index])}; normal {(1-self.alpha)*100:g}% CI "
                         f"({f(self.ci[index, 0])}, {f(self.ci[index, 1])}); "
                         f"z p-value {f(self.p_value[index])}.")
        lines.append("Effects compare categories with the reference; they are not one pooled effect.")
        return "\n".join(lines)


def fit_multinomial(y: Any, exposures: Any, covariates: Any = None, *,
                    levels: Sequence[str], q: int | None = 4,
                    breaks: Any = None, alpha: float = 0.05) -> MultinomialResult:
    """Fit additive multinomial logit and sum exposures by outcome category.

    ``levels`` must name every category in order; the first is the reference.
    This subset uses complete rows and has no weights, clusters, formula,
    nonlinear terms or marginal standardization.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if isinstance(levels, (str, bytes)):
        raise ValueError("levels must be an ordered sequence of distinct strings")
    labels = tuple(levels)
    if len(labels) < 3 or any(not isinstance(v, str) or not v for v in labels) or len(set(labels)) != len(labels):
        raise ValueError("levels must contain at least three distinct nonempty strings")
    try:
        outcomes = np.asarray(y, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be a vector of named outcome categories") from exc
    if np.ma.isMaskedArray(y) and np.any(np.ma.getmaskarray(y)):
        raise ValueError("y contains masked values")
    if outcomes.ndim != 1 or outcomes.size == 0 or any(not isinstance(v, str) or v not in labels for v in outcomes):
        raise ValueError("y must be a nonempty vector whose values appear in levels")
    outcome = np.asarray([labels.index(value) for value in outcomes], dtype=int)
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if len(outcome) != n:
        raise ValueError("y and exposures must have the same rows")
    if covariates is None:
        z, columns = np.empty((n, 0)), None
    else:
        z, columns = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same rows")
    names = _column_names(columns, z.shape[1], "z")
    coefficient_names = ("(Intercept)",) + transformation.feature_names + names
    if len(set(coefficient_names)) != len(coefficient_names):
        raise ValueError("coefficient identifiers must be distinct")
    design = np.column_stack((np.ones(n), transformation.scores, z))
    k = design.shape[1]
    if n <= (len(labels)-1)*k or np.linalg.matrix_rank(design) != k:
        raise ValueError("multinomial design needs full rank and more rows than parameters")
    if np.any(np.bincount(outcome, minlength=len(labels)) <= k):
        raise ValueError("each outcome category needs more rows than coefficients")
    beta, covariance, iterations = _solve_multinomial(design, outcome, len(labels))
    contrast = np.zeros((len(labels)-1, (len(labels)-1)*k))
    for j in range(len(labels)-1):
        contrast[j, j*k+1:j*k+1+d] = 1
    psi = contrast @ beta.ravel()
    psi_covariance = contrast @ covariance @ contrast.T
    if not np.all(np.isfinite(psi_covariance)) or np.linalg.eigvalsh(psi_covariance)[0] <= 0:
        raise ValueError("multinomial joint covariance is not positive definite")
    se = np.sqrt(np.diag(psi_covariance))
    critical = float(-special.ndtri_exp(np.log(alpha)-np.log(2)))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    p_value = 2*special.ndtr(-np.abs(psi/se))
    return MultinomialResult(_readonly(beta), coefficient_names, _readonly(covariance),
                             labels, transformation.feature_names, names, _readonly(psi),
                             _readonly(psi_covariance), _readonly(se), _readonly(ci),
                             _readonly(p_value), _readonly(_probabilities(design, beta)),
                             float(alpha), iterations, transformation, columns)
