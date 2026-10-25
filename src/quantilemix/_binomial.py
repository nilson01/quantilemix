# SPDX-License-Identifier: GPL-2.0-or-later
"""Additive Bernoulli-logit quantile g-computation.

The joint coefficient is conditional log odds, not a marginal risk difference.
Method: Keil et al. (2020), doi:10.1289/EHP5838.
Reference conventions: R qgcomp 2.19.6, GPL (>= 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy import special

from ._core import (
    QGCompWarning, QuantizationResult, _CONDITION_WARNING, _check_columns,
    _column_names, _matrix, _readonly, _real_array, _weights, quantize,
)


def _log_likelihood(design: np.ndarray, y: np.ndarray, beta: np.ndarray,
                    case_weights: np.ndarray) -> float:
    eta = design @ beta
    return float(np.sum(case_weights * (y * special.log_expit(eta)
                        + (1.0 - y) * special.log_expit(-eta))))


def _solve_logit(design: np.ndarray, y: np.ndarray,
                 case_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n, k = design.shape
    beta = np.zeros(k)
    proportion = float(np.average(y, weights=case_weights))
    beta[0] = np.log(proportion / (1.0 - proportion))
    previous = _log_likelihood(design, y, beta, case_weights)
    for iteration in range(1, 101):
        eta = design @ beta
        probability = special.expit(eta)
        weight = case_weights * probability * (1.0 - probability)
        information = design.T @ (weight[:, None] * design)
        score = design.T @ (case_weights * (y - probability))
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("binomial information is singular; check separation") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("binomial update is nonfinite; check separation")
        scale = 1.0
        for _ in range(40):
            candidate = beta + scale * step
            likelihood = _log_likelihood(design, y, candidate, case_weights)
            if np.isfinite(likelihood) and likelihood >= previous - 1e-12:
                break
            scale /= 2.0
        else:
            raise ValueError("binomial line search failed; check separation")
        beta = candidate
        if np.max(np.abs(beta)) > 30:
            raise ValueError("binomial coefficients diverged; check separation")
        if (np.max(np.abs(scale * step)) <= 1e-11 * (1 + np.max(np.abs(beta)))
                and np.max(np.abs(score)) <= 1e-8 * max(1.0, float(np.sum(case_weights) / n))):
            break
        previous = likelihood
    else:
        raise ValueError("binomial fit did not converge in 100 iterations")
    probability = special.expit(design @ beta)
    weight = case_weights * probability * (1.0 - probability)
    if np.min(weight) < 1e-12:
        raise ValueError("fitted probabilities are at numerical boundaries; check separation")
    information = design.T @ (weight[:, None] * design)
    try:
        covariance = np.linalg.solve(information, np.eye(k))
    except np.linalg.LinAlgError as exc:
        raise ValueError("binomial covariance is singular") from exc
    covariance = (covariance + covariance.T) / 2.0
    if not np.all(np.isfinite(covariance)):
        raise ValueError("binomial covariance is nonfinite")
    return beta, covariance, iteration


@dataclass(frozen=True, eq=False)
class BinomialResult:
    """Conditional additive Bernoulli-logit fit and joint log-odds effect."""

    coefficients: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray = field(repr=False)
    exposure_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    fitted_values: np.ndarray = field(repr=False)
    log_likelihood: float
    deviance: float
    null_deviance: float
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    p_value_df: None
    residual_df: int
    alpha: float
    pos_psi: float
    neg_psi: float
    pos_weights: Mapping[str, float]
    neg_weights: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    case_weights: np.ndarray | None = field(repr=False)
    _quantization: QuantizationResult = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    _formula_schema: Any = field(default=None, repr=False)
    reference_version: str = "2.19.6"
    family: str = "binomial"
    link: str = "logit"

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def q(self) -> int | None:
        return self._quantization.q

    @property
    def quantization_mode(self) -> str:
        return self._quantization.mode

    @property
    def odds_ratio(self) -> float:
        """Conditional odds multiplier for one score increase in every exposure."""
        return float(np.exp(self.psi))

    @property
    def odds_ratio_ci(self) -> tuple[float, float]:
        return (float(np.exp(self.ci[0])), float(np.exp(self.ci[1])))

    @property
    def formula(self) -> str | None:
        return None if self._formula_schema is None else self._formula_schema.formula

    @property
    def categorical_levels(self) -> Mapping[str, tuple[str, ...]]:
        if self._formula_schema is None:
            return MappingProxyType({})
        return MappingProxyType(dict(self._formula_schema.category_levels))

    @property
    def training_row_indices(self) -> tuple[int, ...] | None:
        return None if self._formula_schema is None else self._formula_schema.training_row_indices

    @property
    def original_row_count(self) -> int | None:
        return None if self._formula_schema is None else self._formula_schema.original_row_count

    def transform(self, exposures: Any) -> QuantizationResult:
        return self._quantization.transform(exposures)

    def predict(self, exposures: Any, covariates: Any = None) -> np.ndarray:
        """Predict conditional event probabilities from saved exposure cuts."""
        transformed = self.transform(exposures)
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
        prediction = special.expit(design @ self.coefficients)
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn("Some exposures are outside their training ranges; fitted cutpoints are reused.",
                          QGCompWarning, stacklevel=2)
        return _readonly(prediction)

    def predict_data(self, data: Mapping[str, Any]) -> np.ndarray:
        if self._formula_schema is None:
            raise ValueError("predict_data requires a fit_binomial_formula result")
        from ._formula import formula_matrices

        x, z = formula_matrices(data, self._formula_schema)
        return self.predict(x, z)

    def transform_data(self, data: Mapping[str, Any]) -> QuantizationResult:
        if self._formula_schema is None:
            raise ValueError("transform_data requires a fit_binomial_formula result")
        from ._formula import formula_exposures

        return self.transform(formula_exposures(data, self._formula_schema))

    def summary(self, digits: int = 6) -> str:
        """Show the saved log-odds contrast and its normal interval."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: additive binomial logit\n"
                "Joint effect: one score category in every exposure; covariates fixed.\n"
                f"Log odds ratio: {fmt(self.psi)}; SE: {fmt(self.se)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI: ({fmt(self.ci[0])}, {fmt(self.ci[1])})\n"
                f"Odds ratio: {fmt(self.odds_ratio)}\n"
                f"Two-sided normal z p-value (R convention): {fmt(self.p_value)}\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'not supplied'}; model-based SE.\n"
                "The odds ratio is conditional on the adjustment covariates; it is not a risk ratio.")


def fit_binomial(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None,
) -> BinomialResult:
    """Fit an additive Bernoulli-logit model and sum exposure coefficients.

    ``y`` must contain only 0 and 1. Inputs are complete, real and finite.
    Positive case weights act as GLM prior weights. Exposure cuts ignore weights.
    The sum is a conditional log odds ratio. Its confidence interval uses
    normal limits and a normal z p-value, following R qgcomp 2.19.6.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    outcome = _real_array(y, "y")
    if outcome.ndim != 1 or not outcome.size or not np.all((outcome == 0) | (outcome == 1)):
        raise ValueError("binomial y must be a nonempty vector of 0 and 1")
    if np.all(outcome == outcome[0]):
        raise ValueError("binomial y must contain both 0 and 1")
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if len(outcome) != n:
        raise ValueError("y and exposures must have the same number of rows")
    if weights is None:
        case_weights = None
        solver_weights = np.ones(n)
    else:
        case_weights = _real_array(weights, "weights")
        if case_weights.ndim != 1 or case_weights.size != n:
            raise ValueError("weights must be a one-dimensional vector with one value per row")
        if np.any(case_weights <= 0):
            raise ValueError("weights must be strictly positive")
        solver_weights = case_weights
    if covariates is None:
        z = np.empty((n, 0), dtype=float)
        covariate_columns = None
    else:
        z, covariate_columns = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    names = _column_names(covariate_columns, z.shape[1], "z")
    coefficient_names = ("(Intercept)",) + transformation.feature_names + names
    if len(set(coefficient_names)) != len(coefficient_names):
        raise ValueError("coefficient identifiers must be distinct")
    design = np.column_stack((np.ones(n), transformation.scores, z))
    k = design.shape[1]
    if n <= k:
        raise ValueError("the binomial model needs more rows than coefficients")
    try:
        singular_values = np.linalg.svd(design * np.sqrt(solver_weights)[:, None],
                                        compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the binomial design singular-value decomposition failed") from exc
    tolerance = max(n, k) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != k:
        raise ValueError("the binomial design is rank deficient")
    condition = float(singular_values[0] / singular_values[-1])
    if condition > _CONDITION_WARNING:
        warnings.warn("The binomial design is ill conditioned; inspect numerical sensitivity.",
                      QGCompWarning, stacklevel=2)
    coefficients, covariance, iterations = _solve_logit(design, outcome, solver_weights)
    contrast = np.zeros(k)
    contrast[1:1+d] = 1
    psi = float(contrast @ coefficients)
    var_psi = float(contrast @ covariance @ contrast)
    if var_psi <= 0 or not np.isfinite(var_psi):
        raise ValueError("the joint log-odds variance is not positive and finite")
    se = float(np.sqrt(var_psi))
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = (float(psi - critical*se), float(psi + critical*se))
    if not np.all(np.isfinite(ci)):
        raise ValueError("binomial confidence limits are nonfinite")
    test_statistic = psi / se
    p_value = float(2 * special.ndtr(-abs(test_statistic)))
    pos_psi, pos_weights = _weights(coefficients[1:1+d], transformation.feature_names, True)
    neg_psi, neg_weights = _weights(coefficients[1:1+d], transformation.feature_names, False)
    diagnostics = MappingProxyType({
        "rank": rank, "condition_number": condition,
        "iterations": iterations, "converged": True,
        "case_weighted": case_weights is not None,
        "quantization": transformation.diagnostics,
    })
    fitted_values = special.expit(design @ coefficients)
    model_log_likelihood = float(np.sum(solver_weights * (
        special.xlogy(outcome, fitted_values)
        + special.xlogy(1-outcome, 1-fitted_values))))
    null_probability = float(np.average(outcome, weights=solver_weights))
    null_log_likelihood = float(np.sum(solver_weights * (
        special.xlogy(outcome, null_probability)
        + special.xlogy(1-outcome, 1-null_probability))))
    return BinomialResult(
        coefficients=_readonly(coefficients), coefficient_names=coefficient_names,
        covariance=_readonly(covariance), exposure_names=transformation.feature_names,
        covariate_names=names, fitted_values=_readonly(fitted_values),
        log_likelihood=model_log_likelihood,
        deviance=-2*model_log_likelihood,
        null_deviance=-2*null_log_likelihood,
        psi=psi, var_psi=var_psi, se=se, ci=ci, test_statistic=test_statistic,
        p_value=p_value, p_value_df=None, residual_df=n-k, alpha=alpha,
        pos_psi=pos_psi, neg_psi=neg_psi, pos_weights=pos_weights, neg_weights=neg_weights,
        diagnostics=diagnostics,
        case_weights=None if case_weights is None else _readonly(case_weights),
        _quantization=transformation,
        _covariate_columns=covariate_columns,
    )


def fit_binomial_formula(
    formula: str, data: Mapping[str, Any], exposures: Sequence[str], *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    missing: str = "raise", weights: Any = None,
) -> BinomialResult:
    """Fit additive Bernoulli-logit main effects from named columns.

    ``missing='drop'`` removes incomplete rows after estimating automatic
    exposure cutpoints from all observed exposure values, as in R qgcomp.
    """
    from ._formula import _drop_missing_rows, _numeric, _parse, formula_matrices

    schema = _parse(formula, exposures)
    if missing not in ("raise", "drop"):
        raise ValueError("missing must be 'raise' or 'drop'")
    automatic_cuts = missing == "drop" and breaks is None and q is not None
    if missing == "drop":
        data, weights, breaks, rows, original_n = _drop_missing_rows(
            data, schema, weights, q, breaks
        )
    x, z, schema = formula_matrices(data, schema, training=True)
    outcome = _numeric(data, schema.outcome, len(x.values))
    result = fit_binomial(outcome, x, z, q=q, breaks=breaks, alpha=alpha,
                          weights=weights)
    if automatic_cuts and breaks is not None:
        result = replace(result, _quantization=replace(result._quantization,
                                                      mode="quantile", q=int(q)))
    if missing == "raise":
        rows = tuple(range(len(x.values)))
        original_n = len(rows)
    schema = replace(schema, training_row_indices=rows, original_row_count=original_n)
    return replace(result, _formula_schema=schema)
