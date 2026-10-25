# SPDX-License-Identifier: GPL-2.0-or-later
"""Additive Poisson-log quantile g-computation.

The joint coefficient is a conditional log rate ratio. An optional exposure
time offset accounts for unequal observation periods.
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
                    case_weights: np.ndarray, log_time: np.ndarray) -> float:
    eta = design @ beta + log_time
    if np.max(eta) > 700:
        return -np.inf
    return float(np.sum(case_weights *
                        (y * eta - np.exp(eta) - special.gammaln(y + 1))))


def _solve_poisson(design: np.ndarray, y: np.ndarray,
                   case_weights: np.ndarray, log_time: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    _, k = design.shape
    beta = np.zeros(k)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_time = float(np.sum(case_weights * np.exp(log_time)))
        weighted_count = float(np.sum(case_weights * y))
    if not np.isfinite(weighted_time) or not np.isfinite(weighted_count):
        raise ValueError("weighted Poisson counts or observation times exceed the finite range")
    beta[0] = np.log(weighted_count / weighted_time)
    if not np.isfinite(beta[0]):
        raise ValueError("initial Poisson rate is at a numerical boundary")
    previous = _log_likelihood(design, y, beta, case_weights, log_time)
    for iteration in range(1, 101):
        eta = design @ beta + log_time
        mean = np.exp(eta)
        information = design.T @ ((case_weights * mean)[:, None] * design)
        score = design.T @ (case_weights * (y - mean))
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Poisson information is singular") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("Poisson update is nonfinite")
        scale = 1.0
        for _ in range(40):
            candidate = beta + scale * step
            likelihood = _log_likelihood(design, y, candidate, case_weights, log_time)
            if np.isfinite(likelihood) and likelihood >= previous - 1e-12:
                break
            scale /= 2.0
        else:
            raise ValueError("Poisson line search failed; check for a boundary fit")
        beta = candidate
        if np.max(np.abs(beta)) > 30:
            raise ValueError("Poisson coefficients diverged; check for a boundary fit")
        if (np.max(np.abs(scale * step)) <= 1e-11 * (1 + np.max(np.abs(beta)))
                and np.max(np.abs(score)) <= 1e-8 * max(1.0, float(np.mean(case_weights)))):
            break
        previous = likelihood
    else:
        raise ValueError("Poisson fit did not converge in 100 iterations")
    mean = np.exp(design @ beta + log_time)
    if not np.all(np.isfinite(mean)) or np.min(mean) < 1e-12:
        raise ValueError("fitted Poisson means are at numerical boundaries")
    information = design.T @ ((case_weights * mean)[:, None] * design)
    try:
        covariance = np.linalg.solve(information, np.eye(k))
    except np.linalg.LinAlgError as exc:
        raise ValueError("poisson covariance is singular") from exc
    covariance = (covariance + covariance.T) / 2.0
    if not np.all(np.isfinite(covariance)):
        raise ValueError("poisson covariance is nonfinite")
    return beta, covariance, iteration


@dataclass(frozen=True, eq=False)
class PoissonResult:
    """Conditional additive Poisson-log fit and joint log-rate effect."""

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
    exposure_time: np.ndarray | None = field(repr=False)
    _quantization: QuantizationResult = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    _exposure_time_column: str | None = field(default=None, repr=False)
    _formula_schema: Any = field(default=None, repr=False)
    reference_version: str = "2.19.6"
    family: str = "poisson"
    link: str = "log"

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
    def rate_ratio(self) -> float:
        """Conditional rate multiplier for one score increase in every exposure."""
        return float(np.exp(self.psi))

    @property
    def rate_ratio_ci(self) -> tuple[float, float]:
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

    def predict(self, exposures: Any, covariates: Any = None, *,
                exposure_time: Any = None) -> np.ndarray:
        """Predict expected counts; supply time for an offset fit."""
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
        if self.exposure_time is None:
            if exposure_time is not None:
                raise ValueError("this model was fitted without exposure_time")
            log_time = np.zeros(n)
        else:
            if exposure_time is None:
                raise ValueError("exposure_time is required for this fitted model")
            log_time = np.log(_positive_time(exposure_time, n))
        eta = design @ self.coefficients + log_time
        if np.max(eta) > 700:
            raise ValueError("predicted Poisson means exceed the finite range")
        prediction = np.exp(eta)
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn("Some exposures are outside their training ranges; fitted cutpoints are reused.",
                          QGCompWarning, stacklevel=2)
        return _readonly(prediction)

    def predict_data(self, data: Mapping[str, Any]) -> np.ndarray:
        if self._formula_schema is None:
            raise ValueError("predict_data requires a fit_poisson_formula result")
        from ._formula import formula_matrices

        x, z = formula_matrices(data, self._formula_schema)
        time = None
        if self._exposure_time_column is not None:
            from ._formula import _numeric
            time = _numeric(data, self._exposure_time_column, len(x.values))
        return self.predict(x, z, exposure_time=time)

    def transform_data(self, data: Mapping[str, Any]) -> QuantizationResult:
        if self._formula_schema is None:
            raise ValueError("transform_data requires a fit_poisson_formula result")
        from ._formula import formula_exposures

        return self.transform(formula_exposures(data, self._formula_schema))

    def summary(self, digits: int = 6) -> str:
        """Show the saved log-rate contrast and its normal interval."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: additive Poisson log\n"
                "Joint effect: one score category in every exposure; covariates fixed.\n"
                f"Log rate ratio: {fmt(self.psi)}; SE: {fmt(self.se)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI: ({fmt(self.ci[0])}, {fmt(self.ci[1])})\n"
                f"{'Rate' if self.exposure_time is not None else 'Mean'} ratio: {fmt(self.rate_ratio)}\n"
                f"Two-sided normal z p-value (R convention): {fmt(self.p_value)}\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'not supplied'}; model-based SE.\n"
                + ("This rate ratio is conditional on the adjustment covariates and compares "
                   "equal exposure times; fitted counts use the supplied time offset."
                   if self.exposure_time is not None else
                   "This mean ratio is conditional on the adjustment covariates and equal observation periods."))


def _positive_time(value: Any, n: int) -> np.ndarray:
    """Validate observation time before taking its logarithm."""
    result = _real_array(value, "exposure_time")
    raw = np.asanyarray(value)
    if raw.dtype.kind == "b":
        raise ValueError("exposure_time must contain positive real numbers")
    if result.ndim != 1 or len(result) != n or np.any(result <= 0):
        raise ValueError("exposure_time must be a positive vector with one value per row")
    return result


def fit_poisson(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None, exposure_time: Any = None,
) -> PoissonResult:
    """Fit an additive Poisson-log model and sum exposure coefficients.

    ``y`` must contain nonnegative integer counts. Inputs are complete,
    real and finite. Supply positive exposure_time for unequal periods.
    Positive case weights act as GLM prior weights. Exposure cuts ignore weights.
    The sum is a conditional log rate ratio. Its confidence interval uses
    normal limits and a normal z p-value, following R qgcomp 2.19.6.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    outcome = _real_array(y, "y")
    if (outcome.ndim != 1 or not outcome.size or not np.all(outcome >= 0)
            or not np.all(outcome == np.floor(outcome))):
        raise ValueError("Poisson y must be a nonempty vector of nonnegative integer counts")
    if np.sum(outcome) == 0:
        raise ValueError("Poisson y must have a positive total count")
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if len(outcome) != n:
        raise ValueError("y and exposures must have the same number of rows")
    time = None if exposure_time is None else _positive_time(exposure_time, n)
    log_time = np.zeros(n) if time is None else np.log(time)
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
        raise ValueError("the poisson model needs more rows than coefficients")
    try:
        singular_values = np.linalg.svd(design * np.sqrt(solver_weights)[:, None],
                                        compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the poisson design singular-value decomposition failed") from exc
    tolerance = max(n, k) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != k:
        raise ValueError("the poisson design is rank deficient")
    condition = float(singular_values[0] / singular_values[-1])
    if condition > _CONDITION_WARNING:
        warnings.warn("The poisson design is ill conditioned; inspect numerical sensitivity.",
                      QGCompWarning, stacklevel=2)
    coefficients, covariance, iterations = _solve_poisson(design, outcome, solver_weights,
                                                         log_time)
    contrast = np.zeros(k)
    contrast[1:1+d] = 1
    psi = float(contrast @ coefficients)
    var_psi = float(contrast @ covariance @ contrast)
    if var_psi <= 0 or not np.isfinite(var_psi):
        raise ValueError("the joint log-rate variance is not positive and finite")
    se = float(np.sqrt(var_psi))
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = (float(psi - critical*se), float(psi + critical*se))
    if not np.all(np.isfinite(ci)):
        raise ValueError("poisson confidence limits are nonfinite")
    test_statistic = psi / se
    p_value = float(2 * special.ndtr(-abs(test_statistic)))
    pos_psi, pos_weights = _weights(coefficients[1:1+d], transformation.feature_names, True)
    neg_psi, neg_weights = _weights(coefficients[1:1+d], transformation.feature_names, False)
    diagnostics = MappingProxyType({
        "rank": rank, "condition_number": condition,
        "iterations": iterations, "converged": True,
        "case_weighted": case_weights is not None,
        "exposure_time_offset": time is not None,
        "quantization": transformation.diagnostics,
    })
    fitted_values = np.exp(design @ coefficients + log_time)
    log_factorial = special.gammaln(outcome+1)
    saturated_log_likelihood = float(np.sum(solver_weights * (
        special.xlogy(outcome, outcome)-outcome-log_factorial)))
    model_log_likelihood = float(np.sum(solver_weights * (
        special.xlogy(outcome, fitted_values)-fitted_values-log_factorial)))
    null_rate = float(np.sum(solver_weights*outcome)
                      / np.sum(solver_weights*np.exp(log_time)))
    null_mean = null_rate*np.exp(log_time)
    null_log_likelihood = float(np.sum(solver_weights * (
        special.xlogy(outcome, null_mean)-null_mean-log_factorial)))
    return PoissonResult(
        coefficients=_readonly(coefficients), coefficient_names=coefficient_names,
        covariance=_readonly(covariance), exposure_names=transformation.feature_names,
        covariate_names=names, fitted_values=_readonly(fitted_values),
        log_likelihood=model_log_likelihood,
        deviance=max(0.0, 2*(saturated_log_likelihood-model_log_likelihood)),
        null_deviance=max(0.0, 2*(saturated_log_likelihood-null_log_likelihood)),
        psi=psi, var_psi=var_psi, se=se, ci=ci, test_statistic=test_statistic,
        p_value=p_value, p_value_df=None, residual_df=n-k, alpha=alpha,
        pos_psi=pos_psi, neg_psi=neg_psi, pos_weights=pos_weights, neg_weights=neg_weights,
        diagnostics=diagnostics,
        case_weights=None if case_weights is None else _readonly(case_weights),
        exposure_time=None if time is None else _readonly(time),
        _quantization=transformation,
        _covariate_columns=covariate_columns,
    )


def fit_poisson_formula(
    formula: str, data: Mapping[str, Any], exposures: Sequence[str], *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    missing: str = "raise", weights: Any = None, exposure_time: str | None = None,
) -> PoissonResult:
    """Fit additive Poisson-log main effects from named columns.

    ``missing='drop'`` removes incomplete rows after estimating automatic
    exposure cutpoints from all observed exposure values, as in R qgcomp.
    """
    from ._formula import _drop_missing_rows, _numeric, _parse, formula_matrices

    schema = _parse(formula, exposures)
    if exposure_time is not None:
        if not isinstance(exposure_time, str) or not exposure_time.isidentifier():
            raise ValueError("exposure_time must name one numeric data column")
        if exposure_time == schema.outcome or exposure_time in schema.exposure_names:
            raise ValueError("exposure_time must differ from outcome and exposures")
    if missing not in ("raise", "drop"):
        raise ValueError("missing must be 'raise' or 'drop'")
    automatic_cuts = missing == "drop" and breaks is None and q is not None
    if missing == "drop":
        data, weights, breaks, rows, original_n = _drop_missing_rows(
            data, schema, weights, q, breaks,
            extra_numeric=() if exposure_time is None else (exposure_time,)
        )
    x, z, schema = formula_matrices(data, schema, training=True)
    outcome = _numeric(data, schema.outcome, len(x.values))
    time = None if exposure_time is None else _numeric(data, exposure_time, len(x.values))
    result = fit_poisson(outcome, x, z, q=q, breaks=breaks, alpha=alpha,
                         weights=weights, exposure_time=time)
    if automatic_cuts and breaks is not None:
        result = replace(result, _quantization=replace(result._quantization,
                                                      mode="quantile", q=int(q)))
    if missing == "raise":
        rows = tuple(range(len(x.values)))
        original_n = len(rows)
    schema = replace(schema, training_row_indices=rows, original_row_count=original_n)
    return replace(result, _formula_schema=schema, _exposure_time_column=exposure_time)
