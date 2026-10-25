# SPDX-License-Identifier: GPL-2.0-or-later
"""Additive Gaussian quantile g-computation and exposure scoring.

Method: Keil et al. (2020), doi:10.1289/EHP5838.
Reference conventions: Alexander Keil's R qgcomp 2.19.6, GPL (>= 2),
commit 579e5093c894babaa51c9986704c8ffb6bf2825b. The
method and numerical conventions are documented in ``docs/method.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping
import warnings

import numpy as np
import scipy
from scipy import special


_SENTINEL = 1e64
_MAX_EXACT_SCORE = 2**53
_MAX_QUANTILES = 100_000
_CONDITION_WARNING = 1e8


class QGCompWarning(RuntimeWarning):
    """A fit or prediction has a limitation recorded in its diagnostics."""


def _readonly(value: Any) -> np.ndarray:
    """Own the data through an immutable bytes buffer, not the caller's array."""
    a = np.array(value, dtype=np.float64, order="C", copy=True)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _check_mask(value: Any, label: str, prefix: tuple[int, ...] = ()) -> None:
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        locations = [prefix + tuple(int(v) for v in row)
                     for row in np.argwhere(np.ma.getmaskarray(value))[:5]]
        raise ValueError(f"{label} contains masked values at indices {locations}")


def _real_array(value: Any, label: str) -> np.ndarray:
    # np.asarray discards a MaskedArray's mask and exposes its backing data.
    # Reject missing entries before conversion, even when those data are finite.
    pending = [(value, ())]
    visited = set()
    while pending:
        part, prefix = pending.pop()
        _check_mask(part, label, prefix)
        if (isinstance(part, Sequence) and not isinstance(part, (str, bytes))
                and id(part) not in visited):
            visited.add(id(part))
            pending.extend((entry, prefix + (i,)) for i, entry in enumerate(part))
    try:
        raw = np.asanyarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a rectangular numeric array") from exc
    # An array protocol may itself return a masked array. Check before casting
    # to an ordinary float array, which would discard that returned mask.
    _check_mask(raw, label)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{label} must contain real numeric values, not strings or objects")
    with np.errstate(over="ignore", invalid="ignore"):
        a = np.array(raw, dtype=np.float64, copy=True)
    bad = np.argwhere(~np.isfinite(a))
    if bad.size:
        locations = [tuple(int(v) for v in row) for row in bad[:5]]
        raise ValueError(f"{label} contains nonfinite values at indices {locations}")
    if a.ndim == 0 and not np.isfinite(a):
        raise ValueError(f"{label} contains a nonfinite value")
    return a


def _matrix(value: Any, label: str) -> tuple[np.ndarray, tuple[str, ...] | None]:
    a = _real_array(value, label)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{label} must be one- or two-dimensional")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{label} must have at least one row and one column")
    columns = getattr(value, "columns", None)
    names = None if columns is None else tuple(str(name) for name in columns)
    if names is not None:
        if len(names) != a.shape[1]:
            raise ValueError(f"{label} column labels do not match its shape")
        if len(set(names)) != len(names):
            raise ValueError(f"{label} contains duplicate column identifiers")
    return a, names


def _column_names(names: tuple[str, ...] | None, count: int, prefix: str) -> tuple[str, ...]:
    return names if names is not None else tuple(f"{prefix}{i + 1}" for i in range(count))


def _check_columns(
    actual: tuple[str, ...] | None, expected: tuple[str, ...] | None, label: str
) -> None:
    # Arrays are positional. A fit made with labeled columns requires the
    # same labels at prediction, so dropping labels cannot hide reordering.
    if expected is not None and actual != expected:
        raise ValueError(f"{label} must retain the fitted column labels and order: {expected}")


def _score_check(a: np.ndarray) -> None:
    if np.any(a < 0) or np.any(a >= _MAX_EXACT_SCORE) or np.any(a != np.floor(a)):
        raise ValueError(
            "q=None requires nonnegative integer scores smaller than 2**53; "
            "arbitrary raw numeric passthrough is not supported"
        )


def _explicit_breaks(value: Any, count: int) -> tuple[np.ndarray, ...]:
    # A single vector is a convenience for one exposure; otherwise require
    # one complete vector, including both outer endpoints, per exposure.
    single = None
    if count == 1:
        try:
            candidate = np.asarray(value)
            if candidate.ndim == 1 and candidate.dtype.kind in "biuf":
                single = (value,)
        except (TypeError, ValueError):
            pass
    try:
        vectors = single if single is not None else tuple(value)
    except TypeError as exc:
        raise ValueError("breaks must contain one numeric vector per exposure") from exc
    if len(vectors) != count:
        raise ValueError(f"breaks must contain {count} vector(s), one per exposure")
    out = []
    for j, vector in enumerate(vectors):
        cut = _real_array(vector, f"breaks[{j}]")
        if cut.ndim != 1 or cut.size < 2:
            raise ValueError(f"breaks[{j}] must be a vector with at least two endpoints")
        if not np.all(cut[1:] > cut[:-1]):
            raise ValueError(f"breaks[{j}] must be strictly increasing without duplicates")
        out.append(_readonly(cut))
    return tuple(out)


def _apply_breaks(a: np.ndarray, breaks: tuple[np.ndarray, ...]) -> np.ndarray:
    scores = np.empty_like(a)
    for j, cut in enumerate(breaks):
        outside = np.flatnonzero((a[:, j] < cut[0]) | (a[:, j] > cut[-1]))
        if outside.size:
            raise ValueError(
                f"exposures column {j} is outside its break endpoints at rows "
                f"{tuple(int(i) for i in outside[:5])}"
            )
        # Searching interior cuts from the left gives right-closed bins,
        # with both outer endpoints included, as in R cut(include.lowest=TRUE).
        scores[:, j] = np.searchsorted(cut[1:-1], a[:, j], side="left")
    return scores


def _type7_quantiles(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Type-7 interpolation with R's one-based floating-point arithmetic."""
    ordered = np.sort(values)
    index = 1.0 + (ordered.size - 1) * probabilities
    lower = np.floor(index).astype(np.intp)
    upper = np.ceil(index).astype(np.intp)
    result = ordered[lower - 1].copy()
    # Algebraically equivalent zero-based interpolation can round a cut to
    # the other side of an observation. Preserve R's arithmetic order and its
    # equal-neighbor shortcut because those cuts determine discrete scores.
    interpolate = (index > lower) & (ordered[upper - 1] != result)
    fraction = (index - lower)[interpolate]
    result[interpolate] = ((1.0 - fraction) * result[interpolate]
                           + fraction * ordered[upper[interpolate] - 1])
    return result


def _quantization_diagnostics(
    a: np.ndarray,
    scores: np.ndarray,
    breaks: tuple[np.ndarray, ...] | None,
    training_min: np.ndarray,
    training_max: np.ndarray,
    names: tuple[str, ...],
    q: int | None,
) -> Mapping[str, Any]:
    outside_training = (a < training_min) | (a > training_max)
    counts = tuple(int(v) for v in np.sum(outside_training, axis=0))
    observed = tuple(int(np.unique(scores[:, j]).size) for j in range(a.shape[1]))
    if breaks is None:
        # A prequantized input does not specify the full category domain.
        empty = None
        reduced = ()
    else:
        empty = tuple(
            tuple(int(v) for v in np.setdiff1d(np.arange(len(cut) - 1), scores[:, j]))
            for j, cut in enumerate(breaks)
        )
        reduced = tuple(names[j] for j, cut in enumerate(breaks) if q is not None and len(cut) - 1 < q)
    return MappingProxyType(
        {
            "extrapolated_rows": tuple(int(v) for v in np.flatnonzero(np.any(outside_training, axis=1))),
            "extrapolated_columns": tuple(name for name, count in zip(names, counts) if count),
            "extrapolated_counts": counts,
            "observed_category_counts": observed,
            "empty_bins": empty,
            "reduced_category_columns": reduced,
        }
    )


@dataclass(frozen=True, eq=False)
class QuantizationResult:
    """Scores and the immutable fitted transformation.

    ``breaks`` includes outer endpoints and is ``None`` in prequantized mode.
    ``category_counts`` describes fitted bins, or the number of observed
    distinct training scores when no category domain was supplied. Prediction
    keeps these counts; diagnostics separately report new observed counts.
    ``transform`` does not recompute cuts. Labeled inputs must retain their
    labels and column order; unlabeled training arrays use positional columns.
    """

    scores: np.ndarray = field(repr=False)
    breaks: tuple[np.ndarray, ...] | None = field(repr=False)
    category_counts: tuple[int, ...]
    mode: str
    q: int | None
    feature_names: tuple[str, ...]
    training_min: np.ndarray = field(repr=False)
    training_max: np.ndarray = field(repr=False)
    diagnostics: Mapping[str, Any] = field(repr=False)
    _columns: tuple[str, ...] | None = field(repr=False)

    def transform(self, exposures: Any) -> QuantizationResult:
        """Apply the fitted map and report extrapolation; never refit cuts."""
        a, columns = _matrix(exposures, "exposures")
        if a.shape[1] != len(self.feature_names):
            raise ValueError("exposures must have the fitted number of columns")
        _check_columns(columns, self._columns, "exposures")
        if self.breaks is None:
            _score_check(a)
            scores = a
        else:
            scores = _apply_breaks(a, self.breaks)
        diagnostics = _quantization_diagnostics(
            a, scores, self.breaks, self.training_min, self.training_max, self.feature_names, self.q
        )
        return QuantizationResult(
            scores=_readonly(scores),
            breaks=self.breaks,
            category_counts=self.category_counts,
            mode=self.mode,
            q=self.q,
            feature_names=self.feature_names,
            training_min=self.training_min,
            training_max=self.training_max,
            diagnostics=diagnostics,
            _columns=self._columns,
        )


def quantize(exposures: Any, *, q: int | None = 4, breaks: Any = None) -> QuantizationResult:
    """Convert exposure measurements to ordered category scores.

    Parameters
    ----------
    exposures : array-like, shape (n, d) or (n,)
        Measurements for n observations and d exposures. A vector represents
        one exposure. Values must be real, finite and complete. Named columns
        are retained and must have unique names.
    q : int or None, default 4
        Requested number of quantile groups, from 2 to 100,000. Ties can
        reduce the actual number. Booleans are not accepted. With ``q=None``
        and no breaks, inputs must already be nonnegative integer scores
        smaller than 2**53; arbitrary raw measurements are not accepted.
    breaks : array-like or sequence of array-like, optional
        Category boundaries, including both outer endpoints, for each
        exposure. Supply one vector for a single exposure or one vector per
        exposure for multiple columns. Each vector must be finite and strictly
        increasing. Supplied breaks take priority over ``q``.

    Returns
    -------
    QuantizationResult
        Read-only scores, saved boundaries, category counts and diagnostics.
        Its ``transform`` method applies the saved boundaries to new data.

    Raises
    ------
    ValueError
        If inputs or boundaries are invalid, values fall outside supplied
        boundaries, an exposure is constant in automatic mode, or automatic
        input magnitudes are at least 1e64.

    Notes
    -----
    Follows R qgcomp 2.19.6: calculate type-7 quantiles, remove duplicates from
    the full sequence and replace its outer endpoints by +/-1e64. A value on
    an interior boundary belongs to the lower category. Ties can leave one
    category, which cannot then be used as an identifiable regression column.
    The 100,000-group limit bounds memory allocation in this implementation;
    it is not a restriction of the statistical method.

    Examples
    --------
    >>> quantize([0, 1, 2, 3], q=2).scores.ravel().tolist()
    [0.0, 0.0, 1.0, 1.0]
    """
    a, columns = _matrix(exposures, "exposures")
    names = _column_names(columns, a.shape[1], "x")
    fitted_q = None
    if breaks is not None:
        fitted_breaks = _explicit_breaks(breaks, a.shape[1])
        scores = _apply_breaks(a, fitted_breaks)
        mode = "explicit"
    elif q is None:
        _score_check(a)
        fitted_breaks = None
        scores = a
        mode = "prequantized"
    else:
        if isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral) or q < 2:
            raise ValueError("q must be an integer >= 2, or None for prequantized scores")
        if q > _MAX_QUANTILES:
            raise ValueError(
                f"q cannot exceed {_MAX_QUANTILES} in this version; "
                "automatic quantile-grid allocation is bounded"
            )
        fitted_q = int(q)
        if np.any(np.abs(a) >= _SENTINEL):
            raise ValueError("automatic quantization requires finite exposure magnitudes below 1e64")
        # Match R seq(0, 1, by=1/q), including endpoint rounding.
        probabilities = np.minimum(np.arange(fitted_q + 1, dtype=float) * (1.0 / fitted_q), 1.0)
        cut_list = []
        for j in range(a.shape[1]):
            cut = np.unique(_type7_quantiles(a[:, j], probabilities))
            if cut.size < 2:
                raise ValueError(f"exposures column {j} is constant; automatic quantization is undefined")
            cut[0], cut[-1] = -_SENTINEL, _SENTINEL
            cut_list.append(_readonly(cut))
        fitted_breaks = tuple(cut_list)
        scores = _apply_breaks(a, fitted_breaks)
        mode = "quantile"
    category_counts = (
        tuple(int(np.unique(scores[:, j]).size) for j in range(a.shape[1]))
        if fitted_breaks is None
        else tuple(len(cut) - 1 for cut in fitted_breaks)
    )
    training_min = _readonly(np.min(a, axis=0))
    training_max = _readonly(np.max(a, axis=0))
    diagnostics = _quantization_diagnostics(
        a, scores, fitted_breaks, training_min, training_max, names, fitted_q
    )
    return QuantizationResult(
        scores=_readonly(scores),
        breaks=fitted_breaks,
        category_counts=category_counts,
        mode=mode,
        q=fitted_q,
        feature_names=names,
        training_min=training_min,
        training_max=training_max,
        diagnostics=diagnostics,
        _columns=columns,
    )


@dataclass(frozen=True, eq=False)
class QGCompResult:
    """Additive Gaussian fit and descriptive mixture summaries.

    Coefficients are ordered as intercept, exposures, then covariates.
    Covariance uses full-model residual degrees of freedom. For reference
    compatibility, ``ci`` uses normal critical values while ``p_value`` uses
    a Student t law with ``n - 1 - number_of_exposures`` degrees of freedom.
    Therefore the interval need not invert the reported test.

    Weights are separate within-sign proportions, not selection probabilities
    or causal importance. Numerical-zero residual variation gives zero SE
    and a point interval, but NaN test statistic/p-value and an explicit
    ``inference_degenerate`` diagnostic. No inferential claim is then made.
    """

    coefficients: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray = field(repr=False)
    exposure_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    fitted_values: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    case_weights: np.ndarray | None = field(repr=False)
    residual_df: int
    residual_variance: float
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    alpha: float
    test_statistic: float
    p_value: float
    p_value_df: int
    pos_psi: float
    neg_psi: float
    pos_weights: Mapping[str, float]
    neg_weights: Mapping[str, float]
    zero_exposures: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(repr=False)
    library_versions: Mapping[str, str] = field(repr=False)
    _quantization: QuantizationResult = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    _formula_schema: Any = field(default=None, repr=False)
    reference_version: str = "2.19.6"
    inference: str = "qgcomp_2.19.6"

    @property
    def intercept(self) -> float:
        return float(self.coefficients[0])

    @property
    def beta_exposures(self) -> np.ndarray:
        return self.coefficients[1 : 1 + len(self.exposure_names)]

    @property
    def beta_covariates(self) -> np.ndarray:
        return self.coefficients[1 + len(self.exposure_names) :]

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def category_counts(self) -> tuple[int, ...]:
        return self._quantization.category_counts

    @property
    def quantization_mode(self) -> str:
        return self._quantization.mode

    @property
    def q(self) -> int | None:
        return self._quantization.q

    @property
    def formula(self) -> str | None:
        """Original restricted formula, when fitted from named data."""
        return None if self._formula_schema is None else self._formula_schema.formula

    @property
    def categorical_levels(self) -> Mapping[str, tuple[str, ...]]:
        """Saved levels, ordered with each reference level first."""
        if self._formula_schema is None:
            return MappingProxyType({})
        return MappingProxyType(dict(self._formula_schema.category_levels))

    @property
    def training_row_indices(self) -> tuple[int, ...] | None:
        """Original zero-based row positions retained by a named-data fit."""
        if self._formula_schema is None:
            return None
        return self._formula_schema.training_row_indices

    @property
    def original_row_count(self) -> int | None:
        """Number of rows provided to a named-data fit before any omission."""
        if self._formula_schema is None:
            return None
        return self._formula_schema.original_row_count

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    def summary(self, *, digits: int = 6) -> str:
        """Return a plain-text summary of this saved Gaussian fit.

        Parameters
        ----------
        digits : int, default 6
            Significant digits for estimates, uncertainty and sign weights.
            Must be between 1 and 17, inclusive; booleans are not accepted.
            Scientific notation preserves small nonzero values.

        Returns
        -------
        str
            Model size, joint effect and inference, descriptive sign weights,
            coefficient estimates, and explanations of recorded diagnostics.
            Use ``print(result.summary())`` to display the text. This method
            does not print, refit, change the result or test model assumptions.

        Raises
        ------
        ValueError
            If ``digits`` is not an integer from 1 through 17.

        Notes
        -----
        Full-precision values remain in the result attributes. The interval
        uses normal critical values; the p-value uses the stored R-reference
        degrees of freedom. Sign weights are descriptive, not uncertainty
        estimates. Undefined test statistics and p-values appear as ``NA``.
        Column names are quoted with control characters escaped. The text is
        intended for readers, not as a machine-readable serialization format.
        """
        from ._summary import format_summary

        return format_summary(self, digits=digits)

    def transform(self, exposures: Any) -> QuantizationResult:
        """Return scores and diagnostics using the fitted exposure map."""
        return self._quantization.transform(exposures)

    def predict(self, exposures: Any, covariates: Any = None) -> np.ndarray:
        """Predict conditional means; report exposure extrapolation by warning.

        This method returns neither individual-outcome intervals nor causal
        intervention values. Supplied labeled columns must retain fit order.
        """
        transformed = self.transform(exposures)
        n = transformed.scores.shape[0]
        if self.covariate_names:
            if covariates is None:
                raise ValueError("covariates are required because this model was fitted with them")
            z, columns = _matrix(covariates, "covariates")
            if z.shape != (n, len(self.covariate_names)):
                raise ValueError("covariates must match prediction rows and fitted column count")
            _check_columns(columns, self._covariate_columns, "covariates")
        else:
            if covariates is not None:
                raise ValueError("this model was fitted without covariates")
            z = np.empty((n, 0), dtype=np.float64)
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn(
                "Some exposures are outside their training ranges; fitted cutpoints are reused. "
                "Call transform() for row and column diagnostics.",
                QGCompWarning,
                stacklevel=2,
            )
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = self.intercept + transformed.scores @ self.beta_exposures + z @ self.beta_covariates
        if not np.all(np.isfinite(predictions)):
            raise ValueError("prediction is nonfinite; check the scale of the new inputs")
        return _readonly(predictions)

    def predict_data(self, data: Mapping[str, Any]) -> np.ndarray:
        """Predict from named formula columns using saved categorical levels."""
        if self._formula_schema is None:
            raise ValueError("predict_data requires a fit_formula result")
        from ._formula import formula_matrices

        exposures, covariates = formula_matrices(data, self._formula_schema)
        return self.predict(exposures, covariates)

    def transform_data(self, data: Mapping[str, Any]) -> QuantizationResult:
        """Apply saved exposure cuts to named formula columns."""
        if self._formula_schema is None:
            raise ValueError("transform_data requires a fit_formula result")
        from ._formula import formula_exposures

        exposures = formula_exposures(data, self._formula_schema)
        return self.transform(exposures)


def _weights(beta: np.ndarray, names: tuple[str, ...], positive: bool) -> tuple[float, Mapping[str, float]]:
    indices = np.flatnonzero(beta > 0 if positive else beta < 0)
    partial = float(np.sum(beta[indices]))
    if not indices.size:
        return partial, MappingProxyType({})
    if not np.isfinite(partial):
        raise ValueError("a sign-specific coefficient sum is nonfinite")
    values = np.abs(beta[indices]) / abs(partial)
    order = np.argsort(-values, kind="stable")
    return partial, MappingProxyType({names[int(indices[i])]: float(values[i]) for i in order})


def fit(
    y: Any,
    exposures: Any,
    covariates: Any = None,
    *,
    q: int | None = 4,
    breaks: Any = None,
    alpha: float = 0.05,
    weights: Any = None,
) -> QGCompResult:
    """Estimate a joint exposure effect for a continuous outcome.

    Fits additive Gaussian regression: exposure-score effects enter a linear
    model and are added together. An intercept is always included.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome values for n observations. Values must be real, finite and
        complete, with one value per row of ``exposures``.
    exposures : array-like, shape (n, d) or (n,)
        The d exposure columns whose coefficients form the joint effect.
        A vector represents one exposure. Grouping follows ``quantize``.
    covariates : array-like, shape (n, c) or (n,), optional
        Numeric adjustment variables, with c columns. Their coefficients are
        fitted but not included in the joint effect. Encode categorical
        variables before fitting. Do not supply another intercept column.
    q : int or None, default 4
        Number of quantile groups. Use ``None`` for already grouped integer
        scores. See ``quantize`` for input limits and tie handling.
    breaks : array-like or sequence of array-like, optional
        Complete category boundaries per exposure. Takes priority over ``q``.
    alpha : float, default 0.05
        Significance level strictly between zero and one. Confidence limits
        have nominal coverage ``1 - alpha``; 0.05 requests a 95% interval.
    weights : array-like, shape (n,), optional
        Positive finite case weights for the Gaussian regression. One weight
        is required per training row. Exposure quantiles remain unweighted.
        These are not the descriptive sign weights in the result.

    Returns
    -------
    QGCompResult
        Fitted coefficients and covariance, the joint estimate ``psi``, its
        standard error and confidence limits, sign summaries, residuals and
        diagnostics. Coefficients are ordered as intercept, exposures, then
        covariates. Result arrays and mappings are read-only. ``predict``
        returns fitted mean outcomes using the saved exposure boundaries.

    Raises
    ------
    ValueError
        If inputs are invalid, model columns are linearly dependent, there
        are too few observations for residual variance, or finite coefficient,
        covariance and joint-effect estimates cannot be calculated.

    Warns
    -----
    QGCompWarning
        If coefficients are sensitive to small numerical changes, or variation
        is too small for inference. At numerical-zero variation, the standard
        error is zero and the interval is a single point; the test statistic
        and p-value are ``NaN`` (undefined), with ``inference_degenerate`` set
        in diagnostics. These undefined values are intentional, not test evidence.

    Notes
    -----
    The joint estimate is the sum of exposure-score coefficients. Its variance
    includes covariance between those coefficients. Model-based uncertainty
    assumes independent observations and constant conditional residual variance.
    A causal interpretation needs further study assumptions.

    Confidence limits use normal critical values, as in R qgcomp 2.19.6. The
    reference p-value uses a Student t distribution with ``n - 1 - d`` degrees
    of freedom; residual variance instead uses ``n - 1 - d - c``. The interval
    and test therefore need not agree exactly at a significance threshold.

    Rows and variables are not silently removed. Nonlinear terms, interactions,
    other outcome families, clustered observations,
    penalization and bootstrap inference are not supported by this interface.
    Case weights use the R reference's model-based Gaussian covariance. They
    do not provide complex-survey or cluster-robust inference. Zero weights
    and missing rows are not supported.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    outcome = _real_array(y, "y")
    if outcome.ndim != 1 or outcome.size == 0:
        raise ValueError("y must be a nonempty one-dimensional vector")
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if outcome.size != n:
        raise ValueError("y and exposures must have the same number of rows")
    if weights is None:
        case_weights = None
        sqrt_weights = None
    else:
        case_weights = _real_array(weights, "weights")
        if case_weights.ndim != 1 or case_weights.size != n:
            raise ValueError("weights must be a one-dimensional vector with one value per row")
        if np.any(case_weights <= 0):
            raise ValueError("weights must be strictly positive")
        sqrt_weights = np.sqrt(case_weights)
    if covariates is None:
        z = np.empty((n, 0), dtype=np.float64)
        covariate_columns = None
    else:
        z, covariate_columns = _matrix(covariates, "covariates")
        if z.shape[0] != n:
            raise ValueError("covariates and exposures must have the same number of rows")
    covariate_names = _column_names(covariate_columns, z.shape[1], "z")
    coefficient_names = ("(Intercept)",) + transformation.feature_names + covariate_names
    if len(set(coefficient_names)) != len(coefficient_names):
        raise ValueError("coefficient identifiers must be distinct across exposures, covariates and intercept")
    design = np.column_stack((np.ones(n), transformation.scores, z))
    weighted_design = design if sqrt_weights is None else design * sqrt_weights[:, None]
    if not np.all(np.isfinite(weighted_design)):
        raise ValueError("weighted design is nonfinite; check weight and input scales")
    k = design.shape[1]
    if n <= k:
        raise ValueError("the model requires more observations than coefficients (positive residual df)")
    try:
        u, singular_values, vh = np.linalg.svd(weighted_design, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design matrix singular-value decomposition failed") from exc
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("the design matrix has nonfinite numerical scale")
    eps = np.finfo(np.float64).eps
    tolerance = float(max(n, k) * eps * singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != k:
        raise ValueError(
            f"the design is rank deficient (rank {rank}, {k} columns); "
            "check constant or collinear quantized exposures and covariates"
        )
    condition_number = float(singular_values[0] / singular_values[-1])
    ill_conditioned = bool(condition_number > _CONDITION_WARNING)
    if ill_conditioned:
        warnings.warn(
            f"The design condition number is {condition_number:.3g}; inspect numerical sensitivity.",
            QGCompWarning,
            stacklevel=2,
        )
    residual_df = n - k
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # With an intercept, centering y changes only that coefficient.
            # Keep the solve and residuals on the centered scale so a large
            # outcome offset cannot erase resolvable noise or slope precision.
            outcome_center = (float(np.mean(outcome)) if case_weights is None
                              else float(np.average(outcome, weights=case_weights)))
            centered_outcome = outcome - outcome_center
            weighted_outcome = (centered_outcome if sqrt_weights is None
                                else centered_outcome * sqrt_weights)
            centered_coefficients = vh.T @ ((u.T @ weighted_outcome) / singular_values)
            centered_fitted = design @ centered_coefficients
            residuals = centered_outcome - centered_fitted
            weighted_residuals = residuals if sqrt_weights is None else residuals * sqrt_weights
            coefficients = centered_coefficients.copy()
            coefficients[0] += outcome_center
            fitted = centered_fitted + outcome_center
            residual_norm = float(np.linalg.norm(weighted_residuals))
            absolute_fit_scale = np.abs(design) @ np.abs(centered_coefficients)
            if sqrt_weights is not None:
                absolute_fit_scale *= sqrt_weights
            # Treat residual variation below a documented floating-point
            # backward-error scale as numerical zero, not significant evidence.
            residual_tolerance = float(
                10.0 * max(n, k) * eps
                * (np.linalg.norm(weighted_outcome)
                   + np.linalg.norm(absolute_fit_scale))
            )
            degenerate = bool(residual_norm <= residual_tolerance)
            residual_variance = 0.0 if degenerate else float((residual_norm**2) / residual_df)
            scaled_vectors = vh.T / singular_values
            covariance = residual_variance * (scaled_vectors @ scaled_vectors.T)
            covariance = (covariance + covariance.T) / 2.0
            contrast = np.zeros(k)
            contrast[1 : 1 + d] = 1.0
            psi = float(contrast @ coefficients)
            # Equivalent to a.T @ covariance @ a, but does not subtract
            # large covariance entries for a well-estimated joint contrast.
            contrast_factor = (vh @ contrast) / singular_values
            var_psi = float(residual_variance * (contrast_factor @ contrast_factor))
            covariance_contrast_variance = float(contrast @ covariance @ contrast)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("the fit exceeded finite double-precision arithmetic; check input scale") from exc
    if not all(np.all(np.isfinite(v)) for v in (coefficients, fitted, residuals, covariance, psi, var_psi, residual_tolerance)):
        raise ValueError("the fitted coefficients, covariance or residuals are nonfinite")
    # The factor calculation above is nonnegative. Keep the literal covariance
    # contraction and its roundoff bound as diagnostics, since cancellation in
    # that less stable calculation can matter for nearly collinear exposures.
    variance_roundoff_bound = float(100.0 * eps * np.sum(np.abs(covariance[1 : 1 + d, 1 : 1 + d])))
    if var_psi < 0:
        raise ValueError("the joint coefficient variance is materially negative")
    se = float(np.sqrt(var_psi))
    p_value_df = n - 1 - d
    # ndtri_exp avoids underflow in alpha/2 for very small positive alpha.
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2.0)))
    ci = (float(psi - critical * se), float(psi + critical * se))
    if not all(np.isfinite(v) for v in ci):
        raise ValueError("confidence limits are nonfinite")
    if degenerate or se == 0:
        degenerate = True
        test_statistic = float("nan")
        p_value = float("nan")
        warnings.warn(
            "Residual or contrast variation is at numerical zero; inferential tests are undefined.",
            QGCompWarning,
            stacklevel=2,
        )
    else:
        with np.errstate(over="ignore"):
            test_statistic = float(np.divide(psi, se))
        p_value = float(2.0 * special.stdtr(p_value_df, -abs(test_statistic)))
    beta_exposures = coefficients[1 : 1 + d]
    pos_psi, pos_weights = _weights(beta_exposures, transformation.feature_names, True)
    neg_psi, neg_weights = _weights(beta_exposures, transformation.feature_names, False)
    diagnostics = MappingProxyType(
        {
            "rank": rank,
            "rank_tolerance": tolerance,
            "condition_number": condition_number,
            "condition_warning_threshold": _CONDITION_WARNING,
            "ill_conditioned": ill_conditioned,
            "inference_degenerate": degenerate,
            "residual_roundoff_tolerance": residual_tolerance,
            "variance_roundoff_bound": variance_roundoff_bound,
            "variance_clamped": False,
            "contrast_variance_method": "svd_factor",
            "covariance_contrast_variance": covariance_contrast_variance,
            "case_weighted": case_weights is not None,
            "quantization": transformation.diagnostics,
        }
    )
    return QGCompResult(
        coefficients=_readonly(coefficients),
        coefficient_names=coefficient_names,
        covariance=_readonly(covariance),
        exposure_names=transformation.feature_names,
        covariate_names=covariate_names,
        fitted_values=_readonly(fitted),
        residuals=_readonly(residuals),
        case_weights=None if case_weights is None else _readonly(case_weights),
        residual_df=residual_df,
        residual_variance=residual_variance,
        psi=psi,
        var_psi=var_psi,
        se=se,
        ci=ci,
        alpha=alpha,
        test_statistic=test_statistic,
        p_value=p_value,
        p_value_df=p_value_df,
        pos_psi=pos_psi,
        neg_psi=neg_psi,
        pos_weights=pos_weights,
        neg_weights=neg_weights,
        zero_exposures=tuple(name for name, b in zip(transformation.feature_names, beta_exposures) if b == 0),
        diagnostics=diagnostics,
        library_versions=MappingProxyType({"numpy": np.__version__, "scipy": scipy.__version__}),
        _quantization=transformation,
        _covariate_columns=covariate_columns,
    )
