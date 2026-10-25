# SPDX-License-Identifier: GPL-2.0-or-later
"""A deliberately small, main-effects-only named-data interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
import re
from typing import Any

import numpy as np

from ._core import QGCompResult, _real_array, fit, quantize
from ._bootstrap import GaussianBootstrapResult, fit_bootstrap


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_CATEGORY = re.compile(r"C\(([A-Za-z_][A-Za-z_0-9]*)\)\Z")


@dataclass(frozen=True)
class _FormulaSchema:
    formula: str
    outcome: str
    exposure_names: tuple[str, ...]
    covariate_terms: tuple[tuple[str, bool], ...]
    category_levels: tuple[tuple[str, tuple[str, ...]], ...]
    covariate_names: tuple[str, ...]
    training_row_indices: tuple[int, ...] = ()
    original_row_count: int = 0


class _NamedMatrix:
    def __init__(self, values: np.ndarray, columns: tuple[str, ...]):
        self.values = values
        self.columns = columns

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return np.array(self.values, dtype=dtype, copy=copy) if copy is not None else np.asarray(
            self.values, dtype=dtype
        )


def _parse(formula: str, exposures: Sequence[str]) -> _FormulaSchema:
    if not isinstance(formula, str) or formula.count("~") != 1:
        raise ValueError("formula must have the form 'outcome ~ term + term'")
    lhs, rhs = (part.strip() for part in formula.split("~"))
    if not _IDENTIFIER.fullmatch(lhs):
        raise ValueError("formula outcome must be a simple column name")
    raw_terms = [part.strip() for part in rhs.split("+")]
    if not raw_terms or any(not term for term in raw_terms):
        raise ValueError("formula requires nonempty additive terms")
    parsed = []
    for term in raw_terms:
        category = _CATEGORY.fullmatch(term)
        if category:
            parsed.append((category.group(1), True))
        elif _IDENTIFIER.fullmatch(term):
            parsed.append((term, False))
        else:
            raise ValueError("only simple main effects and C(column) are supported")
    names = [name for name, _ in parsed]
    if len(set(names)) != len(names) or lhs in names:
        raise ValueError("formula column names must be distinct")
    if isinstance(exposures, (str, bytes)):
        raise ValueError("exposures must be a sequence of formula column names")
    try:
        exposure_names = tuple(exposures)
    except TypeError as exc:
        raise ValueError("exposures must be a sequence of formula column names") from exc
    if any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in exposure_names):
        raise ValueError("exposure names must be simple column names")
    if not exposure_names or len(set(exposure_names)) != len(exposure_names):
        raise ValueError("exposures must contain distinct column names")
    if any(name not in names for name in exposure_names):
        raise ValueError("every exposure must be a formula term")
    if any(is_category for name, is_category in parsed if name in exposure_names):
        raise ValueError("categorical exposures are not supported")
    covariate_terms = tuple(term for term in parsed if term[0] not in exposure_names)
    return _FormulaSchema(formula, lhs, exposure_names, covariate_terms, (), ())


def _column(data: Mapping[str, Any], name: str, n: int | None = None) -> np.ndarray:
    if not isinstance(data, Mapping) and not (
        hasattr(data, "columns") and hasattr(data, "__getitem__")
    ):
        raise ValueError("data must be a mapping or table with named columns")
    columns = getattr(data, "columns", None)
    if columns is not None and len(set(columns)) != len(columns):
        raise ValueError("data contains duplicate column identifiers")
    try:
        value = data[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"data is missing column {name!r}") from exc
    raw = np.asanyarray(value)
    if raw.ndim != 1 or (n is not None and len(raw) != n) or len(raw) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional column with matching rows")
    return raw


def _numeric(data: Mapping[str, Any], name: str, n: int | None = None) -> np.ndarray:
    raw = _column(data, name, n)
    return _real_array(raw, name)


def _categories(
    data: Mapping[str, Any], name: str, n: int,
    levels: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = _column(data, name, n)
    values = raw.tolist()
    if not all(isinstance(value, str) and value != "" for value in values):
        raise ValueError(f"{name} must contain complete, nonempty text categories")
    if levels is None:
        levels = tuple(sorted(set(values)))
        if len(levels) < 2:
            raise ValueError(f"{name} needs at least two observed levels")
    unknown = set(values).difference(levels)
    if unknown:
        raise ValueError(f"{name} contains an unseen category: {sorted(unknown)[0]!r}")
    return np.asarray(values, dtype=str), levels


def formula_matrices(
    data: Mapping[str, Any], schema: _FormulaSchema,
    *, training: bool = False,
) -> tuple[_NamedMatrix, _NamedMatrix | None] | tuple[
    _NamedMatrix, _NamedMatrix | None, _FormulaSchema
]:
    x_named = formula_exposures(data, schema)
    n = len(x_named.values)
    covariate_values = []
    covariate_names = []
    category_levels = dict(schema.category_levels)
    for name, is_category in schema.covariate_terms:
        if is_category:
            values, levels = _categories(
                data, name, n, None if training else category_levels[name]
            )
            category_levels[name] = levels
            for level in levels[1:]:
                covariate_values.append((values == level).astype(float))
                covariate_names.append(f"{name}[T.{level}]")
        else:
            covariate_values.append(_numeric(data, name, n))
            covariate_names.append(name)
    if training and len(set(covariate_names)) != len(covariate_names):
        raise ValueError("encoded covariate column names must be distinct")
    if not training and tuple(covariate_names) != schema.covariate_names:
        raise ValueError("prediction covariate design does not match training")
    z = (_NamedMatrix(np.column_stack(covariate_values), tuple(covariate_names))
         if covariate_values else None)
    if training:
        updated = replace(
            schema,
            category_levels=tuple((name, category_levels[name])
                                  for name, is_category in schema.covariate_terms if is_category),
            covariate_names=tuple(covariate_names),
        )
        return x_named, z, updated
    return x_named, z


def formula_exposures(data: Mapping[str, Any], schema: _FormulaSchema) -> _NamedMatrix:
    """Read only the fitted exposure columns for saved-cut transforms."""
    first = _numeric(data, schema.exposure_names[0])
    n = len(first)
    x = np.column_stack([first] + [
        _numeric(data, name, n) for name in schema.exposure_names[1:]
    ])
    return _NamedMatrix(x, schema.exposure_names)


def _observed_value(value: Any, name: str, *, category: bool) -> tuple[bool, Any]:
    if value is None or type(value).__name__ == "NAType" or np.ma.is_masked(value):
        return False, None
    if isinstance(value, Real) and np.isnan(value):
        return False, None
    if category:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain nonempty text categories or missing values")
        return True, value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain real numeric values or missing values")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} contains an infinite or out-of-range value")
    return True, converted


def _drop_missing_rows(
    data: Mapping[str, Any], schema: _FormulaSchema, weights: Any,
    q: int | None, breaks: Any, *, extra_numeric: tuple[str, ...] = (),
) -> tuple[dict[str, np.ndarray], Any, Any, tuple[int, ...], int]:
    terms = ((schema.outcome, False),
             *((name, False) for name in schema.exposure_names),
             *schema.covariate_terms,
             *((name, False) for name in extra_numeric
               if name not in (schema.outcome, *schema.exposure_names)
               and name not in (term for term, _ in schema.covariate_terms)))
    raw = _column(data, schema.outcome)
    n = len(raw)
    columns: dict[str, np.ndarray] = {}
    complete = np.ones(n, dtype=bool)
    observed_exposures: dict[str, np.ndarray] = {}
    for name, category in terms:
        values = _column(data, name, n)
        cleaned = np.empty(n, dtype=object)
        present = np.empty(n, dtype=bool)
        for i, value in enumerate(values):
            present[i], cleaned[i] = _observed_value(value, name, category=category)
        complete &= present
        if name in schema.exposure_names:
            observed_exposures[name] = np.asarray(cleaned[present], dtype=float)
        columns[name] = cleaned
    clean_weights = None
    if weights is not None:
        values = np.asanyarray(weights)
        if values.ndim != 1 or len(values) != n:
            raise ValueError("weights must have one value per original data row")
        clean_weights = np.empty(n, dtype=object)
        for i, value in enumerate(values):
            present, clean_weights[i] = _observed_value(value, "weights", category=False)
            complete[i] &= present
    rows = tuple(int(i) for i in np.flatnonzero(complete))
    if not rows:
        raise ValueError("no complete model rows remain")
    selected = {
        name: np.asarray(columns[name][complete], dtype=str if category else float)
        for name, category in terms
    }
    selected_weights = None if clean_weights is None else np.asarray(clean_weights[complete], dtype=float)
    fitted_breaks = breaks
    if breaks is None and q is not None and len(rows) < n:
        # R qgcomp quantizes each observed exposure before the GLM omits rows.
        fitted_breaks = tuple(quantize(observed_exposures[name], q=q).breaks[0]
                              for name in schema.exposure_names)
    return selected, selected_weights, fitted_breaks, rows, n


def fit_formula(
    formula: str,
    data: Mapping[str, Any],
    exposures: Sequence[str],
    *,
    q: int | None = 4,
    breaks: Any = None,
    alpha: float = 0.05,
    weights: Any = None,
    missing: str = "raise",
) -> QGCompResult:
    """Fit the additive Gaussian model from named columns.

    Supports only an intercept and additive main effects. Use simple names for
    the outcome and exposures, and C(name) for a categorical adjustment term.
    Categories use sorted observed text levels; the first is the reference.
    The same levels are retained for :meth:`QGCompResult.predict_data`.
    ``missing='raise'`` requires complete inputs. ``missing='drop'`` removes
    rows with missing model values or weights. Automatic cutpoints use all
    observed exposure values before row removal, as in R qgcomp 2.19.6.
    Interactions and transformations are not supported.
    """
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
    result = fit(outcome, x, z, q=q, breaks=breaks, alpha=alpha, weights=weights)
    if automatic_cuts and breaks is not None:
        # The core sees the pre-omission cuts as explicit. Preserve the user's
        # automatic-quantile mode in the returned transformation.
        transformed = replace(result._quantization, mode="quantile", q=int(q))
        result = replace(result, _quantization=transformed)
    if missing == "raise":
        rows = tuple(range(len(x.values)))
        original_n = len(rows)
    schema = replace(schema, training_row_indices=rows, original_row_count=original_n)
    return replace(result, _formula_schema=schema)


def fit_bootstrap_formula(
    formula: str,
    data: Mapping[str, Any],
    exposures: Sequence[str],
    *,
    q: int | None = 4,
    breaks: Any = None,
    alpha: float = 0.05,
    B: int = 200,
    seed: int | None = None,
    missing: str = "raise",
    weights: Any = None,
) -> GaussianBootstrapResult:
    """Fit the additive Gaussian fixed-cut row bootstrap from named columns.

    The formula allows additive main effects and ``C(name)`` text adjustment.
    ``missing='drop'`` removes incomplete model rows before resampling, while
    automatic exposure cutpoints use observed exposure values before removal.
    The retained row positions and formula are available on ``point_fit``.
    Rows must be independent. Positive case weights remain with sampled rows.
    """
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
    result = fit_bootstrap(
        outcome, x, z, q=q, breaks=breaks, alpha=alpha, B=B, seed=seed,
        weights=weights,
    )
    point = result.point_fit
    if automatic_cuts and breaks is not None:
        transformed = replace(point._quantization, mode="quantile", q=int(q))
        point = replace(point, _quantization=transformed)
    if missing == "raise":
        rows = tuple(range(len(x.values)))
        original_n = len(rows)
    schema = replace(schema, training_row_indices=rows, original_row_count=original_n)
    return replace(result, point_fit=replace(point, _formula_schema=schema))
