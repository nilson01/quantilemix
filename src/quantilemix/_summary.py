# SPDX-License-Identifier: GPL-2.0-or-later
"""Presentation of saved Gaussian results; no fitting or inference calculations."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING
import math

import numpy as np

if TYPE_CHECKING:
    from ._core import QGCompResult


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[j]) for row in [headers, *rows])
              for j in range(len(headers))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
            for row in [headers, *rows]]


def format_summary(result: QGCompResult, *, digits: int) -> str:
    if (isinstance(digits, (bool, np.bool_)) or not isinstance(digits, Integral)
            or not 1 <= digits <= 17):
        raise ValueError("digits must be an integer from 1 through 17")

    def number(value: float) -> str:
        return "NA" if math.isnan(value) else format(value, f".{int(digits)}g")

    lines = [
        "Quantile g-computation: additive Gaussian",
        f"Observations: {len(result.fitted_values)}",
        ("Case weights: none (equal influence in the fitted regression)"
         if result.case_weights is None else
         "Case weights: supplied (positive; regression uses weighted least squares)"),
        f"Exposures: {len(result.exposure_names)}; adjustment covariates: {len(result.covariate_names)}",
        f"Quantization: {result.quantization_mode}; q: {result.q if result.q is not None else 'not used'}",
        f"Residual degrees of freedom: {result.residual_df}",
        f"Residual variance: {number(result.residual_variance)}",
        "",
        "Joint effect: all exposure scores increase by one; covariates held fixed.",
        f"Normal confidence interval; alpha = {result.alpha!r}.",
        f"Two-sided Student t p-value; degrees of freedom = {result.p_value_df}.",
        f"Reference convention: R qgcomp {result.reference_version} ({result.inference}).",
    ]
    lines.extend(_table(
        ("Term", "Estimate", "SE", "CI lower", "CI upper", "t", "p-value"),
        [("psi", *(number(value) for value in (
            result.psi, result.se, result.ci[0], result.ci[1],
            result.test_statistic, result.p_value,
        )))],
    ))
    lines += [
        "The normal interval and t test need not give the same threshold decision.",
        "SE means standard error; CI means confidence interval.",
        "",
        "Sign weights (descriptive proportions within each sign group)",
        f"Positive coefficient sum: {number(result.pos_psi)}",
        f"Negative coefficient sum: {number(result.neg_psi)}",
    ]
    for label, weights in (("Positive", result.pos_weights), ("Negative", result.neg_weights)):
        lines.append(f"{label} weights:")
        lines.extend(_table(
            ("Exposure", "Weight"),
            [(repr(name), number(value)) for name, value in weights.items()],
        ) if weights else ["  None (no coefficients in this sign group)."])
    lines.append("Exactly zero exposure coefficients: " + (
        ", ".join(repr(name) for name in result.zero_exposures) or "none"
    ))
    lines += ["Weights do not measure selection probability or causal importance.", "",
              "Model coefficients (estimates only; fitted column order)"]
    lines.extend(_table(
        ("Term", "Estimate"),
        [(repr(name), number(value)) for name, value in
         zip(result.coefficient_names, result.coefficients)],
    ))
    diagnostics = result.diagnostics
    lines += ["", "Numerical and grouping diagnostics", f"Design rank: {diagnostics['rank']}",
              f"Condition number: {number(diagnostics['condition_number'])}",
              f"Condition warning threshold: {number(diagnostics['condition_warning_threshold'])}"]
    if diagnostics["ill_conditioned"]:
        lines.append("WARNING: The design is ill-conditioned; coefficients can be sensitive to small changes.")
    else:
        lines.append("The design condition number is at or below the numerical warning threshold.")
    if diagnostics["inference_degenerate"]:
        lines.append("WARNING: Residual or joint-effect variation is numerically zero. The point interval")
        lines.append("does not establish certainty; t and p-value are unavailable (NA).")
    quantization = diagnostics["quantization"]
    reduced = quantization["reduced_category_columns"]
    if reduced:
        lines.append("Tied cutpoints reduced the number of categories for: "
                     + ", ".join(repr(name) for name in reduced) + ".")
    category_label = ("Distinct training scores per exposure" if result.quantization_mode == "prequantized"
                      else "Defined bins per exposure")
    lines.append(category_label + ": " + ", ".join(
        f"{name!r}={count}" for name, count in zip(result.exposure_names, result.category_counts)
    ))
    empty_bins = quantization["empty_bins"]
    if empty_bins is not None and any(empty_bins):
        lines.append("Empty fitted bins (zero-based scores): " + "; ".join(
            f"{name!r}: {bins}" for name, bins in zip(result.exposure_names, empty_bins) if bins
        ))
    lines += [
        "Diagnostics describe this fit; they do not test model assumptions or establish causality.",
        "For new exposures, predict() warns about extrapolation; transform() returns its diagnostics.",
        f"Display precision: {int(digits)} significant digits; result attributes retain full precision.",
    ]
    return "\n".join(lines)
