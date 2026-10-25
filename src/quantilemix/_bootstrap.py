# SPDX-License-Identifier: GPL-2.0-or-later
"""Fixed-cut row bootstrap for additive Gaussian quantile g-computation.

The intervention contrast is a marginal mean difference. For an additive
Gaussian model it equals the sum of exposure coefficients, although its
bootstrap uncertainty is calculated from resampled conditional fits.
Method: Keil et al. (2020), doi:10.1289/EHP5838.
Reference: R qgcomp 2.19.6, GPL (>= 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable
import warnings

import numpy as np
from scipy import special

from ._core import QGCompResult, QGCompWarning, _matrix, _readonly, _real_array, fit


@dataclass(frozen=True, eq=False)
class GaussianBootstrapResult:
    """Saved marginal joint contrast and row-bootstrap uncertainty."""

    point_fit: QGCompResult = field(repr=False)
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    p_value_df: int
    alpha: float
    intervention_levels: np.ndarray = field(repr=False)
    intervention_means: np.ndarray = field(repr=False)
    bootstrap_effects: np.ndarray = field(repr=False)
    bootstrap_means: np.ndarray = field(repr=False)
    B: int
    seed: int | None
    reference_version: str = "2.19.6"
    family: str = "gaussian"
    method: str = "fixed_cut_row_bootstrap"

    def summary(self, digits: int = 6) -> str:
        """Display the saved marginal contrast and bootstrap inference."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: Gaussian fixed-cut row bootstrap\n"
                f"Joint mean difference: {fmt(self.psi)}; bootstrap SE: {fmt(self.se)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI: ({fmt(self.ci[0])}, {fmt(self.ci[1])})\n"
                f"Two-sided Student t p-value: {fmt(self.p_value)}; df: {self.p_value_df}\n"
                f"Resamples: {self.B}; cutpoints held fixed.\n"
                + ("Positive case weights follow sampled rows.\n"
                   if self.point_fit.case_weights is not None else "")
                + "Independent rows and a linear joint dose-response are assumed.\n"
                "The interval is a normal Wald interval using bootstrap SD, not a percentile interval."
                + (" This is not complex-survey inference."
                   if self.point_fit.case_weights is not None else ""))


def _levels(point: QGCompResult) -> np.ndarray:
    if point.quantization_mode == "prequantized":
        levels = np.unique(point.scores)
        if not 2 <= len(levels) < 10:
            raise ValueError("prequantized bootstrap requires 2 through 9 pooled score levels")
    elif point.quantization_mode == "explicit":
        if len(set(point.category_counts)) != 1:
            raise ValueError("joint interventions require equal exposure category counts")
        levels = np.arange(point.category_counts[0], dtype=float)
    else:
        levels = np.arange(point.q, dtype=float)
    if len(levels) < 2:
        raise ValueError("the bootstrap requires at least two intervention levels")
    return _readonly(levels)


def _from_indices(
    point: QGCompResult, y: Any, covariates: Any, indices: Any,
    *, seed: int | None,
) -> GaussianBootstrapResult:
    index_values = _real_array(indices, "resample indices")
    n = len(point.scores)
    if (index_values.ndim != 2 or index_values.shape[1] != n
            or index_values.shape[0] < 2 or np.any(index_values < 0)
            or np.any(index_values >= n)
            or np.any(index_values != np.floor(index_values))):
        raise ValueError("resample indices must be a B-by-n matrix of zero-based row positions, B >= 2")
    schedule = index_values.astype(np.intp)
    return _from_rows(point, y, covariates, schedule, len(schedule), seed=seed)


def _from_rows(
    point: QGCompResult, y: Any, covariates: Any,
    rows: Iterable[np.ndarray], B: int, *, seed: int | None,
) -> GaussianBootstrapResult:
    n = len(point.scores)
    outcome = _real_array(y, "y")
    if covariates is None:
        z = None
    else:
        z, _ = _matrix(covariates, "covariates")
    effects = np.empty(B, dtype=float)
    levels = _levels(point)
    profiles = np.empty((B, len(levels)), dtype=float)
    case_weights = point.case_weights
    for b, sampled in enumerate(rows):
        try:
            replica = fit(outcome[sampled], point.scores[sampled],
                          None if z is None else z[sampled], q=None, alpha=point.alpha,
                          weights=None if case_weights is None else case_weights[sampled])
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b + 1} failed: {exc}") from exc
        effects[b] = replica.psi
        sampled_z = None if z is None else z[sampled]
        adjustment = (0.0 if sampled_z is None
                      else float(np.average(sampled_z @ replica.beta_covariates,
                                            weights=None if case_weights is None
                                            else case_weights[sampled])))
        profiles[b] = replica.intercept + levels * replica.psi + adjustment
    se = float(np.std(effects, ddof=1))
    if not np.isfinite(se):
        raise ValueError("bootstrap standard error is nonfinite")
    # In an additive Gaussian model, standardizing each observed covariate row
    # and averaging leaves the exposure contrast equal to sum(beta_exposures).
    mean_adjustment = (0.0 if z is None else float(np.average(
        z @ point.beta_covariates, weights=case_weights)))
    intervention_means = point.intercept + levels * point.psi + mean_adjustment
    psi = point.psi
    critical = float(-special.ndtri_exp(np.log(point.alpha) - np.log(2.0)))
    ci = (float(psi - critical * se), float(psi + critical * se))
    df = n - len(point.exposure_names) - len(point.covariate_names) - 2
    if df <= 0:
        raise ValueError("bootstrap Student t test requires positive reference degrees of freedom")
    if se == 0:
        warnings.warn("All bootstrap effects are identical; inferential test is undefined.",
                      QGCompWarning, stacklevel=2)
        statistic, p_value = float("nan"), float("nan")
    else:
        statistic = psi / se
        p_value = float(2 * special.stdtr(df, -abs(statistic)))
    if not np.all(np.isfinite(ci)) or not np.all(np.isfinite(intervention_means)):
        raise ValueError("bootstrap interval or intervention means are nonfinite")
    return GaussianBootstrapResult(
        point_fit=point, psi=psi, var_psi=se*se, se=se, ci=ci,
        test_statistic=statistic, p_value=p_value, p_value_df=df,
        alpha=point.alpha, intervention_levels=levels,
        intervention_means=_readonly(intervention_means),
        bootstrap_effects=_readonly(effects), bootstrap_means=_readonly(profiles),
        B=len(effects), seed=seed,
    )


def fit_bootstrap(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    B: int = 200, seed: int | None = None, weights: Any = None,
) -> GaussianBootstrapResult:
    """Estimate a Gaussian joint mean contrast with fixed-cut row bootstrap.

    Exposure cutpoints are computed once on all training rows, then held fixed.
    Each of ``B`` replicates samples ``n`` independent rows with replacement.
    Positive case weights stay attached to their rows in each replicate. They
    also weight the covariate distribution used for intervention means. This
    is not complex-survey or cluster bootstrap inference.
    The local NumPy generator leaves global random state unchanged. A failed
    replicate raises an error with its one-based position; no draw is replaced.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    point = fit(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha,
                weights=weights)
    _levels(point)
    n = len(point.scores)
    if n <= 2 + len(point.exposure_names) + len(point.covariate_names):
        raise ValueError("bootstrap Student t test needs more rows than model terms plus two")
    generator = np.random.default_rng(seed)
    rows = (generator.integers(0, n, size=n, dtype=np.intp) for _ in range(int(B)))
    return _from_rows(point, y, covariates, rows, int(B),
                      seed=None if seed is None else int(seed))


def _fit_bootstrap_with_indices(
    y: Any, exposures: Any, indices: Any, covariates: Any = None, *,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None,
) -> GaussianBootstrapResult:
    """Use an external fixed row schedule only for reference verification."""
    point = fit(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha,
                weights=weights)
    return _from_indices(point, y, covariates, indices, seed=None)
