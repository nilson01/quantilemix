# SPDX-License-Identifier: GPL-2.0-or-later
"""Fixed-cut marginal Poisson-log bootstrap for independent complete rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy import special

from ._core import _matrix, _readonly, _real_array
from ._poisson import PoissonResult, fit_poisson


@dataclass(frozen=True, eq=False)
class PoissonBootstrapResult:
    """Marginal joint log mean ratio and bootstrap uncertainty."""

    point_fit: PoissonResult = field(repr=False)
    psi: float
    rate_ratio: float
    se: float
    ci: tuple[float, float]
    rate_ratio_ci: tuple[float, float]
    test_statistic: float
    p_value: float
    intervention_levels: np.ndarray = field(repr=False)
    intervention_means: np.ndarray = field(repr=False)
    bootstrap_effects: np.ndarray = field(repr=False)
    bootstrap_means: np.ndarray = field(repr=False)
    B: int
    seed: int | None
    alpha: float
    reference_version: str = "2.19.6"
    family: str = "poisson"
    method: str = "marginal_fixed_cut_row_bootstrap"

    def summary(self, digits: int = 6) -> str:
        """Display the saved marginal ratio and normal bootstrap interval."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: marginal Poisson row bootstrap\n"
                f"Joint log mean ratio: {fmt(self.psi)}; bootstrap SE: {fmt(self.se)}\n"
                f"Mean ratio: {fmt(self.rate_ratio)}; normal {(1-self.alpha)*100:g}% "
                f"CI: ({fmt(self.rate_ratio_ci[0])}, {fmt(self.rate_ratio_ci[1])})\n"
                f"Two-sided normal z p-value: {fmt(self.p_value)}; resamples: {self.B}.\n"
                "Exposure cutpoints are fixed; rows are resampled independently."
                + (" Observation times are held at each row's value under every intervention."
                   if self.point_fit.exposure_time is not None else
                   " Counts cover equal observation periods; no offset is fitted."))


def _levels(point: PoissonResult) -> np.ndarray:
    if point.quantization_mode == "prequantized":
        observed = np.unique(point.scores)
        if not 2 <= len(observed) < 10 or not np.array_equal(observed, np.arange(len(observed))):
            raise ValueError("prequantized bootstrap needs 2 through 9 consecutive levels starting at zero")
        return _readonly(observed)
    counts = point._quantization.category_counts
    if len(set(counts)) != 1 or counts[0] < 2:
        raise ValueError("joint interventions require equal exposure category counts above one")
    return _readonly(np.arange(counts[0], dtype=float))


def _means(point: PoissonResult, z: np.ndarray | None, levels: np.ndarray) -> np.ndarray:
    n = len(point.scores)
    adjustment = np.zeros(n) if z is None else z @ point.coefficients[1+len(point.exposure_names):]
    slope = float(np.sum(point.coefficients[1:1+len(point.exposure_names)]))
    log_time = (np.zeros(n) if point.exposure_time is None
                else np.log(point.exposure_time))
    means = np.array([np.mean(np.exp(point.coefficients[0] + adjustment + a*slope + log_time))
                      for a in levels])
    if not np.all(np.isfinite(means)) or np.any(means <= 0):
        raise ValueError("standardized means must be positive and finite")
    return means


def _from_indices(y: Any, exposures: Any, indices: Any, covariates: Any = None,
                  *, q: int | None = 4, breaks: Any = None,
                  alpha: float = 0.05, exposure_time: Any = None) -> PoissonBootstrapResult:
    """Reference-only entry point using a supplied zero-based row schedule."""
    schedule = _real_array(indices, "resample indices")
    n = len(_real_array(y, "y"))
    if (schedule.ndim != 2 or schedule.shape[1] != n or schedule.shape[0] < 2
            or np.any(schedule < 0) or np.any(schedule >= n)
            or np.any(schedule != np.floor(schedule))):
        raise ValueError("resample indices must be a B-by-n matrix of zero-based row positions, B >= 2")
    return _fit(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha,
                exposure_time=exposure_time,
                rows=schedule.astype(np.intp), seed=None)


def _fit(y: Any, exposures: Any, covariates: Any, *, q: int | None, breaks: Any,
         alpha: float, exposure_time: Any, rows: Any,
         seed: int | None) -> PoissonBootstrapResult:
    point = fit_poisson(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha,
                        exposure_time=exposure_time)
    levels = _levels(point)
    outcome = _real_array(y, "y")
    z = None if covariates is None else _matrix(covariates, "covariates")[0]
    means = _means(point, z, levels)
    psi = float(np.log(means[1]/means[0]))
    effects = []
    profiles = []
    for b, sampled in enumerate(rows, 1):
        try:
            replica = fit_poisson(outcome[sampled], point.scores[sampled],
                                  None if z is None else z[sampled], q=None, alpha=alpha,
                                  exposure_time=None if point.exposure_time is None
                                  else point.exposure_time[sampled])
            effects.append(float(replica.psi))
            profiles.append(_means(replica, None if z is None else z[sampled], levels))
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b} failed: {exc}") from exc
    effects_array = np.asarray(effects)
    se = float(np.std(effects_array, ddof=1))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("bootstrap log mean ratio standard error must be positive and finite")
    critical = float(special.ndtri(1-alpha/2))
    ci = (float(psi-critical*se), float(psi+critical*se))
    statistic = psi/se
    return PoissonBootstrapResult(
        point_fit=point, psi=psi, rate_ratio=float(np.exp(psi)), se=se, ci=ci,
        rate_ratio_ci=(float(np.exp(ci[0])), float(np.exp(ci[1]))),
        test_statistic=statistic, p_value=float(2*special.ndtr(-abs(statistic))),
        intervention_levels=levels, intervention_means=_readonly(means),
        bootstrap_effects=_readonly(effects_array),
        bootstrap_means=_readonly(np.asarray(profiles)), B=len(effects_array),
        seed=seed, alpha=alpha,
    )


def fit_poisson_bootstrap(y: Any, exposures: Any, covariates: Any = None, *,
                          q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
                          B: int = 200, seed: int | None = None,
                          exposure_time: Any = None) -> PoissonBootstrapResult:
    """Estimate marginal Poisson mean ratios with fixed-cut row resampling.

    Accepts complete numeric rows, optional positive observation times,
    no case weights, and independent rows. Each row's observation time remains
    fixed under every intervention and follows that row during resampling.
    A failed draw is reported, not replaced.
    """
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    n = len(_real_array(y, "y"))
    generator = np.random.default_rng(seed)
    rows = (generator.integers(0, n, size=n, dtype=np.intp) for _ in range(B))
    return _fit(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha,
                exposure_time=exposure_time,
                rows=rows, seed=None if seed is None else int(seed))
