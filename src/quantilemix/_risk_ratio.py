# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal binary risk ratio per one joint exposure-score increase."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy import special

from ._binomial import BinomialResult, fit_binomial
from ._core import _matrix, _readonly, _real_array


@dataclass(frozen=True, eq=False)
class BinomialRiskRatioResult:
    """Standardized risks and fixed-cut row-bootstrap log risk ratio."""

    point_fit: BinomialResult = field(repr=False)
    psi: float
    risk_ratio: float
    se: float
    ci: tuple[float, float]
    risk_ratio_ci: tuple[float, float]
    test_statistic: float
    p_value: float
    intervention_levels: np.ndarray = field(repr=False)
    intervention_risks: np.ndarray = field(repr=False)
    msm_coefficients: np.ndarray = field(repr=False)
    msm_fitted_risks: np.ndarray = field(repr=False)
    bootstrap_effects: np.ndarray = field(repr=False)
    B: int
    seed: int | None
    alpha: float
    reference_version: str = "2.19.6"
    family: str = "binomial"
    method: str = "marginal_risk_ratio_fixed_cut_row_bootstrap"

    def summary(self, digits: int = 6) -> str:
        """Show the population-average ratio and bootstrap uncertainty."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        risks = "; ".join(f"score {int(level)}: {fmt(risk)}" for level, risk in
                          zip(self.intervention_levels, self.intervention_risks))
        return ("Quantile g-computation: marginal binary risk ratio\n"
                f"Standardized risks at {risks}\n"
                "Effect: one score category in every exposure.\n"
                f"Log risk ratio: {fmt(self.psi)}; bootstrap SE: {fmt(self.se)}\n"
                f"Risk ratio: {fmt(self.risk_ratio)}; normal {(1-self.alpha)*100:g}% "
                f"CI: ({fmt(self.risk_ratio_ci[0])}, {fmt(self.risk_ratio_ci[1])})\n"
                f"Two-sided normal z p-value: {fmt(self.p_value)}; resamples: {self.B}.\n"
                "Exposure cutpoints are fixed; rows are resampled independently.")


def _standardized(point: BinomialResult, z: np.ndarray | None,
                  levels: np.ndarray) -> np.ndarray:
    n = len(point.scores)
    adjustment = np.zeros(n) if z is None else z @ point.coefficients[1+len(point.exposure_names):]
    slope = float(np.sum(point.coefficients[1:1+len(point.exposure_names)]))
    risks = np.array([np.mean(special.expit(point.coefficients[0] + adjustment + a*slope))
                      for a in levels])
    if not np.all((risks > 0) & (risks < 1)):
        raise ValueError("standardized risks must be strictly between zero and one")
    return risks


def _log_binomial_msm(risks: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fit R's degree-one log-binomial MSM to standardized fractional risks.

    Each intervention level has the same number of rows and equal prior
    weight. The score equations therefore depend only on its mean risk.
    """
    if len(levels) == 2:
        return np.array([np.log(risks[0]), np.log(risks[1] / risks[0])])
    design = np.column_stack((np.ones(len(levels)), levels))
    beta = np.array([np.log(float(np.mean(risks))), 0.0])

    def log_likelihood(value: np.ndarray) -> float:
        eta = design @ value
        if np.any(eta >= 0) or not np.all(np.isfinite(eta)):
            return -np.inf
        return float(np.sum(risks * eta + (1-risks) * np.log(-np.expm1(eta))))

    previous = log_likelihood(beta)
    for _ in range(100):
        mu = np.exp(design @ beta)
        score = design.T @ ((risks-mu)/(1-mu))
        fisher = design.T @ ((mu/(1-mu))[:, None] * design)
        try:
            step = np.linalg.solve(fisher, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("marginal log-binomial information is singular") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("marginal log-binomial update is nonfinite")
        scale = 1.0
        for _ in range(50):
            candidate = beta + scale*step
            likelihood = log_likelihood(candidate)
            if np.isfinite(likelihood) and likelihood >= previous-1e-13:
                break
            scale /= 2
        else:
            raise ValueError("marginal log-binomial line search failed")
        beta = candidate
        previous = likelihood
        if (np.max(np.abs(scale*step)) <= 1e-10*(1+np.max(np.abs(beta)))
                and np.max(np.abs(score)) <= 1e-9):
            return beta
    raise ValueError("marginal log-binomial fit did not converge")


def _from_indices(y: Any, exposures: Any, indices: Any, covariates: Any = None,
                  *, q: int | None = 2, breaks: Any = None,
                  alpha: float = 0.05) -> BinomialRiskRatioResult:
    """Reference-only entry point with a supplied zero-based resampling schedule."""
    schedule = _real_array(indices, "resample indices")
    n = len(_real_array(y, "y"))
    if (schedule.ndim != 2 or schedule.shape[1] != n or schedule.shape[0] < 2
            or np.any(schedule < 0) or np.any(schedule >= n)
            or np.any(schedule != np.floor(schedule))):
        raise ValueError("resample indices must be a B-by-n matrix of zero-based row positions, B >= 2")
    return _fit(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha,
                rows=schedule.astype(np.intp), seed=None)


def _fit(y: Any, exposures: Any, covariates: Any, *, q: int | None, breaks: Any,
         alpha: float, rows: Any, seed: int | None) -> BinomialRiskRatioResult:
    point = fit_binomial(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha)
    if point.q is not None:
        levels = np.arange(point.q, dtype=float)
    elif point.breaks is not None:
        counts = point._quantization.category_counts
        if len(set(counts)) != 1:
            raise ValueError("marginal risk ratio requires the same number of supplied levels for every exposure")
        levels = np.arange(counts[0], dtype=float)
    else:
        unique = np.unique(point.scores)
        levels = np.arange(len(unique), dtype=float)
        if not np.array_equal(unique, levels):
            raise ValueError("unquantized marginal risk ratio requires contiguous score levels starting at 0")
    if len(levels) < 2:
        raise ValueError("marginal risk ratio requires at least two intervention levels")
    outcome = _real_array(y, "y")
    z = None if covariates is None else _matrix(covariates, "covariates")[0]
    risks = _standardized(point, z, levels)
    msm = _log_binomial_msm(risks, levels)
    psi = float(msm[1])
    effects = []
    for b, sampled in enumerate(rows, 1):
        try:
            replica = fit_binomial(outcome[sampled], point.scores[sampled],
                                   None if z is None else z[sampled], q=None, alpha=alpha)
            replica_risks = _standardized(replica, None if z is None else z[sampled], levels)
            effects.append(float(_log_binomial_msm(replica_risks, levels)[1]))
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {b} failed: {exc}") from exc
    effects_array = np.asarray(effects)
    se = float(np.std(effects_array, ddof=1))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("bootstrap log risk ratio standard error must be positive and finite")
    critical = float(special.ndtri(1-alpha/2))
    ci = (float(psi-critical*se), float(psi+critical*se))
    statistic = psi/se
    return BinomialRiskRatioResult(
        point_fit=point, psi=psi, risk_ratio=float(np.exp(psi)), se=se, ci=ci,
        risk_ratio_ci=(float(np.exp(ci[0])), float(np.exp(ci[1]))),
        test_statistic=statistic, p_value=float(2*special.ndtr(-abs(statistic))),
        intervention_levels=_readonly(levels), intervention_risks=_readonly(risks),
        msm_coefficients=_readonly(msm),
        msm_fitted_risks=_readonly(np.exp(msm[0] + msm[1]*levels)),
        bootstrap_effects=_readonly(effects_array),
        B=len(effects_array), seed=seed, alpha=alpha,
    )


def fit_binomial_risk_ratio(y: Any, exposures: Any, covariates: Any = None, *,
                            q: int | None = 2, breaks: Any = None, alpha: float = 0.05,
                            B: int = 200, seed: int | None = None) -> BinomialRiskRatioResult:
    """Estimate a marginal risk ratio per joint score-category increase.

    For two levels the effect is the ratio of standardized risks at 1 and 0.
    For three or more levels it is the slope of a degree-one log-binomial
    marginal structural model fitted to standardized risks at every level.
    Inputs must have complete numeric rows, no case weights and independent
    rows. A failed bootstrap draw raises an error instead of being replaced.
    R is needed only for optional reference checks, never at Python run time.
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
                rows=rows, seed=None if seed is None else int(seed))
