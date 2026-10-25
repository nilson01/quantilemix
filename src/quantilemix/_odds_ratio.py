# SPDX-License-Identifier: GPL-2.0-or-later
"""Population-average binary odds ratio under additive conditional logistic fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy import special

from ._binomial import BinomialResult, fit_binomial
from ._core import _matrix, _readonly, _real_array
from ._risk_ratio import _standardized


class BootstrapFitError(ValueError):
    """A failed resample with the original fit and completed effects retained."""

    def __init__(self, message: str, *, draw: int, effects: list[float],
                 estimate: float, risks: np.ndarray) -> None:
        super().__init__(message)
        self.draw = draw
        self.effects = tuple(effects)
        self.estimate = estimate
        self.risks = tuple(float(value) for value in risks)


@dataclass(frozen=True, eq=False)
class BinomialOddsRatioResult:
    """Standardized risks and fixed-cut bootstrap marginal odds ratio."""

    point_fit: BinomialResult = field(repr=False)
    psi: float
    odds_ratio: float
    se: float
    ci: tuple[float, float]
    odds_ratio_ci: tuple[float, float]
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
    method: str = "marginal_odds_ratio_fixed_cut_row_bootstrap"

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def fmt(value: float) -> str:
            return format(value, f".{digits}g")
        risks = "; ".join(f"score {int(level)}: {fmt(risk)}" for level, risk in
                          zip(self.intervention_levels, self.intervention_risks))
        interpretation = (
            "With two levels, this compares the standardized odds at score 1 "
            "with score 0. " if len(self.intervention_levels) == 2 else
            "With more than two levels, the fitted odds ratio need not equal "
            "any single adjacent-level marginal odds ratio. "
        )
        return ("Quantile g-computation: marginal binary odds ratio\n"
                f"Standardized risks at {risks}\n"
                "Effect: fitted summary across joint exposure-score levels.\n"
                f"Log odds ratio: {fmt(self.psi)}; bootstrap SE: {fmt(self.se)}\n"
                f"Odds ratio: {fmt(self.odds_ratio)}; normal {(1-self.alpha)*100:g}% "
                f"CI: ({fmt(self.odds_ratio_ci[0])}, {fmt(self.odds_ratio_ci[1])})\n"
                f"Two-sided normal z p-value: {fmt(self.p_value)}; resamples: {self.B}.\n"
                "Exposure cutpoints are fixed; rows are resampled independently. "
                + interpretation +
                "The marginal odds ratio can differ from the conditional odds ratio.")


def _levels(point: BinomialResult) -> np.ndarray:
    if point.q is not None:
        levels = np.arange(point.q, dtype=float)
    elif point.breaks is not None:
        counts = point._quantization.category_counts
        if len(set(counts)) != 1:
            raise ValueError("marginal odds ratio requires the same number of supplied levels for every exposure")
        levels = np.arange(counts[0], dtype=float)
    else:
        unique = np.unique(point.scores)
        levels = np.arange(len(unique), dtype=float)
        if not np.array_equal(unique, levels):
            raise ValueError("unquantized marginal odds ratio requires contiguous score levels starting at 0")
    if len(levels) < 2:
        raise ValueError("marginal odds ratio requires at least two intervention levels")
    return levels


def _logistic_msm(risks: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fit an equal-weight logistic MSM to the standardized fractional risks."""
    if len(levels) == 2:
        return np.array([special.logit(risks[0]),
                         special.logit(risks[1])-special.logit(risks[0])])
    design = np.column_stack((np.ones(len(levels)), levels))
    beta = np.array([special.logit(float(np.mean(risks))), 0.0])
    for _ in range(100):
        fitted = special.expit(design @ beta)
        score = design.T @ (risks-fitted)
        information = design.T @ ((fitted*(1-fitted))[:, None]*design)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("marginal logistic information is singular") from exc
        beta += step
        if np.max(np.abs(step)) <= 1e-11*(1+np.max(np.abs(beta))):
            return beta
    raise ValueError("marginal logistic fit did not converge")


def _fit(y: Any, exposures: Any, covariates: Any, *, q: int | None,
         breaks: Any, alpha: float, rows: Any, seed: int | None) -> BinomialOddsRatioResult:
    point = fit_binomial(y, exposures, covariates, q=q, breaks=breaks, alpha=alpha)
    levels = _levels(point)
    outcome = _real_array(y, "y")
    z = None if covariates is None else _matrix(covariates, "covariates")[0]
    risks = _standardized(point, z, levels)
    msm = _logistic_msm(risks, levels)
    effects = []
    for b, sampled in enumerate(rows, 1):
        try:
            replica = fit_binomial(outcome[sampled], point.scores[sampled],
                                   None if z is None else z[sampled], q=None, alpha=alpha)
            replica_risks = _standardized(replica, None if z is None else z[sampled], levels)
            effects.append(float(_logistic_msm(replica_risks, levels)[1]))
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise BootstrapFitError(
                f"bootstrap replicate {b} failed: {exc}", draw=b,
                effects=effects, estimate=float(msm[1]), risks=risks,
            ) from exc
    effects_array = np.asarray(effects)
    se = float(np.std(effects_array, ddof=1))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("bootstrap log odds ratio standard error must be positive and finite")
    psi = float(msm[1])
    critical = float(special.ndtri(1-alpha/2))
    ci = (float(psi-critical*se), float(psi+critical*se))
    statistic = psi/se
    return BinomialOddsRatioResult(
        point_fit=point, psi=psi, odds_ratio=float(np.exp(psi)), se=se, ci=ci,
        odds_ratio_ci=(float(np.exp(ci[0])), float(np.exp(ci[1]))),
        test_statistic=statistic, p_value=float(2*special.ndtr(-abs(statistic))),
        intervention_levels=_readonly(levels), intervention_risks=_readonly(risks),
        msm_coefficients=_readonly(msm),
        msm_fitted_risks=_readonly(special.expit(msm[0]+msm[1]*levels)),
        bootstrap_effects=_readonly(effects_array), B=len(effects_array), seed=seed,
        alpha=alpha,
    )


def fit_binomial_odds_ratio(
    y: Any, exposures: Any, covariates: Any = None, *,
    q: int | None = 2, breaks: Any = None, alpha: float = 0.05,
    B: int = 200, seed: int | None = None,
) -> BinomialOddsRatioResult:
    """Estimate a fitted marginal odds-ratio summary across joint score levels.

    The conditional outcome model is additive logistic regression. Risks are
    standardized over observed covariates at each joint score, then a degree-one
    logistic marginal structural model is fitted. Independent rows are
    resampled with fixed exposure cuts. Case weights and missing rows are not
    supported. With more than two levels, the fitted odds ratio need not equal
    any one adjacent-level marginal odds ratio or the conditional odds ratio.
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
