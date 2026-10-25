# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal Cox model from simulated potential event times and row bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable

import numpy as np
from scipy import special

from ._core import _matrix, _readonly, _real_array
from ._cox import CoxResult, _baseline_hazard, _solve_cox, fit_cox


class CoxBootstrapFitError(ValueError):
    """A failed resample with the original effect and completed effects retained."""

    def __init__(self, message: str, *, draw: int, effects: np.ndarray,
                 estimate: np.ndarray, levels: np.ndarray) -> None:
        super().__init__(message)
        self.draw = draw
        self.effects = tuple(tuple(float(value) for value in row) for row in effects)
        self.estimate = tuple(float(value) for value in estimate)
        self.levels = tuple(float(value) for value in levels)


@dataclass(frozen=True, eq=False)
class CoxBootstrapResult:
    """Marginal Cox curve coefficients and fixed-cut row-bootstrap uncertainty."""

    conditional_fit: CoxResult = field(repr=False)
    psi: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    ci: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    intervention_levels: np.ndarray = field(repr=False)
    simulated_time: np.ndarray = field(repr=False)
    simulated_event: np.ndarray = field(repr=False)
    simulated_level: np.ndarray = field(repr=False)
    training_covariates: np.ndarray = field(repr=False)
    bootstrap_effects: np.ndarray = field(repr=False)
    marginal_timeline: np.ndarray = field(repr=False)
    marginal_baseline_cumulative_hazard: np.ndarray = field(repr=False)
    B: int
    MCsize: int
    degree: int
    alpha: float
    seed: int | None
    family: str = "cox"
    method: str = "marginal_cumulative_hazard_row_bootstrap"
    reference_version: str = "2.19.6"

    def effect_at(self, level: float = 0.0) -> float:
        """Marginal log hazard change from joint level to level plus one."""
        if not np.isfinite(level):
            raise ValueError("level must be finite")
        return float(self.psi[0] if self.degree == 1 else
                     self.psi[0] + (2*level+1)*self.psi[1])

    def summary(self, digits: int = 6) -> str:
        """Display marginal curve terms and bootstrap normal intervals."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def f(value: float) -> str:
            return format(value, f".{digits}g")
        rows = [f"psi{j+1}: {f(effect)}; SE: {f(se)}; CI: ({f(ci[0])}, {f(ci[1])}); p: {f(p)}"
                for j, (effect, se, ci, p) in enumerate(zip(self.psi, self.se, self.ci, self.p_value))]
        return ("Quantile g-computation: marginal Cox survival bootstrap\n"
                + "\n".join(rows) + "\n"
                + f"Monte Carlo profiles: {self.MCsize} per level; row resamples: {self.B}.\n"
                + "Exposure cuts are fixed. Synthetic event times follow the fitted cumulative hazard.\n"
                + "These are marginal model coefficients; nonlinear one-level changes depend on starting level.\n"
                + "Intervals use bootstrap SD and a normal reference, not percentile limits.")


def _levels(point: CoxResult) -> np.ndarray:
    quantization = point._quantization
    if quantization.mode == "prequantized":
        levels = np.quantile(point.scores.ravel(), [.25, .5, .75])
    else:
        counts = quantization.category_counts
        if len(set(counts)) != 1:
            raise ValueError("joint interventions require equal exposure category counts")
        levels = np.arange(counts[0], dtype=float)
    if len(np.unique(levels)) != len(levels) or len(levels) < 2:
        raise ValueError("joint intervention levels must be distinct")
    return _readonly(np.asarray(levels, dtype=float))


def _simulate_survival(point: CoxResult, levels: np.ndarray, profile_z: np.ndarray,
                       thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert each fitted cumulative hazard at fixed exponential thresholds."""
    n = len(profile_z)
    if (thresholds.shape != (len(levels), n) or not np.all(np.isfinite(thresholds))
            or np.any(thresholds <= 0)):
        raise ValueError("exponential thresholds must be positive, finite, and level-by-profile")
    beta = point.coefficients
    d = len(point.exposure_names)
    adjustment = profile_z @ beta[d:]
    times = np.empty(len(levels)*n)
    events = np.empty(len(times))
    scores = np.repeat(levels, n)
    baseline = point.baseline_cumulative_hazard
    for j, level in enumerate(levels):
        eta = float(np.sum(beta[:d]))*level + adjustment
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            targets = thresholds[j] / np.exp(eta)
        if np.any(np.isnan(targets)) or np.any(targets <= 0):
            raise ValueError("simulated hazard scale is invalid")
        index = np.searchsorted(baseline, targets, side="left")
        sl = slice(j*n, (j+1)*n)
        times[sl] = point.timeline[np.minimum(index, len(point.timeline)-1)]
        events[sl] = index < len(point.timeline)
    return _readonly(times), _readonly(events), _readonly(scores)


def _fit_marginal(time: np.ndarray, event: np.ndarray, level: np.ndarray,
                  degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([level**power for power in range(1, degree+1)])
    if np.count_nonzero(event) <= degree:
        raise ValueError("marginal Cox model needs more events than curve terms")
    order = np.argsort(-time, kind="stable")
    beta, _, _, _ = _solve_cox(design[order], event[order], time[order])
    timeline, baseline = _baseline_hazard(design[order], event[order], time[order], beta)
    return beta, timeline, baseline


def _one_fit(time: np.ndarray, event: np.ndarray, scores: np.ndarray,
             z: np.ndarray, levels: np.ndarray, degree: int,
             rng: np.random.Generator, MCsize: int,
             alpha: float) -> tuple[CoxResult, np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray]:
    conditional = fit_cox(time, event, scores, z if z.shape[1] else None,
                          q=None, alpha=alpha)
    profile_rows = rng.integers(0, len(time), size=MCsize, dtype=np.intp)
    thresholds = rng.exponential(size=(len(levels), MCsize))
    simulated_time, simulated_event, simulated_level = _simulate_survival(
        conditional, levels, z[profile_rows], thresholds)
    beta, timeline, baseline = _fit_marginal(
        simulated_time, simulated_event, simulated_level, degree)
    return conditional, beta, simulated_time, simulated_event, simulated_level, timeline, baseline


def _fit(time: Any, event: Any, exposures: Any, covariates: Any, *,
         q: int | None, breaks: Any, alpha: float, degree: int, B: int,
         MCsize: int | None, seed: int | None,
         rows: Iterable[np.ndarray] | None = None) -> CoxBootstrapResult:
    point = fit_cox(time, event, exposures, covariates, q=q, breaks=breaks, alpha=alpha)
    levels = _levels(point)
    if degree >= len(levels):
        raise ValueError("marginal polynomial degree needs more intervention levels")
    n = len(point.scores)
    if MCsize is None:
        MCsize = max(n, 10000)
    if (isinstance(MCsize, (bool, np.bool_)) or not isinstance(MCsize, Integral)
            or MCsize < n):
        raise ValueError("MCsize must be an integer at least the number of rows")
    rng = np.random.default_rng(seed)
    t = _real_array(time, "durations")
    e = _real_array(event, "events")
    z = (np.empty((n, 0)) if covariates is None
         else _matrix(covariates, "covariates")[0])
    _, psi, simulated_time, simulated_event, simulated_level, timeline, baseline = _one_fit(
        t, e, point.scores, z, levels, degree, rng, int(MCsize), alpha)
    effects = np.empty((B, degree))
    if rows is None:
        rows = (rng.integers(0, n, size=n, dtype=np.intp) for _ in range(B))
    for b, sampled in enumerate(rows):
        if b >= B:
            raise ValueError("row schedule has more than B resamples")
        sampled = np.asarray(sampled)
        if (sampled.shape != (n,) or np.any(sampled < 0) or np.any(sampled >= n)
                or np.any(sampled != np.floor(sampled))):
            raise ValueError(f"bootstrap replicate {b+1} has invalid row positions")
        sampled = sampled.astype(np.intp)
        try:
            _, effect, *_ = _one_fit(t[sampled], e[sampled], point.scores[sampled],
                                      z[sampled], levels, degree, rng, int(MCsize), alpha)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise CoxBootstrapFitError(
                f"bootstrap replicate {b+1} failed: {exc}", draw=b+1,
                effects=effects[:b], estimate=psi, levels=levels,
            ) from exc
        effects[b] = effect
    if b+1 != B:
        raise ValueError("row schedule has fewer than B resamples")
    covariance = np.atleast_2d(np.cov(effects, rowvar=False, ddof=1))
    se = np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("bootstrap standard errors must be positive and finite")
    critical = float(special.ndtri(1-alpha/2))
    ci = np.column_stack((psi-critical*se, psi+critical*se))
    p = 2*special.ndtr(-np.abs(psi/se))
    return CoxBootstrapResult(point, _readonly(psi), _readonly(covariance),
                              _readonly(se), _readonly(ci), _readonly(p), levels,
                              simulated_time, simulated_event, simulated_level, _readonly(z),
                              _readonly(effects), timeline, baseline, B, int(MCsize),
                              degree, alpha, seed)


def fit_cox_bootstrap(durations: Any, events: Any, exposures: Any,
                      covariates: Any = None, *, q: int | None = 4,
                      breaks: Any = None, alpha: float = 0.05, degree: int = 1,
                      B: int = 200, MCsize: int | None = None,
                      seed: int | None = None) -> CoxBootstrapResult:
    """Fit a marginal Cox curve with Monte Carlo events and row-bootstrap SE.

    Parameters
    ----------
    durations, events, exposures, covariates
        Same complete, numeric right-censored inputs as :func:`fit_cox`.
    q, breaks, alpha
        Exposure scoring and confidence level as in :func:`fit_cox`.
    degree : {1, 2}, default 1
        Polynomial degree of the joint intervention score in the marginal Cox model.
    B : int, default 200
        Number of independent row-bootstrap replicates, at least two.
    MCsize : int, optional
        Simulated people per intervention level; at least the input row count.
        The default is max(input rows, 10000), as in R qgcomp 2.19.6.
    seed : int, optional
        NumPy generator seed. Does not change the global random state.

    Returns
    -------
    CoxBootstrapResult
        Marginal curve coefficients and normal intervals based on bootstrap SD.

    Notes
    -----
    Event times are drawn by inverting the fitted cumulative hazard and are
    censored at the last observed follow-up time. R qgcomp 2.19.6 instead
    compares each hazard increment directly with a uniform draw and can produce
    missing times at its last support point. This intentional difference is
    recorded in the validation report. Independent censoring, proportional
    hazards, and causal identification require substantive assessment.
    """
    if isinstance(degree, (bool, np.bool_)) or degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if isinstance(B, (bool, np.bool_)) or not isinstance(B, Integral) or B < 2:
        raise ValueError("B must be an integer at least 2")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, Integral) or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative integer below 2**64 or None")
    return _fit(durations, events, exposures, covariates, q=q, breaks=breaks,
                alpha=alpha, degree=int(degree), B=int(B), MCsize=MCsize,
                seed=None if seed is None else int(seed))
