# SPDX-License-Identifier: GPL-2.0-or-later
"""Restricted conditional Cox model for right-censored observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
import re
from typing import Any
import warnings

import numpy as np
from scipy import special

from ._core import (QGCompWarning, QuantizationResult, _check_columns, _column_names,
                    _matrix, _readonly, _real_array, quantize)
from ._ee import _ee_weights
from ._formula import _numeric, _parse, formula_matrices


_SURV_FORMULA = re.compile(
    r"Surv\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*,\s*([A-Za-z_][A-Za-z_0-9]*)\s*\)\Z"
)


def _partial_likelihood(design: np.ndarray, event: np.ndarray, beta: np.ndarray,
                        durations: np.ndarray | None = None,
                        offset: np.ndarray | None = None):
    """Cox log likelihood, score, and information; Efron handles shared times."""
    eta = design @ beta
    if offset is not None:
        eta = eta + offset
    shift = float(np.max(eta))
    weights = np.exp(eta - shift)
    risk0 = np.cumsum(weights)
    risk1 = np.cumsum(weights[:, None] * design, axis=0)
    risk2 = np.cumsum(np.einsum("ni,nj,n->nij", design, design, weights), axis=0)
    if durations is None or len(np.unique(durations)) == len(durations):
        mean = risk1 / risk0[:, None]
        selected = event.astype(bool)
        likelihood = float(np.sum(eta[selected] - shift - np.log(risk0[selected])))
        score = np.sum(design[selected] - mean[selected], axis=0)
        information = np.sum(risk2[selected] / risk0[selected, None, None]
                             - np.einsum("ni,nj->nij", mean[selected], mean[selected]), axis=0)
        return likelihood, score, (information + information.T) / 2
    likelihood = 0.0
    score = np.zeros(design.shape[1])
    information = np.zeros((design.shape[1], design.shape[1]))
    start = 0
    endpoints = np.flatnonzero(np.r_[np.diff(durations) != 0, True]) + 1
    for stop in endpoints:
        dying = event[start:stop].astype(bool)
        deaths = int(np.count_nonzero(dying))
        if deaths:
            death_design = design[start:stop][dying]
            death_weights = weights[start:stop][dying]
            death_risk0 = np.sum(death_weights)
            death_risk1 = death_design.T @ death_weights
            death_risk2 = np.einsum("ni,nj,n->ij", death_design, death_design,
                                    death_weights)
            likelihood += float(np.sum(eta[start:stop][dying] - shift))
            score += np.sum(death_design, axis=0)
            for step in range(deaths):
                fraction = step / deaths
                denominator = risk0[stop-1] - fraction*death_risk0
                risk_mean = (risk1[stop-1] - fraction*death_risk1) / denominator
                likelihood -= float(np.log(denominator))
                score -= risk_mean
                information += ((risk2[stop-1] - fraction*death_risk2) / denominator
                                - np.outer(risk_mean, risk_mean))
        start = stop
    return likelihood, score, (information + information.T) / 2


def _weighted_partial_likelihood(design: np.ndarray, event: np.ndarray,
                                 beta: np.ndarray, durations: np.ndarray,
                                 case_weights: np.ndarray,
                                 offset: np.ndarray | None = None):
    """Cox case-weighted Efron likelihood, score and observed information."""
    eta = design @ beta
    if offset is not None:
        eta = eta + offset
    shift = float(np.max(eta))
    risk_weight = case_weights * np.exp(eta-shift)
    risk0 = np.cumsum(risk_weight)
    risk1 = np.cumsum(risk_weight[:, None]*design, axis=0)
    risk2 = np.cumsum(np.einsum("ni,nj,n->nij", design, design, risk_weight), axis=0)
    likelihood = 0.0
    score = np.zeros(design.shape[1])
    information = np.zeros((design.shape[1], design.shape[1]))
    start = 0
    endpoints = np.flatnonzero(np.r_[np.diff(durations) != 0, True]) + 1
    for stop in endpoints:
        dying = event[start:stop].astype(bool)
        deaths = int(np.count_nonzero(dying))
        if deaths:
            death_design = design[start:stop][dying]
            death_weights = case_weights[start:stop][dying]
            death_risk_weights = risk_weight[start:stop][dying]
            death_risk0 = float(np.sum(death_risk_weights))
            death_risk1 = death_design.T @ death_risk_weights
            death_risk2 = np.einsum("ni,nj,n->ij", death_design, death_design,
                                    death_risk_weights)
            mean_death_weight = float(np.sum(death_weights)/deaths)
            likelihood += float(death_weights @ (eta[start:stop][dying]-shift))
            score += death_design.T @ death_weights
            for step in range(deaths):
                fraction = step/deaths
                denominator = risk0[stop-1]-fraction*death_risk0
                mean = (risk1[stop-1]-fraction*death_risk1)/denominator
                likelihood -= mean_death_weight*float(np.log(denominator))
                score -= mean_death_weight*mean
                information += mean_death_weight*((risk2[stop-1]-fraction*death_risk2)
                                                  / denominator-np.outer(mean, mean))
        start = stop
    return likelihood, score, (information+information.T)/2


def _weighted_score_residuals(design: np.ndarray, event: np.ndarray,
                              durations: np.ndarray, beta: np.ndarray,
                              case_weights: np.ndarray,
                              offset: np.ndarray | None = None) -> np.ndarray:
    """Per-row weighted Cox score residuals for the robust sandwich."""
    eta = design @ beta
    if offset is not None:
        eta = eta + offset
    risk = np.exp(eta-float(np.max(eta)))
    residuals = np.zeros(design.shape, dtype=float)
    for time in np.unique(durations[event.astype(bool)]):
        at_risk = durations >= time
        dying = (durations == time) & event.astype(bool)
        deaths = int(np.count_nonzero(dying))
        mean_death_weight = float(np.sum(case_weights[dying])/deaths)
        for step in range(deaths):
            fraction = step/deaths
            effective = case_weights*risk*(at_risk-fraction*dying)
            denominator = float(np.sum(effective))
            mean = (effective @ design)/denominator
            residuals[dying] += case_weights[dying, None]*(design[dying]-mean)/deaths
            residuals -= (mean_death_weight/denominator)*effective[:, None]*(design-mean)
    return residuals


def _solve_cox(design: np.ndarray, event: np.ndarray, durations: np.ndarray,
               case_weights: np.ndarray | None = None,
               offset: np.ndarray | None = None,
               cluster_inverse: np.ndarray | None = None):
    def evaluate(beta: np.ndarray):
        if case_weights is None:
            return _partial_likelihood(design, event, beta, durations, offset)
        return _weighted_partial_likelihood(design, event, beta, durations,
                                            case_weights, offset)
    beta = np.zeros(design.shape[1])
    previous, _, _ = evaluate(beta)
    for iteration in range(1, 101):
        _, score, information = evaluate(beta)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Cox information is singular") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("Cox update is nonfinite")
        scale = 1.0
        for _ in range(45):
            candidate = beta + scale * step
            likelihood, candidate_score, _ = evaluate(candidate)
            # In large simulated risk sets, a useful Newton step can change
            # log likelihood by less than floating-point summation error.
            # Accept a roundoff-sized likelihood loss only when the score
            # also improves. The final score check still decides convergence.
            roundoff = 512*np.finfo(float).eps*max(1.0, abs(previous))
            if (np.isfinite(likelihood)
                    and (likelihood >= previous - 1e-9
                         or (likelihood >= previous - roundoff
                             and np.max(np.abs(candidate_score))
                             < np.max(np.abs(score))))):
                break
            scale /= 2
        else:
            raise ValueError("Cox line search failed; check for separation")
        beta = candidate
        if np.max(np.abs(beta)) > 30:
            raise ValueError("Cox coefficients diverged; check for separation")
        if np.max(np.abs(scale * step)) <= 1e-10 * (1 + np.max(np.abs(beta))):
            break
        previous = likelihood
    else:
        raise ValueError("Cox fit did not converge in 100 iterations")
    likelihood, score, information = evaluate(beta)
    # Near an optimum, the likelihood may round to the same value at nearby
    # coefficients. Refine using the score and accept only a smaller score.
    for _ in range(3):
        if np.max(np.abs(score)) <= 1e-9:
            break
        try:
            correction = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            break
        candidate = beta + correction
        candidate_likelihood, candidate_score, candidate_information = evaluate(candidate)
        # A 30,000-row simulated risk set can change the summed likelihood by
        # several ulps even when the Newton correction greatly improves the
        # score. Keep a small roundoff allowance and require strict score gain.
        roundoff = max(1e-9, 512*np.finfo(float).eps*max(1.0, abs(likelihood)))
        if (not np.isfinite(candidate_likelihood)
                or candidate_likelihood < likelihood - roundoff
                or np.max(np.abs(candidate_score)) >= np.max(np.abs(score))):
            break
        beta, likelihood, score, information = (candidate, candidate_likelihood,
                                                candidate_score, candidate_information)
    if np.max(np.abs(score)) > 1e-7:
        raise ValueError("Cox score did not converge")
    try:
        covariance = np.linalg.solve(information, np.eye(len(beta)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Cox covariance is singular") from exc
    covariance = (covariance + covariance.T) / 2
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError("Cox covariance is not positive definite")
    if (cluster_inverse is None and
            (case_weights is None or np.all(case_weights == np.floor(case_weights)))):
        return beta, covariance, iteration, likelihood
    residuals = _weighted_score_residuals(design, event, durations, beta,
                                          (np.ones(len(event)) if case_weights is None
                                           else case_weights), offset)
    if cluster_inverse is None:
        meat = residuals.T @ residuals
    else:
        cluster_scores = np.zeros((int(np.max(cluster_inverse)) + 1, design.shape[1]))
        np.add.at(cluster_scores, cluster_inverse, residuals)
        meat = cluster_scores.T @ cluster_scores
    robust = covariance @ meat @ covariance
    robust = (robust+robust.T)/2
    if not np.all(np.isfinite(robust)) or np.linalg.eigvalsh(robust)[0] <= 0:
        raise ValueError("Cox robust covariance is not positive definite")
    return beta, robust, iteration, likelihood


def _baseline_hazard(design: np.ndarray, event: np.ndarray, durations: np.ndarray,
                     beta: np.ndarray, case_weights: np.ndarray | None = None,
                     offset: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Efron baseline cumulative hazard at every observed follow-up time."""
    eta = design @ beta
    if offset is not None:
        eta = eta + offset
    shift = float(np.max(eta))
    weights = np.exp(eta-shift)
    if case_weights is not None:
        weights = weights*case_weights
    risk = np.cumsum(weights)
    increments = []
    times = []
    start = 0
    endpoints = np.flatnonzero(np.r_[np.diff(durations) != 0, True]) + 1
    for stop in endpoints:
        deaths = event[start:stop].astype(bool)
        count = int(np.count_nonzero(deaths))
        increase = 0.0
        if count:
            death_risk = float(np.sum(weights[start:stop][deaths]))
            mean_death_weight = (1.0 if case_weights is None else
                                 float(np.sum(case_weights[start:stop][deaths])/count))
            for step in range(count):
                denominator = risk[stop-1] - (step/count)*death_risk
                increase += mean_death_weight*float(np.exp(-shift-np.log(denominator)))
        times.append(float(durations[start]))
        increments.append(increase)
        start = stop
    timeline = np.asarray(times[::-1])
    cumulative = np.cumsum(increments[::-1])
    if not np.all(np.isfinite(cumulative)) or np.any(np.diff(cumulative) < -1e-12):
        raise ValueError("baseline cumulative hazard is nonfinite or decreases")
    return _readonly(timeline), _readonly(cumulative)


@dataclass(frozen=True, eq=False)
class CoxResult:
    """Conditional joint Cox log hazard ratio and fitted survival curves."""

    coefficients: np.ndarray = field(repr=False)
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray = field(repr=False)
    exposure_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    psi: float
    var_psi: float
    se: float
    ci: tuple[float, float]
    test_statistic: float
    p_value: float
    alpha: float
    event_count: int
    iterations: int
    tied_time_groups: int
    log_likelihood: float
    _quantization: QuantizationResult = field(repr=False)
    timeline: np.ndarray = field(repr=False)
    baseline_cumulative_hazard: np.ndarray = field(repr=False)
    _covariate_columns: tuple[str, ...] | None = field(repr=False)
    reference_version: str = "2.19.6"
    family: str = "cox"
    case_weights: np.ndarray | None = field(default=None, repr=False)
    offset: np.ndarray | None = field(default=None, repr=False)
    cluster_ids: np.ndarray | None = field(default=None, repr=False)
    cluster_count: int | None = None

    @property
    def scores(self) -> np.ndarray:
        return self._quantization.scores

    @property
    def breaks(self) -> tuple[np.ndarray, ...] | None:
        return self._quantization.breaks

    @property
    def hazard_ratio(self) -> float:
        return float(np.exp(self.psi))

    @property
    def hazard_ratio_ci(self) -> tuple[float, float]:
        return float(np.exp(self.ci[0])), float(np.exp(self.ci[1]))

    @property
    def robust_variance(self) -> bool:
        """Whether clusters or fractional weights require score-sandwich variance."""
        return (self.cluster_ids is not None or
                (self.case_weights is not None
                 and bool(np.any(self.case_weights != np.floor(self.case_weights)))))

    def survival_curve(self, exposures: Any, covariates: Any = None, *,
                       offset: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """Return conditional survival at observed times for new complete rows.

        Parameters
        ----------
        exposures : array-like, shape (n, number of fitted exposures)
            New exposure measurements. Stored quantile cuts are reused.
        covariates : array-like, optional
            New adjustment values in fitted column order; required if fitted.
        offset : array-like, optional
            Known log-hazard offsets for new rows. Required when the fitted
            model used an offset.

        Returns
        -------
        times : read-only ndarray, shape (number of observed times,)
            Positive follow-up times in increasing order, including censor-only times.
        survival : read-only ndarray, shape (number of times, n)
            Conditional survival probabilities. No uncertainty band is supplied.

        Notes
        -----
        These are model-based conditional predictions, not population-average
        survival effects. Proportional hazards and independent censoring remain
        substantive assumptions. Survival before the first returned time is one.
        """
        transformed = self._quantization.transform(exposures)
        n = len(transformed.scores)
        if self.covariate_names:
            if covariates is None:
                raise ValueError("covariates are required because this Cox model was fitted with them")
            z, columns = _matrix(covariates, "covariates")
            if z.shape != (n, len(self.covariate_names)):
                raise ValueError("covariates must match prediction rows and fitted column count")
            _check_columns(columns, self._covariate_columns, "covariates")
        else:
            if covariates is not None:
                raise ValueError("this Cox model was fitted without covariates")
            z = np.empty((n, 0))
        if transformed.diagnostics["extrapolated_rows"]:
            warnings.warn("Some exposures are outside their training ranges; fitted cutpoints are reused.",
                          QGCompWarning, stacklevel=2)
        design = np.column_stack((transformed.scores, z))
        if offset is None:
            if self.offset is not None:
                raise ValueError("prediction offset is required for a Cox fit with an offset")
            prediction_offset = np.zeros(n)
        else:
            prediction_offset = _real_array(offset, "offset")
            if prediction_offset.ndim != 1 or prediction_offset.size != n:
                raise ValueError("offset must have one value per prediction row")
        with np.errstate(over="ignore", invalid="ignore"):
            hazard = (self.baseline_cumulative_hazard[:, None]
                      * np.exp(design @ self.coefficients + prediction_offset)[None, :])
            survival = np.exp(-hazard)
        if not np.all(np.isfinite(survival)) or not np.all(np.isfinite(hazard)):
            raise ValueError("conditional survival prediction is nonfinite")
        return self.timeline, _readonly(survival)

    def summary(self, digits: int = 6) -> str:
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 through 17")
        def f(value: float) -> str:
            return format(value, f".{digits}g")
        return ("Quantile g-computation: conditional Cox proportional hazards\n"
                f"Events: {self.event_count}; Efron tie handling; "
                f"shared-time groups: {self.tied_time_groups}; "
                f"{self.iterations} fitting iterations.\n"
                f"Case weights: {'supplied' if self.case_weights is not None else 'none'}; "
                f"{'cluster-robust' if self.cluster_ids is not None else ('robust score-sandwich' if self.robust_variance else 'model-based')} SE.\n"
                f"Clusters: {self.cluster_count if self.cluster_count is not None else 'none'}.\n"
                f"Known log-hazard offset: {'supplied' if self.offset is not None else 'none'}.\n"
                "Joint effect: one score category in every exposure, covariates fixed.\n"
                f"Log hazard ratio: {f(self.psi)}; SE: {f(self.se)}\n"
                f"Normal {(1-self.alpha)*100:g}% CI: ({f(self.ci[0])}, {f(self.ci[1])})\n"
                f"Conditional hazard ratio: {f(self.hazard_ratio)}\n"
                f"Two-sided normal z p-value: {f(self.p_value)}\n"
                "Baseline curve is available for conditional survival predictions. "
                "No marginal survival effect is estimated.")


def fit_cox(durations: Any, events: Any, exposures: Any, covariates: Any = None, *,
            q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
            weights: Any = None, offset: Any = None,
            cluster_ids: Any = None) -> CoxResult:
    """Fit a right-censored Cox model and sum exposure coefficients.

    Durations must be positive, events must be zero or one, and
    inputs must be complete numeric arrays. Positive case weights enter the
    risk sets and event contributions. As in R's ``coxph`` default, fractional
    weights trigger a robust score-sandwich variance; integer weights use
    model-based variance. A finite ``offset`` gives one known log-hazard
    contribution per row. Exact integer ``cluster_ids`` group row scores for
    robust uncertainty; they do not change the fitted coefficients or exposure
    cuts. This subset has no formula. Shared event
    times use Efron's approximation. Its hazard ratio is conditional, not a
    marginal risk or survival contrast.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    duration = _real_array(durations, "durations")
    event = _real_array(events, "events")
    if duration.ndim != 1 or event.ndim != 1 or len(duration) == 0 or len(duration) != len(event):
        raise ValueError("durations and events must be nonempty vectors of equal length")
    if np.any(duration <= 0):
        raise ValueError("durations must be positive")
    if np.any((event != 0) & (event != 1)):
        raise ValueError("events must contain only zero and one")
    transformation = quantize(exposures, q=q, breaks=breaks)
    n, d = transformation.scores.shape
    if n != len(duration):
        raise ValueError("durations, events, and exposures must have the same rows")
    case_weights, _ = _ee_weights(weights, n)
    if offset is None:
        log_offset = None
    else:
        log_offset = _real_array(offset, "offset")
        if log_offset.ndim != 1 or log_offset.size != n:
            raise ValueError("offset must have one value per row")
    if cluster_ids is None:
        ids = inverse = None
        cluster_count = None
    else:
        ids = _real_array(cluster_ids, "cluster_ids")
        if (ids.ndim != 1 or ids.size != n or np.any(ids != np.floor(ids))
                or np.any(np.abs(ids) >= 2**53)):
            raise ValueError("cluster_ids must contain one finite exact integer ID per row")
        _, inverse = np.unique(ids, return_inverse=True)
        cluster_count = int(np.max(inverse) + 1)
        if cluster_count < 2:
            raise ValueError("Cox cluster-robust variance needs at least two clusters")
    if covariates is None:
        z, names = np.empty((n, 0)), ()
    else:
        z, labels = _matrix(covariates, "covariates")
        if len(z) != n:
            raise ValueError("covariates and exposures must have the same rows")
        names = _column_names(labels, z.shape[1], "z")
    coefficient_names = transformation.feature_names + names
    if len(set(coefficient_names)) != len(coefficient_names):
        raise ValueError("coefficient identifiers must be distinct")
    design = np.column_stack((transformation.scores, z))
    if n <= design.shape[1] or np.count_nonzero(event) <= design.shape[1]:
        raise ValueError("Cox fit needs more events than coefficients")
    order = np.argsort(-duration, kind="stable")
    coefficients, covariance, iterations, log_likelihood = _solve_cox(
        design[order], event[order], duration[order],
        None if case_weights is None else case_weights[order],
        None if log_offset is None else log_offset[order],
        None if inverse is None else inverse[order])
    timeline, baseline_cumulative_hazard = _baseline_hazard(
        design[order], event[order], duration[order], coefficients,
        None if case_weights is None else case_weights[order],
        None if log_offset is None else log_offset[order])
    _, time_counts = np.unique(duration, return_counts=True)
    tied_time_groups = int(np.count_nonzero(time_counts > 1))
    contrast = np.zeros(len(coefficients))
    contrast[:d] = 1
    psi = float(contrast @ coefficients)
    variance = float(contrast @ covariance @ contrast)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("joint log-hazard variance is not positive and finite")
    se = float(np.sqrt(variance))
    critical = float(-special.ndtri_exp(np.log(alpha) - np.log(2)))
    ci = float(psi-critical*se), float(psi+critical*se)
    statistic = psi / se
    return CoxResult(_readonly(coefficients), coefficient_names, _readonly(covariance),
                     transformation.feature_names, names, psi, variance, se, ci,
                     statistic, float(2*special.ndtr(-abs(statistic))), float(alpha),
                     int(np.count_nonzero(event)), iterations, tied_time_groups,
                     log_likelihood, transformation, timeline, baseline_cumulative_hazard,
                     None if covariates is None else labels, case_weights=case_weights,
                     offset=None if log_offset is None else _readonly(log_offset),
                     cluster_ids=None if ids is None else _readonly(ids),
                     cluster_count=cluster_count)


def fit_cox_formula(formula: str, data: Any, exposures: Any, *,
                    q: int | None = 4, breaks: Any = None,
                    alpha: float = 0.05, weights: Any = None,
                    cluster_ids: Any = None, offset: Any = None) -> CoxResult:
    """Fit a complete-data additive survival formula.

    Write ``Surv(time,event) ~ exposure + C(adjustment)``. The two response
    columns must hold positive follow-up times and zero-one event indicators.
    Exposure terms must be plain numeric columns. ``C(name)`` is allowed for
    a text adjustment column, with the first sorted level as reference.
    ``weights``, ``cluster_ids`` and known log-hazard ``offset`` are row
    vectors in the same order as ``data``. Missing rows are not removed.
    """
    if not isinstance(formula, str) or formula.count("~") != 1:
        raise ValueError("survival formula must have the form 'Surv(time,event) ~ terms'")
    lhs, rhs = (part.strip() for part in formula.split("~"))
    match = _SURV_FORMULA.fullmatch(lhs)
    if match is None or match.group(1) == match.group(2):
        raise ValueError("survival formula must start with Surv(time,event)")
    time_name, event_name = match.groups()
    schema = _parse(f"__qgcomp_survival_response__ ~ {rhs}", exposures)
    if time_name in (*schema.exposure_names, *(name for name, _ in schema.covariate_terms)) \
            or event_name in (*schema.exposure_names, *(name for name, _ in schema.covariate_terms)):
        raise ValueError("time and event columns cannot also be model terms")
    x, z, _ = formula_matrices(data, schema, training=True)
    duration = _numeric(data, time_name, len(x.values))
    event = _numeric(data, event_name, len(x.values))
    return fit_cox(duration, event, x, z, q=q, breaks=breaks, alpha=alpha,
                   weights=weights, cluster_ids=cluster_ids, offset=offset)
