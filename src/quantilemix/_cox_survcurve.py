# SPDX-License-Identifier: GPL-2.0-or-later
"""Survival curves from a fitted marginal Cox bootstrap model."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ._core import _readonly
from ._cox_bootstrap import CoxBootstrapResult


@dataclass(frozen=True, eq=False)
class CoxSurvivalCurves:
    """Four read-only curve tables named as in R qgcomp 2.19.6.

    Population tables have ``time`` and ``surv`` columns. By-level tables
    also have a one-based ``q`` column, ordered by intervention level and
    then time. ``levels`` stores the actual joint exposure scores.
    """

    mdfpop: Mapping[str, np.ndarray] = field(repr=False)
    cdfpop: Mapping[str, np.ndarray] = field(repr=False)
    mdfq: Mapping[str, np.ndarray] = field(repr=False)
    cdfq: Mapping[str, np.ndarray] = field(repr=False)
    levels: np.ndarray = field(repr=False)
    reference_version: str = "2.19.6"

    def __getitem__(self, name: str) -> Mapping[str, np.ndarray]:
        if name not in ("mdfpop", "cdfpop", "mdfq", "cdfq"):
            raise KeyError(name)
        return getattr(self, name)


def _table(time: np.ndarray, survival: np.ndarray,
           level: np.ndarray | None = None) -> Mapping[str, np.ndarray]:
    columns = {"time": _readonly(time), "surv": _readonly(survival)}
    if level is not None:
        columns["q"] = _readonly(level)
    return MappingProxyType(columns)


def _by_level(time: np.ndarray, curves: np.ndarray) -> Mapping[str, np.ndarray]:
    nlevels = curves.shape[1]
    return _table(np.tile(time, nlevels), curves.T.reshape(-1),
                  np.repeat(np.arange(1, nlevels+1), len(time)))


def qgcomp_survcurve_boot(result: CoxBootstrapResult) -> CoxSurvivalCurves:
    """Return marginal and conditional survival curves without uncertainty bands.

    ``mdfq`` is the marginal model curve at each fitted joint intervention
    level. ``mdfpop`` averages those curves equally across levels. ``cdfpop``
    averages conditional-model curves over all original adjustment profiles
    and levels. Following R's helper, ``cdfq`` holds each level's conditional
    curve for the *first* original adjustment profile, not its population
    average. The four tables may have different time grids: the marginal
    grid comes from simulated outcomes and the conditional grid from observed
    outcomes. No confidence bands are supplied.

    For a prequantized fit, Python uses the actual pooled-quartile
    intervention levels saved by the bootstrap fit. R qgcomp 2.19.6 instead
    reconstructs zero-based integer levels from its saved breaks when q is
    NULL; the two displays can differ in that setting.
    """
    if not isinstance(result, CoxBootstrapResult):
        raise TypeError("result must be a CoxBootstrapResult")
    levels = result.intervention_levels
    marginal_time = result.marginal_timeline
    conditional = result.conditional_fit
    conditional_time = conditional.timeline
    n = len(conditional.scores)
    z = result.training_covariates
    if z.shape != (n, len(conditional.covariate_names)):
        raise ValueError("saved adjustment profiles do not match the conditional fit")

    powers = np.column_stack([levels**power for power in range(1, result.degree+1)])
    marginal_eta = powers @ result.psi
    conditional_eta = (levels[:, None]*np.sum(conditional.coefficients[:len(conditional.exposure_names)])
                       + z[None, :, :] @ conditional.coefficients[len(conditional.exposure_names):])
    with np.errstate(over="ignore", invalid="ignore"):
        marginal_curves = np.exp(-result.marginal_baseline_cumulative_hazard[:, None]
                                 * np.exp(marginal_eta)[None, :])
        conditional_curves = np.exp(-conditional.baseline_cumulative_hazard[:, None, None]
                                    * np.exp(conditional_eta)[None, :, :])
    if (not np.all(np.isfinite(marginal_curves))
            or not np.all(np.isfinite(conditional_curves))):
        raise ValueError("survival curves exceed the finite numeric range")
    # The conditional matrix is time by level by original adjustment profile.
    first_profile = conditional_curves[:, :, 0]
    return CoxSurvivalCurves(
        _table(marginal_time, marginal_curves.mean(axis=1)),
        _table(conditional_time, conditional_curves.mean(axis=(1, 2))),
        _by_level(marginal_time, marginal_curves),
        _by_level(conditional_time, first_profile),
        levels,
    )
