# SPDX-License-Identifier: GPL-2.0-or-later
"""Formula-based selection among supported additive models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._binomial import BinomialResult, fit_binomial_formula
from ._core import QGCompResult
from ._cox import CoxResult, fit_cox_formula
from ._formula import _numeric, _parse, fit_formula, formula_matrices
from ._poisson import PoissonResult, fit_poisson_formula
from ._risk_ratio import BinomialRiskRatioResult, fit_binomial_risk_ratio


def qgcomp(
    formula: str, data: Mapping[str, Any], exposures: Sequence[str], *,
    family: str = "gaussian", rr: bool = True,
    q: int | None = 4, breaks: Any = None, alpha: float = 0.05,
    weights: Any = None, missing: str = "raise",
    B: int | None = None, seed: int | None = None,
    exposure_time: str | None = None,
    cluster_ids: Any = None, offset: Any = None,
) -> QGCompResult | BinomialResult | BinomialRiskRatioResult | PoissonResult | CoxResult:
    """Choose a supported additive quantile g-computation fit by family.

    Parameters
    ----------
    formula, data, exposures
        A simple additive formula, named columns, and the exposure names.
        ``C(name)`` may encode a categorical adjustment variable. Write an
        observation-time offset as ``exposure_time="column"`` instead of an
        R-style ``offset(log(column))`` formula term.
    family
        ``"gaussian"`` for a continuous outcome, ``"binomial"`` for a
        zero-one outcome, ``"poisson"`` for nonnegative integer counts, or
        ``"cox"`` for a conditional right-censored survival fit.
    rr
        For binomial outcomes, ``True`` (R's default) selects a marginal
        bootstrap risk ratio per one joint-score increase.
        ``False`` selects a conditional odds ratio. R ignores ``rr`` for
        Gaussian and Poisson outcomes, and so does this dispatcher.
    q, breaks, alpha
        Exposure categories or saved boundaries and interval error level,
        passed to the selected fitter.
    weights, missing
        Positive case weights and explicit ``"raise"`` or ``"drop"`` missing
        row handling for conditional fits. The binary risk-ratio branch does
        not yet support weights or missing-row removal.
    B, seed
        Row-bootstrap count and local random seed, accepted only for the
        binary risk-ratio branch. Its default count is 200.
    exposure_time
        Name of a positive observation-time column for Poisson conditional
        fitting. This corresponds to an R ``offset(log(time))`` term; R's
        additive dispatcher sends that term to its non-bootstrap fit.
    cluster_ids, offset
        Optional row-aligned cluster IDs and known log-hazard offsets for
        the conditional Cox route only.

    Returns
    -------
    QGCompResult or BinomialResult or BinomialRiskRatioResult or PoissonResult or CoxResult
        The unchanged result type of the selected public fitter. These result
        types carry different effect measures and uncertainty methods.

    Raises
    ------
    ValueError
        If the family, formula, options or selected route are unsupported.

    Notes
    -----
    The supported routes follow selected behavior of R qgcomp 2.19.6. In
    particular, binomial ``rr=True`` with four quantiles uses a degree-one
    log-binomial marginal structural model, not the two-level risk ratio.
    Marginal survival bootstrap, nonlinear, zero-inflated and other routes
    need separate methods.
    """
    if family not in ("gaussian", "binomial", "poisson", "cox"):
        raise ValueError("family must be 'gaussian', 'binomial', 'poisson' or 'cox'")
    if not isinstance(rr, (bool, np.bool_)):
        raise ValueError("rr must be True or False")
    if exposure_time is not None and family != "poisson":
        raise ValueError("exposure_time is supported only for Poisson outcomes")
    if family != "cox" and (cluster_ids is not None or offset is not None):
        raise ValueError("cluster_ids and offset are supported only for Cox outcomes")
    if B is not None or seed is not None:
        if family != "binomial" or not rr:
            raise ValueError("B and seed apply only to the binomial risk-ratio route")

    if family == "gaussian":
        return fit_formula(formula, data, exposures, q=q, breaks=breaks,
                           alpha=alpha, weights=weights, missing=missing)
    if family == "poisson":
        return fit_poisson_formula(formula, data, exposures, q=q, breaks=breaks,
                                   alpha=alpha, weights=weights, missing=missing,
                                   exposure_time=exposure_time)
    if family == "cox":
        if missing != "raise":
            raise ValueError("Cox formula fitting currently requires missing='raise'")
        return fit_cox_formula(formula, data, exposures, q=q, breaks=breaks,
                               alpha=alpha, weights=weights,
                               cluster_ids=cluster_ids, offset=offset)
    if not rr:
        return fit_binomial_formula(formula, data, exposures, q=q, breaks=breaks,
                                    alpha=alpha, weights=weights, missing=missing)

    if weights is not None:
        raise ValueError("binary rr=True does not yet support case weights")
    if missing != "raise":
        raise ValueError("binary rr=True currently requires missing='raise'")
    schema = _parse(formula, exposures)
    x, z, schema = formula_matrices(data, schema, training=True)
    outcome = _numeric(data, schema.outcome, len(x.values))
    return fit_binomial_risk_ratio(outcome, x, z, q=q, breaks=breaks,
                                   alpha=alpha, B=200 if B is None else B,
                                   seed=seed)
