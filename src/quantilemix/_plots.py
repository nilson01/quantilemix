# SPDX-License-Identifier: GPL-2.0-or-later
"""Small displays of values already saved by fitted results.

Matplotlib is imported only when a new Axes is needed. Neither helper refits
a model, calculates uncertainty, or changes the package's core dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._binomial import BinomialResult
from ._bootstrap import GaussianBootstrapResult
from ._core import QGCompResult
from ._odds_ratio import BinomialOddsRatioResult
from ._poisson import PoissonResult
from ._poisson_bootstrap import PoissonBootstrapResult
from ._risk_ratio import BinomialRiskRatioResult


def _axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs Matplotlib; install quantilemix[plot] "
            "or pass a Matplotlib Axes as ax"
        ) from exc
    _, ax = plt.subplots()
    return ax


def plot_weights(result: QGCompResult | BinomialResult | PoissonResult,
                 *, ax: Any = None) -> Any:
    """Draw descriptive positive and negative exposure weights.

    Parameters
    ----------
    result : QGCompResult, BinomialResult or PoissonResult
        A conditional additive fit. Weights within each sign group sum to
        one when that group has a nonzero coefficient. A zero coefficient is
        omitted. Weights describe fitted coefficients; they are not causal
        importance, selection probabilities, or uncertainty measures.
    ax : Matplotlib Axes, optional
        Axes to draw on. If absent, create one with optional Matplotlib.

    Returns
    -------
    Matplotlib Axes
        Horizontal bars on the supplied or newly created Axes.

    Raises
    ------
    TypeError
        If the result is outside the three supported additive families.
    ImportError
        If new Axes are requested and Matplotlib is not installed.

    Notes
    -----
    The positive and negative groups are normalized separately. Bar lengths
    must not be compared as shares of one common total. The family changes
    the coefficient scale, not this within-sign definition.
    """
    if not isinstance(result, (QGCompResult, BinomialResult, PoissonResult)):
        raise TypeError("plot_weights needs an additive Gaussian, binomial or Poisson result")
    names = list(result.exposure_names)
    positive = np.array([result.pos_weights.get(name, 0.0) for name in names])
    negative = np.array([result.neg_weights.get(name, 0.0) for name in names])
    ax = _axes(ax)
    positions = np.arange(len(names))
    ax.barh(positions - 0.18, positive, height=0.36, label="Positive", color="#176b5b")
    ax.barh(positions + 0.18, negative, height=0.36, label="Negative", color="#a64b3c")
    ax.set_yticks(positions, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share within coefficient-sign group")
    ax.set_title("Descriptive exposure weights")
    ax.legend()
    return ax


def plot_response(result: GaussianBootstrapResult | PoissonBootstrapResult |
                  BinomialRiskRatioResult | BinomialOddsRatioResult,
                  *, ax: Any = None) -> Any:
    """Draw saved standardized responses at joint exposure scores.

    Parameters
    ----------
    result : GaussianBootstrapResult, PoissonBootstrapResult,
        BinomialRiskRatioResult or BinomialOddsRatioResult
        A supported marginal fit with saved intervention levels and
        standardized responses. Binary points are risks; Poisson points are
        expected counts, including any saved observation-time offset.
    ax : Matplotlib Axes, optional
        Axes to draw on. If absent, create one with optional Matplotlib.

    Returns
    -------
    Matplotlib Axes
        A line through the saved response points.

    Notes
    -----
    The line connects fitted points for display. It is not an uncertainty
    band or a new model fit. Each score sets every exposure to that category
    while retaining the fitted method's adjustment distribution.
    """
    if isinstance(result, (GaussianBootstrapResult, PoissonBootstrapResult)):
        values = result.intervention_means
        unit = ("Standardized mean outcome" if isinstance(result, GaussianBootstrapResult)
                else "Standardized expected count")
    elif isinstance(result, (BinomialRiskRatioResult, BinomialOddsRatioResult)):
        values = result.intervention_risks
        unit = "Standardized event risk"
    else:
        raise TypeError("plot_response needs a supported marginal bootstrap result")
    levels = np.asarray(result.intervention_levels, dtype=float)
    response = np.asarray(values, dtype=float)
    if (levels.ndim != 1 or response.shape != levels.shape or len(levels) < 2
            or not np.all(np.isfinite(levels)) or not np.all(np.isfinite(response))):
        raise ValueError("saved intervention levels and responses must be aligned finite vectors")
    ax = _axes(ax)
    ax.plot(levels, response, marker="o", color="#176b5b")
    ax.set_xticks(levels)
    ax.set_xlabel("Joint exposure score (each exposure)")
    ax.set_ylabel(unit)
    ax.set_title("Standardized response by joint score")
    return ax
