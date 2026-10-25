# SPDX-License-Identifier: GPL-2.0-or-later
"""Response-scale prediction from supported bootstrap marginal models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._bootstrap import GaussianBootstrapResult
from ._core import _readonly, _real_array
from ._poisson_bootstrap import PoissonBootstrapResult


def msm_predict(
    result: GaussianBootstrapResult | PoissonBootstrapResult,
    newdata: Any = None,
) -> np.ndarray:
    """Predict from a fitted bootstrap marginal structural model.

    Parameters
    ----------
    result
        An additive Gaussian or Poisson row-bootstrap fit. Conditional fits
        and other bootstrap families are outside this interface.
    newdata
        Optional mapping or data frame with a one-dimensional ``psi`` column
        of joint exposure scores. A one-dimensional numeric array of scores
        is also accepted. Scores may lie outside the fitted levels; these
        predictions extrapolate the fitted linear response on its link scale.
        With ``None``, return one prediction for every original row at every
        fitted intervention level, in consecutive level blocks as in R.

    Returns
    -------
    ndarray
        Read-only one-dimensional predictions on the outcome scale. Gaussian
        values are marginal expected outcomes; Poisson values are marginal
        expected counts over the observation times in the fitted data.

    Raises
    ------
    TypeError
        If ``result`` is not a supported bootstrap fit.
    ValueError
        If the new score column is missing, empty, nonnumeric, nonfinite or
        not one-dimensional, or predictions exceed the finite numeric range.

    Notes
    -----
    This matches the selected additive behavior of R qgcomp 2.19.6
    ``msm.predict``. It predicts from the marginal model, not the conditional
    outcome model. It returns no confidence or individual prediction interval.
    """
    if not isinstance(result, (GaussianBootstrapResult, PoissonBootstrapResult)):
        raise TypeError("result must be a supported Gaussian or Poisson bootstrap fit")

    levels = result.intervention_levels
    means = result.intervention_means
    if newdata is None:
        n = len(result.point_fit.scores)
        values = np.repeat(means, n)
    else:
        if isinstance(newdata, Mapping) or hasattr(newdata, "columns"):
            if "psi" not in newdata:
                raise ValueError("newdata must contain a psi score column")
            raw_scores = newdata["psi"]
        else:
            raw_scores = newdata
        scores = _real_array(raw_scores, "newdata psi")
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("newdata psi must be a nonempty one-dimensional column")
        displacement = scores - float(levels[0])
        if isinstance(result, GaussianBootstrapResult):
            values = float(means[0]) + displacement * float(result.psi)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.exp(np.log(float(means[0])) + displacement * float(result.psi))
    if not np.all(np.isfinite(values)):
        raise ValueError("marginal predictions exceed the finite numeric range")
    return _readonly(values)
