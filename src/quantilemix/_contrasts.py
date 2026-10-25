# SPDX-License-Identifier: GPL-2.0-or-later
"""Covariance of named linear contrasts, adapted from R qgcomp 2.19.6."""

from __future__ import annotations

from numbers import Real
from typing import Any, Sequence

import numpy as np

from ._core import _readonly, _real_array


def _contrast(exposure_names: Sequence[str], covariance: Any,
              *, names: Sequence[str], grad: Any = None):
    matrix = _real_array(covariance, "covariance")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    labels = tuple(names)
    if len(labels) != len(matrix) or any(not isinstance(v, str) or not v for v in labels) or len(set(labels)) != len(labels):
        raise ValueError("names must provide one distinct nonempty string per covariance column")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(matrix)[0] < -1e-10*max(1, float(np.linalg.norm(matrix, ord=2))):
        raise ValueError("covariance must be positive semidefinite")
    if isinstance(exposure_names, (str, bytes)):
        raise ValueError("exposure_names must be a sequence of names")
    selected = tuple(exposure_names)
    if not selected or len(set(selected)) != len(selected) or any(v not in labels for v in selected):
        raise ValueError("exposure_names must be distinct names present in covariance")
    contrast = np.zeros(len(labels))
    if grad is None:
        contrast[[labels.index(v) for v in selected]] = 1
    elif isinstance(grad, Real) and not isinstance(grad, bool):
        if not np.isfinite(grad):
            raise ValueError("grad must be finite")
        contrast[[labels.index(v) for v in selected]] = float(grad)
    else:
        gradient = _real_array(grad, "grad")
        if gradient.ndim != 1 or len(gradient) != len(labels):
            raise ValueError("grad must be a scalar or a full named-order gradient vector")
        contrast = gradient
    return matrix, labels, contrast


def se_comb(exposure_names: Sequence[str], covariance: Any, *,
            names: Sequence[str], grad: Any = None) -> float:
    """Standard error of a named linear combination of coefficients.

    ``names`` gives covariance row and column order. By default each selected
    exposure receives coefficient one; ``grad`` may be a scalar for all
    selected exposures or a full vector in ``names`` order.
    """
    matrix, _, contrast = _contrast(exposure_names, covariance, names=names, grad=grad)
    variance = float(contrast @ matrix @ contrast)
    if variance < -1e-12:
        raise ValueError("contrast variance is negative")
    return float(np.sqrt(max(0, variance)))


def vc_comb(exposure_names: Sequence[str], covariance: Any, *,
            names: Sequence[str], aname: str = "(Intercept)",
            grad: Any = None) -> np.ndarray:
    """Covariance of a named coefficient and an exposure contrast.

    The off-diagonal uses the supplied gradient. R qgcomp 2.19.6 does not
    apply ``grad`` to that entry; the default unweighted behavior agrees.
    """
    matrix, labels, contrast = _contrast(exposure_names, covariance, names=names, grad=grad)
    if aname not in labels:
        raise ValueError("aname must be present in covariance names")
    unit = np.zeros(len(labels))
    unit[labels.index(aname)] = 1
    transform = np.vstack((unit, contrast))
    return _readonly(transform @ matrix @ transform.T)
