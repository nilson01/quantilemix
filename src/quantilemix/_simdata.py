# SPDX-License-Identifier: GPL-2.0-or-later
"""Seeded quantized-exposure examples for supported outcome families."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any
import warnings

import numpy as np
from scipy.special import expit, softmax

from ._core import QGCompWarning, _real_array


def _freeze(values: Any) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class QuantizedSimulation(Mapping[str, np.ndarray]):
    """Read-only named columns and known coefficients from a simulated sample.

    The object is a mapping, so it can be passed directly as ``data`` to the
    package's named-formula fitters. Call ``to_dict`` for editable copies.
    """

    _columns: Mapping[str, np.ndarray] = field(repr=False)
    outcome_type: str
    true_intercept: float
    true_coefficients: np.ndarray = field(repr=False)
    q: int
    n: int
    seed: int | None

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    @property
    def exposure_names(self) -> tuple[str, ...]:
        return tuple(f"x{i+1}" for i in range(self.true_coefficients.shape[0]))

    @property
    def truecoefs(self) -> dict[str, Any]:
        """R-style truth metadata, separate from the simulated data columns."""
        return {"intercept": self.true_intercept,
                "coef": self.true_coefficients.copy()}

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return editable copies of all named data columns."""
        return {name: values.copy() for name, values in self._columns.items()}


def _positive(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def simdata_quantized(
    outcometype: str = "continuous", *, n: int = 100,
    corr: Any = None, b0: float = 0.0, coef: Any = (1., 0., 0., 0.),
    q: int = 4, yscale: float = 1.0, shape0: float = 3.0,
    scale0: float = 5.0, censtime: float = 4.0,
    ncheck: bool = True, seed: int | None = None,
) -> QuantizedSimulation:
    """Generate balanced quantized exposures and one synthetic outcome.

    ``outcometype`` is ``"continuous"``, ``"logistic"``, ``"survival"`` or
    ``"multinomial"``. Each exposure column has scores from zero through
    ``q-1``. ``corr`` gives signed template-copy probabilities for exposures
    after the first, not exact target Pearson correlations. A positive value
    copies the first exposure's shuffled score template; a negative value
    copies its reverse. Other rows are shuffled within that exposure.

    ``coef`` is a vector with one coefficient per exposure, except for
    multinomial outcomes, where it is a matrix of shape
    ``(exposures, nonreference_categories)``. The multinomial outcome ``y``
    uses one-based category labels. Survival returns ``time`` and ``d``;
    ``d=1`` means the event preceded administrative censoring at ``censtime``.
    The returned mapping includes ``x1``, ``x2``, ... and outcome columns;
    the known coefficients are separate attributes, not data columns.

    NumPy's local random generator makes Python runs repeat for the same seed.
    Its draws are not bitwise equal to R's random stream. Two R 2.19.6 source
    errors are corrected deliberately: when ``ncheck`` is true, ``n`` is
    increased to the next multiple of ``q`` (R adds the remainder), and
    survival baseline times use ``scale0`` (R passes ``shape0`` as scale).
    This helper generates synthetic examples; it does not validate a method.
    """
    if outcometype not in ("continuous", "logistic", "survival", "multinomial"):
        raise ValueError("outcometype must be continuous, logistic, survival or multinomial")
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("n must be a positive integer")
    if isinstance(q, bool) or not isinstance(q, Integral) or q < 2:
        raise ValueError("q must be an integer of at least two")
    if not isinstance(ncheck, bool):
        raise ValueError("ncheck must be True or False")
    if (seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)
                              or seed < 0)):
        raise ValueError("seed must be a nonnegative integer or None")
    if isinstance(b0, bool) or not isinstance(b0, Real) or not np.isfinite(b0):
        raise ValueError("b0 must be finite")
    beta = _real_array(coef, "coef")
    if outcometype == "multinomial":
        if beta.ndim != 2 or min(beta.shape) < 1:
            raise ValueError("multinomial coef must be a nonempty two-dimensional matrix")
    elif beta.ndim != 1 or beta.size < 1:
        raise ValueError("coef must be a nonempty one-dimensional vector")
    p = beta.shape[0]
    if corr is None:
        rho = np.zeros(p-1)
    else:
        rho = _real_array(corr, "corr")
        if rho.ndim == 0:
            rho = rho.reshape(1)
        if rho.ndim != 1 or len(rho) > p-1 or np.any(np.abs(rho) > 1):
            raise ValueError("corr needs at most one value per later exposure, each in [-1,1]")
        rho = np.pad(rho, (0, p-1-len(rho)))
    if outcometype == "continuous":
        yscale = _positive(yscale, "yscale")
    elif outcometype == "survival":
        shape0 = _positive(shape0, "shape0")
        scale0 = _positive(scale0, "scale0")
        censtime = _positive(censtime, "censtime")
    original_n = int(n)
    n = int(n)
    q = int(q)
    if ncheck and n % q:
        n += q-n % q
        warnings.warn(f"n increased from {original_n} to {n} for balanced score levels",
                      QGCompWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    template = rng.permutation(np.resize(np.arange(q, dtype=int), n))
    x = np.empty((n, p), dtype=int)
    x[:, 0] = template
    for j in range(1, p):
        base = template if rho[j-1] >= 0 else q-1-template
        copy = rng.random(n) < abs(rho[j-1])
        x[:, j] = base
        x[~copy, j] = rng.permutation(base[~copy])
    mu = b0+x @ beta
    columns = {f"x{j+1}": _freeze(x[:, j]) for j in range(p)}
    if outcometype == "continuous":
        columns["y"] = _freeze(mu+rng.normal(0, yscale, n))
    elif outcometype == "logistic":
        probability = expit(mu)
        if np.mean((probability < 0.005) | (probability > 0.995)) > 0.1:
            warnings.warn("More than 10% of event probabilities are below 0.5% or above 99.5%",
                          QGCompWarning, stacklevel=2)
        columns["y"] = _freeze(rng.binomial(1, probability).astype(int))
    elif outcometype == "survival":
        baseline_time = scale0*rng.weibull(shape0, n)
        event_time = baseline_time*np.exp(-mu/shape0)
        columns["time"] = _freeze(np.minimum(event_time, censtime))
        columns["d"] = _freeze((event_time < censtime).astype(int))
        if np.mean(columns["d"]) < 0.1:
            warnings.warn("Fewer than 10% of rows have an observed event",
                          QGCompWarning, stacklevel=2)
    else:
        probabilities = softmax(np.column_stack((np.full(n, b0), x @ beta)), axis=1)
        if np.mean((probabilities < 0.005) | (probabilities > 0.995)) > 0.1:
            warnings.warn("More than 10% of category probabilities are below 0.5% or above 99.5%",
                          QGCompWarning, stacklevel=2)
        columns["y"] = _freeze(
            (rng.random(n)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)+1)
    return QuantizedSimulation(MappingProxyType(columns), outcometype, float(b0),
                               _freeze(beta), q, n, seed)
