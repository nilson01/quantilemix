# SPDX-License-Identifier: GPL-2.0-or-later
"""One-split additive partial effects, following R qgcomp 2.19.6."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ._core import QGCompResult, _matrix, _real_array, fit
from ._binomial import BinomialResult, fit_binomial
from ._poisson import PoissonResult, fit_poisson


@dataclass(frozen=True, eq=False)
class DataSplit:
    """Disjoint, zero-based training and validation rows and their data."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_data: Any
    validation_data: Any


def _data_length(data: Any) -> int:
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data must have at least one column")
        lengths = [len(np.asarray(v)) for v in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("data columns must have the same number of rows")
        return lengths[0]
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) == 2 and shape[1] > 0:
        return int(shape[0])
    raise ValueError("data must be a named-column mapping or a two-dimensional table")


def _rows(data: Any, indices: np.ndarray) -> Any:
    if isinstance(data, Mapping):
        return {name: np.asarray(values)[indices].copy() for name, values in data.items()}
    if hasattr(data, "iloc"):
        return data.iloc[indices].copy()
    return np.asarray(data)[indices].copy()


def split_data(
    data: Any, *, cluster: str | None = None, prop_train: float = 0.4,
    seed: int | None = None,
) -> DataSplit:
    """Randomly partition rows, with optional whole-cluster assignment.

    The training fraction is rounded to the nearest integer. A seed makes the
    Python split reproducible; it does not reproduce R's random-number stream.
    With ``cluster``, each distinct cluster is assigned wholly to one side.
    This corrects an R 2.19.6 implementation difference: its cluster helper
    samples rows *inside* each cluster despite its group-preserving description.
    Missing cluster IDs and a split with an empty side are rejected.
    """
    n = _data_length(data)
    if isinstance(prop_train, (bool, np.bool_)) or not isinstance(prop_train, Real):
        raise ValueError("prop_train must be a finite fraction strictly between 0 and 1")
    if not np.isfinite(prop_train) or not 0 < prop_train < 1:
        raise ValueError("prop_train must be a finite fraction strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    if cluster is None:
        units = np.arange(n)
        membership = units
    else:
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("cluster must name a column")
        try:
            column = data[cluster]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"cluster column {cluster!r} was not found") from exc
        values = np.asarray(column)
        if values.ndim != 1 or len(values) != n:
            raise ValueError("cluster column must have one value per row")
        if any(v is None or (isinstance(v, Real) and not np.isfinite(v)) for v in values):
            raise ValueError("cluster IDs must be complete")
        try:
            units, membership = np.unique(values, return_inverse=True)
        except TypeError as exc:
            raise ValueError("cluster IDs must have a common comparable type") from exc
    count = int(np.floor(len(units) * float(prop_train) + 0.5))
    if count == 0 or count == len(units):
        raise ValueError("the split needs at least one training and one validation unit")
    selected = rng.choice(len(units), size=count, replace=False)
    train = np.flatnonzero(np.isin(membership, selected))
    validation = np.flatnonzero(~np.isin(membership, selected))
    return DataSplit(tuple(map(int, train)), tuple(map(int, validation)),
                     _rows(data, train), _rows(data, validation))


@dataclass(frozen=True, eq=False)
class GaussianPartialResult:
    """Training-selected exposure groups and held-out Gaussian fits."""

    training_fit: QGCompResult
    positive_exposures: tuple[str, ...]
    negative_exposures: tuple[str, ...]
    positive_fit: QGCompResult | None
    negative_fit: QGCompResult | None
    fixed_breaks: bool

    def summary(self) -> str:
        """Show held-out partial estimates; empty groups have no estimate."""
        lines = ["Gaussian sample-split partial effects",
                 f"Training positive group: {', '.join(self.positive_exposures) or 'none'}",
                 f"Training negative group: {', '.join(self.negative_exposures) or 'none'}"]
        for name, result in (("Positive", self.positive_fit), ("Negative", self.negative_fit)):
            if result is None:
                lines.append(f"{name} held-out effect: unavailable (empty training group)")
            else:
                lines.append(f"{name} held-out effect: {result.psi:.6g}; "
                             f"SE {result.se:.6g}; CI ({result.ci[0]:.6g}, {result.ci[1]:.6g})")
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class GLMPartialResult:
    """Training-selected groups and held-out binary or count model fits."""

    training_fit: BinomialResult | PoissonResult
    positive_exposures: tuple[str, ...]
    negative_exposures: tuple[str, ...]
    positive_fit: BinomialResult | PoissonResult | None
    negative_fit: BinomialResult | PoissonResult | None
    fixed_breaks: bool
    family: str

    def summary(self) -> str:
        """Show conditional held-out contrasts on the fitted link scale."""
        scale = "log odds ratio" if self.family == "binomial" else "log rate ratio"
        lines = [f"{self.family.capitalize()} sample-split partial effects ({scale})",
                 f"Training positive group: {', '.join(self.positive_exposures) or 'none'}",
                 f"Training negative group: {', '.join(self.negative_exposures) or 'none'}"]
        for name, result in (("Positive", self.positive_fit), ("Negative", self.negative_fit)):
            if result is None:
                lines.append(f"{name} held-out effect: unavailable (empty training group)")
            else:
                lines.append(f"{name} held-out effect: {result.psi:.6g}; "
                             f"SE {result.se:.6g}; CI ({result.ci[0]:.6g}, {result.ci[1]:.6g})")
        return "\n".join(lines)


class _NamedMatrix:
    """Pass selected names through the ordinary validated fit interface."""

    def __init__(self, values: np.ndarray, names: tuple[str, ...]):
        self.values = values
        self.columns = names

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=copy)


def qgcomp_partials(
    train_y: Any, train_exposures: Any, validation_y: Any,
    validation_exposures: Any, *, train_covariates: Any = None,
    validation_covariates: Any = None, q: int | None = 4,
    fixed_breaks: bool = True, alpha: float = 0.05,
    family: str = "gaussian",
) -> GaussianPartialResult | GLMPartialResult:
    """Select coefficient signs in training rows and estimate in held-out rows.

    This interface needs an independently chosen train/validation split.
    ``family`` is ``"gaussian"``, ``"binomial"`` or ``"poisson"``. For each
    nonempty sign group, its held-out fit treats the other
    exposures as adjustment variables. With raw exposures, only the selected
    group is quantized in held-out fits, matching R's ``expnms`` convention.
    The default reuses training cutpoints. ``fixed_breaks=False`` computes new
    held-out cutpoints. The binary result is a conditional log odds ratio;
    the count result is a conditional log mean ratio for equal observation
    periods. Cross-fitting, bootstrap, case weights, Poisson time offsets,
    other families and inference for a data-dependent split are outside this
    interface. Keeping clusters on one side with ``split_data`` does not make
    these model-based intervals cluster-robust.
    """
    if not isinstance(fixed_breaks, bool):
        raise ValueError("fixed_breaks must be a boolean")
    if family not in ("gaussian", "binomial", "poisson"):
        raise ValueError("family must be 'gaussian', 'binomial' or 'poisson'")
    fitter = {"gaussian": fit, "binomial": fit_binomial, "poisson": fit_poisson}[family]
    tx, tnames = _matrix(train_exposures, "train_exposures")
    vx, vnames = _matrix(validation_exposures, "validation_exposures")
    if tx.shape[1] != vx.shape[1] or tnames != vnames:
        raise ValueError("training and validation exposures must have matching columns")
    if tx.shape[0] == 0 or vx.shape[0] == 0:
        raise ValueError("both split parts must contain rows")
    if (train_covariates is None) != (validation_covariates is None):
        raise ValueError("training and validation covariates must both be present or absent")
    tz = vz = None
    if train_covariates is not None:
        tz, _ = _matrix(train_covariates, "train_covariates")
        vz, _ = _matrix(validation_covariates, "validation_covariates")
        if tz.shape[0] != len(tx) or vz.shape != (len(vx), tz.shape[1]):
            raise ValueError("covariates must match the split rows and columns")
    ty = _real_array(train_y, "train_y")
    vy = _real_array(validation_y, "validation_y")
    if ty.ndim != 1 or vy.ndim != 1 or len(ty) != len(tx) or len(vy) != len(vx):
        raise ValueError("outcomes must have one value per split row")
    training = fitter(ty, train_exposures, train_covariates, q=q, alpha=alpha)
    names = training.exposure_names
    coefficients = training.coefficients[1:1+len(names)]
    positive = tuple(names[i] for i, b in enumerate(coefficients) if b > 0)
    negative = tuple(names[i] for i, b in enumerate(coefficients) if b < 0)

    def held_out(group: tuple[str, ...]) -> QGCompResult | BinomialResult | PoissonResult | None:
        if not group:
            return None
        chosen = [names.index(name) for name in group]
        other = [i for i in range(len(names)) if i not in chosen]
        adjustment = vx[:, other] if vz is None else np.column_stack((vx[:, other], vz))
        adjustment_names = tuple(names[i] for i in other) + training.covariate_names
        cuts = tuple(training.breaks[i] for i in chosen) if fixed_breaks and training.breaks else None
        return fitter(vy, _NamedMatrix(vx[:, chosen], group),
                      _NamedMatrix(adjustment, adjustment_names) if adjustment.shape[1] else None,
                      q=q, breaks=cuts, alpha=alpha)

    if family == "gaussian":
        return GaussianPartialResult(training, positive, negative,
                                     held_out(positive), held_out(negative), fixed_breaks)
    return GLMPartialResult(training, positive, negative,
                            held_out(positive), held_out(negative), fixed_breaks, family)
