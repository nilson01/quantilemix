# SPDX-License-Identifier: GPL-2.0-or-later
"""Separate reference for the fixed four-level study's case streams."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import expit


def case(mode: str, cell: int, repetition: int):
    if mode not in ("pilot", "study") or cell not in range(4):
        raise ValueError("case outside fixed design")
    limit = 5 if mode == "pilot" else 1000
    if repetition not in range(limit):
        raise ValueError("repetition outside fixed design")
    baseline = (.10, .40)[cell // 2]
    contrast = (0.0, .80)[cell % 2]
    low, high = -30.0, 30.0
    for _ in range(100):
        middle = (low + high) / 2
        average = (expit(middle) + expit(middle + .5)) / 2
        if average < baseline:
            low = middle
        else:
            high = middle
    intercept = (low + high) / 2
    stream = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [2026102402, 111 if mode == "pilot" else 112, cell, repetition])))
    exposure = stream.integers(0, 4, size=(500, 4), dtype=np.int64)
    adjustment = stream.integers(0, 2, size=(500, 1), dtype=np.int64)
    probability = expit(intercept + contrast * exposure.sum(axis=1) / 12
                        + .5 * adjustment[:, 0])
    outcome = stream.binomial(1, probability).astype(np.int64)
    seed = int(np.random.SeedSequence(
        [2026102402, 113, 1 if mode == "pilot" else 2, cell, repetition]
    ).generate_state(1, dtype=np.uint64)[0])
    return exposure, adjustment, outcome, seed


def digest(exposure: np.ndarray, adjustment: np.ndarray, outcome: np.ndarray) -> str:
    raw = b"".join(np.asarray(array, dtype="<i8").tobytes()
                   for array in (exposure, adjustment, outcome))
    return hashlib.sha256(raw).hexdigest()
