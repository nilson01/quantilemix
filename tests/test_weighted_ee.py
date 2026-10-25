"""Independent row-replication and failure checks for weighted cluster EE."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_gaussian_ee, fit_poisson_ee, fit_binomial_ee


CASES = json.loads((Path(__file__).parent / "fixtures/r_weighted_ee_2_19_6.json")
                   .read_text())["cases"]
FITS = {"gaussian": fit_gaussian_ee, "poisson": fit_poisson_ee,
        "binomial": fit_binomial_ee}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_integer_weights_equal_repeated_rows_with_cluster_ids(case):
    """Each integer-weighted score equals that row repeated in its cluster."""
    a = case["inputs"]
    x = np.asarray(case["expected"]["scores"], dtype=float)
    y, ids, z = (np.asarray(a[key], dtype=float) for key in ("y", "id", "z"))
    w = np.asarray(a["w"], dtype=int)
    fit = FITS[case["family"]]
    options = {"rr": case["rr"]} if case["family"] == "binomial" else {}
    weighted = fit(y, x, ids, z, q=None, weights=w, **options)
    rows = np.repeat(np.arange(len(y)), w)
    expanded = fit(y[rows], x[rows], ids[rows], z[rows], q=None, **options)
    np.testing.assert_allclose(weighted.conditional_coefficients,
                               expanded.conditional_coefficients, atol=1e-11)
    np.testing.assert_allclose(weighted.msm_coefficients,
                               expanded.msm_coefficients, atol=1e-11)
    np.testing.assert_allclose(weighted.full_covariance,
                               expanded.full_covariance, atol=1e-10)
    np.testing.assert_allclose(weighted.bread, expanded.bread, atol=1e-9)
    np.testing.assert_allclose(weighted.meat, expanded.meat, atol=1e-9)
    np.testing.assert_allclose(weighted.ci, expanded.ci, atol=1e-10)
    assert weighted.cluster_count == expanded.cluster_count
    np.testing.assert_array_equal(weighted.case_weights, w)
    assert not weighted.case_weights.flags.writeable


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_common_weight_multiplier_leaves_cluster_covariance_unchanged(case):
    a = case["inputs"]
    x = np.asarray(case["expected"]["scores"], dtype=float)
    fit = FITS[case["family"]]
    options = {"rr": case["rr"]} if case["family"] == "binomial" else {}
    one = fit(a["y"], x, a["id"], a["z"], q=None, weights=a["w"], **options)
    twice = fit(a["y"], x, a["id"], a["z"], q=None,
                weights=2*np.asarray(a["w"]), **options)
    np.testing.assert_allclose(one.psi, twice.psi, atol=1e-12)
    np.testing.assert_allclose(one.full_covariance, twice.full_covariance, atol=1e-11)


@pytest.mark.parametrize("bad", [lambda w: w[:-1], lambda w: np.zeros_like(w),
                                 lambda w: -w, lambda w: np.full_like(w, np.nan)])
def test_invalid_weights_rejected(bad):
    case = CASES[0]
    a = case["inputs"]
    x = np.asarray(case["expected"]["scores"], dtype=float)
    with pytest.raises(ValueError, match="weights"):
        fit_gaussian_ee(a["y"], x, a["id"], a["z"], q=None,
                        weights=bad(np.asarray(a["w"], dtype=float)))
