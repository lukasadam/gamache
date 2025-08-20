# tests/test_utils.py
import numpy as np
from numpy.testing import assert_allclose

from gamache.tl.fit import PseudotimeGAM
from gamache.tl.utils import (
    _bh_fdr,
    _center_of_mass,
    _dense_curve,
    _neg_binom_deviance,
    _peak_time,
)


def _fit_one_gene_model(adata, gene="g0", key="nbgam_utils"):
    """Fit a one-gene model quickly for utility tests."""
    m = PseudotimeGAM(
        adata=adata,
        counts_layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        key=key,
        backend="irls",
        nonfinite="error",
    )
    idx = int(np.where(adata.var_names == gene)[0][0])
    m.fit([idx])
    return m


# --------------------
# _bh_fdr
# --------------------
def test_bh_fdr_monotonic_and_in_0_1():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = _bh_fdr(p)
    assert q.shape == p.shape
    assert np.all((q[np.isfinite(q)] >= 0) & (q[np.isfinite(q)] <= 1))
    # q-values should be non-decreasing after sorting by p
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_bh_fdr_handles_nans():
    p = np.array([0.01, np.nan, 0.5, np.nan])
    q = _bh_fdr(p)
    assert np.isnan(q[1]) and np.isnan(q[3])
    assert np.isfinite(q[0]) and np.isfinite(q[2])


def test_bh_fdr_all_nans():
    p = np.array([np.nan, np.nan])
    q = _bh_fdr(p)
    assert np.all(np.isnan(q))


# --------------------
# _center_of_mass
# --------------------
def test_center_of_mass_simple():
    t = np.array([0, 1, 2])
    w = np.array([0, 1, 1])
    com = _center_of_mass(t, w)
    assert np.isclose(com, 1.5)


def test_center_of_mass_with_nans():
    t = np.array([0, 1, np.nan, 3])
    w = np.array([1, np.nan, 1, 1])
    com = _center_of_mass(t, w)
    assert np.isfinite(com)


def test_center_of_mass_zero_weights():
    t = np.array([0, 1, 2])
    w = np.array([0, 0, 0])
    com = _center_of_mass(t, w)
    assert np.isnan(com)


# --------------------
# _peak_time
# --------------------
def test_peak_time_basic():
    t = np.linspace(0, 1, 5)
    y = np.array([0, 1, 5, 2, 1])
    pt = _peak_time(t, y)
    assert np.isclose(pt, t[2])


def test_peak_time_with_nans():
    t = np.array([0, 1, 2])
    y = np.array([np.nan, 5, np.nan])
    pt = _peak_time(t, y)
    assert pt == 1.0


def test_peak_time_empty_or_all_nan():
    assert np.isnan(_peak_time([], []))
    assert np.isnan(_peak_time([1, 2], [np.nan, np.nan]))


# --------------------
# _dense_curve
# --------------------
def test_dense_curve_matches_predict(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0", key="nbgam_utils_dense")
    t_grid, y_grid = _dense_curve(model, "g0", n_grid=50)

    assert len(t_grid) == 50
    assert np.all(np.diff(t_grid) > 0)

    # sanity: values from model.predict directly
    y_check = model.predict("g0", t_new=t_grid)
    assert_allclose(y_grid, y_check)


# --------------------
# _neg_binom_deviance
# --------------------
def test_neg_binom_deviance_zero_for_perfect_fit():
    y = np.array([1, 2, 3])
    mu = np.array([1, 2, 3])
    d = _neg_binom_deviance(y, mu, alpha=1.0)
    assert np.isclose(d, 0.0)


def test_neg_binom_deviance_positive_for_mismatch():
    y = np.array([10, 0, 5])
    mu = np.array([8, 1, 4])
    d = _neg_binom_deviance(y, mu, alpha=0.5)
    assert d >= 0.0


def test_neg_binom_deviance_clipping_edge_cases():
    y = np.array([0, 0])
    mu = np.array([0, 0])
    d = _neg_binom_deviance(y, mu, alpha=1e-12)
    assert np.isfinite(d)
    assert d >= 0.0
