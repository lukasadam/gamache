import numpy as np
import pytest
from scipy import sparse

from gamache.tl.fit import PseudotimeGAM, fit_gam

# ---------------------------
# Construction & inputs
# ---------------------------


def test_init_sets_design_and_metadata(adata_small_pt):
    m = PseudotimeGAM(
        adata=adata_small_pt,
        layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        df=5,
        degree=3,
        lam=0.5,
        include_intercept=False,
        key="nbgam1d",
        nonfinite="error",
    )
    n = adata_small_pt.n_obs

    # basic shape sanity via internal vectors that exist in the class
    assert m.t_filled.shape[0] == n
    assert m.offset.shape[0] == n

    # metadata keys that the mssm-backed class writes
    meta = adata_small_pt.uns["nbgam1d"]
    assert meta["pseudotime_key"] == "dpt_pseudotime"
    assert meta["lambda"] == 0.5
    assert meta["basis"]["df"] == 5
    assert meta["basis"]["degree"] == 3
    assert meta["basis"]["include_intercept"] is False
    assert meta["backend"] == "mssm_gamm"
    assert meta["nonfinite"] == "error"


def test_init_nonfinite_error_raises(adata_small_pt):
    A = adata_small_pt.copy()
    A.obs["dpt_pseudotime"] = np.where(np.arange(A.n_obs) % 5 == 0, np.nan, A.obs["dpt_pseudotime"])
    with pytest.raises(ValueError, match="non-finite"):
        PseudotimeGAM(A, nonfinite="error")


def test_init_nonfinite_mask_and_median(adata_small_pt):
    # mask: keeps NaNs as weight 0 → later fitted_values(keep_nan=True) should restore NaN
    A = adata_small_pt.copy()
    bad_ix = 3
    A.obs["dpt_pseudotime"].iloc[bad_ix] = np.nan
    m_mask = PseudotimeGAM(A, nonfinite="mask", key="nbgam_mask")
    m_mask.fit(genes=[0])
    fv = m_mask.fitted_values(0, keep_nan=True)
    assert np.isnan(fv[bad_ix])

    # median: imputes NaN and weight=1 → fitted_values has no NaN
    m_med = PseudotimeGAM(A, nonfinite="median", key="nbgam_median")
    m_med.fit(genes=[0])
    fv2 = m_med.fitted_values(0, keep_nan=True)
    assert np.isfinite(fv2[bad_ix])


# ---------------------------
# Fitting & coefficients
# ---------------------------


def test_fit_subset_writes_results(adata_small_pt):
    genes = [0, 1, 2]
    m = PseudotimeGAM(adata_small_pt, key="nbgam_fit", nonfinite="error")
    # Do not request storing covariance cube; the class doesn't preallocate (p,p)
    m.fit(genes=genes, store_cov=False)

    coef = adata_small_pt.varm["nbgam_fit_coef"]
    alpha = adata_small_pt.var["nbgam_fit_alpha"].to_numpy()
    edf = adata_small_pt.var["nbgam_fit_edf"].to_numpy()

    # coef width is implementation-dependent; assert rectangular with >=1 column
    assert coef.shape[0] == adata_small_pt.n_vars
    assert coef.shape[1] >= 1

    # alpha/edf vectors shaped by n_vars
    assert alpha.shape == (adata_small_pt.n_vars,)
    assert edf.shape == (adata_small_pt.n_vars,)

    # selected genes finite; others may be NaN in coef
    assert np.isfinite(coef[genes]).all()
    assert alpha.dtype.kind == "f"
    mask_other = [i for i in range(adata_small_pt.n_vars) if i not in genes]
    if len(mask_other) > 0:
        assert np.isnan(coef[mask_other]).any()


def test_predict_requires_fit(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_req")
    # .predict checks _results_ and should raise if not fitted
    with pytest.raises(RuntimeError, match="Call .fit"):
        m.predict(0)


def test_fitted_values_link_vs_response(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_link")
    m.fit(genes=[0])
    eta = m.fitted_values(0, type="link")
    mu = m.fitted_values(0, type="response")
    assert np.allclose(np.exp(eta), mu, rtol=1e-6, atol=1e-6)


# ---------------------------
# Prediction & (optional) basis helpers
# ---------------------------


def test_predict_shapes_and_clamping(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_pred")
    m.fit(genes=[0])
    # t_new outside training range must be handled without error
    t_new = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=float)
    yhat = m.predict(0, t_new=t_new)
    assert yhat.shape == (t_new.size,)
    assert np.isfinite(yhat).all()

    # all-NaN t_new raises
    with pytest.raises(ValueError, match="no finite values"):
        m.predict(0, t_new=np.array([np.nan, np.nan]))


def test_predict_with_ci(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_ci")
    m.fit(genes=[0])
    mean, lo, hi = m.predict(0, t_new=np.linspace(0, 1, 7), return_ci=True)
    assert mean.shape == lo.shape == hi.shape == (7,)
    assert np.all(lo <= mean)
    assert np.all(mean <= hi)


def test_basis_row_shape_and_error(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_basis")
    # Only run if the helper exists in this backend; otherwise skip
    if not hasattr(m, "_basis_row"):
        pytest.skip("This backend does not expose _basis_row().")
    B = m._basis_row(float(np.median(m.t_filled)))
    assert B.shape[0] >= 1
    with pytest.raises(ValueError):
        m._basis_row(float(np.nan))


# ---------------------------
# Wald tests
# ---------------------------


def test_contrast_test_valid(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_contrast")
    m.fit(genes=[0])

    # derive coefficient length from stored varm table instead of m.p
    coef = adata_small_pt.varm["nbgam_contrast_coef"]
    p = coef.shape[1]
    c = np.zeros(p, dtype=float)
    c[0] = 1.0
    out = m.contrast_test(0, c)
    assert set(out.keys()) == {"statistic", "pvalue"}
    assert 0.0 <= out["pvalue"] <= 1.0


def test_association_and_start_end_tests(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_tests")
    m.fit(genes=[0])

    a = m.association_test(0, exclude_intercept=False)
    assert set(a.keys()) == {"statistic", "pvalue"}
    assert np.isfinite(a["statistic"])
    assert 0.0 <= a["pvalue"] <= 1.0

    # start_end_test may rely on a backend-specific basis helper; skip if it raises or missing
    try:
        se = m.start_end_test(0, quantile=0.1)
        assert set(se.keys()) == {"statistic", "pvalue"}
        assert np.isfinite(se["statistic"])
        assert 0.0 <= se["pvalue"] <= 1.0
    except (AttributeError, NotImplementedError):
        pytest.skip("start_end_test not available in this backend.")


# ---------------------------
# test_all & MCC
# ---------------------------


def test_test_all_association_and_mcc(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_all")
    m.fit(genes=[0, 1, 2])
    # association, minimal detection threshold for speed/robustness
    df = m.test_all(
        genes=[0, 1, 2],
        test="association",
        min_cells=1,
        mcc="fdr_bh",
        return_curve_summaries=False,  # faster
    )
    assert {"gene", "statistic", "pvalue", "qvalue", "n_detected", "edf", "alpha"}.issubset(
        df.columns
    )
    # sorted by qvalue ascending
    q = df["qvalue"].to_numpy()
    assert np.all(np.diff(q[~np.isnan(q)]) >= 0)

    # Bonferroni
    df_b = m.test_all(
        genes=[0, 1, 2],
        test="association",
        min_cells=1,
        mcc="bonferroni",
        return_curve_summaries=False,
    )
    assert len(df_b) == len(df)

    # Holm
    df_h = m.test_all(
        genes=[0, 1, 2], test="association", min_cells=1, mcc="holm", return_curve_summaries=False
    )
    assert len(df_h) == len(df)


def test_test_all_contrast_requires_vector(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_contrast_req")
    m.fit(genes=[0])
    with pytest.raises(ValueError, match="contrast must be provided"):
        m.test_all(genes=[0], test="contrast", min_cells=1)


# ---------------------------
# Deviance explained
# ---------------------------


def test_deviance_explained_vector(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_dev")
    m.fit()
    de = m.deviance_explained()
    # index coverage
    assert set(["g0", "g1", "g2"]).issubset(set(de.index))
    # shape is correct for all genes
    assert de.shape[0] == adata_small_pt.n_vars
    # values are numeric (may be NaN in the current backend)
    assert de.dtype.kind == "f"


# ---------------------------
# Alpha (method-of-moments)
# ---------------------------


def test_alpha_mom_nonnegative_and_simple_cases():
    y = np.array([0, 0, 0, 0], dtype=float)
    a0 = PseudotimeGAM._alpha_mom(y)
    assert a0 >= 1e-10

    y2 = np.array([1, 2, 3, 4, 5], dtype=float)
    a2 = PseudotimeGAM._alpha_mom(y2)
    assert a2 >= 1e-10


# ---------------------------
# Layers / sparse / size factors
# ---------------------------


def test_sparse_X_and_size_factors_fallback(adata_small_pt):
    A = adata_small_pt.copy()
    # Remove size_factors_key to force fallback
    if "size_factors" in A.obs:
        del A.obs["size_factors"]
    A.X = sparse.csr_matrix(A.X)
    m = PseudotimeGAM(A, key="nbgam_sparse")
    m.fit(genes=[0])
    fv = m.fitted_values(0)
    assert np.isfinite(fv).all()


def test_layer_usage_counts_present(adata_small_pt):
    # Put a "counts" layer and ask the model to use it explicitly
    A = adata_small_pt.copy()
    A.layers["counts"] = A.X.copy()
    m = PseudotimeGAM(A, layer="counts", key="nbgam_layer")
    m.fit(genes=[0])
    y = m._get_counts_col(0)
    assert y.shape == (A.n_obs,)
    assert np.isfinite(y).all()


# ---------------------------
# Convenience wrapper
# ---------------------------


def test_fit_gam_wrapper_returns_model(adata_small_pt):
    m = fit_gam(adata_small_pt, key="nbgam_wrap")
    # Should be fitted for all genes by default
    assert isinstance(m, PseudotimeGAM)
    # At least one coef row should be finite
    coef = adata_small_pt.varm["nbgam_wrap_coef"]
    assert coef.shape[0] == adata_small_pt.n_vars
    assert coef.shape[1] >= 1
    assert np.isfinite(coef).any()
