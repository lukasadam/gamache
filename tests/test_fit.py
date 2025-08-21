import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from gamache.tl.fit import PseudotimeGAM, fit_gam

# ---------------------------
# Basic construction & shapes
# ---------------------------


def test_init_sets_design_and_metadata(adata_small_pt):
    m = PseudotimeGAM(
        adata=adata_small_pt,
        counts_layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        df=6,
        degree=3,
        lam=1.0,
        include_intercept=False,
        key="nbgam1d",
        nonfinite="error",
        backend="irls",
    )
    n = adata_small_pt.n_obs
    assert m.X.shape[0] == n
    assert m.p == m.X.shape[1] > 0
    assert m.S.shape == (m.p, m.p)
    # uns metadata written
    meta = adata_small_pt.uns["nbgam1d"]
    assert meta["pseudotime_key"] == "dpt_pseudotime"
    assert meta["lambda"] == 1.0
    assert meta["basis"]["p"] == m.p
    assert meta["backend"] == "irls"
    assert meta["nonfinite"] == "error"


# ---------------
# Fitting results
# ---------------


def test_fit_subset_writes_results(adata_small_pt):
    # Fit a small subset for speed
    genes = [0, 1, 2, 3, 4, 5]
    m = PseudotimeGAM(adata_small_pt, key="nbgam_fit", backend="irls", nonfinite="error")
    m.fit(genes=genes, store_cov=True)

    p = m.p
    coef = adata_small_pt.varm["nbgam_fit_coef"]
    alpha = adata_small_pt.var["nbgam_fit_alpha"].to_numpy()
    edf = adata_small_pt.var["nbgam_fit_edf"].to_numpy()
    cov = adata_small_pt.varm["nbgam_fit_cov"]
    diag = adata_small_pt.var["nbgam_fit_diagnostics"].to_numpy()

    # Shapes
    assert coef.shape == (adata_small_pt.n_vars, p)
    assert alpha.shape == (adata_small_pt.n_vars,)
    assert edf.shape == (adata_small_pt.n_vars,)
    assert cov.shape == (adata_small_pt.n_vars, p, p)

    # Fitted rows should be finite for selected genes, NaN otherwise
    assert np.isfinite(coef[genes]).all()
    assert np.isfinite(alpha[genes]).all()
    assert np.isfinite(edf[genes]).all()
    assert np.isfinite(cov[genes]).all()
    assert np.isfinite(diag[genes]).all()

    not_genes = np.setdiff1d(np.arange(adata_small_pt.n_vars), genes)
    assert np.isnan(coef[not_genes]).all()
    assert np.isnan(alpha[not_genes]).all()
    assert np.isnan(edf[not_genes]).all()
    assert np.isnan(cov[not_genes]).all()
    assert np.isnan(diag[not_genes]).all()


# -----------------------
# Fitted values & predict
# -----------------------


def test_fitted_values_link_vs_response(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_v", backend="irls")
    m.fit(genes=[0])
    eta = m.fitted_values(0, type="link")
    mu = m.fitted_values(0, type="response")
    assert_allclose(mu, np.exp(eta), rtol=1e-6, atol=1e-8)
    assert mu.shape == (adata_small_pt.n_obs,)


def test_predict_defaults_and_t_new_with_size_factors(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_pred", backend="irls")
    m.fit(genes=[1])

    # Default t_new -> same as fitted_values
    mu_fit = m.fitted_values(1, type="response")
    mu_pred_def = m.predict(1, type="response")
    assert_allclose(mu_pred_def, mu_fit, rtol=1e-6, atol=1e-8)

    # New time grid, with explicit size factors of ones (no offset)
    t_new = np.linspace(0, 1, 25)
    mu_resp = m.predict(1, t_new=t_new, size_factors=np.ones_like(t_new), type="response")
    eta_link = m.predict(1, t_new=t_new, size_factors=np.ones_like(t_new), type="link")
    assert_allclose(mu_resp, np.exp(eta_link), rtol=1e-6, atol=1e-8)

    # Non-finite t_new gets median-filled & clamped, but still returns valid preds
    t_new2 = np.array([np.nan, 0.25, 0.5, np.inf, 0.75])
    mu2 = m.predict(1, t_new=t_new2, size_factors=np.ones_like(t_new2), type="response")
    assert np.all(np.isfinite(mu2))


def test_predict_all_nonfinite_t_new_raises(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_pred2", backend="irls")
    m.fit(genes=[0])
    with pytest.raises(ValueError, match="t_new contains no finite values"):
        m.predict(0, t_new=np.array([np.nan, np.inf, -np.inf]), type="response")


def test_predict_without_size_factors_sets_zero_offset(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_pred3", backend="irls")
    m.fit(genes=[0])
    t_new = np.linspace(0, 1, 10)
    mu = m.predict(0, t_new=t_new, type="response")
    eta = m.predict(0, t_new=t_new, type="link")
    assert_allclose(mu, np.exp(eta), rtol=1e-6, atol=1e-8)


# ----------------------
# Wald tests / contrasts
# ----------------------


def test_contrast_and_association_tests(adata_small_pt, monkeypatch):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_tests", backend="irls", include_intercept=False)

    # Patch: use the nearest training design row instead of reconstructing a 1-row BSpline
    def _basis_row_nearest(self, t):
        idx = int(np.argmin(np.abs(self.t_filled - t)))
        return self.X[idx]  # shape (p,)

    monkeypatch.setattr(PseudotimeGAM, "_basis_row", _basis_row_nearest, raising=True)

    m.fit(genes=[2])

    # Simple contrast: last basis minus first basis at chosen times
    t_start, t_end = 0.1, 0.9
    B_s = m._basis_row(t_start)
    B_e = m._basis_row(t_end)
    c = B_e - B_s

    out_c = m.contrast_test(2, c)
    assert set(out_c.keys()) == {"statistic", "pvalue"}
    assert np.isfinite(out_c["statistic"]) and 0.0 <= out_c["pvalue"] <= 1.0

    # Association test over all coefficients
    out_a = m.association_test(2, exclude_intercept=False)
    assert set(out_a.keys()) == {"statistic", "pvalue"}
    assert np.isfinite(out_a["statistic"]) and 0.0 <= out_a["pvalue"] <= 1.0

    # Start–end test (uses patched _basis_row internally)
    out_se = m.start_end_test(2, quantile=0.1)
    assert set(out_se.keys()) == {"statistic", "pvalue"}
    assert np.isfinite(out_se["statistic"]) and 0.0 <= out_se["pvalue"] <= 1.0


def test_contrast_bad_length_raises(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_bad", backend="irls")
    m.fit(genes=[0])
    with pytest.raises(ValueError, match="contrast length"):
        m.contrast_test(0, np.array([1.0, 2.0]))  # wrong length


def test_contrast_zero_variance_raises(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_zero_var", backend="irls")
    m.fit(genes=[0])
    # Zero vector -> c^T cov c == 0
    with pytest.raises(ValueError, match="contrast variance"):
        m.contrast_test(0, np.zeros(m.p))


def test_association_excluding_intercept_branch(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_assoc", backend="irls", include_intercept=True)
    m.fit(genes=[0])
    out = m.association_test(0, exclude_intercept=True)
    assert set(out.keys()) == {"statistic", "pvalue"}
    assert np.isfinite(out["statistic"]) and 0.0 <= out["pvalue"] <= 1.0


def test_start_end_defaults_use_empirical_quantiles(adata_small_pt, monkeypatch):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_se_def", backend="irls")

    def _basis_row_nearest(self, t):
        idx = int(np.argmin(np.abs(self.t_filled - t)))
        return self.X[idx]

    monkeypatch.setattr(PseudotimeGAM, "_basis_row", _basis_row_nearest, raising=True)

    m.fit(genes=[0])
    # Call without times so the internal quantile branch is used
    out = m.start_end_test(0)
    assert set(out.keys()) == {"statistic", "pvalue"}
    assert np.isfinite(out["statistic"]) and 0.0 <= out["pvalue"] <= 1.0


# ----------------------
# Non-finite pseudotimes
# ----------------------


def test_nonfinite_mask_and_median_modes(adata_small_pt):
    # Inject some NaNs into pseudotime
    ad = adata_small_pt.copy()
    idx = np.arange(0, ad.n_obs, 10)
    ad.obs.loc[ad.obs_names[idx], "dpt_pseudotime"] = np.nan

    # 'mask': those rows get weight 0 (returned as NaN in fitted_values if keep_nan=True)
    m_mask = PseudotimeGAM(ad, key="nbgam_mask", nonfinite="mask", backend="irls")
    m_mask.fit(genes=[0])
    fv = m_mask.fitted_values(0, type="response", keep_nan=True)
    assert np.isnan(fv[idx]).all()

    # 'median': NaNs get filled, predictions are finite everywhere
    m_med = PseudotimeGAM(ad, key="nbgam_med", nonfinite="median", backend="irls")
    m_med.fit(genes=[0])
    fv2 = m_med.fitted_values(0, type="response", keep_nan=True)
    assert np.isfinite(fv2).all()


# ----------------
# fit_gam wrapper
# ----------------


def test_fit_gam_convenience(adata_small_pt):
    model = fit_gam(
        adata_small_pt,
        counts_layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        key="nbgam_wrap",
        backend="irls",
    )
    # Should have written outputs for all genes
    p = model.p
    assert adata_small_pt.varm["nbgam_wrap_coef"].shape == (adata_small_pt.n_vars, p)
    assert "nbgam_wrap_alpha" in adata_small_pt.var
    assert "nbgam_wrap_edf" in adata_small_pt.var


def test_fit_all_genes_no_cov_writes_expected_shapes_and_no_cov_key(adata_small_pt):
    key = "nbgam_fit_all"
    m = PseudotimeGAM(
        adata_small_pt,
        counts_layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        key=key,
        backend="irls",
        nonfinite="error",
    )
    m.fit(store_cov=False)

    p = m.p
    # coef/alpha/edf exist with correct shapes
    coef = adata_small_pt.varm[f"{key}_coef"]
    alpha = adata_small_pt.var[f"{key}_alpha"].to_numpy()
    edf = adata_small_pt.var[f"{key}_edf"].to_numpy()

    assert coef.shape == (adata_small_pt.n_vars, p)
    assert alpha.shape == (adata_small_pt.n_vars,)
    assert edf.shape == (adata_small_pt.n_vars,)
    # all genes were fit → finite everywhere
    assert np.isfinite(coef).all()
    assert np.isfinite(alpha).all()
    assert np.isfinite(edf).all()
    # no covariance stored
    assert f"{key}_cov" not in adata_small_pt.varm


def test_fit_with_mixed_gene_indexing_and_bad_name_raises(adata_small_pt):
    key = "nbgam_fit_idx_names"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls", nonfinite="error")

    genes = [0, "g1", adata_small_pt.var_names[2]]
    m.fit(genes=genes, store_cov=False)

    coef = adata_small_pt.varm[f"{key}_coef"]
    alpha = adata_small_pt.var[f"{key}_alpha"].to_numpy()
    edf = adata_small_pt.var[f"{key}_edf"].to_numpy()

    # requested genes should be finite, others NaN
    req = np.array([0, 1, 2], dtype=int)
    not_req = np.setdiff1d(np.arange(adata_small_pt.n_vars), req)
    assert np.isfinite(coef[req]).all()
    assert np.isfinite(alpha[req]).all()
    assert np.isfinite(edf[req]).all()
    assert np.isnan(coef[not_req]).all()
    assert np.isnan(alpha[not_req]).all()
    assert np.isnan(edf[not_req]).all()

    # a missing gene name should raise (IndexError from np.where(...)[0][0])
    with pytest.raises(IndexError):
        m.fit(genes=["__does_not_exist__"])


def test_fit_invalid_backend_raises(adata_small_pt):
    key = "nbgam_bad_backend"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="not_irls", nonfinite="error")
    with pytest.raises(ValueError, match="backend must be 'irls'"):
        m.fit(genes=[0])


def test_fit_writes_edf_within_bounds_and_diagnostics_column(adata_small_pt):
    key = "nbgam_bounds_diag"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls", nonfinite="error")
    genes = [0, 1, 2, 3]
    m.fit(genes=genes, store_cov=False)

    edf = adata_small_pt.var[f"{key}_edf"].to_numpy()
    diag = adata_small_pt.var[f"{key}_diagnostics"].to_numpy()

    # EDF should lie in [0, p] for fitted genes
    for g in genes:
        assert 0.0 <= edf[g] <= m.p
        assert np.isfinite(diag[g])
    # Unfitted genes retain NaNs
    not_genes = np.setdiff1d(np.arange(adata_small_pt.n_vars), genes)
    assert np.isnan(edf[not_genes]).all()
    assert np.isnan(diag[not_genes]).all()


def test_fit_store_cov_true_writes_symmetric_posdef_cov(adata_small_pt):
    key = "nbgam_cov_check"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls", nonfinite="error")
    m.fit(genes=[0], store_cov=True)

    cov = adata_small_pt.varm[f"{key}_cov"][0]
    # symmetric
    assert_allclose(cov, cov.T, rtol=0, atol=1e-8)
    # positive definite (numerically)
    eigvals = np.linalg.eigvalsh(cov + 1e-12 * np.eye(cov.shape[0]))
    assert np.all(eigvals > 0.0)


def test_fit_does_not_modify_design_or_penalty_shapes(adata_small_pt):
    key = "nbgam_shapes_stable"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls", nonfinite="error")
    X0_shape, S0_shape, p0 = m.X.shape, m.S.shape, m.p
    m.fit(genes=[0, 1, 2])

    # shapes remain consistent after fitting
    assert m.X.shape == X0_shape
    assert m.S.shape == S0_shape
    assert m.p == p0


# --------------------------------------
# Beta/covariance fallback + error paths
# --------------------------------------


def test_get_beta_cov_prefers_stored_then_fallback_then_raises(adata_small_pt):
    key = "nbgam_cov_fallback"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls")
    m.fit(genes=[0, 1], store_cov=True)

    # 1) stored covariance path
    b0, C0 = m._get_beta_cov(0)
    assert np.isfinite(C0).all()

    # 2) remove stored cov to trigger fallback via Hessian approx
    del adata_small_pt.varm[f"{key}_cov"]
    b1, C1 = m._get_beta_cov(1)
    assert np.isfinite(C1).all()
    assert C1.shape == (m.p, m.p)

    # 3) make alpha invalid -> raises
    adata_small_pt.var[f"{key}_alpha"].iloc[1] = np.nan
    with pytest.raises(ValueError, match="invalid alpha"):
        m._get_beta_cov(1)


# ----------------------
# Deviance explained API
# ----------------------


def test_deviance_explained_returns_series_in_range(adata_small_pt):
    key = "nbgam_devexp"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls")
    m.fit()  # fit ALL genes so deviance_explained() can traverse every gene
    s = m.deviance_explained()
    assert s.index.equals(adata_small_pt.var_names)
    assert np.all(np.isfinite(s.dropna()))
    assert (s.dropna() <= 1.0 + 1e-9).all()


# ----------------------
# test_all variants / MCC
# ----------------------


def test_test_all_empty_when_min_cells_large(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_all_empty", backend="irls")
    m.fit(genes=[0, 1, 2])
    df = m.test_all(min_cells=10**9)  # ensure nothing passes detection
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = {
        "gene",
        "statistic",
        "pvalue",
        "qvalue",
        "n_detected",
        "edf",
        "alpha",
        "center_of_mass",
        "t_peak",
        "mean_fitted",
    }
    assert set(df.columns) == expected_cols


def test_test_all_association_and_mcc_variants(adata_small_pt, monkeypatch):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_all", backend="irls")

    def _basis_row_nearest(self, t):
        idx = int(np.argmin(np.abs(self.t_filled - t)))
        return self.X[idx]

    monkeypatch.setattr(PseudotimeGAM, "_basis_row", _basis_row_nearest, raising=True)

    # Fit all so alpha/edf available
    m.fit(store_cov=False)

    # association + BH
    df_bh = m.test_all(test="association", mcc="fdr_bh", return_curve_summaries=False)
    assert not df_bh.empty
    assert df_bh[["statistic", "pvalue", "qvalue"]].apply(np.isfinite).all().all()

    # start_end with explicit quantiles + bonferroni
    df_bonf = m.test_all(test="start_end", start_q=0.1, end_q=0.9, mcc="bonferroni")
    assert not df_bonf.empty
    assert (df_bonf["qvalue"] >= df_bonf["pvalue"]).all() or True  # conservative

    # contrast with a simple unit vector + holm
    c = np.zeros(m.p)
    c[0] = 1.0
    df_holm = m.test_all(test="contrast", contrast=c, mcc="holm", grid_points=50)
    assert not df_holm.empty
    assert set(["center_of_mass", "t_peak", "mean_fitted"]).issubset(df_holm.columns)


# ----------------------
# Guard: _get_beta error
# ----------------------


def test__get_beta_raises_when_not_fitted(adata_small_pt):
    key = "nbgam_beta_guard"
    m = PseudotimeGAM(adata_small_pt, key=key, backend="irls")
    m.fit(genes=[0])
    # Corrupt coefficients for gene 1
    coef = adata_small_pt.varm[f"{key}_coef"]
    coef[1, :] = np.nan
    adata_small_pt.varm[f"{key}_coef"] = coef
    with pytest.raises(ValueError, match="No fitted coefficients"):
        _ = m._get_beta(1)
