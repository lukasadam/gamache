import numpy as np
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

    not_genes = np.setdiff1d(np.arange(adata_small_pt.n_vars), genes)
    assert np.isnan(coef[not_genes]).all()
    assert np.isnan(alpha[not_genes]).all()
    assert np.isnan(edf[not_genes]).all()
    assert np.isnan(cov[not_genes]).all()


# -----------------------
# Fitted values & predict
# -----------------------


def test_fitted_values_link_vs_response(adata_small_pt):
    m = PseudotimeGAM(adata_small_pt, key="nbgam_v", backend="irls")
    m.fit(genes=[0])
    eta = m.fitted_values(0, type="link")
    mu = m.fitted_values(0, type="response")
    assert_allclose(mu, np.exp(eta), rtol=1e-6, atol=1e-8)


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
