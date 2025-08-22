import numpy as np

from gamache.tl.fit import PseudotimeGAM

# ---------------------------
# Basic construction & shapes
# ---------------------------


def test_init_sets_design_and_metadata(adata_small_pt):
    m = PseudotimeGAM(
        adata=adata_small_pt,
        layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        df=6,
        degree=3,
        lam=1.0,
        include_intercept=False,
        key="nbgam1d",
        nonfinite="error",
    )
    n = adata_small_pt.n_obs
    assert m.X.shape[0] == n
    assert m.p == m.X.shape[1] > 0
    # uns metadata written
    meta = adata_small_pt.uns["nbgam1d"]
    assert meta["pseudotime_key"] == "dpt_pseudotime"
    assert meta["lambda"] == 1.0
    assert meta["basis"]["p"] == m.p
    assert meta["backend"] == "statsmodels_glmgam"
    assert meta["nonfinite"] == "error"


# ---------------
# Fitting results
# ---------------


def test_fit_subset_writes_results(adata_small_pt):
    # Fit a small subset for speed
    genes = [0, 1, 2, 3, 4, 5]
    m = PseudotimeGAM(adata_small_pt, key="nbgam_fit", nonfinite="error")
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
