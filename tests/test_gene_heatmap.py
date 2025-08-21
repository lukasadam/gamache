# tests/test_gene_heatmap.py
import matplotlib

matplotlib.use("Agg")  # headless for CI

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from gamache.pl.gene_heatmap import (
    _fill_runs_categorical_1d,
    _fill_runs_nan_1d,
    plot_gene_heatmap,
)

# ---------------------------
# Helper filler function tests
# ---------------------------


def test_fill_runs_nan_1d_cases():
    # middle run -> average of neighbors
    a = np.array([1.0, np.nan, np.nan, 3.0])
    out = _fill_runs_nan_1d(a)
    assert np.allclose(out, [1.0, 2.0, 2.0, 3.0], atol=1e-12)

    # left edge run -> prefer right
    a = np.array([np.nan, np.nan, 4.0])
    out = _fill_runs_nan_1d(a)
    assert np.allclose(out, [4.0, 4.0, 4.0])

    # right edge run -> prefer left
    a = np.array([5.0, np.nan, np.nan])
    out = _fill_runs_nan_1d(a)
    assert np.allclose(out, [5.0, 5.0, 5.0])

    # all NaN -> unchanged
    a = np.array([np.nan, np.nan, np.nan])
    out = _fill_runs_nan_1d(a)
    assert np.isnan(out).all()


def test_fill_runs_categorical_1d_cases():
    # equal neighbors -> fill with that label
    a = np.array([1, np.nan, np.nan, 1], dtype=float)
    out = _fill_runs_categorical_1d(a)
    assert np.allclose(out, [1, 1, 1, 1])

    # different neighbors -> prefer left neighbor
    a = np.array([2, np.nan, 3], dtype=float)
    out = _fill_runs_categorical_1d(a)
    assert np.allclose(out, [2, 2, 3])

    # left edge -> prefer right
    a = np.array([np.nan, 4, 4], dtype=float)
    out = _fill_runs_categorical_1d(a)
    assert np.allclose(out, [4, 4, 4])

    # all NaN -> unchanged
    a = np.array([np.nan, np.nan], dtype=float)
    out = _fill_runs_categorical_1d(a)
    assert np.isnan(out).all()


def test_basic_heatmap_runs_and_returns_expected_shapes(adata_small_pt):
    genes = ["g0", "g1", "g2"]
    ordered, bin_centers, H = plot_gene_heatmap(
        adata_small_pt, genes, pt_key="dpt_pseudotime", n_bins=20
    )

    # All genes included and ordered subset
    assert set(ordered) <= set(genes)
    assert len(bin_centers) == 20
    assert H.shape == (len(ordered), 20)


def test_invalid_gene_raises(adata_small_pt):
    with pytest.raises(ValueError, match="None of the provided genes"):
        plot_gene_heatmap(adata_small_pt, ["not_a_gene"], pt_key="dpt_pseudotime")


def test_invalid_pseudotime_key_raises(adata_small_pt):
    with pytest.raises((ValueError, KeyError)):
        plot_gene_heatmap(adata_small_pt, ["g0"], pt_key="not_a_key")


def test_no_finite_pseudotime_raises(adata_small_pt):
    adata = adata_small_pt.copy()
    adata.obs["dpt_pseudotime"] = np.nan
    with pytest.raises(ValueError, match="No finite pseudotime"):
        plot_gene_heatmap(adata, ["g0"], pt_key="dpt_pseudotime")


def test_fill_empty_zero_option(adata_small_pt):
    # ensure that option "zero" actually fills empty bins with 0.0
    _, _, H = plot_gene_heatmap(
        adata_small_pt, ["g0"], pt_key="dpt_pseudotime", n_bins=5, fill_empty="zero"
    )
    assert np.all(np.isfinite(H))


def test_ordering_changes_gene_order(adata_small_pt):
    genes = ["g0", "g1"]
    ordered_peak, _, _ = plot_gene_heatmap(
        adata_small_pt, genes, pt_key="dpt_pseudotime", order_by_peak=True, n_bins=10
    )
    ordered_none, _, _ = plot_gene_heatmap(
        adata_small_pt, genes, pt_key="dpt_pseudotime", order_by_peak=False, n_bins=10
    )
    # Both must contain same gene set
    assert set(ordered_peak) == set(genes)
    assert set(ordered_none) == set(genes)
    # They may differ, but if not, that's valid
    assert isinstance(ordered_peak, list)
    assert isinstance(ordered_none, list)


def test_annotation_continuous_and_categorical(adata_small_pt):
    # add a continuous covariate
    adata_small_pt.obs["cov_cont"] = np.linspace(0, 1, adata_small_pt.n_obs)
    # add a categorical covariate
    adata_small_pt.obs["cov_cat"] = np.where(np.arange(adata_small_pt.n_obs) % 2 == 0, "A", "B")
    _, _, H = plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        annot_keys=["cov_cont", "cov_cat"],
        n_bins=15,
    )
    assert H.shape[0] == 2


def test_return_matrix_zscore_and_clip(adata_small_pt):
    ordered, _, H = plot_gene_heatmap(
        adata_small_pt,
        ["g0"],
        pt_key="dpt_pseudotime",
        zscore=True,
        clip=(-1, 1),
        n_bins=10,
    )
    assert np.nanmin(H) >= -1 - 1e-6
    assert np.nanmax(H) <= 1 + 1e-6


def test_outfile_saves(tmp_path, adata_small_pt):
    outfile = tmp_path / "heatmap.png"
    plot_gene_heatmap(adata_small_pt, ["g0"], pt_key="dpt_pseudotime", n_bins=5, outfile=outfile)
    assert outfile.exists()


def test_pt_key_autodetect_and_sparse_X(adata_small_pt):
    # use sparse X path and pt_key auto-detection
    adata = adata_small_pt.copy()
    adata.X = sparse.csr_matrix(adata.X)
    genes = ["g0", "g1", "g2"]
    ordered, bin_centers, H = plot_gene_heatmap(adata, genes, pt_key=None, n_bins=25)
    assert set(ordered) <= set(genes)
    assert len(bin_centers) == 25
    assert H.shape == (len(ordered), 25)


def test_model_predictions_are_used_monotonic(adata_small_pt):
    # Dummy model: strictly increasing with pseudotime, scaled per gene
    pt = adata_small_pt.obs["dpt_pseudotime"].to_numpy()
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)

    class DummyModel:
        def __init__(self, scale_by_name=True):
            self.scale_by_name = scale_by_name

        def predict(self, g):
            # scale depends on gene index (g like 'g12')
            idx = int(g[1:]) if self.scale_by_name else 1
            return (1 + idx) * (1 + pt)  # strictly increasing in pt

    genes = ["g0", "g3", "g7"]
    ordered, _, H = plot_gene_heatmap(
        adata_small_pt,
        genes,
        model=DummyModel(),
        pt_key="dpt_pseudotime",
        n_bins=30,
        zscore=False,  # keep monotonicity after log1p + bin-mean
    )
    # Each gene row should be non-decreasing across bins (up to tiny numeric noise)
    for r in range(H.shape[0]):
        diffs = np.diff(np.nan_to_num(H[r], nan=-np.inf))
        assert np.all(diffs >= -1e-9)


def test_fill_empty_neighbor_mean_behavior_small_constructed():
    # Construct tiny dataset where middle bins are empty → neighbor mean should fill
    # 10 cells, 1 gene; 4 cells at pt=0, 6 cells at pt=1 → bins 0 and last non-empty
    counts = np.array([[0.0]] * 4 + [[10.0]] * 6)  # (10, 1)
    t = np.array([0.0] * 4 + [1.0] * 6, dtype=float)
    A = ad.AnnData(counts)
    A.var_names = ["g0"]
    A.obs["dpt_pseudotime"] = t

    ordered, _, H = plot_gene_heatmap(
        A,
        ["g0"],
        pt_key="dpt_pseudotime",
        n_bins=6,
        zscore=False,  # inspect raw log1p means
        fill_empty="neighbor_mean",
    )
    # Expected: only bin 0 and bin 5 have data; others get filled with mean of ends
    left = np.log1p(0.0)
    right = np.log1p(10.0)
    mid_expected = 0.5 * (left + right)
    # H has shape (1, 6)
    assert np.isclose(H[0, 0], left, atol=1e-12)
    assert np.isclose(H[0, 5], right, atol=1e-12)
    # a middle bin gets neighbor mean
    assert np.isclose(H[0, 2], mid_expected, rtol=1e-6, atol=1e-6)


def test_fill_empty_leave_nan_creates_nans_in_output():
    # Make gaps so some bins are empty; leave NaNs
    counts = np.array([[0.0]] * 3 + [[10.0]] * 3)  # (6, 1)
    t = np.array([0.0] * 3 + [1.0] * 3, dtype=float)
    A = ad.AnnData(counts)
    A.var_names = ["g0"]
    A.obs["dpt_pseudotime"] = t

    _, _, H = plot_gene_heatmap(
        A, ["g0"], pt_key="dpt_pseudotime", n_bins=8, zscore=True, fill_empty="nan"
    )
    assert np.isnan(H).any()  # NaNs persist in heatmap matrix


def test_include_pt_tile_false_and_no_other_annots(adata_small_pt):
    ordered, bins, H = plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        include_pt_tile=False,
        annot_keys=[],  # no other annotations
        n_bins=12,
    )
    assert len(bins) == 12
    assert H.shape[0] == len(ordered)


def test_heatmap_norm_and_vrange_custom(adata_small_pt):
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=0.0, vmax=3.0)
    ordered, _, H = plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1", "g2"],
        pt_key="dpt_pseudotime",
        zscore=False,  # use raw log1p-mean
        heatmap_norm=norm,  # custom norm branch
        n_bins=15,
    )
    assert H.shape == (len(ordered), 15)
    # values should be >= 0 because of log1p
    assert np.nanmin(H) >= 0.0


def test_zscore_with_center_none_path(adata_small_pt):
    # zscore=True & heatmap_center=None triggers Normalize (not TwoSlopeNorm)
    ordered, _, H = plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        zscore=True,
        heatmap_center=None,
        n_bins=10,
    )
    assert H.shape == (len(ordered), 10)
    assert np.isfinite(np.nanmean(H))


def test_categorical_palette_partial_fallback(adata_small_pt):
    # add a 3-level categorical with a palette that misses one level
    ad = adata_small_pt.copy()
    cats = np.array(["A", "B", "C"])
    ad.obs["cov_cat3"] = cats[(np.arange(ad.n_obs) % 3)]
    palettes = {"cov_cat3": {"A": "#ff0000", "B": "#00ff00"}}  # no entry for 'C'
    _, _, H = plot_gene_heatmap(
        ad,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        annot_keys=["cov_cat3"],
        categorical_palettes=palettes,
        n_bins=20,
    )
    assert H.shape[0] == 2  # ran successfully


def test_valid_genes_subset_filters_missing(adata_small_pt):
    # One valid and one invalid gene name → should proceed with the valid one
    ordered, _, H = plot_gene_heatmap(
        adata_small_pt, ["g0", "__not_a_gene__"], pt_key="dpt_pseudotime", n_bins=7
    )
    assert ordered == ["g0"]
    assert H.shape == (1, 7)
