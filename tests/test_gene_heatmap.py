# tests/test_gene_heatmap.py
import matplotlib

matplotlib.use("Agg")  # headless for CI

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import sparse

from gamache.pl.gene_heatmap import _impute_1d_linear, plot_gene_heatmap

# ---------------------------
# Helpers
# ---------------------------


def _close_all():
    plt.close("all")


def _fig_image_axes():
    """
    Return (fig, image_axes) where image_axes are those axes that have at least one
    imshow image (annotation tiles and main heatmap), excluding colorbar axes.
    """
    fig = plt.gcf()
    image_axes = [ax for ax in fig.axes if len(ax.images) > 0]
    assert len(image_axes) >= 1, "No image axes found on the figure."
    return fig, image_axes


def _get_main_ax_and_im():
    """
    Identify the main heatmap axis.
    1) Prefer the image axis whose first image has the largest number of rows (>1).
    2) If all images are 1×N (e.g., single-gene heatmap), fallback to the LAST image axis.
    Returns (ax_main, im_main).
    """
    _, image_axes = _fig_image_axes()
    candidates = []
    for ax in image_axes:
        im = ax.images[0]
        arr = np.array(im.get_array())
        candidates.append((arr.shape[0], ax, im))

    rows, ax_main, im_main = max(candidates, key=lambda x: x[0])
    if rows <= 1:
        # single-gene case: pick the last image axis (GridSpec draws main last)
        ax_main = image_axes[-1]
        im_main = ax_main.images[0]
        arr = np.array(im_main.get_array())
        assert arr.shape[0] == 1, "Fallback expected a 1×N image."
    return ax_main, im_main


# ---------------------------
# _impute_1d_linear tests
# ---------------------------


def test_impute_1d_linear_middle_run():
    _close_all()
    a = np.array([1.0, np.nan, np.nan, 3.0])
    out = _impute_1d_linear(a)
    assert np.allclose(out, [1.0, 1.6666667, 2.3333333, 3.0], atol=1e-6)


def test_impute_1d_linear_left_edge():
    _close_all()
    a = np.array([np.nan, np.nan, 4.0])
    out = _impute_1d_linear(a)
    assert np.allclose(out, [4.0, 4.0, 4.0], atol=1e-12)


def test_impute_1d_linear_right_edge():
    _close_all()
    a = np.array([5.0, np.nan, np.nan])
    out = _impute_1d_linear(a)
    assert np.allclose(out, [5.0, 5.0, 5.0], atol=1e-12)


def test_impute_1d_linear_all_nan_returns_zeros():
    _close_all()
    a = np.array([np.nan, np.nan, np.nan])
    out = _impute_1d_linear(a)
    assert np.allclose(out, [0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------
# Heatmap tests
# ---------------------------


def test_basic_heatmap_runs_and_shapes(adata_small_pt):
    _close_all()
    genes = ["g0", "g1", "g2"]
    plot_gene_heatmap(adata_small_pt, genes, pt_key="dpt_pseudotime", n_bins=20)
    _, im = _get_main_ax_and_im()
    H = np.array(im.get_array())
    assert H.shape == (len(genes), 20)


def test_invalid_gene_raises(adata_small_pt):
    _close_all()
    with pytest.raises(ValueError, match="None of the provided genes"):
        plot_gene_heatmap(adata_small_pt, ["not_a_gene"], pt_key="dpt_pseudotime")


def test_invalid_pseudotime_key_raises(adata_small_pt):
    _close_all()
    with pytest.raises((ValueError, KeyError)):
        plot_gene_heatmap(adata_small_pt, ["g0"], pt_key="not_a_key")


def test_no_finite_pseudotime_raises(adata_small_pt):
    _close_all()
    adata = adata_small_pt.copy()
    adata.obs["dpt_pseudotime"] = np.nan
    with pytest.raises(ValueError, match="No finite pseudotime"):
        plot_gene_heatmap(adata, ["g0"], pt_key="dpt_pseudotime")


def test_pt_key_autodetect_and_sparse_X(adata_small_pt):
    _close_all()
    adata = adata_small_pt.copy()
    adata.X = sparse.csr_matrix(adata.X)
    genes = ["g0", "g1", "g2"]
    plot_gene_heatmap(adata, genes, pt_key=None, n_bins=25)
    _, im = _get_main_ax_and_im()
    H = np.array(im.get_array())
    assert H.shape == (len(genes), 25)


def test_annotation_continuous_and_categorical(adata_small_pt):
    _close_all()
    adata = adata_small_pt.copy()
    # Use numeric categoricals to avoid SciPy mode error on strings
    adata.obs["cov_cont"] = np.linspace(0, 1, adata.n_obs)
    adata.obs["cov_cat"] = (np.arange(adata.n_obs) % 2).astype(int)  # 0/1 numeric

    n_bins = 15
    plot_gene_heatmap(
        adata,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        annot_keys=["cov_cont", "cov_cat"],
        n_bins=n_bins,
    )
    fig, image_axes = _fig_image_axes()
    # include_pt_tile (True) + 2 annotations + 1 main heatmap = 4 image axes
    assert len(image_axes) == 4

    # Among image axes, exactly 3 should be 1×n_bins tiles; 1 should be G×n_bins (G=2) or 1×n_bins in edge cases
    tile_axes = []
    main_axes = []
    for ax in image_axes:
        arr = np.array(ax.images[0].get_array())
        if arr.shape[0] == 1 and arr.shape[1] == n_bins:
            tile_axes.append(ax)
        else:
            main_axes.append(ax)
    assert len(tile_axes) == 3
    assert len(main_axes) == 1
    H = np.array(main_axes[0].images[0].get_array())
    assert H.shape[1] == n_bins


def test_zscore_and_clip_reflected_in_image_data(adata_small_pt):
    _close_all()
    plot_gene_heatmap(
        adata_small_pt, ["g0"], pt_key="dpt_pseudotime", zscore=True, clip=(-1, 1), n_bins=10
    )
    _, im = _get_main_ax_and_im()
    H = np.array(im.get_array())
    # Should be clipped to [-1, 1]
    assert np.nanmin(H) >= -1 - 1e-6
    assert np.nanmax(H) <= 1 + 1e-6


def test_outfile_saves(tmp_path, adata_small_pt):
    _close_all()
    outfile = tmp_path / "heatmap.png"
    plot_gene_heatmap(adata_small_pt, ["g0"], pt_key="dpt_pseudotime", n_bins=5, outfile=outfile)
    assert outfile.exists()


def test_model_predictions_are_used_monotonic(adata_small_pt):
    _close_all()
    pt = adata_small_pt.obs["dpt_pseudotime"].to_numpy()
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)

    class DummyModel:
        def __init__(self, scale_by_name=True):
            self.scale_by_name = scale_by_name

        def predict(self, g):
            idx = int(g[1:]) if self.scale_by_name else 1
            return (1 + idx) * (1 + pt)  # strictly increasing in pt

    genes = ["g0", "g3", "g7"]
    plot_gene_heatmap(
        adata_small_pt,
        genes,
        model=DummyModel(),
        pt_key="dpt_pseudotime",
        n_bins=30,
        zscore=False,  # implementation z-scores anyway; we check monotonic bin-means after pipeline
    )
    _, im = _get_main_ax_and_im()
    H = np.array(im.get_array())
    for r in range(H.shape[0]):
        diffs = np.diff(np.nan_to_num(H[r], nan=-np.inf))
        # After z-scoring a strictly increasing series, diffs remain >= 0 (up to tiny noise)
        assert np.all(diffs >= -1e-9)


def test_linear_imputation_constructed():
    _close_all()
    # 10 cells, 1 gene; 4 at pt=0, 6 at pt=1 -> only bin 0 and last have data; others imputed
    counts = np.array([[0.0]] * 4 + [[10.0]] * 6)  # (10, 1)
    t = np.array([0.0] * 4 + [1.0] * 6, dtype=float)
    A = ad.AnnData(counts)
    A.var_names = ["g0"]
    A.obs["dpt_pseudotime"] = t

    n_bins = 6
    plot_gene_heatmap(
        A,
        ["g0"],
        pt_key="dpt_pseudotime",
        n_bins=n_bins,
        zscore=True,  # implementation z-scores anyway
    )
    _, im = _get_main_ax_and_im()
    H = np.array(im.get_array())  # shape (1, n_bins) after single gene
    # Expect monotonic increase across bins (linear interpolation + z-score preserves order)
    diffs = np.diff(np.nan_to_num(H[0], nan=-np.inf))
    assert np.all(diffs >= -1e-9)


def test_include_pt_tile_false_and_no_other_annots(adata_small_pt):
    _close_all()
    plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        include_pt_tile=False,
        annot_keys=[],  # no other annotations
        n_bins=12,
    )
    _, image_axes = _fig_image_axes()
    # Only one image axis should exist (the main heatmap)
    assert len(image_axes) == 1
    arr = np.array(image_axes[0].images[0].get_array())
    assert arr.shape[1] == 12  # n_bins


def test_heatmap_vrange_custom_and_norm_type(adata_small_pt):
    _close_all()
    plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1", "g2"],
        pt_key="dpt_pseudotime",
        zscore=False,  # code still z-scores internally; we only check the norm object
        heatmap_vmin=0.0,
        heatmap_vmax=3.0,
        n_bins=15,
    )
    _, im = _get_main_ax_and_im()
    # Norm should be a basic Normalize with our vmin/vmax when not using TwoSlopeNorm path
    assert isinstance(im.norm, Normalize)
    assert np.isclose(im.norm.vmin, 0.0)
    assert np.isclose(im.norm.vmax, 3.0)


def test_zscore_with_center_none_uses_normalize(adata_small_pt):
    _close_all()
    plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        zscore=True,
        heatmap_center=None,  # triggers Normalize instead of TwoSlopeNorm
        n_bins=10,
    )
    _, im = _get_main_ax_and_im()
    assert isinstance(im.norm, Normalize)
    assert not isinstance(im.norm, TwoSlopeNorm)


def test_zscore_with_center_uses_twoslopenorm(adata_small_pt):
    _close_all()
    plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        zscore=True,
        heatmap_center=0.0,  # default path → TwoSlopeNorm
        n_bins=10,
    )
    _, im = _get_main_ax_and_im()
    assert isinstance(im.norm, TwoSlopeNorm)


def test_categorical_palette_partial_fallback(adata_small_pt):
    _close_all()
    adata = adata_small_pt.copy()
    # numeric 3-level categorical
    cats = (np.arange(adata.n_obs) % 3).astype(int)  # values 0,1,2
    adata.obs["cov_cat3"] = cats
    palettes = {"cov_cat3": {0: "#ff0000", 1: "#00ff00"}}  # no entry for '2'
    n_bins = 20
    plot_gene_heatmap(
        adata,
        ["g0", "g1"],
        pt_key="dpt_pseudotime",
        annot_keys=["cov_cat3"],
        categorical_palettes=palettes,
        n_bins=n_bins,
    )
    _, image_axes = _fig_image_axes()
    # include_pt_tile + 1 categorical annotation + main = 3 image axes
    assert len(image_axes) == 3
    # annotation tiles are 1×n_bins
    tile_shapes = [np.array(ax.images[0].get_array()).shape for ax in image_axes]
    assert tile_shapes.count((1, n_bins)) == 2


def test_valid_genes_subset_filters_missing(adata_small_pt):
    _close_all()
    plot_gene_heatmap(adata_small_pt, ["g0", "__not_a_gene__"], pt_key="dpt_pseudotime", n_bins=7)
    _, im = _get_main_ax_and_im()
    H = np.array(im.get_array())
    # Only g0 remained → 1×n_bins
    assert H.shape == (1, 7)


def test_show_gene_labels_controls_ticks(adata_small_pt):
    _close_all()
    # all labels
    plot_gene_heatmap(
        adata_small_pt, ["g0", "g1"], pt_key="dpt_pseudotime", n_bins=6, show_gene_labels="all"
    )
    ax_main, _ = _get_main_ax_and_im()
    assert len([t for t in ax_main.get_yticklabels() if t.get_text()]) == 2
    _close_all()

    # no labels
    plot_gene_heatmap(
        adata_small_pt, ["g0", "g1"], pt_key="dpt_pseudotime", n_bins=6, show_gene_labels=None
    )
    ax_main, _ = _get_main_ax_and_im()
    assert len([t for t in ax_main.get_yticklabels() if t.get_text()]) == 0
    _close_all()

    # partial labels
    plot_gene_heatmap(
        adata_small_pt,
        ["g0", "g1", "g2"],
        pt_key="dpt_pseudotime",
        n_bins=6,
        show_gene_labels=["g1"],
    )
    ax_main, _ = _get_main_ax_and_im()
    labels = [t.get_text() for t in ax_main.get_yticklabels()]
    assert "g1" in labels
    assert labels.count("") >= 1
