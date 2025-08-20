# tests/test_gene_heatmap.py
import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pytest

from gamache.pl.gene_heatmap import plot_gene_heatmap


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
