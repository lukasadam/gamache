# tests/test_gene_fit_scanpy.py
import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pytest

from gamache.pl.gene_fit import plot_gene_fit
from gamache.tl.fit import PseudotimeGAM


def _fit_one_gene_model(adata, gene_name="g0", key="nbgam_scanpy_plot"):
    """Fit a model for a single gene (fast, minimal)."""
    m = PseudotimeGAM(
        adata=adata,
        counts_layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        key=key,
        backend="irls",
        nonfinite="error",
    )
    idx = int(np.where(adata.var_names == gene_name)[0][0])
    m.fit(genes=[idx])
    return m


def test_returns_fig_and_ax(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    fig, ax = plot_gene_fit(model, "g0")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    fig.clf()


def test_scatter_and_smooth_line_present(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    fig, ax = plot_gene_fit(model, "g0")
    # scatter from scanpy
    assert any(isinstance(coll, matplotlib.collections.PathCollection) for coll in ax.collections)
    # smooth line
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "smooth" in labels
    fig.clf()


def test_color_forwarded(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    # Use another gene as coloring variable
    fig, ax = plot_gene_fit(model, "g0", color="g1")
    # Should still produce scatter + smooth
    assert len(ax.collections) >= 1
    assert any(ln.get_label() == "smooth" for ln in ax.get_lines())
    fig.clf()


def test_jitter_changes_pseudotime(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    t_before = model.t_filled.copy()
    fig, ax = plot_gene_fit(model, "g0", jitter=0.05)
    # Temporary obs columns must be gone
    assert not any(
        col.endswith("_tmp_expr") or col == "_tmp_pseudotime" for col in model.adata.obs.columns
    )
    # x data of scatter ≠ original pseudotime (due to jitter)
    scatter_x = ax.collections[0].get_offsets()[:, 0]
    assert not np.allclose(scatter_x, t_before, rtol=1e-6)
    fig.clf()


def test_title_and_labels(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    fig, ax = plot_gene_fit(model, "g0", title="custom title")
    assert ax.get_title() == "custom title"
    assert ax.get_xlabel() == "Pseudotime"
    assert ax.get_ylabel() == "Expression g0"
    fig.clf()


def test_invalid_gene_raises(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    with pytest.raises(ValueError, match="not found"):
        plot_gene_fit(model, "no_such_gene")


def test_cleanup_of_temp_obs_keys(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0")
    fig, ax = plot_gene_fit(model, "g0")
    # tmp keys should have been removed
    assert all(not k.startswith("_g0_tmp") for k in model.adata.obs.columns)
    fig.clf()
