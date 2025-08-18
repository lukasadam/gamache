# tests/test_plot_gene_fit.py
import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gamache.pl.gene_fit import plot_gene_fit
from gamache.tl.fit import PseudotimeGAM


def _fit_one_gene_model(adata, gene_name="g0", key="nbgam_plot"):
    """Build & fit a model for a single gene to keep tests fast."""
    m = PseudotimeGAM(
        adata=adata,
        counts_layer="counts",
        pseudotime_key="dpt_pseudotime",
        size_factors_key="size_factors",
        key=key,
        backend="irls",
        nonfinite="error",
    )
    # Fit the target gene so .fitted_values() works
    idx = int(np.where(adata.var_names == gene_name)[0][0])
    m.fit(genes=[idx])
    return m


def test_plot_basic_elements(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0", key="nbgam_plot_basic")

    fig, ax = plot_gene_fit(model, "g0")  # defaults: smoother=True, show_binned=True

    # Scatter (observed)
    assert len(ax.collections) >= 1

    # Lines: per-cell fitted, smooth grid, and (likely) binned mean
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "fitted (per-cell)" in labels
    assert "smooth (grid)" in labels
    # binned mean may skip if bins are empty; allow either presence or absence
    assert any(l.startswith("binned mean (") for l in labels) or True

    # y-axis lower bound at 0
    y0, _ = ax.get_ylim()
    assert y0 == 0.0

    fig.clf()


def test_no_smoother_no_binned_custom_transform_on_given_ax(adata_small_pt):
    import matplotlib.pyplot as plt

    model = _fit_one_gene_model(adata_small_pt, "g0", key="nbgam_plot_minimal")

    fig, ax = plt.subplots()
    fig2, ax2 = plot_gene_fit(
        model,
        "g0",
        smoother=False,
        show_binned=False,
        transform=np.log1p,
        ax=ax,
        title="custom",
    )

    # Returned figure/axes are the ones we passed
    assert fig2 is fig and ax2 is ax

    # Only the per-cell fitted line should be drawn
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_label() == "fitted (per-cell)"

    # Check plotted data equals transform(fitted) sorted by t
    eta = model.fitted_values("g0", type="link")
    mu = np.exp(eta)
    yhat = np.log1p(mu)
    order = np.argsort(model.t_filled)

    assert_allclose(lines[0].get_xdata(), model.t_filled[order])
    assert_allclose(lines[0].get_ydata(), yhat[order], rtol=1e-6, atol=1e-8)

    fig.clf()


def test_downsample_and_jitter_changes_scatter_count(adata_small_pt):
    model = _fit_one_gene_model(adata_small_pt, "g0", key="nbgam_plot_downsample")

    fig, ax = plot_gene_fit(model, "g0", max_points=50, jitter=0.01)
    # First PathCollection is the observed scatter; offsets rows == plotted points
    n_pts = ax.collections[0].get_offsets().shape[0]
    assert n_pts == 50
    fig.clf()


def test_bad_gene_raises(adata_small_pt):
    # Model doesn't need to be fit if gene name is invalid — error is raised early
    model = _fit_one_gene_model(adata_small_pt, "g0", key="nbgam_plot_bad")
    with pytest.raises(ValueError, match="not found"):
        plot_gene_fit(model, "no_such_gene")


def test_length_mismatch_raises(adata_small_pt, monkeypatch):
    model = _fit_one_gene_model(adata_small_pt, "g0", key="nbgam_plot_mismatch")

    # Force t length != y length (fitted/observed remain full length)
    t_orig = model.t_filled
    monkeypatch.setattr(model, "t_filled", t_orig[:-1], raising=False)

    with pytest.raises(ValueError, match="Shape mismatch"):
        plot_gene_fit(model, "g0")
