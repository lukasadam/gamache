"""Plotting functionality to visualize gene expression over pseudotime."""

from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def plot_gene_fit(
    model,
    gene: str,
    *,
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: np.log1p(x),
    smooth_points: int = 200,
    scatter_alpha: float = 1.0,
    scatter_size: float = 80.0,
    jitter: float = 0.0,
    grid: bool = True,
    title: Optional[str] = None,
    color: Optional[str] = None,  # forward to sc.pl.scatter
    **scatter_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the fitted gene expression over pseudotime.

    Parameters
    ----------
    model : object
        A fitted model with a `predict` method and `adata` attribute.
    gene : str
        The gene for which to plot the fit.
    transform : Callable, optional
        Function to transform the response values, by default `np.log1p`.
    smooth_points : int, optional
        Number of points in the dense grid for smoothing, by default 200.
    scatter_alpha : float, optional
        The alpha transparency for the scatter points, by default 1.0.
    scatter_size : float, optional
        The size of the scatter points, by default 80.0.
    jitter : float, optional
        The amount of jitter to add to the pseudotime values, by default 0.0.
    grid : bool, optional
        Whether to show a grid in the plot, by default True.
    title : Optional[str], optional
        The title of the plot, by default None.
    color : Optional[str], optional
        The color for the scatter points, by default None.
    scatter_kwargs : dict, optional
        Additional keyword arguments to pass to `sc.pl.scatter`.
    """
    # --- check gene
    var_names = np.asarray(model.adata.var_names)
    if gene not in var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
    j = int(np.where(var_names == gene)[0][0])

    # --- get raw arrays
    y_raw = model._get_counts_col(j)
    if sp.issparse(y_raw):
        y_raw = y_raw.A.ravel()
    y_raw = np.asarray(y_raw).ravel()
    t = np.asarray(model.t_filled).ravel()

    if jitter and jitter > 0:
        rng = np.random.default_rng(0)
        t = t + rng.normal(0.0, jitter, size=t.shape)

    y = transform(y_raw)

    # --- temporarily add obs columns for plotting
    tmp_key = f"_{gene}_tmp_expr"
    tmp_pt = "_tmp_pseudotime"
    model.adata.obs[tmp_key] = y
    model.adata.obs[tmp_pt] = t

    # --- scatter via scanpy
    ax = sc.pl.scatter(
        model.adata,
        x=tmp_pt,
        y=tmp_key,
        color=color,
        alpha=scatter_alpha,
        size=scatter_size,
        show=False,
        **scatter_kwargs,
    )
    fig = ax.figure

    # --- smooth prediction
    t_grid = np.linspace(np.nanmin(t), np.nanmax(t), int(smooth_points))
    y_grid_pred = np.asarray(model.predict(gene, t_new=t_grid)).ravel()
    y_grid = transform(y_grid_pred)
    ax.plot(t_grid, y_grid, color="black", lw=2, label="smooth", linestyle="dashed")

    # cosmetics
    ax.set_title(title or f"Fit for {gene}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(f"Expression {gene}")
    if grid:
        ax.grid(True, linewidth=0.6, alpha=0.4)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)

    # clean up
    del model.adata.obs[tmp_key]
    del model.adata.obs[tmp_pt]

    return fig, ax
