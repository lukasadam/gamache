from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

ArrayLike = Union[np.ndarray, "scipy.sparse.spmatrix"]


def plot_gene_fit(
    model,  # PseudotimeGAM-like: .adata, .t_filled, .predict(gene, t_new), .fitted_values(gene), ._get_counts_col(j)
    gene: str,
    *,
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: np.log1p(x),
    smoother: bool = True,  # draw dense-grid smooth curve via model.predict
    smooth_points: int = 200,
    show_binned: bool = True,  # draw a binned mean with CI (on transformed scale)
    bins: int = 30,
    ci: float = 0.95,
    scatter_alpha: float = 0.35,
    scatter_size: float = 8.0,
    jitter: float = 0.0,  # add small jitter to t (e.g., 0.002) to reduce overplotting
    max_points: Optional[int] = None,  # downsample scatter for huge n
    grid: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed counts vs. pseudotime with fitted curve(s) for a single gene.

    - Observed points (optionally downsampled/jittered)
    - Fitted values at observed t
    - Optional dense-grid smoother via `model.predict`
    - Optional binned mean with simple CI on the transformed scale

    Notes
    -----
    * By default applies log1p to both observed and fitted so scales match.
    * The function avoids hard-coded colors to play nicely with your mpl style/theme.
    """
    # --- resolve gene index
    var_names = np.asarray(model.adata.var_names)
    matches = np.where(var_names == gene)[0]
    if matches.size == 0:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
    j = int(matches[0])

    # --- get data
    y_raw: ArrayLike = model._get_counts_col(j)  # 1D counts for cells
    # convert sparse -> dense if needed
    if sp.issparse(y_raw):
        y_raw = y_raw.A.ravel()

    y_raw = np.asarray(y_raw).ravel()

    t = np.asarray(model.t_filled).ravel()
    if t.shape[0] != y_raw.shape[0]:
        raise ValueError(f"Shape mismatch: t has {t.shape[0]} cells, y has {y_raw.shape[0]}.")

    # --- fitted at observed t
    yhat_raw = np.asarray(model.fitted_values(gene)).ravel()
    if yhat_raw.shape[0] != y_raw.shape[0]:
        raise ValueError("model.fitted_values(gene) must return per-cell fitted values.")

    y = transform(y_raw)
    yhat = transform(yhat_raw)

    # --- order for lines
    order = np.argsort(t)
    t_sorted = t[order]
    yhat_sorted = yhat[order]

    # --- optional downsampling
    if max_points is not None and y.size > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(y.size, size=max_points, replace=False))
        t_scatter = t[keep]
        y_scatter = y[keep]
    else:
        t_scatter = t
        y_scatter = y

    # --- jitter (helps when many identical t)
    if jitter and jitter > 0:
        rng = np.random.default_rng(0)
        t_scatter = t_scatter + rng.normal(0.0, jitter, size=t_scatter.shape)

    # --- prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.figure

    # --- draw
    # scatter of observed
    ax.scatter(t_scatter, y_scatter, s=scatter_size, alpha=scatter_alpha, label="observed")

    # fitted line at observed t (sorted)
    ax.plot(t_sorted, yhat_sorted, linewidth=2.0, label="fitted (per-cell)")

    # optional dense-grid smooth
    if smoother:
        t_grid = np.linspace(np.nanmin(t), np.nanmax(t), int(smooth_points))
        y_grid_raw = np.asarray(model.predict(gene, t_new=t_grid)).ravel()
        y_grid = transform(y_grid_raw)
        ax.plot(t_grid, y_grid, linestyle="--", linewidth=2.0, label="smooth (grid)")

    # optional binned mean with simple CI (on transformed scale)
    if show_binned and bins > 1:
        # digitize and compute mean + CI within each bin
        edges = np.linspace(np.nanmin(t), np.nanmax(t), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(bins, np.nan)
        lowers = np.full(bins, np.nan)
        uppers = np.full(bins, np.nan)

        from math import sqrt

        from scipy.stats import t as t_dist  # only used if available

        for i in range(bins):
            mask = (t >= edges[i]) & (t < edges[i + 1])
            yy = y[mask]
            if yy.size > 1:
                m = float(np.nanmean(yy))
                s = float(np.nanstd(yy, ddof=1))
                n = int(np.sum(~np.isnan(yy)))
                means[i] = m
                if n > 1:
                    # t-based CI on transformed scale
                    try:
                        q = float(t_dist.ppf(0.5 + ci / 2.0, df=n - 1))
                    except Exception:
                        q = 1.96  # fallback ~95%
                    half = q * s / sqrt(n)
                    lowers[i] = m - half
                    uppers[i] = m + half

        # plot binned mean line
        valid = ~np.isnan(means)
        if np.any(valid):
            ax.plot(
                centers[valid],
                means[valid],
                linewidth=2.0,
                linestyle=":",
                label=f"binned mean ({bins})",
            )
            # CI band if available
            ci_ok = valid & ~np.isnan(lowers) & ~np.isnan(uppers)
            if np.any(ci_ok):
                ax.fill_between(
                    centers[ci_ok], lowers[ci_ok], uppers[ci_ok], alpha=0.15, linewidth=0
                )

    # --- labels & cosmetics
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(f"Expression {gene}")
    ax.set_title(title or f"Fit for {gene}")
    if grid:
        ax.grid(True, linewidth=0.6, alpha=0.4)
    # Set ylim lower to zero
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax
