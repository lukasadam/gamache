"""Plotting functionality to visualize gene expression as a heatmap over pseudotime."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import sparse


def _fill_runs_nan_1d(arr: np.ndarray) -> np.ndarray:
    """Fill consecutive NaN runs with the mean of the two neighbors.

    If both neighbors exist and match, use that; otherwise prefer left if present else right.
    If no neighbors exist, leave NaN.

    Parameters
    ----------
    arr : np.ndarray
        1D array with NaNs to fill.

    Returns
    -------
    np.ndarray
        Array with NaN runs filled.
    """
    a = np.asarray(arr, float).copy()
    n, i = a.size, 0
    while i < n:
        if np.isnan(a[i]):
            start = i
            while i < n and np.isnan(a[i]):
                i += 1
            end = i
            left = a[start - 1] if start > 0 and np.isfinite(a[start - 1]) else np.nan
            right = a[end] if end < n and np.isfinite(a[end]) else np.nan
            if np.isfinite(left) and np.isfinite(right):
                fill_val = 0.5 * (left + right)
            elif np.isfinite(left):
                fill_val = left
            elif np.isfinite(right):
                fill_val = right
            else:
                fill_val = np.nan
            a[start:end] = fill_val
        else:
            i += 1
    return a


def _fill_runs_categorical_1d(arr: np.ndarray) -> np.ndarray:
    """Fill NaN runs in integer-coded categories using nearest neighbor(s).

    Parameters
    ----------
    arr : np.ndarray
        1D array with NaNs to fill, where NaNs are treated as missing categories
        and the rest are integers representing categories.

    Returns
    -------
    np.ndarray
        Array with NaN runs filled.
    """
    a = np.asarray(arr, float).copy()
    n, i = a.size, 0
    while i < n:
        if np.isnan(a[i]):
            start = i
            while i < n and np.isnan(a[i]):
                i += 1
            end = i
            left = a[start - 1] if start > 0 and np.isfinite(a[start - 1]) else np.nan
            right = a[end] if end < n and np.isfinite(a[end]) else np.nan
            if np.isfinite(left) and np.isfinite(right) and left == right:
                fill_val = left
            elif np.isfinite(left):
                fill_val = left
            elif np.isfinite(right):
                fill_val = right
            else:
                fill_val = np.nan
            a[start:end] = fill_val
        else:
            i += 1
    return a


def plot_gene_heatmap(
    adata,
    genes,
    model=None,
    pt_key=None,
    n_bins=50,
    *,
    zscore=True,
    clip=(-2, 2),
    cmap="RdBu_r",
    heatmap_center=0.0,
    heatmap_vmin=None,
    heatmap_vmax=None,
    heatmap_norm=None,
    include_pt_tile=True,
    annot_keys=None,
    pt_cmap="viridis",
    pt_vmin=None,
    pt_vmax=None,
    continuous_cmaps=None,
    continuous_ranges=None,
    categorical_palettes=None,
    annot_row_height=0.28,
    order_by_peak=True,
    fill_empty="neighbor_mean",
    figsize=None,
    outfile=None,
):
    """Plot a heatmap of gene expression over pseudotime, with optional annotations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the data.
    genes : list of str
        List of gene names to include in the heatmap.
    model : Optional[object], optional
        A fitted model with a `predict` method, used to get gene expression predictions.
        If None, uses adata.X.
    pt_key : Optional[str], optional
        Key in adata.obs for pseudotime values. If None, tries to find a suitable key automatically.
    n_bins : int, optional
        Number of bins to use for pseudotime discretization, by default 50.
    zscore : bool, optional
        Whether to z-score the mean expression values, by default True.
    clip : tuple, optional
        Clipping range for z-scored values, by default (-2, 2).
    cmap : str, optional
        Colormap for the heatmap, by default "RdBu_r".
    heatmap_center : float, optional
        Center value for the heatmap colormap, by default 0.0.
    heatmap_vmin : Optional[float], optional
        Minimum value for the heatmap colormap, by default None (auto).
    heatmap_vmax : Optional[float], optional
        Maximum value for the heatmap colormap, by default None (auto).
    heatmap_norm : Optional[Normalize], optional
        Custom normalization for the heatmap, by default None (auto).
    include_pt_tile : bool, optional
        Whether to include a tile showing mean pseudotime per bin, by default True.
    annot_keys : Optional[list of str], optional
        List of keys in adata.obs to include as annotations. If None, no annotations are included.
    pt_cmap : str, optional
        Colormap for the pseudotime annotation, by default "viridis".
    pt_vmin : Optional[float], optional
        Minimum value for the pseudotime annotation colormap, by default None (auto).
    pt_vmax : Optional[float], optional
        Maximum value for the pseudotime annotation colormap, by default None (auto).
    continuous_cmaps : Optional[dict], optional
        Dictionary mapping continuous annotation keys to colormaps, by default None.
    continuous_ranges : Optional[dict], optional
        Dictionary mapping continuous annotation keys to value ranges for normalization, by default None.
    categorical_palettes : Optional[dict], optional
        Dictionary mapping categorical annotation keys to color palettes, by default None.
    annot_row_height : float, optional
        Height of each annotation row in the heatmap, by default 0.28.
    order_by_peak : bool, optional
        Whether to order genes by the peak expression bin, by default True.
    fill_empty : str, optional
        Method to fill empty bins: "neighbor_mean" (default), "zero", or leave as NaN.
    figsize : Optional[tuple], optional
        Figure size for the heatmap, by default None.
    outfile : Optional[str], optional
        If provided, saves the heatmap to this file path, by default None (does not save).
    """
    # --- pick pseudotime key
    if pt_key is None:
        for k in [
            "latent_time",
            "dpt_pseudotime",
            "curve_pseudotime",
            "pseudotime",
            "palantir_pseudotime",
            "pt",
        ]:
            if k in adata.obs:
                pt_key = k
                break
        if pt_key is None:
            raise ValueError("Provide pt_key (e.g. 'dpt_pseudotime' or 'latent_time').")

    # --- mask finite pseudotime and scale to [0,1]
    pt_all = adata.obs[pt_key].astype(float).values
    finite = np.isfinite(pt_all)
    if not np.any(finite):
        raise ValueError("No finite pseudotime values found.")
    pt = pt_all[finite]
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)

    # --- keep genes that exist
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        raise ValueError("None of the provided genes are in adata.var_names.")

    # --- get matrix X (cells x genes), aligned to finite mask; then log1p
    if model is not None:
        X_full = np.column_stack([model.predict(g) for g in genes]).astype(float, copy=False)
        X = X_full[finite, :]
    else:
        ad = adata[finite].copy()
        if sparse.issparse(ad.X):
            X = ad[:, genes].X.toarray().astype(float, copy=False)
        else:
            X = ad[:, genes].X.astype(float, copy=False)

    X = np.log1p(X)  # <- transform used for bin means

    # --- uniform bins in [0,1]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(pt, edges[:-1], right=False) - 1
    bin_ids = np.clip(bin_ids, 0, int(n_bins) - 1)

    # --- compute bin means (genes x bins)
    M = np.full((len(genes), n_bins), np.nan, dtype=float)
    for b in range(n_bins):
        m = bin_ids == b
        if np.any(m):
            M[:, b] = X[m, :].mean(axis=0)

    # --- fill empty bins by neighbor mean (constant across runs of NaNs)
    if fill_empty == "neighbor_mean":
        for i in range(M.shape[0]):
            row = M[i]
            idx = 0
            while idx < row.size:
                if np.isnan(row[idx]):
                    start = idx
                    while idx < row.size and np.isnan(row[idx]):
                        idx += 1
                    end = idx  # first non-NaN after block (or len)
                    left = row[start - 1] if start > 0 and np.isfinite(row[start - 1]) else np.nan
                    right = row[end] if end < row.size and np.isfinite(row[end]) else np.nan
                    if np.isfinite(left) and np.isfinite(right):
                        fill_val = 0.5 * (left + right)
                    elif np.isfinite(left):
                        fill_val = left
                    elif np.isfinite(right):
                        fill_val = right
                    else:
                        fill_val = np.nan  # can't fill (all-NaN row)
                    row[start:end] = fill_val
                else:
                    idx += 1
    elif fill_empty == "zero":
        M = np.nan_to_num(M, nan=0.0)
    else:
        pass  # leave NaNs

    # --- order genes by peak bin (on M, before z-scoring)
    if order_by_peak:
        # if a row is all-NaN, argmax fails; guard by replacing with -inf, then argmax->0
        M_for_order = np.where(np.isfinite(M), M, -np.inf)
        peak_bin = np.argmax(M_for_order, axis=1)
        order = np.argsort(peak_bin)  # early→late
    else:
        order = np.arange(len(genes))
    genes_ordered = [genes[i] for i in order]
    M_ord = M[order, :]

    # --- build the displayed matrix H (z-scored or raw)
    if zscore:
        mean = np.nanmean(M_ord, axis=1, keepdims=True)
        std = np.nanstd(M_ord, axis=1, keepdims=True) + 1e-12
        H = (M_ord - mean) / std
        if clip is not None:
            H = np.clip(H, *clip)
        # default diverging norm around 0 unless user passed heatmap_norm
        if heatmap_norm is not None:
            hm_norm = heatmap_norm
        else:
            if heatmap_center is not None:
                vmin_eff = np.nanmin(H) if heatmap_vmin is None else heatmap_vmin
                vmax_eff = np.nanmax(H) if heatmap_vmax is None else heatmap_vmax
                hm_norm = TwoSlopeNorm(vcenter=heatmap_center, vmin=vmin_eff, vmax=vmax_eff)
            else:
                vmin_eff = np.nanmin(H) if heatmap_vmin is None else heatmap_vmin
                vmax_eff = np.nanmax(H) if heatmap_vmax is None else heatmap_vmax
                hm_norm = Normalize(vmin=vmin_eff, vmax=vmax_eff)
        cbar_label = "Z-scored mean expression"
    else:
        H = M_ord
        if heatmap_norm is not None:
            hm_norm = heatmap_norm
        else:
            vmin_eff = np.nanmin(H) if heatmap_vmin is None else heatmap_vmin
            vmax_eff = np.nanmax(H) if heatmap_vmax is None else heatmap_vmax
            hm_norm = Normalize(vmin=vmin_eff, vmax=vmax_eff)
        cbar_label = "log1p(mean expression)"

    # --- build annotation rows
    annot_rows = []
    # per-bin mean pseudotime
    if include_pt_tile:
        pt_bin_means = np.full(n_bins, np.nan, dtype=float)
        for b in range(n_bins):
            m = bin_ids == b
            if np.any(m):
                pt_bin_means[b] = pt[m].mean()

        pt_bin_means = _fill_runs_nan_1d(pt_bin_means)

        pt_norm = Normalize(
            vmin=np.nanmin(pt_bin_means) if pt_vmin is None else pt_vmin,
            vmax=np.nanmax(pt_bin_means) if pt_vmax is None else pt_vmax,
        )
        annot_rows.append(
            {
                "name": pt_key,
                "type": "continuous",
                "array": pt_bin_means[None, :],
                "cmap": pt_cmap,
                "norm": pt_norm,
            }
        )

    # other covariates
    if annot_keys is None:
        annot_keys = []
    continuous_cmaps = {} if continuous_cmaps is None else dict(continuous_cmaps)
    continuous_ranges = {} if continuous_ranges is None else dict(continuous_ranges)
    categorical_palettes = {} if categorical_palettes is None else dict(categorical_palettes)

    # use obs aligned to finite mask
    ad_obs = adata.obs.iloc[finite].copy()
    # also carry the bin id so we can aggregate by bin
    ad_obs["_bin_id_tmp_"] = bin_ids

    def _cont_norm_for_key(key, arr):
        rng = continuous_ranges.get(key, None)
        if rng is not None:
            return Normalize(
                vmin=np.nanmin(arr) if rng[0] is None else rng[0],
                vmax=np.nanmax(arr) if rng[1] is None else rng[1],
            )
        return None

    for key in annot_keys:
        if key not in ad_obs.columns:
            continue
        s = ad_obs[key]
        # numeric covariate → per-bin mean
        if pd.api.types.is_numeric_dtype(s):
            vals = ad_obs.groupby("_bin_id_tmp_")[key].mean().reindex(range(n_bins)).values
            vals = _fill_runs_nan_1d(vals)
            annot_rows.append(
                {
                    "name": key,
                    "type": "continuous",
                    "array": vals[None, :],
                    "cmap": continuous_cmaps.get(key, "viridis"),
                    "norm": _cont_norm_for_key(key, vals),
                }
            )
        else:
            # categorical → per-bin mode
            s_cat = s.astype("category")
            cats = list(s_cat.cat.categories)
            by = ad_obs.groupby("_bin_id_tmp_")[key].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan
            )
            int_vals = (
                by.reindex(range(n_bins))
                .map({c: i for i, c in enumerate(cats)})
                .astype(float)
                .values
            )
            int_vals = _fill_runs_categorical_1d(int_vals)
            # palette
            if key in categorical_palettes:
                pal = categorical_palettes[key]
                colors, fallback = [], plt.get_cmap("tab20")
                for i, c in enumerate(cats):
                    colors.append(pal.get(c, fallback(i / max(len(cats) - 1, 1))))
            else:
                cmap_tab = plt.get_cmap("tab20")
                colors = [cmap_tab(i / max(len(cats) - 1, 1)) for i in range(len(cats))]
            annot_rows.append(
                {
                    "name": key,
                    "type": "categorical",
                    "array": int_vals[None, :],
                    "categories": cats,
                    "cmap": ListedColormap(colors),
                    "norm": Normalize(vmin=-0.5, vmax=(len(cats) - 0.5)),
                }
            )

    # --- layout: annotation tiles + main heatmap
    n_annot = len(annot_rows)
    base_main_h = max(3.0, 0.22 * len(genes_ordered))
    if figsize is None:
        figsize = (10.0, base_main_h + n_annot * annot_row_height + 0.6)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(
        n_annot + 1, 1, figure=fig, height_ratios=[annot_row_height] * n_annot + [base_main_h]
    )

    ax_main = fig.add_subplot(gs[-1, 0])
    annot_axes = [fig.add_subplot(gs[i, 0], sharex=ax_main) for i in range(n_annot)]

    extent = (-0.5, n_bins - 0.5, -0.5, 0.5)
    for axa, row in zip(annot_axes, annot_rows):
        axa.imshow(
            row["array"],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=row["cmap"],
            norm=row["norm"],
            extent=extent,
        )
        axa.set_yticks([0])
        axa.set_yticklabels([row["name"]], fontsize=9)
        axa.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for s in axa.spines.values():
            s.set_visible(False)

    im = ax_main.imshow(
        H,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=hm_norm,
    )

    # y ticks
    ax_main.set_yticks(range(len(genes_ordered)))
    ax_main.set_yticklabels(genes_ordered)

    # x ticks: show pseudotime 0→1
    xticks = np.linspace(0, n_bins - 1, 6)
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels([f"{t:.2f}" for t in np.linspace(0, 1, 6)])
    ax_main.set_xlabel(f"Pseudotime ({pt_key}, 0→early, 1→late)")
    ax_main.set_ylabel("Genes (early → late)")
    ax_main.set_xlim(-0.5, n_bins - 0.5)

    cbar = fig.colorbar(im, ax=[*annot_axes, ax_main], fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)

    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.show()

    return genes_ordered, bin_centers, H
