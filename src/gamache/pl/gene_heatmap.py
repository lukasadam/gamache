"""Plotting functionality to visualize gene expression as a heatmap over pseudotime."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import sparse
from scipy.stats import mode


def _impute_1d_linear(arr: np.ndarray) -> np.ndarray:
    """Impute NaNs in a 1D array using linear interpolation.

    Parameters
    ----------
    arr
        1D array with NaNs to be imputed.

    Returns
    -------
    np.ndarray
        1D array with NaNs imputed by linear interpolation.

    """
    x = np.asarray(arr, dtype=float)
    n = x.size
    idx = np.arange(n)
    mask = np.isfinite(x)

    if mask.sum() == 0:
        return np.zeros(n, dtype=float)

    # np.interp linearly interpolates and extends ends with boundary values
    return np.interp(idx, idx[mask], x[mask])


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
    include_pt_tile=True,
    annot_keys=None,
    pt_cmap="viridis",
    continuous_cmaps=None,
    categorical_palettes=None,
    annot_row_height=0.28,
    figsize=(6, 10),
    outfile=None,
    show_gene_labels="all",
):
    """Plot a heatmap of gene expression over pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix with cells x genes.
    genes
        List of gene names to include in the heatmap.
    model
        Optional fitted GAM model. If provided, expression values will be taken from model predictions.
        Otherwise, expression values will be taken directly from `adata`.
    pt_key
        Key in `adata.obs` containing pseudotime values. If None, will search for common keys.
    n_bins
        Number of pseudotime bins to use.
    zscore
        Whether to z-score expression values per gene.
    clip
        Tuple specifying min and max values to clip z-scored expression values.
    cmap
        Colormap for the main heatmap.
    heatmap_center
        Center value for diverging colormaps. Only used if `zscore` is True and `cmap` is diverging.
    heatmap_vmin
        Minimum value for heatmap color scaling. If None, determined from data.
    heatmap_vmax
        Maximum value for heatmap color scaling. If None, determined from data.
    include_pt_tile
        Whether to include a pseudotime annotation tile above the heatmap.
    annot_keys
        List of keys in `adata.obs` to include as annotation rows above the heatmap.
    pt_cmap
        Colormap for the pseudotime annotation tile.
    continuous_cmaps
        Dictionary mapping continuous annotation keys to colormaps.
    categorical_palettes
        Dictionary mapping categorical annotation keys to color palettes (dicts mapping category to color).
    annot_row_height
        Height of each annotation row in inches.
    figsize
        Figure size in inches.
    outfile
        If provided, path to save the figure.
    show_gene_labels
        'all' to show all gene labels, None to hide all labels, or a list of gene names to label.
    """
    # 1) Pseudotime extraction
    if pt_key is None:
        for candidate in [
            "latent_time",
            "dpt_pseudotime",
            "curve_pseudotime",
            "pseudotime",
            "palantir_pseudotime",
            "pt",
        ]:
            if candidate in adata.obs:
                pt_key = candidate
                break
        if pt_key is None:
            raise ValueError("Provide pt_key (e.g. 'dpt_pseudotime' or 'latent_time').")

    pt_all = adata.obs[pt_key].astype(float).values
    finite_mask = np.isfinite(pt_all)
    if not np.any(finite_mask):
        raise ValueError("No finite pseudotime values found.")

    pt = pt_all[finite_mask]
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)

    # 2) Filter valid genes
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        raise ValueError("None of the provided genes are in adata.var_names.")

    # 3) Expression matrix
    if model is not None:
        X_full = np.column_stack([model.predict(g) for g in genes]).astype(float, copy=False)
        X = X_full[finite_mask, :]
    else:
        adata_subset = adata[finite_mask, genes].copy()
        X = adata_subset.X.toarray() if sparse.issparse(adata_subset.X) else adata_subset.X
        X = X.astype(float, copy=False)
    X = np.log1p(X)

    # 4) Binning pseudotime
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_assignments = np.digitize(pt, bin_edges[:-1], right=False) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    # 5) Mean per bin (NaN for empty bins)
    bin_means = np.full((len(genes), n_bins), np.nan, dtype=float)
    for b in range(n_bins):
        cells = bin_assignments == b
        if np.any(cells):
            bin_means[:, b] = X[cells, :].mean(axis=0)

    # 6) Impute bins (per gene, linear across bin index)
    for i in range(len(genes)):
        bin_means[i, :] = _impute_1d_linear(bin_means[i, :])
    bin_means = pd.DataFrame(bin_means, index=genes, columns=bin_centers)

    # 7) Z-score rows and sort by peak time
    bin_means = (bin_means - bin_means.mean(axis=1).values[:, None]) / (
        bin_means.std(axis=1).values[:, None] + 1e-12
    )
    bin_means = np.clip(bin_means, clip[0], clip[1])

    # identify peak bin + value
    bin_means["max_col"] = bin_means.idxmax(axis=1)
    bin_means["max_val"] = bin_means.max(axis=1)

    # sort by (bin index asc, then peak value asc)
    col_order = {col: i for i, col in enumerate(bin_means.columns[:-2])}
    bin_means = bin_means.sort_values(
        by=["max_col", "max_val"],
        key=lambda x: x.map(col_order) if x.name == "max_col" else x,
        ascending=[False, False],
    )

    # drop helper columns
    bin_means = bin_means.drop(columns=["max_col", "max_val"])
    display_matrix = bin_means.values

    # 8) Color normalization
    if heatmap_vmin is None:
        heatmap_vmin = np.nanmin(display_matrix)
    if heatmap_vmax is None:
        heatmap_vmax = np.nanmax(display_matrix)

    if zscore and heatmap_center is not None:
        norm = TwoSlopeNorm(vcenter=heatmap_center, vmin=heatmap_vmin, vmax=heatmap_vmax)
    else:
        norm = Normalize(vmin=heatmap_vmin, vmax=heatmap_vmax)

    # 9) Build annotation rows
    annot_rows = []

    # pseudotime tile
    if include_pt_tile:
        pt_bin_means = np.full(n_bins, np.nan, dtype=float)
        for b in range(n_bins):
            cells = bin_assignments == b
            if np.any(cells):
                pt_bin_means[b] = pt[cells].mean()
        pt_bin_means = _impute_1d_linear(pt_bin_means)

        annot_rows.append(
            {
                "name": pt_key,
                "type": "continuous",
                "values": pt_bin_means,
                "cmap": pt_cmap,
                "vmin": float(np.nanmin(pt_bin_means)),
                "vmax": float(np.nanmax(pt_bin_means)),
            }
        )

    if annot_keys is not None:
        continuous_cmaps = continuous_cmaps or {}
        categorical_palettes = categorical_palettes or {}

        obs_subset = adata.obs.loc[finite_mask].copy()
        obs_subset["_bin_"] = bin_assignments

        for key in annot_keys:
            if key not in obs_subset.columns:
                continue

            # numeric → mean per bin
            if pd.api.types.is_numeric_dtype(obs_subset[key]):
                bin_means_key = obs_subset.groupby("_bin_")[key].mean()
                vals = np.array([bin_means_key.get(b, np.nan) for b in range(n_bins)], dtype=float)
                vals = _impute_1d_linear(vals)

                annot_rows.append(
                    {
                        "name": key,
                        "type": "continuous",
                        "values": vals,
                        "cmap": continuous_cmaps.get(key, "viridis"),
                        "vmin": float(np.nanmin(vals)),
                        "vmax": float(np.nanmax(vals)),
                    }
                )
            else:
                # categorical → mode per bin
                def safe_mode(s):
                    if len(s) == 0:
                        return np.nan
                    m = mode(s, keepdims=False)
                    return m.mode if m.count > 0 else np.nan

                bin_modes = obs_subset.groupby("_bin_")[key].apply(safe_mode)

                s = bin_modes.reindex(range(n_bins)).ffill().bfill()
                if s.isna().any():
                    global_mode = safe_mode(obs_subset[key].dropna())
                    s = s.fillna(global_mode)

                categories = obs_subset[key].astype("category").cat.categories.tolist()
                cat_to_idx = {c: i for i, c in enumerate(categories)}
                vals = np.array([cat_to_idx.get(v, np.nan) for v in s.values])

                if key in categorical_palettes:
                    colors = [
                        categorical_palettes[key].get(cat, plt.cm.tab10(i))
                        for i, cat in enumerate(categories)
                    ]
                else:
                    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

                annot_rows.append(
                    {
                        "name": key,
                        "type": "categorical",
                        "values": vals,
                        "categories": categories,
                        "cmap": ListedColormap(colors),
                        "vmin": -0.5,
                        "vmax": len(categories) - 0.5,
                    }
                )

    # 10) Figure layout
    n_annot = len(annot_rows)
    main_height = max(3.0, 10)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if n_annot > 0:
        gs = GridSpec(
            n_annot + 1, 1, figure=fig, height_ratios=[annot_row_height] * n_annot + [main_height]
        )
        ax_main = fig.add_subplot(gs[-1, 0])
        ax_annots = [fig.add_subplot(gs[i, 0], sharex=ax_main) for i in range(n_annot)]
    else:
        ax_main = fig.add_subplot(111)
        ax_annots = []

    # 11) Draw annotation rows
    for ax, row in zip(ax_annots, annot_rows):
        ax.imshow(
            row["values"].reshape(1, -1),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=row["cmap"],
            vmin=row["vmin"],
            vmax=row["vmax"],
            extent=(-0.5, n_bins - 0.5, -0.5, 0.5),
        )
        ax.set_yticks([0])
        ax.set_yticklabels([row["name"]], fontsize=9)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 12) Main heatmap
    im_main = ax_main.imshow(
        bin_means, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, norm=norm
    )

    ordered_genes = bin_means.index.tolist()
    ax_main.set_yticks(range(len(ordered_genes)))

    if show_gene_labels == "all":
        ax_main.set_yticklabels(ordered_genes, fontsize=8)
    elif show_gene_labels is None:
        ax_main.set_yticks([])
        ax_main.set_yticklabels([])
    elif isinstance(show_gene_labels, (list, tuple, set)):
        # Label only selected genes
        labels = [g if g in show_gene_labels else "" for g in ordered_genes]
        ax_main.set_yticklabels(labels, fontsize=8)
    else:
        raise ValueError("show_gene_labels must be 'all', None, or a list of gene names.")

    # 13) X-axis and colorbar
    xtick_positions = np.linspace(0, n_bins - 1, 6)
    xtick_labels = [f"{x:.2f}" for x in np.linspace(0, 1, 6)]
    ax_main.set_xticks(xtick_positions)
    ax_main.set_xticklabels(xtick_labels)
    ax_main.set_xlabel(f"Pseudotime ({pt_key})")
    ax_main.set_ylabel("Genes (ordered by peak time)")
    ax_main.set_xlim(-0.5, n_bins - 0.5)

    # 13) Colorbar
    cbar = fig.colorbar(
        im_main, ax=(ax_annots + [ax_main]) if ax_annots else [ax_main], fraction=0.025, pad=0.02
    )
    cbar.set_label("Expression (z-scored)")

    # 14) Save / Show
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight", dpi=300)

    plt.show()
