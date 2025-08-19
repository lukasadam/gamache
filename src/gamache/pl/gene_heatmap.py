"""Plotting function for gene heatmaps along pseudotime."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import sparse
from scipy.ndimage import gaussian_filter1d


def plot_gene_heatmap(
    adata,
    genes,
    model=None,
    pt_key=None,
    n_bins=50,
    smoothing_sigma=1.0,  # Gaussian smoothing in bin space (0 = off)
    clip=(-2, 2),  # clip z-scores for nicer colors
    heatmap_cmap="RdBu_r",
    heatmap_center=0.0,  # center value for TwoSlopeNorm; set None to disable centering
    heatmap_vmin=None,
    heatmap_vmax=None,
    heatmap_norm=None,  # custom matplotlib norm overrides center/vmin/vmax
    pt_cmap="viridis",
    pt_vmin=None,
    pt_vmax=None,
    continuous_cmaps=None,  # dict: {key: cmap}
    continuous_ranges=None,  # dict: {key: (vmin, vmax)}
    annotation_keys=None,
    categorical_palettes=None,
    show_row_labels="all",  # "all" or "selected"
    row_label_subset=None,  # used when show_row_labels == "selected"
    annot_row_height=0.3,
    figsize=None,
    outfile="pseudotime_heatmap.pdf",
):
    """Plot a heatmap of gene expression along pseudotime.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the gene expression data.
    genes : list of str
        List of gene names to include in the heatmap.
    model : object, optional
        A fitted model with a `predict` method. If None, uses the data directly. Otherwise,
        the model is used to predict gene expression along pseudotime.
    pt_key : str, optional
        Key in `adata.obs` that contains pseudotime values. If None, tries to
        automatically detect common pseudotime keys.
    n_bins : int, optional
        Number of bins to use for pseudotime discretization, by default 50.
    smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing applied to the binned data,
        by default 1.0 (set to 0 to disable smoothing).
    clip : tuple, optional
        Tuple specifying the clipping range for z-scores, by default (-2, 2).
    heatmap_cmap : str, optional
        Colormap for the heatmap, by default "RdBu_r"
    heatmap_center : float, optional
        Center value for TwoSlopeNorm; if None, no centering is applied.
    heatmap_vmin : float, optional
        Minimum value for heatmap normalization; if None, uses data min.
    heatmap_vmax : float, optional
        Maximum value for heatmap normalization; if None, uses data max.
    heatmap_norm : matplotlib.colors.Normalize, optional
        Custom normalization for the heatmap; overrides center/vmin/vmax.
    pt_cmap : str, optional
        Colormap for pseudotime annotations, by default "viridis".
    pt_vmin : float, optional
        Minimum value for pseudotime normalization; if None, uses data min.
    pt_vmax : float, optional
        Maximum value for pseudotime normalization; if None, uses data max.
    continuous_cmaps : dict, optional
        Dictionary mapping annotation keys to colormaps for continuous data.
    continuous_ranges : dict, optional
        Dictionary mapping annotation keys to (vmin, vmax) tuples for continuous data.
    annotation_keys : list of str, optional
        List of keys in `adata.obs` to include as annotations in the heatmap.
    categorical_palettes : dict, optional
        Dictionary mapping annotation keys to categorical color palettes.
    show_row_labels : str, optional
        How to show row labels: "all" for all genes, "selected" for a subset.
    row_label_subset : list of str, optional
        Subset of genes to show when `show_row_labels` is "selected".
    annot_row_height : float, optional
        Height of each annotation row in the heatmap, by default 0.3.
    figsize : tuple, optional
        Figure size for the heatmap, by default None (automatically determined).
    outfile : str, optional
        Output file path for saving the heatmap, by default "pseudotime_heatmap.pdf".
    """

    def _two_slope_or_norm(vmin, vmax, center, arr):  # noqa: D202
        """Return a TwoSlopeNorm or Normalize based on provided parameters."""
        if heatmap_norm is not None:
            return heatmap_norm
        if center is not None:
            vmin_eff = np.min(arr) if vmin is None else vmin
            vmax_eff = np.max(arr) if vmax is None else vmax
            return TwoSlopeNorm(vcenter=center, vmin=vmin_eff, vmax=vmax_eff)
        if (vmin is not None) or (vmax is not None):
            return Normalize(
                vmin=np.min(arr) if vmin is None else vmin,
                vmax=np.max(arr) if vmax is None else vmax,
            )
        return None

    def _cont_norm_for_key(key, arr):
        """Get normalization for a continuous key based on provided ranges."""
        rng = None if continuous_ranges is None else continuous_ranges.get(key, None)
        if rng is not None:
            return Normalize(
                vmin=np.min(arr) if rng[0] is None else rng[0],
                vmax=np.max(arr) if rng[1] is None else rng[1],
            )
        return None

    # --- 0) pseudotime key
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

    pt = adata.obs[pt_key].astype(float).values
    finite = np.isfinite(pt)
    ad = adata[finite].copy()
    pt = pt[finite]
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-9)  # [0,1]

    # --- 1) genes & matrix
    genes = [g for g in genes if g in ad.var_names]
    if not genes:
        raise ValueError("None of the provided genes are in adata.var_names.")

    if model is not None:
        # Predict on a dense grid if a model is provided
        X = np.array([model.predict(gene) for gene in genes]).T
    else:
        # Use the raw data directly
        if sparse.issparse(ad.X):
            X = ad[:, genes].X.toarray()
        else:
            X = ad[:, genes].X
    X = X.astype(float, copy=False)

    # --- 2) binning
    bin_ids = pd.qcut(pt, q=min(n_bins, len(np.unique(pt))), labels=False, duplicates="drop")
    n_bins = int(bin_ids.max() + 1)

    M = np.zeros((len(genes), n_bins), dtype=float)  # genes x bins
    for b in range(n_bins):
        mask = (bin_ids == b).values if isinstance(bin_ids, pd.Series) else (bin_ids == b)
        if mask.sum():
            M[:, b] = X[mask].mean(axis=0)

    # --- 3) smoothing
    if smoothing_sigma and smoothing_sigma > 0:
        M = gaussian_filter1d(M, sigma=smoothing_sigma, axis=1, mode="nearest")

    # --- 4) peak positions (center of mass)

    # find the bin of maximum expression (mode / peak)
    idx_max = M.argmax(axis=1)  # bin index of max
    peak_pos = idx_max.astype(float) / max(n_bins - 1, 1)  # scale to [0,1]

    df_com = pd.DataFrame({"peak_pos": peak_pos}, index=pd.Index(genes, name="gene"))
    df_com = df_com.sort_values("peak_pos", kind="mergesort")  # stable sort early→late

    genes_ord = df_com.index.tolist()  # <-- use DF order everywhere below
    M_df = pd.DataFrame(M, index=genes, columns=np.arange(n_bins))
    M_ord = M_df.loc[genes_ord].values

    # --- 5) z-score
    mean = M_ord.mean(axis=1, keepdims=True)
    std = M_ord.std(axis=1, keepdims=True) + 1e-9
    Z = (M_ord - mean) / std
    if clip is not None:
        Z = np.clip(Z, *clip)

    # --- 6) annotations
    pt_bin_means = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        mask = (bin_ids == b).values if isinstance(bin_ids, pd.Series) else (bin_ids == b)
        pt_bin_means[b] = pt[mask].mean() if mask.sum() else np.nan

    annotation_keys = [] if annotation_keys is None else list(annotation_keys)
    cat_palettes = categorical_palettes or {}
    continuous_cmaps = {} if continuous_cmaps is None else dict(continuous_cmaps)
    continuous_ranges = {} if continuous_ranges is None else dict(continuous_ranges)

    annot_rows = []

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

    def is_numeric_series(s: pd.Series) -> bool:
        """Check if a pandas Series is numeric."""
        return pd.api.types.is_numeric_dtype(s)

    def compute_mode_or_nan(series):
        """Compute the mode of a pandas Series, returning NaN if empty."""
        vc = series.value_counts(dropna=True)
        return vc.index[0] if len(vc) else np.nan

    for key in annotation_keys:
        if key not in ad.obs.columns:
            continue
        s = ad.obs[key]
        if is_numeric_series(s):
            vals = np.full(n_bins, np.nan, dtype=float)
            for b in range(n_bins):
                mask = (bin_ids == b).values if isinstance(bin_ids, pd.Series) else (bin_ids == b)
                if mask.sum():
                    vals[b] = pd.to_numeric(s.iloc[mask], errors="coerce").mean()
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
            s_cat = s.astype("category")
            cats = list(s_cat.cat.categories)
            cat_to_int = {c: i for i, c in enumerate(cats)}
            int_vals = np.full(n_bins, np.nan, dtype=float)
            for b in range(n_bins):
                mask = (bin_ids == b).values if isinstance(bin_ids, pd.Series) else (bin_ids == b)
                if mask.sum():
                    mode_val = compute_mode_or_nan(s_cat.iloc[mask])
                    int_vals[b] = np.nan if pd.isna(mode_val) else cat_to_int.get(mode_val, np.nan)
            if key in cat_palettes:
                colors, fallback = [], plt.get_cmap("tab20")
                for i, c in enumerate(cats):
                    col = cat_palettes[key].get(c, None)
                    colors.append(col if col is not None else fallback(i / max(len(cats) - 1, 1)))
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

    # --- 7) layout (aligned)
    n_annot = len(annot_rows)
    base_main_h = max(3.0, 0.22 * len(genes_ord))  # uses DF-based order length
    if figsize is None:
        figsize = (10.0, base_main_h + n_annot * annot_row_height + 0.6)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(
        n_annot + 1,
        1,
        figure=fig,
        height_ratios=[annot_row_height] * n_annot + [base_main_h],
    )

    ax = fig.add_subplot(gs[-1, 0])  # main heatmap
    annot_axes = [fig.add_subplot(gs[i, 0], sharex=ax) for i in range(n_annot)]

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

    hm_norm = _two_slope_or_norm(heatmap_vmin, heatmap_vmax, heatmap_center, Z)
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=heatmap_cmap,
        norm=hm_norm,
    )

    # y-ticks from DF order
    if show_row_labels == "all":
        ax.set_yticks(range(len(genes_ord)))
        ax.set_yticklabels(genes_ord)
    elif show_row_labels == "selected":
        sel = set(row_label_subset or [])
        idxs = [i for i, g in enumerate(genes_ord) if g in sel]
        ax.set_yticks(idxs)
        ax.set_yticklabels([genes_ord[i] for i in idxs])
    else:
        raise ValueError("show_row_labels must be 'all' or 'selected'.")

    ax.set_xticks(np.linspace(0, n_bins - 1, 6))
    ax.set_xticklabels([f"{t:.2f}" for t in np.linspace(0, 1, 6)])
    ax.set_xlabel(f"Pseudotime ({pt_key}, 0→early, 1→late)")
    ax.set_ylabel("Genes (early → late)")

    fig.colorbar(im, ax=[*annot_axes, ax], fraction=0.025, pad=0.02).set_label(
        "Z-scored mean expression"
    )
    ax.set_xlim(-0.5, n_bins - 0.5)
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.show()

    # Return peak positions in the **same DF order** as the heatmap rows
    return genes_ord, df_com.loc[genes_ord, "peak_pos"].values, Z
