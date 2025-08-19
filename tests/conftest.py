"""Fixtures for testing."""

import anndata as ad
import numpy as np
import pytest
import scanpy as sc


def _ensure_pseudotime(adata, t, root_idx, n_dcs=10):
    """Compute DPT pseudotime across Scanpy versions.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to compute pseudotime for.
    t : np.ndarray
        The pseudotime values to use.
    root_idx : int
        The index of the root cell in `adata.obs_names`.
    n_dcs : int, optional
        Number of diffusion components to compute (default: 10).
    """
    root_name = adata.obs_names[root_idx]
    adata.obs["__is_root__"] = adata.obs_names == root_name

    # Try new -> old APIs
    try:
        # Newer API
        sc.tl.dpt(adata, n_dcs=n_dcs, root_cells=[root_name])
    except TypeError:
        try:
            # Some versions use a boolean obs column via root_key
            sc.tl.dpt(adata, n_dcs=n_dcs, root_key="__is_root__")
        except TypeError:
            # Old API: needs diffmap + adata.uns['iroot'] (int index)
            try:
                sc.tl.diffmap(adata, n_comps=n_dcs)
            except Exception:
                # neighbors already computed; diffmap may still be required by old DPT
                pass
            adata.uns["iroot"] = int(np.where(adata.obs_names == root_name)[0][0])
            sc.tl.dpt(adata, n_dcs=n_dcs)

    # Safety fallback if pseudotime still missing
    if "dpt_pseudotime" not in adata.obs:
        tt = (t - t.min()) / (t.max() - t.min() + 1e-12)
        adata.obs["dpt_pseudotime"] = tt

    # Clean up helper column
    if "__is_root__" in adata.obs:
        del adata.obs["__is_root__"]


@pytest.fixture
def adata_small_pt():
    """Create a small AnnData object with pseudotime and synthetic gene expression data."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 300, 60

    # latent 1D trajectory (shuffled order to be realistic)
    t = np.linspace(0, 1, n_cells)
    t = np.clip(t + rng.normal(0, 0.02, n_cells), 0, 1)
    rng.shuffle(t)

    # gene-wise mean trends
    g = n_genes // 3
    mu = np.empty((n_cells, n_genes), dtype=float)
    mu[:, :g] = 2 + 8 * t[:, None]
    mu[:, g : 2 * g] = 2 + 8 * (1 - t)[:, None]
    mu[:, 2 * g :] = 2 + 8 * np.exp(-0.5 * ((t - 0.5) / 0.15) ** 2)[:, None]
    mu *= rng.lognormal(mean=0.0, sigma=0.3, size=(1, n_genes))

    # Gamma-Poisson (NB2) sampling
    theta = 2.0
    rate = rng.gamma(shape=theta, scale=mu / theta)
    counts = rng.poisson(rate).astype(np.int32)

    # AnnData
    adata = ad.AnnData(counts)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers["counts"] = adata.X.copy()

    # size factors
    sf = adata.layers["counts"].sum(axis=1).astype(float)
    if hasattr(sf, "A1"):  # sparse guard
        sf = sf.A1
    adata.obs["size_factors"] = sf / np.median(sf)

    # normalize/log1p for X; keep full snapshot in raw
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    # neighbors + DPT pseudotime (version-robust)
    sc.pp.pca(adata, n_comps=20, random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
    root_idx = int(np.argmin(t))
    _ensure_pseudotime(adata, t, root_idx, n_dcs=10)

    # store ground-truth for debugging
    adata.obs["true_t"] = t
    return adata
