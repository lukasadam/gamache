"""Fixtures for testing."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from scipy import sparse

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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


def _make_counts_from_trends(rng: np.random.Generator, t: np.ndarray, n_genes: int):
    """Utility to synthesize NB2 counts across three trend archetypes."""
    g = n_genes // 3
    mu = np.empty((t.size, n_genes), dtype=float)
    mu[:, :g] = 2 + 8 * t[:, None]
    mu[:, g : 2 * g] = 2 + 8 * (1 - t)[:, None]
    mu[:, 2 * g :] = 2 + 8 * np.exp(-0.5 * ((t - 0.5) / 0.15) ** 2)[:, None]
    mu *= rng.lognormal(mean=0.0, sigma=0.3, size=(1, n_genes))

    theta = 2.0  # NB2 dispersion
    rate = rng.gamma(shape=theta, scale=mu / theta)
    counts = rng.poisson(rate).astype(np.int32)
    return counts


# ---------------------------------------------------------------------
# Global marks / utilities
# ---------------------------------------------------------------------

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="session")
def rng():
    """Session-scoped RNG for deterministic tests."""
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------
# Core dataset from your original fixture
# ---------------------------------------------------------------------


@pytest.fixture
def adata_small_pt(rng):
    """Create a small AnnData object with pseudotime and synthetic gene expression data."""
    n_cells, n_genes = 300, 60

    # latent 1D trajectory (shuffled order to be realistic)
    t = np.linspace(0, 1, n_cells)
    t = np.clip(t + rng.normal(0, 0.02, n_cells), 0, 1)
    rng.shuffle(t)

    # Gamma-Poisson (NB2) sampling via helper
    counts = _make_counts_from_trends(rng, t, n_genes)

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


# ---------------------------------------------------------------------
# Extra datasets for edge cases & fast unit tests
# ---------------------------------------------------------------------


@pytest.fixture
def adata_tiny_dense(rng):
    """Very small dense AnnData with simple pseudotime and counts layer.

    No Scanpy graph is computed here to keep tests fast; you can
    rely on the pre-set 'dpt_pseudotime'.
    """
    n_cells, n_genes = 40, 12
    t = np.linspace(0, 1, n_cells)
    counts = _make_counts_from_trends(rng, t, n_genes)
    adata = ad.AnnData(counts)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers["counts"] = counts.copy()
    adata.obs["dpt_pseudotime"] = t
    # No size_factors on purpose to exercise fallback to row-sums
    return adata


@pytest.fixture
def adata_tiny_sparse(rng):
    """Tiny AnnData with CSR 'counts' to test sparse code paths (_get_counts_col)."""
    n_cells, n_genes = 30, 9
    t = np.linspace(0, 1, n_cells)
    counts = _make_counts_from_trends(rng, t, n_genes)
    # Introduce some exact zeros to make detection thresholds meaningful
    counts[rng.uniform(size=counts.shape) < 0.2] = 0
    X = sparse.csr_matrix(counts, dtype=np.int32)
    adata = ad.AnnData(X)  # X is sparse; no counts layer initially
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["dpt_pseudotime"] = t
    # No layers['counts'] on purpose to hit ".X" fallback
    return adata


@pytest.fixture
def adata_nonfinite_time(rng):
    """Anndata whose pseudotime includes NaN/+inf/-inf to test nonfinite handling.

    The fixture DOES NOT normalize or log1p X; it's raw counts only.
    """
    n_cells, n_genes = 50, 15
    t = np.linspace(0, 1, n_cells)
    # Inject non-finite values
    t[5] = np.nan
    t[17] = np.inf
    t[33] = -np.inf
    counts = _make_counts_from_trends(
        rng, np.clip(np.nan_to_num(t, nan=0.5, posinf=1.0, neginf=0.0), 0, 1), n_genes
    )
    adata = ad.AnnData(counts)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers["counts"] = counts.copy()
    adata.obs["dpt_pseudotime"] = t  # intentionally non-finite
    return adata


@pytest.fixture
def adata_constant_time(rng):
    """Anndata with constant pseudotime to trigger the jitter/clamp logic."""
    n_cells, n_genes = 40, 10
    t = np.full(n_cells, 0.42, dtype=float)
    counts = _make_counts_from_trends(rng, np.linspace(0, 1, n_cells), n_genes)
    adata = ad.AnnData(counts)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers["counts"] = counts.copy()
    adata.obs["dpt_pseudotime"] = t  # constant
    return adata


@pytest.fixture
def adata_detection_mix(rng):
    """Anndata where some genes have too few nonzeros to pass detection filtering.

    Useful for exercising 'min_cells' behavior in test_all().
    """
    n_cells, n_genes = 60, 20
    t = np.linspace(0, 1, n_cells)
    counts = _make_counts_from_trends(rng, t, n_genes)

    # Make the last 5 genes extremely sparse (few non-zeros)
    mask = rng.uniform(size=(n_cells, 5)) < 0.95
    counts[:, -5:][mask] = 0

    adata = ad.AnnData(counts)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers["counts"] = counts.copy()
    adata.obs["dpt_pseudotime"] = t
    return adata


# ---------------------------------------------------------------------
# Small utility fixtures used across tests
# ---------------------------------------------------------------------


@pytest.fixture
def contrast_unit(request):
    """A simple unit-contrast factory.

    Usage:
        def test_something(contrast_unit):
            # later: c = contrast_unit(p) to match p coefficients
    """

    def _factory(p: int, k: int | None = None):
        c = np.zeros(p, dtype=float)
        c[0 if k is None else int(k)] = 1.0
        return c

    return _factory


@pytest.fixture
def t_new_mixed():
    """Mixed-finite t_new vector to test predict() sanitization."""
    return np.array([np.nan, -np.inf, 0.0, 0.25, 0.5, 0.75, np.inf, 1.0], dtype=float)


@pytest.fixture
def t_grid_dense():
    """A dense grid for curve summarization tests."""
    return np.linspace(0, 1, 200, dtype=float)


@pytest.fixture
def gene_names_small():
    """Helper for selecting a few gene names consistently."""
    return [f"g{i}" for i in range(5)]


# ---------------------------------------------------------------------
# Optional: lightweight IRLS backend stub (if you want to avoid heavy backends)
# ---------------------------------------------------------------------
# If your real IRLS backend is fast and available, you can delete this.
# Otherwise, enable the autouse fixture below to monkeypatch the backend used by
# PseudotimeGAM to a deterministic, lightweight linearized solver.


@pytest.fixture
def dummy_irls_backend_cls():
    """A tiny drop-in replacement for IRLSBackend with deterministic outputs.

    Returns a class with the same interface:
      DummyIRLS(return_cov=False, **kwargs).fit(y, X, S, offset, obs_weights)
    """

    class _Res:
        __slots__ = ("beta", "alpha", "edf", "cov", "diagnostics")

        def __init__(self, beta, alpha, edf, cov):
            self.beta = beta
            self.alpha = alpha
            self.edf = edf
            self.cov = cov
            self.diagnostics = {"converged_mu": 1.0}

    class DummyIRLS:
        def __init__(self, return_cov: bool = False, **kwargs):
            self.return_cov = bool(return_cov)

        def fit(self, y, X, S, offset, obs_weights):
            y = np.asarray(y, float)
            X = np.asarray(X, float)
            S = np.asarray(S, float)
            w = np.asarray(obs_weights, float)

            # Simple working-response: z = log1p(y) - offset
            z = np.log1p(np.clip(y, 0, None)) - np.asarray(offset, float)
            # Weighted/pentalized normal equations
            W = np.diag(w) if w.ndim == 1 else np.asarray(w)
            XtWX = X.T @ W @ X
            A = XtWX + S + 1e-6 * np.eye(X.shape[1])
            XtWz = X.T @ W @ z
            try:
                beta = np.linalg.solve(A, XtWz)
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A, XtWz, rcond=None)[0]
                Ainv = np.linalg.pinv(A)

            # Pseudo alpha and edf
            alpha = 0.5
            # edf via trace of hat matrix (approx)
            H = X @ Ainv @ X.T
            edf = float(np.trace(W @ H) / (np.trace(W) + 1e-12))
            cov = Ainv if self.return_cov else None
            return _Res(beta=beta, alpha=alpha, edf=edf, cov=cov)

    return DummyIRLS


@pytest.fixture
def patch_dummy_irls(monkeypatch, dummy_irls_backend_cls):
    """Monkeypatch the module-under-test to use DummyIRLS instead of IRLSBackend.

    Adjust the import path below to match your package layout if needed.
    If the import fails, the fixture will be skipped and tests will use the real backend.
    """
    import importlib

    candidates = [
        # Put your actual module path(s) here; the first importable one will be patched.
        "cellflux.gam",  # example package.module
        "cellflux.models.gam",  # alternative nested path
        "nbgam.gam",  # another possible name
    ]
    target_mod = None
    for name in candidates:
        try:
            target_mod = importlib.import_module(name)
            break
        except Exception:
            continue
    if target_mod is None:
        pytest.skip(
            "Could not locate module-under-test to patch IRLSBackend. "
            "Edit `patch_dummy_irls` candidates to your module path."
        )
    monkeypatch.setattr(target_mod, "IRLSBackend", dummy_irls_backend_cls)
    # Also patch where it's imported in the same package (relative import .backend)
    try:
        backend_mod = importlib.import_module(target_mod.__package__ + ".backend")
        monkeypatch.setattr(backend_mod, "IRLSBackend", dummy_irls_backend_cls)
    except Exception:
        # If backend module isn't importable, ignore
        pass
    return target_mod
