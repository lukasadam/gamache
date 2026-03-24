from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np


_TRUTHY = {"jax", "1", "true", "yes", "on"}


def jax_is_requested() -> bool:
    """Return whether JAX autodiff was requested via MSSM_AUTODIFF."""
    return os.getenv("MSSM_AUTODIFF", "").strip().lower() in _TRUTHY


@lru_cache(maxsize=1)
def _jax_modules() -> tuple[Any, Any, Any | None] | None:
    """Lazily import JAX modules.

    Returns
    -------
    tuple
        (jax, jnp, jsparse) if import succeeds. `jsparse` may be None.
    None
        If JAX could not be imported.
    """
    try:
        import jax
        import jax.numpy as jnp

        try:
            from jax.experimental import sparse as jsparse
        except Exception:
            jsparse = None

        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

        return jax, jnp, jsparse
    except Exception:
        return None


def _require_jax() -> tuple[Any, Any, Any | None]:
    """Return imported JAX modules or raise if unavailable."""
    mods = _jax_modules()
    if mods is None:
        raise RuntimeError("JAX is not available")
    return mods


def jax_is_available() -> bool:
    """Return whether JAX can be imported."""
    return _jax_modules() is not None


def to_jax_array(x: Any) -> Any:
    """Convert array-like input to a JAX array."""
    if x is None:
        return None
    _, jnp, _ = _require_jax()
    return jnp.asarray(x)


def to_jax_mat(mat: Any) -> Any:
    """Convert matrix-like input to a JAX dense or sparse matrix.

    If JAX sparse is available and the input is SciPy sparse, returns BCOO.
    Otherwise returns a dense JAX array.
    """
    if mat is None:
        return None

    _, jnp, jsparse = _require_jax()

    # Already JAX-backed
    if type(mat).__module__.startswith("jax"):
        return mat

    if jsparse is not None:
        try:
            return jsparse.BCOO.from_scipy_sparse(mat)
        except Exception:
            pass

    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return jnp.asarray(mat)


def as_numpy_1d(x: Any) -> np.ndarray:
    """Convert array-like input to a flattened NumPy array."""
    return np.asarray(x).reshape(-1)