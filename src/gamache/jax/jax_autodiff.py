from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np


_JAX_CACHE: dict[str, Any] = {}


def jax_is_requested() -> bool:
    """Whether the user requested JAX autodiff.

    Controlled via env var `MSSM_AUTODIFF`.
    Accepted truthy values: `jax`, `1`, `true`, `yes`, `on`.
    """

    v = os.getenv("MSSM_AUTODIFF", "").strip().lower()
    return v in {"jax", "1", "true", "yes", "on"}


def _import_jax() -> Optional[Tuple[Any, Any, Any]]:
    """Lazy import for jax/jnp/jsparse.

    Returns (jax, jnp, jsparse) on success, else None.
    """

    cached = _JAX_CACHE.get("jax_triplet")
    if cached is not None:
        return cached

    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore

        try:
            from jax.experimental import sparse as jsparse  # type: ignore
        except Exception:
            jsparse = None

        # Prefer numerical parity with NumPy/SciPy.
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

        _JAX_CACHE["jax_triplet"] = (jax, jnp, jsparse)
        return _JAX_CACHE["jax_triplet"]
    except Exception:
        _JAX_CACHE["jax_triplet"] = None
        return None


def jax_is_available() -> bool:
    return _import_jax() is not None


def to_jax_array(x: Any) -> Any:
    triplet = _import_jax()
    if triplet is None:
        raise RuntimeError("JAX is not available")

    _, jnp, _ = triplet
    if x is None:
        return None
    return jnp.asarray(x)


def to_jax_mat(mat: Any) -> Any:
    """Convert a SciPy sparse matrix/array to a JAX matrix.

    - If JAX sparse is available, converts to `BCOO`.
    - Otherwise falls back to dense `jnp.asarray(mat.toarray())`.
    """

    triplet = _import_jax()
    if triplet is None:
        raise RuntimeError("JAX is not available")

    _, jnp, jsparse = triplet
    if mat is None:
        return None

    # Already a JAX array or a JAX sparse object.
    mod = getattr(type(mat), "__module__", "")
    if mod.startswith("jax"):
        return mat

    if jsparse is not None:
        try:
            # Newer JAX supports direct conversion.
            return jsparse.BCOO.from_scipy_sparse(mat)
        except Exception:
            pass

    # Dense fallback.
    if hasattr(mat, "toarray"):
        return jnp.asarray(mat.toarray())
    return jnp.asarray(mat)


def as_numpy_1d(x: Any) -> np.ndarray:
    """Convert a JAX array (or array-like) to a 1D NumPy array."""

    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    try:
        return np.asarray(x).reshape(-1)
    except Exception:
        # Last resort for JAX DeviceArray-like objects
        return np.array(x).reshape(-1)
