from __future__ import annotations

from typing import Any

import numpy as np


def _bspline_state_1d(x: np.ndarray, nk: int, degree: int) -> dict[str, Any]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if nk < degree + 2:
        raise ValueError(f"nk must be >= {degree + 2}")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax == xmin:
        return {
            "constant": True,
            "xmin": xmin,
            "xmax": xmax,
            "degree": int(degree),
            "t": np.asarray([0.0, 1.0], dtype=float),
            "drop_first": False,
        }

    xs = (x - xmin) / (xmax - xmin)

    n_internal = max(nk - degree - 1, 0)
    if n_internal > 0:
        qs = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
        internal = np.unique(np.quantile(xs, qs))
    else:
        internal = np.array([], dtype=float)

    tvec = np.concatenate(
        [np.zeros(degree + 1), internal, np.ones(degree + 1)],
        axis=0,
    )

    return {
        "constant": False,
        "xmin": xmin,
        "xmax": xmax,
        "degree": int(degree),
        "t": np.asarray(tvec, dtype=float),
        "drop_first": False,
    }


def _bspline_basis_1d_from_state(x: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if state.get("constant", False):
        return np.ones((x.shape[0], 1), dtype=float)

    xmin = float(state["xmin"])
    xmax = float(state["xmax"])
    degree = int(state["degree"])
    tvec = np.asarray(state["t"], dtype=float)

    xs = (x - xmin) / (xmax - xmin)
    n_basis = int(len(tvec) - degree - 1)
    B = np.empty((xs.shape[0], n_basis), dtype=float)

    from scipy.interpolate import BSpline

    for j in range(n_basis):
        c = np.zeros((n_basis,), dtype=float)
        c[j] = 1.0
        spl = BSpline(tvec, c, degree, extrapolate=True)
        B[:, j] = spl(xs)
    return B


def _bspline_basis_1d_predict(x: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Basis for prediction time (applies state's `drop_first`)."""
    B = _bspline_basis_1d_from_state(x, state)
    if bool(state.get("drop_first", False)) and B.shape[1] > 1:
        B = B[:, 1:]
    return B
