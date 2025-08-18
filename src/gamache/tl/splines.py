from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import BSpline


def _bspline_design(t: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Construct B-spline design matrix (n × K) for given knots and degree."""
    n_basis = len(knots) - degree - 1
    B = np.zeros((t.size, n_basis))
    # basis by unit vectors
    for k in range(n_basis):
        c = np.zeros(n_basis)
        c[k] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=True)
        B[:, k] = spl(t)
    return B


def bspline_basis(
    t: np.ndarray,
    df: int = 6,
    degree: int = 3,
    include_intercept: bool = False,
    t_min: float | None = None,
    t_max: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """B-spline basis and knots.

    Returns
    -------
    B : (n, K) design matrix
    knots : (K+degree+1,) knot vector
    """
    t = np.asarray(t, float)
    if t_min is None:
        t_min = float(np.nanmin(t))
    if t_max is None:
        t_max = float(np.nanmax(t))
    # interior knots equally spaced over [t_min, t_max]
    n_interior = max(df - degree - (0 if include_intercept else 1), 0)
    interior = np.linspace(t_min, t_max, num=max(n_interior, 0), endpoint=True)
    # clamped knots
    knots = np.concatenate(
        [
            np.repeat(t_min, degree + 1),
            interior,
            np.repeat(t_max, degree + 1),
        ]
    )
    print(f"Using {len(knots)} knots for B-spline basis (df={df}, degree={degree})")
    B = _bspline_design(t, knots, degree)
    if not include_intercept:
        # drop the first column to avoid implicit intercept
        B = B[:, 1:]
    return B, knots


def penalty_matrix(n_basis: int, order: int = 2) -> np.ndarray:
    """Difference penalty matrix (D^T D) of given order (default 2: curvature).

    Produces a (n_basis × n_basis) matrix K such that beta^T K beta penalizes
    (approximate) squared differences of order `order` between adjacent coefficients.
    """
    if n_basis <= order:
        return np.zeros((n_basis, n_basis))
    D = np.eye(n_basis)
    for _ in range(order):
        D = np.diff(D, n=1, axis=0)
    return D.T @ D
