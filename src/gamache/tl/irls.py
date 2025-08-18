from __future__ import annotations

import numpy as np


def solve_penalized_wls(
    X: np.ndarray, z: np.ndarray, w: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (X^T W X + S) beta = X^T W z; return beta and covariance.
    W is diagonal with entries w.
    """
    WX = X * w[:, None]
    XtWX = X.T @ WX
    XtWz = X.T @ (w * z)
    A = XtWX + S
    beta = np.linalg.solve(A, XtWz)
    cov = np.linalg.inv(A)
    return beta, cov
