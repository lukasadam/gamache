"""Iteratively Reweighted Least Squares (IRLS) for fitting penalized weighted least squares models."""

from __future__ import annotations

import numpy as np


def solve_penalized_wls(
    X: np.ndarray, z: np.ndarray, w: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (X^T W X + S) beta = X^T W z; return beta and covariance.

    W is diagonal with entries w.

    Parameters
    ----------
    X : (n, p) design matrix
        Design matrix of predictors.
    z : (n,) pseudo-response vector
        Pseudo-response values, typically computed from the linear predictor and response.
    w : (n,) weights vector
        Weights for each observation, often derived from the variance function.
    S : (p, p) penalty matrix
        Penalty matrix for regularization (e.g., for smoothness in GAMs).

    Returns
    -------
    beta : (p,) estimated coefficients
        Estimated coefficients for the weighted least squares problem.
    cov : (p, p) covariance matrix
        Covariance matrix of the estimated coefficients, computed as the inverse of the Hessian.
    """
    WX = X * w[:, None]
    XtWX = X.T @ WX
    XtWz = X.T @ (w * z)
    A = XtWX + S
    beta = np.linalg.solve(A, XtWz)
    cov = np.linalg.inv(A)
    return beta, cov
