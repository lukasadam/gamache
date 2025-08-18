from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

from .irls import solve_penalized_wls

# ===========================
# core utilities
# ===========================


def nb2_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Elementwise NB2 log pmf (ignoring constants not depending on mu/alpha).
    Var(Y)=mu + alpha*mu^2, alpha>0.
    """

    r = 1.0 / alpha
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * np.log(r / (r + mu))
        + y * np.log(mu / (r + mu))
    )


def mu_eta(eta: np.ndarray) -> np.ndarray:
    return np.exp(eta)


def irls_weights(mu: np.ndarray, alpha: float) -> np.ndarray:
    # w_i = (dmu/deta)^2 / Var(Y) with log link: (mu^2) / (mu + alpha mu^2)
    return (mu * mu) / (mu + alpha * mu * mu + 1e-12)


def pseudo_response(eta: np.ndarray, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    # z = eta + (y - mu) / dmu/deta; with log link dmu/deta = mu
    return eta + (y - mu) / (mu + 1e-12)


def estimate_alpha_pearson(
    y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """Method-of-moments dispersion estimate from Pearson residuals; clipped to small positive."""
    if weights is None:
        weights = np.ones_like(y)
    num = np.sum(weights * (y - mu) ** 2) - np.sum(weights * mu)
    den = np.sum(weights * mu * mu) + 1e-12
    alpha = max(num / den, 1e-8)
    return float(alpha)


# =======================================
# Penalty + design assembly for B-splines
# =======================================


def penalty_matrix(k: int, order: int = 2) -> np.ndarray:
    """RW(order) difference penalty: K = D^T D, with D the finite-difference matrix."""
    if order < 1:
        return np.zeros((k, k), dtype=float)
    # Build D (k - order) x k
    D = np.eye(k, dtype=float)
    for _ in range(order):
        D = np.diff(D, n=1, axis=0)
    return D.T @ D


def block_diag(mats: Sequence[np.ndarray]) -> np.ndarray:
    """Lightweight block diagonal assembly."""
    p = int(sum(m.shape[0] for m in mats))
    out = np.zeros((p, p), dtype=float)
    c = 0
    for M in mats:
        k = M.shape[0]
        out[c : c + k, c : c + k] = M
        c += k
    return out


def apply_lineage_weights(
    X_blocks: Sequence[np.ndarray], lineage_weights: Optional[np.ndarray]
) -> List[np.ndarray]:
    """Scale columns in block l by sqrt(w_{i,l}) row-wise (classic trick)."""
    if lineage_weights is None:
        return [np.asarray(X, float) for X in X_blocks]
    Xw = []
    for l, X in enumerate(X_blocks):
        w = lineage_weights[:, l : l + 1]
        Xw.append(np.asarray(X, float) * np.sqrt(w))
    return Xw


def build_block_design(
    X_blocks: Sequence[np.ndarray],
    lambdas: Sequence[float],
    lineage_weights: Optional[np.ndarray] = None,
    penalty_order: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[slice]]:
    """Concatenate a list of design blocks into a single X and build block-diagonal penalty S."""
    Xw_blocks = apply_lineage_weights(X_blocks, lineage_weights)
    slices: List[slice] = []
    cols = 0
    for X in Xw_blocks:
        k = X.shape[1]
        slices.append(slice(cols, cols + k))
        cols += k
    X = np.concatenate(Xw_blocks, axis=1) if len(Xw_blocks) > 1 else Xw_blocks[0]
    K_blocks = [
        lmb * penalty_matrix(X.shape[1], order=penalty_order) for X, lmb in zip(Xw_blocks, lambdas)
    ]
    S = block_diag(K_blocks)
    return X, S, slices


# ======================
# Unified fit result
# ======================


@dataclass
class FitResult:
    beta: np.ndarray  # (p,)
    alpha: float  # NB2 dispersion
    edf: float  # effective degrees of freedom
    cov: Optional[np.ndarray] = None  # (p,p) covariance (approx)
    diagnostics: Dict[str, float] = field(default_factory=dict)


# ============================
# Frequentist (IRLS) backend
# ============================


@dataclass
class IRLSBackend:
    """Penalized WLS (IRLS) backend for NB2-GAM."""

    max_iter: int = 50
    tol: float = 1e-6
    alpha_init: float = 0.1
    return_cov: bool = False

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        S: np.ndarray,
        offset: np.ndarray,
        obs_weights: Optional[np.ndarray] = None,
    ) -> FitResult:
        n, p = X.shape
        y = y.astype(float)
        offset = offset.astype(float)
        if obs_weights is None:
            obs_w = np.ones(n, dtype=float)
        else:
            obs_w = obs_weights.astype(float)

        beta = np.zeros(p, dtype=float)
        alpha = float(self.alpha_init)
        eta = offset + X @ beta
        mu = mu_eta(eta)

        cov = None
        for _ in range(int(self.max_iter)):
            w = irls_weights(mu, alpha) * obs_w
            if np.all(w == 0):
                break
            z = pseudo_response(eta, y, mu)
            # solve (X, z, w, S)
            beta_new, cov = solve_penalized_wls(X, z, w, S)

            eta_new = offset + X @ beta_new
            mu_new = mu_eta(eta_new)
            alpha_new = estimate_alpha_pearson(y, mu_new, weights=w)

            d_beta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-8)
            d_mu = np.mean(np.abs(mu_new - mu) / (mu + 1e-8))
            beta, eta, mu, alpha = beta_new, eta_new, mu_new, float(alpha_new)
            if max(d_beta, d_mu) < float(self.tol):
                break

        # EDF: trace(A^{-1} XtWX) with XtWX = X^T W X, A = XtWX + S
        w_final = irls_weights(mu, alpha) * obs_w
        if np.all(w_final == 0):
            edf = 0.0
        else:
            Xw = X * np.sqrt(w_final)[:, None]
            XtWX = Xw.T @ Xw
            A = XtWX + S
            try:
                M = np.linalg.solve(A, XtWX)
            except np.linalg.LinAlgError:
                M = np.linalg.solve(A + 1e-8 * np.eye(p), XtWX)
            edf = float(np.trace(M))

        return FitResult(
            beta=beta,
            alpha=alpha,
            edf=edf,
            cov=(cov if self.return_cov else None),
            diagnostics={"converged_mu": float(np.mean(mu))},
        )
