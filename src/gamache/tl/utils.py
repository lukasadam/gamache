"""Utility functions."""

import numpy as np


def _bh_fdr(p):
    """Benjamini–Hochberg FDR (vectorized, returns q-values in original order).

    Parameters
    ----------
    p : array-like
        Array of p-values, can contain NaNs.

    Returns
    -------
    q : np.ndarray
        Array of q-values (FDR-adjusted p-values) in the same order as input `p`.
    """
    p = np.asarray(p, float)
    n = np.sum(np.isfinite(p))
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p) + 1)
    q = np.full_like(p, np.nan, dtype=float)
    # only finite p-values get adjusted
    finite_idx = np.isfinite(p)
    if n > 0:
        p_fin = p[finite_idx]
        o = np.argsort(p_fin)
        ro = np.empty_like(o)
        ro[o] = np.arange(1, n + 1)
        q_fin = p_fin * n / ro
        # enforce monotonicity from largest to smallest p
        q_fin = np.minimum.accumulate(q_fin[o[::-1]])[::-1]
        q[finite_idx] = np.clip(q_fin, 0.0, 1.0)
    return q


def _center_of_mass(t, w):
    """Center of mass along pseudotime using nonnegative weights w (e.g., fitted means).

    Parameters
    ----------
    t : array-like
        Pseudotime values, can contain NaNs.
    w : array-like
        Weights corresponding to `t`, can contain NaNs.

    Returns
    -------
    float
        The weighted center of mass along pseudotime, or NaN if undefined.
    """
    t = np.asarray(t, float)
    w = np.asarray(w, float)
    w = np.where(np.isfinite(w), w, 0.0)
    t = np.where(np.isfinite(t), t, np.nan)
    s = np.nansum(w)
    if s <= 0:
        return np.nan
    return float(np.nansum(t * w) / s)


def _peak_time(t, y):
    """Argmax time of y on a dense grid; returns np.nan if undefined.

    Parameters
    ----------
    t : array-like
        Pseudotime values, can contain NaNs.
    y : array-like
        Response values corresponding to `t`, can contain NaNs.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if y.size == 0 or not np.isfinite(y).any():
        return np.nan
    i = int(np.nanargmax(y))
    return float(t[i])


def _dense_curve(model, gene, n_grid=200):
    """Predict on a dense, sorted pseudotime grid spanning observed range.

    Parameters
    ----------
    model : object
        A fitted model with a `predict` method.
    gene : str
        The gene for which to predict.
    n_grid : int, optional
        Number of points in the dense grid, by default 200.

    Returns
    -------
    t_grid : np.ndarray
        Dense pseudotime grid.
    y_grid : np.ndarray
        Predicted response values on the dense grid.
    """
    t_obs = np.asarray(model.t_filled, float)
    lo, hi = np.nanmin(t_obs), np.nanmax(t_obs)
    t_grid = np.linspace(lo, hi, n_grid)
    y_grid = model.predict(gene, t_new=t_grid)  # response scale
    return t_grid, y_grid


def _neg_binom_deviance(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Negative binomial deviance for a given response vector and fitted mean.

    Parameters
    ----------
    y : np.ndarray
        Response vector (e.g., counts).
    mu : np.ndarray
        Fitted mean vector (e.g., predicted counts).
    alpha : float
        Dispersion parameter (should be positive).

    Returns
    -------
    float
        The negative binomial deviance.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # sanitize inputs
    alpha = float(max(alpha, 1e-12))
    # Replace NaNs/±inf first, then clip
    y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=0.0)
    mu = np.nan_to_num(mu, nan=1.0, posinf=1e12, neginf=1e-12)

    # ensure strictly positive mu, and keep everything finite
    y = np.clip(y, 0.0, 1e12)
    mu = np.clip(mu, 1e-12, 1e12)

    # term1: y * (log y - log mu), with 0*log(0) defined as 0
    # (avoid log(0) by clipping y inside the log)
    logy = np.log(np.clip(y, 1e-12, None))
    logmu = np.log(mu)
    term1 = np.where(y > 0.0, y * (logy - logmu), 0.0)

    # term2: (y + 1/alpha) * [log(y + 1/alpha) - log(mu + 1/alpha)]
    inva = 1.0 / alpha
    num = y + inva
    den = mu + inva
    # num, den are strictly positive; still clip to avoid extreme logs
    num = np.clip(num, 1e-12, 1e12)
    den = np.clip(den, 1e-12, 1e12)
    term2 = num * (np.log(num) - np.log(den))

    dev = 2.0 * np.sum(term1 - term2)
    return float(max(dev, 0.0))
