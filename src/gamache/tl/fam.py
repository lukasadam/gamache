"""Negative Binomial (NB2) Family for mssm."""

import warnings

import mssm
import numpy as np
import scipy as scp


class NegBin(mssm.models.Family):
    """Negative Binomial (NB2) Family (Poisson–Gamma mixture).

    We assume: Y_i ~ NB(mu_i, alpha), with Var(Y_i) = mu_i + alpha * mu_i^2.
    Here we use the NB2 parameterization where the dispersion alpha = 1 / size.

    Internally, scipy's nbinom is parameterized by (n, p) with pmf:
        P(Y=k) = C(k+n-1, k) * p^n * (1-p)^k
    and mean/var:
        E[Y] = n * (1-p) / p
        Var[Y] = n * (1-p) / p^2 = mu + mu^2 / n

    Thus:
        n = 1 / alpha
        p = n / (n + mu) = 1 / (1 + alpha * mu)

    Notes
    -----
    - `scale` in this Family is the NB dispersion alpha.
    - If `scale` is None, likelihood methods require providing `alpha` explicitly
      via kwargs (defaulted to 1.0 below to stay usable).
    - Common link is log (not canonical), so default link = LOG().

    References:
      - Hilbe, J. M. (2011). Negative Binomial Regression.
      - McCullagh & Nelder (1989). Generalized Linear Models.
      - Wood (2017). Generalized Additive Models.

    Parameters
    ----------
    link : mssm.models.Link, optional
        Link function; default is log link.
    scale : float, optional
        Dispersion parameter alpha; if None, must be provided in likelihood calls.
    """

    def __init__(
        self, link: mssm.models.Link = mssm.models.LOG(), scale: float | None = None
    ) -> None:
        """Initialize NegBin family with link and scale (dispersion)."""
        super().__init__(link, True, scale)
        self.is_canonical: bool = False  # log link is common but not canonical

    # --- Variance function and its derivative (NB2) ---
    def V(self, mu: np.ndarray) -> np.ndarray:
        """Var(Y|mu) = mu + alpha * mu^2.

        Parameters
        ----------
        mu : np.ndarray
            Mean vector (shape (-1,1) or (-1,))

        Returns
        -------
        np.ndarray
            Variance vector (same shape as mu)
        """
        alpha = 1.0 if (self.scale is None) else float(self.scale)
        return mu + alpha * np.power(mu, 2)

    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """D Var / D mu = 1 + 2 * alpha * mu.

        Parameters
        ----------
        mu : np.ndarray
            Mean vector (shape (-1,1) or (-1,))

        Returns
        -------
        np.ndarray
            Derivative vector (same shape as mu)
        """
        alpha = 1.0 if (self.scale is None) else float(self.scale)
        return 1.0 + 2.0 * alpha * mu

    # --- Log-likelihood / log-pmf ---
    def _alpha(self, alpha_kw: float | None) -> float:
        """Determine dispersion alpha to use.

        Parameters
        ----------
        alpha_kw : float, optional
            Dispersion provided via keyword argument.

        Returns
        -------
        float
            Dispersion to use (alpha_kw if provided, else self.scale, else 1.
        """
        if alpha_kw is not None:
            return float(alpha_kw)
        if self.scale is not None:
            return float(self.scale)
        # Fallback to 1.0 so methods remain usable without explicit scale
        return 1.0

    @staticmethod
    def _nb_n_p_from_mu_alpha(mu: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Convert (mu, alpha) -> (n, p) for scipy.stats.nbinom.

        n = 1/alpha; p = n / (n + mu) = 1 / (1 + alpha * mu)

        Parameters
        ----------
        mu : np.ndarray
            Mean vector (shape (-1,1) or (-1,))
        alpha : float
            Dispersion (alpha > 0)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            n, p parameters for scipy.stats.nbinom
        """
        # Guard alpha
        a = max(alpha, np.finfo(float).tiny)
        n = 1.0 / a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = 1.0 / (1.0 + a * mu)
        # Clamp p into (0,1) to avoid numerical issues
        p = np.clip(p, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        return n, p

    def lp(self, y: np.ndarray, mu: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Log-probability of each observation under NB2 with mean=mu and dispersion=alpha.

        Parameters
        ----------
        y : np.ndarray
            Observed non-negative integer counts (shape (-1,1) or (-1,))
        mu : np.ndarray
            Mean vector (same shape as y)
        alpha : float, optional
            Dispersion; if omitted, falls back to self.scale or 1.0

        Returns
        -------
        np.ndarray
            Log-probability vector (same shape as y and mu)
        """
        a = self._alpha(alpha)
        n, p = self._nb_n_p_from_mu_alpha(mu, a)
        # scipy expects y as integers; let scipy handle domain warnings
        return scp.stats.nbinom.logpmf(y, n, p)

    def llk(self, y: np.ndarray, mu: np.ndarray, alpha: float | None = None) -> float:
        """Sum of log pmf values (log-likelihood).

        Parameters
        ----------
        y : np.ndarray
            Observed non-negative integer counts (shape (-1,1) or (-1,))
        mu : np.ndarray
            Mean vector (same shape as y)
        alpha : float, optional
            Dispersion; if omitted, falls back to self.scale or 1.0

        Returns
        -------
        float
            Total log-likelihood
        """
        return float(np.sum(self.lp(y, mu, alpha)))

    # --- Deviance (NB2) ---
    # per-observation deviance contribution:
    # D_i = 2 * [ y_i * log(y_i / mu_i) - (y_i + 1/alpha) * log( (y_i + 1/alpha) / (mu_i + 1/alpha) ) ]
    # with the convention 0 * log(0/.) := 0
    @staticmethod
    def _nb2_D_vec(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
        """Per-observation deviance contributions for NB2.

        Parameters
        ----------
        n : float
            Size parameter for the NB2 distribution.
        y : np.ndarray
            Observed non-negative integer counts (shape (-1,1) or (-1,)).
        mu : np.ndarray
            Mean vector (same shape as y).
        alpha : float
            Dispersion (alpha > 0).

        Returns
        -------
        np.ndarray
            Per-observation deviance contributions (same shape as y).
        """
        n = 1.0 / max(alpha, np.finfo(float).tiny)  # size
        y = y.astype(float)
        mu = mu.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            term1 = y * (np.log(y) - np.log(mu))
            term2 = (y + n) * (np.log(y + n) - np.log(mu + n))

        # Handle limits: when y==0, y*log(y/mu) -> 0
        term1[~np.isfinite(term1)] = 0.0
        term2[~np.isfinite(term2)] = 0.0

        return 2.0 * (term1 - term2)

    def D(self, y: np.ndarray, mu: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Per-observation deviance contributions for NB2.

        Parameters
        ----------
        y : np.ndarray
            Observed non-negative integer counts (shape (-1,1) or (-1,)).
        mu : np.ndarray
            Mean vector (same shape as y).
        alpha : float, optional
            Dispersion; if omitted, falls back to self.scale or 1.0.

        Returns
        -------
        np.ndarray
            Per-observation deviance contributions (same shape as y).
        """
        a = self._alpha(alpha)
        return self._nb2_D_vec(y, mu, a)

    def deviance(self, y: np.ndarray, mu: np.ndarray, alpha: float | None = None) -> float:
        """Total deviance for NB2.

        Parameters
        ----------
        y : np.ndarray
            Observed non-negative integer counts (shape (-1,1) or (-1,)).
        mu : np.ndarray
            Mean vector (same shape as y).
        alpha : float, optional
            Dispersion; if omitted, falls back to self.scale or 1.0.

        Returns
        -------
        float
            Total deviance
        """
        return float(np.sum(self.D(y, mu, alpha)))

    # --- Initialization for mu (optional but handy) ---
    def init_mu(self, y: np.ndarray) -> np.ndarray:
        """Initialize mu by shrinking counts toward the global mean (like Poisson.init_mu).

        Parameters
        ----------
        y : np.ndarray
            Observed non-negative integer counts (shape (-1,1) or (-1,)).

        Returns
        -------
        np.ndarray
            Initialized mean vector (same shape as y).
        """
        gmu = float(np.mean(y))
        gmu = max(gmu, np.finfo(float).tiny)
        norm = y / gmu
        norm[norm > 1.9] = 1.9
        norm[norm < 0.1] = 0.1
        mu0 = gmu * norm
        return mu0
