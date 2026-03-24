from __future__ import annotations

from typing import Any

from .exp_fam import JAXGSMMFamily


class Gaussian(JAXGSMMFamily):
    r"""Gaussian likelihood with identity mean link.

    This is a small compatibility family primarily intended for `GAMM.fit()`.
    The log-likelihood is returned up to an additive constant.

    Notes
    -----
    If `sigma` is None, the log-likelihood is evaluated with sigma profiled out:
    $\ell(b) = -n\log(\hat\sigma(b)) - n/2$ (up to a constant).
    """

    def __init__(self, sigma: float | None = None):
        super().__init__(pars=1, links=[])
        self.sigma = None if sigma is None else float(sigma)

    def llk_jax(self, coef, coef_split_idx, ys, Xs):
        import jax.numpy as jnp  # type: ignore

        X = Xs[0]
        y = ys[0].reshape(-1)
        b = jnp.asarray(coef).reshape(-1)

        mu = X @ b
        resid = y - mu

        sigma = self.sigma
        if sigma is None:
            n = resid.shape[0]
            rss = jnp.sum(resid**2)
            sigma_hat = jnp.sqrt(rss / n + jnp.finfo(resid.dtype).tiny)
            return -n * jnp.log(sigma_hat) - 0.5 * n

        n = resid.shape[0]
        return -n * jnp.log(sigma) - 0.5 * jnp.sum((resid / sigma) ** 2)


class Poisson(JAXGSMMFamily):
    r"""Poisson GLM with log link (canonical).

    Model:
        $y \sim \mathrm{Poisson}(\mu)$, with $\log(\mu) = Xb$.

    The log-likelihood includes the $-\log(y!)$ term.
    """

    def __init__(self):
        super().__init__(pars=1, links=[])

    def llk_jax(self, coef, coef_split_idx, ys, Xs):
        import jax.numpy as jnp  # type: ignore
        from jax.scipy.special import gammaln  # type: ignore

        X = Xs[0]
        y = jnp.asarray(ys[0]).reshape(-1)
        b = jnp.asarray(coef).reshape(-1)

        eta = X @ b
        mu = jnp.exp(eta)

        return jnp.sum(y * eta - mu - gammaln(y + 1.0))


class Binomial(JAXGSMMFamily):
    r"""Bernoulli/Binomial GLM with logit link.

    This minimal family expects a binary response $y \in \{0, 1\}$.

    Model:
        $\Pr(y=1) = \sigma(Xb)$.
    """

    def __init__(self):
        super().__init__(pars=1, links=[])

    def llk_jax(self, coef, coef_split_idx, ys, Xs):
        import jax.numpy as jnp  # type: ignore
        import jax.nn as jnn  # type: ignore

        X = Xs[0]
        y = jnp.asarray(ys[0]).reshape(-1)
        b = jnp.asarray(coef).reshape(-1)

        eta = X @ b
        return jnp.sum(y * jnn.log_sigmoid(eta) + (1.0 - y) * jnn.log_sigmoid(-eta))


class NegativeBinomial(JAXGSMMFamily):
    r"""Negative Binomial GLM (NB2) with log link.

    Parameterization:
        $y \sim \mathrm{NB}(\mu, \theta)$ with mean $\mu = \exp(Xb)$.

    Variance:
        $\mathrm{Var}(y) = \mu + \mu^2/\theta$.

    Notes
    -----
    - `theta` is the size (inverse overdispersion). Larger `theta` approaches Poisson.
    - This pruned build treats `theta` as fixed (not estimated).
    """

    def __init__(self, theta: float = 10.0):
        super().__init__(pars=1, links=[])
        if theta <= 0:
            raise ValueError("theta must be > 0")
        self.theta = float(theta)

    def llk_jax(self, coef, coef_split_idx, ys, Xs):
        import jax.numpy as jnp  # type: ignore
        from jax.scipy.special import gammaln  # type: ignore

        X = Xs[0]
        y = jnp.asarray(ys[0]).reshape(-1)
        b = jnp.asarray(coef).reshape(-1)

        eta = X @ b
        mu = jnp.exp(eta)

        theta = jnp.asarray(self.theta, dtype=mu.dtype)
        # log PMF: log Gamma(y+theta) - log Gamma(theta) - log(y!)
        #          + theta log(theta/(theta+mu)) + y log(mu/(theta+mu))
        tpmu = theta + mu
        return jnp.sum(
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1.0)
            + theta * (jnp.log(theta) - jnp.log(tpmu))
            + y * (jnp.log(mu) - jnp.log(tpmu))
        )


__all__ = ["Gaussian", "Poisson", "Binomial", "NegativeBinomial"]
