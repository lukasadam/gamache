from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy as scp

from .jax_autodiff import as_numpy_1d, jax_is_available, jax_is_requested, to_jax_array, to_jax_mat


class Link:
    """Link function base class."""

    def f(self, mu: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def fi(self, eta: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def dy1(self, mu: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def dy2(self, mu: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Identity(Link):
    r"""Identity link: $\eta = \mu$."""

    def f(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def fi(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def dy1(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def dy2(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros_like(mu)


class GSMMFamily:
    """Minimal base class for GSMM-style likelihoods with optional JAX autodiff.

    This is a pruned version of MSSM's original GSMMFamily, keeping only what is
    needed for JAX-first gradient/Hessian computation and the PropHaz example.
    """

    def __init__(self, pars: int, links: list[Link], *llkargs: Any) -> None:
        self.n_par = int(pars)
        self.links = links
        self.llkargs = llkargs

        self.extra_coef: int | None = None

        self.use_jax_autodiff: bool = False

        self._jax_cache_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None
        self._jax_cache: dict[str, object] = {}

    def enable_jax_autodiff(self, enabled: bool = True) -> None:
        self.use_jax_autodiff = bool(enabled)

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
    ) -> float:
        raise NotImplementedError

    def llk_jax(
        self,
        coef: Any,
        coef_split_idx: tuple[int, ...],
        ys: tuple[Any, ...],
        Xs: tuple[Any, ...],
    ) -> Any:
        raise NotImplementedError

    def _can_use_jax(self) -> bool:
        if not (self.use_jax_autodiff or jax_is_requested()):
            return False
        if not jax_is_available():
            return False
        return type(self).llk_jax is not GSMMFamily.llk_jax

    def _jax_prepare(
        self,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
    ) -> tuple[tuple[int, ...], tuple[object, ...], tuple[object, ...]]:
        key = (id(self), tuple(id(y) for y in ys), tuple(id(X) for X in Xs))
        coef_split_idx_t = tuple(int(i) for i in coef_split_idx)

        if self._jax_cache_key == key and "ys" in self._jax_cache and "Xs" in self._jax_cache:
            return (
                coef_split_idx_t,
                self._jax_cache["ys"],  # type: ignore[return-value]
                self._jax_cache["Xs"],  # type: ignore[return-value]
            )

        ys_jax = tuple(to_jax_array(y) if y is not None else None for y in ys)
        Xs_jax = tuple(to_jax_mat(X) if X is not None else None for X in Xs)

        self._jax_cache_key = key
        self._jax_cache["ys"] = ys_jax
        self._jax_cache["Xs"] = Xs_jax

        return coef_split_idx_t, ys_jax, Xs_jax

    def _jax_cached_grad_fn(self, llk_fn: Any, cache_sig: object) -> Any:
        import jax  # type: ignore

        grad_sig = self._jax_cache.get("_jax_grad_sig")
        grad_fn = self._jax_cache.get("_jax_grad_fn")

        if grad_fn is None or grad_sig != cache_sig:
            grad_fn = jax.jit(jax.grad(llk_fn))
            self._jax_cache["_jax_grad_fn"] = grad_fn
            self._jax_cache["_jax_grad_sig"] = cache_sig

        return grad_fn

    def _jax_cached_hvp_fn(self, llk_fn: Any, cache_sig: object) -> Any:
        import jax  # type: ignore

        hvp_sig = self._jax_cache.get("_jax_hvp_sig")
        hvp_fn = self._jax_cache.get("_jax_hvp_fn")

        if hvp_fn is None or hvp_sig != cache_sig:
            grad_fn = jax.grad(llk_fn)

            def hvp(c_flat: Any, v: Any) -> Any:
                return jax.jvp(grad_fn, (c_flat,), (v,))[1]

            hvp_fn = jax.jit(hvp)
            self._jax_cache["_jax_hvp_fn"] = hvp_fn
            self._jax_cache["_jax_hvp_sig"] = cache_sig

        return hvp_fn

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
    ) -> np.ndarray:
        if self._can_use_jax():
            import jax.numpy as jnp  # type: ignore

            coef_split_idx_t, ys_jax, Xs_jax = self._jax_prepare(coef_split_idx, ys, Xs)

            def llk_fn(c_flat: Any) -> Any:
                c_flat = jnp.asarray(c_flat).reshape(-1)
                return self.llk_jax(c_flat, coef_split_idx_t, ys_jax, Xs_jax)

            cache_sig = ("grad", self._jax_cache_key, coef_split_idx_t)
            grad_fn = self._jax_cached_grad_fn(llk_fn, cache_sig)
            g = grad_fn(jnp.asarray(coef).reshape(-1))
            return as_numpy_1d(g).reshape(-1, 1)

        def llk_warp(x: np.ndarray) -> float:
            return float(self.llk(x.reshape(-1, 1), coef_split_idx, ys, Xs))

        grad = scp.optimize.approx_fprime(np.asarray(coef).reshape(-1), llk_warp)
        return np.asarray(grad).reshape(-1, 1)

    def jhessian(
        self,
        jcols: np.ndarray,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
    ) -> scp.sparse.csc_array:
        if self._can_use_jax():
            import jax.numpy as jnp  # type: ignore

            coef_split_idx_t, ys_jax, Xs_jax = self._jax_prepare(coef_split_idx, ys, Xs)

            def llk_fn(c_flat: Any) -> Any:
                c_flat = jnp.asarray(c_flat).reshape(-1)
                return self.llk_jax(c_flat, coef_split_idx_t, ys_jax, Xs_jax)

            cache_sig = ("hvp", self._jax_cache_key, coef_split_idx_t)
            hvp_fn = self._jax_cached_hvp_fn(llk_fn, cache_sig)

            p = int(np.asarray(coef).size)
            c_flat0 = jnp.asarray(coef).reshape(-1)
            jcols = np.asarray(jcols, dtype=int).reshape(-1)

            Hdat: list[float] = []
            Hrows: list[int] = []
            Hcols: list[int] = []

            for j in jcols:
                v = jnp.zeros((p,), dtype=c_flat0.dtype).at[int(j)].set(1.0)
                Hj = as_numpy_1d(hvp_fn(c_flat0, v))
                rows = np.arange(p, dtype=int)

                Hdat.extend(Hj.tolist())
                Hrows.extend(rows.tolist())
                Hcols.extend([int(j)] * p)

            Ha = scp.sparse.csc_array((Hdat, (Hrows, Hcols)), shape=(p, p))
            Ha = (Ha + Ha.T) * 0.5
            return Ha

        # Finite difference fallback: approximate all requested columns via jacobian-of-gradient.
        ccols: list[int] = []
        Hdat: list[float] = []
        Hrows: list[int] = []
        Hcols: list[int] = []
        Hdim = int(len(coef))

        for j in np.asarray(jcols, dtype=int).reshape(-1):

            def __d2llkj(r: float) -> np.ndarray:
                n_coef = np.array(coef, copy=True)
                n_coef[j] = r
                n_grad = self.gradient(n_coef, coef_split_idx, ys, Xs)
                return np.asarray(n_grad).reshape(-1)

            def vectorized_d2(rr: np.ndarray) -> np.ndarray:
                return np.apply_along_axis(__d2llkj, axis=0, arr=rr)

            Hsk = scp.differentiate.jacobian(vectorized_d2, np.asarray(coef[j]).reshape(()), order=2)
            Hj = np.asarray(Hsk.df).reshape(-1)

            Hjrows = np.arange(Hdim)
            Hjc = np.delete(Hj, ccols)
            Hjrows = np.delete(Hjrows, ccols)

            Hdat.extend(Hjc.tolist())
            Hrows.extend(Hjrows.tolist())
            Hcols.extend([int(j)] * len(Hjrows))

            ccols.append(int(j))

            Hjcols = np.arange(Hdim)
            Hjr = np.delete(Hj, ccols)
            Hjcols = np.delete(Hjcols, ccols)

            Hdat.extend(Hjr.tolist())
            Hcols.extend(Hjcols.tolist())
            Hrows.extend([int(j)] * len(Hjcols))

        return scp.sparse.csc_array((Hdat, (Hrows, Hcols)), shape=(Hdim, Hdim))

    def hessian(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
    ) -> scp.sparse.csc_array:
        return self.jhessian(np.arange(len(coef)), coef, coef_split_idx, ys, Xs)

    def get_resid(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
        **kwargs: Any,
    ) -> np.ndarray | None:
        return None

    def init_coef(self, models: list[Callable]) -> np.ndarray | None:
        return None


class JAXGSMMFamily(GSMMFamily):
    """Convenience base class for JAX-traceable GSMM families."""

    def __init__(self, pars: int, links: list[Link], *llkargs: Any) -> None:
        super().__init__(pars, links, *llkargs)
        self.use_jax_autodiff = True

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray | None],
    ) -> float:
        if not jax_is_available():
            raise RuntimeError(
                "JAXGSMMFamily requires JAX installed to evaluate llk via llk_jax."
            )

        coef_split_idx_t, ys_jax, Xs_jax = self._jax_prepare(coef_split_idx, ys, Xs)
        c_flat = to_jax_array(np.asarray(coef).reshape(-1))
        val = self.llk_jax(c_flat, coef_split_idx_t, ys_jax, Xs_jax)
        return float(np.asarray(val))


class PropHaz(GSMMFamily):
    """Proportional hazards family (WPS 2016) with analytic and JAX-traceable likelihood."""

    def __init__(self, ut: np.ndarray, r: np.ndarray):
        super().__init__(1, [Identity()], np.asarray(ut), np.asarray(r))

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray],
    ) -> float:
        delta = np.asarray(ys[0]).reshape(-1, 1)
        ut = np.asarray(self.llkargs[0])
        r = np.asarray(self.llkargs[1])
        nt = int(len(ut))

        X = Xs[0]
        eta = X @ coef
        gamma = np.exp(eta)

        llk = float(np.sum(delta * eta))

        gamma_p = 0.0
        for j in range(nt):
            ri = r == j
            dj = float(np.sum(delta[ri]))
            gamma_p += float(np.sum(gamma[ri]))
            llk -= dj * float(np.log(gamma_p))

        return float(llk)

    def llk_jax(
        self,
        coef: Any,
        coef_split_idx: tuple[int, ...],
        ys: tuple[Any, ...],
        Xs: tuple[Any, ...],
    ) -> Any:
        import jax.numpy as jnp  # type: ignore

        delta = ys[0].reshape(-1)
        X = Xs[0]
        coef = jnp.asarray(coef).reshape(-1)

        eta = X @ coef
        gamma = jnp.exp(eta)

        llk = jnp.sum(delta * eta)

        r = jnp.asarray(self.llkargs[1]).reshape(-1).astype(jnp.int32)
        nt = int(len(self.llkargs[0]))

        dj = jnp.bincount(r, weights=delta, length=nt)
        sj = jnp.bincount(r, weights=gamma, length=nt)
        gamma_p = jnp.cumsum(sj)

        tiny = jnp.finfo(gamma_p.dtype).tiny
        llk = llk - jnp.sum(dj * jnp.log(jnp.maximum(gamma_p, tiny)))
        return llk

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray],
    ) -> np.ndarray:
        if self._can_use_jax():
            return super().gradient(coef, coef_split_idx, ys, Xs)

        delta = np.asarray(ys[0]).reshape(-1, 1)
        ut = np.asarray(self.llkargs[0])
        r = np.asarray(self.llkargs[1])
        nt = int(len(ut))

        X = Xs[0]
        eta = X @ coef
        gamma = np.exp(eta).reshape(-1, 1)

        g = (delta.T @ X).astype(float)
        b_p = np.zeros_like(g)

        gamma_p = 0.0
        for j in range(nt):
            ri = r == j
            dj = float(np.sum(delta[ri]))
            gamma_i = gamma[ri, 0].reshape(-1, 1)
            gamma_p += float(np.sum(gamma_i))

            X_i = X[ri, :]
            bi = gamma_i.T @ X_i
            b_p += bi

            g -= dj * (b_p / gamma_p)

        return np.asarray(g).reshape(-1, 1)

    def hessian(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.spmatrix | scp.sparse.sparray],
    ) -> scp.sparse.csc_array:
        if self._can_use_jax():
            return super().hessian(coef, coef_split_idx, ys, Xs)

        delta = np.asarray(ys[0]).reshape(-1, 1)
        ut = np.asarray(self.llkargs[0])
        r = np.asarray(self.llkargs[1])
        nt = int(len(ut))

        X = Xs[0]
        eta = X @ coef
        gamma = np.exp(eta).reshape(-1, 1)

        b_p = np.zeros((1, X.shape[1]), dtype=float)
        gamma_p = 0.0
        A_p = scp.sparse.csc_array((X.shape[1], X.shape[1]))
        H = scp.sparse.csc_array((X.shape[1], X.shape[1]))

        for j in range(nt):
            ri = r == j
            dj = float(np.sum(delta[ri]))
            gamma_i = gamma[ri, 0].reshape(-1, 1)
            gamma_p += float(np.sum(gamma_i))

            X_i = X[ri, :]
            bi = gamma_i.T @ X_i
            b_p += bi

            A_i = (gamma_i * X_i).T @ X_i
            A_p += A_i

            Hj = dj * (b_p.T @ b_p) / (gamma_p**2) - dj * (A_p / gamma_p)
            H += Hj

        return scp.sparse.csc_array(H)
