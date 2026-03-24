from __future__ import annotations

from typing import Any

import numpy as np
import scipy as scp

from .families import Binomial, Gaussian, NegativeBinomial, Poisson
from .exp_fam import GAMMFamily
from .formula import Formula, f, fs, i, l, lhs

__all__ = [
    "lhs",
    "i",
    "l",
    "f",
    "fs",
    "Formula",
    "Gaussian",
    "Poisson",
    "Binomial",
    "NegativeBinomial",
    "GAMM",
]


class GAMM:
    """Minimal GAMM-compatible wrapper exposing `.fit()`.

    In the full upstream project this class handled smoothing penalties, random
    effects, REML, C++ solvers, etc. This pruned JAX-only version focuses on a
    core API that many scripts expect: `GAMM(formula, family).fit()`.

    `formula` can be a `Formula` or a minimal formula string.
    """

    def __init__(
        self,
        formula: Formula | str,
        family: GAMMFamily | str | None = None,
        *,
        data: Any | None = None,
        progress_bar: bool = False,
    ):
        self.progress_bar = bool(progress_bar)

        self._formula_str: str | None = None
        if isinstance(formula, str):
            self._formula_str = formula
            self.formula: Formula | None
            self.formula = None if data is None else Formula.from_string(formula, data=data)
        else:
            self.formula = formula

        if family is None:
            self.family = Gaussian()
        elif isinstance(family, str):
            fam = family.strip().lower()
            if fam in {"gaussian", "normal"}:
                self.family = Gaussian()
            elif fam in {"poisson"}:
                self.family = Poisson()
            elif fam in {"binomial", "bernoulli"}:
                self.family = Binomial()
            elif fam in {"negative_binomial", "negativebinomial", "neg-binomial", "nb", "nb2"}:
                # Default theta; can be overridden by passing a family object.
                self.family = NegativeBinomial()
            else:
                raise ValueError(f"Unknown family string: {family!r}")
        else:
            self.family = family

        self.coef: np.ndarray | None = None
        self.optim_result: Any | None = None

        self.edf: float = float("nan")

    def fit(
        self,
        method: str = "BFGS",
        maxiter: int = 200,
        x0: np.ndarray | None = None,
        jax_autodiff: bool = True,
        adata: Any | None = None,
        obs_mask: np.ndarray | None = None,
        layer: str | None = None,
        genes: list[str] | None = None,
        store_key: str = "jaxgamx",
        ridge: float = 1e-8,
        irls_maxiter: int = 25,
        gene_batch_size: int | None = None,
        **minimize_kwargs: Any,
    ) -> "GAMM":
        if self.formula is None:
            if self._formula_str is None:
                raise RuntimeError("Internal error: missing formula")
            if adata is None:
                raise ValueError(
                    "This GAMM was created with a formula string; pass `data=...` to GAMM(...) "
                    "or pass `adata=...` to fit()."
                )
            self.formula = Formula.from_string(self._formula_str, data=adata)

        if adata is not None:
            is_gaussian = isinstance(self.family, Gaussian)
            is_poisson = isinstance(self.family, Poisson)
            is_nb = isinstance(self.family, NegativeBinomial)
            if not (is_gaussian or is_poisson or is_nb):
                raise NotImplementedError(
                    "AnnData per-gene fitting in the pruned JAX-only build supports Gaussian(), Poisson(), and NegativeBinomial()."
                )

            # Build design matrix from adata.obs (or provided Formula.data if it's already adata-like)
            _y, cov_flat, _notNA, *_rest = self.formula.encode_data(adata, prediction=True)
            X = np.asarray(cov_flat, dtype=float)

            Y_src = getattr(adata, "X") if layer is None else getattr(adata, "layers")[layer]
            if scp.sparse.issparse(Y_src):
                Y = np.asarray(Y_src.todense())
            else:
                Y = np.asarray(Y_src)

            if Y.ndim != 2:
                raise ValueError("AnnData matrix must be 2D (n_obs, n_vars)")
            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    f"Design matrix row count ({X.shape[0]}) does not match adata rows ({Y.shape[0]})."
                )

            if obs_mask is not None:
                mask = np.asarray(obs_mask).reshape(-1)
                if mask.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"obs_mask length ({mask.shape[0]}) does not match n_obs ({X.shape[0]})."
                    )
                mask_bool = mask.astype(bool)
                if not np.any(mask_bool):
                    raise ValueError("obs_mask selects zero observations")
                X = X[mask_bool]
                Y = Y[mask_bool]

            var_names = getattr(adata, "var_names", None)
            if var_names is None:
                raise TypeError("adata.var_names is required")
            var_names = [str(v) for v in list(var_names)]

            selected_idx: list[int] | None
            if genes is not None:
                gene_set = {str(g) for g in genes}
                idx = [i for i, g in enumerate(var_names) if g in gene_set]
                if not idx:
                    raise ValueError("No requested genes found in adata.var_names")
                gene_names = [var_names[i] for i in idx]
                Y = Y[:, idx]
                selected_idx = idx
            else:
                gene_names = var_names
                selected_idx = None

            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore

            Xj = jnp.asarray(X)
            ridge = float(ridge)

            g_total = int(Y.shape[1])
            if gene_batch_size is not None:
                gene_batch_size_i = int(gene_batch_size)
                if gene_batch_size_i <= 0:
                    raise ValueError("gene_batch_size must be > 0")
            else:
                gene_batch_size_i = 64 if self.progress_bar else g_total

            def _iter_gene_slices() -> list[slice]:
                if g_total == 0:
                    return []
                return [
                    slice(s, min(s + gene_batch_size_i, g_total))
                    for s in range(0, g_total, gene_batch_size_i)
                ]

            gene_slices = _iter_gene_slices()

            use_tqdm = bool(self.progress_bar) and (len(gene_slices) > 1)
            if use_tqdm:
                try:
                    from tqdm.auto import tqdm  # type: ignore
                except Exception:  # pragma: no cover
                    tqdm = None  # type: ignore
                    use_tqdm = False

            # Gaussian: keep closed-form batched ridge solve

            if is_gaussian:
                @jax.jit
                def _fit_all_gaussian(Xj: Any, Yj: Any) -> tuple[Any, Any]:
                    XtX = Xj.T @ Xj
                    p = XtX.shape[0]
                    XtX = XtX + ridge * jnp.eye(p, dtype=XtX.dtype)
                    XtY = Xj.T @ Yj

                    L = jnp.linalg.cholesky(XtX)
                    try:
                        from jax.scipy.linalg import solve_triangular  # type: ignore

                        Z = solve_triangular(L, XtY, lower=True)
                        B = solve_triangular(L.T, Z, lower=False)
                    except Exception:
                        B = jnp.linalg.solve(XtX, XtY)

                    resid = Yj - (Xj @ B)
                    sigma_hat = jnp.sqrt(
                        jnp.mean(resid**2, axis=0) + jnp.finfo(resid.dtype).tiny
                    )
                    return B, sigma_hat

                coef = np.empty((X.shape[1], g_total), dtype=float)
                sigma_hat_np = np.empty((g_total,), dtype=float)

                it = gene_slices
                if use_tqdm and tqdm is not None:  # type: ignore[truthy-bool]
                    it = tqdm(it, desc="Fitting genes", unit="batch")  # type: ignore[assignment]

                for sl in it:
                    Yj_chunk = jnp.asarray(Y[:, sl])
                    B, sigma_hat = _fit_all_gaussian(Xj, Yj_chunk)
                    coef[:, sl] = np.asarray(jax.device_get(B))
                    sigma_hat_np[sl] = np.asarray(jax.device_get(sigma_hat)).reshape(-1)
            else:
                # Poisson / NegativeBinomial: batched IRLS
                irls_maxiter_i = int(irls_maxiter)
                if irls_maxiter_i <= 0:
                    raise ValueError("irls_maxiter must be > 0")

                theta = float(self.family.theta) if is_nb else float("inf")
                # alpha = 1/theta; for Poisson alpha=0
                alpha = 0.0 if is_poisson else 1.0 / theta

                @jax.jit
                def _fit_all_irls(Xj: Any, Yj: Any) -> Any:
                    n, p = Xj.shape
                    g = Yj.shape[1]
                    B = jnp.zeros((p, g), dtype=Xj.dtype)
                    eye = jnp.eye(p, dtype=Xj.dtype)

                    def body(_, B):
                        eta = Xj @ B  # (n, g)
                        mu = jnp.exp(eta)
                        mu = jnp.clip(mu, a_min=jnp.finfo(mu.dtype).tiny)

                        # Working response
                        z = eta + (Yj - mu) / mu

                        # Weights
                        if alpha == 0.0:
                            w = mu
                        else:
                            w = mu / (1.0 + alpha * mu)

                        # XtWX: (g,p,p)
                        XtWX = jnp.einsum('ni,nj,ng->gij', Xj, Xj, w)
                        XtWX = XtWX + ridge * eye[None, :, :]

                        # XtWz: (g,p)
                        XtWz = jnp.einsum('ni,ng,ng->gi', Xj, w, z)

                        sol = jnp.linalg.solve(XtWX, XtWz[..., None]).squeeze(-1)  # (g,p)
                        return sol.T

                    B = jax.lax.fori_loop(0, irls_maxiter_i, body, B)
                    return B

                coef = np.empty((X.shape[1], g_total), dtype=float)
                sigma_hat_np = np.full((g_total,), np.nan, dtype=float)

                it = gene_slices
                if use_tqdm and tqdm is not None:  # type: ignore[truthy-bool]
                    it = tqdm(it, desc="Fitting genes", unit="batch")  # type: ignore[assignment]

                for sl in it:
                    Yj_chunk = jnp.asarray(Y[:, sl])
                    B = _fit_all_irls(Xj, Yj_chunk)
                    coef[:, sl] = np.asarray(jax.device_get(B))

            varm = getattr(adata, "varm", None)
            if varm is None:
                raise TypeError("adata.varm is required (dict-like)")

            n_vars = len(var_names)
            if selected_idx is None:
                coef_store = coef.T
                sigma_store = sigma_hat_np.reshape(-1, 1)
            else:
                coef_store = np.full((n_vars, coef.shape[0]), np.nan, dtype=float)
                sigma_store = np.full((n_vars, 1), np.nan, dtype=float)
                coef_store[np.asarray(selected_idx, dtype=int), :] = coef.T
                sigma_store[np.asarray(selected_idx, dtype=int), 0] = sigma_hat_np

            varm[f"{store_key}_coef"] = coef_store
            varm[f"{store_key}_sigma"] = sigma_store

            uns = getattr(adata, "uns", None)
            if uns is not None:
                try:
                    uns[f"{store_key}_feature_names"] = list(
                        getattr(self.formula, "coef_names", [])
                    )
                    uns[f"{store_key}_gene_names"] = list(gene_names)
                    if isinstance(self.family, Gaussian):
                        uns[f"{store_key}_family"] = "gaussian"
                    elif isinstance(self.family, Poisson):
                        uns[f"{store_key}_family"] = "poisson"
                    elif isinstance(self.family, NegativeBinomial):
                        uns[f"{store_key}_family"] = "negative_binomial"
                    elif isinstance(self.family, Binomial):
                        uns[f"{store_key}_family"] = "binomial"
                except Exception:
                    pass

            # Store on model for convenience
            self.coef = None
            self.optim_result = None
            self.edf = float(coef.shape[0])
            self.coef_by_gene = coef
            self.sigma_by_gene = sigma_hat_np
            self.gene_names = gene_names

            return self

        y = self.formula.y
        if y is None:
            raise RuntimeError(
                "Formula has no dependent variable stored; pass `adata=...` to fit per gene from adata.X."
            )
        Xs = self.formula.Xs
        coef_split_idx = self.formula.coef_split_idx

        p = int(Xs[0].shape[1])
        if x0 is None:
            x0 = np.zeros((p,), dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)

        if jax_autodiff and hasattr(self.family, "enable_jax_autodiff"):
            self.family.enable_jax_autodiff(True)

        def obj(x: np.ndarray) -> float:
            c = np.asarray(x, dtype=float).reshape(-1, 1)
            return -float(self.family.llk(c, coef_split_idx, [y], Xs))

        def jac(x: np.ndarray) -> np.ndarray:
            c = np.asarray(x, dtype=float).reshape(-1, 1)
            g = self.family.gradient(c, coef_split_idx, [y], Xs)
            return -np.asarray(g).reshape(-1)

        res = scp.optimize.minimize(
            obj,
            x0,
            method=method,
            jac=jac,
            options={"maxiter": int(maxiter), **minimize_kwargs.pop("options", {})},
            **minimize_kwargs,
        )

        self.optim_result = res
        self.coef = np.asarray(res.x, dtype=float).reshape(-1, 1)
        self.edf = float(self.coef.size)
        return self

    def get_llk(self, *_: Any, **__: Any) -> float:
        if self.coef is None:
            raise RuntimeError("Model is not fitted yet")
        y = self.formula.y
        return float(self.family.llk(self.coef, self.formula.coef_split_idx, [y], self.formula.Xs))

    def get_pars(self) -> tuple[np.ndarray, float | None]:
        if self.coef is None:
            raise RuntimeError("Model is not fitted yet")

        sigma: float | None = None
        if isinstance(self.family, Gaussian):
            if self.family.sigma is not None:
                sigma = float(self.family.sigma)
            else:
                X = self.formula.Xs[0]
                resid = self.formula.y - (X @ self.coef)
                sigma = float(np.sqrt(np.mean(np.asarray(resid) ** 2)))

        return self.coef, sigma

    def get_reml(self) -> float:
        raise NotImplementedError(
            "REML is not implemented in the pruned JAX-only build."
        )
