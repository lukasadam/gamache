"""Fitting method for `PseudotimeGAM` (split out for browseability)."""

from __future__ import annotations

from typing import Any

import numpy as np


class _FitMixin:
    """Mixin providing `.fit()`."""

    def fit(self, genes=None, *, store_cov: bool = False) -> None:
        """Fit the model to the specified genes.

        :param genes: Genes to fit, defaults to None (fit all genes)
        :type genes: list[str] | list[int] | None, optional
        :param store_cov: Whether to store the covariance matrix, defaults to False
        :type store_cov: bool, optional
        :raises RuntimeError: If the coefficient matrix has an unexpected shape
        """

        # Get gene indices to fit
        if genes is None:
            idx = np.arange(self.adata.n_vars, dtype=int)
        else:
            idx = np.asarray(
                [
                    g if isinstance(g, int) else int(np.where(self.adata.var_names == g)[0][0])
                    for g in genes
                ],
                dtype=int,
            )

        # Fit the model using the JAX backend
        self._backend_model.fit(
            adata=self.adata,
            obs_mask=None,
            layer=self.layer,
            genes=[str(self.adata.var_names[i]) for i in idx],
            store_key=self.key,
            ridge=float(self.lam) + float(self.ridge),
            irls_maxiter=int(self.irls_maxiter),
        )

        # Retrieve the fitted coefficient matrix and check its shape
        coef_mat = np.asarray(self.adata.varm[self.key + "_coef"], float)
        if coef_mat.shape[1] != self.p:
            raise RuntimeError("Internal error: coef matrix has unexpected shape")

        # Store effective degrees of freedom and alpha (overdispersion) for fitted genes, NaN for unfitted genes
        edf_vec = np.full(self.adata.n_vars, np.nan, float)
        edf_vec[idx] = float(self.p)
        alpha_val = float(1.0 / float(self.theta))
        alpha_vec = np.full(self.adata.n_vars, np.nan, float)
        alpha_vec[idx] = alpha_val
        diagnostic_vec = np.full(self.adata.n_vars, np.nan, float)
        diagnostic_vec[idx] = 1.0

        # Store results in adata
        self.adata.var[self.key + "_edf"] = edf_vec
        self.adata.var[self.key + "_alpha"] = alpha_vec
        self.adata.var[self.key + "_diagnostics"] = diagnostic_vec

        # Optionally compute and store covariance matrices for fitted genes
        if store_cov:
            cov_stack = np.full((self.adata.n_vars, self.p, self.p), np.nan, float)
            X_fit = self.X
            ridge = float(self.lam) + float(self.ridge)
            for j in idx:
                beta = coef_mat[int(j)]
                eta = X_fit @ beta
                mu = np.exp(np.clip(eta, -50.0, 50.0))
                alpha = alpha_val
                w = mu / (1.0 + alpha * mu)
                XtWX = X_fit.T @ (X_fit * w[:, None]) + ridge * np.eye(self.p)
                try:
                    cov = np.linalg.inv(XtWX)
                except np.linalg.LinAlgError:
                    cov = np.linalg.pinv(XtWX)
                cov_stack[int(j)] = cov
            self.adata.varm[self.key + "_cov"] = cov_stack
