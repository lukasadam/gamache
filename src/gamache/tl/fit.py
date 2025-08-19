"""Functionality to fit a single-trajectory Negative Binomial Generalized Additive Model (NB-GAM) to pseudotime data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2

from .backend import IRLSBackend, irls_weights
from .splines import bspline_basis, penalty_matrix
from .utils import _bh_fdr, _center_of_mass, _dense_curve, _peak_time


@dataclass
class PseudotimeGAM:
    """Single-trajectory NB-GAM with a selectable backend."""

    # config / inputs
    adata: ad.AnnData = field(repr=False)
    counts_layer: Optional[str] = "counts"
    pseudotime_key: str = "dpt_pseudotime"
    size_factors_key: str = "size_factors"
    df: int = 6
    degree: int = 3
    lam: float = 1.0
    include_intercept: bool = False
    key: str = "nbgam1d"
    nonfinite: str = "error"
    backend: str = "irls"
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)

    # derived (built in __post_init__)
    t: np.ndarray = field(init=False, repr=False)
    t_filled: np.ndarray = field(init=False, repr=False)
    _obs_weight_mask: np.ndarray = field(init=False, repr=False)
    sf: np.ndarray = field(init=False, repr=False)
    offset: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    p: int = field(init=False)
    S: np.ndarray = field(init=False, repr=False)
    _counts_layer: Optional[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the PseudotimeGAM object by validating inputs and constructing the model."""
        # ----- pseudotime -----
        if self.pseudotime_key not in self.adata.obs:
            raise ValueError(
                f"Expected a single pseudotime column '{self.pseudotime_key}' in .obs."
            )
        t_raw = self.adata.obs[self.pseudotime_key].to_numpy(dtype=float)
        finite = np.isfinite(t_raw)
        n_bad = int((~finite).sum())
        if self.nonfinite not in {"error", "mask", "median"}:
            raise ValueError("nonfinite must be 'error', 'mask', or 'median'.")
        if self.nonfinite == "error":
            if n_bad:
                raise ValueError(
                    f"Pseudotime contains {n_bad} non-finite values (NaN/±inf). "
                    "Fix your input or use nonfinite='mask' or 'median'."
                )
            t_clean = t_raw
            mask_w = np.ones_like(t_raw, dtype=float)
        else:
            if finite.sum() < 2:
                raise ValueError("Need ≥2 finite pseudotime values after sanitization.")
            t_med = float(np.median(t_raw[finite]))
            t_clean = np.where(finite, t_raw, t_med)
            mask_w = (
                finite.astype(float) if self.nonfinite == "mask" else np.ones_like(t_raw, float)
            )

        # small clamp & jitter to stabilize knot placement
        lo, hi = np.quantile(t_clean, [0.001, 0.999])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            t_clean = np.clip(t_clean, lo, hi)
        if float(np.max(t_clean) - np.min(t_clean)) == 0.0:
            n = t_clean.size
            t_clean = t_clean + np.linspace(-1e-9, 1e-9, n)

        self.t = t_raw
        self.t_filled = t_clean
        self._obs_weight_mask = mask_w

        # ----- size factors / offset -----
        if self.size_factors_key in self.adata.obs:
            sf = self.adata.obs[self.size_factors_key].to_numpy(dtype=float)
        else:
            if "n_counts" in self.adata.obs:
                sf = self.adata.obs["n_counts"].to_numpy(dtype=float)
            else:
                Xmat = (
                    self.adata.layers[self.counts_layer]
                    if self.counts_layer
                    and (self.adata.layers is not None)
                    and (self.counts_layer in self.adata.layers)
                    else self.adata.X
                )
                sf = np.asarray(Xmat.sum(axis=1)).ravel().astype(float)
        self.sf = sf / np.median(np.clip(sf, 1e-12, None))
        self.offset = np.log(self.sf + 1e-12)

        # ----- design & penalty -----
        B, basis_info = bspline_basis(
            self.t_filled,
            df=int(self.df),
            degree=int(self.degree),
            include_intercept=bool(self.include_intercept),
        )  # (n, p)
        if not np.isfinite(B).all():
            raise ValueError("B-spline basis contains non-finite values after sanitization.")
        self.X = B
        self.p = int(B.shape[1])
        self.S = float(self.lam) * penalty_matrix(self.p, order=2)
        self._counts_layer = self.counts_layer

        self.lineage_slices = [slice(0, self.p)]  # single-trajectory slice

        # metadata
        meta_basis = {
            "df": int(self.df),
            "degree": int(self.degree),
            "include_intercept": bool(self.include_intercept),
            "p": self.p,
            "basis_info": basis_info,
        }
        if self.nonfinite != "error":
            meta_basis["t_median_fill"] = float(np.median(self.t_filled))
        self.adata.uns[self.key] = {
            "pseudotime_key": self.pseudotime_key,
            "size_factors_key": self.size_factors_key,
            "counts_layer": self.counts_layer,
            "basis": meta_basis,
            "lambda": float(self.lam),
            "nonfinite": self.nonfinite,
            "backend": self.backend,
            "backend_kwargs": dict(self.backend_kwargs),
        }

    # -------- fit / predict ----------
    def fit(
        self,
        genes: Optional[Sequence[Union[str, int]]] = None,
        *,
        store_cov: bool = False,
    ) -> None:
        """Fit per gene with the selected backend and write results to AnnData.

        Parameters
        ----------
        genes : Optional[Sequence[Union[str, int]]], optional
            Genes to fit. If None, fit all genes in adata.var_names (default: None).
        store_cov : bool, optional
            Whether to store the covariance matrix of the estimated coefficients in adata.varm
            under key f"{self.key}_cov". This is only applicable for the IRLS backend
            (default: False).

        Raises
        ------
        ValueError
            If the backend is not recognized or if the gene indices do not match.

        Notes
        -----
        This method fits a penalized weighted least squares model for each gene using the specified backend.
        The results are stored in the AnnData object under the specified keys.
        """
        adata = self.adata
        X, S, offset = self.X, self.S, self.offset
        obs_w = self._obs_weight_mask
        n, p = X.shape

        # resolve gene indices
        if genes is None:
            idx = np.arange(adata.n_vars, dtype=int)
        else:
            idx = np.asarray(
                [
                    g if isinstance(g, int) else int(np.where(adata.var_names == g)[0][0])
                    for g in genes
                ],
                dtype=int,
            )

        # outputs
        coef_mat = np.full((adata.n_vars, p), np.nan, dtype=float)
        alpha_vec = np.full(adata.n_vars, np.nan, dtype=float)
        edf_vec = np.full(adata.n_vars, np.nan, dtype=float)
        cov_stack = np.full((adata.n_vars, p, p), np.nan, dtype=float) if store_cov else None

        # choose backend
        if self.backend == "irls":
            bk = IRLSBackend(return_cov=store_cov, **self.backend_kwargs)
            # per-gene fits
            for j in idx:
                y = self._get_counts_col(j).astype(float)
                res = bk.fit(y=y, X=X, S=S, offset=offset, obs_weights=obs_w)
                coef_mat[j] = res.beta
                alpha_vec[j] = res.alpha
                edf_vec[j] = res.edf
                if store_cov and res.cov is not None:
                    cov_stack[j] = res.cov
        else:
            raise ValueError("backend must be 'irls'.")

        # write back
        adata.var[self.key + "_alpha"] = alpha_vec
        adata.var[self.key + "_edf"] = edf_vec
        adata.varm[self.key + "_coef"] = coef_mat
        if store_cov and cov_stack is not None:
            adata.varm[self.key + "_cov"] = cov_stack

    def fitted_values(
        self, gene: Union[str, int], *, type: str = "response", keep_nan: bool = True
    ) -> np.ndarray:
        """Get fitted values for a specific gene.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to get fitted values for.
        type : str, optional
            The type of fitted values to return. Can be "link" or "response" (default: "response").
        keep_nan : bool, optional
            Whether to keep NaN values in the output (default: True).

        Returns
        -------
        np.ndarray
            The fitted values for the specified gene.
        """
        beta = self._get_beta(gene)
        eta = self.offset + self.X @ beta
        out = eta if type == "link" else np.exp(eta)
        if keep_nan and np.any(self._obs_weight_mask == 0.0):
            out = out.copy()
            out[self._obs_weight_mask == 0.0] = np.nan
        return out

    def predict(
        self,
        gene: Union[str, int],
        t_new: Optional[np.ndarray] = None,
        size_factors: Optional[np.ndarray] = None,
        *,
        type: str = "response",
    ) -> np.ndarray:
        """Predict fitted values for a specific gene at new times.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to predict fitted values for.
        t_new : Optional[np.ndarray], optional
            New time points to predict fitted values at (default: None).
        size_factors : Optional[np.ndarray], optional
            Size factors to use for prediction (default: None).
        type : str, optional
            The type of fitted values to return. Can be "link" or "response" (default: "response").

        Returns
        -------
        np.ndarray
            The predicted fitted values for the specified gene at the new time points.
        """
        beta = self._get_beta(gene)

        if t_new is None:
            B = self.X
            off = self.offset
        else:
            t_new = np.asarray(t_new, dtype=float)
            finite = np.isfinite(t_new)
            if finite.sum() == 0:
                raise ValueError("t_new contains no finite values.")
            t_med = float(np.median(t_new[finite]))
            t_new = np.where(finite, t_new, t_med)

            lo, hi = np.quantile(t_new, [0.001, 0.999])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                t_new = np.clip(t_new, lo, hi)
            if float(np.max(t_new) - np.min(t_new)) == 0.0:
                n = t_new.size
                t_new = t_new + np.linspace(-1e-9, 1e-9, n)

            B, _ = bspline_basis(
                t_new,
                df=int(self.df),
                degree=int(self.degree),
                include_intercept=bool(self.include_intercept),
            )
            off = (
                0.0
                if size_factors is None
                else np.log(np.asarray(size_factors, dtype=float) + 1e-12)
            )

        eta = off + B @ beta
        return eta if type == "link" else np.exp(eta)

    # -------- helpers --------
    def _get_counts_col(self, j: int) -> np.ndarray:
        """Get raw counts for gene at index `j` (1D array for cells).

        Parameters
        ----------
        j : int
            The index of the gene to get raw counts for.

        Returns
        -------
        np.ndarray
            The raw counts for the specified gene.
        """
        Xmat = (
            self.adata.layers[self._counts_layer]
            if self._counts_layer
            and (self.adata.layers is not None)
            and (self._counts_layer in self.adata.layers)
            else self.adata.X
        )
        if sparse.issparse(Xmat):
            return np.asarray(Xmat[:, j].toarray()).ravel()
        return np.asarray(Xmat[:, j]).ravel()

    def _get_beta(self, gene: Union[str, int]) -> np.ndarray:
        """Get fitted coefficients for a specific gene.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to get fitted coefficients for.

        Returns
        -------
        np.ndarray
            The fitted coefficients for the specified gene.
        """
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        beta = self.adata.varm[self.key + "_coef"][j]
        if not np.isfinite(beta).all():
            raise ValueError(
                f"No fitted coefficients for gene index/name: {gene}. Did you call .fit()? "
                f"Or this gene was not included in 'genes'."
            )
        return beta

    # --- internal: one-row basis at a given t (sanitized like training t) ---
    def _basis_row(self, t: float) -> np.ndarray:
        """Get a single row of the B-spline design matrix at time `t`.

        Parameters
        ----------
        t : float
            The time point to get the B-spline basis row for.

        Returns
        -------
        np.ndarray
            The B-spline basis row for the specified time point.
        """
        x = np.asarray([t], dtype=float)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            raise ValueError("Provided t is non-finite.")
        # sanitize same way as training
        t_med = float(np.median(self.t_filled))
        x = np.where(finite, x, t_med)
        lo, hi = np.quantile(self.t_filled, [0.001, 0.999])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            x = np.clip(x, lo, hi)
        if float(np.max(x) - np.min(x)) == 0.0:
            x = x + np.array([0.0])  # keeps shape consistent
        B, _ = bspline_basis(
            x,
            df=int(self.df),
            degree=int(self.degree),
            include_intercept=bool(self.include_intercept),
        )
        return B.ravel()  # (p,)

    # --- internal: get beta and a covariance for Wald tests ---
    def _get_beta_cov(self, gene: Union[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Get fitted coefficients and covariance for a specific gene.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to get fitted coefficients and covariance for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The fitted coefficients and covariance for the specified gene.
        """
        beta = self._get_beta(gene)
        # 1) prefer stored covariance (IRLS with store_cov=True)
        cov_key = self.key + "_cov"
        if cov_key in self.adata.varm:
            cov_stack = self.adata.varm[cov_key]
            j = (
                gene
                if isinstance(gene, (int, np.integer))
                else int(np.where(self.adata.var_names == gene)[0][0])
            )
            cov = cov_stack[j]
            if np.isfinite(cov).all():
                return beta, cov

        # 2) otherwise, approximate from final fit using penalized Hessian
        #    cov ≈ (X^T W X + S)^{-1}
        # get alpha_hat for this gene
        alpha = float(
            self.adata.var[self.key + "_alpha"][
                (
                    gene
                    if isinstance(gene, (int, np.integer))
                    else np.where(self.adata.var_names == gene)[0][0]
                )
            ]
        )
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Cannot approximate covariance: invalid alpha for gene.")
        mu = np.exp(self.offset + self.X @ beta)
        w = irls_weights(mu, alpha) * self._obs_weight_mask
        if np.all(w == 0):
            raise ValueError("All observation weights are zero; cannot form covariance.")
        Xw = self.X * np.sqrt(w)[:, None]
        XtWX = Xw.T @ Xw
        A = XtWX + self.S
        # numerical safety ridge
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        return beta, A_inv

    # --------- public Wald tests ---------

    def contrast_test(self, gene: Union[str, int], c: np.ndarray) -> Dict[str, float]:
        """Generic Wald test for H0: c^T beta = 0.

        Returns (statistic, pvalue) with 1 df.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to test the contrast for.
        c : np.ndarray
            Contrast vector (1D array) to test against the fitted coefficients.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the Wald statistic and p-value for the contrast test.

        Raises
        ------
        ValueError
            If the contrast vector length does not match the number of coefficients,
            or if the contrast variance is non-positive or non-finite.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the Wald statistic and p-value for the contrast test.
        """
        beta, cov = self._get_beta_cov(gene)
        c = np.asarray(c, dtype=float).ravel()
        if c.shape[0] != beta.size:
            raise ValueError(f"contrast length {c.size} != number of coefficients {beta.size}")
        denom = float(c @ cov @ c)
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Non-positive or non-finite contrast variance.")
        stat = float((c @ beta) ** 2 / denom)
        p = 1.0 - chi2.cdf(stat, 1)
        return {"statistic": float(stat), "pvalue": float(p)}

    def association_test(
        self, gene: Union[str, int], exclude_intercept: bool = False
    ) -> Dict[str, float]:
        """H0: all smooth coefficients == 0.

        If include_intercept=True and exclude_intercept=True, drops the first column.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to test the association for.
        exclude_intercept : bool, optional
            Whether to exclude the intercept term from the test (default: False).

        Returns
        -------
        Dict[str, float]
            A dictionary containing the Wald statistic and p-value for the association test.
        """
        beta, cov = self._get_beta_cov(gene)
        if self.include_intercept and exclude_intercept:
            idx = np.arange(1, beta.size, dtype=int)
        else:
            idx = np.arange(0, beta.size, dtype=int)
        if idx.size == 0:
            raise ValueError("No coefficients to test.")
        b = beta[idx]
        C = cov[np.ix_(idx, idx)]
        # Wald stat: b^T C^{-1} b with df = len(idx)
        try:
            Cinvb = np.linalg.solve(C, b)
        except np.linalg.LinAlgError:
            Cinvb = np.linalg.lstsq(C, b, rcond=None)[0]
        stat = float(b.T @ Cinvb)
        p = 1.0 - chi2.cdf(stat, idx.size)
        return {"statistic": float(stat), "pvalue": float(p)}

    def start_end_test(
        self,
        gene: Union[str, int],
        *,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        quantile: float = 0.05,
    ) -> Dict[str, float]:
        """H0: f(t_end) - f(t_start) = 0 for the single trajectory.

        If times are omitted, uses empirical quantiles (q, 1-q).

        Parameters
        ----------
        gene : Union[str, int]
            The gene to test the start-end difference for.
        t_start : Optional[float], optional
            Start time for the test. If None, uses the quantile (default: None).
        t_end : Optional[float], optional
            End time for the test. If None, uses the 1-quantile (default: None).
        quantile : float, optional
            Quantile to use for start and end times if not provided (default: 0.05).

        Returns
        -------
        Dict[str, float]
            A dictionary containing the Wald statistic and p-value for the start-end test.
        """
        if t_start is None or t_end is None:
            # quantiles computed on the *filled* training times
            t_start = float(np.quantile(self.t_filled, quantile))
            t_end = float(np.quantile(self.t_filled, 1.0 - quantile))
        B_start = self._basis_row(t_start)  # (p,)
        B_end = self._basis_row(t_end)  # (p,)
        c = B_end - B_start  # (p,)
        return self.contrast_test(gene, c)

    def test_all(
        self,
        genes: Optional[Sequence[Union[str, int]]] = None,
        *,
        test: str = "association",  # "association" | "start_end" | "contrast"
        exclude_intercept: bool = False,  # for association()
        start_q: float = 0.05,  # for start_end()
        end_q: float = 0.95,  # for start_end()
        contrast: Optional[np.ndarray] = None,  # for contrast()
        min_cells: int = 10,  # detection: ≥ this many cells with count > 0
        mcc: str = "fdr_bh",  # multiple testing correction
        return_curve_summaries: bool = True,  # adds CoM, t_peak, mean_fitted
        grid_points: int = 200,  # for curve summaries
    ) -> pd.DataFrame:
        """Run a chosen pseudotime test for all (detected) genes.

        Parameters
        ----------
        genes : Optional[Sequence[Union[str, int]]], optional
            Genes to test. If None, test all genes in adata.var_names (default: None).
        test : str, optional
            The type of test to run. Can be "association", "start_end", or "contrast" (default: "association").
        exclude_intercept : bool, optional
            Whether to exclude the intercept term from the association test (default: False).
        start_q : float, optional
            Quantile to use for the start time in the start-end test (default: 0.05).
        end_q : float, optional
            Quantile to use for the end time in the start-end test (default: 0.95).
        contrast : Optional[np.ndarray], optional
            Contrast vector to use for the contrast test. Must be provided if test is "contrast" (default: None).
        min_cells : int, optional
            Minimum number of cells with counts > 0 to consider a gene detected (default: 10).
        mcc : str, optional
            Method for multiple testing correction. Can be "fdr_bh", "bonferroni", or "holm" (default: "fdr_bh").
        return_curve_summaries : bool, optional
            Whether to return curve summaries for each gene (default: True).
        grid_points : int, optional
            Number of points in the dense grid for curve summaries (default: 200).

        Raises
        ------
        ValueError
            If the test type is not recognized, or if the contrast vector is not provided for the contrast test.

        Returns
        -------
        pd.DataFrame
            A tidy DataFrame containing the results of the pseudotime tests for all detected genes.
        """
        # --- resolve gene indices ---
        adata = self.adata
        if genes is None:
            idx = np.arange(adata.n_vars, dtype=int)
        else:
            idx = np.asarray(
                [
                    g if isinstance(g, int) else int(np.where(adata.var_names == g)[0][0])
                    for g in genes
                ],
                dtype=int,
            )
        gene_names = adata.var_names.to_numpy()

        # --- detection filter ---
        detected = []
        n_detected = []
        for j in idx:
            y = self._get_counts_col(j)
            n_det = int(np.sum(y > 0))
            if n_det >= int(min_cells):
                detected.append(j)
                n_detected.append(n_det)
        if len(detected) == 0:
            return pd.DataFrame(
                columns=[
                    "gene",
                    "statistic",
                    "pvalue",
                    "qvalue",
                    "n_detected",
                    "edf",
                    "alpha",
                    "center_of_mass",
                    "t_peak",
                    "mean_fitted",
                ]
            )

        # --- run tests ---
        stats = []
        pvals = []
        edf = self.adata.var.get(self.key + "_edf", np.full(adata.n_vars, np.nan)).to_numpy()
        alpha = self.adata.var.get(self.key + "_alpha", np.full(adata.n_vars, np.nan)).to_numpy()

        CoM = []
        Tpeak = []
        MeanFit = []

        for j in detected:
            # run the selected test
            if test == "association":
                out = self.association_test(j, exclude_intercept=exclude_intercept)
            elif test == "start_end":
                # use quantiles on training times
                t_start = float(np.quantile(self.t_filled, start_q))
                t_end = float(np.quantile(self.t_filled, end_q))
                out = self.start_end_test(j, t_start=t_start, t_end=t_end, quantile=start_q)
            elif test == "contrast":
                if contrast is None:
                    raise ValueError("contrast must be provided for test='contrast'.")
                out = self.contrast_test(j, contrast)
            else:
                raise ValueError("test must be 'association', 'start_end', or 'contrast'.")

            stats.append(out["statistic"])
            pvals.append(out["pvalue"])

            # curve summaries on response scale
            if return_curve_summaries:
                tg, yg = _dense_curve(self, j, n_grid=grid_points)
                CoM.append(_center_of_mass(tg, yg))
                Tpeak.append(_peak_time(tg, yg))
                MeanFit.append(float(np.nanmean(yg)))
            else:
                CoM.append(np.nan)
                Tpeak.append(np.nan)
                MeanFit.append(np.nan)

        # --- multiple testing correction ---
        if mcc == "fdr_bh":
            qvals = _bh_fdr(pvals)
        elif mcc in {"bonferroni", "holm"}:
            # quick simple alternatives
            p = np.asarray(pvals, float)
            m = np.isfinite(p).sum()
            if mcc == "bonferroni":
                qvals = np.clip(p * m, 0.0, 1.0)
            else:  # Holm step-down (conservative, simple implementation)
                order = np.argsort(np.where(np.isfinite(p), p, np.inf))
                q = np.full_like(p, np.nan, float)
                k = 1
                for r in order:
                    if np.isfinite(p[r]):
                        q[r] = min((m - k + 1) * p[r], 1.0)
                        k += 1
                # monotone adjustment from small to large p
                for i in range(1, len(order)):
                    a, b = order[i - 1], order[i]
                    if np.isfinite(q[a]) and np.isfinite(q[b]):
                        q[b] = max(q[b], q[a])
                qvals = q
        else:
            raise ValueError("Unsupported mcc. Use 'fdr_bh', 'bonferroni', or 'holm'.")

        # --- pack results ---
        df = (
            pd.DataFrame(
                {
                    "gene": gene_names[np.array(detected, dtype=int)],
                    "statistic": np.asarray(stats, float),
                    "pvalue": np.asarray(pvals, float),
                    "qvalue": np.asarray(qvals, float),
                    "n_detected": np.asarray(n_detected, int),
                    "edf": edf[np.array(detected, dtype=int)],
                    "alpha": alpha[np.array(detected, dtype=int)],
                    "center_of_mass": np.asarray(CoM, float),
                    "t_peak": np.asarray(Tpeak, float),
                    "mean_fitted": np.asarray(MeanFit, float),
                }
            )
            .sort_values("qvalue", kind="mergesort", na_position="last")
            .reset_index(drop=True)
        )

        return df


# Standalone function for PseudotimeGAM
def fit_gam(
    adata: ad.AnnData,
    counts_layer: Optional[str] = "counts",
    pseudotime_key: str = "dpt_pseudotime",
    size_factors_key: str = "size_factors",
    df: int = 6,
    degree: int = 3,
    lam: float = 1.0,
    include_intercept: bool = False,
    key: str = "nbgam1d",
    nonfinite: str = "error",
    backend: str = "irls",
    **kwargs,
) -> PseudotimeGAM:
    """Fit a PseudotimeGAM model to the given AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing the data.
    counts_layer : Optional[str], optional
        The layer in adata containing the raw counts (default: "counts").
    pseudotime_key : str, optional
        The key in adata.obs containing the pseudotime values (default: "dpt_pseudotime").
    size_factors_key : str, optional
        The key in adata.obs containing size factors (default: "size_factors").
    df : int, optional
        Degrees of freedom for the B-spline basis (default: 6).
    degree : int, optional
        Degree of the B-spline (default: 3).
    lam : float, optional
        Regularization parameter (default: 1.0).
    include_intercept : bool, optional
        Whether to include an intercept term in the model (default: False).
    key : str, optional
        Key prefix for storing results in adata.var (default: "nbgam1d").
    nonfinite : str, optional
        How to handle non-finite pseudotime values: "error", "mask", or "median" (default: "error").
    backend : str, optional
        The backend to use for fitting. Currently only "irls" is supported (default: "irls").
    **kwargs : Any
        Additional keyword arguments for the backend.

    Returns
    -------
    PseudotimeGAM
        The fitted PseudotimeGAM model.
    """
    model = PseudotimeGAM(
        adata=adata,
        counts_layer=counts_layer,
        pseudotime_key=pseudotime_key,
        size_factors_key=size_factors_key,
        df=df,
        degree=degree,
        lam=lam,
        include_intercept=include_intercept,
        key=key,
        nonfinite=nonfinite,
        backend=backend,
        backend_kwargs=kwargs,
    )
    model.fit()
    return model
