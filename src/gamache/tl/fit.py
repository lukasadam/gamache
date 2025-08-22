"""Functionality to fit a single-trajectory Negative Binomial Generalized Additive Model (NB-GAM) to pseudotime data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import chi2
from statsmodels.gam.api import BSplines, GLMGam

from .utils import (
    _bh_fdr,
    _center_of_mass,
    _dense_curve,
    _neg_binom_deviance,
    _peak_time,
)


@dataclass
class PseudotimeGAM:
    """Model to fit a single-trajectory Negative Binomial Generalized Additive Model (NB-GAM) to pseudotime data."""

    # config / inputs
    adata: ad.AnnData = field(repr=False)
    layer: Optional[str] = "counts"
    pseudotime_key: str = "dpt_pseudotime"
    size_factors_key: str = "size_factors"
    df: int = 6
    degree: int = 3
    lam: float = 1.0
    include_intercept: bool = False
    key: str = "nbgam1d"
    nonfinite: str = "error"  # "error" | "mask" | "median"
    nb_alpha: Optional[float] = None  # if None, estimate per gene via method-of-moments

    # derived (built in __post_init__)
    t: np.ndarray = field(init=False, repr=False)
    t_filled: np.ndarray = field(init=False, repr=False)
    _obs_weight_mask: np.ndarray = field(init=False, repr=False)
    sf: np.ndarray = field(init=False, repr=False)
    offset: np.ndarray = field(init=False, repr=False)
    smoother: BSplines = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)  # (n, p) spline design used by statsmodels
    p: int = field(init=False)
    _layer: Optional[str] = field(init=False, repr=False)

    # ----------------------- construction -----------------------
    def __post_init__(self) -> None:
        """Initialize the PseudotimeGAM object by validating inputs and constructing the model."""
        if self.pseudotime_key not in self.adata.obs:
            raise ValueError(
                f"Expected a single pseudotime column '{self.pseudotime_key}' in .obs."
            )
        # Store pseudotime as np.ndarray, sanitize non-finite values
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

        # size factors & offset
        if self.size_factors_key in self.adata.obs:
            sf = self.adata.obs[self.size_factors_key].to_numpy(dtype=float)
        else:
            if "n_counts" in self.adata.obs:
                sf = self.adata.obs["n_counts"].to_numpy(dtype=float)
            else:
                Xmat = (
                    self.adata.layers[self.layer]
                    if self.layer
                    and (self.adata.layers is not None)
                    and (self.layer in self.adata.layers)
                    else self.adata.X
                )
                sf = np.asarray(Xmat.sum(axis=1)).ravel().astype(float)
        self.sf = sf / np.median(np.clip(sf, 1e-12, None))
        self.offset = np.log(self.sf + 1e-12)

        # B-spline basis
        self.smoother = BSplines(
            self.t_filled[:, None],
            df=[int(self.df)],
            degree=[int(self.degree)],
            include_intercept=bool(self.include_intercept),
        )
        self.X = np.asarray(self.smoother.basis)  # (n, p)
        if not np.isfinite(self.X).all():
            raise ValueError("B-spline basis contains non-finite values after sanitization.")
        self.p = int(self.smoother.dim_basis)
        self._layer = self.layer

        # metadata
        meta_basis = {
            "df": int(self.df),
            "degree": int(self.degree),
            "include_intercept": bool(self.include_intercept),
            "p": self.p,
        }
        if self.nonfinite != "error":
            meta_basis["t_median_fill"] = float(np.median(self.t_filled))
        self.adata.uns[self.key] = {
            "pseudotime_key": self.pseudotime_key,
            "size_factors_key": self.size_factors_key,
            "layer": self.layer,
            "basis": meta_basis,
            "lambda": float(self.lam),
            "nonfinite": self.nonfinite,
            "backend": "statsmodels_glmgam",
        }

    # ----------------------- fitting & prediction -----------------------
    @staticmethod
    def _alpha_mom(y: np.ndarray) -> float:
        """Method-of-moments NB2 dispersion (alpha) with floor.

        Parameters
        ----------
        y
            (n,) array of counts for a single gene

        Returns
        -------
        alpha
            estimated dispersion parameter (≥ 1e-10)
        """
        m = float(np.mean(y))
        v = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
        if m <= 0.0:
            return 1.0
        return float(max(1e-10, (v - m) / (m**2 + 1e-12)))

    def fit(self, genes=None, *, store_cov: bool = False) -> None:
        """Fit per gene and write results back to AnnData.

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
        This method uses the implementation from statsmodels to fit a penalized weighted least squares model
        for each gene. The results are stored in the AnnData object under the specified keys.
        """
        adata = self.adata

        # build smoother on training pseudotime
        x = np.asarray(self.t_filled, float)
        self._tmin, self._tmax = float(np.min(x)), float(np.max(x))
        self.smoother = BSplines(x[:, None], df=[int(self.df)], degree=[int(self.degree)])

        n = x.size
        p = int(self.smoother.dim_basis)
        exog_lin = np.ones((n, 1))  # intercept

        # gene indices
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
        coef_mat = np.full((adata.n_vars, p), np.nan, float)
        edf_vec = np.full(adata.n_vars, np.nan, float)
        alpha_vec = np.full(adata.n_vars, 1.0, float)  # fixed NB alpha
        cov_stack = np.full((adata.n_vars, p, p), np.nan, float) if store_cov else None
        diagnostic_vec = np.full(adata.n_vars, np.nan, float)

        self._results_ = {}

        for j in idx:
            y = self._get_counts_col(j).astype(float)

            model = GLMGam(
                y,
                exog=exog_lin,
                smoother=self.smoother,
                family=sm.families.NegativeBinomial(alpha=1.0),
                # uncomment if you want library-size normalization:
                # offset=self.offset,
            )
            res = model.fit()
            self._results_[int(j)] = res

            k_lin = int(model.k_exog_linear)  # = 1
            params = np.asarray(res.params)
            coef_mat[j] = params[k_lin : k_lin + p]

            try:
                edf_vec[j] = float(np.nansum(np.asarray(res.edf)[k_lin : k_lin + p]))
            except Exception:
                edf_vec[j] = np.nan

            if store_cov:
                C = np.asarray(res.cov_params())
                cov_stack[j] = C[k_lin : k_lin + p, k_lin : k_lin + p]

            diagnostic_vec[j] = float(np.isfinite(res.fittedvalues).all())

        # write back
        adata.varm[self.key + "_coef"] = coef_mat
        adata.var[self.key + "_edf"] = edf_vec
        adata.var[self.key + "_alpha"] = alpha_vec
        if store_cov and cov_stack is not None:
            adata.varm[self.key + "_cov"] = cov_stack
        adata.var[self.key + "_diagnostics"] = diagnostic_vec

    def fitted_values(
        self, gene: Union[str, int], *, type: str = "response", keep_nan: bool = True
    ) -> np.ndarray:
        """Get fitted values for a specific gene at the training pseudotime.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to get fitted values for.
        type : str, optional
            The type of fitted values to return ("link" or "response", default: "response").
        keep_nan : bool, optional
            Whether to keep NaN values in the output (default: True).

        Returns
        -------
        np.ndarray
            The fitted values for the specified gene at the training pseudotime.
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
        *,
        return_ci: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Predict fitted values for a specific gene at new times.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to predict fitted values for.
        t_new : Optional[np.ndarray], optional
            New time points to predict fitted values at (default: None).
        return_ci : bool, optional
            Whether to return 95% confidence intervals along with the mean (default: False).

        Returns
        -------
        np.ndarray
            The predicted mean (and optional CI) for the specified gene at the new time points.
        """
        # resolve gene index
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        if not hasattr(self, "_results_") or int(j) not in self._results_:
            raise RuntimeError("No fitted model for this gene. Call .fit() first.")

        res = self._results_[int(j)]

        # default to training times
        x_new = np.asarray(self.t_filled if t_new is None else t_new, float)

        # handle non-finite
        finite = np.isfinite(x_new)
        if not finite.any():
            raise ValueError("t_new contains no finite values.")
        t_med = float(np.median(x_new[finite]))
        x_new = np.where(finite, x_new, t_med)

        # clamp strictly inside [tmin, tmax] to satisfy BSplines.transform
        tmin, tmax = float(self._tmin), float(self._tmax)
        # nudge to open interval to be extra safe
        left = np.nextafter(tmin, np.inf)
        right = np.nextafter(tmax, -np.inf)
        x_new = np.clip(x_new, left, right)

        exog_lin = np.ones((x_new.size, 1))
        exog_spl = x_new[:, None]

        pred = res.get_prediction(exog=exog_lin, exog_smooth=exog_spl)
        sf = pred.summary_frame()  # mean, mean_se, mean_ci_lower, mean_ci_upper

        mean = np.asarray(sf["mean"], float)
        if return_ci:
            lower = np.asarray(sf["mean_ci_lower"], float)
            upper = np.asarray(sf["mean_ci_upper"], float)
            return mean, lower, upper
        return mean

    # ----------------------- model quality -----------------------
    def _deviance_explained(self, gene: str) -> float:
        """Calculate the proportion of deviance explained by the fitted model for a gene.

        Parameters
        ----------
        gam : PseudotimeGAM-like object
            Must have attributes:
            - .adata: AnnData object with gene expression data
            - .t_filled: Pseudotime values for each cell
            - .predict(gene, t_new): Predict expression for `gene` at new pseudotime values
            - .fitted_values(gene): Fitted expression values for `gene` at observed pseudotime
            - ._get_counts_col(j): Get raw counts for gene at index `j` (1D array for cells)
        gene : str
            The gene name for which to calculate deviance explained.

        Returns
        -------
        float
            Proportion of deviance explained by the model for the specified gene.
        """
        if isinstance(gene, str):
            gene = int(np.where(self.adata.var_names == gene)[0][0])
        y = self._get_counts_col(gene)
        mu_fit = self.fitted_values(gene)
        mu_null = np.full_like(y, fill_value=y.mean())  # null model = flat curve
        alpha = self.adata.var[self.key + "_alpha"][gene]

        dev_resid = _neg_binom_deviance(y, mu_fit, alpha)
        dev_null = _neg_binom_deviance(y, mu_null, alpha)
        return 1 - dev_resid / dev_null

    def deviance_explained(
        self,
    ) -> float:
        """Calculate the proportion of deviance explained by the fitted model for all fitted genes.

        Returns
        -------
        float
            Proportion of deviance explained by the model for all fitted genes.
        """
        return pd.DataFrame(
            {
                "gene": self.adata.var_names,
                "deviance_explained": [
                    self._deviance_explained(gene) for gene in self.adata.var_names
                ],
            }
        ).set_index("gene")["deviance_explained"]

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
            self.adata.layers[self._layer]
            if self._layer
            and (self.adata.layers is not None)
            and (self._layer in self.adata.layers)
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

    def _basis_row(self, t: float) -> np.ndarray:
        """Get B-spline basis row for a single time point `t`.

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
        # sanitize like training
        t_med = float(np.median(self.t_filled))
        x = np.where(finite, x, t_med)
        lo, hi = np.quantile(self.t_filled, [0.001, 0.999])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            x = np.clip(x, lo, hi)
        B = np.asarray(self.smoother.transform(x[:, None]))
        return B.ravel()  # (p,)

    def _get_beta_cov(self, gene: Union[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (beta, cov) for Wald tests. If cov not stored, refit quickly for that gene.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to get fitted coefficients and covariance for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The fitted coefficients and covariance matrix for the specified gene.
        """
        beta = self._get_beta(gene)
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

        # fallback: refit just this gene to get covariance
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        y = self._get_counts_col(j).astype(float)
        nb_alpha = float(self.adata.var[self.key + "_alpha"][j])
        fam = sm.families.NegativeBinomial(
            alpha=nb_alpha if np.isfinite(nb_alpha) and nb_alpha > 0 else self._alpha_mom(y)
        )
        mod = GLMGam(
            y,
            exog=None,
            smoother=self.smoother,
            alpha=float(self.lam),
            family=fam,
            offset=self.offset,
        )
        weights = self._obs_weight_mask if (self.nonfinite == "mask") else None
        res = mod.fit(weights=weights)
        C = np.asarray(res.cov_params())[: self.p, : self.p]
        return beta, C

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


# ----------------------- convenience wrapper -----------------------
def fit_gam(
    adata: ad.AnnData,
    layer: Optional[str] = "counts",
    pseudotime_key: str = "dpt_pseudotime",
    size_factors_key: str = "size_factors",
    df: int = 6,
    degree: int = 3,
    lam: float = 1.0,
    include_intercept: bool = False,
    key: str = "nbgam1d",
    nonfinite: str = "error",
    nb_alpha: Optional[float] = None,
) -> PseudotimeGAM:
    """Fit a PseudotimeGAM model to the given AnnData using statsmodels GLMGam."""
    model = PseudotimeGAM(
        adata=adata,
        layer=layer,
        pseudotime_key=pseudotime_key,
        size_factors_key=size_factors_key,
        df=df,
        degree=degree,
        lam=lam,
        include_intercept=include_intercept,
        key=key,
        nonfinite=nonfinite,
        nb_alpha=nb_alpha,
    )
    model.fit()
    return model
