"""Functionality to fit a Negative Binomial Generalized Additive (Mixture) Models (NB-GAMM) to pseudotime data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import anndata as ad
import mssm
import numpy as np
import pandas as pd
from mssm.src.python.exp_fam import LOG
from scipy import sparse
from scipy.stats import chi2

from .fam import NegBin
from .utils import (
    _bh_fdr,
    _center_of_mass,
    _dense_curve,
    _neg_binom_deviance,
    _peak_time,
)


@dataclass
class PseudotimeGAM:
    """Model to fit Negative Binomial Generalized Additive Mixture Model (NB-GAMM) to pseudotime data."""

    adata: ad.AnnData = field(repr=False)
    layer: Optional[str] = "counts"
    pseudotime_key: str = "dpt_pseudotime"
    size_factors_key: str = "size_factors"
    df: int = 6
    degree: int = 3
    lam: float = (
        1.0  # smoothing multiplier (passed to family as scale or to GAMM via lambda handling)
    )
    include_intercept: bool = False
    key: str = "nbgam1d"
    nonfinite: str = "error"  # "error" | "mask" | "median"
    nb_alpha: Optional[float] = None  # if None -> per-gene method-of-moments

    # derived
    t: np.ndarray = field(init=False, repr=False)
    t_filled: np.ndarray = field(init=False, repr=False)
    _obs_weight_mask: np.ndarray = field(init=False, repr=False)
    sf: np.ndarray = field(init=False, repr=False)
    offset: np.ndarray = field(init=False, repr=False)
    _results_: dict = field(init=False, repr=False, default_factory=dict)

    # ----------------------- construction -----------------------
    def __post_init__(self) -> None:
        """Initialize the PseudotimeGAM object by validating inputs and constructing the model."""
        if self.pseudotime_key not in self.adata.obs:
            raise ValueError(
                f"Expected a single pseudotime column '{self.pseudotime_key}' in .obs."
            )

        # pseudotime clean-up (keep your logic)
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

        # clamp + tiny jitter to stabilize basis placement
        lo, hi = np.quantile(t_clean, [0.001, 0.999])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            t_clean = np.clip(t_clean, lo, hi)
        if float(np.max(t_clean) - np.min(t_clean)) == 0.0:
            n = t_clean.size
            t_clean = t_clean + np.linspace(-1e-9, 1e-9, n)

        self.t = t_raw
        self.t_filled = t_clean
        self._obs_weight_mask = mask_w

        # size factors & offset on log-scale
        if self.size_factors_key in self.adata.obs:
            sf = self.adata.obs[self.size_factors_key].to_numpy(dtype=float)
        elif "n_counts" in self.adata.obs:
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

        # metadata in .uns
        self.adata.uns[self.key] = {
            "backend": "mssm_gamm",
            "pseudotime_key": self.pseudotime_key,
            "size_factors_key": self.size_factors_key,
            "layer": self.layer,
            "basis": {
                "df": int(self.df),
                "degree": int(self.degree),
                "include_intercept": bool(self.include_intercept),
            },
            "lambda": float(self.lam),
            "nonfinite": self.nonfinite,
        }

    # ----------------------- util -----------------------
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
            self.adata.layers[self.layer]
            if self.layer and (self.adata.layers is not None) and (self.layer in self.adata.layers)
            else self.adata.X
        )
        if sparse.issparse(Xmat):
            return np.asarray(Xmat[:, j].toarray()).ravel()
        return np.asarray(Xmat[:, j]).ravel()

    # ----------------------- fitting -----------------------
    def fit(
        self, genes: Optional[Sequence[Union[str, int]]] = None, *, store_cov: bool = False
    ) -> None:
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
        # NOTE: with mssm we don’t precompute spline basis here; we store betas per gene
        coef_list = [None] * adata.n_vars
        edf_vec = np.full(
            adata.n_vars, np.nan, float
        )  # mssm provides edf via fit info; keep NaN if not extracted
        alpha_vec = np.full(adata.n_vars, np.nan, float)
        cov_stack = (
            np.full((adata.n_vars, 1, 1), np.nan, float) if store_cov else None
        )  # filled later per gene
        diagnostic_vec = np.full(adata.n_vars, np.nan, float)

        self._results_ = {}

        # data frame shared columns
        base_df = pd.DataFrame(
            {
                self.pseudotime_key: self.t_filled,
                "offset": self.offset,  # we’ll add to eta manually
            }
        )

        for j in idx:
            y = self._get_counts_col(j).astype(float)
            fam = NegBin(
                link=LOG(),
                scale=float(self.lam),
            )

            data = base_df.copy()
            data["y"] = y

            # --- 2) Build the GAM: y ~ intercept + s(time), NB(log) ---
            formula = mssm.models.Formula(
                lhs=mssm.models.lhs("y"),
                terms=[
                    mssm.models.i(),  # intercept
                    mssm.models.f([self.pseudotime_key]),  # smooth f(time)
                ],
                data=data,
            )
            model = mssm.models.GAMM(formula, fam)
            model.fit(progress_bar=False)

            # store whole model to predict later
            self._results_[int(j)] = model

            # coefficients (marginal) –  smooth part only (formula has just one smooth)
            coef_list[j] = np.asarray(model.coef).astype(float)

            # edf: available via solver utils; if not exposed, leave NaN
            try:
                # model.fit_info may expose edf per term; if not, skip
                edf_vec[j] = float(getattr(model, "edf", np.nan))
            except Exception:
                edf_vec[j] = np.nan

            alpha_vec[j] = float(fam.alpha) if hasattr(fam, "alpha") else np.nan

            # quick diagnostic: predictions exist and finite
            mu = self.fitted_values(j)
            diagnostic_vec[j] = float(np.isfinite(mu).all())

            # covariance (optional): from model.lvi (lower-tri Cholesky of V⁻¹?) -> Vβ = LVIᵀ LVI
            if store_cov and hasattr(model, "lvi"):
                LVI = (
                    model.lvi.toarray() if hasattr(model.lvi, "toarray") else np.asarray(model.lvi)
                )
                Vbeta = LVI.T @ LVI
                cov_stack[j] = Vbeta

        # write back
        # pack coef as a matrix; pad to common length by NaN (mssm keeps fixed by df/degree so length is constant)
        max_p = max((len(c) for c in coef_list if c is not None), default=0)
        coef_mat = np.full((adata.n_vars, max_p), np.nan, float)
        for g, c in enumerate(coef_list):
            if c is not None:
                coef_mat[g, : len(c)] = np.ravel(c)

        adata.varm[self.key + "_coef"] = coef_mat
        adata.var[self.key + "_edf"] = edf_vec
        adata.var[self.key + "_alpha"] = alpha_vec
        if store_cov and cov_stack is not None:
            adata.varm[self.key + "_cov"] = cov_stack
        adata.var[self.key + "_diagnostics"] = diagnostic_vec

    # ----------------------- fitted & predict -----------------------
    def _predict_eta_and_ci(self, model: "mssm.models.GAMM", new_df: pd.DataFrame, z: float = 1.96):
        """mssm-based pointwise CI on η, then add offset and inverse-link.

        Parameters
        ----------
        model : mssm.models.GAMM
            The fitted GAMM model.
        new_df : pd.DataFrame
            New data frame with the same structure as the training data.
        z : float, optional
            Z-score for the confidence interval (default: 1.96).
        """
        # design and beta via mssm
        _, Xnew, _ = model.predict(
            n_dat=new_df, use_terms=None
        )  # returns (pred, X, terms) per docs
        X = Xnew.toarray() if hasattr(Xnew, "toarray") else np.asarray(Xnew)
        beta = model.coef.reshape(-1, 1)
        eta_hat = (X @ beta).ravel()

        # coefficient covariance
        LVI = model.lvi.toarray() if hasattr(model.lvi, "toarray") else np.asarray(model.lvi)
        Vbeta = LVI.T @ LVI
        XV = X @ Vbeta
        var_eta = np.maximum(np.sum(XV * X, axis=1), 0.0)
        se_eta = np.sqrt(var_eta)

        # add offset to η and return CIs on μ = exp(η)
        eta_hat += new_df["offset"].to_numpy()
        lower = eta_hat - z * se_eta
        upper = eta_hat + z * se_eta
        mu = np.exp(eta_hat)
        return mu, np.exp(lower), np.exp(upper)

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
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        if int(j) not in self._results_:
            raise RuntimeError("No fitted model for this gene. Call .fit() first.")
        model = self._results_[int(j)]

        new_df = pd.DataFrame({self.pseudotime_key: self.t_filled, "offset": self.offset})
        mu, _, _ = self._predict_eta_and_ci(model, new_df, z=0.0)
        out = mu if type == "response" else np.log(mu)
        if keep_nan and np.any(self._obs_weight_mask == 0.0):
            out = out.copy()
            out[self._obs_weight_mask == 0.0] = np.nan
        return out

    def predict(
        self, gene: Union[str, int], t_new: Optional[np.ndarray] = None, *, return_ci: bool = False
    ):
        """Predict at new pseudotime values for a specific gene.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to predict for.
        t_new : Optional[np.ndarray], optional
            New pseudotime values to predict at (default: None).
        return_ci : bool, optional
            Whether to return confidence intervals (default: False).

        Returns
        -------
        np.ndarray
            The predicted values for the specified gene at the new pseudotime.
        """
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        if int(j) not in self._results_:
            raise RuntimeError("No fitted model for this gene. Call .fit() first.")
        model = self._results_[int(j)]

        x_new = np.asarray(self.t_filled if t_new is None else t_new, float)
        finite = np.isfinite(x_new)
        if not finite.any():
            raise ValueError("t_new contains no finite values.")
        t_med = float(np.median(x_new[finite]))
        x_new = np.where(finite, x_new, t_med)

        new_df = pd.DataFrame(
            {
                self.pseudotime_key: x_new,
                "offset": (
                    np.full_like(x_new, np.median(self.offset), dtype=float)
                    if t_new is not None
                    else self.offset
                ),
            }
        )
        if return_ci:
            mean, lower, upper = self._predict_eta_and_ci(model, new_df, z=1.96)
            return mean, lower, upper
        mean, _, _ = self._predict_eta_and_ci(model, new_df, z=0.0)
        return mean

    # ----------------------- model quality -----------------------
    def _deviance_explained(self, gene: Union[str, int]) -> float:
        """Calculate the proportion of deviance explained by the fitted model for a gene.

        Parameters
        ----------
        gene : str
            The gene name for which to calculate deviance explained.

        Returns
        -------
        float
            Proportion of deviance explained by the model for the specified gene.
        """
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        y = self._get_counts_col(j)
        mu_fit = self.fitted_values(j)
        mu_null = np.full_like(y, fill_value=y.mean())
        alpha = float(self.adata.var[self.key + "_alpha"].iloc[j])
        dev_resid = _neg_binom_deviance(y, mu_fit, alpha)
        dev_null = _neg_binom_deviance(y, mu_null, alpha)
        return 1 - dev_resid / dev_null

    def deviance_explained(self) -> pd.Series:
        """Calculate the proportion of deviance explained by the fitted model for all fitted genes.

        Returns
        -------
        float
            Proportion of deviance explained by the model for all fitted genes.
        """
        return pd.DataFrame(
            {
                "gene": self.adata.var_names,
                "deviance_explained": [self._deviance_explained(g) for g in self.adata.var_names],
            }
        ).set_index("gene")["deviance_explained"]

    # --------- covariance for Wald tests (direct from mssm) ---------
    def _get_beta_cov(self, gene: Union[str, int]):
        """Return (beta, cov) for Wald tests.

        Parameters
        ----------
        gene : Union[str, int]
            The gene to get fitted coefficients and covariance for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The fitted coefficients and covariance matrix for the specified gene.
        """
        j = (
            gene
            if isinstance(gene, (int, np.integer))
            else int(np.where(self.adata.var_names == gene)[0][0])
        )
        beta = self.adata.varm[self.key + "_coef"][j]
        if not np.isfinite(beta).all():
            raise ValueError(
                f"No fitted coefficients for gene {gene}. Did you call .fit()? Or this gene was not included."
            )
        model = self._results_[int(j)]
        LVI = model.lvi.toarray() if hasattr(model.lvi, "toarray") else np.asarray(model.lvi)
        Vbeta = LVI.T @ LVI
        return beta, Vbeta

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
    lam: float = 0.01,
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
