"""Tests and summary statistics for `PseudotimeGAM`."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..utils import _bh_fdr, _center_of_mass, _dense_curve, _neg_binom_deviance, _peak_time


class _TestsMixin:
    """Mixin providing hypothesis tests and curve summaries."""

    def _deviance_explained(self, gene: Union[str, int]) -> float:
        if isinstance(gene, str):
            gene = int(np.where(self.adata.var_names == gene)[0][0])
        y = self._get_counts_col(int(gene))
        mu_fit = self.fitted_values(gene)
        mu_null = np.full_like(y, fill_value=y.mean())
        alpha = float(self.adata.var[self.key + "_alpha"].iloc[int(gene)])

        dev_resid = _neg_binom_deviance(y, mu_fit, alpha)
        dev_null = _neg_binom_deviance(y, mu_null, alpha)
        return 1 - dev_resid / dev_null

    def deviance_explained(self) -> pd.Series:
        return pd.DataFrame(
            {
                "gene": self.adata.var_names,
                "deviance_explained": [self._deviance_explained(g) for g in self.adata.var_names],
            }
        ).set_index("gene")["deviance_explained"]

    def contrast_test(self, gene: Union[str, int], c: np.ndarray) -> Dict[str, float]:
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

    def association_test(self, gene: Union[str, int], exclude_intercept: bool = False) -> Dict[str, float]:
        beta, cov = self._get_beta_cov(gene)
        if self.include_intercept and exclude_intercept:
            idx = np.arange(1, beta.size, dtype=int)
        else:
            idx = np.arange(0, beta.size, dtype=int)
        if idx.size == 0:
            raise ValueError("No coefficients to test.")
        b = beta[idx]
        C = cov[np.ix_(idx, idx)]
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
        if t_start is None or t_end is None:
            t_start = float(np.quantile(self.t, quantile))
            t_end = float(np.quantile(self.t, 1.0 - quantile))
        B_start = self._basis_row(float(t_start))
        B_end = self._basis_row(float(t_end))
        c = B_end - B_start
        return self.contrast_test(gene, c)

    def test_all(
        self,
        genes: Optional[Sequence[Union[str, int]]] = None,
        *,
        test: str = "association",
        exclude_intercept: bool = False,
        start_q: float = 0.05,
        end_q: float = 0.95,
        contrast: Optional[np.ndarray] = None,
        min_cells: int = 10,
        mcc: str = "fdr_bh",
        return_curve_summaries: bool = True,
        grid_points: int = 200,
    ) -> pd.DataFrame:
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

        detected: list[int] = []
        n_detected: list[int] = []
        for j in idx:
            y = self._get_counts_col(int(j))
            n_det = int(np.sum(y > 0))
            if n_det >= int(min_cells):
                detected.append(int(j))
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

        stats: list[float] = []
        pvals: list[float] = []
        edf = adata.var.get(self.key + "_edf", np.full(adata.n_vars, np.nan)).to_numpy()
        alpha = adata.var.get(self.key + "_alpha", np.full(adata.n_vars, np.nan)).to_numpy()

        CoM: list[float] = []
        Tpeak: list[float] = []
        MeanFit: list[float] = []

        for j in detected:
            if test == "association":
                out = self.association_test(j, exclude_intercept=exclude_intercept)
            elif test == "start_end":
                t_start = float(np.quantile(self.t, start_q))
                t_end = float(np.quantile(self.t, end_q))
                out = self.start_end_test(j, t_start=t_start, t_end=t_end, quantile=start_q)
            elif test == "contrast":
                if contrast is None:
                    raise ValueError("contrast must be provided for test='contrast'.")
                out = self.contrast_test(j, contrast)
            else:
                raise ValueError("test must be 'association', 'start_end', or 'contrast'.")

            stats.append(float(out["statistic"]))
            pvals.append(float(out["pvalue"]))

            if return_curve_summaries:
                tg, yg = _dense_curve(self, j, n_grid=grid_points)
                CoM.append(float(_center_of_mass(tg, yg)))
                Tpeak.append(float(_peak_time(tg, yg)))
                MeanFit.append(float(np.nanmean(yg)))
            else:
                CoM.append(np.nan)
                Tpeak.append(np.nan)
                MeanFit.append(np.nan)

        if mcc == "fdr_bh":
            qvals = _bh_fdr(pvals)
        elif mcc in {"bonferroni", "holm"}:
            p = np.asarray(pvals, float)
            m = np.isfinite(p).sum()
            if mcc == "bonferroni":
                qvals = np.clip(p * m, 0.0, 1.0)
            else:
                order = np.argsort(np.where(np.isfinite(p), p, np.inf))
                q = np.full_like(p, np.nan, float)
                k = 1
                for r in order:
                    if np.isfinite(p[r]):
                        q[r] = min((m - k + 1) * p[r], 1.0)
                        k += 1
                for i in range(1, len(order)):
                    a, b = order[i - 1], order[i]
                    if np.isfinite(q[a]) and np.isfinite(q[b]):
                        q[b] = max(q[b], q[a])
                qvals = q
        else:
            raise ValueError("Unsupported mcc. Use 'fdr_bh', 'bonferroni', or 'holm'.")

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
