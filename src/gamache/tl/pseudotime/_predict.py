"""Prediction helpers for `PseudotimeGAM` (split out for browseability)."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class _PredictMixin:
    """Mixin providing `.fitted_values()` and `.predict()`."""

    def fitted_values(
        self, gene: Union[str, int], *, type: str = "response", keep_nan: bool = True
    ) -> np.ndarray:
        """Obtain fitted values for a gene on the original data. Requires that `.fit()` has been called.

        :param gene: Gene for which to obtain fitted values
        :type gene: Union[str, int]
        :param type: Type of fitted values to return ("response" or "link"), defaults to "response"
        :type type: str, optional
        :param keep_nan: Kept for compatibility; ignored (pseudotime is fully finite).
        :type keep_nan: bool, optional
        :return: Fitted values for the specified gene
        :rtype: np.ndarray
        """

        # Check that the model has been fitted for this gene
        try:
            _ = self._get_beta(gene)
        except Exception as e:
            raise RuntimeError("No fitted model for this gene. Call .fit() first.") from e
        
        # Get the fitted coefficients for the gene
        beta = self._get_beta(gene)
        
        # Compute the linear predictor (eta) and the fitted values on the response scale
        eta = self.X @ beta
        out = eta if type == "link" else np.exp(np.clip(eta, -50.0, 50.0))
        return out

    def predict(
        self,
        gene: Union[str, int],
        t_new: Optional[np.ndarray] = None,
        obs_new: Optional[pd.DataFrame] = None,
        *,
        return_ci: bool = False,
    ):
        """Predict fitted values for a gene at new pseudotime points or new observations. Requires that `.fit()` has been called.

        :param gene: Gene for which to obtain predictions
        :type gene: Union[str, int]
        :param t_new: New pseudotime points for prediction, defaults to None
        :type t_new: Optional[np.ndarray], optional
        :param obs_new: New observations for prediction, defaults to None
        :type obs_new: Optional[pd.DataFrame], optional
        :param return_ci: Whether to return confidence intervals, defaults to False
        :type return_ci: bool, optional
        :raises RuntimeError: If the model has not been fitted for the specified gene
        :raises ValueError: If neither `t_new` nor `obs_new` is provided
        :raises ValueError: If `t_new` contains no finite values
        :return: Predicted values (and optionally confidence intervals) for the specified gene
        :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        """
        try:
            _ = self._get_beta(gene)
        except Exception as e:
            raise RuntimeError("No fitted model for this gene. Call .fit() first.") from e

        # Determine covariate names for prediction
        covs = [] if self.covariates is None else [str(c) for c in self.covariates]

        # Check if we are predicting at the original data points (no new t or obs provided)
        if t_new is None and obs_new is None:
            B = self.X
            eta = B @ self._get_beta(gene)
            mean = np.exp(np.clip(eta, -50.0, 50.0))
            if not return_ci:
                return mean
            _, cov = self._get_beta_cov(gene)
            var_eta = np.einsum("ij,jk,ik->i", B, cov, B)
            se_eta = np.sqrt(np.clip(var_eta, 0.0, np.inf))
            z = 1.959963984540054
            lower = np.exp(np.clip(eta - z * se_eta, -50.0, 50.0))
            upper = np.exp(np.clip(eta + z * se_eta, -50.0, 50.0))
            return mean, lower, upper

        # Prepare the new data for prediction
        if t_new is None:
            if obs_new is None:
                raise ValueError("Provide either t_new or obs_new.")
            x_new = np.asarray(obs_new.get(self.pseudotime_key, np.nan), dtype=float).reshape(-1)
        else:
            x_new = np.asarray(t_new, dtype=float).reshape(-1)

        # Require pseudotime values to be finite
        if not np.isfinite(x_new).all():
            raise ValueError("t_new contains non-finite values (NaN/±inf).")
        x_new = np.clip(x_new, float(self._tmin), float(self._tmax))

        # Construct the design matrix for the new data using the model formula, and make predictions
        if obs_new is None:
            frame = pd.DataFrame({self._pt_sanitized_key: x_new})
        else:
            frame = obs_new.copy()
            frame[self._pt_sanitized_key] = x_new

        # Ensure that all covariates expected by the model are present in the new data, filling with default values if necessary
        for c in covs:
            if c not in frame:
                frame[c] = self._covariate_defaults.get(c, 0.0)

        _y, B, _notNA, *_rest = self._formula.encode_data(frame, prediction=True)
        B = np.asarray(B, dtype=float)
        beta = self._get_beta(gene)
        eta = B @ beta
        mean = np.exp(np.clip(eta, -50.0, 50.0))
        if not return_ci:
            return mean

        _, cov = self._get_beta_cov(gene)
        var_eta = np.einsum("ij,jk,ik->i", B, cov, B)
        se_eta = np.sqrt(np.clip(var_eta, 0.0, np.inf))
        z = 1.959963984540054
        lower = np.exp(np.clip(eta - z * se_eta, -50.0, 50.0))
        upper = np.exp(np.clip(eta + z * se_eta, -50.0, 50.0))
        return mean, lower, upper
