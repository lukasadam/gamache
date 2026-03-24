"""Core data + design-matrix construction for `PseudotimeGAM`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ... import jax as jaxgamx


@dataclass
class _Core:
    """Core implementation for `PseudotimeGAM`."""

    # AnnData config
    adata: ad.AnnData = field(repr=False)
    layer: str | None = "counts"
    pseudotime_key: str = "dpt_pseudotime"
    covariates: tuple[str, ...] | list[str] | None = None

    # GAM config
    df: int = 6
    degree: int = 3
    lam: float = 1.0
    include_intercept: bool = False

    # Additional config
    key: str = "nbgam1d"
    # Kept for backwards compatibility only; non-finite pseudotime handling has
    # been removed. Pseudotime must be fully finite.
    nonfinite: str = "error"
    nb_alpha: float | None = None  # kept for compatibility; ignored by backend

    # NB2 parameterization in jaxgamx: Var(y) = mu + mu^2/theta
    theta: float = 10.0
    irls_maxiter: int = 25
    ridge: float = 1e-8

    # derived
    t: np.ndarray = field(init=False, repr=False)

    # Internal key used for building the backend design matrix.
    # (Historically this held sanitized values; now it is an internal copy.)
    _pt_sanitized_key: str = field(init=False, repr=False)
    _formula: Any = field(init=False, repr=False)
    _backend_family: Any = field(init=False, repr=False)
    _backend_model: Any = field(init=False, repr=False)
    _covariates: tuple[str, ...] = field(init=False, repr=False)
    _covariate_defaults: dict[str, float | str] = field(init=False, repr=False)

    X: np.ndarray = field(init=False, repr=False)
    p: int = field(init=False)
    _fit_mask: np.ndarray = field(init=False, repr=False)
    _tmin: float = field(init=False, repr=False)
    _tmax: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the model by validating inputs, reading pseudotime, constructing the design matrix, and initializing the backend model.
        """
        self._validate_inputs()
        self._covariates = tuple() if self.covariates is None else tuple(map(str, self.covariates))
        self._initialize_pseudotime()
        self._initialize_covariates()
        self._initialize_formula()
        self._initialize_backend_model()
        self._initialize_ranges_and_masks()
        self._write_metadata()

    def _initialize_backend_model(self) -> None:
        """Initialize the JAX backend model based on the constructed formula and specified family.
        """
        self._backend_family = jaxgamx.families.NegativeBinomial(theta=float(self.theta))
        self._backend_model = jaxgamx.models.GAMM(self._formula, self._backend_family)

    def _validate_inputs(self) -> None:
        """Validate user-supplied configuration.

        :raises ValueError: If pseudotime column is not found in `adata.obs`
        :raises ValueError: If nonfinite handling is not set to 'error'
        :raises ValueError: If df is less than 1
        :raises ValueError: If degree is less than 1
        :raises ValueError: If theta is less than or equal to 0
        :raises ValueError: If irls_maxiter is less than 1
        :raises ValueError: If lam is less than 0
        :raises ValueError: If ridge is less than 0
        """
        if self.pseudotime_key not in self.adata.obs:
            raise ValueError(
                f"Expected a pseudotime column '{self.pseudotime_key}' in adata.obs."
            )
        if self.nonfinite != "error":
            raise ValueError(
                "Non-finite pseudotime handling has been removed. "
                "Ensure pseudotime is fully finite before fitting (nonfinite must be 'error')."
            )
        if self.df < 1:
            raise ValueError("df must be >= 1.")
        if self.degree < 1:
            raise ValueError("degree must be >= 1.")
        if self.theta <= 0:
            raise ValueError("theta must be > 0.")
        if self.irls_maxiter < 1:
            raise ValueError("irls_maxiter must be >= 1.")
        if self.lam < 0:
            raise ValueError("lam must be >= 0.")
        if self.ridge < 0:
            raise ValueError("ridge must be >= 0.")

    def _initialize_pseudotime(self) -> None:
        """Read and store pseudotime-related state.

        :raises ValueError: If pseudotime contains non-finite values
        """
        t = self.adata.obs[self.pseudotime_key].to_numpy(dtype=float)
        if t.ndim != 1:
            t = np.asarray(t, dtype=float).reshape(-1)

        n_bad = int((~np.isfinite(t)).sum())
        if n_bad:
            raise ValueError(
                f"Pseudotime contains {n_bad} non-finite values (NaN/±inf). "
                "Please fix your input; this model no longer supports masking/filling pseudotime."
            )

        self.t = np.asarray(t, dtype=float)

        self._pt_sanitized_key = f"__gm_{self.key}_pt"
        self.adata.obs[self._pt_sanitized_key] = np.asarray(self.t, dtype=float)

    def _initialize_covariates(self) -> None:
        """Validate covariates and compute default values for prediction.

        :raises ValueError: If a covariate is not found in `adata.obs`
        """
        for cov in self._covariates:
            if cov not in self.adata.obs:
                raise ValueError(f"Covariate {cov!r} not found in adata.obs")

        self._covariate_defaults = {
            cov: self._default_value_for_covariate(cov) for cov in self._covariates
        }

    def _default_value_for_covariate(self, cov: str) -> float | str:
        """Choose a stable default value for a covariate.

        :param cov: Covariate name
        :type cov: str
        :return: Default value for the covariate
        :rtype: float | str
        """
        values = self.adata.obs[cov]
        arr = np.asarray(values)

        if arr.dtype.kind in {"i", "u", "f"}:
            x = np.asarray(arr, dtype=float)
            finite = np.isfinite(x)
            return float(np.median(x[finite])) if np.any(finite) else 0.0

        obj = np.asarray(values.astype(object))
        for item in obj.reshape(-1):
            if item is None:
                continue
            s = str(item)
            if s != "nan":
                return s
        return ""

    def _initialize_formula(self) -> None:
        """Build backend formula and dense design matrix.

        :raises ValueError: If the design matrix contains non-finite values
        """
        terms: list[Any] = []

        if self.include_intercept:
            terms.append(jaxgamx.models.i())
        if self._covariates:
            terms.append(jaxgamx.models.l(list(self._covariates)))
        terms.append(
            jaxgamx.models.f(
                [self._pt_sanitized_key],
                nk=int(self.df),
                degree=int(self.degree),
            )
        )

        self._formula = jaxgamx.models.Formula(
            lhs=jaxgamx.models.lhs("y"),
            terms=terms,
            data=self.adata,
        )

        self.X = np.asarray(self._formula.Xs[0].todense(), dtype=float)
        if not np.isfinite(self.X).all():
            raise ValueError("Design matrix contains non-finite values after sanitization.")

        self.p = int(self.X.shape[1])

    def _initialize_ranges_and_masks(self) -> None:
        """Initialize fit mask and training-time range.

        With strict finiteness requirements, all observations participate in fitting.
        """
        self._fit_mask = np.ones_like(self.t, dtype=bool)
        self._tmin = float(np.min(self.t))
        self._tmax = float(np.max(self.t))

    def _metadata_dict(self) -> dict[str, Any]:
        """Assemble model metadata stored in `adata.uns[self.key]`.

        :return: Model metadata
        :rtype: dict[str, Any]
        """
        basis = {
            "df": int(self.df),
            "degree": int(self.degree),
            "include_intercept": bool(self.include_intercept),
            "p": int(self.p),
        }
        return {
            "pseudotime_key": self.pseudotime_key,
            "layer": self.layer,
            "covariates": list(self._covariates),
            "basis": basis,
            "lambda": float(self.lam),
            "backend": "jaxgamx",
            "theta": float(self.theta),
        }

    def _write_metadata(self) -> None:
        """Write model metadata to `adata.uns[self.key]` for later reference by tests and prediction.
        """
        self.adata.uns[self.key] = self._metadata_dict()

    def _expression_matrix(self) -> Any:
        """Get the active expression matrix, preferring the requested layer.

        :return: Active expression matrix
        :rtype: Any
        """
        if self.layer and self.adata.layers is not None and self.layer in self.adata.layers:
            return self.adata.layers[self.layer]
        return self.adata.X

    def _gene_index(self, gene: str | int) -> int:
        """Resolve a gene name or integer index to an integer index.

        :param gene: Gene name or index
        :type gene: str | int
        :raises IndexError: If the gene index is out of bounds
        :raises KeyError: If the gene name is not found in `adata.var_names`
        :return: Integer index of the gene
        :rtype: int
        """
        if isinstance(gene, (int, np.integer)):
            j = int(gene)
            if j < 0 or j >= self.adata.n_vars:
                raise IndexError(f"Gene index {j} is out of bounds for n_vars={self.adata.n_vars}.")
            return j

        matches = np.where(self.adata.var_names == gene)[0]
        if len(matches) == 0:
            raise KeyError(f"Gene {gene!r} not found in adata.var_names.")
        return int(matches[0])

    def _get_counts_col(self, j: int) -> np.ndarray:
        """Get the counts column for a gene index, handling sparse/dense matrices.

        :param j: Gene index
        :type j: int
        :return: Counts column for the specified gene
        :rtype: np.ndarray
        """
        Xmat = self._expression_matrix()
        if sparse.issparse(Xmat):
            return np.asarray(Xmat[:, j].toarray()).ravel()
        return np.asarray(Xmat[:, j]).ravel()

    def _get_beta(self, gene: str | int) -> np.ndarray:
        """Return fitted coefficients for a gene.

        :param gene: Gene name or index
        :type gene: str | int
        :raises ValueError: If no fitted coefficients are found for the specified gene, which may indicate that `.fit()` has not been called or that the gene was excluded from fitting.
        :return: Fitted coefficients for the specified gene
        :rtype: np.ndarray
        """
        # Get gene index
        j = self._gene_index(gene)

        # Retrieve the coefficient matrix from adata.varm and extract the row for the specified gene
        coef_key = f"{self.key}_coef"
        if coef_key not in self.adata.varm:
            raise ValueError(
                f"No coefficient matrix found under adata.varm[{coef_key!r}]. "
                "Did you call .fit()?"
            )

        # Extract the coefficient vector for the specified gene index and validate that it contains finite values
        beta = np.asarray(self.adata.varm[coef_key][j], dtype=float)
        if not np.isfinite(beta).all():
            raise ValueError(
                f"No fitted coefficients for gene index/name: {gene}. "
                "Did you call .fit()? Or was this gene excluded from fitting?"
            )
        return beta

    def _approx_beta_cov(self, beta: np.ndarray) -> np.ndarray:
        """Approximate coefficient covariance from working weights.

        :param beta: Coefficient vector
        :type beta: np.ndarray
        :return: Approximated covariance matrix
        :rtype: np.ndarray
        """

        # 
        X_fit = self.X[self._fit_mask]
        eta = X_fit @ beta
        mu = np.exp(np.clip(eta, -50.0, 50.0))
        alpha = 1.0 / float(self.theta)
        w = mu / (1.0 + alpha * mu)

        ridge = float(self.lam) + float(self.ridge)
        XtWX = X_fit.T @ (X_fit * w[:, None]) + ridge * np.eye(self.p)

        try:
            return np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(XtWX)

    def _get_beta_cov(self, gene: str | int) -> tuple[np.ndarray, np.ndarray]:
        """Return `(beta, cov)` for a gene, using stored or approximated covariance.

        :param gene: Gene name or index
        :type gene: str | int
        :return: Tuple of `(beta, cov)` where `beta` is the coefficient vector and `cov` is the covariance matrix
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        # Get index for gene
        j = self._gene_index(gene)
        # Get beta coefficients for gene
        beta = self._get_beta(j)

        # Check if covariance is already stored; if not, approximate it
        cov_key = f"{self.key}_cov"
        if cov_key in self.adata.varm:
            cov = np.asarray(self.adata.varm[cov_key][j], dtype=float)
            if np.isfinite(cov).all():
                return beta, cov

        # Return beta and approximated covariance
        return beta, self._approx_beta_cov(beta)

    def _basis_row(self, t: float) -> np.ndarray:
        """Return a single design-matrix row for one pseudotime value.

        :param t: Pseudotime value
        :type t: float
        :raises ValueError: If the provided pseudotime value is non-finite
        :return: Design-matrix row for the provided pseudotime value
        :rtype: np.ndarray
        """
        # Cast float to np.ndarray and validate finiteness; 
        x = np.asarray([t], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Provided t contains non-finite values.")

        # Clip to range
        x = np.clip(x, self._tmin, self._tmax)

        # Place in a DataFrame and apply the formula's encoding to get the design-matrix row
        frame = pd.DataFrame({self._pt_sanitized_key: x})

        # Add covariates if provided
        for cov in self._covariates:
            frame[cov] = self._covariate_defaults.get(cov, 0.0)

        # Encode using the formula's data encoding, which applies the same 
        # sanitization and basis construction as during fitting
        _y, B, _notNA, *_rest = self._formula.encode_data(frame, prediction=True)

        # Returns the design-matrix row as a 1D array; the formula encoding returns a 2D array with shape (1, p)
        return np.asarray(B, dtype=float).reshape(-1)