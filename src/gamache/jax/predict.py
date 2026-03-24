from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def predict_gene_linear_predictor(
    adata: Any,
    *,
    gene: str,
    formula: Any,
    store_key: str = "jaxgamx",
) -> np.ndarray:
    """Predict the linear predictor eta = X @ beta for a gene.

    This is the untransformed value (before applying the family link function).
    """

    var_names = getattr(adata, "var_names", None)
    if var_names is None:
        raise TypeError("adata.var_names is required")

    var_names = [str(v) for v in list(var_names)]
    gene = str(gene)

    try:
        gene_idx = var_names.index(gene)
    except ValueError as e:
        raise KeyError(f"gene {gene!r} not found in adata.var_names") from e

    varm = getattr(adata, "varm", None)
    if varm is None:
        raise TypeError("adata.varm is required")

    key = f"{store_key}_coef"
    if key not in varm:
        raise KeyError(
            f"No fitted coefficients found at adata.varm[{key!r}]. "
            "Run `GAMM.fit(adata=..., store_key=...)` first."
        )

    coef_row = np.asarray(varm[key])[gene_idx]
    coef_row = np.asarray(coef_row, dtype=float).reshape(-1)

    if not np.any(np.isfinite(coef_row)):
        raise ValueError(
            f"No fitted coefficients available for gene {gene!r} in adata.varm[{key!r}]"
        )

    _y, cov, *_rest = formula.encode_data(adata, prediction=True)
    cov = np.asarray(cov, dtype=float)

    if cov.ndim != 2:
        raise ValueError("Encoded design matrix must be 2D")
    if cov.shape[1] != coef_row.shape[0]:
        raise ValueError(
            f"Design matrix has {cov.shape[1]} columns but coef has {coef_row.shape[0]} entries"
        )

    return (cov @ coef_row).reshape(-1)


def predict_gene_mean(
    adata: Any,
    *,
    gene: str,
    formula: Any,
    store_key: str = "jaxgamx",
    family: str | None = None,
    transform: str | None = None,
) -> np.ndarray:
    """Predict fitted mean for a gene using coefficients stored in `adata.varm`.

    This is intended to pair with `GAMM.fit(adata=..., store_key=...)`.

    Parameters
    ----------
    adata:
        AnnData-like object with `.obs`, `.var_names`, and `.varm`.
    gene:
        Gene name in `adata.var_names`.
    formula:
        The `Formula` used for fitting (or an equivalent one). Used to rebuild
        the design matrix for `adata`.
    store_key:
        Prefix used when storing fit results in `adata.varm`.
    family:
        Optional family name. If omitted, this tries to infer it from
        `adata.uns[f"{store_key}_family"]` (written by `GAMM.fit`).
        Supported values: "gaussian", "poisson", "negative_binomial", "binomial".
    transform:
        Optional post-transform applied to the mean. Currently supported: "log1p".

    Returns
    -------
    y_hat:
        Array of shape `(n_obs,)` with fitted values.
    """

    eta = predict_gene_linear_predictor(
        adata,
        gene=gene,
        formula=formula,
        store_key=store_key,
    )

    fam = family
    if fam is None:
        uns = getattr(adata, "uns", None)
        if isinstance(uns, dict):
            fam = uns.get(f"{store_key}_family")

    fam = (fam or "gaussian").strip().lower()
    if fam in {"gaussian", "normal"}:
        mu = eta
    elif fam in {"poisson"}:
        mu = np.exp(eta)
    elif fam in {"negative_binomial", "negativebinomial", "nb", "nb2"}:
        mu = np.exp(eta)
    elif fam in {"binomial", "bernoulli"}:
        mu = _sigmoid(eta)
    else:
        raise ValueError(
            f"Unsupported family {fam!r} for predict_gene_mean(). "
            "Pass family='gaussian'|'poisson'|'negative_binomial'|'binomial'."
        )

    if transform is None:
        return mu.reshape(-1)

    t = transform.strip().lower()
    if t == "log1p":
        return np.log1p(mu).reshape(-1)

    raise ValueError(f"Unsupported transform {transform!r}. Supported: 'log1p'.")


__all__ = ["predict_gene_mean", "predict_gene_linear_predictor"]
