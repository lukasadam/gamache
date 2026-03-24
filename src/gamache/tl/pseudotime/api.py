"""Convenience API for fitting a pseudotime GAM."""

from __future__ import annotations

from typing import Optional, Sequence

import anndata as ad

from .model import PseudotimeGAM


def fit_gam(
    adata: ad.AnnData,
    layer: Optional[str] = "counts",
    pseudotime_key: str = "dpt_pseudotime",
    df: int = 6,
    degree: int = 3,
    lam: float = 0.01,
    include_intercept: bool = False,
    covariates: Optional[Sequence[str]] = None,
    key: str = "nbgam1d",
    nonfinite: str = "error",
    nb_alpha: Optional[float] = None,
) -> PseudotimeGAM:
    """Fit a `PseudotimeGAM` model to the given AnnData using the JAX backend."""
    if nonfinite != "error":
        raise ValueError(
            "Non-finite pseudotime handling has been removed. "
            "Ensure pseudotime is fully finite before fitting (nonfinite must be 'error')."
        )
    model = PseudotimeGAM(
        adata=adata,
        layer=layer,
        pseudotime_key=pseudotime_key,
        df=df,
        degree=degree,
        lam=lam,
        include_intercept=include_intercept,
        covariates=covariates,
        key=key,
        nonfinite=nonfinite,
        nb_alpha=nb_alpha,
    )
    model.fit()
    return model


__all__ = ["fit_gam"]
