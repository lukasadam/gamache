"""Pseudotime GAM API.

This subpackage exists to keep the public AnnData-centric pseudotime GAM code
browseable.

Backwards compatibility:
- `gamache.tl.fit` re-exports `PseudotimeGAM` and `fit_gam`.
"""

from .api import fit_gam
from .model import PseudotimeGAM

__all__ = ["PseudotimeGAM", "fit_gam"]
