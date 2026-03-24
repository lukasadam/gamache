"""Pseudotime GAM fitting API (backwards-compatible shim).

Historically, `gamache.tl.fit` contained the full `PseudotimeGAM` implementation.
To make the codebase easier to browse, the implementation now lives under the
`gamache.tl.pseudotime` subpackage.

Public API
----------
- `PseudotimeGAM`
- `fit_gam`

Import paths preserved:
- `from gamache.tl.fit import PseudotimeGAM, fit_gam`
"""

from __future__ import annotations

from .pseudotime import PseudotimeGAM, fit_gam

__all__ = ["PseudotimeGAM", "fit_gam"]
