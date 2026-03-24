"""Public `PseudotimeGAM` class."""

from __future__ import annotations

from ._core import _Core
from ._fit import _FitMixin
from ._predict import _PredictMixin
from ._tests import _TestsMixin


class PseudotimeGAM(
    _Core,
    _FitMixin,
    _PredictMixin,
    _TestsMixin,
):
    """Pseudotime NB-GAM with a JAX backend."""


__all__ = ["PseudotimeGAM"]
