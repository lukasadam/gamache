"""Gamache: A Python package for fitting Generalized Additive Models (GAMs) to single cell data."""

from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]

__version__ = version("gamache")
