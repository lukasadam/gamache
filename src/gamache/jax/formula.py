from __future__ import annotations

"""Public formula API.

This module is intentionally small: the implementation is split into internal
modules to make it easier to browse, while preserving backwards-compatible
imports like `from gamache.jax.formula import Formula, f, fs, i, l, lhs`.
"""

from ._formula_impl import Formula
from ._formula_terms import (
    InterceptTerm,
    LinearTerm,
    SmoothTerm,
    UnsupportedTerm,
    VarType,
    f,
    fs,
    i,
    l,
    lhs,
)

__all__ = [
    "Formula",
    "VarType",
    "lhs",
    "InterceptTerm",
    "LinearTerm",
    "SmoothTerm",
    "UnsupportedTerm",
    "i",
    "l",
    "f",
    "fs",
]
