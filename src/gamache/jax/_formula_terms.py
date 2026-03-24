from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable


class VarType(str, Enum):
    NUMERIC = "numeric"
    FACTOR = "factor"


class lhs:
    """The Left-hand side of a regression equation.

    This is a small, compatibility-oriented version of the original MSSM `lhs`.

    Parameters
    ----------
    variable:
        Name of the dependent variable in `data`.
    f:
        Optional transform applied to the dependent variable before fitting.
    """

    def __init__(self, variable: str, f: Callable | None = None) -> None:
        self.variable = str(variable)
        self.f = f


@dataclass(frozen=True)
class InterceptTerm:
    pass


def i() -> InterceptTerm:
    return InterceptTerm()


@dataclass(frozen=True)
class LinearTerm:
    vars: tuple[str, ...]


def l(vars: Iterable[str]) -> LinearTerm:
    return LinearTerm(tuple(str(v) for v in vars))


@dataclass(frozen=True)
class SmoothTerm:
    vars: tuple[str, ...]
    nk: int = 9
    degree: int = 3
    by: str | None = None
    kind: str = "f"  # "f" or "fs"


@dataclass(frozen=True)
class UnsupportedTerm:
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def f(*args: Any, **kwargs: Any) -> UnsupportedTerm | SmoothTerm:
    # Minimal support: f(["time"], nk=9, by="cond") for univariate smooths.
    if len(args) != 1:
        return UnsupportedTerm("f", args, dict(kwargs))

    try:
        vars_t = tuple(str(v) for v in args[0])
    except Exception:
        return UnsupportedTerm("f", args, dict(kwargs))

    nk = int(kwargs.pop("nk", 9))
    degree = int(kwargs.pop("degree", 3))
    by = kwargs.pop("by", None)
    if kwargs:
        return UnsupportedTerm("f", (vars_t,), dict(kwargs, nk=nk, degree=degree, by=by))

    return SmoothTerm(
        vars=vars_t,
        nk=nk,
        degree=degree,
        by=None if by is None else str(by),
        kind="f",
    )


def fs(*args: Any, **kwargs: Any) -> UnsupportedTerm | SmoothTerm:
    # Minimal support: fs(["time"], rf="sub", nk=9). In this pruned build we treat
    # this like a by-factor smooth (no random-effect penalty).
    if len(args) != 1:
        return UnsupportedTerm("fs", args, dict(kwargs))

    try:
        vars_t = tuple(str(v) for v in args[0])
    except Exception:
        return UnsupportedTerm("fs", args, dict(kwargs))

    nk = int(kwargs.pop("nk", 9))
    degree = int(kwargs.pop("degree", 3))
    rf = kwargs.pop("rf", None)
    by = kwargs.pop("by", None)
    if rf is not None and by is not None:
        return UnsupportedTerm("fs", (vars_t,), dict(kwargs, nk=nk, degree=degree, rf=rf, by=by))

    by_var = rf if rf is not None else by
    if kwargs:
        return UnsupportedTerm("fs", (vars_t,), dict(kwargs, nk=nk, degree=degree, rf=rf, by=by))

    return SmoothTerm(
        vars=vars_t,
        nk=nk,
        degree=degree,
        by=None if by_var is None else str(by_var),
        kind="fs",
    )
