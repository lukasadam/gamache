from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import numpy as np
import scipy as scp


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


def f(*args: Any, **kwargs: Any) -> UnsupportedTerm:
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


def fs(*args: Any, **kwargs: Any) -> UnsupportedTerm:
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


class Formula:
    """The formula of a regression equation (JAX-only compatibility layer).

    This pruned build keeps the *constructor signature* and a small subset of the
    behavior needed to assemble a design matrix for `GAMM.fit()`.

    Supported terms:
    - `i()` intercept
    - `l([...])` numeric columns + simple factor one-hot encoding

    Unsupported (will raise `NotImplementedError`): smooth/random terms (`f`, `fs`),
    penalty building, streaming-from-files, nested smooth detection.
    """

    def __init__(
        self,
        lhs: lhs,
        terms: list[Any],
        data: Any,
        series_id: str | None = None,
        codebook: dict | None = None,
        print_warn: bool = True,
        keep_cov: bool = False,
        find_nested: bool = True,
        file_paths: list[str] = [],
        file_loading_nc: int = 1,
        file_loading_kwargs: dict = {"header": 0, "index_col": False},
    ) -> None:
        self.lhs = lhs
        self.terms = list(terms)
        self.data = data

        self.series_id = series_id
        self.codebook = codebook
        self.print_warn = bool(print_warn)
        self.keep_cov = bool(keep_cov)
        self.find_nested = bool(find_nested)
        self.file_paths = list(file_paths)
        self.file_loading_nc = int(file_loading_nc)
        self.file_loading_kwargs = dict(file_loading_kwargs)

        if self.file_paths:
            raise NotImplementedError(
                "Streaming from files is not supported in the pruned JAX-only build. "
                "Pass an in-memory `data` mapping / DataFrame."
            )
        if self.data is None:
            raise TypeError("`data` is required in the pruned JAX-only build")

        # Core artifacts used by `GAMM`
        self.y: np.ndarray | None
        self.Xs: list[scp.sparse.csc_array]
        self.coef_split_idx: list[int] = []

        # Lightweight metadata / compatibility getters
        self.coef_names: list[str] = []
        self.var_map: dict[str, int] = {}
        self.var_types: dict[str, VarType] = {}
        self.factor_levels: dict[str, np.ndarray] = {}
        self.factor_codings: dict[str, dict[str, int]] = {}
        self.coding_factors: dict[str, dict[int, str]] = {}
        self.var_mins: dict[str, float | None] = {}
        self.var_maxs: dict[str, float | None] = {}

        # Training-time state for consistent encoding on new data.
        # Keys are tuples describing the SmoothTerm: (kind, vars, nk, degree, by).
        self._smooth_states: dict[tuple, dict[str, Any]] = {}

        self._build_design()

    @classmethod
    def from_xy(
        cls,
        y: Any,
        X: Any,
        feature_names: list[str] | None = None,
    ) -> "Formula":
        obj = cls.__new__(cls)
        obj.lhs = lhs("y")
        obj.terms = []
        obj.data = None
        obj.series_id = None
        obj.codebook = None
        obj.print_warn = True
        obj.keep_cov = False
        obj.find_nested = True
        obj.file_paths = []
        obj.file_loading_nc = 1
        obj.file_loading_kwargs = {"header": 0, "index_col": False}

        obj.coef_split_idx = []

        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2 or y_arr.shape[1] != 1:
            raise ValueError("y must be shape (n,) or (n, 1)")

        if scp.sparse.issparse(X):
            Xc = scp.sparse.csc_array(X)
        else:
            Xc = scp.sparse.csc_array(np.asarray(X))

        obj.y = y_arr.astype(float)
        obj.Xs = [Xc]
        obj.coef_names = feature_names or [f"x{i}" for i in range(Xc.shape[1])]

        obj.var_map = {n: i for i, n in enumerate(obj.coef_names)}
        obj.var_types = {}
        obj.factor_levels = {}
        obj.factor_codings = {}
        obj.coding_factors = {}
        obj.var_mins = {}
        obj.var_maxs = {}

        return obj

    @classmethod
    def from_string(
        cls,
        formula: str,
        *,
        data: Any,
        series_id: str | None = None,
        codebook: dict | None = None,
        print_warn: bool = True,
    ) -> "Formula":
        """Create a `Formula` from a compact string.

        Supported (pruned build):
        - `y ~ x1 + x2` (adds intercept by default)
        - `f(time, nk=7)` and `f(time, by=cond)`
        - `f(time, x, nk=7)` and `f(time, x, by=cond)` (tensor-product basis)
        - `fs(time, rf=sub)` treated like `by=sub` (no random-effect penalty)

        Notes
        -----
        This parser is intentionally minimal: it does not support nested terms,
        `-` subtraction, `:` interactions, or arbitrary Python expressions.
        """

        s = str(formula).strip()
        if "~" not in s:
            raise ValueError("Formula string must contain '~'")

        lhs_s, rhs_s = [p.strip() for p in s.split("~", 1)]
        if not lhs_s:
            raise ValueError("Missing LHS in formula")

        include_intercept = True
        rhs_terms = [t.strip() for t in rhs_s.split("+") if t.strip()]

        terms: list[Any] = []

        def _parse_call(call: str) -> SmoothTerm | UnsupportedTerm:
            call = call.strip()
            if not call.endswith(")"):
                return UnsupportedTerm("call", (call,), {})

            if call.startswith("f("):
                kind = "f"
                inside = call[2:-1]
            elif call.startswith("fs("):
                kind = "fs"
                inside = call[3:-1]
            else:
                return UnsupportedTerm("call", (call,), {})

            parts = [p.strip() for p in inside.split(",") if p.strip()]
            pos: list[str] = []
            kwargs: dict[str, str] = {}
            for p in parts:
                if "=" in p:
                    k, v = [q.strip() for q in p.split("=", 1)]
                    kwargs[k] = v
                else:
                    pos.append(p)

            nk = int(kwargs.pop("nk", "9"))
            by = kwargs.pop("by", None)
            rf = kwargs.pop("rf", None)
            if kwargs:
                return UnsupportedTerm(kind, tuple(pos), dict(kwargs, nk=nk, by=by, rf=rf))

            by_var = None
            if kind == "f":
                by_var = by
            else:
                by_var = rf if rf is not None else by

            return SmoothTerm(
                vars=tuple(pos),
                nk=nk,
                by=None if by_var is None else str(by_var),
                kind=kind,
            )

        for t in rhs_terms:
            if t in {"1"}:
                continue
            if t in {"0", "-1"}:
                include_intercept = False
                continue

            if t.startswith("f(") or t.startswith("fs("):
                st = _parse_call(t)
                if isinstance(st, UnsupportedTerm):
                    terms.append(st)
                else:
                    terms.append(st)
                continue

            # Default: treat as linear term
            terms.append(LinearTerm((t,)))

        if include_intercept:
            terms.insert(0, InterceptTerm())

        return cls(
            lhs=lhs(lhs_s),
            terms=terms,
            data=data,
            series_id=series_id,
            codebook=codebook,
            print_warn=print_warn,
        )

    def _build_design(self) -> None:
        for t in self.terms:
            if isinstance(t, UnsupportedTerm):
                raise NotImplementedError(
                    "Unsupported term/kwargs in the pruned JAX-only build. "
                    "Supported: i(), l([...]), f(var, nk=.., by=..), f(var1, var2, nk=.., by=..), and fs(var, rf=..) (treated as by-factor)."
                )

        data = self.data

        def _frame(obj: Any) -> Any:
            return obj.obs if hasattr(obj, "obs") else obj

        frame = _frame(data)

        def _has_col(frame_obj: Any, key: str) -> bool:
            try:
                return key in frame_obj
            except Exception:
                try:
                    frame_obj[key]
                    return True
                except Exception:
                    return False

        def _col(frame_obj: Any, key: str) -> np.ndarray:
            return np.asarray(frame_obj[key])

        y: np.ndarray | None
        if _has_col(frame, self.lhs.variable):
            y = _col(frame, self.lhs.variable)
            if self.lhs.f is not None:
                y = np.asarray(self.lhs.f(y))
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.ndim != 2 or y.shape[1] != 1:
                raise ValueError("lhs response must be vector-like")
            n = int(y.shape[0])
        else:
            # AnnData use-case: y comes from adata.X; allow missing lhs in obs.
            y = None
            n = None
            for t in self.terms:
                if isinstance(t, LinearTerm):
                    x0 = _col(frame, t.vars[0])
                    n = int(np.asarray(x0).shape[0])
                    break
                if isinstance(t, SmoothTerm):
                    x0 = _col(frame, t.vars[0])
                    n = int(np.asarray(x0).shape[0])
                    break
            if n is None:
                raise ValueError("Cannot infer number of rows from terms")

        n = int(n)

        cols: list[np.ndarray] = []
        names: list[str] = []

        has_intercept = any(isinstance(t, InterceptTerm) for t in self.terms)
        if has_intercept:
            cols.append(np.ones((n, 1), dtype=float))
            names.append("Intercept")

        # Codebook support for factors (optional)
        codebook = self.codebook or {}

        def _bspline_state_1d(x: np.ndarray, nk: int, degree: int) -> dict[str, Any]:
            x = np.asarray(x, dtype=float).reshape(-1)
            if nk < degree + 2:
                raise ValueError(f"nk must be >= {degree + 2}")

            xmin = float(np.min(x))
            xmax = float(np.max(x))
            if xmax == xmin:
                return {
                    "constant": True,
                    "xmin": xmin,
                    "xmax": xmax,
                    "degree": int(degree),
                    "t": np.asarray([0.0, 1.0], dtype=float),
                    "drop_first": False,
                }

            xs = (x - xmin) / (xmax - xmin)

            n_internal = max(nk - degree - 1, 0)
            if n_internal > 0:
                qs = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
                internal = np.unique(np.quantile(xs, qs))
            else:
                internal = np.array([], dtype=float)

            tvec = np.concatenate(
                [np.zeros(degree + 1), internal, np.ones(degree + 1)],
                axis=0,
            )

            return {
                "constant": False,
                "xmin": xmin,
                "xmax": xmax,
                "degree": int(degree),
                "t": np.asarray(tvec, dtype=float),
                "drop_first": False,
            }

        def _bspline_basis_1d_from_state(x: np.ndarray, state: dict[str, Any]) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            if state.get("constant", False):
                return np.ones((x.shape[0], 1), dtype=float)

            xmin = float(state["xmin"])
            xmax = float(state["xmax"])
            degree = int(state["degree"])
            tvec = np.asarray(state["t"], dtype=float)

            xs = (x - xmin) / (xmax - xmin)
            n_basis = int(len(tvec) - degree - 1)
            B = np.empty((xs.shape[0], n_basis), dtype=float)

            from scipy.interpolate import BSpline

            for j in range(n_basis):
                c = np.zeros((n_basis,), dtype=float)
                c[j] = 1.0
                spl = BSpline(tvec, c, degree, extrapolate=True)
                B[:, j] = spl(xs)
            return B

        for t in self.terms:
            if isinstance(t, InterceptTerm):
                continue
            if isinstance(t, LinearTerm):
                for var in t.vars:
                    x = np.asarray(frame[var])
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    if x.shape[0] != n:
                        raise ValueError(f"Column {var!r} has wrong number of rows")

                    if x.dtype.kind in {"O", "U", "S"}:
                        levels = np.unique(x.reshape(-1))
                        self.factor_levels[var] = np.array(levels, dtype=object)

                        if var in codebook:
                            mapping = {str(k): int(v) for k, v in codebook[var].items()}
                            levels_sorted = sorted(mapping.items(), key=lambda kv: kv[1])
                            ordered_levels = [k for k, _ in levels_sorted]
                        else:
                            ordered_levels = [str(lv) for lv in levels]
                            mapping = {lev: i for i, lev in enumerate(ordered_levels)}

                        self.factor_codings[var] = dict(mapping)
                        self.coding_factors[var] = {v: k for k, v in mapping.items()}

                        if len(ordered_levels) <= 1:
                            continue

                        base = ordered_levels[0]
                        for lev in ordered_levels[1:]:
                            cols.append(
                                (x.reshape(-1).astype(object) == lev)
                                .astype(float)
                                .reshape(-1, 1)
                            )
                            names.append(f"{var}[{lev}]")

                        self.var_types[var] = VarType.FACTOR
                        self.var_mins[var] = None
                        self.var_maxs[var] = None
                    else:
                        x = x.astype(float)
                        cols.append(x)
                        names.append(var)

                        self.var_types[var] = VarType.NUMERIC
                        self.var_mins[var] = float(np.nanmin(x))
                        self.var_maxs[var] = float(np.nanmax(x))
                continue

            if isinstance(t, SmoothTerm):
                if len(t.vars) not in {1, 2}:
                    raise NotImplementedError("Only f(var, ...) and f(var1, var2, ...) are supported")

                if len(t.vars) == 1:
                    v1 = t.vars[0]
                    x1 = np.asarray(frame[v1]).reshape(-1)
                    if x1.shape[0] != n:
                        raise ValueError(f"Column {v1!r} has wrong number of rows")
                    if x1.dtype.kind in {"O", "U", "S"}:
                        raise NotImplementedError("Smooth terms require numeric covariates")

                    state = _bspline_state_1d(x1, nk=int(t.nk), degree=int(t.degree))
                    B = _bspline_basis_1d_from_state(x1, state)
                    if B.shape[1] > 1:
                        B = B[:, 1:]
                        state["drop_first"] = True
                    else:
                        state["drop_first"] = False

                    smooth_key = (t.kind, t.vars, int(t.nk), int(t.degree), t.by)

                    base_name = f"{t.kind}({v1})"

                    # Optional by-factor
                    if t.by is None:
                        cols.append(B)
                        for j in range(B.shape[1]):
                            names.append(f"{base_name}[{j}]")
                    else:
                        byv = np.asarray(frame[t.by]).reshape(-1)
                        if byv.dtype.kind not in {"O", "U", "S"}:
                            raise NotImplementedError("by= is only supported for factor-like columns")
                        levels = [str(v) for v in np.unique(byv.astype(object))]
                        state["by_levels"] = list(levels)
                        for lev in levels:
                            mask = (byv.astype(object) == lev).astype(float).reshape(-1, 1)
                            Blev = B * mask
                            cols.append(Blev)
                            for j in range(B.shape[1]):
                                names.append(f"{base_name}:{t.by}={lev}[{j}]")

                    # store training-time state for stable prediction encodings
                    self._smooth_states[smooth_key] = dict(state)

                    self.var_types[v1] = VarType.NUMERIC
                    self.var_mins[v1] = float(np.nanmin(x1))
                    self.var_maxs[v1] = float(np.nanmax(x1))
                    continue

                v1, v2 = t.vars
                x1 = np.asarray(frame[v1]).reshape(-1)
                x2 = np.asarray(frame[v2]).reshape(-1)
                if x1.shape[0] != n or x2.shape[0] != n:
                    raise ValueError("Smooth covariates have wrong number of rows")
                if x1.dtype.kind in {"O", "U", "S"} or x2.dtype.kind in {"O", "U", "S"}:
                    raise NotImplementedError("Smooth terms require numeric covariates")

                state1 = _bspline_state_1d(x1, nk=int(t.nk), degree=int(t.degree))
                state2 = _bspline_state_1d(x2, nk=int(t.nk), degree=int(t.degree))
                B1 = _bspline_basis_1d_from_state(x1, state1)
                B2 = _bspline_basis_1d_from_state(x2, state2)
                if B1.shape[1] > 1:
                    B1 = B1[:, 1:]
                    state1["drop_first"] = True
                else:
                    state1["drop_first"] = False
                if B2.shape[1] > 1:
                    B2 = B2[:, 1:]
                    state2["drop_first"] = True
                else:
                    state2["drop_first"] = False
                B = (B1[:, :, None] * B2[:, None, :]).reshape(x1.shape[0], -1)
                base_name = f"{t.kind}({v1},{v2})"

                smooth_key = (t.kind, t.vars, int(t.nk), int(t.degree), t.by)
                state2d: dict[str, Any] = {"state1": dict(state1), "state2": dict(state2)}

                if t.by is None:
                    cols.append(B)
                    for j in range(B.shape[1]):
                        names.append(f"{base_name}[{j}]")
                else:
                    byv = np.asarray(frame[t.by]).reshape(-1)
                    if byv.dtype.kind not in {"O", "U", "S"}:
                        raise NotImplementedError("by= is only supported for factor-like columns")
                    levels = [str(v) for v in np.unique(byv.astype(object))]
                    state2d["by_levels"] = list(levels)
                    for lev in levels:
                        mask = (byv.astype(object) == lev).astype(float).reshape(-1, 1)
                        Blev = B * mask
                        cols.append(Blev)
                        for j in range(B.shape[1]):
                            names.append(f"{base_name}:{t.by}={lev}[{j}]")

                self._smooth_states[smooth_key] = state2d

                self.var_types[v1] = VarType.NUMERIC
                self.var_mins[v1] = float(np.nanmin(x1))
                self.var_maxs[v1] = float(np.nanmax(x1))
                self.var_types[v2] = VarType.NUMERIC
                self.var_mins[v2] = float(np.nanmin(x2))
                self.var_maxs[v2] = float(np.nanmax(x2))
                continue

            raise NotImplementedError(
                "Only i(), l([...]), f(...), and fs(...) terms are supported in the pruned JAX-only build."
            )

        if not cols:
            raise ValueError("No design columns produced; did you forget i() or l([...])?")

        X = np.concatenate(cols, axis=1)
        self.y = None if y is None else y.astype(float)
        self.Xs = [scp.sparse.csc_array(X)]
        self.coef_names = names
        self.var_map = {name: i for i, name in enumerate(names)}

    # --- Compatibility getters ---
    def get_lhs(self) -> lhs:
        return lhs(self.lhs.variable, self.lhs.f)

    def get_terms(self) -> list[Any]:
        return list(self.terms)

    def get_data(self) -> Any:
        d = self.data
        if hasattr(d, "copy"):
            try:
                return d.copy()
            except Exception:
                pass
        if isinstance(d, dict):
            return dict(d)
        return d

    def get_depvar(self) -> np.ndarray:
        if self.y is None:
            raise RuntimeError("No dependent variable stored in Formula (lhs not found in data)")
        return np.array(self.y, copy=True)

    def get_notNA(self) -> np.ndarray:
        if self.y is None:
            raise RuntimeError("No dependent variable stored in Formula (lhs not found in data)")
        return np.isfinite(self.y.reshape(-1))

    def get_has_intercept(self) -> bool:
        return any(isinstance(t, InterceptTerm) for t in self.terms)

    def get_n_coef(self) -> int:
        return int(self.Xs[0].shape[1])

    def get_term_names(self) -> list[str]:
        out: list[str] = []
        for t in self.terms:
            if isinstance(t, InterceptTerm):
                out.append("i")
            elif isinstance(t, LinearTerm):
                out.append("l")
            elif isinstance(t, SmoothTerm):
                out.append("f")
            elif isinstance(t, UnsupportedTerm):
                out.append(t.name)
            else:
                out.append(type(t).__name__)
        return out

    def get_linear_term_idx(self) -> list[int]:
        return [i for i, t in enumerate(self.terms) if isinstance(t, LinearTerm)]

    def get_smooth_term_idx(self) -> list[int]:
        return [i for i, t in enumerate(self.terms) if isinstance(t, SmoothTerm)]

    def get_random_term_idx(self) -> list[int]:
        return []

    def get_ir_smooth_term_idx(self) -> list[int]:
        return []

    def has_ir_terms(self) -> bool:
        return False

    def get_subgroup_variables(self) -> list:
        return []

    def get_var_map(self) -> dict:
        return dict(self.var_map)

    def get_var_types(self) -> dict:
        return dict(self.var_types)

    def get_factor_levels(self) -> dict:
        return {k: np.array(v, copy=True) for k, v in self.factor_levels.items()}

    def get_factor_codings(self) -> dict:
        return {k: dict(v) for k, v in self.factor_codings.items()}

    def get_coding_factors(self) -> dict:
        return {k: dict(v) for k, v in self.coding_factors.items()}

    def get_var_mins(self) -> dict:
        return dict(self.var_mins)

    def get_var_maxs(self) -> dict:
        return dict(self.var_maxs)

    def get_var_mins_maxs(self) -> tuple[dict, dict]:
        return self.get_var_mins(), self.get_var_maxs()

    # --- Encoding entry point ---
    def encode_data(self, data: Any, prediction: bool = False):
        """Encode a new dataset.

        Returns a simplified version of the original API. For this pruned build:
        - `cov_flat` is the design matrix matching `i()`/`l()` terms.
        - If `series_id` is provided, returns per-series splits.
        """

        if self.file_paths:
            raise NotImplementedError("File streaming is not supported")

        if prediction:
            frame = data.obs if hasattr(data, "obs") else data

            # infer row count
            n: int | None = None
            for term in self.terms:
                if isinstance(term, InterceptTerm):
                    continue
                if isinstance(term, LinearTerm):
                    n = int(np.asarray(frame[term.vars[0]]).shape[0])
                    break
                if isinstance(term, SmoothTerm):
                    n = int(np.asarray(frame[term.vars[0]]).shape[0])
                    break
            if n is None:
                raise ValueError("Cannot infer number of rows from terms")
            n = int(n)

            cols: list[np.ndarray] = []

            if self.get_has_intercept():
                cols.append(np.ones((n, 1), dtype=float))

            from scipy.interpolate import BSpline

            def _basis_1d(x: np.ndarray, state: dict[str, Any]) -> np.ndarray:
                x = np.asarray(x, dtype=float).reshape(-1)
                if state.get("constant", False):
                    B = np.ones((x.shape[0], 1), dtype=float)
                else:
                    xmin = float(state["xmin"])
                    xmax = float(state["xmax"])
                    degree = int(state["degree"])
                    tvec = np.asarray(state["t"], dtype=float)
                    xs = (x - xmin) / (xmax - xmin)
                    n_basis = int(len(tvec) - degree - 1)
                    B = np.empty((xs.shape[0], n_basis), dtype=float)
                    for j in range(n_basis):
                        c = np.zeros((n_basis,), dtype=float)
                        c[j] = 1.0
                        spl = BSpline(tvec, c, degree, extrapolate=True)
                        B[:, j] = spl(xs)
                if bool(state.get("drop_first", False)) and B.shape[1] > 1:
                    B = B[:, 1:]
                return B

            for term in self.terms:
                if isinstance(term, InterceptTerm):
                    continue

                if isinstance(term, LinearTerm):
                    for var in term.vars:
                        x = np.asarray(frame[var])
                        if x.ndim == 1:
                            x = x.reshape(-1, 1)
                        if x.shape[0] != n:
                            raise ValueError(f"Column {var!r} has wrong number of rows")

                        if x.dtype.kind in {"O", "U", "S"}:
                            mapping = self.factor_codings.get(var)
                            if mapping is None:
                                # fall back to current levels
                                levels = [str(v) for v in np.unique(x.reshape(-1).astype(object))]
                                mapping = {lev: i for i, lev in enumerate(levels)}
                                coding_factors = {i: lev for lev, i in mapping.items()}
                            else:
                                coding_factors = self.coding_factors[var]

                            ordered_levels = [coding_factors[i] for i in sorted(mapping.values())]
                            if len(ordered_levels) <= 1:
                                continue
                            xv = x.reshape(-1).astype(object)
                            for lev in ordered_levels[1:]:
                                cols.append((xv == lev).astype(float).reshape(-1, 1))
                            continue

                        cols.append(x.astype(float))
                    continue

                if isinstance(term, SmoothTerm):
                    smooth_key = (term.kind, term.vars, int(term.nk), int(term.degree), term.by)
                    st = self._smooth_states.get(smooth_key)
                    if st is None:
                        raise RuntimeError(
                            "Smooth term state missing; build the Formula on training data before encoding prediction data."
                        )

                    if len(term.vars) == 1:
                        v1 = term.vars[0]
                        x1 = np.asarray(frame[v1]).reshape(-1)
                        B = _basis_1d(x1, st)
                        if term.by is None:
                            cols.append(B)
                        else:
                            by_levels = [str(v) for v in st.get("by_levels", [])]
                            byv = np.asarray(frame[term.by]).reshape(-1).astype(object)
                            for lev in by_levels:
                                mask = (byv == lev).astype(float).reshape(-1, 1)
                                cols.append(B * mask)
                        continue

                    if len(term.vars) == 2:
                        v1, v2 = term.vars
                        x1 = np.asarray(frame[v1]).reshape(-1)
                        x2 = np.asarray(frame[v2]).reshape(-1)
                        B1 = _basis_1d(x1, st["state1"])
                        B2 = _basis_1d(x2, st["state2"])
                        B = (B1[:, :, None] * B2[:, None, :]).reshape(x1.shape[0], -1)
                        if term.by is None:
                            cols.append(B)
                        else:
                            by_levels = [str(v) for v in st.get("by_levels", [])]
                            byv = np.asarray(frame[term.by]).reshape(-1).astype(object)
                            for lev in by_levels:
                                mask = (byv == lev).astype(float).reshape(-1, 1)
                                cols.append(B * mask)
                        continue

                    raise NotImplementedError("Only 1D/2D smooths are supported")

                raise NotImplementedError(
                    "Only i(), l([...]), f(...), and fs(...) terms are supported in the pruned JAX-only build."
                )

            if not cols:
                raise ValueError("No design columns produced")

            cov_flat = np.concatenate(cols, axis=1)
            y_flat = None
            notNA = None

        else:
            # Backward-compatible path: rebuild using the same spec on the provided data.
            tmp = Formula(
                lhs=self.lhs,
                terms=self.terms,
                data=data,
                series_id=self.series_id,
                codebook=self.codebook,
                print_warn=self.print_warn,
                keep_cov=self.keep_cov,
                find_nested=self.find_nested,
                file_paths=[],
                file_loading_nc=self.file_loading_nc,
                file_loading_kwargs=self.file_loading_kwargs,
            )

            y_flat = tmp.get_depvar()
            cov_flat = np.asarray(tmp.Xs[0].todense())
            notNA = tmp.get_notNA()

        if self.series_id is None:
            return y_flat, cov_flat, notNA, None, None, None, None

        frame = data.obs if hasattr(data, "obs") else data
        sid = np.asarray(frame[self.series_id]).reshape(-1)
        _, idx = np.unique(sid, return_index=True)
        # preserve first occurrence order
        levels = sid[np.sort(idx)]

        y_s: list[np.ndarray] = []
        cov_s: list[np.ndarray] = []
        notNA_s: list[np.ndarray] = []
        split_points: list[tuple[int, int]] = []

        start = 0
        for lev in levels:
            mask = sid == lev
            rows = np.where(mask)[0]
            if rows.size == 0:
                continue
            y_i = y_flat[rows] if y_flat is not None else None
            cov_i = cov_flat[rows]
            notNA_i = notNA[rows] if notNA is not None else None

            end = start + int(rows.size)
            split_points.append((start, end))
            start = end

            if y_i is not None:
                y_s.append(y_i)
            cov_s.append(cov_i)
            if notNA_i is not None:
                notNA_s.append(notNA_i)

        return y_flat, cov_flat, notNA, y_s or None, cov_s or None, notNA_s or None, np.asarray(split_points)


__all__ = [
    "VarType",
    "lhs",
    "Formula",
    "InterceptTerm",
    "i",
    "LinearTerm",
    "l",
    "SmoothTerm",
    "UnsupportedTerm",
    "f",
    "fs",
]
