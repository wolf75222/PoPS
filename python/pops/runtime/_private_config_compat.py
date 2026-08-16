"""Private constructor compatibility for the retired scalar runtime shorthand.

The native config PODs intentionally expose only exact-ranked geometry.  A small
number of internal tests still construct the private ``System``/``AmrSystem``
seams directly with the historical scalar keywords.  This module translates
those keywords *before* the pybind setters run; it is not part of public layout
resolution and does not add fields to either native config type.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any


_EXACT_GEOMETRY = frozenset(("shape", "lower", "upper"))
_COMMON_ALIASES = frozenset(("n", "L", "xlo"))
_AMR_AXIS_ONE_ALIASES = frozenset(("ny", "Ly", "ylo"))
_RETIRED_ALIASES = frozenset(("regrid_grow", "regrid_margin"))
_UNSUPPORTED_Z_ALIASES = frozenset(("nz", "Lz", "zlo"))


def _ranked_default(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("private runtime %s default must be one non-empty tuple" % name)
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("private runtime %s must be one integer scalar" % name)
    return int(value)


def _real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("private runtime %s must be one real scalar" % name)
    return float(value)


def _validate_periodicity(value: Any, *, dimension: int) -> tuple[bool, ...]:
    if not isinstance(value, tuple) or len(value) != dimension or any(
        type(item) is not bool for item in value
    ):
        raise TypeError(
            "private runtime periodicity must be an exact bool tuple of length %d" % dimension
        )
    return value


def private_constructor_config(
    config_type: Any,
    keywords: dict[str, Any],
    *,
    runtime: str,
    adaptive: bool,
) -> Any:
    """Construct an exact native config, translating only documented private aliases.

    ``dimension`` comes exclusively from the fresh config's default ``shape``.
    In particular, scalar aliases never select a native specialization and no
    third-axis spelling is invented.
    """
    unsupported = _RETIRED_ALIASES.intersection(keywords)
    if unsupported:
        raise NotImplementedError(
            "%s no longer accepts retired %s"
            % (runtime, "/".join(sorted(unsupported)))
        )
    z_aliases = _UNSUPPORTED_Z_ALIASES.intersection(keywords)
    if z_aliases:
        raise TypeError(
            "%s accepts no z-axis scalar aliases; pass exact shape/lower/upper"
            % runtime
        )
    axis_one = _AMR_AXIS_ONE_ALIASES.intersection(keywords)
    if axis_one and not adaptive:
        raise TypeError("System accepts no ny/Ly/ylo aliases")

    config = config_type()
    shape = list(_ranked_default(config.shape, name="shape"))
    lower = list(_ranked_default(config.lower, name="lower"))
    upper = list(_ranked_default(config.upper, name="upper"))
    dimension = len(shape)
    if len(lower) != dimension or len(upper) != dimension:
        raise ValueError("private runtime config defaults do not share one rank")

    if "periodicity" in keywords:
        _validate_periodicity(keywords["periodicity"], dimension=dimension)

    aliases = _COMMON_ALIASES | (_AMR_AXIS_ONE_ALIASES if adaptive else frozenset())
    used_aliases = aliases.intersection(keywords)
    exact_geometry = _EXACT_GEOMETRY.intersection(keywords)
    if used_aliases and exact_geometry:
        raise ValueError(
            "%s cannot mix scalar aliases (%s) with exact geometry (%s)"
            % (runtime, ", ".join(sorted(used_aliases)), ", ".join(sorted(exact_geometry)))
        )

    if not used_aliases:
        for name, value in keywords.items():
            if adaptive and name == "coarse_max_grid" and not isinstance(value, tuple):
                value = (_integer(value, name="coarse_max_grid"),) * dimension
            setattr(config, name, value)
        return config

    lengths = [high - low for low, high in zip(lower, upper, strict=True)]
    if "n" in keywords:
        shape[:] = [_integer(keywords["n"], name="n")] * dimension
    if "L" in keywords:
        lengths[:] = [_real(keywords["L"], name="L")] * dimension
    if "xlo" in keywords:
        lower[0] = _real(keywords["xlo"], name="xlo")

    if axis_one:
        if dimension < 2:
            raise ValueError("AmrSystem ny/Ly/ylo aliases require a native rank of at least two")
        if "ny" in keywords:
            shape[1] = _integer(keywords["ny"], name="ny")
        if "Ly" in keywords:
            lengths[1] = _real(keywords["Ly"], name="Ly")
        if "ylo" in keywords:
            lower[1] = _real(keywords["ylo"], name="ylo")

    config.shape = tuple(shape)
    config.lower = tuple(lower)
    config.upper = tuple(low + length for low, length in zip(lower, lengths, strict=True))

    for name, value in keywords.items():
        if name in _COMMON_ALIASES or name in _AMR_AXIS_ONE_ALIASES:
            continue
        if adaptive and name == "coarse_max_grid" and not isinstance(value, tuple):
            value = (_integer(value, name="coarse_max_grid"),) * dimension
        setattr(config, name, value)
    return config


__all__ = ["private_constructor_config"]
