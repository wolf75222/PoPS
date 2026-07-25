"""Exact lowering of an adaptive layout through its open runtime-data protocol."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _runtime_data(layout: Any) -> dict[str, Any]:
    protocol = getattr(layout, "runtime_layout_data", None)
    if not callable(protocol):
        raise TypeError(
            "adaptive runtime layouts must implement runtime_layout_data(); "
            "concrete layout classes are not dispatched centrally"
        )
    first, second = protocol(), protocol()
    if type(first) is not dict or first != second:
        raise TypeError("runtime_layout_data() must return one deterministic dict")
    if first.get("schema_version") != 1 \
            or first.get("layout_type") != "adaptive_cartesian":
        raise ValueError("adaptive runtime layout uses an unsupported protocol schema")
    return first


def _regrid_every(data: dict[str, Any]) -> int:
    schedule = data["regrid"]["schedule"]
    if schedule["domain"]["type"] != "accepted_step":
        raise ValueError("native AMR regrid schedule must use AcceptedStep")
    trigger = schedule["trigger"]
    if trigger["type"] == "always":
        return 1
    if trigger["type"] == "every":
        value = trigger["n"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("native AMR Every cadence must be an integer >= 1")
        return value
    raise ValueError("native AMR supports Always/Every regrid triggers")


def _native_amr_grid_values(
    data: Any,
) -> tuple[
    tuple[int, int], tuple[float, float], tuple[float, float], tuple[bool, bool]
]:
    """Authenticate one Cartesian grid without collapsing its axis topology."""
    from pops.mesh.grid import CartesianGrid

    grid = CartesianGrid.from_dict(data)
    periodic_axes = grid.topology.periodic_axes
    periodic_indices = {axis.index for axis in periodic_axes}
    return (
        grid.cells,
        grid.frame.lower,
        grid.frame.upper,
        (0 in periodic_indices, 1 in periodic_indices),
    )


def _physical_patch_rectangles(
    patch_boxes: Any,
    *,
    cells: tuple[int, int],
    lengths: tuple[float, float],
    lower: tuple[float, float],
) -> list[tuple[float, float, float, float]]:
    """Map inclusive AMR index boxes to exact Cartesian physical rectangles."""
    nx, ny = cells
    lx, ly = lengths
    xlo, ylo = lower
    result: list[tuple[float, float, float, float]] = []
    for level, ilo, jlo, ihi, jhi in patch_boxes:
        dx = lx / (nx << level)
        dy = ly / (ny << level)
        result.append((
            xlo + ilo * dx,
            ylo + jlo * dy,
            (ihi - ilo + 1) * dx,
            (jhi - jlo + 1) * dy,
        ))
    return result


def _native_binary64(value: Any, *, where: str) -> float:
    if isinstance(value, Mapping) and set(value) == {"binary64"} \
            and isinstance(value["binary64"], str):
        result = float.fromhex(value["binary64"])
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be one canonical binary64 value" % where)
    else:
        result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError("%s must be in (0, 1]" % where)
    return result


def _native_patch_generation_values(options: Any) -> tuple[bool, int]:
    """Lower the exact public patch authority into the current native provider ABI."""
    expected = {"native_route", "distribute_coarse", "coarse_max_grid"}
    if type(options) is not dict or set(options) != expected:
        raise TypeError("native AMR patch generation requires the exact box_array option schema")
    if options["native_route"] != "box_array":
        raise NotImplementedError(
            "native AMR patch generation requires native_route='box_array'"
        )
    distribute_coarse = options["distribute_coarse"]
    if type(distribute_coarse) is not bool:
        raise TypeError("native AMR distribute_coarse must be an exact bool")
    authored_max_grid = options["coarse_max_grid"]
    if authored_max_grid is None:
        return distribute_coarse, 0
    if type(authored_max_grid) is not int:
        raise TypeError("native AMR coarse_max_grid must be None or an exact integer")
    if authored_max_grid < 1:
        raise ValueError("native AMR coarse_max_grid must be positive when provided")
    if authored_max_grid > 2_147_483_647:
        raise OverflowError("native AMR coarse_max_grid exceeds the signed 32-bit provider ABI")
    return distribute_coarse, authored_max_grid


def _native_load_balance_options(options: dict[str, Any]) -> dict[str, Any]:
    """Decode the canonical provider value language into the native variant ABI."""
    result: dict[str, Any] = {}
    for key, value in options.items():
        if type(value) is dict and set(value) == {"binary64"}:
            result[key] = float.fromhex(value["binary64"])
        else:
            result[key] = value
    return result


def amr_config_from_layout(layout: Any, *, hierarchy: Any = None) -> Any:
    """Build ``AmrSystemConfig`` without inferring or dropping authored facts."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh._amr import ResolvedHierarchy

    data = _runtime_data(layout)
    cells, lower, upper, periodicity = _native_amr_grid_values(data["grid"])
    lengths = (upper[0] - lower[0], upper[1] - lower[1])
    if type(hierarchy) is not ResolvedHierarchy:
        raise TypeError("adaptive runtime requires an exact resolved hierarchy")
    from pops.mesh._amr.hierarchy_native import lower_native_hierarchy

    native_hierarchy = lower_native_hierarchy(hierarchy)

    cfg = AmrSystemConfig()
    cfg.n = cells[0]
    cfg.ny = cells[1]
    cfg.L = lengths[0]
    cfg.Ly = lengths[1]
    cfg.xlo = lower[0]
    cfg.ylo = lower[1]
    cfg.periodicity = periodicity
    cfg.level_count = native_hierarchy.level_count
    cfg.regrid_margin = native_hierarchy.nesting_buffer
    cfg.regrid_grow = native_hierarchy.nesting_lookahead
    cfg.regrid_every = _regrid_every(data)
    cfg.explicit_bootstrap = True

    cluster = hierarchy.plan.clustering.options.to_data()
    clustering_provider = cluster.get("provider")
    patches = hierarchy.plan.patch_generation.options.to_data()
    balance = hierarchy.plan.load_balance.options.to_data()
    distribute_coarse, coarse_max_grid = _native_patch_generation_values(patches)
    if type(balance) is not dict or set(balance) != {"provider"}:
        raise TypeError(
            "resolved AMR load balance must preserve one exact provider authority")
    from pops.amr._load_balance_contract import validate_load_balance_provider_data
    from pops.amr.providers import prepare_amr_provider_native_config

    balance_provider = validate_load_balance_provider_data(balance["provider"])
    if data.get("load_balance") != balance_provider:
        raise ValueError(
            "resolved hierarchy load balance differs from the adaptive layout authority")
    prepared_clustering = prepare_amr_provider_native_config(clustering_provider)
    if prepared_clustering.role != "clustering":
        raise ValueError("resolved hierarchy selected a non-clustering provider")
    native_config_converters = {
        "cluster_min_efficiency": lambda value: _native_binary64(
            value, where="AMR clustering minimum_efficiency"),
        "cluster_min_box_size": int,
        "cluster_max_box_size": int,
    }
    if not set(prepared_clustering.config) <= set(native_config_converters):
        raise NotImplementedError("AMR clustering provider emitted an unsupported native control")
    for name, value in prepared_clustering.config.items():
        setattr(cfg, name, native_config_converters[name](value))
    cfg.distribute_coarse = distribute_coarse
    cfg.coarse_max_grid = coarse_max_grid
    cfg._set_load_balance_provider(
        balance_provider["native_route"],
        balance_provider["provider_identity"],
        balance_provider["option_schema_identity"],
        _native_load_balance_options(balance_provider["options"]),
    )
    return cfg


__all__ = ["amr_config_from_layout"]
