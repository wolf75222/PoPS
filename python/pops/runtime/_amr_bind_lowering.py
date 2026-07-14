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
) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float], bool]:
    """Authenticate one Cartesian grid and project only native-representable topology.

    The native AMR config has one global periodic flag. A partial axis partition is therefore an
    unavailable backend route, never a reason to erase the authored topology.
    """
    from pops.mesh.grid import CartesianGrid

    grid = CartesianGrid.from_dict(data)
    periodic_axes = grid.topology.periodic_axes
    if periodic_axes and len(periodic_axes) != len(grid.axes):
        raise NotImplementedError(
            "native AmrSystemConfig has one global periodic flag and cannot represent a partially "
            "periodic CartesianGrid topology"
        )
    return grid.cells, grid.frame.lower, grid.frame.upper, bool(periodic_axes)


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


def amr_config_from_layout(layout: Any, *, hierarchy: Any = None) -> Any:
    """Build ``AmrSystemConfig`` without inferring or dropping authored facts."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh._amr import ResolvedHierarchy

    data = _runtime_data(layout)
    cells, lower, upper, periodic = _native_amr_grid_values(data["grid"])
    lengths = (upper[0] - lower[0], upper[1] - lower[1])
    if cells[0] != cells[1] or lower != (0.0, 0.0) or lengths[0] != lengths[1]:
        raise NotImplementedError(
            "the installed native AMR provider currently requires a square [0,L]^2 grid; "
            "the full rectangular grid remains authenticated and is never collapsed"
        )
    if type(hierarchy) is not ResolvedHierarchy:
        raise TypeError("adaptive runtime requires an exact resolved hierarchy")
    transitions = hierarchy.plan.transitions
    if any(row.dimension != 2 or row.ratio != (2, 2) for row in transitions):
        raise NotImplementedError(
            "native AMR requires exact two-dimensional ratio-(2,2) transitions")
    buffers = {row.buffer for row in transitions}
    lookaheads = {row.lookahead for row in transitions}
    if len(buffers) != 1 or len(lookaheads) != 1:
        raise NotImplementedError(
            "native AMR exposes one global nesting buffer/lookahead")
    buffer = next(iter(buffers))
    if len(set(buffer)) != 1:
        raise NotImplementedError("native AMR cannot lower anisotropic nesting buffers")

    cfg = AmrSystemConfig()
    cfg.n = cells[0]
    cfg.L = lengths[0]
    cfg.periodic = periodic
    cfg.level_count = hierarchy.plan.level_count
    cfg.regrid_margin = buffer[0]
    cfg.regrid_grow = next(iter(lookaheads))
    cfg.regrid_every = _regrid_every(data)
    cfg.explicit_bootstrap = True

    cluster = hierarchy.plan.clustering.options.to_data()
    clustering_provider = cluster.get("provider")
    patches = hierarchy.plan.patch_generation.options.to_data()
    balance = hierarchy.plan.load_balance.options.to_data()
    if not isinstance(clustering_provider, dict) \
            or clustering_provider.get("provider_type") not in {
                "builtin_amr_clustering", "external_amr_clustering"} \
            or patches.get("native_route") != "box_array" \
            or balance != {"native_route": "round_robin"}:
        raise NotImplementedError("resolved hierarchy selected an unavailable native provider")
    if clustering_provider["provider_type"] == "builtin_amr_clustering":
        cfg.cluster_min_efficiency = _native_binary64(
            clustering_provider["minimum_efficiency"],
            where="AMR clustering minimum_efficiency")
        cfg.cluster_min_box_size = int(clustering_provider["minimum_box_size"])
        cfg.cluster_max_box_size = int(clustering_provider["maximum_box_size"])
    cfg.distribute_coarse = bool(patches.get("distribute_coarse", False))
    cfg.coarse_max_grid = int(patches.get("coarse_max_grid", 0))
    return cfg


__all__ = ["amr_config_from_layout"]
