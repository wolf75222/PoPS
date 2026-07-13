"""Exact lowering of an adaptive layout through its open runtime-data protocol."""
from __future__ import annotations

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


def amr_config_from_layout(layout: Any, *, hierarchy: Any = None) -> Any:
    """Build ``AmrSystemConfig`` without inferring or dropping authored facts."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh.amr import ResolvedHierarchy

    data = _runtime_data(layout)
    grid = data["grid"]
    cells = tuple(grid["cells"])
    lower = tuple(grid["extent"]["lower"])
    upper = tuple(grid["extent"]["upper"])
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
    cfg.periodic = False
    cfg.level_count = hierarchy.plan.level_count
    cfg.regrid_margin = buffer[0]
    cfg.regrid_grow = next(iter(lookaheads))
    cfg.regrid_every = _regrid_every(data)
    cfg.explicit_bootstrap = True

    cluster = hierarchy.plan.clustering.options.to_data()
    patches = hierarchy.plan.patch_generation.options.to_data()
    balance = hierarchy.plan.load_balance.options.to_data()
    if cluster.get("native_route") != "berger_rigoutsos" \
            or patches.get("native_route") != "box_array" \
            or balance != {"native_route": "round_robin"}:
        raise NotImplementedError("resolved hierarchy selected an unavailable native provider")
    cfg.cluster_min_efficiency = float(cluster.get("minimum_efficiency", 0.0))
    cfg.cluster_min_box_size = int(cluster.get("minimum_box_size", 0))
    cfg.cluster_max_box_size = int(cluster.get("maximum_box_size", 0))
    cfg.distribute_coarse = bool(patches.get("distribute_coarse", False))
    cfg.coarse_max_grid = int(patches.get("coarse_max_grid", 0))
    return cfg


__all__ = ["amr_config_from_layout"]
