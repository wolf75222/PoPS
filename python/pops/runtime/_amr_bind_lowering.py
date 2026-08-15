"""Exact lowering of an adaptive layout through its open runtime-data protocol."""

from __future__ import annotations

from collections.abc import Mapping
import math
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
    if first.get("schema_version") != 1 or first.get("layout_type") != "adaptive_cartesian":
        raise ValueError("adaptive runtime layout uses an unsupported protocol schema")
    return first


def _regrid_every(data: dict[str, Any]) -> int:
    regrid = data["regrid"]
    if regrid == {
        "schema_version": 1,
        "authority_type": "amr_regrid",
        "mode": "frozen",
    }:
        return 0
    schedule = regrid["schedule"]
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
    native_layout: Any,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[bool, ...]]:
    """Authenticate the exact layout-derived geometry before allocating ``AmrSystemConfig``."""
    from pops.mesh import NativeSpatialLayout

    if type(native_layout) is not NativeSpatialLayout:
        raise TypeError("native AMR lowering requires an exact NativeSpatialLayout")
    dimension = native_layout.dimension
    expected_coordinates = "pops://coordinates/cartesian-%dd@1" % dimension
    if (
        native_layout.coordinate_system != expected_coordinates
        or native_layout.centering != "cell"
        or native_layout.decomposition.get("kind") != "adaptive"
    ):
        raise NotImplementedError(
            "native AmrSystemConfig requires cell-centered Cartesian AMR matching its exact "
            "spatial rank"
        )
    return (
        native_layout.shape,
        native_layout.lower,
        native_layout.upper,
        native_layout.periodicity,
    )


def _physical_patch_bounds(
    patch_boxes: Any,
    *,
    cells: tuple[int, ...],
    lengths: tuple[float, ...],
    lower: tuple[float, ...],
) -> list[tuple[float, ...]]:
    """Map ranked inclusive AMR index boxes to physical lower bounds and extents."""
    dimension = len(cells)
    if dimension not in (1, 2, 3) or len(lengths) != dimension or len(lower) != dimension:
        raise ValueError("physical AMR patch projection requires one exact spatial rank")
    if any(type(value) is not int or value < 1 for value in cells):
        raise TypeError("physical AMR patch cell extents must be exact positive integers")
    if any(not math.isfinite(value) or value <= 0.0 for value in lengths):
        raise ValueError("physical AMR patch lengths must be finite and positive")
    if any(not math.isfinite(value) for value in lower):
        raise ValueError("physical AMR patch lower bounds must be finite")
    result: list[tuple[float, ...]] = []
    for position, row in enumerate(patch_boxes):
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise TypeError("AMR patch %d must contain level, lower, and upper" % position)
        level, index_lower, index_upper = row
        if type(level) is not int or level < 0:
            raise TypeError("AMR patch levels must be exact non-negative integers")
        index_lower = tuple(index_lower)
        index_upper = tuple(index_upper)
        if (
            len(index_lower) != dimension
            or len(index_upper) != dimension
            or any(type(value) is not int for value in index_lower + index_upper)
        ):
            raise TypeError("AMR patch bounds must match the exact spatial rank")
        if any(high < low for low, high in zip(index_lower, index_upper, strict=True)):
            raise ValueError("AMR patch bounds must be non-empty on every axis")
        spacing = tuple(
            length / (extent << level) for length, extent in zip(lengths, cells, strict=True)
        )
        physical_lower = tuple(
            origin + index * width
            for origin, index, width in zip(lower, index_lower, spacing, strict=True)
        )
        physical_extents = tuple(
            (high - low + 1) * width
            for low, high, width in zip(index_lower, index_upper, spacing, strict=True)
        )
        result.append(physical_lower + physical_extents)
    return result


def _native_binary64(value: Any, *, where: str) -> float:
    if (
        isinstance(value, Mapping)
        and set(value) == {"binary64"}
        and isinstance(value["binary64"], str)
    ):
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
        raise NotImplementedError("native AMR patch generation requires native_route='box_array'")
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


def _install_native_hierarchy_config(
    config: Any, lowering: Any, *, dimension: int
) -> None:
    """Install every hierarchy-v2 transition without reducing ranked facts to scalars."""
    from pops.mesh._amr.hierarchy_native import PreparedHierarchyNativeLowering

    if type(lowering) is not PreparedHierarchyNativeLowering:
        raise TypeError(
            "native AMR config requires an exact PreparedHierarchyNativeLowering"
        )
    if lowering.dimension != dimension:
        raise ValueError(
            "native AMR hierarchy dimension differs from the selected config specialization"
        )
    config.level_count = lowering.level_count
    config.transition_ratios = tuple(tuple(row) for row in lowering.transition_ratios)
    config.transition_buffers = tuple(tuple(row) for row in lowering.transition_buffers)
    config.transition_lookaheads = tuple(
        (value,) * dimension for value in lowering.transition_lookaheads
    )


def amr_config_from_layout(
    layout: Any,
    *,
    hierarchy: Any = None,
    native_layout: Any,
) -> Any:
    """Build ``AmrSystemConfig`` without inferring or dropping authored facts."""
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh._amr import ResolvedHierarchy

    data = _runtime_data(layout)
    cells, lower, upper, periodicity = _native_amr_grid_values(native_layout)
    if type(hierarchy) is not ResolvedHierarchy:
        raise TypeError("adaptive runtime requires an exact resolved hierarchy")
    from pops.mesh._amr.hierarchy_native import lower_native_hierarchy

    native_hierarchy = lower_native_hierarchy(hierarchy)

    cfg = AmrSystemConfig()
    cfg.shape = cells
    cfg.lower = lower
    cfg.upper = upper
    cfg.periodicity = periodicity
    _install_native_hierarchy_config(
        cfg, native_hierarchy, dimension=len(cells)
    )
    cfg.regrid_every = _regrid_every(data)
    cfg.explicit_bootstrap = True

    cluster = hierarchy.plan.clustering.options.to_data()
    clustering_provider = cluster.get("provider")
    patches = hierarchy.plan.patch_generation.options.to_data()
    balance = hierarchy.plan.load_balance.options.to_data()
    distribute_coarse, coarse_max_grid = _native_patch_generation_values(patches)
    if type(balance) is not dict or set(balance) != {"provider"}:
        raise TypeError("resolved AMR load balance must preserve one exact provider authority")
    from pops.amr._load_balance_contract import validate_load_balance_provider_data
    from pops.amr.providers import prepare_amr_provider_native_config

    balance_provider = validate_load_balance_provider_data(balance["provider"])
    if data.get("load_balance") != balance_provider:
        raise ValueError(
            "resolved hierarchy load balance differs from the adaptive layout authority"
        )
    prepared_clustering = prepare_amr_provider_native_config(clustering_provider)
    if prepared_clustering.role != "clustering":
        raise ValueError("resolved hierarchy selected a non-clustering provider")
    native_config_converters = {
        "cluster_min_efficiency": lambda value: _native_binary64(
            value, where="AMR clustering minimum_efficiency"
        ),
        "cluster_min_box_size": int,
        "cluster_max_box_size": int,
    }
    if not set(prepared_clustering.config) <= set(native_config_converters):
        raise NotImplementedError("AMR clustering provider emitted an unsupported native control")
    for name, value in prepared_clustering.config.items():
        setattr(cfg, name, native_config_converters[name](value))
    cfg.distribute_coarse = distribute_coarse
    cfg.coarse_max_grid = (coarse_max_grid,) * len(cells)
    cfg._set_load_balance_provider(
        balance_provider["native_route"],
        balance_provider["provider_identity"],
        balance_provider["option_schema_identity"],
        _native_load_balance_options(balance_provider["options"]),
    )
    return cfg


__all__ = ["amr_config_from_layout"]
