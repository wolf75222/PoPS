"""Pre-launch platform and field-view gates used by the typed bind/install path."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from pops._platform_contracts import (
    ExecutionContext,
    FieldViewDescriptor,
    PlatformContractError,
    PlatformManifest,
    validate_launch,
)


def validate_platform_bind(
    platform: Any,
    context: Any,
    initial: Any,
    compiled_plan: Any,
) -> list[str]:
    """Return one actionable line; validation always finishes before native engine construction."""
    if type(platform) is not PlatformManifest:
        return ["compiled artifact carries no exact PlatformManifest"]
    if type(context) is not ExecutionContext:
        return ["InstallPlan carries no exact ExecutionContext"]
    try:
        dimension, mesh_shapes = _compiled_spatial_facts(compiled_plan)
        fields = tuple(
            _initial_field(name, array, dimension, mesh_shapes)
            for name, array in (initial or {}).items()
        )
        validate_launch(platform, context, fields)
    except (PlatformContractError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _compiled_spatial_facts(
    compiled_plan: Any,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Authenticate the one compiled rank and every exact layout shape."""
    dimension = getattr(compiled_plan, "resolved_dimension", None)
    native_layouts = getattr(compiled_plan, "native_layouts", None)
    if isinstance(dimension, bool) or type(dimension) is not int \
            or dimension not in (1, 2, 3):
        raise TypeError("compiled plan has no exact resolved spatial dimension")
    if not isinstance(native_layouts, Mapping) or not native_layouts:
        raise TypeError("compiled plan has no exact native spatial layouts")
    shapes = tuple(tuple(getattr(layout, "shape", ())) for layout in native_layouts.values())
    if any(
        len(shape) != dimension
        or any(isinstance(extent, bool) or type(extent) is not int or extent < 1
               for extent in shape)
        for shape in shapes
    ):
        raise TypeError("compiled native spatial layouts differ from the resolved rank")
    return dimension, tuple(dict.fromkeys(shapes))


def _right_strides(extents: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(extents[axis + 1:]) for axis in range(len(extents)))


def _initial_field(
    name: str,
    array: Any,
    dimension: int,
    mesh_shapes: tuple[tuple[int, ...], ...],
) -> FieldViewDescriptor:
    shape = tuple(int(item) for item in getattr(array, "shape", ()) or ())
    inferred_from_layout = False
    if len(shape) == 1:
        candidates = tuple(
            candidate for candidate in mesh_shapes
            if shape[0] % math.prod(candidate) == 0
        )
        if len(candidates) == 1:
            # Native mesh axes are coordinate ordered; NumPy exposes the same
            # field with its spatial axes reversed at the binding boundary.
            extents = tuple(reversed(candidates[0]))
            inferred_from_layout = True
        elif dimension == 1:
            extents = shape
        else:
            extents = (1,) * dimension
    elif len(shape) >= dimension:
        extents = shape[-dimension:]
    else:
        # Keep malformed/rank-deficient arrays representable so the ordinary initial-state gate can
        # report its richer shape error.  The generic descriptor itself must remain well formed.
        extents = (1,) * dimension
    itemsize = int(getattr(getattr(array, "dtype", None), "itemsize", 8) or 8)
    byte_strides = tuple(int(item) for item in getattr(array, "strides", ()) or ())
    if not inferred_from_layout and len(byte_strides) >= dimension:
        strides = tuple(
            max(abs(item) // itemsize, 1) for item in byte_strides[-dimension:]
        )
    else:
        strides = _right_strides(extents)
    flags = getattr(array, "flags", None)
    if flags is not None and bool(getattr(flags, "c_contiguous", False)):
        field_layout = "right"
    elif flags is not None and bool(getattr(flags, "f_contiguous", False)):
        field_layout = "left"
    else:
        field_layout = "strided"
    scalar = _scalar_name(array)
    memory = "device" if hasattr(array, "__cuda_array_interface__") else "host"
    ownership = "owned" if getattr(array, "base", None) is None else "borrowed"
    return FieldViewDescriptor(
        name=str(name), dimension=dimension, extents=tuple(extents), strides=strides,
        centering="cell", ghosts=((0, 0),) * dimension, scalar=scalar,
        memory_space=memory, patch=str(name), layout=field_layout, ownership=ownership)


def _scalar_name(array: Any) -> str:
    dtype = getattr(array, "dtype", None)
    name = getattr(dtype, "name", None) or str(dtype or "")
    return {"double": "float64", "float": "float32"}.get(name, name)


__all__ = ["validate_platform_bind"]
