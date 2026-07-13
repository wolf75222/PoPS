"""Pre-launch platform and field-view gates used by the typed bind/install path."""
from __future__ import annotations

from typing import Any

from pops._platform_contracts import (
    ExecutionContext,
    FieldViewDescriptor,
    PlatformContractError,
    PlatformManifest,
    validate_launch,
)


def validate_platform_bind(platform: Any, context: Any, initial: Any, layout: Any) -> list[str]:
    """Return one actionable line; validation always finishes before native engine construction."""
    if type(platform) is not PlatformManifest:
        return ["compiled artifact carries no exact PlatformManifest"]
    if type(context) is not ExecutionContext:
        return ["InstallPlan carries no exact ExecutionContext"]
    try:
        fields = tuple(_initial_field(name, array, layout) for name, array in (initial or {}).items())
        validate_launch(platform, context, fields)
    except (PlatformContractError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _initial_field(name: str, array: Any, layout: Any) -> FieldViewDescriptor:
    shape = tuple(int(item) for item in getattr(array, "shape", ()) or ())
    mesh = _mesh_extent(layout)
    if len(shape) >= 2:
        extents = shape[-2:]
    elif len(shape) == 1 and mesh is not None and shape[0] == mesh * mesh:
        extents = (mesh, mesh)
    else:
        # Keep malformed/rank-deficient arrays representable so the ordinary initial-state gate can
        # report its richer shape error.  The generic descriptor itself must remain well formed.
        extents = (max(mesh or 1, 1), max(mesh or 1, 1))
    itemsize = int(getattr(getattr(array, "dtype", None), "itemsize", 8) or 8)
    byte_strides = tuple(int(item) for item in getattr(array, "strides", ()) or ())
    if len(byte_strides) >= 2:
        strides = tuple(max(abs(item) // itemsize, 1) for item in byte_strides[-2:])
    else:
        strides = (extents[1], 1)
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
        name=str(name), dimension=2, extents=tuple(extents), strides=strides,
        centering="cell", ghosts=((0, 0), (0, 0)), scalar=scalar,
        memory_space=memory, patch=str(name), layout=field_layout, ownership=ownership)


def _mesh_extent(layout: Any) -> int | None:
    mesh = getattr(layout, "mesh", None) or getattr(layout, "base", None)
    value = getattr(mesh, "n", None)
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return int(value) if value is not None else None


def _scalar_name(array: Any) -> str:
    dtype = getattr(array, "dtype", None)
    name = getattr(dtype, "name", None) or str(dtype or "")
    return {"double": "float64", "float": "float32"}.get(name, name)


__all__ = ["validate_platform_bind"]
