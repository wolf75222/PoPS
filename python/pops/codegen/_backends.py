"""Typed authority for the sole native production compiler.

The final public lifecycle deliberately has one compilation route. ``Production`` selects that
compiler and lowers to the internal ``"production"`` token consumed by the native package builder.
The exact platform manifest is evidence derived from the authenticated compiled components. Runtime
resources are supplied separately at ``pops.bind``; authoring never fabricates platform evidence.
Prototype/JIT and host-marshalled AOT descriptors do not exist here.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.params.use_sites import ParamUse, resolve_param_use


_PRODUCTION = "production"


class _Backend(Descriptor):
    """Private base used only to authenticate a compile-route descriptor."""

    category = "backend"


class Production(_Backend):
    """The native fixed-ABI production backend used by ``pops.compile``."""

    def __init__(self) -> None:
        pass

    @property
    def scheme(self) -> str:
        return _PRODUCTION

    @property
    def tier(self) -> str:
        return _PRODUCTION

    def lower(self, context: Any = None) -> str:
        return _PRODUCTION

    def options(self) -> dict[str, Any]:
        return {"backend": _PRODUCTION}

    def capabilities(self) -> Any:
        from pops.codegen._compile_emit import _BACKEND_CAPS
        from pops.descriptors_report import CapabilitySet

        return CapabilitySet(dict(_BACKEND_CAPS[_PRODUCTION]))

BACKEND_DESCRIPTORS = {_PRODUCTION: Production}


def lower_backend(backend: Any) -> str:
    """Lower the public descriptor or private native token; reject every other route."""

    backend = resolve_param_use(backend, ParamUse.BACKEND, where="compile(backend=)")
    if isinstance(backend, Production):
        return backend.lower()
    if backend == _PRODUCTION:
        return _PRODUCTION
    raise TypeError(
        "compile backend must be pops.codegen.Production(); prototype/JIT and "
        "host-marshalled AOT routes are not part of the final runtime"
    )


__all__ = ["Production"]
