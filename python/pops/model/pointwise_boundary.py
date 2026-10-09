"""Generic pointwise declaration reads and canonical resolver contract @1.

Shared by boundary and field consumers; no field-solve, compiler or runtime authority.
The existing expression and reference semantics are retained.
"""
from __future__ import annotations
from typing import Any
from pops._ir.expr import Const, Expr
from pops.model.handles import Handle


def resolve_handle(reference: Any, resolver: Any, *, where: str) -> Any:
    """Authenticate one declaration through an assembly resolver."""
    if not isinstance(reference, Handle):
        raise TypeError("%s must be a declaration Handle" % where)
    resolve = resolver if callable(resolver) else getattr(resolver, "resolve", None)
    if not callable(resolve):
        raise TypeError(
            "%s reference resolution requires a callable resolver or an object exposing "
            "resolve(handle)" % where
        )
    resolved = resolve(reference)
    if not isinstance(resolved, Handle) or not resolved.is_resolved:
        raise TypeError("%s resolver must return a canonical Handle" % where)
    return resolved


def _component_index(handle: Handle, component: Any) -> tuple[int, str | None]:
    if handle.kind == "field":
        if component not in (None, 0):
            raise ValueError("a scalar field boundary read accepts only component 0")
        return 0, None
    if handle.kind != "state":
        raise TypeError("boundary_value requires a state or field Handle")
    declaration = handle.declaration_ref or handle
    space = getattr(declaration, "space", None)
    components = tuple(getattr(space, "components", ()))
    if not components:
        components = tuple(getattr(declaration, "components", ()))
    if not components:
        raise TypeError(
            "state boundary reads require authoritative component metadata on the Handle")
    if isinstance(component, str):
        try:
            return components.index(component), component
        except ValueError:
            raise KeyError(
                "state %r has no component %r (have: %s)"
                % (handle.local_id, component, ", ".join(components))) from None
    if isinstance(component, bool) or not isinstance(component, int):
        raise TypeError("a state boundary read requires a component name or integer index")
    if component < 0 or component >= len(components):
        raise IndexError("state boundary component index %d is out of range" % component)
    return component, components[component]


class BoundaryValue(Expr):
    """Pointwise read of one resolved state component or scalar field."""

    __slots__ = ("handle", "component", "component_name")

    def __init__(self, handle: Any, component: Any = None) -> None:
        if not isinstance(handle, Handle):
            raise TypeError("BoundaryValue requires a declaration Handle")
        index, name = _component_index(handle, component)
        self.handle = handle
        self.component = index
        self.component_name = name

    def resolve_references(self, resolver: Any) -> BoundaryValue:
        resolved = resolve_handle(self.handle, resolver, where="BoundaryValue handle")
        return BoundaryValue(resolved, self.component)

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.handle,)

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("boundary_value", self.handle.qualified_id, self.component)

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Expr:
        target_handle = getattr(target, "handle", target)
        same = isinstance(target_handle, Handle) and target_handle == self.handle
        return Const(1 if same and self.component == 0 else 0)

    def eval(self, env: Any) -> Any:
        key = (self.handle.qualified_id, self.component)
        if key not in env:
            raise KeyError("boundary value %r missing from the environment" % (key,))
        return env[key]

    def deps(self) -> set[Any]:
        return {(self.handle.qualified_id, self.component)}

    def _str(self) -> str:
        suffix = self.component_name if self.component_name is not None else self.component
        return "boundary_value(%s, %r)" % (self.handle.qualified_id, suffix)
