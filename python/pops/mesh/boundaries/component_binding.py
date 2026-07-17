"""Qualified native-component bindings for executable boundary Handles.

A binding is deliberately smaller than an external component package.  It retains only the exact
Handle and authenticated native identity that must survive resolve -> compile -> bind.  Installation
then resolves that identity against ``InstallPlan.components``; it never selects an implementation
because it happens to be the only component of a given interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity
from pops.model import Handle


from pops._generated_component_interfaces import NATIVE_COMPONENT_BOUNDARY_HANDLE_ROUTES


def _component_identity(component: Any) -> tuple[str, Identity, Any]:
    from pops.external import CompiledComponentArtifact, ExternalComponent

    if type(component) is ExternalComponent:
        return (
            component.component_manifest.component_id,
            component.component_manifest.manifest_digest,
            component.component_type.interface,
        )
    if type(component) is CompiledComponentArtifact:
        component.verify()
        return component.component_id, component.component_manifest, component.interface
    raise TypeError(
        "BoundaryComponentBinding component must be an exact ExternalComponent or "
        "CompiledComponentArtifact"
    )


@dataclass(frozen=True, slots=True, init=False)
class BoundaryComponentBinding:
    """Bind one owner-qualified boundary operation Handle to one exact native component.

    The operation and required interface are inferred from ``Handle.kind``.  Callers cannot claim a
    residual table for a ghost producer, or a JVP table for a residual Handle, by passing strings.
    """

    target: Handle
    component_id: str
    component_manifest_identity: Identity
    native_interface: Any
    operation: str
    interface_version: int = field(init=False)

    def __init__(self, target: Handle, component: Any) -> None:
        if isinstance(target, str) or not isinstance(target, Handle) or not target.is_resolved:
            raise TypeError(
                "BoundaryComponentBinding.target requires a canonical owner-qualified Handle"
            )
        try:
            interface_name, operation = NATIVE_COMPONENT_BOUNDARY_HANDLE_ROUTES[target.kind]
        except KeyError:
            raise TypeError(
                "BoundaryComponentBinding.target kind %r is not an executable boundary "
                "provider/resolver/residual/JVP Handle" % target.kind
            ) from None
        component_id, manifest_identity, interface = _component_identity(component)
        from pops import interfaces

        expected = interfaces.resolve(interface_name)
        if interface != expected:
            raise TypeError(
                "BoundaryComponentBinding target %s requires exact interface %s@%d, got %s@%d"
                % (
                    target.qualified_id,
                    expected.uri,
                    expected.version,
                    interface.uri,
                    interface.version,
                )
            )
        if operation not in interface.operations:
            raise ValueError(
                "native interface %s@%d does not export required operation %s"
                % (interface.uri, interface.version, operation)
            )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "component_manifest_identity", manifest_identity)
        object.__setattr__(self, "native_interface", interface)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "interface_version", interface.version)

    @property
    def qualified_id(self) -> str:
        return self.target.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "binding_type": "boundary_native_component",
            "target": self.target.canonical_identity(),
            "operation": self.operation,
            "component_id": self.component_id,
            "component_manifest_identity": self.component_manifest_identity.token,
            "native_interface": self.native_interface.to_data(),
            "interface_version": self.interface_version,
        }

    def require_component(self, component: Any) -> None:
        """Prove one resolve/compile component input is exactly the authored authority."""
        component_id, manifest_identity, interface = _component_identity(component)
        if component_id != self.component_id:
            raise ValueError(
                "boundary binding for %s requires component %r, got %r"
                % (self.target.qualified_id, self.component_id, component_id)
            )
        if manifest_identity != self.component_manifest_identity:
            raise ValueError(
                "boundary binding for %s changed component manifest identity"
                % self.target.qualified_id
            )
        if interface != self.native_interface or interface.version != self.interface_version:
            raise ValueError(
                "boundary binding for %s changed native interface identity/version"
                % self.target.qualified_id
            )


__all__ = ["BoundaryComponentBinding"]
