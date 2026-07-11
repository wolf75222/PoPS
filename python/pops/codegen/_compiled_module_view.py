"""Detached operator/type introspection retained by a compiled artifact."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

from pops.model.manifest_data import freeze_json, thaw_json


class CompiledModuleView:
    """Small immutable view; never retains a Module, registry, or model builder."""

    __slots__ = ("state_spaces", "field_spaces", "_operators")

    def __init__(self, model: Any, manifest: Any = None) -> None:
        registry = None
        registry_of = getattr(model, "operator_registry", None)
        if callable(registry_of):
            registry = registry_of()

        if manifest is not None:
            state_spaces = tuple(manifest.state_spaces)
            field_spaces = tuple(manifest.field_spaces)
            entries = tuple(manifest.operators)
        elif registry is not None:
            state_spaces = _space_names(model, "list_state_spaces")
            field_spaces = _space_names(model, "list_field_spaces")
            entries = tuple(registry)
        else:
            state_spaces = field_spaces = entries = ()

        rows = {}
        for entry in entries:
            name = entry.name
            signature = getattr(entry, "signature", None)
            if registry is not None:
                # Signature/Space values are immutable value objects.  Capture the typed value so
                # the public operator_signature() contract survives disposal of the registry.
                try:
                    signature = registry.get(name).signature
                except KeyError:
                    pass
            requirements = freeze_json(
                getattr(entry, "requirements", {}),
                where="compiled operator %s requirements" % name,
            )
            capabilities = freeze_json(
                getattr(entry, "capabilities", {}),
                where="compiled operator %s capabilities" % name,
            )
            rows[name] = (signature, requirements, capabilities)

        object.__setattr__(self, "state_spaces", tuple(state_spaces))
        object.__setattr__(self, "field_spaces", tuple(field_spaces))
        object.__setattr__(self, "_operators", MappingProxyType(rows))

    @property
    def available(self) -> bool:
        return bool(self._operators or self.state_spaces or self.field_spaces)

    def operator_names(self) -> tuple[str, ...]:
        return tuple(self._operators)

    def signature(self, name: str) -> Any:
        return self._row(name)[0]

    def requirements(self, name: str) -> dict[str, Any]:
        return thaw_json(self._row(name)[1])

    def capabilities(self, name: str) -> dict[str, Any]:
        return thaw_json(self._row(name)[2])

    def _row(self, name: str) -> Any:
        try:
            return self._operators[name]
        except KeyError:
            raise KeyError(
                "operator %r is not in this compiled module (registered: %s)"
                % (name, ", ".join(self._operators) or "<none>")
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CompiledModuleView is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CompiledModuleView is immutable")


def _space_names(model: Any, method_name: str) -> tuple[str, ...]:
    method = getattr(model, method_name, None)
    return tuple(method() or ()) if callable(method) else ()

