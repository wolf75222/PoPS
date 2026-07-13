"""Manifest-driven adapters for the small PoPS component interfaces.

The adapter is the only behavioral trust boundary.  It binds the exact interface rows carried by
``ComponentManifest`` and never branches on a scientific concrete class, a route family, or an
``isinstance`` allowlist.  Builtins, source extensions, and AOT entry points therefore produce the
same immutable registration/provenance record.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import Any, NoReturn

from ._component_manifest import ComponentManifest
from ._generated_component_schema import COMPONENT_INTERFACE_SPECS


class ComponentInterfaceError(TypeError):
    """Structured refusal while adapting or invoking a component interface."""

    def __init__(self, code: str, component_id: str, interface: str, message: str, *,
                 evidence: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.component_id = component_id
        self.interface = interface
        self.evidence = evidence

    def to_data(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "component_id": self.component_id,
            "interface": self.interface,
            "message": str(self),
            "evidence": self.evidence,
        }


def _refuse(code: str, manifest: ComponentManifest, interface: str, message: str, *,
            evidence: Any = None) -> NoReturn:
    raise ComponentInterfaceError(code, manifest.component_id, interface, message,
                                  evidence=evidence)


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    name: str
    method: str
    required_args: int


INTERFACE_SPECS = MappingProxyType({
    row["name"]: InterfaceSpec(row["name"], row["method"], row["required_args"])
    for row in COMPONENT_INTERFACE_SPECS
})


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Explicit outcome of a fallible scientific evaluation.

    ``retry`` and ``reject`` are recoverable transaction actions; ``failed`` is terminal for the
    current run.  Python truth conversion is refused so a caller cannot silently turn any failure
    into success with ``if outcome``.
    """

    status: str
    value: Any = field(default=None, compare=False)
    reason: str = ""
    evidence: Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "retry", "reject", "failed"}:
            raise ValueError("EvaluationOutcome status must be ok, retry, reject, or failed")
        if self.status == "ok" and self.reason:
            raise ValueError("successful EvaluationOutcome cannot carry a failure reason")
        if self.status != "ok" and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("non-success EvaluationOutcome requires a non-empty reason")
        if self.status != "ok" and self.value is not None:
            raise ValueError("non-success EvaluationOutcome cannot carry a value")

    def __bool__(self) -> bool:
        raise TypeError("EvaluationOutcome has no Python truth value; inspect .status explicitly")

    @property
    def transaction_action(self) -> str:
        """Exact Program/step action; no caller may reinterpret a failure as a value."""
        return {
            "ok": "continue",
            "retry": "retry_step",
            "reject": "reject_step",
            "failed": "abort_run",
        }[self.status]

    @classmethod
    def ok(cls, value: Any = None) -> EvaluationOutcome:
        return cls("ok", value=value)

    @classmethod
    def retry(cls, reason: str, *, evidence: Any = None) -> EvaluationOutcome:
        return cls("retry", reason=reason, evidence=evidence)

    @classmethod
    def reject(cls, reason: str, *, evidence: Any = None) -> EvaluationOutcome:
        return cls("reject", reason=reason, evidence=evidence)

    @classmethod
    def failed(cls, reason: str, *, evidence: Any = None) -> EvaluationOutcome:
        return cls("failed", reason=reason, evidence=evidence)


@dataclass(frozen=True, slots=True)
class ComponentProvenance:
    """Origin evidence with the same shape for builtin and external components."""

    origin: str
    source_uri: str
    semantic_identity: str
    manifest_identity: str

    @classmethod
    def from_manifest(cls, manifest: ComponentManifest, *, origin: str,
                      source_uri: str | None = None) -> ComponentProvenance:
        if not isinstance(origin, str) or not origin:
            raise ValueError("component provenance origin must be non-empty")
        uri = manifest.uri if source_uri is None else source_uri
        if not isinstance(uri, str) or not uri:
            raise ValueError("component provenance source_uri must be non-empty")
        return cls(origin, uri, manifest.semantic_digest.token, manifest.manifest_digest.token)

    def to_data(self) -> dict[str, str]:
        return {
            "origin": self.origin,
            "source_uri": self.source_uri,
            "semantic_identity": self.semantic_identity,
            "manifest_identity": self.manifest_identity,
        }


@dataclass(frozen=True, slots=True)
class InterfaceBinding:
    name: str
    mode: str
    binding: str
    native_symbol: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "mode": self.mode, "binding": self.binding,
                "native_symbol": self.native_symbol}


class ComponentAdapter:
    """Immutable exact interface table for one registered component."""

    __slots__ = ("manifest", "component", "provenance", "_bindings", "_entry_points",
                 "_sealed")

    def __init__(self, manifest: ComponentManifest, component: Any,
                 provenance: ComponentProvenance,
                 bindings: Mapping[str, InterfaceBinding],
                 entry_points: Mapping[str, Any]) -> None:
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "_bindings", MappingProxyType(dict(bindings)))
        object.__setattr__(self, "_entry_points", MappingProxyType(dict(entry_points)))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ComponentAdapter is immutable")
        object.__setattr__(self, name, value)

    @property
    def component_id(self) -> str:
        return self.manifest.component_id

    @property
    def interfaces(self) -> tuple[str, ...]:
        return tuple(self._bindings)

    def binding(self, interface: str) -> InterfaceBinding:
        try:
            return self._bindings[interface]
        except KeyError:
            _refuse("missing_component_interface", self.manifest, interface,
                    f"component {self.component_id!r} does not provide interface {interface!r}",
                    evidence={"provided": list(self._bindings)})

    def invoke(self, interface: str, *args: Any, **kwargs: Any) -> Any:
        binding = self.binding(interface)
        if binding.mode == "value":
            if args or kwargs:
                _refuse("value_interface_arguments", self.manifest, interface,
                        f"value interface {interface!r} does not accept arguments")
            result = getattr(self.component, binding.binding)
        elif binding.mode == "method":
            result = getattr(self.component, binding.binding)(*args, **kwargs)
        else:
            target = self._entry_points.get(binding.binding)
            if target is None:
                _refuse(
                    "unbound_component_entry_point", self.manifest, interface,
                    f"component interface {interface!r} requires unbound entry point "
                    f"{binding.binding!r} ({binding.native_symbol!r})",
                    evidence={"entry_point": binding.binding,
                              "native_symbol": binding.native_symbol},
                )
            result = target(*args, **kwargs)
        if interface == "fallible_evaluation" and not isinstance(result, EvaluationOutcome):
            _refuse(
                "implicit_evaluation_outcome", self.manifest, interface,
                "fallible_evaluation must return an explicit EvaluationOutcome",
                evidence={"returned_type": type(result).__name__},
            )
        return result

    def to_data(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.manifest.component_type,
            "interfaces": [self._bindings[name].to_data() for name in self._bindings],
            "requirements": list(self.manifest.requirements),
            "capabilities": list(self.manifest.capabilities),
            "effects": list(self.manifest.effects),
            "target": {"variants": [dict(row) for row in self.manifest.target["variants"]]},
            "provenance": self.provenance.to_data(),
        }


def _validate_callable(manifest: ComponentManifest, interface: str, target: Any,
                       required_args: int) -> None:
    if not callable(target):
        _refuse("interface_not_callable", manifest, interface,
                f"interface {interface!r} must bind a callable member")
    try:
        inspect.signature(target).bind(*([object()] * required_args))
    except (TypeError, ValueError):
        _refuse(
            "interface_signature_mismatch", manifest, interface,
            f"interface {interface!r} cannot accept its {required_args} required argument(s)",
            evidence={"binding": getattr(target, "__name__", repr(target)),
                      "required_args": required_args},
        )


def adapt_component(component: Any, manifest: ComponentManifest, *, origin: str,
                    source_uri: str | None = None,
                    entry_points: Mapping[str, Any] | None = None,
                    platform: Mapping[str, Any] | None = None) -> ComponentAdapter:
    """Validate and adapt one component without executing scientific code."""
    if not isinstance(manifest, ComponentManifest):
        raise TypeError("adapt_component requires a ComponentManifest")
    if platform is not None:
        manifest.require_target(platform)
    resolvers = dict(entry_points or {})
    unknown_resolvers = sorted(set(resolvers) - set(manifest.entry_points))
    if unknown_resolvers:
        _refuse("unknown_entry_point_resolver", manifest, "entry_points",
                "entry point resolvers contain undeclared names",
                evidence={"unknown": unknown_resolvers})
    bindings: dict[str, InterfaceBinding] = {}
    for declaration in manifest.interfaces:
        name = declaration["name"]
        spec = INTERFACE_SPECS[name]
        mode = declaration["mode"]
        member = declaration["binding"]
        native_symbol = manifest.entry_points.get(member) if mode == "entry_point" else None
        if mode == "method":
            if not hasattr(component, member):
                _refuse("missing_interface_binding", manifest, name,
                        f"component has no member {member!r} for interface {name!r}")
            _validate_callable(manifest, name, getattr(component, member), spec.required_args)
        elif mode == "value":
            if not hasattr(component, member):
                _refuse("missing_interface_binding", manifest, name,
                        f"component has no value {member!r} for interface {name!r}")
            if callable(getattr(component, member)):
                _refuse("value_interface_callable", manifest, name,
                        f"value interface {name!r} binds callable member {member!r}")
            if spec.required_args != 0:
                _refuse("value_interface_arity", manifest, name,
                        f"interface {name!r} requires arguments and cannot use value mode")
        elif member in resolvers:
            _validate_callable(manifest, name, resolvers[member], spec.required_args)
        bindings[name] = InterfaceBinding(name, mode, member, native_symbol)
    provenance = ComponentProvenance.from_manifest(
        manifest, origin=origin, source_uri=source_uri)
    return ComponentAdapter(manifest, component, provenance, bindings, resolvers)


__all__ = [
    "ComponentInterfaceError", "InterfaceSpec", "INTERFACE_SPECS", "EvaluationOutcome",
    "ComponentProvenance", "InterfaceBinding", "ComponentAdapter", "adapt_component",
]
