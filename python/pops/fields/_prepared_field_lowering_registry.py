"""Authenticated, append-only providers for field-method lowering.

The registry is intentionally ignorant of equations, layouts, boundary families and runtime
targets.  A numerical method binds to one provider; that provider owns the complete validation and
derivation of the native field contract.  The compiler merely authenticates the result and combines
it with independently prepared solver/nullspace providers.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

from pops.fields._identity import field_identity, strict_field_data
from pops.fields._prepared_field_nullspace_registry import PreparedFieldNullspaceFacts
from pops.fields._prepared_field_solver_registry import PreparedFieldSolverFacts


_INTERFACE = "pops.prepared-field-lowering-provider@1"
_RESERVED_NATIVE_KEYS = frozenset({
    "method_provider", "solver_provider", "nullspace_provider", "nonlinear",
})


def _exact_nonempty(value: Any, *, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be a non-empty exact string" % where)
    return value


def _freeze(value: Any, *, where: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite binary64 value" % where)
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty exact strings" % where)
        return MappingProxyType({
            key: _freeze(item, where="%s.%s" % (where, key))
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        )
    raise TypeError("%s contains opaque %s" % (where, type(value).__name__))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PreparedFieldLoweringRequest:
    """Opaque authoring objects supplied to one explicitly selected method provider."""

    name: str
    operator: Any
    discretization: Any
    target: str
    output_components: tuple[str, ...]
    layout: Any

    def __post_init__(self) -> None:
        _exact_nonempty(self.name, where="field lowering name")
        _exact_nonempty(self.target, where="field lowering target identity")
        if type(self.output_components) is not tuple or not self.output_components or any(
            type(component) is not str or not component for component in self.output_components
        ):
            raise TypeError(
                "field lowering output components must be a non-empty exact string tuple"
            )


@dataclass(frozen=True, slots=True)
class PreparedFieldLoweringEvidence:
    """One provider-owned coverage fact, converted by the compiler without interpretation."""

    source: str
    disposition: str
    targets: tuple[str, ...] = ()
    rule: str | None = None
    gate: str | None = None

    def __post_init__(self) -> None:
        _exact_nonempty(self.source, where="field lowering evidence source")
        _exact_nonempty(self.disposition, where="field lowering evidence disposition")
        if type(self.targets) is not tuple or any(
            type(target) is not str or not target for target in self.targets
        ):
            raise TypeError("field lowering evidence targets must be exact strings")
        for name in ("rule", "gate"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise TypeError("field lowering evidence %s must be None or non-empty" % name)

    def to_data(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "disposition": self.disposition,
            "targets": list(self.targets),
            "rule": self.rule,
            "gate": self.gate,
        }

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldLoweringEvidence:
        expected = {"source", "disposition", "targets", "rule", "gate"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared field lowering evidence has an invalid shape")
        targets = value["targets"]
        if not isinstance(targets, (list, tuple)):
            raise TypeError("prepared field lowering evidence targets must be a sequence")
        return cls(
            value["source"], value["disposition"], tuple(targets),
            value["rule"], value["gate"],
        )


@dataclass(frozen=True, slots=True)
class PreparedFieldLoweringResolution:
    """Complete provider-owned native contract and downstream provider facts."""

    native_options: Mapping[str, Any]
    solver_facts: PreparedFieldSolverFacts
    nullspace_facts: PreparedFieldNullspaceFacts
    evidence: tuple[PreparedFieldLoweringEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.native_options, Mapping):
            raise TypeError("prepared field lowering native options must be a mapping")
        if _RESERVED_NATIVE_KEYS.intersection(self.native_options):
            raise ValueError(
                "prepared field lowering native options use compiler-reserved keys"
            )
        native = _freeze(self.native_options, where="prepared field lowering native options")
        strict_field_data(_plain(native))
        object.__setattr__(self, "native_options", native)
        if type(self.solver_facts) is not PreparedFieldSolverFacts:
            raise TypeError("prepared field lowering requires exact solver facts")
        if type(self.nullspace_facts) is not PreparedFieldNullspaceFacts:
            raise TypeError("prepared field lowering requires exact nullspace facts")
        if type(self.evidence) is not tuple or any(
            type(item) is not PreparedFieldLoweringEvidence for item in self.evidence
        ):
            raise TypeError("prepared field lowering evidence must be an exact tuple")

    def to_data(self) -> dict[str, Any]:
        return {
            "native_options": _plain(self.native_options),
            "solver_facts": self.solver_facts.to_data(),
            "nullspace_facts": self.nullspace_facts.to_data(),
            "evidence": [item.to_data() for item in self.evidence],
        }

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldLoweringResolution:
        expected = {"native_options", "solver_facts", "nullspace_facts", "evidence"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared field lowering resolution has an invalid shape")
        evidence = value["evidence"]
        if not isinstance(evidence, (list, tuple)):
            raise TypeError("prepared field lowering evidence must be a sequence")
        return cls(
            value["native_options"],
            PreparedFieldSolverFacts.from_data(value["solver_facts"]),
            PreparedFieldNullspaceFacts.from_data(value["nullspace_facts"]),
            tuple(PreparedFieldLoweringEvidence.from_data(item) for item in evidence),
        )


@dataclass(frozen=True, slots=True)
class PreparedFieldRuntimePreflightContext:
    """Engine-free resources offered while a provider prepares canonical install payloads."""

    target: str
    resources: Mapping[str, Any]
    slot: str

    def __post_init__(self) -> None:
        _exact_nonempty(self.target, where="field runtime preflight target identity")
        _exact_nonempty(self.slot, where="field runtime preflight provider slot")
        if not isinstance(self.resources, Mapping):
            raise TypeError("field runtime preflight resources must be a mapping")
        if any(type(name) is not str or not name for name in self.resources):
            raise TypeError(
                "field runtime preflight resource names must be non-empty exact strings"
            )
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))


@dataclass(frozen=True, slots=True)
class PreparedFieldRuntimeInstallContext:
    """Opaque target engine plus the resources authenticated by the pure preflight phase."""

    target: str
    engine: Any
    resources: Mapping[str, Any]
    slot: str

    def __post_init__(self) -> None:
        preflight = PreparedFieldRuntimePreflightContext(
            self.target, self.resources, self.slot
        )
        object.__setattr__(self, "resources", preflight.resources)

    def preflight(self) -> PreparedFieldRuntimePreflightContext:
        return PreparedFieldRuntimePreflightContext(
            self.target, self.resources, self.slot
        )


class PreparedFieldLoweringResolver(Protocol):
    def __call__(
        self,
        options: Mapping[str, Any],
        request: PreparedFieldLoweringRequest,
        where: str,
    ) -> PreparedFieldLoweringResolution: ...


class PreparedFieldLoweringResolutionValidator(Protocol):
    def __call__(self, binding: PreparedFieldLoweringBinding, where: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedFieldLoweringProvider:
    """Immutable authority for one spatial-method lowering implementation."""

    provider_id: str
    version: int
    resolver_id: str
    resolution_validator_id: str
    runtime_binder_id: str
    output_preparer_id: str
    bound_options_installer_id: str
    output_installer_id: str
    capabilities: Mapping[str, Any]
    resolver: PreparedFieldLoweringResolver
    resolution_validator: PreparedFieldLoweringResolutionValidator
    parameter_handles: Callable[
        [PreparedFieldLoweringBinding, Any, Any], Mapping[str, tuple[Any, ...]]
    ]
    bind_native_options: Callable[
        [PreparedFieldLoweringBinding, Any, Any, Mapping[Any, Any]], Mapping[str, Any]
    ]
    prepare_output: Callable[
        [PreparedFieldLoweringBinding, PreparedFieldRuntimePreflightContext, Any, Any],
        Mapping[str, Any]
    ]
    install_bound_options: Callable[
        [PreparedFieldLoweringBinding, PreparedFieldRuntimeInstallContext,
         Mapping[str, Any]], None
    ]
    install_output: Callable[
        [PreparedFieldLoweringBinding, PreparedFieldRuntimeInstallContext,
         Mapping[str, Any]], None
    ]

    def __post_init__(self) -> None:
        for name in (
            "provider_id", "resolver_id", "resolution_validator_id", "runtime_binder_id",
            "output_preparer_id", "bound_options_installer_id", "output_installer_id",
        ):
            _exact_nonempty(
                getattr(self, name), where="prepared field lowering %s" % name
            )
        if type(self.version) is not int or self.version < 1:
            raise ValueError("prepared field lowering provider version must be positive")
        if not isinstance(self.capabilities, Mapping):
            raise TypeError("prepared field lowering capabilities must be a mapping")
        capabilities = _freeze(
            self.capabilities, where="prepared field lowering capabilities"
        )
        strict_field_data(_plain(capabilities))
        object.__setattr__(self, "capabilities", capabilities)
        for name in (
            "resolver", "resolution_validator", "parameter_handles",
            "bind_native_options", "prepare_output", "install_bound_options", "install_output",
        ):
            if not callable(getattr(self, name)):
                raise TypeError("prepared field lowering %s must be callable" % name)

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "interface": _INTERFACE,
            "provider_id": self.provider_id,
            "version": self.version,
            "resolver_id": self.resolver_id,
            "resolution_validator_id": self.resolution_validator_id,
            "runtime_binder_id": self.runtime_binder_id,
            "output_preparer_id": self.output_preparer_id,
            "bound_options_installer_id": self.bound_options_installer_id,
            "output_installer_id": self.output_installer_id,
            "capabilities": _plain(self.capabilities),
        }

    def prepare(
        self,
        *,
        options: Mapping[str, Any],
        request: PreparedFieldLoweringRequest,
        where: str,
    ) -> PreparedFieldLoweringBinding:
        if not isinstance(options, Mapping):
            raise TypeError("%s lowering options must be a mapping" % where)
        frozen = _freeze(options, where="%s lowering options" % where)
        resolution = self.resolver(frozen, request, where)
        if type(resolution) is not PreparedFieldLoweringResolution:
            raise TypeError("prepared field lowering resolver returned a foreign resolution")
        binding = PreparedFieldLoweringBinding.create(
            provider=self, options=frozen, resolution=resolution
        )
        self.validate_binding(binding, where=where)
        return binding

    def validate_binding(
        self, binding: PreparedFieldLoweringBinding, *, where: str,
    ) -> None:
        if type(binding) is not PreparedFieldLoweringBinding:
            raise TypeError("%s requires an exact field lowering binding" % where)
        if _plain(binding.provider) != self.authority():
            raise ValueError("%s field lowering provider authority changed" % where)
        if binding.identity != binding.expected_identity():
            raise ValueError("%s field lowering binding identity is not canonical" % where)
        result = self.resolution_validator(binding, where)
        if result is not None:
            raise TypeError("field lowering resolution validator must return None")

    def install_runtime(
        self,
        binding: PreparedFieldLoweringBinding,
        context: PreparedFieldRuntimeInstallContext,
        operator: Any,
        discretization: Any,
        params: Mapping[Any, Any],
    ) -> None:
        """Run one provider's complete install protocol without interpreting its payload."""
        self.validate_binding(binding, where="prepared field runtime install")
        if type(context) is not PreparedFieldRuntimeInstallContext:
            raise TypeError("prepared field runtime install requires an exact context")
        if context.target != binding.resolution.solver_facts.target:
            raise ValueError("prepared field runtime target disagrees with resolved solver facts")
        if not isinstance(params, Mapping):
            raise TypeError("prepared field runtime bind parameters must be a mapping")
        bound_options = self.prepare_bound_options(
            binding, operator, discretization, params
        )
        output_payload = self.prepare_output_payload(
            binding, context.preflight(), operator, discretization
        )
        output_result = self.install_output(
            binding, context, output_payload
        )
        if output_result is not None:
            raise TypeError("field lowering output installer must return None")
        options_result = self.install_bound_options(
            binding, context, bound_options
        )
        if options_result is not None:
            raise TypeError("field lowering native-options installer must return None")

    def prepare_bound_options(
        self,
        binding: PreparedFieldLoweringBinding,
        operator: Any,
        discretization: Any,
        params: Mapping[Any, Any],
    ) -> Mapping[str, Any]:
        """Compute and canonicalize provider options before the first runtime mutation."""
        if not isinstance(params, Mapping):
            raise TypeError("prepared field runtime bind parameters must be a mapping")
        result = self.bind_native_options(binding, operator, discretization, params)
        if not isinstance(result, Mapping):
            raise TypeError("field lowering runtime binder must return a mapping")
        payload = _freeze(result, where="prepared field bound-options payload")
        strict_field_data(_plain(payload))
        return payload

    def prepare_output_payload(
        self,
        binding: PreparedFieldLoweringBinding,
        context: PreparedFieldRuntimePreflightContext,
        operator: Any,
        discretization: Any,
    ) -> Mapping[str, Any]:
        """Compute and canonicalize output routing without exposing a mutable native engine."""
        if type(context) is not PreparedFieldRuntimePreflightContext:
            raise TypeError("field output preflight requires an exact engine-free context")
        if context.target != binding.resolution.solver_facts.target:
            raise ValueError("field output preflight target disagrees with resolved solver facts")
        result = self.prepare_output(binding, context, operator, discretization)
        if not isinstance(result, Mapping):
            raise TypeError("field lowering output preparer must return a mapping")
        payload = _freeze(result, where="prepared field output payload")
        strict_field_data(_plain(payload))
        return payload


@dataclass(frozen=True, slots=True)
class PreparedFieldLoweringBinding:
    provider: Mapping[str, Any]
    options: Mapping[str, Any]
    resolution: PreparedFieldLoweringResolution
    identity: str

    def __post_init__(self) -> None:
        provider = _freeze(self.provider, where="field lowering provider authority")
        options = _freeze(self.options, where="field lowering authored options")
        if not isinstance(provider, Mapping) or not isinstance(options, Mapping):
            raise TypeError("field lowering provider/options must be mappings")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "options", options)
        if type(self.resolution) is not PreparedFieldLoweringResolution:
            raise TypeError("field lowering binding requires an exact resolution")
        _exact_nonempty(self.identity, where="field lowering binding identity")

    @classmethod
    def create(
        cls,
        *,
        provider: PreparedFieldLoweringProvider,
        options: Mapping[str, Any],
        resolution: PreparedFieldLoweringResolution,
    ) -> PreparedFieldLoweringBinding:
        pending = cls(provider.authority(), options, resolution, "pending")
        return cls(
            provider.authority(), options, resolution, pending.expected_identity()
        )

    def identity_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": _plain(self.provider),
            "options": _plain(self.options),
            "resolution": self.resolution.to_data(),
        }

    def expected_identity(self) -> str:
        return field_identity(
            "prepared-field-lowering-binding", self.identity_data()
        ).token

    def to_data(self) -> dict[str, Any]:
        return {**self.identity_data(), "identity": self.identity}

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldLoweringBinding:
        expected = {"schema_version", "provider", "options", "resolution", "identity"}
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("prepared field lowering binding has an invalid shape")
        binding = cls(
            value["provider"], value["options"],
            PreparedFieldLoweringResolution.from_data(value["resolution"]),
            value["identity"],
        )
        provider = prepared_field_lowering_provider_from_identity(binding.provider)
        provider.validate_binding(binding, where="resolved field lowering binding")
        return binding


_lock = RLock()
_providers: dict[str, PreparedFieldLoweringProvider] = {}


def register_prepared_field_lowering_provider(
    provider: PreparedFieldLoweringProvider,
) -> PreparedFieldLoweringProvider:
    if type(provider) is not PreparedFieldLoweringProvider:
        raise TypeError("field lowering plugins must register an exact Provider")
    with _lock:
        if provider.resolver_id in _providers or any(
            current.provider_id == provider.provider_id
            or current.resolution_validator_id == provider.resolution_validator_id
            or current.runtime_binder_id == provider.runtime_binder_id
            or current.output_preparer_id == provider.output_preparer_id
            or current.bound_options_installer_id == provider.bound_options_installer_id
            or current.output_installer_id == provider.output_installer_id
            for current in _providers.values()
        ):
            raise ValueError("prepared field lowering provider identity is already registered")
        _providers[provider.resolver_id] = provider
    return provider


def prepared_field_lowering_provider_by_resolver_id(
    resolver_id: Any,
) -> PreparedFieldLoweringProvider:
    if type(resolver_id) is not str:
        raise TypeError("field lowering resolver id must be an exact string")
    with _lock:
        provider = _providers.get(resolver_id)
    if provider is None:
        raise NotImplementedError(
            "prepared field lowering resolver %r is not registered" % resolver_id
        )
    return provider


def prepared_field_lowering_provider_from_identity(
    identity: Any,
) -> PreparedFieldLoweringProvider:
    expected = {
        "schema_version", "interface", "provider_id", "version", "resolver_id",
        "resolution_validator_id", "runtime_binder_id", "capabilities",
        "output_preparer_id", "bound_options_installer_id", "output_installer_id",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected:
        raise ValueError("prepared field lowering provider authority is not exact")
    provider = prepared_field_lowering_provider_by_resolver_id(identity.get("resolver_id"))
    if _plain(identity) != provider.authority():
        raise ValueError("prepared field lowering provider authority is inconsistent")
    return provider


def prepared_field_lowering_binding_from_descriptor(
    descriptor: Any,
    *,
    request: PreparedFieldLoweringRequest,
    where: str,
) -> PreparedFieldLoweringBinding:
    protocol = getattr(descriptor, "_prepared_field_lowering", None)
    if not callable(protocol):
        raise TypeError("%s has no prepared field lowering provider" % where)
    value = protocol()
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not PreparedFieldLoweringProvider
        or not isinstance(value[1], Mapping)
    ):
        raise TypeError("%s returned an invalid prepared field lowering binding" % where)
    provider = prepared_field_lowering_provider_by_resolver_id(value[0].resolver_id)
    if provider is not value[0]:
        raise ValueError("%s field lowering provider is not the registered authority" % where)
    return provider.prepare(options=value[1], request=request, where=where)


def prepared_field_lowering_binding_from_data(
    value: Any,
) -> PreparedFieldLoweringBinding:
    return PreparedFieldLoweringBinding.from_data(value)


__all__ = [
    "PreparedFieldLoweringBinding",
    "PreparedFieldLoweringEvidence",
    "PreparedFieldLoweringProvider",
    "PreparedFieldLoweringRequest",
    "PreparedFieldLoweringResolution",
    "PreparedFieldRuntimeInstallContext",
    "PreparedFieldRuntimePreflightContext",
    "prepared_field_lowering_binding_from_data",
    "prepared_field_lowering_binding_from_descriptor",
    "prepared_field_lowering_provider_by_resolver_id",
    "prepared_field_lowering_provider_from_identity",
    "register_prepared_field_lowering_provider",
]
