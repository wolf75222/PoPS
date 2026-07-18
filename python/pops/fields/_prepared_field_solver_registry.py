"""Authenticated providers for resolved ``FieldSolvePlan`` installation.

The field compiler owns only mathematical and layout facts.  A provider owns its option schema,
compatibility policy, component bindings and native installation.  Builtin solvers and external
components are registered through this exact protocol; consumers never branch on a backend name
or a ``builtin``/``external`` discriminator.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

from pops.fields._identity import field_identity, strict_field_data


_PROVIDER_INTERFACE = "pops.prepared-field-solver-provider@1"
_BINDING_SCHEMA_VERSION = 1


def _freeze(value: Any, *, where: str) -> Any:
    """Freeze the exact field-provider carrier, preserving binary64 values."""
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


def _exact_nonempty(value: Any, *, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be a non-empty exact string" % where)
    return value


@dataclass(frozen=True, slots=True)
class PreparedFieldSolverFacts:
    """Provider-independent facts established by field lowering.

    The compiler deliberately supplies no solver token.  ``operator``, ``layout``, ``hierarchy``
    and ``boundary`` are canonical semantic records and may grow in a versioned provider interface
    without introducing a class hierarchy in the compiler.
    """

    target: str
    operator: Mapping[str, Any]
    layout: Mapping[str, Any]
    hierarchy: Mapping[str, Any]
    boundary: Mapping[str, Any]
    nonlinear: bool

    def __post_init__(self) -> None:
        # ``target`` is a qualified consumer identity, not a closed enum owned by this protocol.
        # Each provider decides which consumers it implements; keeping the carrier open lets a
        # third-party provider add a runtime target without editing the field compiler registry.
        _exact_nonempty(self.target, where="prepared field solver target identity")
        for name in ("operator", "layout", "hierarchy", "boundary"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError("prepared field solver %s facts must be a mapping" % name)
            frozen = _freeze(value, where="prepared field solver %s facts" % name)
            strict_field_data(_plain(frozen))
            object.__setattr__(self, name, frozen)
        if type(self.nonlinear) is not bool:
            raise TypeError("prepared field solver nonlinear fact must be an exact bool")

    def to_data(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "operator": _plain(self.operator),
            "layout": _plain(self.layout),
            "hierarchy": _plain(self.hierarchy),
            "boundary": _plain(self.boundary),
            "nonlinear": self.nonlinear,
        }

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldSolverFacts:
        expected = {"target", "operator", "layout", "hierarchy", "boundary", "nonlinear"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared field solver facts have an invalid shape")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class PreparedFieldSolverResolution:
    """Provider-owned native contract and exact external component authorities."""

    native_contract: Mapping[str, Any]
    topology_contract: Mapping[str, Any]
    component_bindings: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("native_contract", "topology_contract"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError("prepared field solver %s must be a mapping" % name)
            frozen = _freeze(value, where="prepared field solver %s" % name)
            strict_field_data(_plain(frozen))
            object.__setattr__(self, name, frozen)
        if type(self.component_bindings) is not tuple:
            raise TypeError("prepared field solver component bindings must be an exact tuple")
        bindings = tuple(
            _freeze(binding, where="prepared field solver component binding")
            for binding in self.component_bindings
        )
        if any(not isinstance(binding, Mapping) for binding in bindings):
            raise TypeError("prepared field solver component bindings must be mappings")
        identities = tuple(
            field_identity("prepared-field-solver-component-binding", _plain(binding)).token
            for binding in bindings
        )
        if len(identities) != len(set(identities)):
            raise ValueError("prepared field solver component bindings contain a duplicate")
        object.__setattr__(self, "component_bindings", bindings)

    def to_data(self) -> dict[str, Any]:
        return {
            "native_contract": _plain(self.native_contract),
            "topology_contract": _plain(self.topology_contract),
            "component_bindings": [_plain(binding) for binding in self.component_bindings],
        }

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldSolverResolution:
        expected = {"native_contract", "topology_contract", "component_bindings"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared field solver resolution has an invalid shape")
        bindings = value["component_bindings"]
        if not isinstance(bindings, (list, tuple)):
            raise TypeError("prepared field solver component bindings must be a sequence")
        return cls(
            value["native_contract"], value["topology_contract"], tuple(bindings)
        )


@dataclass(frozen=True, slots=True)
class PreparedFieldSolverUse:
    options: Mapping[str, Any]
    facts: PreparedFieldSolverFacts
    resolution: PreparedFieldSolverResolution

    def __post_init__(self) -> None:
        if not isinstance(self.options, Mapping):
            raise TypeError("prepared field solver use requires exact provider options")
        if type(self.facts) is not PreparedFieldSolverFacts:
            raise TypeError("prepared field solver use requires exact facts")
        if type(self.resolution) is not PreparedFieldSolverResolution:
            raise TypeError("prepared field solver use requires an exact resolution")


@dataclass(frozen=True, slots=True)
class PreparedFieldSolverUsePolicy:
    policy_id: str
    version: int
    capabilities: Mapping[str, Any]
    validator: Callable[[PreparedFieldSolverUse, str], None]

    def __post_init__(self) -> None:
        _exact_nonempty(self.policy_id, where="prepared field solver use-policy id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("prepared field solver use-policy version must be positive")
        if not isinstance(self.capabilities, Mapping):
            raise TypeError("prepared field solver capabilities must be a mapping")
        capabilities = _freeze(
            self.capabilities, where="prepared field solver use-policy capabilities"
        )
        strict_field_data(_plain(capabilities))
        object.__setattr__(self, "capabilities", capabilities)
        if not callable(self.validator):
            raise TypeError("prepared field solver use-policy validator must be callable")

    def authority(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "capabilities": _plain(self.capabilities),
        }

    def validate(self, use: PreparedFieldSolverUse, *, where: str) -> None:
        result = self.validator(use, where)
        if result is not None:
            raise TypeError("prepared field solver use-policy validator must return None")


class PreparedFieldSolverResolver(Protocol):
    def __call__(
        self,
        options: Mapping[str, Any],
        facts: PreparedFieldSolverFacts,
        where: str,
    ) -> PreparedFieldSolverResolution:
        ...


class PreparedFieldSolverNativeInstaller(Protocol):
    def __call__(
        self,
        context: Any,
        binding: PreparedFieldSolverBinding,
    ) -> None:
        """Install a provider route and the common plan; numerical work remains native."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedFieldSolverProvider:
    """Complete immutable authority for one field-solver implementation family."""

    provider_id: str
    version: int
    resolver_id: str
    installer_id: str
    use_policy: PreparedFieldSolverUsePolicy
    resolver: PreparedFieldSolverResolver
    native_installer: PreparedFieldSolverNativeInstaller

    def __post_init__(self) -> None:
        for name in ("provider_id", "resolver_id", "installer_id"):
            _exact_nonempty(
                getattr(self, name), where="prepared field solver %s" % name
            )
        if type(self.version) is not int or self.version < 1:
            raise ValueError("prepared field solver provider version must be positive")
        if type(self.use_policy) is not PreparedFieldSolverUsePolicy:
            raise TypeError("prepared field solver provider requires an exact use policy")
        if not callable(self.resolver) or not callable(self.native_installer):
            raise TypeError("prepared field solver provider requires resolver and installer")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "interface": _PROVIDER_INTERFACE,
            "provider_id": self.provider_id,
            "version": self.version,
            "resolver_id": self.resolver_id,
            "installer_id": self.installer_id,
            "use_policy": self.use_policy.authority(),
        }

    def prepare(
        self,
        *,
        options: Mapping[str, Any],
        facts: PreparedFieldSolverFacts,
        where: str,
    ) -> PreparedFieldSolverBinding:
        if not isinstance(options, Mapping):
            raise TypeError("%s provider options must be a mapping" % where)
        frozen_options = _freeze(options, where="%s provider options" % where)
        resolution = self.resolver(frozen_options, facts, where)
        if type(resolution) is not PreparedFieldSolverResolution:
            raise TypeError("prepared field solver resolver must return an exact resolution")
        self.use_policy.validate(
            PreparedFieldSolverUse(frozen_options, facts, resolution), where=where
        )
        return PreparedFieldSolverBinding.create(
            provider=self,
            options=frozen_options,
            facts=facts,
            resolution=resolution,
        )

    def validate_binding(self, binding: PreparedFieldSolverBinding, *, where: str) -> None:
        if type(binding) is not PreparedFieldSolverBinding:
            raise TypeError("%s requires an exact prepared field solver binding" % where)
        if _plain(binding.provider) != self.authority():
            raise ValueError("%s field solver provider authority changed" % where)
        replay = self.resolver(binding.options, binding.facts, where)
        if replay != binding.resolution:
            raise ValueError("%s field solver provider resolution is nondeterministic" % where)
        self.use_policy.validate(
            PreparedFieldSolverUse(binding.options, binding.facts, binding.resolution),
            where=where,
        )
        if binding.identity != binding.expected_identity():
            raise ValueError("%s field solver binding identity is not canonical" % where)

    def install(self, context: Any, binding: PreparedFieldSolverBinding) -> None:
        self.validate_binding(binding, where="native field solver install")
        result = self.native_installer(context, binding)
        if result is not None:
            raise TypeError("prepared field solver native installer must return None")


@dataclass(frozen=True, slots=True)
class PreparedFieldSolverBinding:
    """Immutable provider selection crossing resolve, compile, bind and MPI identity."""

    provider: Mapping[str, Any]
    options: Mapping[str, Any]
    facts: PreparedFieldSolverFacts
    resolution: PreparedFieldSolverResolution
    identity: str

    def __post_init__(self) -> None:
        provider = _freeze(self.provider, where="prepared field solver provider authority")
        options = _freeze(self.options, where="prepared field solver authored options")
        if not isinstance(provider, Mapping) or not isinstance(options, Mapping):
            raise TypeError("prepared field solver provider/options must be mappings")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "options", options)
        if type(self.facts) is not PreparedFieldSolverFacts:
            raise TypeError("prepared field solver binding requires exact facts")
        if type(self.resolution) is not PreparedFieldSolverResolution:
            raise TypeError("prepared field solver binding requires an exact resolution")
        _exact_nonempty(self.identity, where="prepared field solver binding identity")

    @classmethod
    def create(
        cls,
        *,
        provider: PreparedFieldSolverProvider,
        options: Mapping[str, Any],
        facts: PreparedFieldSolverFacts,
        resolution: PreparedFieldSolverResolution,
    ) -> PreparedFieldSolverBinding:
        provisional = cls(provider.authority(), options, facts, resolution, "pending")
        return cls(
            provider.authority(), options, facts, resolution,
            provisional.expected_identity(),
        )

    def identity_data(self) -> dict[str, Any]:
        return {
            "schema_version": _BINDING_SCHEMA_VERSION,
            "provider": _plain(self.provider),
            "options": _plain(self.options),
            "facts": self.facts.to_data(),
            "resolution": self.resolution.to_data(),
        }

    def expected_identity(self) -> str:
        return field_identity("prepared-field-solver-binding", self.identity_data()).token

    def to_data(self) -> dict[str, Any]:
        return {**self.identity_data(), "identity": self.identity}

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldSolverBinding:
        expected = {
            "schema_version", "provider", "options", "facts", "resolution", "identity"
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != _BINDING_SCHEMA_VERSION
        ):
            raise ValueError("prepared field solver binding has an invalid shape")
        binding = cls(
            value["provider"],
            value["options"],
            PreparedFieldSolverFacts.from_data(value["facts"]),
            PreparedFieldSolverResolution.from_data(value["resolution"]),
            value["identity"],
        )
        provider = prepared_field_solver_provider_from_identity(binding.provider)
        provider.validate_binding(binding, where="resolved field solver binding")
        return binding


_registry_lock = RLock()
_providers_by_resolver: dict[str, PreparedFieldSolverProvider] = {}


def register_prepared_field_solver_provider(
    provider: PreparedFieldSolverProvider,
) -> PreparedFieldSolverProvider:
    if type(provider) is not PreparedFieldSolverProvider:
        raise TypeError("field solver plugins must register an exact Provider")
    with _registry_lock:
        if provider.resolver_id in _providers_by_resolver:
            raise ValueError(
                "prepared field solver resolver %r is already registered" % provider.resolver_id
            )
        if any(
            existing.provider_id == provider.provider_id
            for existing in _providers_by_resolver.values()
        ):
            raise ValueError(
                "prepared field solver provider %r is already registered" % provider.provider_id
            )
        if any(
            existing.installer_id == provider.installer_id
            for existing in _providers_by_resolver.values()
        ):
            raise ValueError(
                "prepared field solver installer %r is already registered" % provider.installer_id
            )
        _providers_by_resolver[provider.resolver_id] = provider
    return provider


def prepared_field_solver_provider_by_resolver_id(
    resolver_id: Any,
) -> PreparedFieldSolverProvider:
    if type(resolver_id) is not str:
        raise TypeError("prepared field solver resolver id must be an exact string")
    with _registry_lock:
        provider = _providers_by_resolver.get(resolver_id)
    if provider is None:
        raise NotImplementedError(
            "prepared field solver resolver %r is not registered" % resolver_id
        )
    return provider


def prepared_field_solver_provider_from_identity(
    identity: Any,
) -> PreparedFieldSolverProvider:
    expected = {
        "schema_version", "interface", "provider_id", "version", "resolver_id",
        "installer_id", "use_policy",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected:
        raise ValueError("prepared field solver provider authority is not exact")
    provider = prepared_field_solver_provider_by_resolver_id(identity.get("resolver_id"))
    if _plain(identity) != provider.authority():
        raise ValueError("prepared field solver provider authority is inconsistent")
    return provider


def prepared_field_solver_binding_from_descriptor(
    descriptor: Any,
    *,
    facts: PreparedFieldSolverFacts,
    where: str,
) -> PreparedFieldSolverBinding:
    protocol = getattr(descriptor, "_prepared_field_solver", None)
    if not callable(protocol):
        raise TypeError("%s must implement the prepared field solver provider protocol" % where)
    value = protocol()
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not PreparedFieldSolverProvider
        or not isinstance(value[1], Mapping)
    ):
        raise TypeError("%s returned an invalid prepared field solver provider binding" % where)
    provider = prepared_field_solver_provider_by_resolver_id(value[0].resolver_id)
    if provider is not value[0]:
        raise ValueError("%s field solver provider is not the registered authority" % where)
    return provider.prepare(options=value[1], facts=facts, where=where)


def prepared_field_solver_binding_from_data(value: Any) -> PreparedFieldSolverBinding:
    return PreparedFieldSolverBinding.from_data(value)


__all__ = [
    "PreparedFieldSolverBinding",
    "PreparedFieldSolverFacts",
    "PreparedFieldSolverNativeInstaller",
    "PreparedFieldSolverProvider",
    "PreparedFieldSolverResolution",
    "PreparedFieldSolverUse",
    "PreparedFieldSolverUsePolicy",
    "prepared_field_solver_binding_from_data",
    "prepared_field_solver_binding_from_descriptor",
    "prepared_field_solver_provider_by_resolver_id",
    "prepared_field_solver_provider_from_identity",
    "register_prepared_field_solver_provider",
]
