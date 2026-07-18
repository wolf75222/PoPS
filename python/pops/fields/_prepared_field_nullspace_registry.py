"""Provider-owned nullspace policy for resolved field installation.

This protocol is deliberately separate from matrix-free Program nullspaces: a field provider owns
the topology-derived assertion, gauge contract and native installation route.  The compiler supplies
only the derived kernel dimension and exact topology identity.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

from pops.fields._identity import field_identity, strict_field_data


_INTERFACE = "pops.prepared-field-nullspace-provider@1"


def _freeze(value: Any, *, where: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite value" % where)
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
class PreparedFieldNullspaceFacts:
    topology_identity: str
    kernel_components: int
    operator: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.topology_identity) is not str or not self.topology_identity:
            raise TypeError("field nullspace topology identity must be a non-empty exact string")
        if type(self.kernel_components) is not int or self.kernel_components < 0:
            raise ValueError("field nullspace kernel component count must be nonnegative")
        if not isinstance(self.operator, Mapping):
            raise TypeError("field nullspace operator facts must be a mapping")
        operator = _freeze(self.operator, where="field nullspace operator facts")
        strict_field_data(_plain(operator))
        object.__setattr__(self, "operator", operator)

    def to_data(self) -> dict[str, Any]:
        return {
            "topology_identity": self.topology_identity,
            "kernel_components": self.kernel_components,
            "operator": _plain(self.operator),
        }

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldNullspaceFacts:
        expected = {"topology_identity", "kernel_components", "operator"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared field nullspace facts have an invalid shape")
        return cls(
            value["topology_identity"], value["kernel_components"], value["operator"]
        )


@dataclass(frozen=True, slots=True)
class PreparedFieldNullspaceResolution:
    native_contract: Mapping[str, Any]
    singular: bool

    def __post_init__(self) -> None:
        if not isinstance(self.native_contract, Mapping):
            raise TypeError("prepared field nullspace native contract must be a mapping")
        contract = _freeze(
            self.native_contract, where="prepared field nullspace native contract"
        )
        strict_field_data(_plain(contract))
        object.__setattr__(self, "native_contract", contract)
        if type(self.singular) is not bool:
            raise TypeError("prepared field nullspace singular capability must be an exact bool")

    def to_data(self) -> dict[str, Any]:
        return {"native_contract": _plain(self.native_contract), "singular": self.singular}

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldNullspaceResolution:
        if not isinstance(value, Mapping) or set(value) != {"native_contract", "singular"}:
            raise ValueError("prepared field nullspace resolution has an invalid shape")
        return cls(value["native_contract"], value["singular"])


class PreparedFieldNullspaceAuthor(Protocol):
    def __call__(
        self,
        options: Mapping[str, Any],
        gauge: Any,
        facts: PreparedFieldNullspaceFacts,
        where: str,
    ) -> PreparedFieldNullspaceResolution:
        ...


class PreparedFieldNullspaceDefaultResolver(Protocol):
    def __call__(
        self, facts: PreparedFieldNullspaceFacts,
    ) -> tuple[PreparedFieldNullspaceProvider, Mapping[str, Any]]:
        ...


class PreparedFieldNullspaceResolutionValidator(Protocol):
    def __call__(
        self,
        binding: PreparedFieldNullspaceBinding,
        where: str,
    ) -> None:
        """Re-authenticate one provider-owned resolution after deserialization."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedFieldNullspaceProvider:
    provider_id: str
    version: int
    resolver_id: str
    resolution_validator_id: str
    installer_id: str
    capabilities: Mapping[str, Any]
    author: PreparedFieldNullspaceAuthor
    resolution_validator: PreparedFieldNullspaceResolutionValidator
    native_installer: Callable[[Any, PreparedFieldNullspaceBinding], None]

    def __post_init__(self) -> None:
        for name in (
            "provider_id", "resolver_id", "resolution_validator_id", "installer_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError("prepared field nullspace %s must be non-empty" % name)
        if type(self.version) is not int or self.version < 1:
            raise ValueError("prepared field nullspace provider version must be positive")
        if not isinstance(self.capabilities, Mapping):
            raise TypeError("prepared field nullspace capabilities must be a mapping")
        capabilities = _freeze(
            self.capabilities, where="prepared field nullspace capabilities"
        )
        strict_field_data(_plain(capabilities))
        object.__setattr__(self, "capabilities", capabilities)
        if not callable(self.author) \
                or not callable(self.resolution_validator) \
                or not callable(self.native_installer):
            raise TypeError("prepared field nullspace provider protocol is incomplete")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "interface": _INTERFACE,
            "provider_id": self.provider_id,
            "version": self.version,
            "resolver_id": self.resolver_id,
            "resolution_validator_id": self.resolution_validator_id,
            "installer_id": self.installer_id,
            "capabilities": _plain(self.capabilities),
        }

    def prepare(
        self,
        *,
        options: Mapping[str, Any],
        gauge: Any,
        facts: PreparedFieldNullspaceFacts,
        where: str,
    ) -> PreparedFieldNullspaceBinding:
        if not isinstance(options, Mapping):
            raise TypeError("%s nullspace options must be a mapping" % where)
        frozen = _freeze(options, where="%s nullspace options" % where)
        resolution = self.author(frozen, gauge, facts, where)
        if type(resolution) is not PreparedFieldNullspaceResolution:
            raise TypeError("prepared field nullspace author returned an invalid resolution")
        binding = PreparedFieldNullspaceBinding.create(
            provider=self, options=frozen, facts=facts, resolution=resolution
        )
        self.validate_binding(binding, where=where)
        return binding

    def validate_binding(
        self, binding: PreparedFieldNullspaceBinding, *, where: str,
    ) -> None:
        if type(binding) is not PreparedFieldNullspaceBinding:
            raise TypeError("%s requires an exact field nullspace binding" % where)
        if _plain(binding.provider) != self.authority():
            raise ValueError("%s field nullspace provider authority changed" % where)
        if binding.identity != binding.expected_identity():
            raise ValueError("%s field nullspace binding identity is not canonical" % where)
        result = self.resolution_validator(binding, where)
        if result is not None:
            raise TypeError("field nullspace resolution validator must return None")

    def install(self, context: Any, binding: PreparedFieldNullspaceBinding) -> None:
        self.validate_binding(binding, where="native field nullspace install")
        result = self.native_installer(context, binding)
        if result is not None:
            raise TypeError("field nullspace native installer must return None")


@dataclass(frozen=True, slots=True)
class PreparedFieldNullspaceBinding:
    provider: Mapping[str, Any]
    options: Mapping[str, Any]
    facts: PreparedFieldNullspaceFacts
    resolution: PreparedFieldNullspaceResolution
    identity: str

    def __post_init__(self) -> None:
        provider = _freeze(self.provider, where="field nullspace provider authority")
        options = _freeze(self.options, where="field nullspace options")
        if not isinstance(provider, Mapping) or not isinstance(options, Mapping):
            raise TypeError("field nullspace provider/options must be mappings")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "options", options)
        if type(self.facts) is not PreparedFieldNullspaceFacts:
            raise TypeError("field nullspace binding requires exact facts")
        if type(self.resolution) is not PreparedFieldNullspaceResolution:
            raise TypeError("field nullspace binding requires an exact resolution")
        if type(self.identity) is not str or not self.identity:
            raise TypeError("field nullspace binding identity must be non-empty")

    @classmethod
    def create(
        cls,
        *,
        provider: PreparedFieldNullspaceProvider,
        options: Mapping[str, Any],
        facts: PreparedFieldNullspaceFacts,
        resolution: PreparedFieldNullspaceResolution,
    ) -> PreparedFieldNullspaceBinding:
        pending = cls(provider.authority(), options, facts, resolution, "pending")
        return cls(
            provider.authority(), options, facts, resolution, pending.expected_identity()
        )

    def identity_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": _plain(self.provider),
            "options": _plain(self.options),
            "facts": self.facts.to_data(),
            "resolution": self.resolution.to_data(),
        }

    def expected_identity(self) -> str:
        return field_identity("prepared-field-nullspace-binding", self.identity_data()).token

    def to_data(self) -> dict[str, Any]:
        return {**self.identity_data(), "identity": self.identity}

    @classmethod
    def from_data(cls, value: Any) -> PreparedFieldNullspaceBinding:
        expected = {
            "schema_version", "provider", "options", "facts", "resolution", "identity"
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema_version") != 1
            or type(value.get("schema_version")) is not int
        ):
            raise ValueError("prepared field nullspace binding has an invalid shape")
        binding = cls(
            value["provider"], value["options"],
            PreparedFieldNullspaceFacts.from_data(value["facts"]),
            PreparedFieldNullspaceResolution.from_data(value["resolution"]),
            value["identity"],
        )
        provider = prepared_field_nullspace_provider_from_identity(binding.provider)
        provider.validate_binding(binding, where="resolved field nullspace binding")
        return binding


_lock = RLock()
_providers: dict[str, PreparedFieldNullspaceProvider] = {}
_default_policy: PreparedFieldNullspaceDefaultPolicy | None = None


@dataclass(frozen=True, slots=True)
class PreparedFieldNullspaceDefaultPolicy:
    """Single versioned authority for omitted field-nullspace descriptors.

    Provider registration is deliberately not an implicit-selection mechanism.  An extension is
    selected explicitly through ``PreparedFieldNullspace`` unless the application deliberately
    installs one default policy before authoring begins.
    """

    policy_id: str
    version: int
    resolver: PreparedFieldNullspaceDefaultResolver

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id:
            raise TypeError("field nullspace default policy id must be non-empty")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("field nullspace default policy version must be positive")
        if not callable(self.resolver):
            raise TypeError("field nullspace default policy requires a resolver")

    def resolve(
        self, facts: PreparedFieldNullspaceFacts,
    ) -> tuple[PreparedFieldNullspaceProvider, Mapping[str, Any]]:
        value = self.resolver(facts)
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or type(value[0]) is not PreparedFieldNullspaceProvider
            or not isinstance(value[1], Mapping)
        ):
            raise TypeError("field nullspace default policy returned an invalid provider binding")
        provider, options = value
        if prepared_field_nullspace_provider_by_resolver_id(provider.resolver_id) is not provider:
            raise ValueError("field nullspace default policy returned an unregistered provider")
        return provider, options


def register_prepared_field_nullspace_default_policy(
    policy: PreparedFieldNullspaceDefaultPolicy,
) -> PreparedFieldNullspaceDefaultPolicy:
    """Install the one append-only authority used when the author omits a descriptor."""
    if type(policy) is not PreparedFieldNullspaceDefaultPolicy:
        raise TypeError("field nullspace default policy must be an exact Policy")
    global _default_policy
    with _lock:
        if _default_policy is not None:
            raise ValueError("field nullspace default policy is already registered")
        _default_policy = policy
    return policy


def register_prepared_field_nullspace_provider(
    provider: PreparedFieldNullspaceProvider,
) -> PreparedFieldNullspaceProvider:
    if type(provider) is not PreparedFieldNullspaceProvider:
        raise TypeError("field nullspace plugins must register an exact Provider")
    with _lock:
        if provider.resolver_id in _providers or any(
            current.provider_id == provider.provider_id
            or current.resolution_validator_id == provider.resolution_validator_id
            or current.installer_id == provider.installer_id
            for current in _providers.values()
        ):
            raise ValueError("prepared field nullspace provider identity is already registered")
        _providers[provider.resolver_id] = provider
    return provider


def prepared_field_nullspace_provider_by_resolver_id(
    resolver_id: Any,
) -> PreparedFieldNullspaceProvider:
    if type(resolver_id) is not str:
        raise TypeError("field nullspace resolver id must be an exact string")
    with _lock:
        provider = _providers.get(resolver_id)
    if provider is None:
        raise NotImplementedError(
            "prepared field nullspace resolver %r is not registered" % resolver_id
        )
    return provider


def prepared_field_nullspace_provider_from_identity(
    identity: Any,
) -> PreparedFieldNullspaceProvider:
    expected = {
        "schema_version", "interface", "provider_id", "version", "resolver_id",
        "resolution_validator_id", "installer_id", "capabilities",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected:
        raise ValueError("prepared field nullspace provider authority is not exact")
    provider = prepared_field_nullspace_provider_by_resolver_id(identity.get("resolver_id"))
    if _plain(identity) != provider.authority():
        raise ValueError("prepared field nullspace provider authority is inconsistent")
    return provider


def prepared_field_nullspace_binding(
    descriptor: Any,
    gauge: Any,
    *,
    facts: PreparedFieldNullspaceFacts,
    where: str,
) -> PreparedFieldNullspaceBinding:
    if descriptor is None:
        with _lock:
            policy = _default_policy
        if policy is None:
            raise RuntimeError("field nullspace inference has no registered default policy")
        provider, options = policy.resolve(facts)
    else:
        protocol = getattr(descriptor, "_prepared_field_nullspace", None)
        if not callable(protocol):
            raise TypeError("%s nullspace descriptor has no prepared field provider" % where)
        value = protocol()
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or type(value[0]) is not PreparedFieldNullspaceProvider
            or not isinstance(value[1], Mapping)
        ):
            raise TypeError("%s nullspace descriptor returned an invalid provider binding" % where)
        provider, options = value
        if prepared_field_nullspace_provider_by_resolver_id(provider.resolver_id) is not provider:
            raise ValueError("%s nullspace provider is not the registered authority" % where)
    return provider.prepare(options=options, gauge=gauge, facts=facts, where=where)


def prepared_field_nullspace_binding_from_data(value: Any) -> PreparedFieldNullspaceBinding:
    return PreparedFieldNullspaceBinding.from_data(value)


__all__ = [
    "PreparedFieldNullspaceBinding",
    "PreparedFieldNullspaceDefaultPolicy",
    "PreparedFieldNullspaceFacts",
    "PreparedFieldNullspaceProvider",
    "PreparedFieldNullspaceResolution",
    "PreparedFieldNullspaceResolutionValidator",
    "prepared_field_nullspace_binding",
    "prepared_field_nullspace_binding_from_data",
    "prepared_field_nullspace_provider_by_resolver_id",
    "prepared_field_nullspace_provider_from_identity",
    "register_prepared_field_nullspace_provider",
    "register_prepared_field_nullspace_default_policy",
]
