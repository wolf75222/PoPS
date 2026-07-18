"""Authenticated prepared-nullspace providers for matrix-free Program solves.

The compiler core carries one opaque provider contract.  A provider owns authoring validation,
native emission and its complete native build input; no nullspace family is selected by a branch in
the Program or codegen layers.  Numerical preparation and compatibility checks remain in C++.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

from pops._ir.literals import ScalarLiteral
from pops.identity import canonical_bytes
from pops.native_components import PreparedNativeComponent


_PROVIDER_SCHEMA_VERSION = 1
_NATIVE_INTERFACE = "pops.prepared-field-nullspace@1"


def _freeze_ir_data(value: Any, *, where: str) -> Any:
    """Freeze the exact Program-metadata language used by provider contracts."""
    if value is None or isinstance(value, (bool, int, str, ScalarLiteral)):
        return value
    if isinstance(value, float):
        raise TypeError("%s cannot contain an untyped binary float" % where)
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty exact strings" % where)
        return MappingProxyType(
            {
                key: _freeze_ir_data(item, where="%s.%s" % (where, key))
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_ir_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        )
    raise TypeError("%s contains opaque %s" % (where, type(value).__name__))


def _plain_ir_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_ir_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_ir_data(item) for item in value)
    return value


def _identity_data(value: Any, *, where: str) -> Any:
    """Project immutable IR data to the strict deterministic identity language."""
    if isinstance(value, ScalarLiteral):
        return {"scalar": value.to_data()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {
            key: _identity_data(item, where="%s.%s" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _identity_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    raise TypeError("%s contains opaque %s" % (where, type(value).__name__))


@dataclass(frozen=True, slots=True)
class PreparedNullspaceContracts:
    """One provider-owned mathematical declaration and representative constraint."""

    nullspace: Mapping[str, Any]
    gauge: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.nullspace, Mapping) or not isinstance(self.gauge, Mapping):
            raise TypeError("prepared nullspace and gauge contracts must be exact mappings")
        nullspace = _freeze_ir_data(self.nullspace, where="prepared nullspace contract")
        gauge = _freeze_ir_data(self.gauge, where="prepared gauge contract")
        canonical_bytes(
            _identity_data(nullspace, where="prepared nullspace contract identity")
        )
        canonical_bytes(_identity_data(gauge, where="prepared gauge contract identity"))
        object.__setattr__(self, "nullspace", nullspace)
        object.__setattr__(self, "gauge", gauge)

    def detached(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return dict(_plain_ir_data(self.nullspace)), dict(_plain_ir_data(self.gauge))


@dataclass(frozen=True, slots=True)
class PreparedNullspaceUse:
    """Exact problem facts offered to a provider-owned validation policy."""

    components: int | None
    operator_properties: Mapping[str, bool]
    contracts: PreparedNullspaceContracts

    def __post_init__(self) -> None:
        if self.components is not None and (
            type(self.components) is not int or self.components < 1
        ):
            raise ValueError("prepared nullspace component count must be positive or unresolved")
        expected = {
            "symmetric",
            "positive_definite",
            "positive_definite_on_nullspace_complement",
        }
        if (
            not isinstance(self.operator_properties, Mapping)
            or set(self.operator_properties) != expected
            or any(type(self.operator_properties[key]) is not bool for key in expected)
        ):
            raise TypeError("prepared nullspace operator properties are not canonical")
        object.__setattr__(
            self,
            "operator_properties",
            MappingProxyType(dict(self.operator_properties)),
        )
        if type(self.contracts) is not PreparedNullspaceContracts:
            raise TypeError("prepared nullspace use requires exact typed contracts")


@dataclass(frozen=True, slots=True)
class PreparedNullspaceUsePolicy:
    """Provider-owned validation with an inspectable, authenticated capability contract."""

    policy_id: str
    version: int
    capabilities: Any
    validator: Callable[[PreparedNullspaceUse, str], None]

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id:
            raise TypeError("prepared nullspace use policy id must be a non-empty exact string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("prepared nullspace use policy version must be positive")
        capabilities = _freeze_ir_data(
            self.capabilities, where="prepared nullspace use-policy capabilities"
        )
        canonical_bytes(
            _identity_data(
                capabilities, where="prepared nullspace use-policy capability identity"
            )
        )
        object.__setattr__(self, "capabilities", capabilities)
        if not callable(self.validator):
            raise TypeError("prepared nullspace use-policy validator must be callable")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "version": self.version,
            "capabilities": _plain_ir_data(self.capabilities),
        }

    def validate(self, use: PreparedNullspaceUse, *, where: str) -> None:
        result = self.validator(use, where)
        if result is not None:
            raise TypeError("prepared nullspace use policy must return None")


@dataclass(frozen=True, slots=True)
class PreparedNullspaceNativeEmission:
    """One provider-owned C++ expression constructing a real ``FieldNullspacePlan``."""

    plan: str | None

    @classmethod
    def nonsingular(cls) -> PreparedNullspaceNativeEmission:
        return cls(None)

    def __post_init__(self) -> None:
        if self.plan is not None and (type(self.plan) is not str or not self.plan):
            raise TypeError("prepared nullspace native plan must be a non-empty C++ expression")


class PreparedNullspaceEmitter(Protocol):
    def __call__(
        self,
        node: Any,
        prelude: list[str],
        contracts: PreparedNullspaceContracts,
        plan_identity: str,
        provider: PreparedNullspaceProvider,
    ) -> PreparedNullspaceNativeEmission:
        """Emit installation-time C++; scientific work remains in the native component."""
        ...


class PreparedNullspaceAuthor(Protocol):
    def __call__(
        self,
        options: Mapping[str, Any],
        gauge: Any,
        operator_properties: Mapping[str, bool],
        where: str,
    ) -> PreparedNullspaceContracts:
        """Snapshot one descriptor/gauge pair into exact provider-owned contracts."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedNullspaceProvider:
    """Complete immutable compiler contract for one prepared nullspace family."""

    provider_id: str
    emitter_id: str
    singular: bool
    use_policy: PreparedNullspaceUsePolicy
    author: PreparedNullspaceAuthor
    emitter: PreparedNullspaceEmitter
    native_component: PreparedNativeComponent

    def __post_init__(self) -> None:
        for label, value in (
            ("provider_id", self.provider_id),
            ("emitter_id", self.emitter_id),
        ):
            if type(value) is not str or not value:
                raise TypeError("prepared nullspace %s must be a non-empty exact string" % label)
        if type(self.singular) is not bool:
            raise TypeError("prepared nullspace singular capability must be an exact bool")
        if type(self.use_policy) is not PreparedNullspaceUsePolicy:
            raise TypeError("prepared nullspace provider requires an exact use policy")
        if not callable(self.author) or not callable(self.emitter):
            raise TypeError("prepared nullspace provider requires author and emitter callables")
        if type(self.native_component) is not PreparedNativeComponent:
            raise TypeError("prepared nullspace provider requires an exact native component")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": _PROVIDER_SCHEMA_VERSION,
            "interface": _NATIVE_INTERFACE,
            "provider_id": self.provider_id,
            "emitter_id": self.emitter_id,
            "singular": self.singular,
            "use_policy": self.use_policy.authority(),
            "native_component": self.native_component.authority(),
        }

    def prepare(
        self,
        *,
        options: Mapping[str, Any],
        gauge: Any,
        operator_properties: Mapping[str, bool],
        where: str,
    ) -> PreparedNullspaceContracts:
        if not isinstance(options, Mapping):
            raise TypeError("%s options must be an exact mapping" % where)
        contracts = self.author(options, gauge, operator_properties, where)
        if type(contracts) is not PreparedNullspaceContracts:
            raise TypeError("prepared nullspace author must return PreparedNullspaceContracts")
        self.validate_use(
            contracts=contracts,
            components=None,
            operator_properties=operator_properties,
            where=where,
        )
        return contracts

    def validate_use(
        self,
        *,
        contracts: PreparedNullspaceContracts,
        components: int | None,
        operator_properties: Mapping[str, bool],
        where: str,
    ) -> None:
        self.use_policy.validate(
            PreparedNullspaceUse(components, operator_properties, contracts), where=where
        )

    def enveloped_contract(self, contracts: PreparedNullspaceContracts) -> dict[str, Any]:
        nullspace, _ = contracts.detached()
        return {
            "schema_version": 1,
            "provider": self.authority(),
            "contract": nullspace,
        }

    def emit(
        self,
        *,
        node: Any,
        prelude: list[str],
        contracts: PreparedNullspaceContracts,
    ) -> str:
        exact_parameters = canonical_bytes(
            {
                "interface": _NATIVE_INTERFACE,
                "provider": self.authority(),
                "contracts": _identity_data(
                    {
                        "nullspace": contracts.nullspace,
                        "gauge": contracts.gauge,
                    },
                    where="prepared nullspace native contract",
                ),
            }
        )
        digest = hashlib.sha256(exact_parameters).hexdigest()
        plan_identity = "pops://prepared-nullspace/%s@1" % digest
        emission = self.emitter(node, prelude, contracts, plan_identity, self)
        if type(emission) is not PreparedNullspaceNativeEmission:
            raise TypeError("prepared nullspace emitter must return a typed native emission")
        if (emission.plan is not None) != self.singular:
            raise ValueError("prepared nullspace emission disagrees with its singular capability")
        if emission.plan is None:
            return "pops::PreparedNullspacePolicy::nonsingular()"
        plan_name = "krylov_nullspace_plan%d" % node.id
        prelude.append("auto %s = %s;" % (plan_name, emission.plan))
        prelude.append(
            "%s.identity = %s;"
            % (plan_name, json.dumps(plan_identity, ensure_ascii=True))
        )
        prelude.append(
            "%s.layout_identity = %s;"
            % (plan_name, json.dumps(plan_identity + ":layout", ensure_ascii=True))
        )
        prelude.append("ctx.configure_program_resource_field_nullspace(%s);" % plan_name)
        return (
            "pops::PreparedNullspacePolicy::preserving(std::move(%s), "
            "ctx.program_resource_field_level())" % plan_name
        )


_registry_lock = RLock()
_providers_by_emitter_id: dict[str, PreparedNullspaceProvider] = {}


def register_prepared_nullspace_provider(
    provider: PreparedNullspaceProvider,
) -> PreparedNullspaceProvider:
    if type(provider) is not PreparedNullspaceProvider:
        raise TypeError("prepared nullspace plugins must register an exact Provider")
    with _registry_lock:
        if provider.emitter_id in _providers_by_emitter_id:
            raise ValueError(
                "prepared nullspace emitter %r is already registered" % provider.emitter_id
            )
        if any(
            existing.provider_id == provider.provider_id
            for existing in _providers_by_emitter_id.values()
        ):
            raise ValueError(
                "prepared nullspace provider id %r is already registered" % provider.provider_id
            )
        _providers_by_emitter_id[provider.emitter_id] = provider
    return provider


def prepared_nullspace_provider_by_emitter_id(
    emitter_id: Any,
) -> PreparedNullspaceProvider:
    if type(emitter_id) is not str:
        raise TypeError("prepared nullspace emitter identity must be an exact string")
    with _registry_lock:
        provider = _providers_by_emitter_id.get(emitter_id)
    if provider is None:
        raise NotImplementedError(
            "prepared nullspace emitter %r is not registered" % emitter_id
        )
    return provider


def prepared_nullspace_provider_from_identity(identity: Any) -> PreparedNullspaceProvider:
    expected = {
        "schema_version",
        "interface",
        "provider_id",
        "emitter_id",
        "singular",
        "use_policy",
        "native_component",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected:
        raise ValueError("prepared nullspace provider identity is not exact")
    emitter_id = identity.get("emitter_id")
    provider = prepared_nullspace_provider_by_emitter_id(emitter_id)
    if identity != provider.authority():
        raise ValueError("prepared nullspace provider identity is inconsistent")
    return provider


def _contracts_from_attrs(
    attrs: Mapping[str, Any], provider: PreparedNullspaceProvider
) -> PreparedNullspaceContracts:
    envelope = attrs.get("nullspace_contract")
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"schema_version", "provider", "contract"}
        or type(envelope.get("schema_version")) is not int
        or envelope.get("schema_version") != 1
        or envelope.get("provider") != provider.authority()
        or not isinstance(envelope.get("contract"), Mapping)
    ):
        raise ValueError("solve_linear nullspace contract is not an exact provider envelope")
    gauge = attrs.get("gauge_contract")
    if not isinstance(gauge, Mapping):
        raise ValueError("solve_linear gauge contract must be a provider-owned mapping")
    return PreparedNullspaceContracts(dict(envelope["contract"]), dict(gauge))


def prepared_nullspace_provider_from_attrs(
    attrs: Mapping[str, Any],
) -> PreparedNullspaceProvider:
    if not isinstance(attrs, Mapping):
        raise TypeError("solve_linear attributes must be a mapping")
    provider = prepared_nullspace_provider_from_identity(attrs.get("nullspace_provider"))
    envelope = attrs.get("nullspace_contract")
    if not isinstance(envelope, Mapping) or envelope.get("provider") != provider.authority():
        raise ValueError("solve_linear nullspace provider and contract disagree")
    return provider


def prepared_nullspace_contracts_from_attrs(
    attrs: Mapping[str, Any],
) -> tuple[PreparedNullspaceProvider, PreparedNullspaceContracts]:
    provider = prepared_nullspace_provider_from_attrs(attrs)
    return provider, _contracts_from_attrs(attrs, provider)


def _none_author(
    options: Mapping[str, Any], gauge: Any, _properties: Mapping[str, bool], where: str
) -> PreparedNullspaceContracts:
    if options:
        raise TypeError("%s nonsingular provider takes no options" % where)
    if gauge is not None:
        raise ValueError("%s gauge must be None for a nonsingular problem" % where)
    return PreparedNullspaceContracts(
        {"declaration": "nonsingular"}, {"constraint": "none"}
    )


def _constant_author(
    options: Mapping[str, Any], gauge: Any, _properties: Mapping[str, bool], where: str
) -> PreparedNullspaceContracts:
    if options:
        raise TypeError("%s constant provider takes no options" % where)
    from pops.fields.gauges import MeanValueGauge
    from pops._ir.literals import scalar_literal

    if type(gauge) is not MeanValueGauge:
        raise TypeError("%s constant nullspace requires exactly MeanValueGauge(value)" % where)
    try:
        value = scalar_literal(gauge.value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("%s MeanValueGauge value must be one finite scalar literal" % where) from exc
    return PreparedNullspaceContracts(
        {"basis": "constant-function", "basis_count": 1},
        {"constraint": "mean-value", "value": value},
    )


def _validate_none(use: PreparedNullspaceUse, where: str) -> None:
    nullspace, gauge = use.contracts.detached()
    if nullspace != {"declaration": "nonsingular"} or gauge != {"constraint": "none"}:
        raise ValueError("%s nonsingular provider contract is inconsistent" % where)
    if use.operator_properties["positive_definite_on_nullspace_complement"]:
        raise ValueError(
            "%s positive_definite_on_nullspace_complement requires nullspace with a singular "
            "provider" % where
        )


def _validate_constant(use: PreparedNullspaceUse, where: str) -> None:
    nullspace, gauge = use.contracts.detached()
    if nullspace != {"basis": "constant-function", "basis_count": 1}:
        raise ValueError("%s constant-basis contract is inconsistent" % where)
    if (
        set(gauge) != {"constraint", "value"}
        or gauge.get("constraint") != "mean-value"
        or type(gauge.get("value")) is not ScalarLiteral
    ):
        raise ValueError("%s constant-basis gauge contract is inconsistent" % where)
    if use.components is not None and use.components != 1:
        raise ValueError("%s constant-basis provider is scalar-only" % where)
    if not use.operator_properties["symmetric"]:
        raise ValueError(
            "%s constant-basis provider requires a symmetric operator certificate" % where
        )
    if use.operator_properties["positive_definite"]:
        raise ValueError(
            "%s singular operator cannot carry a global positive_definite certificate" % where
        )


def _emit_none(
    _node: Any,
    _prelude: list[str],
    _contracts: PreparedNullspaceContracts,
    _plan_identity: str,
    _provider: PreparedNullspaceProvider,
) -> PreparedNullspaceNativeEmission:
    return PreparedNullspaceNativeEmission.nonsingular()


def _emit_constant(
    _node: Any,
    _prelude: list[str],
    contracts: PreparedNullspaceContracts,
    plan_identity: str,
    _provider: PreparedNullspaceProvider,
) -> PreparedNullspaceNativeEmission:
    from pops._ir.literals import scalar_cpp

    gauge = contracts.gauge
    expression = (
        "[&]() { auto plan = pops::constant_mean_zero_nullspace(%s, "
        "\"authored constant-basis prepared provider\"); "
        "plan.gauges.front().value = static_cast<pops::Real>(%s); return plan; }()"
        % (json.dumps(plan_identity, ensure_ascii=True), scalar_cpp(gauge["value"]))
    )
    return PreparedNullspaceNativeEmission(expression)


_NONE_PROVIDER = register_prepared_nullspace_provider(
    PreparedNullspaceProvider(
        provider_id="pops.prepared-nullspace.nonsingular",
        emitter_id="pops.prepared-nullspace.nonsingular@1",
        singular=False,
        use_policy=PreparedNullspaceUsePolicy(
            "pops.prepared-nullspace.nonsingular-use",
            1,
            {"components": "any", "operator": "invertible-certificate"},
            _validate_none,
        ),
        author=_none_author,
        emitter=_emit_none,
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.prepared-nullspace.nonsingular"
        ),
    )
)

_CONSTANT_PROVIDER = register_prepared_nullspace_provider(
    PreparedNullspaceProvider(
        provider_id="pops.prepared-nullspace.constant-basis",
        emitter_id="pops.prepared-nullspace.constant-basis@1",
        singular=True,
        use_policy=PreparedNullspaceUsePolicy(
            "pops.prepared-nullspace.constant-basis-use",
            1,
            {"components": {"minimum": 1, "maximum": 1}, "gauge": "mean-value"},
            _validate_constant,
        ),
        author=_constant_author,
        emitter=_emit_constant,
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.prepared-nullspace.constant-basis",
            entry_headers=(
                "pops/numerics/elliptic/interface/field_nullspace.hpp",
                "pops/numerics/elliptic/linear/prepared_affine_problem.hpp",
            ),
        ),
    )
)


def none_prepared_nullspace_provider() -> PreparedNullspaceProvider:
    return _NONE_PROVIDER


def constant_prepared_nullspace_provider() -> PreparedNullspaceProvider:
    return _CONSTANT_PROVIDER


def prepared_nullspace_binding_from_descriptor(
    nullspace: Any,
) -> tuple[PreparedNullspaceProvider, Mapping[str, Any]]:
    if nullspace is None:
        return _NONE_PROVIDER, {}
    binding = getattr(nullspace, "_program_prepared_nullspace", None)
    if not callable(binding):
        raise TypeError(
            "LinearProblem.nullspace must be None or a prepared nullspace descriptor"
        )
    value = binding()
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not PreparedNullspaceProvider
        or not isinstance(value[1], Mapping)
    ):
        raise TypeError("prepared nullspace descriptor returned an invalid provider binding")
    registered = prepared_nullspace_provider_by_emitter_id(value[0].emitter_id)
    if registered is not value[0]:
        raise ValueError("prepared nullspace descriptor provider is not the registered authority")
    return value[0], value[1]


__all__ = [
    "PreparedNullspaceContracts",
    "PreparedNullspaceEmitter",
    "PreparedNullspaceNativeEmission",
    "PreparedNullspaceProvider",
    "PreparedNullspaceUse",
    "PreparedNullspaceUsePolicy",
    "constant_prepared_nullspace_provider",
    "none_prepared_nullspace_provider",
    "prepared_nullspace_binding_from_descriptor",
    "prepared_nullspace_contracts_from_attrs",
    "prepared_nullspace_provider_by_emitter_id",
    "prepared_nullspace_provider_from_attrs",
    "prepared_nullspace_provider_from_identity",
    "register_prepared_nullspace_provider",
]
