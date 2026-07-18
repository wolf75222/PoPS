"""Prepared-runtime preconditioner provider registry.

The registry in this private module is an append-only *compiler extension* surface.  One provider
owns the complete provider metadata and its C++ emitter, so authoring, validation, allocation
planning, and emission cannot drift across parallel registries. A plugin may register a provider
during compiler startup, but it must also ship the authenticated native source tree named by that
provider.  Registration does not create a dynamic native ABI and never moves numerical work into
Python.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
import json
from threading import RLock
from typing import Any, Protocol

from pops._frozen_data import freeze_data
from pops.identity import canonical_bytes
from pops.native_components import PreparedNativeComponent


_PROVIDER_SCHEMA_VERSION = 6
_GEOMETRIC_MG_SCRATCH_NOTE = (
    "geometric MG hierarchy ~4/3 of the fine grid (geometric V-cycle hierarchy)"
)


def _plain_authority_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_authority_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_authority_data(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerScratchResource:
    """One provider-owned persistent resource reported by inert scratch inspection."""

    kind: str
    buffers: int
    exact: bool
    note: str

    def __post_init__(self) -> None:
        _require_exact_nonempty_string(self.kind, field="scratch resource kind")
        if type(self.buffers) is not int or self.buffers < 0:
            raise ValueError("prepared preconditioner scratch buffers must be non-negative")
        if type(self.exact) is not bool:
            raise TypeError("prepared preconditioner scratch exactness must be an exact bool")
        if type(self.note) is not str:
            raise TypeError("prepared preconditioner scratch note must be an exact string")

    def authority(self) -> dict[str, Any]:
        """Return the inert allocation facts carried verbatim by prepared IR."""
        return {
            "schema_version": 1,
            "kind": self.kind,
            "buffers": self.buffers,
            "exact": self.exact,
            "note": self.note,
        }

    @classmethod
    def from_authority(cls, value: Any) -> PreparedPreconditionerScratchResource:
        expected = {"schema_version", "kind", "buffers", "exact", "note"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared preconditioner scratch resource authority is not exact")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("prepared preconditioner scratch resource schema is unsupported")
        return cls(value["kind"], value["buffers"], value["exact"], value["note"])


class PreparedPreconditionerOption(Protocol):
    """Immutable value-option protocol consumed generically by compilation providers."""

    @property
    def name(self) -> str:
        """Stable option key."""
        ...

    def validate(self, value: Any, *, where: str) -> Any:
        """Return one canonical value or raise without coercion."""
        ...

    def resolve(self, values: Mapping[str, Any], *, where: str) -> Any:
        """Return the authored canonical value or this option's canonical default."""
        ...

    def emit_cpp_literal(self, value: Any) -> str:
        """Render one already-resolved canonical value as a C++ expression."""
        ...

    def contract_data(self, value: Any) -> Any:
        """Project one resolved value to the strict canonical identity language."""
        ...

    def authority(self) -> Mapping[str, Any]:
        """Describe key, type, default and constraints without executing validation."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerNativeEmission:
    """Native workspace-session factory emitted before the authenticated wrapper.

    Providers may declare immutable authority in the prelude, but mutable execution state must be
    constructed afresh by ``make_session``. The registry wraps that factory with the provider
    identity, ABI version, component manifest and resolved options.
    """

    make_session: str | None

    @classmethod
    def identity(cls) -> PreparedPreconditionerNativeEmission:
        return cls(None)

    def __post_init__(self) -> None:
        if self.make_session is not None and (
            type(self.make_session) is not str or not self.make_session
        ):
            raise TypeError(
                "prepared preconditioner native session factory must be a non-empty expression"
            )


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerUse:
    """Typed problem facts offered to one provider-owned compatibility policy."""

    method_provider: Mapping[str, Any]
    components: int
    nullspace_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        from pops.solvers.krylov._prepared_method_registry import (
            prepared_krylov_method_provider_from_identity,
        )

        provider = prepared_krylov_method_provider_from_identity(self.method_provider)
        object.__setattr__(self, "method_provider", freeze_data(
            provider.authority(), "prepared preconditioner method provider"
        ))
        if type(self.components) is not int or self.components < 1:
            raise ValueError("prepared preconditioner use components must be positive")
        if not isinstance(self.nullspace_contract, Mapping):
            raise TypeError("prepared preconditioner use nullspace contract must be a mapping")
        contract = freeze_data(
            dict(self.nullspace_contract), "prepared preconditioner use nullspace contract"
        )
        canonical_bytes(_plain_authority_data(contract))
        object.__setattr__(self, "nullspace_contract", contract)


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerUsePolicy:
    """Authenticated provider-owned validator for one prepared problem use.

    The compiler core supplies typed facts and never branches on a provider name, nullspace family
    or component limit. A provider may implement any compatibility rule through ``validator``;
    ``capabilities`` is its canonical, inspectable contract and therefore participates in IR,
    artifact and native MPI identity.
    """

    policy_id: str
    version: int
    capabilities: Any
    validator: Callable[[PreparedPreconditionerUse, str], None]

    def __post_init__(self) -> None:
        _require_exact_nonempty_string(self.policy_id, field="use policy id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("prepared preconditioner use policy version must be positive")
        capabilities = freeze_data(
            self.capabilities, "prepared preconditioner use policy capabilities"
        )
        canonical_bytes(_plain_authority_data(capabilities))
        object.__setattr__(self, "capabilities", capabilities)
        if not callable(self.validator):
            raise TypeError("prepared preconditioner use policy validator must be callable")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "version": self.version,
            "capabilities": _plain_authority_data(self.capabilities),
        }

    def validate(self, use: PreparedPreconditionerUse, *, where: str) -> None:
        result = self.validator(use, where)
        if result is not None:
            raise TypeError(
                "prepared preconditioner use policy %r must return None" % self.policy_id
            )


class PreparedPreconditionerEmitter(Protocol):
    """Python-side C++ source emitter; numerical execution remains in the native provider."""

    def __call__(
        self,
        node: Any,
        prelude: Any,
        prototype: str,
        vector_distribution_expr: str,
        provider: PreparedPreconditionerProvider,
    ) -> PreparedPreconditionerNativeEmission:
        """Emit provider-owned native callbacks for the authenticated common wrapper."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerIntOption:
    """One exact signed-native-int option in constructor order."""

    name: str
    default: int
    minimum: int
    maximum: int = (1 << 31) - 1

    def __post_init__(self) -> None:
        _require_exact_nonempty_string(self.name, field="option name")
        for label, value in (
            ("default", self.default),
            ("minimum", self.minimum),
            ("maximum", self.maximum),
        ):
            if type(value) is not int:
                raise TypeError("prepared preconditioner int option %s must be an exact int" % label)
        if self.minimum > self.maximum or not self.minimum <= self.default <= self.maximum:
            raise ValueError("prepared preconditioner int option bounds/default are incoherent")

    def validate(self, value: Any, *, where: str) -> int:
        """Validate one authored or deserialised option against the native contract."""
        if type(value) is not int:
            raise TypeError("%s %s must be a Python int; got %r" % (where, self.name, value))
        if value < self.minimum or value > self.maximum:
            raise ValueError(
                "%s %s must be in [%d, %d]; got %r"
                % (where, self.name, self.minimum, self.maximum, value)
            )
        return value

    def resolve(self, values: Mapping[str, Any], *, where: str) -> int:
        """Resolve an authored value or the schema-owned native default."""
        return self.validate(values.get(self.name, self.default), where=where)

    def emit_cpp_literal(self, value: Any) -> str:
        """Render one validated signed integer without implicit conversion."""
        if type(value) is not int:
            raise TypeError("resolved option %s is not an exact integer" % self.name)
        return str(value)

    def contract_data(self, value: Any) -> int:
        """Return the exact signed integer consumed by the native constructor."""
        if type(value) is not int:
            raise TypeError("resolved option %s is not an exact integer" % self.name)
        return value

    def authority(self) -> dict[str, Any]:
        """Return the complete canonical option schema consumed by the compiler."""
        return {
            "schema_version": 1,
            "type_id": "pops.prepared-preconditioner.option.signed-int@1",
            "name": self.name,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def _option_authority(option: PreparedPreconditionerOption) -> Mapping[str, Any]:
    authority = getattr(option, "authority", None)
    if not callable(authority):
        raise TypeError("prepared preconditioner option is missing authority")
    value = authority()
    required = {"schema_version", "type_id", "name", "default"}
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError(
            "prepared preconditioner option authority must declare schema, type, name and default"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] < 1:
        raise ValueError("prepared preconditioner option schema version must be positive")
    _require_versioned_identity(value["type_id"], field="option type id")
    if type(value["name"]) is not str or value["name"] != option.name:
        raise ValueError("prepared preconditioner option authority name is inconsistent")
    frozen = freeze_data(dict(value), "prepared preconditioner option authority")
    canonical_bytes(_plain_authority_data(frozen))
    return frozen


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerAllocationPlan:
    """Authenticated, inert allocation plan consumed without consulting the live registry."""

    planner_id: str
    prepared_buffers: int
    scratch_resources: tuple[PreparedPreconditionerScratchResource, ...]

    def __post_init__(self) -> None:
        _require_versioned_identity(self.planner_id, field="planner id")
        if type(self.prepared_buffers) is not int or self.prepared_buffers < 0:
            raise ValueError("prepared preconditioner prepared_buffers must be non-negative")
        if type(self.scratch_resources) is not tuple or any(
            type(resource) is not PreparedPreconditionerScratchResource
            for resource in self.scratch_resources
        ):
            raise TypeError("prepared preconditioner allocation resources must be exact typed records")

    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "planner_id": self.planner_id,
            "prepared_buffers": self.prepared_buffers,
            "scratch_resources": tuple(
                resource.authority() for resource in self.scratch_resources
            ),
        }

    @classmethod
    def from_authority(cls, value: Any) -> PreparedPreconditionerAllocationPlan:
        expected = {"schema_version", "planner_id", "prepared_buffers", "scratch_resources"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("prepared preconditioner allocation plan authority is not exact")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("prepared preconditioner allocation plan schema is unsupported")
        resources = value["scratch_resources"]
        if not isinstance(resources, (list, tuple)):
            raise TypeError("prepared preconditioner scratch resources must be a sequence")
        return cls(
            value["planner_id"],
            value["prepared_buffers"],
            tuple(PreparedPreconditionerScratchResource.from_authority(item) for item in resources),
        )


@dataclass(frozen=True, slots=True)
class PreparedPreconditionerProvider:
    """Complete immutable compiler contract for one prepared preconditioner plugin."""

    provider_id: str
    interface_version: int
    options_schema: str
    scheme: str
    descriptor_name: str
    display_name: str
    native_id: str
    validator_id: str
    planner_id: str
    emitter_id: str
    preconditioned: bool
    prepared_buffers: int
    use_policy: PreparedPreconditionerUsePolicy
    options: tuple[PreparedPreconditionerOption, ...]
    emitter: PreparedPreconditionerEmitter
    native_component: PreparedNativeComponent
    scratch_resources: tuple[PreparedPreconditionerScratchResource, ...] = ()
    _option_contracts: tuple[Mapping[str, Any], ...] = dataclass_field(
        init=False, repr=False, compare=True
    )

    def __post_init__(self) -> None:
        if type(self.options) is not tuple:
            raise TypeError("prepared preconditioner provider options must be an exact tuple")
        object.__setattr__(
            self,
            "_option_contracts",
            tuple(_option_authority(option) for option in self.options),
        )

    def allocation_plan(self) -> PreparedPreconditionerAllocationPlan:
        """Snapshot the only allocation facts compiler consumers are allowed to use."""
        return PreparedPreconditionerAllocationPlan(
            self.planner_id, self.prepared_buffers, self.scratch_resources
        )

    def authority(self) -> dict[str, Any]:
        """Return the exact IR identity re-authenticated by every compiler consumer."""
        return {
            "schema_version": _PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "interface_version": self.interface_version,
            "options_schema": self.options_schema,
            "validator_id": self.validator_id,
            "emitter_id": self.emitter_id,
            "native_component": self.native_component.authority(),
            "native_id": self.native_id,
            "preconditioned": self.preconditioned,
            "use_policy": self.use_policy.authority(),
            "option_contracts": tuple(
                _plain_authority_data(contract) for contract in self._option_contracts
            ),
            "allocation_plan": self.allocation_plan().authority(),
        }

    def validate_use(
        self, *, method_provider: Mapping[str, Any], components: int,
        nullspace_contract: Mapping[str, Any], where: str
    ) -> None:
        """Delegate compatibility entirely to this provider's authenticated policy."""
        self.use_policy.validate(
            PreparedPreconditionerUse(method_provider, components, nullspace_contract), where=where
        )

    @property
    def option_names(self) -> tuple[str, ...]:
        """Option names in exact native-constructor order."""
        return tuple(option.name for option in self.options)

    def validate_options(
        self,
        values: Any,
        *,
        where: str,
        require_nonempty: bool = False,
    ) -> dict[str, Any]:
        """Validate an exact option mapping without filling omitted native defaults."""
        if not isinstance(values, Mapping):
            raise TypeError("%s options must be an exact mapping" % where)
        if require_nonempty and not values:
            raise ValueError("%s options must not be empty when explicitly carried" % where)
        unknown = set(values) - set(self.option_names)
        if unknown:
            raise TypeError(
                "%s got unknown option(s) %s; allowed options are %s"
                % (where, sorted(unknown), list(self.option_names))
            )
        validated: dict[str, Any] = {}
        for option in self.options:
            if option.name in values:
                validated[option.name] = option.validate(values[option.name], where=where)
        return validated

    def prepare_options(self, values: Any, *, where: str) -> dict[str, Any]:
        """Return the complete canonical provider-owned option mapping.

        The compiler core treats this mapping as opaque data.  Filling defaults here makes the
        native constructor contract explicit in IR and removes the former omit-when-default branch.
        """
        validated = self.validate_options(values, where=where)
        prepared = {
            option.name: option.resolve(validated, where=where) for option in self.options
        }
        frozen = freeze_data(prepared, "%s canonical options" % where)
        canonical_bytes(_plain_authority_data(frozen))
        return _plain_authority_data(frozen)

    def resolved_option_values(
        self,
        values: Any,
        *,
        where: str,
        require_nonempty: bool = False,
    ) -> tuple[Any, ...]:
        """Return validated values in constructor order, filling provider-owned defaults."""
        validated = self.validate_options(
            values, where=where, require_nonempty=require_nonempty
        )
        return tuple(option.resolve(validated, where=where) for option in self.options)

    def resolved_cpp_option_literals(
        self,
        values: Any,
        *,
        where: str,
        require_nonempty: bool = False,
    ) -> tuple[str, ...]:
        """Resolve and render options through their implementations, without type dispatch."""
        resolved = self.resolved_option_values(
            values, where=where, require_nonempty=require_nonempty
        )
        return tuple(
            option.emit_cpp_literal(value)
            for option, value in zip(self.options, resolved, strict=True)
        )

    def emit(
        self,
        node: Any,
        prelude: Any,
        prototype: str,
        vector_distribution_expr: str,
    ) -> str:
        """Emit native callbacks and wrap them in the exact provider contract."""
        emission = self.emitter(
            node, prelude, prototype, vector_distribution_expr, self
        )
        if type(emission) is not PreparedPreconditionerNativeEmission:
            raise TypeError(
                "prepared preconditioner provider %r must emit a typed native emission"
                % self.scheme
            )
        if (emission.make_session is not None) != self.preconditioned:
            raise ValueError(
                "prepared preconditioner provider %r emission disagrees with its preconditioned "
                "capability" % self.scheme
            )
        if emission.make_session is None:
            return "pops::PreparedLinearPreconditioner::identity()"

        options = node.attrs.get("preconditioner_options")
        prepared_options = self.prepare_options(
            options, where="%s options" % self.display_name
        )
        if prepared_options != options:
            raise ValueError(
                "prepared preconditioner options are not canonical for %s" % self.display_name
            )
        values = tuple(prepared_options[option.name] for option in self.options)
        option_contract = []
        for option, value in zip(self.options, values, strict=True):
            projector = getattr(option, "contract_data", None)
            if not callable(projector):
                raise TypeError(
                    "prepared preconditioner option %r has no exact contract_data projector"
                    % option.name
                )
            option_contract.append({"name": option.name, "value": projector(value)})
        exact_parameters = canonical_bytes({
            "interface": "pops.prepared-linear-preconditioner@1",
            "provider": self.authority(),
            "options": option_contract,
        })
        byte_literal = "".join("\\x%02x" % value for value in exact_parameters)
        implementation = json.dumps(self.emitter_id, ensure_ascii=True)
        provider_expression = (
            "pops::PreparedLinearPreconditionerProvider::trusted_extension("
            "pops::PreparedProviderIdentity{%s, %dull}, "
            "std::string(\"%s\", std::size_t{%d}), %s)"
            % (
                implementation,
                self.native_component.abi_version,
                byte_literal,
                len(exact_parameters),
                emission.make_session,
            )
        )
        return "pops::PreparedLinearPreconditioner(*%s, %s, %s)" % (
            prototype,
            provider_expression,
            vector_distribution_expr,
        )


_registry_lock = RLock()
_providers_by_id: dict[str, PreparedPreconditionerProvider] = {}
_providers_by_emitter_id: dict[str, PreparedPreconditionerProvider] = {}


def _require_exact_nonempty_string(value: Any, *, field: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(
            "prepared preconditioner provider %s must be a non-empty exact string" % field
        )


def _require_versioned_identity(value: Any, *, field: str) -> None:
    _require_exact_nonempty_string(value, field=field)
    _, separator, version = value.rpartition("@")
    if separator != "@" or not version.isdecimal() or int(version) < 1:
        raise ValueError(
            "prepared preconditioner provider %s must end in a positive @version" % field
        )


def _validate_provider(provider: Any) -> PreparedPreconditionerProvider:
    if not isinstance(provider, PreparedPreconditionerProvider):
        raise TypeError("prepared preconditioner plugins must register a provider record")
    for field in (
        "provider_id", "options_schema", "scheme", "descriptor_name", "display_name",
        "native_id",
    ):
        _require_exact_nonempty_string(getattr(provider, field), field=field)
    for field in ("options_schema", "validator_id", "planner_id", "emitter_id"):
        _require_versioned_identity(getattr(provider, field), field=field)
    if type(provider.interface_version) is not int or provider.interface_version < 1:
        raise ValueError("prepared preconditioner interface_version must be positive")
    if type(provider.preconditioned) is not bool:
        raise TypeError("prepared preconditioner provider preconditioned must be an exact bool")
    if type(provider.prepared_buffers) is not int or provider.prepared_buffers < 0:
        raise ValueError("prepared preconditioner provider prepared_buffers must be non-negative")
    if type(provider.use_policy) is not PreparedPreconditionerUsePolicy:
        raise TypeError("prepared preconditioner provider must carry an exact use policy")
    if type(provider.options) is not tuple:
        raise TypeError("prepared preconditioner provider options must be an exact tuple")
    if len(set(provider.option_names)) != len(provider.option_names):
        raise ValueError("prepared preconditioner provider has duplicate option names")
    if not callable(provider.emitter):
        raise TypeError("prepared preconditioner provider emitter must be callable")
    if type(provider.native_component) is not PreparedNativeComponent:
        raise TypeError(
            "prepared preconditioner provider native_component must be an exact typed component"
        )
    if type(provider.scratch_resources) is not tuple:
        raise TypeError("prepared preconditioner provider scratch resources must be an exact tuple")
    for resource in provider.scratch_resources:
        if not isinstance(resource, PreparedPreconditionerScratchResource):
            raise TypeError(
                "prepared preconditioner provider scratch resources must use typed records"
            )
    for option in provider.options:
        _require_exact_nonempty_string(getattr(option, "name", None), field="option name")
        for operation in (
            "validate", "resolve", "emit_cpp_literal", "contract_data", "authority",
        ):
            if not callable(getattr(option, operation, None)):
                raise TypeError("prepared preconditioner option is missing %s" % operation)
        _option_authority(option)
        option.resolve({}, where="prepared provider %r" % provider.scheme)
    return provider


def register_prepared_preconditioner_provider(
    provider: PreparedPreconditionerProvider,
) -> PreparedPreconditionerProvider:
    """Append one unique compiler provider; registrations are never replaced or removed."""
    provider = _validate_provider(provider)
    with _registry_lock:
        if provider.provider_id in _providers_by_id:
            raise ValueError(
                "prepared preconditioner provider %r is already registered" % provider.provider_id
            )
        if provider.emitter_id in _providers_by_emitter_id:
            raise ValueError(
                "prepared preconditioner emitter %r is already registered" % provider.emitter_id
            )
        _providers_by_id[provider.provider_id] = provider
        _providers_by_emitter_id[provider.emitter_id] = provider
    return provider


def _emit_identity(
    node: Any,
    prelude: Any,
    prototype: str,
    vector_distribution_expr: str,
    provider: PreparedPreconditionerProvider,
) -> PreparedPreconditionerNativeEmission:
    del prelude, prototype, vector_distribution_expr, provider
    if node.attrs.get("preconditioner_options") != {}:
        raise ValueError("identity preconditioner requires canonical empty provider options")
    return PreparedPreconditionerNativeEmission.identity()


def _emit_geometric_mg(
    node: Any,
    prelude: Any,
    prototype: str,
    vector_distribution_expr: str,
    provider: PreparedPreconditionerProvider,
) -> PreparedPreconditionerNativeEmission:
    name = "make_precond_mg_session%d" % node.id
    options = provider.prepare_options(
        node.attrs.get("preconditioner_options"),
        where="GeometricMG preconditioner options",
    )
    if options != node.attrs.get("preconditioner_options"):
        raise ValueError("GeometricMG preconditioner options are not canonical")
    constructor_arguments = ", ".join(
        option.emit_cpp_literal(options[option.name]) for option in provider.options
    )
    prelude.append(
        "pops::PreparedLinearPreconditionerSessionFactory %s = "
        "[ctx_owner, %s, vector_distribution = %s]("
        "const pops::ExecutionLane& lane) {"
        % (name, prototype, vector_distribution_expr)
    )
    prelude.append(
        "  auto state = std::make_shared<"
        "pops::runtime::program::GeometricMgPreconditioner>(%s);"
        % constructor_arguments
    )
    distribution_argument = (
        ", vector_distribution"
        + ", ctx.program_resource_field_storage_distribution()"
    )
    prelude.append("  return pops::PreparedLinearPreconditionerSessionCallbacks{")
    prelude.append(
        "      [ctx_owner, state, %s, vector_distribution, execution_lane = &lane]() {"
        % prototype
    )
    prelude.append("  auto& ctx = *ctx_owner;")
    prelude.append(
        "  state->prepare(ctx, *%s, *execution_lane%s);"
        % (prototype, distribution_argument)
    )
    prelude.append("      },")
    prelude.append(
        "      [ctx_owner, state, execution_lane = &lane]("
        "pops::MultiFab& out, const pops::MultiFab& in) {"
    )
    prelude.append("  auto& ctx = *ctx_owner;")
    prelude.append("  state->apply(ctx, out, in, *execution_lane);")
    prelude.append("      },")
    prelude.append("      [state]() { return state->persistent_field_count(); }};")
    prelude.append("};")
    return PreparedPreconditionerNativeEmission(name)


def _validate_identity_use(_use: PreparedPreconditionerUse, _where: str) -> None:
    return None


_GEOMETRIC_MG_METHOD_PRECONDITIONING_PLACEMENTS = ("left", "right")


def _require_supported_method_preconditioning_placement(
    use: PreparedPreconditionerUse,
    where: str,
    *,
    preconditioner: str,
    supported: tuple[str, ...],
) -> None:
    capabilities = use.method_provider.get("capabilities")
    placement = (
        capabilities.get("preconditioning_placement")
        if isinstance(capabilities, Mapping)
        else None
    )
    if type(placement) is not str or placement not in supported:
        choices = " or ".join(repr(item) for item in supported)
        raise ValueError(
            "%s: %s requires an authenticated method preconditioning placement in (%s); got %r"
            % (where, preconditioner, choices, placement)
        )


def _validate_geometric_mg_use(use: PreparedPreconditionerUse, where: str) -> None:
    _require_supported_method_preconditioning_placement(
        use,
        where,
        preconditioner="geometric multigrid",
        supported=_GEOMETRIC_MG_METHOD_PRECONDITIONING_PLACEMENTS,
    )
    if use.components != 1:
        raise ValueError(
            "%s: geometric multigrid preconditioning is scalar-only; got %d components"
            % (where, use.components)
        )
    nullspace_provider = use.nullspace_contract.get("provider")
    if not isinstance(nullspace_provider, Mapping) or type(
        nullspace_provider.get("singular")
    ) is not bool:
        raise ValueError(
            "%s: prepared nullspace contract has no authenticated provider capability"
            % where
        )
    if nullspace_provider["singular"]:
        raise NotImplementedError(
            "%s: geometric multigrid has no explicit public certificate for singular "
            "nullspace contract %r"
            % (where, dict(use.nullspace_contract))
        )


_IDENTITY_USE_POLICY = PreparedPreconditionerUsePolicy(
    "pops.prepared-preconditioner.identity-use",
    1,
    {"method_capability": "any", "components": "any", "nullspace_contracts": "any-certified"},
    _validate_identity_use,
)
_GEOMETRIC_MG_USE_POLICY = PreparedPreconditionerUsePolicy(
    "pops.prepared-preconditioner.geometric-mg-use",
    2,
    {
        "supported_method_preconditioning_placements": (
            _GEOMETRIC_MG_METHOD_PRECONDITIONING_PLACEMENTS
        ),
        "components": {"minimum": 1, "maximum": 1},
        "nullspace_contracts": "registered nonsingular providers",
    },
    _validate_geometric_mg_use,
)


_IDENTITY_PROVIDER = register_prepared_preconditioner_provider(
    PreparedPreconditionerProvider(
        provider_id="pops.preconditioner.identity",
        interface_version=1,
        options_schema="pops.preconditioner.identity.options@1",
        scheme="identity",
        descriptor_name="identity",
        display_name="preconditioners.Identity()",
        native_id="pops::ApplyFn",
        validator_id="pops.prepared-preconditioner.identity.validate@1",
        planner_id="pops.prepared-preconditioner.identity.plan@1",
        emitter_id="pops.prepared-preconditioner.identity@1",
        preconditioned=False,
        prepared_buffers=0,
        use_policy=_IDENTITY_USE_POLICY,
        options=(),
        emitter=_emit_identity,
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.prepared-preconditioner.identity"
        ),
    )
)
_GEOMETRIC_MG_PROVIDER = register_prepared_preconditioner_provider(
    PreparedPreconditionerProvider(
        provider_id="pops.preconditioner.geometric-mg",
        interface_version=1,
        options_schema="pops.preconditioner.geometric-mg.options@1",
        scheme="geometric_mg",
        descriptor_name="geometric_mg",
        display_name="preconditioners.GeometricMG()",
        native_id="pops::GeometricMG",
        validator_id="pops.prepared-preconditioner.geometric-mg.validate@1",
        planner_id="pops.prepared-preconditioner.geometric-mg.plan@1",
        emitter_id="pops.prepared-preconditioner.geometric-mg@1",
        preconditioned=True,
        prepared_buffers=2,
        use_policy=_GEOMETRIC_MG_USE_POLICY,
        options=(
            PreparedPreconditionerIntOption("pre_sweeps", default=2, minimum=0),
            PreparedPreconditionerIntOption("post_sweeps", default=2, minimum=0),
            PreparedPreconditionerIntOption("bottom_sweeps", default=50, minimum=1),
            PreparedPreconditionerIntOption("min_coarse", default=2, minimum=1),
            PreparedPreconditionerIntOption("n_vcycles", default=1, minimum=1),
        ),
        emitter=_emit_geometric_mg,
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.prepared-preconditioner.geometric-mg",
            entry_headers=("pops/runtime/program/coeff_elliptic_ops.hpp",),
        ),
        scratch_resources=(
            PreparedPreconditionerScratchResource(
                kind="multigrid_preconditioner",
                buffers=1,
                exact=False,
                note=_GEOMETRIC_MG_SCRATCH_NOTE,
            ),
        ),
    )
)


def prepared_preconditioner_providers() -> tuple[PreparedPreconditionerProvider, ...]:
    """Return a stable snapshot of all registered compilation providers."""
    with _registry_lock:
        return tuple(_providers_by_id.values())


def prepared_preconditioner_provider_by_id(
    provider_id: Any,
) -> PreparedPreconditionerProvider:
    """Resolve an exact stable provider identity through the append-only registry."""
    if type(provider_id) is not str:
        raise TypeError("prepared preconditioner provider_id must be an exact string")
    with _registry_lock:
        provider = _providers_by_id.get(provider_id)
        available = tuple(sorted(_providers_by_id))
    if provider is None:
        raise NotImplementedError(
            "preconditioner provider %r is not registered; available providers: %s"
            % (provider_id, list(available))
        )
    return provider


def prepared_preconditioner_provider_by_emitter_id(
    emitter_id: Any,
) -> PreparedPreconditionerProvider:
    """Resolve one exact versioned compiler-emitter identity without a fallback."""
    if type(emitter_id) is not str:
        raise TypeError("prepared preconditioner emitter identity must be an exact string")
    with _registry_lock:
        provider = _providers_by_emitter_id.get(emitter_id)
    if provider is None:
        raise NotImplementedError(
            "prepared preconditioner emitter %r is not registered" % emitter_id
        )
    return provider


def prepared_preconditioner_provider_from_identity(
    identity: Any,
) -> PreparedPreconditionerProvider:
    """Authenticate one serialised provider identity without name-based dispatch."""
    expected_keys = {
        "schema_version",
        "provider_id",
        "interface_version",
        "options_schema",
        "validator_id",
        "emitter_id",
        "native_component",
        "native_id",
        "preconditioned",
        "use_policy",
        "option_contracts",
        "allocation_plan",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_keys:
        raise ValueError(
            "prepared preconditioner identity is not authenticated: expected the exact provider "
            "authority"
        )
    if (
        type(identity["schema_version"]) is not int
        or identity["schema_version"] != _PROVIDER_SCHEMA_VERSION
    ):
        raise ValueError("prepared preconditioner provider uses an unsupported schema version")
    provider_id = identity["provider_id"]
    if type(provider_id) is not str:
        raise TypeError("prepared preconditioner provider identity must contain an exact provider id")
    provider = prepared_preconditioner_provider_by_id(provider_id)
    emitter = prepared_preconditioner_provider_by_emitter_id(identity["emitter_id"])
    if emitter is not provider:
        raise ValueError("prepared preconditioner provider/emitter authorities disagree")
    if identity != provider.authority():
        raise ValueError("prepared preconditioner provider identity is inconsistent")
    return provider


def prepared_preconditioner_allocation_plan_from_identity(
    identity: Any,
) -> PreparedPreconditionerAllocationPlan:
    """Read a plan from authenticated IR rather than from mutable registry fields."""
    prepared_preconditioner_provider_from_identity(identity)
    return PreparedPreconditionerAllocationPlan.from_authority(identity["allocation_plan"])


def prepared_preconditioner_provider_from_attrs(
    attrs: Mapping[str, Any],
) -> PreparedPreconditionerProvider:
    """Authenticate the provider identity carried by one solve-linear IR node."""
    if not isinstance(attrs, Mapping):
        raise TypeError("solve_linear attributes must be a mapping")
    provider = prepared_preconditioner_provider_from_identity(
        attrs.get("preconditioner_provider")
    )
    options = provider.prepare_options(
        attrs.get("preconditioner_options"),
        where="solve_linear preconditioner options",
    )
    if options != attrs.get("preconditioner_options"):
        raise ValueError("solve_linear preconditioner options are not canonical")
    return provider


__all__ = [
    "PreparedPreconditionerAllocationPlan",
    "PreparedPreconditionerEmitter",
    "PreparedPreconditionerIntOption",
    "PreparedPreconditionerNativeEmission",
    "PreparedPreconditionerOption",
    "PreparedPreconditionerProvider",
    "PreparedPreconditionerScratchResource",
    "PreparedPreconditionerUse",
    "PreparedPreconditionerUsePolicy",
    "PreparedNativeComponent",
    "prepared_preconditioner_provider_by_id",
    "prepared_preconditioner_provider_by_emitter_id",
    "prepared_preconditioner_allocation_plan_from_identity",
    "prepared_preconditioner_provider_from_attrs",
    "prepared_preconditioner_provider_from_identity",
    "prepared_preconditioner_providers",
    "register_prepared_preconditioner_provider",
]
