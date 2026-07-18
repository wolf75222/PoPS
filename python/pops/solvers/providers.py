"""Authenticated native providers for hierarchy-scoped mathematical solves.

The Program and code generator consume only :class:`PreparedHierarchySolverProvider` records.
Concrete algorithms own their authoring validation, native component and C++ emitter here; no
hierarchy backend is selected or numerically executed in Python.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from copy import deepcopy
import json
from threading import RLock
from typing import Any

from pops.identity import Identity, canonical_bytes, make_identity
from pops.native_components import PreparedNativeComponent
from pops.solvers._numeric import (
    exact_nonnegative_real,
    exact_open_unit_real,
    optional_positive_int,
)


_HIERARCHY_PROVIDER_SCHEMA_VERSION = 1
_HIERARCHY_INSTANCE_SCHEMA_VERSION = 1
_DEFAULT_MAX_ITER = 30
_DEFAULT_REL_TOL = 1.0e-9


def _exact_nonempty_string(value: Any, *, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be a non-empty exact string" % where)
    return value


def _plain_data(value: Any) -> Any:
    """Detach the Program IR's immutable mapping/tuple wrappers into canonical JSON data."""
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain_data(item) for item in value)
    return value


def _positive_max_iter(value: Any) -> int:
    checked = optional_positive_int(value, where="CompositeTensorFAC(max_iter=)")
    if checked is None:
        raise ValueError("CompositeTensorFAC(max_iter=) must be a positive int")
    return checked


@dataclass(frozen=True, slots=True)
class PreparedHierarchySolverUseFacts:
    """Typed common facts plus an append-only provider-owned extension envelope."""

    target: str | None
    scope: str
    problem_kind: str
    domain: str
    range: str
    components: int
    singular_nullspace: bool
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target is not None:
            _exact_nonempty_string(self.target, where="hierarchy use target")
        for name in ("scope", "problem_kind", "domain", "range"):
            _exact_nonempty_string(getattr(self, name), where="hierarchy use " + name)
        if type(self.components) is not int or self.components < 1:
            raise TypeError("hierarchy use components must be a positive exact integer")
        if type(self.singular_nullspace) is not bool:
            raise TypeError("hierarchy use singular_nullspace must be an exact bool")
        if not isinstance(self.extensions, Mapping):
            raise TypeError("hierarchy use extensions must be a mapping")
        if any(type(key) is not str or not key for key in self.extensions):
            raise TypeError("hierarchy use extension keys must be non-empty exact strings")
        # Canonical serialization is the authority check for arbitrary future facts. Reject values
        # that cannot participate in an immutable provider identity instead of coercing them.
        try:
            canonical_bytes(_plain_data(self.extensions))
        except Exception as exc:
            raise TypeError("hierarchy use extensions are not canonical identity data") from exc

    def canonical_data(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "scope": self.scope,
            "problem_kind": self.problem_kind,
            "domain": self.domain,
            "range": self.range,
            "components": self.components,
            "singular_nullspace": self.singular_nullspace,
            "extensions": _plain_data(self.extensions),
        }


HierarchyUseValidator = Callable[
    [PreparedHierarchySolverUseFacts, Any, str], PreparedHierarchySolverUseFacts
]


@dataclass(frozen=True, slots=True)
class PreparedHierarchySolverUsePolicy:
    """Authenticated provider-owned validator for an extensible use-fact envelope."""

    policy_id: str
    interface_version: int
    capabilities: frozenset[str]
    validator: HierarchyUseValidator = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _exact_nonempty_string(self.policy_id, where="hierarchy use policy id")
        if type(self.interface_version) is not int or self.interface_version < 1:
            raise TypeError("hierarchy use policy interface_version must be positive")
        if type(self.capabilities) is not frozenset:
            raise TypeError("hierarchy use policy capabilities must be an exact frozenset")
        if any(type(item) is not str or not item for item in self.capabilities):
            raise TypeError("hierarchy use policy capabilities must be non-empty exact strings")
        if not callable(self.validator):
            raise TypeError("hierarchy use policy requires a callable provider validator")

    def authority(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "interface_version": self.interface_version,
            "capabilities": sorted(self.capabilities),
        }

    def validate(
        self, facts: PreparedHierarchySolverUseFacts, *, operator: Any, where: str
    ) -> PreparedHierarchySolverUseFacts:
        if type(facts) is not PreparedHierarchySolverUseFacts:
            raise TypeError("%s requires exact hierarchy use facts" % where)
        validated = self.validator(facts, operator, where)
        if type(validated) is not PreparedHierarchySolverUseFacts:
            raise TypeError("%s use-policy validator returned an invalid fact record" % where)
        if canonical_bytes(validated.canonical_data()) != canonical_bytes(facts.canonical_data()):
            raise ValueError("%s use-policy validator changed the authenticated facts" % where)
        return validated


@dataclass(frozen=True, slots=True)
class PreparedHierarchyKrylovFallback:
    """Exact flat-topology fallback selected by provider capability, never by provider name."""

    method_provider_id: str
    preconditioner_provider_id: str
    method_options: Mapping[str, Any] = field(default_factory=dict)
    preconditioner_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _exact_nonempty_string(self.method_provider_id, where="flat fallback method provider")
        _exact_nonempty_string(
            self.preconditioner_provider_id,
            where="flat fallback preconditioner provider",
        )
        from pops.solvers._prepared_preconditioner_registry import (
            prepared_preconditioner_provider_by_id,
        )
        from pops.solvers.krylov._prepared_method_registry import (
            prepared_krylov_method_provider_by_id,
        )

        provider = prepared_krylov_method_provider_by_id(self.method_provider_id)
        object.__setattr__(self, "method_options", provider.prepare_options(self.method_options))
        preconditioner = prepared_preconditioner_provider_by_id(
            self.preconditioner_provider_id
        )
        object.__setattr__(
            self,
            "preconditioner_options",
            preconditioner.prepare_options(
                self.preconditioner_options,
                where="hierarchy flat-fallback preconditioner options",
            ),
        )

    def authority(self) -> dict[str, Any]:
        from pops.solvers._prepared_preconditioner_registry import (
            prepared_preconditioner_provider_by_id,
        )
        from pops.solvers.krylov._prepared_method_registry import (
            prepared_krylov_method_provider_by_id,
        )

        method_provider = prepared_krylov_method_provider_by_id(self.method_provider_id)
        preconditioner = prepared_preconditioner_provider_by_id(
            self.preconditioner_provider_id
        )
        return {
            "interface": "pops.prepared-hierarchy-flat-krylov@1",
            "method_provider": method_provider.authority(),
            "method_options": method_provider.prepare_options(self.method_options),
            "preconditioner_provider": preconditioner.authority(),
            "preconditioner_options": preconditioner.prepare_options(
                self.preconditioner_options,
                where="hierarchy flat-fallback preconditioner options",
            ),
        }

    def ir_attributes(
        self,
        *,
        components: int,
        input_ghosts: int,
        nullspace_contract: Mapping[str, Any],
        operator_properties: Mapping[str, bool],
        declared_nullspace: bool,
        relative_tolerance: Any,
        absolute_tolerance: Any,
        max_iterations: int,
    ) -> dict[str, Any]:
        from pops.solvers._prepared_preconditioner_registry import (
            prepared_preconditioner_provider_by_id,
        )
        from pops.solvers.krylov._prepared_method_registry import (
            PreparedKrylovMethodUse,
            prepared_krylov_method_provider_by_id,
        )

        method_provider = prepared_krylov_method_provider_by_id(self.method_provider_id)
        method_options = method_provider.prepare_options(self.method_options)
        preconditioner = prepared_preconditioner_provider_by_id(
            self.preconditioner_provider_id
        )
        preconditioner.validate_use(
            method_provider=method_provider.authority(),
            components=components,
            nullspace_contract=nullspace_contract,
            where="hierarchy flat-fallback preconditioner",
        )
        use = PreparedKrylovMethodUse(
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
            max_iterations=max_iterations,
            components=components,
            input_ghosts=input_ghosts,
            preconditioned=preconditioner.preconditioned,
            operator_properties=operator_properties,
            declared_nullspace=declared_nullspace,
            method_options=method_options,
        )
        method_provider.validate_use(use, where="hierarchy flat-fallback method")
        return {
            "method_provider": method_provider.authority(),
            "method_options": method_options,
            "preconditioner_provider": preconditioner.authority(),
            "preconditioner_options": preconditioner.prepare_options(
                self.preconditioner_options,
                where="hierarchy flat-fallback preconditioner options",
            ),
            "krylov_footprint": {
                "components": components,
                "input_ghosts": input_ghosts,
                "preconditioned": preconditioner.preconditioned,
            },
        }

    def validate_ir(self, attrs: Mapping[str, Any], *, where: str) -> None:
        authority = self.authority()
        if (
            attrs.get("method_provider") != authority["method_provider"]
            or attrs.get("method_options") != authority["method_options"]
            or attrs.get("preconditioner_provider") != authority["preconditioner_provider"]
            or attrs.get("preconditioner_options") != authority["preconditioner_options"]
        ):
            raise ValueError("%s flat Krylov fallback disagrees with provider authority" % where)


@dataclass(frozen=True, slots=True)
class PreparedHierarchyFlatExecution:
    """Provider-owned flat-topology execution contract.

    The two modes are protocol-level execution shapes, not solver names. A hierarchy provider may
    either delegate to a separately authenticated prepared Krylov contract, or own storage, solve and
    publication directly even for one level. Core lowering never branches on a provider identity.
    """

    mode: str
    krylov: PreparedHierarchyKrylovFallback | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("prepared_krylov_fallback", "direct_provider"):
            raise ValueError("unknown hierarchy flat execution mode %r" % self.mode)
        if self.mode == "prepared_krylov_fallback":
            if type(self.krylov) is not PreparedHierarchyKrylovFallback:
                raise TypeError("prepared Krylov flat execution requires an exact fallback contract")
        elif self.krylov is not None:
            raise TypeError("direct-provider flat execution cannot carry a Krylov fallback")

    @classmethod
    def prepared_krylov(
        cls, fallback: PreparedHierarchyKrylovFallback
    ) -> PreparedHierarchyFlatExecution:
        return cls("prepared_krylov_fallback", fallback)

    @classmethod
    def direct_provider(cls) -> PreparedHierarchyFlatExecution:
        return cls("direct_provider")

    @property
    def uses_prepared_krylov_fallback(self) -> bool:
        return self.mode == "prepared_krylov_fallback"

    def authority(self) -> dict[str, Any]:
        return {
            "interface": "pops.prepared-hierarchy-flat-execution@1",
            "mode": self.mode,
            "krylov": None if self.krylov is None else self.krylov.authority(),
        }

    def ir_attributes(self, **kwargs: Any) -> dict[str, Any]:
        if self.krylov is None:
            return {}
        return self.krylov.ir_attributes(**kwargs)

    def validate_ir(self, attrs: Mapping[str, Any], *, where: str) -> None:
        if self.krylov is not None:
            self.krylov.validate_ir(attrs, where=where)
            return
        forbidden = sorted(
            key
            for key in (
                "method_provider",
                "method_options",
                "preconditioner",
                "preconditioner_provider",
                "krylov_footprint",
                "krylov_workspace",
            )
            if key in attrs
        )
        if forbidden:
            raise ValueError(
                "%s direct-provider execution carries unexpected Krylov attributes %s"
                % (where, forbidden)
            )


@dataclass(frozen=True, slots=True)
class PreparedHierarchyConvergenceContract:
    relative_tolerance_option: str
    absolute_tolerance_option: str
    max_iterations_option: str

    def __post_init__(self) -> None:
        for name in (
            "relative_tolerance_option",
            "absolute_tolerance_option",
            "max_iterations_option",
        ):
            _exact_nonempty_string(getattr(self, name), where="convergence " + name)

    def authority(self) -> dict[str, str]:
        return {
            "relative_tolerance_option": self.relative_tolerance_option,
            "absolute_tolerance_option": self.absolute_tolerance_option,
            "max_iterations_option": self.max_iterations_option,
        }

    def values(self, options: Mapping[str, Any], *, where: str) -> tuple[Any, Any, int]:
        from pops.identity.scalar import exact_cpp_int
        from pops.model._bind_schema_data import literal_value

        try:
            relative = literal_value(
                options[self.relative_tolerance_option], where=where + " relative tolerance"
            )
            absolute = literal_value(
                options[self.absolute_tolerance_option], where=where + " absolute tolerance"
            )
            maximum = exact_cpp_int(
                options[self.max_iterations_option],
                where=where + " max iterations",
                minimum=1,
            )
        except KeyError as exc:
            raise ValueError("%s lacks its provider-owned convergence option" % where) from exc
        return relative, absolute, maximum


@dataclass(frozen=True, slots=True)
class PreparedHierarchySolverEmitRequest:
    node: Any
    target: str
    report_name: str
    solution_name: str
    components: int
    block_index: int
    relative_tolerance_cpp: str
    absolute_tolerance_cpp: str
    max_iterations: int


@dataclass(frozen=True, slots=True)
class PreparedHierarchySolverNativeEmission:
    configure: tuple[str, ...]
    solve: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("configure", "solve"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(line) is not str or not line for line in value):
                raise TypeError("hierarchy native emission %s must contain exact C++ lines" % name)
        if not self.solve:
            raise ValueError("hierarchy native emission must publish a solve report")


HierarchyOptionValidator = Callable[[Any, str], dict[str, Any]]
HierarchyAuthor = Callable[[Any, Any, Any, Any, Any], Any]
HierarchyEmitter = Callable[
    [PreparedHierarchySolverEmitRequest, Any, Mapping[str, Any]],
    PreparedHierarchySolverNativeEmission,
]


@dataclass(frozen=True, slots=True)
class PreparedHierarchySolver:
    """Immutable prepared provider instance passed from a descriptor to ``Program.solve``."""

    _provider_json: str
    _options_json: str
    identity: Identity

    @property
    def provider_authority(self) -> dict[str, Any]:
        return json.loads(self._provider_json)

    @property
    def options(self) -> dict[str, Any]:
        return json.loads(self._options_json)

    @property
    def identity_data(self) -> dict[str, Any]:
        return {
            "schema_version": _HIERARCHY_INSTANCE_SCHEMA_VERSION,
            "provider": self.provider_authority,
            "options": self.options,
        }

    def build_program_solve(self, *, program: Any, problem: Any, name: Any = None) -> Any:
        provider = prepared_hierarchy_solver_provider_from_identity(self.provider_authority)
        provider.authenticate_prepared(self)
        return provider.author(program, problem, self, name, provider)


@dataclass(frozen=True, slots=True)
class PreparedHierarchySolverProvider:
    """Complete append-only compiler contract for one native hierarchy solver provider."""

    provider_id: str
    interface_version: int
    emitter_id: str
    option_schema: str
    capabilities: frozenset[str]
    use_policy: PreparedHierarchySolverUsePolicy
    convergence: PreparedHierarchyConvergenceContract
    flat_execution: PreparedHierarchyFlatExecution
    native_component: PreparedNativeComponent
    option_validator: HierarchyOptionValidator = field(repr=False, compare=False)
    author: HierarchyAuthor = field(repr=False, compare=False)
    emitter: HierarchyEmitter = field(repr=False, compare=False)

    def authority(self) -> dict[str, Any]:
        return _plain_data({
            "schema_version": _HIERARCHY_PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "interface_version": self.interface_version,
            "emitter_id": self.emitter_id,
            "option_schema": self.option_schema,
            "capabilities": sorted(self.capabilities),
            "use_policy": self.use_policy.authority(),
            "convergence": self.convergence.authority(),
            "flat_execution": self.flat_execution.authority(),
            "native_component": self.native_component.authority(),
        })

    def validate_options(self, values: Any, *, where: str) -> dict[str, Any]:
        validated = self.option_validator(values, where)
        if type(validated) is not dict:
            raise TypeError("hierarchy provider option validator must return an exact dict")
        return validated

    def instance_data(self, options: Any) -> dict[str, Any]:
        return {
            "schema_version": _HIERARCHY_INSTANCE_SCHEMA_VERSION,
            "provider": self.authority(),
            "options": self.validate_options(
                options, where="hierarchy provider %r" % self.provider_id
            ),
        }

    def prepare(self, options: Any) -> PreparedHierarchySolver:
        data = self.instance_data(options)
        provider_json = json.dumps(
            data["provider"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        options_json = json.dumps(
            data["options"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return PreparedHierarchySolver(
            provider_json,
            options_json,
            make_identity("hierarchy-solver", data),
        )

    def authenticate_prepared(self, prepared: PreparedHierarchySolver) -> dict[str, Any]:
        if type(prepared) is not PreparedHierarchySolver:
            raise TypeError("prepared hierarchy solver has an invalid immutable record type")
        data = prepared.identity_data
        expected = self.instance_data(data["options"])
        if data != expected or prepared.identity != make_identity("hierarchy-solver", expected):
            raise ValueError("prepared hierarchy solver identity is inconsistent")
        return expected["options"]

    def validate_node(self, node: Any, *, target: str) -> dict[str, Any]:
        attrs = getattr(node, "attrs", None)
        inputs = getattr(node, "inputs", None)
        if not isinstance(attrs, Mapping) or not isinstance(inputs, (tuple, list)) or not inputs:
            raise TypeError("hierarchy solve node is malformed")
        provider = prepared_hierarchy_solver_provider_from_attrs(attrs)
        if provider is not self:
            raise ValueError("hierarchy solve resolved a different registered provider")
        options = self.validate_options(
            attrs.get("hierarchy_solver_options"), where="hierarchy solve options"
        )
        operator = inputs[0]
        operator_attrs = getattr(operator, "attrs", {})
        from pops.fields._prepared_nullspace_registry import (
            prepared_nullspace_provider_from_attrs,
        )

        nullspace = prepared_nullspace_provider_from_attrs(attrs)
        use_facts = PreparedHierarchySolverUseFacts(
            target=target,
            scope=attrs.get("scope"),
            problem_kind=attrs.get("problem_kind"),
            domain=operator_attrs.get("domain"),
            range=operator_attrs.get("range"),
            components=attrs.get("ncomp"),
            singular_nullspace=nullspace.singular,
            extensions=attrs.get("hierarchy_use_facts", {}),
        )
        self.use_policy.validate(
            use_facts,
            operator=operator,
            where="hierarchy provider %r" % self.provider_id,
        )
        self.flat_execution.validate_ir(attrs, where="hierarchy solve")
        from pops.identity.scalar import exact_cpp_int, scalar_data

        relative, absolute, maximum = self.convergence.values(
            options, where="hierarchy solve"
        )
        if (
            scalar_data(attrs.get("tol")) != scalar_data(relative)
            or scalar_data(attrs.get("abs_tol")) != scalar_data(absolute)
            or exact_cpp_int(
                attrs.get("max_iter"), where="hierarchy emitted max iterations", minimum=1
            )
            != maximum
        ):
            raise ValueError("hierarchy solve convergence controls disagree with provider identity")
        exact_cpp_int(
            attrs.get("hierarchy_block_index"),
            where="hierarchy provider block index",
            minimum=0,
        )
        return options

    def emit(
        self, request: PreparedHierarchySolverEmitRequest
    ) -> PreparedHierarchySolverNativeEmission:
        if type(request) is not PreparedHierarchySolverEmitRequest:
            raise TypeError("hierarchy provider emitter requires a typed request")
        options = self.validate_node(request.node, target=request.target)
        emission = self.emitter(request, self, options)
        if type(emission) is not PreparedHierarchySolverNativeEmission:
            raise TypeError("hierarchy provider emitter must return a typed native emission")
        return emission


_registry_lock = RLock()
_providers_by_id: dict[str, PreparedHierarchySolverProvider] = {}
_providers_by_emitter: dict[str, PreparedHierarchySolverProvider] = {}


def _validate_provider(provider: Any) -> PreparedHierarchySolverProvider:
    if type(provider) is not PreparedHierarchySolverProvider:
        raise TypeError("hierarchy solver plugins must register an exact provider record")
    _exact_nonempty_string(provider.provider_id, where="hierarchy provider id")
    _exact_nonempty_string(provider.emitter_id, where="hierarchy emitter id")
    _exact_nonempty_string(provider.option_schema, where="hierarchy option schema")
    if type(provider.interface_version) is not int or provider.interface_version < 1:
        raise ValueError("hierarchy provider interface_version must be a positive exact integer")
    if type(provider.capabilities) is not frozenset:
        raise TypeError("hierarchy provider capabilities must be an exact frozenset")
    if any(type(item) is not str or not item for item in provider.capabilities):
        raise TypeError("hierarchy provider capabilities must be non-empty exact strings")
    if type(provider.use_policy) is not PreparedHierarchySolverUsePolicy:
        raise TypeError("hierarchy provider requires an exact use policy")
    if type(provider.convergence) is not PreparedHierarchyConvergenceContract:
        raise TypeError("hierarchy provider requires an exact convergence contract")
    if type(provider.flat_execution) is not PreparedHierarchyFlatExecution:
        raise TypeError("hierarchy provider requires an exact flat execution contract")
    if type(provider.native_component) is not PreparedNativeComponent:
        raise TypeError("hierarchy provider requires a PreparedNativeComponent")
    for name in ("option_validator", "author", "emitter"):
        if not callable(getattr(provider, name)):
            raise TypeError("hierarchy provider is missing callable %s" % name)
    return provider


def register_prepared_hierarchy_solver_provider(
    provider: PreparedHierarchySolverProvider,
) -> PreparedHierarchySolverProvider:
    """Append one unique provider. Registered authority is never replaced or removed."""
    provider = _validate_provider(provider)
    with _registry_lock:
        if provider.provider_id in _providers_by_id:
            raise ValueError("hierarchy provider %r is already registered" % provider.provider_id)
        if provider.emitter_id in _providers_by_emitter:
            raise ValueError("hierarchy emitter %r is already registered" % provider.emitter_id)
        _providers_by_id[provider.provider_id] = provider
        _providers_by_emitter[provider.emitter_id] = provider
    return provider


def prepared_hierarchy_solver_provider_by_id(
    provider_id: Any,
) -> PreparedHierarchySolverProvider:
    provider_id = _exact_nonempty_string(provider_id, where="hierarchy provider id")
    with _registry_lock:
        provider = _providers_by_id.get(provider_id)
    if provider is None:
        raise NotImplementedError("hierarchy provider %r is not registered" % provider_id)
    return provider


def prepared_hierarchy_solver_provider_from_identity(
    identity: Any,
) -> PreparedHierarchySolverProvider:
    if not isinstance(identity, Mapping):
        raise TypeError("hierarchy provider identity must be an exact mapping")
    expected_keys = set(next(iter(_providers_by_id.values())).authority()) if _providers_by_id else {
        "schema_version", "provider_id", "interface_version", "emitter_id", "option_schema",
        "capabilities", "use_policy", "convergence", "flat_execution", "native_component",
    }
    if set(identity) != expected_keys:
        raise ValueError("hierarchy provider identity has an unauthenticated shape")
    if identity.get("schema_version") != _HIERARCHY_PROVIDER_SCHEMA_VERSION:
        raise ValueError("hierarchy provider identity uses an unsupported schema")
    provider_id = identity.get("provider_id")
    emitter_id = identity.get("emitter_id")
    if type(provider_id) is not str or type(emitter_id) is not str:
        raise TypeError("hierarchy provider identity requires exact string ids")
    provider = prepared_hierarchy_solver_provider_by_id(provider_id)
    if (
        provider.emitter_id != emitter_id
        or canonical_bytes(_plain_data(identity)) != canonical_bytes(provider.authority())
    ):
        raise ValueError("hierarchy provider identity is inconsistent with its registry authority")
    return provider


def prepared_hierarchy_solver_provider_from_attrs(
    attrs: Mapping[str, Any],
) -> PreparedHierarchySolverProvider:
    if not isinstance(attrs, Mapping):
        raise TypeError("hierarchy solve attributes must be a mapping")
    provider = prepared_hierarchy_solver_provider_from_identity(
        attrs.get("hierarchy_solver_provider")
    )
    options = provider.validate_options(
        attrs.get("hierarchy_solver_options"), where="hierarchy solve options"
    )
    data = {
        "schema_version": _HIERARCHY_INSTANCE_SCHEMA_VERSION,
        "provider": provider.authority(),
        "options": options,
    }
    identity = make_identity("hierarchy-solver", data).token
    if attrs.get("hierarchy_solver_identity") != identity or attrs.get("solver_identity") != identity:
        raise ValueError("hierarchy solver instance identity is inconsistent")
    return provider


def prepared_hierarchy_solver_providers() -> tuple[PreparedHierarchySolverProvider, ...]:
    with _registry_lock:
        return tuple(_providers_by_id.values())


_COMPOSITE_OPTION_NAMES = {
    "max_iter",
    "rel_tol",
    "abs_tol",
    "fine_sweeps",
    "coarse_rel_tol",
    "coarse_abs_tol",
    "coarse_cycles",
    "verbose",
}


def _validate_composite_options(values: Any, where: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        if values == {}:
            values = CompositeTensorFAC().canonical_options()
        else:
            raise TypeError("%s options must be an exact mapping" % where)
    if not values:
        values = CompositeTensorFAC().canonical_options()
    if set(values) != _COMPOSITE_OPTION_NAMES:
        raise TypeError("%s options do not match the provider schema" % where)
    from pops.identity.scalar import exact_cpp_int, scalar_data
    from pops.model._bind_schema_data import literal_value

    maximum = exact_cpp_int(values["max_iter"], where=where + " max_iter", minimum=1)
    fine = values["fine_sweeps"]
    coarse_cycles = values["coarse_cycles"]
    if fine is not None:
        fine = exact_cpp_int(fine, where=where + " fine_sweeps", minimum=1)
    if coarse_cycles is not None:
        coarse_cycles = exact_cpp_int(
            coarse_cycles, where=where + " coarse_cycles", minimum=1
        )
    verbose = values["verbose"]
    if verbose is not None and type(verbose) is not bool:
        raise TypeError("%s verbose must be a Python bool or None" % where)
    relative = literal_value(values["rel_tol"], where=where + " rel_tol")
    absolute = literal_value(values["abs_tol"], where=where + " abs_tol")
    if isinstance(relative, bool) or not 0 < relative < 1:
        raise ValueError("%s rel_tol must be in (0, 1)" % where)
    if isinstance(absolute, bool) or absolute < 0:
        raise ValueError("%s abs_tol must be nonnegative" % where)
    coarse_relative = values["coarse_rel_tol"]
    if coarse_relative is not None:
        coarse_relative = literal_value(coarse_relative, where=where + " coarse_rel_tol")
        if isinstance(coarse_relative, bool) or not 0 < coarse_relative < 1:
            raise ValueError("%s coarse_rel_tol must be in (0, 1) or None" % where)
    coarse_absolute = values["coarse_abs_tol"]
    if coarse_absolute is not None:
        coarse_absolute = literal_value(coarse_absolute, where=where + " coarse_abs_tol")
        if isinstance(coarse_absolute, bool) or coarse_absolute < 0:
            raise ValueError("%s coarse_abs_tol must be nonnegative or None" % where)
    return {
        "max_iter": maximum,
        "rel_tol": scalar_data(relative),
        "abs_tol": scalar_data(absolute),
        "fine_sweeps": fine,
        "coarse_rel_tol": None if coarse_relative is None else scalar_data(coarse_relative),
        "coarse_abs_tol": None if coarse_absolute is None else scalar_data(coarse_absolute),
        "coarse_cycles": coarse_cycles,
        "verbose": verbose,
    }


def _composite_tensor_apply_contract(attrs: Mapping[str, Any]) -> Any:
    """Provider-owned proof that the flat callback and refined native operator are equivalent."""
    from pops.time.values import ProgramValue, _Affine

    if (
        attrs.get("scope") != "hierarchy"
        or attrs.get("domain") != "scalar"
        or attrs.get("range") != "scalar"
        or int(attrs.get("ncomp", 0)) != 1
    ):
        raise ValueError("CompositeTensorFAC requires a scalar hierarchy operator with ncomp=1")
    block = attrs.get("apply_block")
    if not isinstance(block, (list, tuple)):
        raise ValueError("CompositeTensorFAC operator has no authenticated apply")
    ops = sorted(node.op for node in block)
    expected = sorted(("apply_out", "apply_in", "scalar_field", "apply_laplacian_coeff"))
    if ops != expected:
        raise ValueError(
            "CompositeTensorFAC accepts one scalar scratch and one coefficiented Laplacian"
        )
    tensor = next(node for node in block if node.op == "apply_laplacian_coeff")
    scratch = next(node for node in block if node.op == "scalar_field")
    apply_in = attrs.get("apply_in")
    if (
        len(tensor.inputs) != 3
        or tensor.inputs[0].id != scratch.id
        or not isinstance(apply_in, ProgramValue)
        or tensor.inputs[1].id != apply_in.id
    ):
        raise ValueError("CompositeTensorFAC tensor apply must read its exact apply input")
    if int(scratch.attrs.get("ncomp", 1)) != 1:
        raise ValueError("CompositeTensorFAC tensor apply scratch must have ncomp=1")
    coefficients = tensor.inputs[2]
    if not (
        isinstance(coefficients, ProgramValue)
        and coefficients.vtype == "condensed_coeffs"
        and coefficients.block is not None
    ):
        raise ValueError("CompositeTensorFAC requires owner-qualified condensed coefficients")
    result = attrs.get("apply_result")
    if not isinstance(result, _Affine):
        raise ValueError("CompositeTensorFAC apply must return exactly the negative tensor stencil")
    terms = result._merge()
    if (
        len(terms) != 1
        or terms[0][0].id != tensor.id
        or terms[0][1].as_dict() != {0: -1}
    ):
        raise ValueError("CompositeTensorFAC apply must return exactly the negative tensor stencil")
    return coefficients


def _author_composite_tensor_fac(
    program: Any,
    problem: Any,
    prepared: PreparedHierarchySolver,
    name: Any,
    provider: PreparedHierarchySolverProvider,
) -> Any:
    from pops.identity.scalar import scalar_literal
    from pops.fields._prepared_nullspace_registry import (
        PreparedNullspaceContracts,
        prepared_nullspace_provider_from_identity,
    )
    from pops.linalg import LinearProblem
    from pops.solvers.scopes import solve_scope_id
    from pops.time.solve_outcome import SolveOutcome
    from pops.time.value_metadata import positive_scalar_literal
    from pops.time.values import ProgramValue, _resolve_handle

    if not isinstance(problem, LinearProblem):
        raise TypeError("CompositeTensorFAC requires a pops.linalg.LinearProblem")
    options = provider.authenticate_prepared(prepared)
    nullspace_identity = problem.canonical_nullspace_provider()
    nullspace_contract = problem.canonical_nullspace_contract()
    gauge_contract = problem.canonical_gauge_contract()
    nullspace = prepared_nullspace_provider_from_identity(nullspace_identity)
    contracts = PreparedNullspaceContracts(nullspace_contract["contract"], gauge_contract)
    nullspace.validate_use(
        contracts=contracts,
        components=1,
        operator_properties=problem.properties.canonical_data(),
        where="hierarchy nullspace provider %r" % nullspace.provider_id,
    )
    operator = program._canonical_value(_resolve_handle(problem.operator))
    rhs = program._canonical_value(_resolve_handle(problem.rhs))
    initial_guess = (
        None
        if problem.initial_guess is None
        else program._canonical_value(_resolve_handle(problem.initial_guess))
    )
    if not isinstance(operator, ProgramValue) or operator.vtype != "matrix_free_op":
        raise ValueError("hierarchy provider requires a matrix_free_operator")
    scope = operator.attrs.get("scope", "level") if problem.scope is None else solve_scope_id(
        problem.scope
    )
    coefficients = _composite_tensor_apply_contract(operator.attrs)
    problem_kind = "scalar_tensor_elliptic_hierarchy"
    hierarchy_use_facts: dict[str, Any] = {}
    provider.use_policy.validate(
        PreparedHierarchySolverUseFacts(
            target=None,
            scope=scope,
            problem_kind=problem_kind,
            domain=operator.attrs.get("domain"),
            range=operator.attrs.get("range"),
            components=operator.attrs.get("ncomp"),
            singular_nullspace=nullspace.singular,
            extensions=hierarchy_use_facts,
        ),
        operator=operator,
        where="hierarchy provider %r" % provider.provider_id,
    )
    owner = coefficients.block
    if not (
        isinstance(rhs, ProgramValue)
        and rhs.vtype == "scalar_field"
        and rhs.op == "condensed_rhs"
    ):
        raise ValueError("CompositeTensorFAC rhs must be an owner-qualified condensed_rhs")
    if rhs.block != owner:
        raise ValueError("CompositeTensorFAC coefficients and rhs must share one owner")
    rhs_storage = rhs.inputs[0]
    if int(rhs_storage.attrs.get("ncomp", 1)) != 1:
        raise ValueError("CompositeTensorFAC supports exactly one scalar component")
    coefficient_state = coefficients.inputs[0]
    rhs_state = rhs.inputs[2]
    if coefficient_state.id != rhs_state.id:
        raise ValueError("CompositeTensorFAC coefficients and rhs must use the same State")
    for key in ("linear_operator", "subset"):
        if coefficients.attrs.get(key) != rhs.attrs.get(key):
            raise ValueError("CompositeTensorFAC coefficients and rhs disagree on %s" % key)
    if initial_guess is not None:
        if not isinstance(initial_guess, ProgramValue) or initial_guess.vtype != "scalar_field":
            raise ValueError("CompositeTensorFAC initial_guess must be a scalar field")
        if initial_guess.block != owner or int(initial_guess.attrs.get("ncomp", 1)) != 1:
            raise ValueError("CompositeTensorFAC initial_guess has an incompatible owner/layout")
    block_indices = program._block_indices()
    if owner not in block_indices:
        raise ValueError("hierarchy provider owner has no installed Program state")
    block_index = int(block_indices[owner])
    relative, absolute, maximum = provider.convergence.values(
        options, where="hierarchy provider"
    )
    tolerance = positive_scalar_literal(relative, where="hierarchy provider relative tolerance")
    absolute_tolerance = scalar_literal(absolute)
    inputs = (operator, rhs) if initial_guess is None else (operator, rhs, initial_guess)
    flat_execution = provider.flat_execution.ir_attributes(
        components=1,
        input_ghosts=operator.attrs["stencil_access"].required_ghost_depth,
        nullspace_contract=nullspace_contract,
        operator_properties=problem.properties.canonical_data(),
        declared_nullspace=nullspace.singular,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
        max_iterations=maximum,
    )
    attrs = {
        **flat_execution,
        "tol": tolerance,
        "abs_tol": absolute_tolerance,
        "max_iter": maximum,
        "has_guess": initial_guess is not None,
        "ncomp": 1,
        "operator_properties": problem.properties.canonical_data(),
        "nullspace_provider": nullspace_identity,
        "nullspace_contract": nullspace_contract,
        "gauge_contract": gauge_contract,
        "scope": scope,
        "hierarchy_solver_provider": provider.authority(),
        "hierarchy_solver_options": deepcopy(options),
        "hierarchy_solver_identity": prepared.identity.token,
        "hierarchy_block_index": block_index,
        "hierarchy_tensor_coefficients": coefficients.id,
        "solver_identity": prepared.identity.token,
        "problem_kind": problem_kind,
        "hierarchy_use_facts": hierarchy_use_facts,
    }
    token = program._new(
        "scalar_field",
        "solve_linear",
        inputs,
        attrs,
        name,
        owner,
        space=rhs.space,
        point=rhs.point if problem.at is None else problem.at,
    )
    outcome_name = name or token.name

    def project(outcome: Any) -> Any:
        return program._new(
            "scalar_field",
            "solve_outcome_component",
            (outcome,),
            {"index": 0, "ncomp": 1},
            outcome_name,
            owner,
            space=rhs.space,
            point=token.point,
        )

    return SolveOutcome(program, token, project, outcome_name)


def _emit_composite_tensor_fac(
    request: PreparedHierarchySolverEmitRequest,
    provider: PreparedHierarchySolverProvider,
    options: Mapping[str, Any],
) -> PreparedHierarchySolverNativeEmission:
    from pops.identity.scalar import scalar_cpp
    from pops.model._bind_schema_data import literal_value

    fine = options["fine_sweeps"]
    coarse_relative = options["coarse_rel_tol"]
    coarse_absolute = options["coarse_abs_tol"]
    coarse_cycles = options["coarse_cycles"]
    verbose = options["verbose"]
    native_options: list[str] = []
    if fine is not None:
        native_options.append(
            '{"fac.fine_sweeps", std::int64_t{%d}}' % fine
        )
    if coarse_relative is not None:
        value = literal_value(
            coarse_relative, where="hierarchy coarse relative tolerance"
        )
        native_options.append(
            '{"fac.coarse_rel_tol", static_cast<double>(%s)}' % scalar_cpp(value)
        )
    if coarse_absolute is not None:
        value = literal_value(
            coarse_absolute, where="hierarchy coarse absolute tolerance"
        )
        native_options.append(
            '{"fac.coarse_abs_tol", static_cast<double>(%s)}' % scalar_cpp(value)
        )
    if coarse_cycles is not None:
        native_options.append(
            '{"fac.coarse_cycles", std::int64_t{%d}}' % coarse_cycles
        )
    if verbose is not None:
        native_options.append(
            '{"fac.verbose", %s}' % ("true" if verbose else "false")
        )
    option_map = "{" + ", ".join(native_options) + "}"
    plan_identity = request.node.attrs["hierarchy_solver_identity"]
    operator_contract = "pops.operator.scalar-tensor-elliptic-2d@1"
    assembly_slots = (
        "pops.tensor-elliptic.diagonal.x",
        "pops.tensor-elliptic.diagonal.y",
        "pops.tensor-elliptic.cross.xy",
        "pops.tensor-elliptic.cross.yx",
        "pops.tensor-elliptic.rhs",
        "pops.tensor-elliptic.flux",
    )
    assembly_slot_cpp = "std::vector<std::string>{%s}" % ", ".join(
        json.dumps(slot) for slot in assembly_slots
    )
    configure = (
        "ctx.configure_hierarchy_tensor_solver(%d, %d, %s, %s, %s, %s, %s, "
        "pops::PreparedProviderOptions{%s, %s});"
        % (
            request.block_index,
            request.components,
            json.dumps(provider.provider_id),
            json.dumps(plan_identity),
            json.dumps(operator_contract),
            assembly_slot_cpp,
            json.dumps("pops.tensor-elliptic.solution"),
            json.dumps(provider.option_schema),
            option_map,
        ),
    )
    solve = (
        "pops::SolveReport %s = ctx.solve_hierarchy_tensor(%d, %d, %s, %s, %d);"
        % (
            request.report_name,
            request.block_index,
            request.components,
            request.relative_tolerance_cpp,
            request.absolute_tolerance_cpp,
            request.max_iterations,
        ),
    )
    return PreparedHierarchySolverNativeEmission(configure=configure, solve=solve)


def _validate_composite_tensor_fac_use(
    facts: PreparedHierarchySolverUseFacts, operator: Any, where: str
) -> PreparedHierarchySolverUseFacts:
    """Builtin policy implementation; the registry core knows none of these restrictions."""
    if facts.target not in (None, "amr_system"):
        raise ValueError("%s target %r is unsupported" % (where, facts.target))
    if facts.scope != "hierarchy":
        raise ValueError("%s scope %r is unsupported" % (where, facts.scope))
    if facts.problem_kind != "scalar_tensor_elliptic_hierarchy":
        raise ValueError("%s problem kind %r is unsupported" % (where, facts.problem_kind))
    if facts.domain != "scalar" or facts.range != facts.domain:
        raise ValueError("%s requires an authenticated square scalar domain" % where)
    if facts.components != 1:
        raise ValueError("%s component count %r is unsupported" % (where, facts.components))
    if facts.singular_nullspace:
        raise NotImplementedError("%s does not support a singular nullspace contract" % where)
    if facts.extensions:
        raise ValueError("%s received unsupported provider use facts" % where)
    operator_attrs = getattr(operator, "attrs", None)
    if not isinstance(operator_attrs, Mapping):
        raise TypeError("%s requires an authenticated operator record" % where)
    _composite_tensor_apply_contract(operator_attrs)
    return facts


_COMPOSITE_PROVIDER = register_prepared_hierarchy_solver_provider(
    PreparedHierarchySolverProvider(
        provider_id="pops.hierarchy.composite-tensor-fac",
        interface_version=1,
        emitter_id="pops.codegen.hierarchy.composite-tensor-fac@1",
        option_schema="pops.hierarchy.composite-tensor-fac.options@1",
        capabilities=frozenset(
            {
                "pops.hierarchy.composite-tensor-fac.flat-krylov@1",
                "pops.hierarchy.composite-tensor-fac.refined-direct@1",
                "pops.hierarchy.composite-tensor-fac.mixed-level-distribution@1",
                "pops.hierarchy.composite-tensor-fac.exact-preparation@1",
            }
        ),
        use_policy=PreparedHierarchySolverUsePolicy(
            policy_id="pops.use-policy.scalar-tensor-elliptic-amr",
            interface_version=1,
            capabilities=frozenset(
                {
                    "pops.hierarchy.use-facts.common@1",
                    "pops.operator.scalar-tensor-elliptic-2d@1",
                    "pops.target.amr-system@1",
                }
            ),
            validator=_validate_composite_tensor_fac_use,
        ),
        convergence=PreparedHierarchyConvergenceContract(
            "rel_tol", "abs_tol", "max_iter"
        ),
        flat_execution=PreparedHierarchyFlatExecution.prepared_krylov(
            PreparedHierarchyKrylovFallback(
                "pops.krylov.bicgstab",
                "pops.preconditioner.identity",
            )
        ),
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.hierarchy.composite-tensor-fac",
            entry_headers=(
                "pops/runtime/amr/amr_tensor_elliptic.hpp",
                "pops/runtime/program/amr_program_context.hpp",
            ),
        ),
        option_validator=_validate_composite_options,
        author=_author_composite_tensor_fac,
        emitter=_emit_composite_tensor_fac,
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeTensorFAC:
    """Builtin scalar tensor-elliptic provider over one AMR hierarchy."""

    max_iter: int = _DEFAULT_MAX_ITER
    rel_tol: Any = _DEFAULT_REL_TOL
    abs_tol: Any = 0.0
    fine_sweeps: int | None = None
    coarse_rel_tol: Any = None
    coarse_abs_tol: Any = None
    coarse_cycles: int | None = None
    verbose: bool | None = None
    solver_id: str = field(init=False, default="composite_tensor_fac")
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_iter", _positive_max_iter(self.max_iter))
        object.__setattr__(
            self,
            "rel_tol",
            exact_open_unit_real(self.rel_tol, where="CompositeTensorFAC(rel_tol=)"),
        )
        object.__setattr__(
            self,
            "abs_tol",
            exact_nonnegative_real(self.abs_tol, where="CompositeTensorFAC(abs_tol=)"),
        )
        object.__setattr__(
            self,
            "fine_sweeps",
            optional_positive_int(self.fine_sweeps, where="CompositeTensorFAC(fine_sweeps=)"),
        )
        object.__setattr__(
            self,
            "coarse_cycles",
            optional_positive_int(self.coarse_cycles, where="CompositeTensorFAC(coarse_cycles=)"),
        )
        if self.coarse_rel_tol is not None:
            object.__setattr__(
                self,
                "coarse_rel_tol",
                exact_open_unit_real(
                    self.coarse_rel_tol, where="CompositeTensorFAC(coarse_rel_tol=)"
                ),
            )
        if self.coarse_abs_tol is not None:
            object.__setattr__(
                self,
                "coarse_abs_tol",
                exact_nonnegative_real(
                    self.coarse_abs_tol, where="CompositeTensorFAC(coarse_abs_tol=)"
                ),
            )
        if self.verbose is not None and type(self.verbose) is not bool:
            raise TypeError("CompositeTensorFAC(verbose=) must be a Python bool or None")

    @property
    def capabilities(self) -> frozenset[str]:
        return _COMPOSITE_PROVIDER.capabilities

    def canonical_options(self) -> dict[str, Any]:
        from pops.identity.scalar import scalar_data

        return {
            "max_iter": self.max_iter,
            "rel_tol": scalar_data(self.rel_tol),
            "abs_tol": scalar_data(self.abs_tol),
            "fine_sweeps": self.fine_sweeps,
            "coarse_rel_tol": (
                None if self.coarse_rel_tol is None else scalar_data(self.coarse_rel_tol)
            ),
            "coarse_abs_tol": (
                None if self.coarse_abs_tol is None else scalar_data(self.coarse_abs_tol)
            ),
            "coarse_cycles": self.coarse_cycles,
            "verbose": self.verbose,
        }

    def canonical_identity(self) -> dict[str, Any]:
        return _COMPOSITE_PROVIDER.instance_data(self.canonical_options())

    def to_data(self) -> dict[str, Any]:
        return self.canonical_identity()

    @property
    def identity(self) -> Identity:
        return make_identity("hierarchy-solver", self.canonical_identity())

    def prepare_program_solve(self) -> PreparedHierarchySolver:
        return _COMPOSITE_PROVIDER.prepare(self.canonical_options())


__all__ = [
    "CompositeTensorFAC",
    "PreparedHierarchyConvergenceContract",
    "PreparedHierarchyFlatExecution",
    "PreparedHierarchyKrylovFallback",
    "PreparedHierarchySolver",
    "PreparedHierarchySolverEmitRequest",
    "PreparedHierarchySolverNativeEmission",
    "PreparedHierarchySolverProvider",
    "PreparedHierarchySolverUseFacts",
    "PreparedHierarchySolverUsePolicy",
    "prepared_hierarchy_solver_provider_by_id",
    "prepared_hierarchy_solver_provider_from_attrs",
    "prepared_hierarchy_solver_provider_from_identity",
    "prepared_hierarchy_solver_providers",
    "register_prepared_hierarchy_solver_provider",
]
