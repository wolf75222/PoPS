"""Typed external AMR provider bindings.

These values select one exact native component at authoring time.  They retain no
Python callback: resolve authenticates the component and its graph-evaluator capabilities,
then compile carries only the immutable native binding record to installation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from pops.identity import make_identity
from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI


_TAGGER_LEAF_OPCODES = dict(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"])
_TAGGER_LOGICAL_OPCODES = dict(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])
_TAGGER_OUTPUTS = tuple(NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"])


@dataclass(frozen=True, slots=True)
class AMRProviderLoweringContext:
    """Resolved facts available to every open AMR provider authority.

    Providers decide which facts they consume.  In particular, a tagger authenticates the
    resolved tagging graph while a clustering provider remains independent of it; the resolver
    never needs to name either concrete implementation.
    """

    layout_identity: str
    components: Any
    tagging_graph: Any
    clock_identity: str

    def __post_init__(self) -> None:
        for name in ("layout_identity", "clock_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError("AMR provider lowering context requires a non-empty %s" % name)
        if not isinstance(getattr(self.tagging_graph, "qualified_id", None), str):
            raise TypeError("AMR provider lowering context requires one resolved tagging graph")


@dataclass(frozen=True, slots=True)
class ResolvedAMRProviderBinding:
    """One role-qualified, data-only provider binding returned by an open authority."""

    role: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise TypeError("resolved AMR provider binding requires a non-empty role")
        if type(self.data) is not dict:
            raise TypeError("resolved AMR provider binding data must be an exact dict")
        installation = self.data.get("runtime_installation")
        if self.data.get("schema_version") != 1 \
                or not isinstance(self.data.get("provider_identity"), str) \
                or not self.data["provider_identity"] \
                or not isinstance(self.data.get("layout_identity"), str) \
                or not self.data["layout_identity"] \
                or not isinstance(self.data.get("native_interface"), Mapping) \
                or not isinstance(installation, Mapping) \
                or set(installation) != {"schema_version", "protocol"} \
                or installation.get("schema_version") != 1 \
                or installation.get("protocol") not in {
                    "builtin", "external_component"
                }:
            raise TypeError(
                "resolved AMR provider binding is incomplete or lacks its runtime protocol"
            )
        object.__setattr__(
            self,
            "data",
            validate_amr_provider_binding(
                role=self.role,
                frozen_binding=self.data,
                layout_identity=self.data["layout_identity"],
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedAMRProviderInstallation:
    """Validated runtime job; ``installer`` is absent for an intrinsic builtin provider."""

    role: str
    binding: dict[str, Any]
    installer: Callable[..., Any] | None = None
    native_handle: Any = None


@dataclass(frozen=True, slots=True)
class PreparedAMRProviderNativeConfig:
    """Provider-owned bind-time config plus options retained for component preparation."""

    role: str
    config: dict[str, Any]
    provider_options: dict[str, Any]


def _external_component(value: Any, *, interface: Any, where: str) -> Any:
    from pops.external import ExternalComponent

    if type(value) is not ExternalComponent:
        raise TypeError("%s requires an exact pops.external.ExternalComponent" % where)
    actual = value.component_type.interface
    if actual != interface:
        raise TypeError(
            "%s requires exact interface %s@%d, got %s@%d"
            % (where, interface.uri, interface.version, actual.uri, actual.version)
        )
    interface.require_manifest(value.component_manifest)
    interface.resolve_native_target(value)
    determinism = value.component_manifest.determinism.get("classification")
    if determinism not in {"bitwise", "reproducible"}:
        raise ValueError(
            "%s requires determinism.classification 'bitwise' or 'reproducible'; "
            "every rank must derive the same hierarchy" % where
        )
    return value


def _require_component(component: Any, values: Any, *, where: str) -> None:
    from pops.external import ExternalComponent

    rows = tuple(values)
    matches = [
        value for value in rows
        if type(value) is ExternalComponent
        and value.component_manifest.component_id
        == component.component_manifest.component_id
    ]
    if len(matches) != 1 or matches[0].to_data() != component.to_data():
        raise ValueError(
            "%s requires its exact ExternalComponent in pops.resolve(..., components=...)"
            % where
        )


def _component_binding(component: Any, interface: Any) -> dict[str, Any]:
    return {
        "component_id": component.component_manifest.component_id,
        "component_manifest_identity": component.component_manifest.manifest_digest.token,
        "component": component.to_data(),
        "native_interface": interface.to_data(),
        "interface_version": interface.version,
    }


def _normalize_tagger_capability(capabilities: Any) -> dict[str, Any]:
    matches = []
    for row in capabilities:
        if isinstance(row, dict) and row.get("capability_type") == "amr_tagging_program":
            matches.append(row)
        elif hasattr(row, "get") and row.get("capability_type") == "amr_tagging_program":
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "TaggerProvider.component must declare exactly one amr_tagging_program capability")
    row = matches[0]
    expected = {
        "schema_version", "capability_type", "leaf_opcodes", "logical_opcodes",
        "candidate_outputs", "indicator_stencil_routes", "maximum_stencil_terms",
        "maximum_instruction_count", "non_finite_policy", "persistent_hysteresis",
        "execution_mode", "collective_scope", "memory_spaces",
    }
    if set(row) != expected or row.get("schema_version") != 1:
        raise ValueError("AMR Tagger capability has an unsupported schema")
    leaves = tuple(row["leaf_opcodes"])
    logical = tuple(row["logical_opcodes"])
    outputs = tuple(row["candidate_outputs"])
    maximum = row["maximum_instruction_count"]
    stencil_routes = tuple(row["indicator_stencil_routes"])
    maximum_stencil_terms = row["maximum_stencil_terms"]
    non_finite_policy = row["non_finite_policy"]
    execution_mode = row["execution_mode"]
    collective_scope = row["collective_scope"]
    memory_spaces = tuple(row["memory_spaces"])
    if not leaves or len(set(leaves)) != len(leaves) \
            or any(value not in _TAGGER_LEAF_OPCODES for value in leaves):
        raise ValueError("AMR Tagger capability declares invalid leaf opcodes")
    if not logical or len(set(logical)) != len(logical) \
            or any(value not in _TAGGER_LOGICAL_OPCODES for value in logical):
        raise ValueError("AMR Tagger capability declares invalid logical opcodes")
    if outputs != _TAGGER_OUTPUTS:
        raise ValueError(
            "AMR Tagger must return exact refine/coarsen candidate and equality masks")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("AMR Tagger maximum_instruction_count must be an integer >= 1")
    known_routes = tuple(NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"])
    if not stencil_routes or len(set(stencil_routes)) != len(stencil_routes) \
            or any(route not in known_routes for route in stencil_routes):
        raise ValueError("AMR Tagger declares invalid indicator_stencil_routes")
    if isinstance(maximum_stencil_terms, bool) \
            or not isinstance(maximum_stencil_terms, int) \
            or maximum_stencil_terms < 1 \
            or maximum_stencil_terms > NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"]:
        raise ValueError("AMR Tagger maximum_stencil_terms is outside the native ABI")
    if non_finite_policy != NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"]:
        raise ValueError("AMR Tagger must reject every non-finite indicator sample")
    if type(row["persistent_hysteresis"]) is not bool:
        raise TypeError("AMR Tagger persistent_hysteresis capability must be Boolean")
    if execution_mode not in NATIVE_TAGGING_PROGRAM_ABI["execution_modes"]:
        raise ValueError("AMR Tagger declares an invalid execution_mode")
    if collective_scope not in NATIVE_TAGGING_PROGRAM_ABI["collective_scopes"] \
            or collective_scope != "none":
        raise ValueError("AMR Tagger callbacks must be explicitly noncollective")
    known_memory_spaces = tuple(NATIVE_TAGGING_PROGRAM_ABI["memory_spaces"])
    if not memory_spaces or len(set(memory_spaces)) != len(memory_spaces) \
            or any(value not in known_memory_spaces for value in memory_spaces):
        raise ValueError("AMR Tagger declares invalid memory_spaces")
    if execution_mode == "host" and memory_spaces != ("host",):
        raise ValueError(
            "host AMR Tagger execution must declare exactly the host memory space")
    return {
        "schema_version": 1,
        "capability_type": "amr_tagging_program",
        "leaf_opcodes": list(leaves),
        "leaf_opcode_ids": [_TAGGER_LEAF_OPCODES[value] for value in leaves],
        "logical_opcodes": list(logical),
        "logical_opcode_ids": [_TAGGER_LOGICAL_OPCODES[value] for value in logical],
        "candidate_outputs": list(outputs),
        "indicator_stencil_routes": list(stencil_routes),
        "maximum_stencil_terms": maximum_stencil_terms,
        "maximum_instruction_count": maximum,
        "non_finite_policy": non_finite_policy,
        "persistent_hysteresis": row["persistent_hysteresis"],
        "execution_mode": execution_mode,
        "collective_scope": collective_scope,
        "memory_spaces": list(memory_spaces),
    }


def _tagger_capability(component: Any) -> dict[str, Any]:
    return _normalize_tagger_capability(component.component_manifest.capabilities)


def _require_tagger_target_execution(component: Any, capability: Mapping[str, Any]) -> None:
    from pops import interfaces

    target = interfaces.Tagger.resolve_native_target(component)
    if capability["execution_mode"] != "native_backend":
        return
    required_space = "host" if target["device"] == "cpu" else "managed"
    if required_space not in capability["memory_spaces"]:
        raise ValueError(
            "native-backend AMR Tagger target %r requires %r field memory"
            % (target["device"], required_space)
        )


@dataclass(frozen=True, slots=True)
class TaggerProvider:
    """Bind one external evaluator to the resolved AMRTagging graph."""

    component: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops import interfaces

        _external_component(
            self.component, interface=interfaces.Tagger, where="TaggerProvider.component")
        capability = _tagger_capability(self.component)
        _require_tagger_target_execution(self.component, capability)

    def resolve_references(self, resolver: Any) -> TaggerProvider:
        if not callable(resolver):
            raise TypeError("TaggerProvider.resolve_references requires a callable resolver")
        return self

    def require_component_inputs(self, components: Any) -> None:
        _require_component(self.component, components, where="TaggerProvider")

    def require_tagging_graph(self, graph: Any) -> None:
        capability = _tagger_capability(self.component)
        if capability["persistent_hysteresis"] is not NATIVE_TAGGING_PROGRAM_ABI[
                "persistent_hysteresis"]:
            raise NotImplementedError(
                "AMR Tagger persistent_hysteresis is not implemented by the native adapter")
        registrations = getattr(graph, "registrations", None)
        authoring = getattr(graph, "graph", None)
        if not isinstance(registrations, tuple) or authoring is None:
            raise TypeError("TaggerProvider requires one resolved AMRTagging graph")
        used = {row.node_type for row in registrations}
        supported = set(capability["leaf_opcodes"]) | set(capability["logical_opcodes"])
        missing = sorted(used - supported)
        if missing:
            raise NotImplementedError(
                "external AMR Tagger lacks resolved opcode(s): %s" % ", ".join(missing))

        def count(node: Any) -> int:
            operands = node.operands()
            return 1 + sum(count(child) for child in operands)

        instruction_count = count(authoring.refine)
        if authoring.coarsen is not None:
            instruction_count += count(authoring.coarsen)
        if instruction_count > capability["maximum_instruction_count"]:
            raise NotImplementedError(
                "external AMR Tagger graph exceeds maximum_instruction_count")
        def require_stencils(node: Any) -> None:
            if getattr(node, "node_type", None) in {"gradient_above", "gradient_below"}:
                from pops.numerics.indicator_stencils import DiscreteGradientStencil

                lowering = getattr(getattr(node, "context", None), "lowering", None)
                if type(lowering) is not DiscreteGradientStencil:
                    raise TypeError("resolved AMR gradient has no typed stencil lowering")
                if lowering.route not in capability["indicator_stencil_routes"]:
                    raise NotImplementedError(
                        "external AMR Tagger lacks indicator stencil route %r" % lowering.route)
                if any(len(axis.offsets) > capability["maximum_stencil_terms"]
                       for axis in lowering.axes):
                    raise NotImplementedError(
                        "external AMR Tagger stencil exceeds maximum_stencil_terms")
            for child in node.operands():
                require_stencils(child)

        require_stencils(authoring.refine)
        if authoring.coarsen is not None:
            require_stencils(authoring.coarsen)
        if authoring.hysteresis.min_cycles != 0:
            raise NotImplementedError(
                "AMR hysteresis min_cycles requires native persistent tagging state; "
                "it is never accepted then ignored")

    def lower_amr_provider(
        self, context: AMRProviderLoweringContext,
    ) -> ResolvedAMRProviderBinding:
        """Authenticate component, graph and clock before emitting detached runtime data."""
        if type(context) is not AMRProviderLoweringContext:
            raise TypeError("TaggerProvider requires an AMRProviderLoweringContext")
        self.require_component_inputs(context.components)
        self.require_tagging_graph(context.tagging_graph)
        data = {
            **self.runtime_binding_data(),
            "layout_identity": context.layout_identity,
            "clock_identity": context.clock_identity,
            "tagging_graph_identity": context.tagging_graph.qualified_id,
        }
        data["provider_identity"] = amr_provider_binding_identity("tagger", data)
        return ResolvedAMRProviderBinding("tagger", data)

    def runtime_binding_data(self) -> dict[str, Any]:
        from pops import interfaces

        data = {
            "schema_version": 1,
            "provider_type": "external_amr_tagger",
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "external_component",
            },
            **_component_binding(self.component, interfaces.Tagger),
            "tagging_capability": _tagger_capability(self.component),
        }
        data["provider_identity"] = make_identity("amr-tagger-provider", data).token
        return data

    def inspect(self) -> dict[str, Any]:
        return self.runtime_binding_data()

    canonical_identity = inspect


@dataclass(frozen=True, slots=True)
class ClusteringProvider:
    """Bind one external Clustering table to the AMR hierarchy authority."""

    component: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops import interfaces

        _external_component(
            self.component,
            interface=interfaces.Clustering,
            where="ClusteringProvider.component",
        )

    def resolve_references(self, resolver: Any) -> ClusteringProvider:
        if not callable(resolver):
            raise TypeError("ClusteringProvider.resolve_references requires a callable resolver")
        return self

    def require_component_inputs(self, components: Any) -> None:
        _require_component(self.component, components, where="ClusteringProvider")

    def lower_amr_provider(
        self, context: AMRProviderLoweringContext,
    ) -> ResolvedAMRProviderBinding:
        """Authenticate the component and emit its role-qualified detached binding."""
        if type(context) is not AMRProviderLoweringContext:
            raise TypeError("ClusteringProvider requires an AMRProviderLoweringContext")
        self.require_component_inputs(context.components)
        data = {
            **self.runtime_binding_data(),
            "layout_identity": context.layout_identity,
        }
        data["provider_identity"] = amr_provider_binding_identity("clustering", data)
        return ResolvedAMRProviderBinding("clustering", data)

    def runtime_binding_data(self) -> dict[str, Any]:
        from pops import interfaces

        data = {
            "schema_version": 1,
            "provider_type": "external_amr_clustering",
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "external_component",
            },
            **_component_binding(self.component, interfaces.Clustering),
        }
        data["provider_identity"] = make_identity("amr-clustering-provider", data).token
        return data

    inspect = runtime_binding_data
    canonical_identity = runtime_binding_data


@dataclass(frozen=True, slots=True)
class _AMRRuntimeInterfaceProtocol:
    """Native-interface-owned validation and installation route."""

    role: str
    native_interface: Mapping[str, Any]
    builtin_provider_id: str
    component_installer: str

    @property
    def resolved_identity_namespace(self) -> str:
        return "resolved-amr-%s-provider" % self.role

    def validate_resolved_capability(
        self, binding: Mapping[str, Any], resolved_tagging_identity: str | None,
    ) -> None:
        del binding, resolved_tagging_identity

    def validate_installed_capability(
        self, binding: Mapping[str, Any], installed: Any,
        resolved_tagging_identity: str | None,
    ) -> None:
        del binding, installed, resolved_tagging_identity

    def builtin_native_config(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        return {}


@dataclass(frozen=True, slots=True)
class _ClusteringRuntimeInterfaceProtocol(_AMRRuntimeInterfaceProtocol):
    """Clustering controls that belong to the builtin native implementation."""

    def validate_resolved_capability(
        self, binding: Mapping[str, Any], resolved_tagging_identity: str | None,
    ) -> None:
        del resolved_tagging_identity
        installation = binding.get("runtime_installation")
        if not isinstance(installation, Mapping) \
                or installation.get("protocol") != "builtin":
            return
        efficiency = binding.get("minimum_efficiency")
        minimum = binding.get("minimum_box_size")
        maximum = binding.get("maximum_box_size")
        if isinstance(efficiency, Mapping) and set(efficiency) == {"binary64"} \
                and isinstance(efficiency.get("binary64"), str):
            try:
                decoded_efficiency = float.fromhex(efficiency["binary64"])
            except ValueError:
                decoded_efficiency = None
            if decoded_efficiency is not None \
                    and decoded_efficiency.hex() != efficiency["binary64"]:
                decoded_efficiency = None
        elif not isinstance(efficiency, bool) and isinstance(efficiency, (int, float)):
            decoded_efficiency = float(efficiency)
        else:
            decoded_efficiency = None
        if decoded_efficiency is None or not isfinite(decoded_efficiency) \
                or not 0.0 < decoded_efficiency <= 1.0:
            raise ValueError("builtin AMR clustering minimum_efficiency must be in (0, 1]")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1 \
                or isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1 \
                or minimum > maximum:
            raise ValueError("builtin AMR clustering box-size controls are invalid")

    def builtin_native_config(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        required = ("minimum_efficiency", "minimum_box_size", "maximum_box_size")
        if any(name not in binding for name in required):
            raise TypeError("builtin AMR clustering binding lacks exact algorithm controls")
        return {
            "cluster_min_efficiency": binding["minimum_efficiency"],
            "cluster_min_box_size": binding["minimum_box_size"],
            "cluster_max_box_size": binding["maximum_box_size"],
        }


@dataclass(frozen=True, slots=True)
class _TaggerRuntimeInterfaceProtocol(_AMRRuntimeInterfaceProtocol):
    """Candidate-program capability owned by the Tagger native interface."""

    def validate_resolved_capability(
        self, binding: Mapping[str, Any], resolved_tagging_identity: str | None,
    ) -> None:
        capability = binding.get("tagging_capability")
        expected_keys = {
            "schema_version", "capability_type", "leaf_opcodes", "leaf_opcode_ids",
            "logical_opcodes", "logical_opcode_ids", "candidate_outputs",
            "indicator_stencil_routes", "maximum_stencil_terms",
            "maximum_instruction_count", "non_finite_policy", "persistent_hysteresis",
            "execution_mode", "collective_scope", "memory_spaces",
        }
        leaves = tuple(capability.get("leaf_opcodes", ())) \
            if isinstance(capability, Mapping) else ()
        leaf_ids = tuple(capability.get("leaf_opcode_ids", ())) \
            if isinstance(capability, Mapping) else ()
        logical = tuple(capability.get("logical_opcodes", ())) \
            if isinstance(capability, Mapping) else ()
        logical_ids = tuple(capability.get("logical_opcode_ids", ())) \
            if isinstance(capability, Mapping) else ()
        maximum_instructions = capability.get("maximum_instruction_count") \
            if isinstance(capability, Mapping) else None
        maximum_stencil_terms = (
            capability.get("maximum_stencil_terms")
            if isinstance(capability, Mapping)
            else None
        )
        execution_mode = capability.get("execution_mode") \
            if isinstance(capability, Mapping) else None
        collective_scope = capability.get("collective_scope") \
            if isinstance(capability, Mapping) else None
        memory_spaces = tuple(capability.get("memory_spaces", ())) \
            if isinstance(capability, Mapping) else ()
        if not isinstance(capability, Mapping) or set(capability) != expected_keys \
                or capability.get("schema_version") != 1 \
                or capability.get("capability_type") != "amr_tagging_program" \
                or not leaves or len(set(leaves)) != len(leaves) \
                or any(value not in _TAGGER_LEAF_OPCODES for value in leaves) \
                or leaf_ids != tuple(_TAGGER_LEAF_OPCODES[value] for value in leaves) \
                or not logical or len(set(logical)) != len(logical) \
                or any(value not in _TAGGER_LOGICAL_OPCODES for value in logical) \
                or logical_ids != tuple(_TAGGER_LOGICAL_OPCODES[value] for value in logical) \
                or tuple(capability.get("candidate_outputs", ())) != _TAGGER_OUTPUTS \
                or not set(capability.get("indicator_stencil_routes", ())) <= set(
                    NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]) \
                or not capability.get("indicator_stencil_routes") \
                or isinstance(maximum_instructions, bool) \
                or not isinstance(maximum_instructions, int) \
                or maximum_instructions < 1 \
                or maximum_instructions > NATIVE_TAGGING_PROGRAM_ABI[
                    "maximum_instruction_count"] \
                or isinstance(maximum_stencil_terms, bool) \
                or not isinstance(maximum_stencil_terms, int) \
                or maximum_stencil_terms < 1 \
                or maximum_stencil_terms > NATIVE_TAGGING_PROGRAM_ABI[
                    "maximum_stencil_terms"] \
                or capability.get("non_finite_policy") != NATIVE_TAGGING_PROGRAM_ABI[
                    "non_finite_policy"] \
                or capability.get("persistent_hysteresis") is not NATIVE_TAGGING_PROGRAM_ABI[
                    "persistent_hysteresis"] \
                or execution_mode not in NATIVE_TAGGING_PROGRAM_ABI["execution_modes"] \
                or collective_scope not in NATIVE_TAGGING_PROGRAM_ABI["collective_scopes"] \
                or collective_scope != "none" \
                or not memory_spaces or len(set(memory_spaces)) != len(memory_spaces) \
                or any(space not in NATIVE_TAGGING_PROGRAM_ABI["memory_spaces"]
                       for space in memory_spaces) \
                or (execution_mode == "host" and memory_spaces != ("host",)) \
                or not isinstance(binding.get("tagging_graph_identity"), str) \
                or not binding.get("tagging_graph_identity") \
                or not isinstance(binding.get("clock_identity"), str) \
                or not binding.get("clock_identity") \
                or (resolved_tagging_identity is not None
                    and binding.get("tagging_graph_identity") != resolved_tagging_identity):
            raise ValueError(
                "AMR Tagger lacks the exact resolved candidate-program authority")

    def validate_installed_capability(
        self, binding: Mapping[str, Any], installed: Any,
        resolved_tagging_identity: str | None,
    ) -> None:
        from pops.identity.semantic import semantic_value

        self.validate_resolved_capability(binding, resolved_tagging_identity)
        manifest_capability = _normalize_tagger_capability(
            installed.runtime_contract.capabilities)
        if semantic_value(
                binding.get("tagging_capability"),
                where="installed AMR Tagger capability") != semantic_value(
                    manifest_capability,
                    where="manifest AMR Tagger capability") \
                or not isinstance(binding.get("clock_identity"), str) \
                or not binding["clock_identity"]:
            raise ValueError(
                "external AMR Tagger lacks its exact graph/capability/clock contract")


def _runtime_interface_key(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("AMR provider binding has no native-interface protocol")
    return (
        value.get("uri"), value.get("version"), value.get("catalog_sha256"),
        value.get("protocol_abi"), value.get("cpp_table"),
    )


def _runtime_interface_matches(value: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    from pops.identity.semantic import semantic_value

    return semantic_value(value, where="AMR provider native interface") == semantic_value(
        expected, where="AMR provider protocol interface")


def _runtime_interface_protocols() -> dict[tuple[Any, ...], _AMRRuntimeInterfaceProtocol]:
    from pops import interfaces

    protocols: tuple[_AMRRuntimeInterfaceProtocol, ...] = (
        _ClusteringRuntimeInterfaceProtocol(
            role="clustering",
            native_interface=interfaces.Clustering.to_data(),
            builtin_provider_id="pops.lib.amr::berger_rigoutsos",
            component_installer="_install_amr_clustering_component",
        ),
        _TaggerRuntimeInterfaceProtocol(
            role="tagger",
            native_interface=interfaces.Tagger.to_data(),
            builtin_provider_id="pops.lib.amr::symbolic_tagger",
            component_installer="_install_amr_tagger_component",
        ),
    )
    return {_runtime_interface_key(row.native_interface): row for row in protocols}


def amr_provider_binding_identity(role: str, data: Mapping[str, Any]) -> str:
    """Return the canonical identity for one resolved third-party provider binding.

    The namespace belongs to the selected native interface.  Extension providers use this
    helper instead of reproducing PoPS identity details or inventing a provider-name branch.
    """
    if not isinstance(role, str) or not role or not isinstance(data, Mapping):
        raise TypeError("AMR provider identity requires one role and one binding mapping")
    protocol = _runtime_interface_protocols().get(
        _runtime_interface_key(data.get("native_interface")))
    if protocol is None or protocol.role != role \
            or not _runtime_interface_matches(
                data.get("native_interface"), protocol.native_interface):
        raise ValueError("AMR provider identity selects an unsupported role/interface")
    from pops.identity.semantic import semantic_value

    payload = {key: value for key, value in data.items() if key != "provider_identity"}
    return make_identity(
        protocol.resolved_identity_namespace,
        semantic_value(payload, where="resolved AMR %s provider" % role),
    ).token


def _validate_builtin_binding(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: Mapping[str, Any],
    component_inputs: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    del component_inputs
    if binding.get("provider_type") != "builtin_amr_%s" % protocol.role \
            or binding.get("provider_id") != protocol.builtin_provider_id \
            or any(name in binding for name in (
                "component_id", "component_manifest_identity", "component",
                "interface_version",
            )):
        raise ValueError("builtin AMR %s provider is not canonical" % protocol.role)


def _validate_external_binding(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: Mapping[str, Any],
    component_inputs: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    component_id = binding.get("component_id")
    component = binding.get("component")
    manifest_identity = binding.get("component_manifest_identity")
    if binding.get("provider_type") != "external_amr_%s" % protocol.role \
            or not isinstance(component_id, str) or not component_id \
            or not isinstance(manifest_identity, str) or not manifest_identity \
            or binding.get("interface_version") != protocol.native_interface.get("version") \
            or not isinstance(component, Mapping) \
            or component.get("component_id") != component_id \
            or component.get("component_manifest") != manifest_identity \
            or not _runtime_interface_matches(
                component.get("interface"), protocol.native_interface) \
            or "provider_id" in binding:
        raise ValueError("external AMR %s provider lost exact component identity" % protocol.role)
    if component_inputs is not None:
        from pops.identity.semantic import semantic_value

        installed = component_inputs.get(component_id)
        if installed is None or semantic_value(
                installed, where="AMR component authority") != semantic_value(
                    component, where="AMR provider component authority"):
            raise ValueError(
                "external AMR %s provider differs from its component authority"
                % protocol.role)


_BINDING_PROTOCOLS: dict[
    str,
    Callable[
        [_AMRRuntimeInterfaceProtocol, Mapping[str, Any],
         Mapping[str, Mapping[str, Any]] | None],
        None,
    ],
] = {
    "builtin": _validate_builtin_binding,
    "external_component": _validate_external_binding,
}


def validate_amr_provider_binding(
    *,
    role: str,
    frozen_binding: Any,
    layout_identity: str,
    component_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    resolved_tagging_identity: str | None = None,
) -> dict[str, Any]:
    """Authenticate detached provider data through its interface and installation protocols."""
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR %s provider binding must be a mapping" % role)
    binding = dict(frozen_binding)
    protocol = _runtime_interface_protocols().get(
        _runtime_interface_key(binding.get("native_interface")))
    if protocol is None or protocol.role != role \
            or not _runtime_interface_matches(
                binding.get("native_interface"), protocol.native_interface):
        raise ValueError("AMR provider binding role disagrees with its native interface")
    installation = binding.get("runtime_installation")
    if binding.get("schema_version") != 1 \
            or not isinstance(binding.get("provider_identity"), str) \
            or not binding["provider_identity"] \
            or binding.get("layout_identity") != layout_identity \
            or not isinstance(installation, Mapping) \
            or set(installation) != {"schema_version", "protocol"} \
            or installation.get("schema_version") != 1:
        raise ValueError("AMR %s provider binding is incomplete or unauthenticated" % role)
    validator = _BINDING_PROTOCOLS.get(installation.get("protocol"))
    if validator is None:
        raise NotImplementedError("AMR %s provider binding protocol is not implemented" % role)
    validator(protocol, binding, component_inputs)
    protocol.validate_resolved_capability(binding, resolved_tagging_identity)
    if binding["provider_identity"] != amr_provider_binding_identity(role, binding):
        raise ValueError(
            "AMR %s provider_identity does not authenticate its resolved authority" % role)
    return binding


def _prepare_builtin_provider(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: dict[str, Any],
    **_: Any,
) -> PreparedAMRProviderInstallation:
    if binding.get("provider_id") != protocol.builtin_provider_id \
            or any(name in binding for name in (
                "component_id", "component_manifest_identity", "component")):
        raise ValueError("builtin AMR %s provider is not canonical" % protocol.role)
    return PreparedAMRProviderInstallation(protocol.role, binding)


def _prepare_external_provider(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: dict[str, Any],
    *,
    components: Mapping[str, Any],
    native: Any,
    resolved_tagging_identity: str | None,
) -> PreparedAMRProviderInstallation:
    component_id = binding.get("component_id")
    installed = components.get(component_id)
    if installed is None:
        raise ValueError(
            "AMR %s provider requires exact component %r; it is not installed"
            % (protocol.role, component_id))
    if installed.component_manifest.token != binding.get("component_manifest_identity"):
        raise ValueError("AMR %s provider changed component manifest identity" % protocol.role)
    if installed.interface.to_data() != binding.get("native_interface") \
            or installed.interface.version != binding.get("interface_version"):
        raise ValueError("AMR %s provider changed native interface/version" % protocol.role)
    if installed.native_handle is None:
        raise ValueError("AMR %s component must be loaded before installation" % protocol.role)
    component = binding.get("component")
    if not isinstance(component, Mapping) \
            or component.get("component_id") != component_id \
            or component.get("component_manifest") != installed.component_manifest.token \
            or component.get("interface") != installed.interface.to_data():
        raise ValueError("AMR %s provider lost its exact component declaration" % protocol.role)
    protocol.validate_installed_capability(
        binding, installed, resolved_tagging_identity)
    installer = getattr(native, protocol.component_installer, None)
    if not callable(installer):
        raise NotImplementedError(
            "the selected native provider cannot install external AMR %s" % protocol.role)
    return PreparedAMRProviderInstallation(
        protocol.role, binding, installer, installed.native_handle)


_INSTALLATION_PROTOCOLS: dict[str, Callable[..., PreparedAMRProviderInstallation]] = {
    "builtin": _prepare_builtin_provider,
    "external_component": _prepare_external_provider,
}


def _builtin_native_config(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: dict[str, Any],
) -> PreparedAMRProviderNativeConfig:
    if binding.get("provider_id") != protocol.builtin_provider_id:
        raise ValueError("builtin AMR %s provider is not canonical" % protocol.role)
    return PreparedAMRProviderNativeConfig(
        protocol.role, protocol.builtin_native_config(binding), {})


def _external_native_config(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: dict[str, Any],
) -> PreparedAMRProviderNativeConfig:
    component = binding.get("component")
    parameters = component.get("parameters") if isinstance(component, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise TypeError(
            "external AMR %s provider lost its canonical component parameters"
            % protocol.role)
    # The external component receives these exact options through LoadedComponent preparation;
    # they are deliberately not reinterpreted as controls of a builtin algorithm.
    return PreparedAMRProviderNativeConfig(protocol.role, {}, dict(parameters))


_NATIVE_CONFIG_PROTOCOLS: dict[
    str, Callable[[_AMRRuntimeInterfaceProtocol, dict[str, Any]],
                  PreparedAMRProviderNativeConfig]
] = {
    "builtin": _builtin_native_config,
    "external_component": _external_native_config,
}


def prepare_amr_provider_native_config(
    frozen_binding: Any,
) -> PreparedAMRProviderNativeConfig:
    """Prepare bind-time config without dispatching on concrete provider names."""
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR provider native config requires one binding mapping")
    raw = dict(frozen_binding)
    protocols = _runtime_interface_protocols()
    protocol = protocols.get(_runtime_interface_key(raw.get("native_interface")))
    if protocol is None:
        raise ValueError("AMR provider native config selects an unsupported interface")
    binding = validate_amr_provider_binding(
        role=protocol.role,
        frozen_binding=raw,
        layout_identity=raw.get("layout_identity"),
    )
    installation = binding.get("runtime_installation")
    route = installation.get("protocol") if isinstance(installation, Mapping) else None
    lowering = _NATIVE_CONFIG_PROTOCOLS.get(route)
    if lowering is None:
        raise NotImplementedError("AMR provider native config protocol is not implemented")
    return lowering(protocol, binding)


def prepare_amr_provider_installation(
    *,
    role: str,
    frozen_binding: Any,
    layout_identity: str,
    components: Mapping[str, Any],
    native: Any,
    resolved_tagging_identity: str | None,
) -> PreparedAMRProviderInstallation:
    """Lower one detached provider through its native-interface protocol.

    The runtime does not infer behavior from the mapping slot or from ``provider_type``.  The
    authority carries an explicit installation protocol, while the exact native interface owns
    capability validation and the native installation route.
    """
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR %s provider binding must be an immutable mapping" % role)
    binding = validate_amr_provider_binding(
        role=role,
        frozen_binding=frozen_binding,
        layout_identity=layout_identity,
        resolved_tagging_identity=resolved_tagging_identity,
    )
    protocol = _runtime_interface_protocols()[
        _runtime_interface_key(binding.get("native_interface"))]
    installation = binding.get("runtime_installation")
    if not isinstance(installation, Mapping) \
            or set(installation) != {"schema_version", "protocol"} \
            or installation.get("schema_version") != 1:
        raise ValueError("AMR %s provider lacks its exact runtime protocol" % role)
    lowering = _INSTALLATION_PROTOCOLS.get(installation.get("protocol"))
    if lowering is None:
        raise NotImplementedError(
            "AMR %s provider runtime protocol is not implemented" % role)
    return lowering(
        protocol,
        binding,
        components=components,
        native=native,
        resolved_tagging_identity=resolved_tagging_identity,
    )


__all__ = [
    "AMRProviderLoweringContext",
    "amr_provider_binding_identity",
    "ClusteringProvider",
    "PreparedAMRProviderNativeConfig",
    "ResolvedAMRProviderBinding",
    "TaggerProvider",
    "validate_amr_provider_binding",
]
