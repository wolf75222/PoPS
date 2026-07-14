"""Typed external AMR provider bindings.

These values select one exact native component at authoring time.  They retain no
Python callback: resolve authenticates the component and its graph-evaluator capabilities,
then compile carries only the immutable native binding record to installation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI


_TAGGER_LEAF_OPCODES = dict(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"])
_TAGGER_LOGICAL_OPCODES = dict(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])
_TAGGER_OUTPUTS = tuple(NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"])


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
    }


def _tagger_capability(component: Any) -> dict[str, Any]:
    return _normalize_tagger_capability(component.component_manifest.capabilities)


@dataclass(frozen=True, slots=True)
class TaggerProvider:
    """Bind one external evaluator to the resolved AMRTagging graph."""

    component: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops import interfaces

        _external_component(
            self.component, interface=interfaces.Tagger, where="TaggerProvider.component")
        _tagger_capability(self.component)

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

    def runtime_binding_data(self) -> dict[str, Any]:
        from pops import interfaces

        data = {
            "schema_version": 1,
            "provider_type": "external_amr_tagger",
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

    def runtime_binding_data(self) -> dict[str, Any]:
        from pops import interfaces

        data = {
            "schema_version": 1,
            "provider_type": "external_amr_clustering",
            **_component_binding(self.component, interfaces.Clustering),
        }
        data["provider_identity"] = make_identity("amr-clustering-provider", data).token
        return data

    inspect = runtime_binding_data
    canonical_identity = runtime_binding_data


__all__ = ["ClusteringProvider", "TaggerProvider"]
