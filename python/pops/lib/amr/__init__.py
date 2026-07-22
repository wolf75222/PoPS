"""Pre-implemented AMR transfer and materialization policies.

These are authoring descriptors.  Their order, halo and conservation capabilities are intrinsic;
users select physics-level policies and never author compiler ``AccuracyRequirement`` objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from pops.identity import make_identity


class _BuiltinLoadBalance:
    """Constructor-only value implementing the open AMR load-balance data protocol."""

    __pops_ir_immutable__: ClassVar[bool] = True
    provider_id: ClassVar[str]
    native_route: ClassVar[str]
    option_schema_identity: ClassVar[str]
    consumes_weights: ClassVar[bool]

    def load_balance_provider_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": 1,
            "provider_type": "amr_load_balance_provider",
            "provider_id": self.provider_id,
            "native_route": self.native_route,
            "option_schema_identity": self.option_schema_identity,
            "options": {},
            "weight_capability": {
                "authenticated": True,
                "consumed": self.consumes_weights,
            },
        }
        data["provider_identity"] = make_identity(
            "amr-load-balance-provider", data).token
        return data

    inspect = load_balance_provider_data
    canonical_identity = load_balance_provider_data


@dataclass(frozen=True, slots=True)
class SpaceFillingCurve(_BuiltinLoadBalance):
    """Weighted Morton-order partitions; the default locality-preserving policy."""

    provider_id: ClassVar[str] = "pops.lib.amr::space_filling_curve"
    native_route: ClassVar[str] = "space_filling_curve"
    option_schema_identity: ClassVar[str] = (
        "pops.amr.load-balance.space-filling-curve@1"
    )
    consumes_weights: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class Knapsack(_BuiltinLoadBalance):
    """Weighted longest-processing-time ownership balancing."""

    provider_id: ClassVar[str] = "pops.lib.amr::knapsack"
    native_route: ClassVar[str] = "knapsack"
    option_schema_identity: ClassVar[str] = "pops.amr.load-balance.knapsack@1"
    consumes_weights: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class RoundRobin(_BuiltinLoadBalance):
    """Index policy; weights are authenticated but intentionally do not select owners."""

    provider_id: ClassVar[str] = "pops.lib.amr::round_robin"
    native_route: ClassVar[str] = "round_robin"
    option_schema_identity: ClassVar[str] = "pops.amr.load-balance.round-robin@1"
    consumes_weights: ClassVar[bool] = False


class _ImmutableTransferPolicy:
    """Marker shared by constructor-only, frozen transfer policy values."""

    __pops_ir_immutable__: ClassVar[bool] = True

    def amr_transfer_kernel_data(self) -> dict[str, Any]:
        required = (
            "native_route", "order", "ghost_depth", "dimensions",
            "refinement_ratios", "conservative", "temporal",
        )
        return {
            "schema_version": 1,
            "kernel_type": "amr_transfer_kernel",
            **{name: getattr(self, name) for name in required},
        }

    def amr_transfer_policy_data(self) -> dict[str, Any]:
        kind = getattr(self, "policy_kind", None)
        data: dict[str, Any] = {
            "schema_version": 1,
            "authority_type": "amr_transfer_policy",
            "policy_kind": kind,
        }
        routes = {}
        for name in getattr(type(self), "__dataclass_fields__", {}):
            value = getattr(self, name)
            candidates = getattr(value, "amr_transfer_kernel_candidates", None)
            if callable(candidates):
                values = tuple(candidates())
                if not values:
                    raise ValueError("AMR transfer kernel family must not be empty")
                routes[name] = [item.amr_transfer_kernel_data() for item in values]
                continue
            protocol = getattr(value, "amr_transfer_kernel_data", None)
            if callable(protocol):
                routes[name] = protocol()
        if routes:
            data["routes"] = routes
        native_route = getattr(self, "native_route", None)
        if native_route is not None:
            if not isinstance(native_route, str) or not native_route:
                raise TypeError("AMR transfer native_route must be non-empty text")
            data["native_route"] = native_route
        return data


@dataclass(frozen=True, slots=True)
class ConservativeLinear(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "conservative_linear"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class VolumeAverage(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "volume_average"
    order: ClassVar[int] = 1
    ghost_depth: ClassVar[tuple[int, ...]] = (0,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class _ConservativeCoarseFineSecondOrder(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "conservative_coarse_fine"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (2,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class _ConservativeCoarseFineFifthOrder(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "conservative_polynomial5_coarse_fine"
    order: ClassVar[int] = 5
    ghost_depth: ClassVar[tuple[int, ...]] = (3,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class ConservativeCoarseFine(_ImmutableTransferPolicy):
    """Capability family selected from the resolved spatial order and halo."""

    def amr_transfer_kernel_candidates(self) -> tuple[Any, ...]:
        return (
            _ConservativeCoarseFineSecondOrder(),
            _ConservativeCoarseFineFifthOrder(),
        )


@dataclass(frozen=True, slots=True)
class LinearTimeInterpolation(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "linear_time_interpolation"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (0,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class DivergencePreservingFace(_ImmutableTransferPolicy):
    """Coupled normal-face vector prolongation; never an independent scalar interpolation."""

    native_route: ClassVar[str] = "face_divergence_preserving"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class BilinearNode(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "node_bilinear"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = False
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class StateTransfer(_ImmutableTransferPolicy):
    policy_kind: ClassVar[str] = "state"
    prolongation: ConservativeLinear = field(default_factory=ConservativeLinear)
    restriction: VolumeAverage = field(default_factory=VolumeAverage)
    coarse_fine: ConservativeCoarseFine = field(default_factory=ConservativeCoarseFine)
    time_interpolation: LinearTimeInterpolation = field(
        default_factory=LinearTimeInterpolation
    )


@dataclass(frozen=True, slots=True)
class FaceTransfer(_ImmutableTransferPolicy):
    policy_kind: ClassVar[str] = "face"
    prolongation: DivergencePreservingFace = field(
        default_factory=DivergencePreservingFace
    )


@dataclass(frozen=True, slots=True)
class NodeTransfer(_ImmutableTransferPolicy):
    policy_kind: ClassVar[str] = "node"
    prolongation: BilinearNode = field(default_factory=BilinearNode)


@dataclass(frozen=True, slots=True)
class EllipticRecompute(_ImmutableTransferPolicy):
    policy_kind: ClassVar[str] = "field"
    native_route: ClassVar[str] = "elliptic_solve"


@dataclass(frozen=True, slots=True)
class PatchTopologyRebuild(_ImmutableTransferPolicy):
    policy_kind: ClassVar[str] = "cache"
    native_route: ClassVar[str] = "patch_topology"


@dataclass(frozen=True, slots=True)
class SymbolicTagger:
    """Builtin data-only tag-graph VM, selected through the AMR provider protocol."""

    __pops_ir_immutable__: ClassVar[bool] = True

    def resolve_references(self, resolver: Any) -> SymbolicTagger:
        if not callable(resolver):
            raise TypeError("SymbolicTagger.resolve_references requires a callable resolver")
        return self

    def require_component_inputs(self, components: Any) -> None:
        del components

    def require_tagging_graph(self, graph: Any) -> None:
        from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI

        registrations = getattr(graph, "registrations", None)
        authoring = getattr(graph, "graph", None)
        if not isinstance(registrations, tuple) or authoring is None:
            raise TypeError("SymbolicTagger requires one resolved AMRTagging graph")
        supported = set(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"]) | set(
            NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])
        missing = sorted({row.node_type for row in registrations} - supported)
        if missing:
            raise NotImplementedError(
                "builtin AMR Tagger lacks resolved opcode(s): %s" % ", ".join(missing))
        def require_stencils(node: Any) -> None:
            if getattr(node, "node_type", None) in {"gradient_above", "gradient_below"}:
                from pops.numerics.indicator_stencils import DiscreteGradientStencil

                lowering = getattr(getattr(node, "context", None), "lowering", None)
                if type(lowering) is not DiscreteGradientStencil:
                    raise TypeError("resolved AMR gradient has no typed stencil lowering")
                if lowering.route not in NATIVE_TAGGING_PROGRAM_ABI[
                        "indicator_stencil_routes"]:
                    raise NotImplementedError(
                        "builtin AMR Tagger lacks indicator stencil route %r" % lowering.route)
                if any(len(axis.offsets) > NATIVE_TAGGING_PROGRAM_ABI[
                        "maximum_stencil_terms"] for axis in lowering.axes):
                    raise NotImplementedError(
                        "builtin AMR Tagger stencil exceeds maximum_stencil_terms")
            for child in node.operands():
                require_stencils(child)

        require_stencils(authoring.refine)
        if authoring.coarsen is not None:
            require_stencils(authoring.coarsen)
        if (authoring.hysteresis.min_cycles != 0
                and not NATIVE_TAGGING_PROGRAM_ABI["persistent_hysteresis"]):
            raise NotImplementedError(
                "AMR hysteresis min_cycles requires native persistent tagging state; "
                "it is never accepted then ignored")

    def lower_amr_provider(self, context: Any) -> Any:
        from pops.amr.providers import (
            AMRProviderLoweringContext,
            ResolvedAMRProviderBinding,
        )
        from pops.amr.providers import amr_provider_binding_identity

        if type(context) is not AMRProviderLoweringContext:
            raise TypeError("SymbolicTagger requires an AMRProviderLoweringContext")
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
        from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI

        leaf_opcodes = dict(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"])
        logical_opcodes = dict(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])

        data = {
            "schema_version": 1,
            "provider_type": "builtin_amr_tagger",
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "builtin",
            },
            "provider_id": "pops.lib.amr::symbolic_tagger",
            "native_interface": interfaces.Tagger.to_data(),
            "tagging_capability": {
                "schema_version": 1,
                "capability_type": "amr_tagging_program",
                "leaf_opcodes": list(leaf_opcodes),
                "leaf_opcode_ids": list(leaf_opcodes.values()),
                "logical_opcodes": list(logical_opcodes),
                "logical_opcode_ids": list(logical_opcodes.values()),
                "candidate_outputs": list(
                    NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"]),
                "indicator_stencil_routes": list(
                    NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]),
                "maximum_stencil_terms": NATIVE_TAGGING_PROGRAM_ABI[
                    "maximum_stencil_terms"],
                "maximum_instruction_count": NATIVE_TAGGING_PROGRAM_ABI[
                    "maximum_instruction_count"],
                "non_finite_policy": NATIVE_TAGGING_PROGRAM_ABI[
                    "non_finite_policy"],
                "persistent_hysteresis": NATIVE_TAGGING_PROGRAM_ABI[
                    "persistent_hysteresis"],
                "execution_mode": "native_backend",
                "collective_scope": "none",
                "memory_spaces": list(NATIVE_TAGGING_PROGRAM_ABI["memory_spaces"]),
            },
        }
        data["provider_identity"] = make_identity("amr-tagger-provider", data).token
        return data

    inspect = runtime_binding_data
    canonical_identity = runtime_binding_data


@dataclass(frozen=True, slots=True)
class BergerRigoutsos:
    """Builtin clustering provider with intrinsic validated algorithm controls."""

    minimum_efficiency: float = 0.7
    minimum_box_size: int = 1
    maximum_box_size: int = 32
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if isinstance(self.minimum_efficiency, bool) or not isinstance(
                self.minimum_efficiency, (int, float)):
            raise TypeError("BergerRigoutsos.minimum_efficiency must be numeric")
        if not 0.0 < float(self.minimum_efficiency) <= 1.0:
            raise ValueError("BergerRigoutsos.minimum_efficiency must be in (0, 1]")
        for name in ("minimum_box_size", "maximum_box_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("BergerRigoutsos.%s must be an integer >= 1" % name)
        if self.minimum_box_size > self.maximum_box_size:
            raise ValueError(
                "BergerRigoutsos.minimum_box_size must not exceed maximum_box_size")
        object.__setattr__(self, "minimum_efficiency", float(self.minimum_efficiency))

    def resolve_references(self, resolver: Any) -> BergerRigoutsos:
        if not callable(resolver):
            raise TypeError("BergerRigoutsos.resolve_references requires a callable resolver")
        return self

    def require_component_inputs(self, components: Any) -> None:
        del components

    def lower_amr_provider(self, context: Any) -> Any:
        from pops.amr.providers import (
            AMRProviderLoweringContext,
            ResolvedAMRProviderBinding,
        )

        if type(context) is not AMRProviderLoweringContext:
            raise TypeError("BergerRigoutsos requires an AMRProviderLoweringContext")
        self.require_component_inputs(context.components)
        from pops.amr.providers import amr_provider_binding_identity

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
            "provider_type": "builtin_amr_clustering",
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "builtin",
            },
            "provider_id": "pops.lib.amr::berger_rigoutsos",
            "native_interface": interfaces.Clustering.to_data(),
            "minimum_efficiency": self.minimum_efficiency,
            "minimum_box_size": self.minimum_box_size,
            "maximum_box_size": self.maximum_box_size,
        }
        identity_data = {
            **data,
            "minimum_efficiency": self.minimum_efficiency.hex(),
        }
        data["provider_identity"] = make_identity(
            "amr-clustering-provider", identity_data).token
        return data

    inspect = runtime_binding_data
    canonical_identity = runtime_binding_data


__all__ = [
    "ConservativeCoarseFine",
    "ConservativeLinear",
    "BilinearNode",
    "BergerRigoutsos",
    "DivergencePreservingFace",
    "EllipticRecompute",
    "FaceTransfer",
    "LinearTimeInterpolation",
    "Knapsack",
    "NodeTransfer",
    "PatchTopologyRebuild",
    "StateTransfer",
    "SpaceFillingCurve",
    "SymbolicTagger",
    "RoundRobin",
    "VolumeAverage",
]
