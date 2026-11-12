"""Inert geometry evolution requests, contract pops.geometry-evolution@1.

This extension describes the coordinate law through the existing analytic algebra.
Preparation selects a coupled native realization with trial/accepted measures
and swept-volume exchanges, or reports the exact missing provider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.analytic import ScalarExpr
from pops.descriptors_report import CapabilitySet, RequirementSet
from ._descriptor import MeshDescriptor


GEOMETRY_EVOLUTION_CONTRACT = "pops://geometry-evolution/moving-control-volumes@1"
GEOMETRY_EVOLUTION_OBLIGATIONS = (
    "trial_and_accepted_coordinates_and_cell_measures",
    "oriented_swept_volumes_from_same_time_quadrature",
    "relative_physical_flux_minus_density_times_mesh_flux",
    "state_geometry_exchange_atomic_publication_and_rollback",
    "duration_geometry_and_input_qualified_reuse",
    "saved_geometry_and_inventory_receipt",
)


@dataclass(frozen=True, slots=True)
class GeometryEvolution:
    """A coordinate map on a fixed physical domain and fixed topology.

    Coordinate, clock, parameter and discrete-input identities belong to the
    common ScalarExpr body, never to an opaque physics-specific callback.
    Boundary displacement compatibility must be checked by the eventual native
    geometry provider. This declaration alone establishes no capability.
    """

    coordinate_map: tuple[ScalarExpr, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        values = self.coordinate_map
        if type(values) is not tuple or len(values) not in (1, 2, 3) \
                or any(type(value) is not ScalarExpr for value in values):
            raise TypeError("GeometryEvolution.coordinate_map requires 1..3 ScalarExpr values")
        frames = {value.frame_id for value in values if value.frame_id is not None}
        if len(frames) != 1:
            raise ValueError("GeometryEvolution requires one exact reference frame")
        for value in values:
            value.validate()

    @property
    def frame_id(self) -> str:
        return next(value.frame_id for value in self.coordinate_map if value.frame_id is not None)

    def to_data(self) -> dict[str, Any]:
        return {"contract": GEOMETRY_EVOLUTION_CONTRACT, "schema_version": 1,
                "domain_motion": "fixed_physical_domain", "topology": "fixed",
                "frame_id": self.frame_id,
                "coordinate_map": [value.to_data() for value in self.coordinate_map],
                "obligations": list(GEOMETRY_EVOLUTION_OBLIGATIONS)}

    __pops_semantic_data__ = to_data

    def resolve_references(self, resolver: Any) -> GeometryEvolution:
        return type(self)(tuple(value.resolve_references(resolver)
                                for value in self.coordinate_map))


def geometry_rule_data(rule: Any) -> dict[str, Any]:
    """The stated rule of a configuration's measures, as data on [0, 1] (D0, D0b).

    ``rule`` is a ``pops.Quadrature`` (or any value with its ``nodes``, ``weights`` and bounded
    ``interval``); the core names no rule (D0). The nodes and weights are mapped once from the
    reference interval to [0, 1], t = (v - a) / (b - a) and w / (b - a), and kept as exact binary64
    values, so that two spellings of one rule give one identity.
    """
    import math

    try:
        nodes, weights, interval = rule.nodes, rule.weights, rule.interval
    except AttributeError:
        raise TypeError("MovingControlVolumes(geometry=...) takes a quadrature rule, "
                        "pops.Quadrature(nodes=, weights=, interval=) or a rule of "
                        "pops.lib.quadrature, got %r" % (rule,)) from None
    nodes = tuple(float(node) for node in nodes)
    weights = tuple(float(weight) for weight in weights)
    lower, upper = (float(end) for end in interval)
    if not nodes or len(nodes) != len(weights):
        raise ValueError("MovingControlVolumes(geometry=...) needs as many weights as nodes, "
                         "at least one")
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("MovingControlVolumes(geometry=...) needs a rule on a bounded interval; "
                         "%r is not one" % ((lower, upper),))
    if any(not math.isfinite(value) for value in nodes + weights) \
            or any(not lower <= node <= upper for node in nodes):
        raise ValueError("MovingControlVolumes(geometry=...) needs finite nodes inside the "
                         "interval of the rule and finite weights")
    span = upper - lower
    return {"rule": "tensor-quadrature@1",
            "nodes": [((node - lower) / span).hex() for node in nodes],
            "weights": [(weight / span).hex() for weight in weights]}


class MovingControlVolumes(MeshDescriptor):
    """Compose a layout with an explicit moving-geometry requirement.

    The wrapper declares the reference layout and exact geometry obligation.
    Preparation selects an authenticated coupled native realization or refuses
    the missing provider before JIT; the descriptor executes no solver itself.

    ``geometry=`` states the quadrature rule of the measures (D0b: the arrays of a mapped
    configuration are generated « with a stated quadrature »), applied on every logical axis. It
    is recorded as the requirement ``geometry_rule``; a layout that states none keeps the identity
    it had.
    """

    category = "layout"

    def __init__(self, layout: Any, *, evolution: GeometryEvolution,
                 geometry: Any = None) -> None:
        if type(evolution) is not GeometryEvolution:
            raise TypeError("MovingControlVolumes.evolution requires GeometryEvolution")
        geometry_of_layout = layout.normalized_geometry()
        if geometry_of_layout.frame_id != evolution.frame_id \
                or geometry_of_layout.dimension != len(evolution.coordinate_map):
            raise ValueError("geometry evolution must match the exact layout frame and rank")
        self.layout = layout
        self.evolution = evolution
        # The rule is kept as its data only: a descriptor crosses the compiled boundary.
        self._rule = None if geometry is None else geometry_rule_data(geometry)

    def _rule_data(self) -> dict[str, Any]:
        return {} if self._rule is None else {"geometry_rule": self._rule}

    def semantic_data(self) -> dict[str, Any]:
        return {"kind": "moving-control-volumes@1", "layout": self.layout,
                "evolution": self.evolution.to_data(), **self._rule_data()}

    def options(self) -> dict[str, Any]:
        return {**self.layout.options(), "geometry_evolution": self.evolution.to_data(),
                **self._rule_data()}

    def capabilities(self) -> CapabilitySet:
        # Dynamic support is an obligation, not a provided native capability.
        return CapabilitySet(self.layout.capabilities().to_dict())

    def requirements(self) -> RequirementSet:
        return RequirementSet({**self.layout.requirements().to_dict(),
                               "geometry_evolution": self.evolution.to_data(),
                               **self._rule_data()})

    def normalized_geometry(self) -> Any:
        # This is the reference geometry. The requirement prevents serving it as
        # an evolved physical geometry during production resolution.
        return self.layout.normalized_geometry()

    def native_spatial_data(self) -> Any:
        return self.layout.native_spatial_data()

    def resolve_for_case(self, resolver: Any) -> MovingControlVolumes:
        result = type(self)(self.layout.resolve_for_case(resolver),
                            evolution=self.evolution.resolve_references(resolver))
        result._rule = self._rule
        return result

    def validate(self, context: Any = None) -> bool:
        self.layout.validate(context)
        return True


__all__ = ["GeometryEvolution", "MovingControlVolumes", "GEOMETRY_EVOLUTION_CONTRACT"]
