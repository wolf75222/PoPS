"""Typed transport-boundary authoring and exact low-level port resolution."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, ClassVar

from pops.domain import DomainBoundary
from pops._ir import Expr
from pops._ir.expr import Const
from pops._ir.visitors import _key
from pops.identity import make_identity
from pops.identity.semantic import semantic_value
from pops.model import Handle, OwnerPath, ParamHandle
from pops.representations import Representation


_SCHEMA_VERSION = 1


def _expression(value: Any, *, where: str) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, str) or callable(value):
        raise TypeError("%s must be a PoPS Expr or an exact scalar, never text/callable" % where)
    try:
        return Const(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a PoPS Expr or an exact scalar" % where) from exc


def _expression_data(value: Expr) -> Any:
    """Return the same stable structural protocol used by derived parameter expressions."""
    return {
        "protocol": "pops.expr.key.v1",
        "value": json.loads(json.dumps(
            _key(value), sort_keys=True, separators=(",", ":"), allow_nan=False)),
    }


def _state(value: Any, *, where: str, require_instance: bool = True) -> Handle:
    if not isinstance(value, Handle) or value.kind != "state":
        raise TypeError("%s requires a typed state Handle, never a name" % where)
    if require_instance and not value.is_instance:
        raise TypeError(
            "%s requires a block-qualified state such as block[state]; model-local states are "
            "ambiguous at a physical boundary" % where
        )
    return value


def _converter(value: Any) -> Handle | None:
    if value is None:
        return None
    if not isinstance(value, Handle) or value.kind != "representation_conversion":
        raise TypeError("boundary converter must be a representation_conversion Handle or None")
    return value


def _condition_protocol(value: Any, *, where: str) -> Any:
    _state(getattr(value, "state", None), where="%s.state" % where)
    for method in ("inspect", "resolve_references", "resolve_condition"):
        if not callable(getattr(value, method, None)):
            raise TypeError(
                "%s must implement the transport-boundary condition protocol (%s missing)"
                % (where, method)
            )
    return value


def _unique_references(*groups: Any) -> tuple[Handle, ...]:
    rows: list[Handle] = []
    for group in groups:
        for value in group:
            if not isinstance(value, Handle):
                raise TypeError("boundary declaration references must contain only Handle values")
            if value not in rows:
                rows.append(value)
    return tuple(rows)


def _representation_handle(state: Handle, name: str) -> Handle:
    digest = hashlib.sha256(
        (state.qualified_id + "\0" + name).encode("utf-8")
    ).hexdigest()[:24]
    return Handle(
        "state-representation-%s" % digest,
        kind="representation",
        owner=state.owner_path,
    )


def _provider_handle(state: Handle, boundary: Any, condition_type: str) -> Handle:
    payload = "%s\0%s\0%s" % (
        state.qualified_id, boundary.canonical_id, condition_type)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return Handle(
        "%s-%s" % (condition_type, digest),
        kind="boundary_provider",
        owner=state.owner_path,
    )


def _dependency_handles(
    values: tuple[Expr, ...], *, include_state: Handle | None = None
) -> tuple[tuple[Handle, ...], tuple[Handle, ...], tuple[Handle, ...], tuple[ParamHandle, ...]]:
    references = _unique_references(*(value.declaration_references() for value in values))
    states = [reference for reference in references if reference.kind == "state"]
    fields = [reference for reference in references if reference.kind == "field"]
    time = [reference for reference in references if reference.kind == "time"]
    params = [
        reference for reference in references
        if isinstance(reference, ParamHandle) and reference.param_kind == "runtime"
    ]
    supported = {"state", "field", "time", "parameter"}
    unsupported = sorted({reference.kind for reference in references} - supported)
    if unsupported:
        raise TypeError(
            "transport boundary expression has unsupported dependency Handle kinds %s"
            % unsupported
        )
    if include_state is not None and include_state not in states:
        states.append(include_state)
    return tuple(states), tuple(fields), tuple(time), tuple(params)


def _closure() -> Any:
    from pops.mesh.boundaries import (
        CharacteristicClosure,
        ClosureMode,
        IncomingMultiplicity,
        SignDependence,
        SonicPolicy,
    )

    return CharacteristicClosure(
        mode=ClosureMode.NONE,
        sign_dependence=SignDependence.FIXED,
        sonic=SonicPolicy.NEUTRAL,
        incoming=IncomingMultiplicity.SINGLE,
        characteristics=(),
    )


@dataclass(frozen=True, slots=True)
class BoundaryStencilRequirement:
    """Stencil facts derived from every rate method that reads one state."""

    state: Handle
    ghost_depth: int
    formal_orders: tuple[int, ...]
    rates: tuple[str, ...]

    def __post_init__(self) -> None:
        _state(self.state, where="BoundaryStencilRequirement.state")
        if not self.state.is_resolved:
            raise TypeError("BoundaryStencilRequirement.state must be canonical")
        if isinstance(self.ghost_depth, bool) or not isinstance(self.ghost_depth, int) \
                or self.ghost_depth < 1:
            raise ValueError("BoundaryStencilRequirement.ghost_depth must be an integer >= 1")
        if not isinstance(self.formal_orders, tuple) or not self.formal_orders \
                or any(isinstance(row, bool) or not isinstance(row, int) or row < 1
                       for row in self.formal_orders):
            raise TypeError("BoundaryStencilRequirement.formal_orders must contain positive ints")
        if tuple(sorted(set(self.formal_orders))) != self.formal_orders:
            raise ValueError("BoundaryStencilRequirement.formal_orders must be canonical")
        if not isinstance(self.rates, tuple) or not self.rates \
                or any(not isinstance(row, str) or not row for row in self.rates):
            raise TypeError("BoundaryStencilRequirement.rates must contain qualified ids")
        if tuple(sorted(set(self.rates))) != self.rates:
            raise ValueError("BoundaryStencilRequirement.rates must be canonical")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "state": self.state.canonical_identity(),
            "ghost_depth": self.ghost_depth,
            "formal_orders": list(self.formal_orders),
            "rates": list(self.rates),
        }


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedTransportCondition:
    geometry: DomainBoundary
    condition_type: str
    state: Handle
    values: tuple[Expr, ...]
    requirement: BoundaryStencilRequirement
    provider: Any

    def __post_init__(self) -> None:
        from pops.mesh.boundaries import BoundaryProvider

        if not isinstance(self.geometry, DomainBoundary):
            raise TypeError("ResolvedTransportCondition.geometry must be a DomainBoundary")
        if self.condition_type not in {"inflow", "outflow"}:
            raise ValueError("unsupported built-in transport condition type")
        _state(self.state, where="ResolvedTransportCondition.state")
        if not self.state.is_resolved:
            raise TypeError("ResolvedTransportCondition.state must be canonical")
        if not isinstance(self.values, tuple) or any(not isinstance(row, Expr) for row in self.values):
            raise TypeError("ResolvedTransportCondition.values must contain Expr values")
        if self.requirement.state != self.state:
            raise ValueError("transport condition and stencil requirement refer to different states")
        if not isinstance(self.provider, BoundaryProvider):
            raise TypeError("ResolvedTransportCondition.provider must be a BoundaryProvider")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "condition_type": self.condition_type,
            "geometry": self.geometry.canonical_identity(),
            "state": self.state.canonical_identity(),
            "values": [_expression_data(value) for value in self.values],
            "stencil": self.requirement.canonical_identity(),
            "provider": self.provider.canonical_identity(),
        }

    inspect = canonical_identity


def _resolved_condition(
    condition: Any,
    *,
    condition_type: str,
    geometry: DomainBoundary,
    boundary: Any,
    requirement: BoundaryStencilRequirement,
    include_state_dependency: bool,
) -> ResolvedTransportCondition:
    from pops.mesh.boundaries import (
        BoundaryDependencies,
        GhostState,
        Inflow as LowLevelInflow,
        Outflow as LowLevelOutflow,
        RepresentationFlow,
    )

    state = condition.state
    target_name = state.space.representation
    selected = condition.representation
    source_name = target_name if selected is None else selected.name
    source = _representation_handle(state, source_name)
    target = _representation_handle(state, target_name)
    converter = condition.converter
    flow = RepresentationFlow(source=source, target=target, converter=converter)
    states, fields, time, params = _dependency_handles(
        condition.values,
        include_state=state if include_state_dependency else None,
    )
    dependencies = BoundaryDependencies(
        states=states,
        fields=fields,
        time=time,
        runtime_params=params,
        representation=flow,
        characteristic=_closure(),
    )
    output = GhostState(boundary=boundary, subject=state, representation=target)
    factory = LowLevelInflow if condition_type == "inflow" else LowLevelOutflow
    provider = factory(
        handle=_provider_handle(state, geometry, condition_type),
        outputs=(output,),
        dependencies=dependencies,
    )
    return ResolvedTransportCondition(
        geometry=geometry,
        condition_type=condition_type,
        state=state,
        values=condition.values,
        requirement=requirement,
        provider=provider,
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Inflow:
    """Prescribe every component of one block-qualified state on an inflow face."""

    condition_type: ClassVar[str] = "inflow"
    state: Handle
    values: tuple[Expr, ...]
    representation: Representation | None
    converter: Handle | None

    def __init__(
        self,
        *,
        state: Any,
        value: Any,
        representation: Representation | None = None,
        converter: Any = None,
    ) -> None:
        checked_state = _state(state, where="Inflow.state")
        if representation is not None and not isinstance(representation, Representation):
            raise TypeError("Inflow.representation must be a typed Representation or None")
        raw_values = value if isinstance(value, tuple) else (value,)
        if not raw_values:
            raise ValueError("Inflow.value must prescribe at least one state component")
        object.__setattr__(self, "state", checked_state)
        object.__setattr__(self, "values", tuple(
            _expression(row, where="Inflow.value[%d]" % index)
            for index, row in enumerate(raw_values)
        ))
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "converter", _converter(converter))

    def declaration_references(self) -> tuple[Handle, ...]:
        converter = () if self.converter is None else (self.converter,)
        return _unique_references(
            (self.state,),
            *(value.declaration_references() for value in self.values),
            converter,
        )

    def resolve_references(self, resolver: Any) -> Inflow:
        if not callable(resolver):
            raise TypeError("Inflow.resolve_references requires a callable resolver")
        converter = None if self.converter is None else resolver(self.converter)
        return type(self)(
            state=resolver(self.state),
            value=tuple(value.resolve_references(resolver) for value in self.values),
            representation=self.representation,
            converter=converter,
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "condition_type": self.condition_type,
            "state": self.state.inspect(),
            "values": [_expression_data(value) for value in self.values],
            "representation": (
                None if self.representation is None else self.representation.canonical_identity()),
            "converter": None if self.converter is None else self.converter.inspect(),
        }

    def resolve_condition(
        self,
        *,
        geometry: DomainBoundary,
        boundary: Any,
        requirement: BoundaryStencilRequirement,
    ) -> ResolvedTransportCondition:
        components = getattr(self.state.space, "components", ())
        if len(self.values) != len(components):
            raise ValueError(
                "Inflow for state %s must prescribe exactly %d components, got %d"
                % (self.state.qualified_id, len(components), len(self.values))
            )
        return _resolved_condition(
            self,
            condition_type=self.condition_type,
            geometry=geometry,
            boundary=boundary,
            requirement=requirement,
            include_state_dependency=False,
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Outflow:
    """Extrapolate one block-qualified state at a physical outflow face."""

    condition_type: ClassVar[str] = "outflow"
    state: Handle
    values: tuple[Expr, ...]
    representation: Representation | None
    converter: Handle | None

    def __init__(self, *, state: Any) -> None:
        object.__setattr__(self, "state", _state(state, where="Outflow.state"))
        object.__setattr__(self, "values", ())
        object.__setattr__(self, "representation", None)
        object.__setattr__(self, "converter", None)

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.state,)

    def resolve_references(self, resolver: Any) -> Outflow:
        if not callable(resolver):
            raise TypeError("Outflow.resolve_references requires a callable resolver")
        return type(self)(state=resolver(self.state))

    def inspect(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "condition_type": self.condition_type,
            "state": self.state.inspect(),
        }

    def resolve_condition(
        self,
        *,
        geometry: DomainBoundary,
        boundary: Any,
        requirement: BoundaryStencilRequirement,
    ) -> ResolvedTransportCondition:
        return _resolved_condition(
            self,
            condition_type=self.condition_type,
            geometry=geometry,
            boundary=boundary,
            requirement=requirement,
            include_state_dependency=True,
        )


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedTransportBoundarySet:
    domain_geometry_id: str
    conditions: tuple[ResolvedTransportCondition, ...]
    plan: Any

    def __post_init__(self) -> None:
        from pops.mesh.boundaries import ResolvedBoundaryPlan

        if not isinstance(self.domain_geometry_id, str) or not self.domain_geometry_id:
            raise TypeError("resolved transport domain identity must be non-empty text")
        if not isinstance(self.conditions, tuple) or not self.conditions \
                or any(not isinstance(row, ResolvedTransportCondition)
                       for row in self.conditions):
            raise TypeError("resolved transport conditions must be a non-empty tuple")
        if not isinstance(self.plan, ResolvedBoundaryPlan):
            raise TypeError("resolved transport plan must be a ResolvedBoundaryPlan")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "authority_type": "transport_boundary_set",
            "domain_geometry_id": self.domain_geometry_id,
            "conditions": [row.canonical_identity() for row in self.conditions],
            "plan": self.plan.canonical_identity(),
        }

    inspect = canonical_identity

    def ghost_plan_composer_capability(self) -> dict[str, Any]:
        """Advertise the narrow open composer protocol; this authority composes only itself."""
        return {"schema_version": 1, "scope": "self"}

    def compose_ghost_plan(self, context: Any) -> Any:
        from pops.mesh.boundaries.composition import (
            GhostPlanCompositionContext,
            compose_transport_boundary,
        )

        if not isinstance(context, GhostPlanCompositionContext):
            raise TypeError("transport boundary composition requires GhostPlanCompositionContext")
        if context.authorities != (self,):
            raise ValueError(
                "TransportBoundarySet composes only itself; use an explicit scope='all' composer "
                "for multiple authorities"
            )
        return compose_transport_boundary(self, context=context)

    def _native_contract(self) -> tuple[Handle, int, tuple[ResolvedTransportCondition, ...], int]:
        """Validate the complete compile-time shape of the built-in native provider."""
        states = {row.state for row in self.conditions}
        if len(states) != 1:
            raise NotImplementedError(
                "the installed native block provider requires one state per boundary plan"
            )
        state = next(iter(states))
        components = tuple(getattr(state.space, "components", ()))
        if not components:
            raise TypeError("resolved transport boundary state has no component manifest")
        ncomp = len(components)
        face_rows: list[ResolvedTransportCondition | None] = [None, None, None, None]
        depth = 0
        for condition in self.conditions:
            geometry = condition.geometry
            if geometry.axis.index not in (0, 1):
                raise NotImplementedError(
                    "the installed native transport boundary provider is two-dimensional"
                )
            face = 2 * geometry.axis.index + (0 if geometry.side.value == "lower" else 1)
            if face_rows[face] is not None:
                raise ValueError("native transport boundary contains overlapping face producers")
            face_rows[face] = condition
            depth = max(depth, condition.requirement.ghost_depth)
            dependencies = condition.provider.dependencies
            flow = dependencies.representation
            if flow.converter is not None or flow.source != flow.target:
                raise NotImplementedError(
                    "native transport boundary lowering requires an authored compiled "
                    "representation converter"
                )
            if condition.condition_type == "inflow":
                if dependencies.states or dependencies.fields or dependencies.time:
                    raise NotImplementedError(
                        "state/field/time-dependent inflow requires a compiled boundary kernel; "
                        "the built-in native provider accepts only constants and RuntimeParams"
                    )
                if len(condition.values) != ncomp:
                    raise ValueError(
                        "native inflow must prescribe exactly %d state components" % ncomp
                    )
                for expression in condition.values:
                    if _expression_data(expression).get("protocol") != "pops.expr.key.v1":
                        raise NotImplementedError("unsupported boundary expression protocol")
        if any(row is None for row in face_rows):
            raise ValueError("native transport boundary has incomplete physical-face coverage")
        return state, ncomp, tuple(row for row in face_rows if row is not None), depth

    def compile_boundary_data(self) -> dict[str, Any]:
        """Return deterministic evidence that the authority has a total native lowering.

        RuntimeParam values intentionally remain unbound here.  Their expression protocol and
        dependency set are authenticated now; numeric evaluation happens exactly once at bind.
        """
        state, ncomp, conditions, depth = self._native_contract()
        return {
            "schema_version": 1,
            "authority_type": "prepared_boundary_plan_compile",
            "source_plan": self.plan.canonical_id,
            "state": state.canonical_identity(),
            "ncomp": ncomp,
            "required_depth": depth,
            "faces": [
                {
                    "ordinal": 2 * row.geometry.axis.index + (
                        0 if row.geometry.side.value == "lower" else 1),
                    "condition_type": row.condition_type,
                    "producer": row.provider.qualified_id,
                    "geometry": row.geometry.canonical_identity(),
                    "type": ("foextrap" if row.condition_type == "outflow"
                             else "dirichlet"),
                    "values": (
                        [] if row.condition_type == "outflow" else
                        [_expression_data(expression)["value"]
                         for expression in row.values]
                    ),
                }
                for row in conditions
            ],
        }

    def runtime_boundary_data(self, params: Any) -> dict[str, Any]:
        """Lower this resolved authority to the executable native v1 transport contract.

        The built-in provider intentionally supports only data that can be executed without a
        Python callback: outflow and scalar expressions closed over BindSchema parameters.  A
        state/field/time-dependent inflow needs a compiled boundary kernel and therefore fails here
        instead of being retained as ignored metadata.
        """
        from pops.model._bind_expression import eval_expression_key

        if not isinstance(params, Mapping):
            raise TypeError("runtime boundary lowering requires resolved BindSchema values")
        env: dict[str, Any] = {}
        local_counts: dict[str, int] = {}
        for handle, value in params.items():
            if not isinstance(handle, ParamHandle) or not handle.is_resolved:
                raise TypeError("runtime boundary parameters must use canonical ParamHandle keys")
            env[handle.qualified_id] = value
            local_counts[handle.local_id] = local_counts.get(handle.local_id, 0) + 1
        for handle, value in params.items():
            if local_counts[handle.local_id] == 1:
                env[handle.local_id] = value

        state, ncomp, conditions, depth = self._native_contract()
        face_rows: list[dict[str, Any] | None] = [None, None, None, None]
        for condition in conditions:
            geometry = condition.geometry
            face = 2 * geometry.axis.index + (0 if geometry.side.value == "lower" else 1)
            if condition.condition_type == "outflow":
                values = [0.0] * ncomp
                face_type = "foextrap"
            else:
                values = []
                for index, expression in enumerate(condition.values):
                    data = _expression_data(expression)
                    value = eval_expression_key(
                        data["value"], env,
                        where="transport boundary %s component %d" % (geometry.name, index),
                    )
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError(
                            "transport boundary values must lower to real scalars, got %r" % value
                        )
                    values.append(float(value))
                face_type = "dirichlet"
            face_rows[face] = {
                "ordinal": face,
                "geometry": geometry.canonical_identity(),
                "producer": condition.provider.qualified_id,
                "type": face_type,
                "values": values,
            }
        rows = tuple(row for row in face_rows if row is not None)
        evidence = {
            "schema_version": 1,
            "authority_type": "prepared_boundary_plan",
            "source_plan": self.plan.canonical_id,
            "state": state.canonical_identity(),
            "required_depth": depth,
            "faces": list(rows),
            # Dimension-split FV reconstruction never reads diagonal corner ghosts.  A future
            # multidimensional stencil must set this from its stencil manifest and supply a resolver.
            "corner_required": False,
            "residual_contributions": [],
            "linearization_contributions": [],
            "interfaces": [],
        }
        evidence["identity"] = make_identity(
            "prepared-boundary-plan", semantic_value(
                evidence, where="prepared transport boundary plan")
        ).token
        return evidence

    def amr_boundary_requirement(self, *, owner: Any, dimension: int) -> Any:
        """Project exact ghost-fill needs through the AMR nesting extension protocol."""
        from pops.mesh._amr import NestingRequirementSource

        if isinstance(dimension, bool) or dimension not in (1, 2, 3):
            raise ValueError("AMR boundary dimension must be 1, 2, or 3")
        depth = max(row.requirement.ghost_depth for row in self.conditions)
        lookahead = max(
            max(row.requirement.formal_orders) - 1 for row in self.conditions
        )
        evidence = {
            "boundary": self.canonical_identity(),
            "dimension": dimension,
            "ghost_depth": depth,
            "lookahead": lookahead,
        }
        provider = Handle(
            "boundary_%s" % make_identity(
                "amr-boundary-requirement",
                semantic_value(evidence, where="AMR boundary requirement"),
            ).token,
            kind="amr_boundary_requirement",
            owner=OwnerPath.coerce(owner).canonical(),
        )
        return NestingRequirementSource(provider, (depth,) * dimension, lookahead)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class TransportBoundarySet:
    """One exact physical-boundary authority for all FV states of a block.

    A mapping value may be one condition or a tuple of conditions, which keeps the same API usable
    for systems evolving several independent states.  Coverage is checked only after numerical
    resolution, when the frame and every evolved state are known exactly.
    """

    entries: tuple[tuple[DomainBoundary, tuple[Any, ...]], ...]

    def __init__(self, bindings: Any) -> None:
        if not isinstance(bindings, Mapping) or not bindings:
            raise TypeError("TransportBoundarySet requires a non-empty boundary mapping")
        rows = []
        for boundary, raw_conditions in bindings.items():
            if not isinstance(boundary, DomainBoundary):
                raise TypeError(
                    "TransportBoundarySet keys must be typed DomainBoundary values, never names")
            conditions = raw_conditions if isinstance(raw_conditions, tuple) else (raw_conditions,)
            if not conditions:
                raise ValueError("every transport boundary must declare at least one condition")
            checked = tuple(
                _condition_protocol(value, where="TransportBoundarySet[%s]" % boundary.name)
                for value in conditions
            )
            states = [condition.state for condition in checked]
            if len(states) != len(set(states)):
                raise ValueError(
                    "transport boundary %r declares a state more than once" % boundary.name)
            rows.append((boundary, checked))
        geometry_ids = {boundary.domain_geometry_id for boundary, _ in rows}
        if len(geometry_ids) != 1:
            raise ValueError("TransportBoundarySet cannot mix boundaries from several domains")
        orientations = {(boundary.axis.index, boundary.side.value) for boundary, _ in rows}
        if len(orientations) != len(rows):
            raise ValueError("TransportBoundarySet contains duplicate geometric orientations")
        object.__setattr__(self, "entries", tuple(sorted(
            rows, key=lambda row: row[0].canonical_id)))

    def inspect(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "authority_type": "transport_boundary_set_authoring",
            "bindings": [
                {
                    "boundary": boundary.canonical_identity(),
                    "conditions": [condition.inspect() for condition in conditions],
                }
                for boundary, conditions in self.entries
            ],
        }

    @staticmethod
    def _requirements(context: Any) -> dict[Handle, BoundaryStencilRequirement]:
        accumulated: dict[Handle, dict[str, Any]] = {}
        for row in context.rates:
            state = row.method.variables.options.get("state")
            _state(state, where="FiniteVolume.variables state")
            if not state.is_resolved:
                raise TypeError("resolved FiniteVolume variables retain an authoring state")
            record = accumulated.setdefault(
                state, {"ghost_depth": [], "formal_orders": [], "rates": []})
            record["ghost_depth"].append(row.method.ghost_depth)
            record["formal_orders"].append(row.method.formal_order)
            record["rates"].append(row.rate.qualified_id)
        return {
            state: BoundaryStencilRequirement(
                state=state,
                ghost_depth=max(record["ghost_depth"]),
                formal_orders=tuple(sorted(set(record["formal_orders"]))),
                rates=tuple(sorted(set(record["rates"]))),
            )
            for state, record in accumulated.items()
        }

    def resolve_for_numerics(self, context: Any) -> ResolvedTransportBoundarySet:
        from pops.domain import BoundarySide as DomainBoundarySide
        from pops.mesh.boundaries import (
            BoundaryHandle,
            BoundaryOrientation,
            BoundaryProviderRegistry,
            BoundarySide,
            BoundaryTopology,
        )

        for attribute in ("owner", "block", "frame", "rates", "resolve"):
            if not hasattr(context, attribute):
                raise TypeError(
                    "transport boundary resolution context is missing %r" % attribute)
        if not callable(context.resolve):
            raise TypeError("transport boundary context resolver must be callable")
        frame_boundaries = getattr(context.frame, "boundaries", None)
        expected = getattr(frame_boundaries, "all", None)
        if not isinstance(expected, tuple) or not expected \
                or any(not isinstance(row, DomainBoundary) for row in expected):
            raise TypeError(
                "TransportBoundarySet requires a frame exposing typed boundaries.all")
        authored = tuple(boundary for boundary, _ in self.entries)
        missing = set(expected) - set(authored)
        extra = set(authored) - set(expected)
        if missing or extra:
            raise ValueError(
                "transport boundary geometry coverage mismatch: missing=%s extra=%s"
                % (sorted(row.name for row in missing), sorted(row.name for row in extra))
            )

        low_level = {}
        for geometry in expected:
            side = (
                BoundarySide.LOWER
                if geometry.side is DomainBoundarySide.LOWER
                else BoundarySide.UPPER
            )
            low_level[geometry] = BoundaryHandle(
                "%s@%s" % (geometry.name, geometry.domain_geometry_id),
                owner=context.owner,
                orientation=BoundaryOrientation(geometry.axis.index, side),
            )
        topology = BoundaryTopology(
            owner=context.owner,
            boundaries=tuple(low_level.values()),
            periodic=(),
            physical=tuple(low_level.values()),
        )
        requirements = self._requirements(context)
        resolved_conditions = []
        covered = set()
        for geometry, conditions in self.entries:
            for condition in conditions:
                resolved = condition.resolve_references(context.resolve)
                _condition_protocol(
                    resolved, where="resolved TransportBoundarySet[%s]" % geometry.name)
                requirement = requirements.get(resolved.state)
                if requirement is None:
                    raise ValueError(
                        "transport condition for %s refers to state %s that no resolved rate evolves"
                        % (geometry.name, resolved.state.qualified_id)
                    )
                key = (geometry, resolved.state)
                if key in covered:
                    raise ValueError("resolved transport boundary contains duplicate state coverage")
                covered.add(key)
                resolved_conditions.append(resolved.resolve_condition(
                    geometry=geometry,
                    boundary=low_level[geometry],
                    requirement=requirement,
                ))
        expected_coverage = {
            (geometry, state) for geometry in expected for state in requirements
        }
        missing_coverage = expected_coverage - covered
        extra_coverage = covered - expected_coverage
        if missing_coverage or extra_coverage:
            def labels(rows: Any) -> list[str]:
                return sorted("%s:%s" % (geometry.name, state.qualified_id)
                              for geometry, state in rows)

            raise ValueError(
                "transport state coverage mismatch: missing=%s extra=%s"
                % (labels(missing_coverage), labels(extra_coverage))
            )
        resolved_conditions.sort(key=lambda row: (
            row.geometry.canonical_id, row.state.qualified_id))
        providers = tuple(row.provider for row in resolved_conditions)
        needs = tuple(row.provider.outputs[0] for row in resolved_conditions)
        plan = BoundaryProviderRegistry(*providers).resolve(topology, needs)
        return ResolvedTransportBoundarySet(
            domain_geometry_id=expected[0].domain_geometry_id,
            conditions=tuple(resolved_conditions),
            plan=plan,
        )


__all__ = [
    "BoundaryStencilRequirement",
    "Inflow",
    "Outflow",
    "ResolvedTransportBoundarySet",
    "ResolvedTransportCondition",
    "TransportBoundarySet",
]
