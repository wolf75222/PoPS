"""Typed transport-boundary authoring and exact low-level port resolution."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, ClassVar, cast

from pops.analytic import ScalarExpr
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


def _expression_data(value: Expr | ScalarExpr, *, qualified: bool = False) -> Any:
    """Return the same stable structural protocol used by derived parameter expressions."""
    if isinstance(value, ScalarExpr):
        return {
            "protocol": "pops.analytic.scalar.v1",
            "value": value.to_data(),
        }
    if qualified:
        from pops.model._bind_expression import qualified_expression_key

        key = qualified_expression_key(
            value, where="resolved transport boundary expression")
    else:
        key = _key(value)
    return {
        "protocol": "pops.expr.key.v1",
        "value": json.loads(json.dumps(
            key, sort_keys=True, separators=(",", ":"), allow_nan=False)),
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


def _state_components(state: Handle, *, where: str) -> tuple[Any, ...]:
    """Return the authenticated state-space component manifest."""
    space = getattr(state, "space", None)
    raw = getattr(space, "components", None)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise TypeError("%s state has no iterable component manifest" % where)
    return tuple(raw)


def _converter(value: Any) -> Handle | None:
    if value is None:
        return None
    if not isinstance(value, Handle) or value.kind != "representation_conversion":
        raise TypeError("boundary converter must be a representation_conversion Handle or None")
    return value


def model_primitive_to_conservative(state: Any) -> Handle:
    """Return the exact block-model primitive-to-conservative boundary provider.

    The returned Handle names the already compiled ``Model.to_conservative`` kernel; it is not a
    Python callback and it cannot select an unrelated conversion implementation by string. The
    corresponding ``Inflow.value`` tuple follows the model's declared primitive-variable order.
    """
    checked = _state(state, where="model_primitive_to_conservative.state")
    representation = getattr(getattr(checked, "space", None), "representation", None)
    if representation != "conservative":
        raise ValueError(
            "model_primitive_to_conservative requires a conservative target state"
        )
    digest = hashlib.sha256(checked.qualified_id.encode("utf-8")).hexdigest()[:24]
    return Handle(
        "model-primitive-to-conservative-%s" % digest,
        kind="representation_conversion",
        owner=checked.owner_path,
    )


def model_characteristic_no_inflow(state: Any) -> Handle:
    """Return the exact block-model flux-Jacobian characteristic provider.

    The provider is generated only for models compiled with
    ``m.roe_from_jacobian()``.  It projects the authored conservative reference state onto the
    incoming eigenspace of the outward-normal flux Jacobian.  The returned Handle is data-only and
    block-qualified; it never names a Python callback or an Euler-specific implementation.
    """
    checked = _state(state, where="model_characteristic_no_inflow.state")
    if getattr(getattr(checked, "space", None), "representation", None) != "conservative":
        raise ValueError(
            "model_characteristic_no_inflow requires a conservative target state"
        )
    digest = hashlib.sha256(checked.qualified_id.encode("utf-8")).hexdigest()[:24]
    return Handle(
        "model-characteristic-no-inflow-%s" % digest,
        kind="boundary_eigenstructure",
        owner=checked.owner_path,
    )


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


def _analytic_time_handle(clock: Any) -> Handle:
    from pops.time import Clock

    if type(clock) is not Clock or clock.owner is None:
        raise TypeError("analytic boundary time requires one owner-qualified exact Clock")
    digest = hashlib.sha256(clock.qualified_id.encode("utf-8")).hexdigest()[:24]
    return Handle(
        "clock-%s" % digest,
        kind="time",
        owner=clock.owner,
    )


def _dependency_handles(
    values: tuple[Expr | ScalarExpr, ...], *, include_state: Handle | None = None
) -> tuple[tuple[Handle, ...], tuple[Handle, ...], tuple[Handle, ...], tuple[ParamHandle, ...]]:
    references = _unique_references(
        *(
            value.declaration_references() if isinstance(value, Expr) else value.parameter_handles()
            for value in values
        )
    )
    states = [reference for reference in references if reference.kind == "state"]
    fields = [reference for reference in references if reference.kind == "field"]
    time = [reference for reference in references if reference.kind == "time"]
    for value in values:
        if isinstance(value, ScalarExpr):
            for clock in value.time_clocks():
                handle = _analytic_time_handle(clock)
                if handle not in time:
                    time.append(handle)
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


def _closure(characteristic: Handle | None = None) -> Any:
    from pops.mesh.boundaries import (
        CharacteristicClosure,
        ClosureMode,
        IncomingMultiplicity,
        SignDependence,
        SonicPolicy,
    )

    if characteristic is None:
        return CharacteristicClosure(
            mode=ClosureMode.NONE,
            sign_dependence=SignDependence.FIXED,
            sonic=SonicPolicy.NEUTRAL,
            incoming=IncomingMultiplicity.SINGLE,
            characteristics=(),
        )
    return CharacteristicClosure(
        mode=ClosureMode.DIRECTIONAL,
        sign_dependence=SignDependence.SPATIAL,
        sonic=SonicPolicy.NEUTRAL,
        incoming=IncomingMultiplicity.MULTIPLE,
        characteristics=(characteristic,),
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
    values: tuple[Expr | ScalarExpr, ...]
    requirement: BoundaryStencilRequirement
    provider: Any

    def __post_init__(self) -> None:
        from pops.mesh.boundaries import BoundaryProvider, BoundaryProviderKind

        if not isinstance(self.geometry, DomainBoundary):
            raise TypeError("ResolvedTransportCondition.geometry must be a DomainBoundary")
        if self.condition_type not in {"inflow", "outflow", "no_flux", "slip_wall"}:
            raise ValueError("unsupported built-in transport condition type")
        _state(self.state, where="ResolvedTransportCondition.state")
        if not self.state.is_resolved:
            raise TypeError("ResolvedTransportCondition.state must be canonical")
        if not isinstance(self.values, tuple) \
                or any(not isinstance(row, (Expr, ScalarExpr)) for row in self.values):
            raise TypeError("ResolvedTransportCondition.values must contain Expr or ScalarExpr values")
        if self.requirement.state != self.state:
            raise ValueError("transport condition and stencil requirement refer to different states")
        if not isinstance(self.provider, BoundaryProvider):
            raise TypeError("ResolvedTransportCondition.provider must be a BoundaryProvider")
        allowed_kinds = {
            "inflow": frozenset((
                BoundaryProviderKind.INFLOW,
                BoundaryProviderKind.DIRECTIONAL_TRANSPORT,
            )),
            "outflow": frozenset((
                BoundaryProviderKind.OUTFLOW,
                BoundaryProviderKind.DIRECTIONAL_TRANSPORT,
            )),
            "no_flux": frozenset((BoundaryProviderKind.NO_FLUX,)),
            "slip_wall": frozenset((BoundaryProviderKind.GHOST_FORMULA,)),
        }[self.condition_type]
        if self.provider.kind not in allowed_kinds:
            raise ValueError(
                "transport condition %r cannot use boundary provider law %r"
                % (self.condition_type, self.provider.kind.value)
            )

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "condition_type": self.condition_type,
            "geometry": self.geometry.canonical_identity(),
            "state": self.state.canonical_identity(),
            "values": [_expression_data(value, qualified=True) for value in self.values],
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
        GhostFormula,
        GhostState,
        Inflow as LowLevelInflow,
        NoFlux as LowLevelNoFlux,
        NumericalFlux,
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
    characteristic = getattr(condition, "characteristic", None)
    if characteristic is not None and state not in states:
        states = (*states, state)
    dependencies = BoundaryDependencies(
        states=states,
        fields=fields,
        time=time,
        runtime_params=params,
        representation=flow,
        characteristic=_closure(characteristic),
    )
    output = (
        NumericalFlux(boundary=boundary, subject=state, representation=target)
        if condition_type == "no_flux"
        else GhostState(boundary=boundary, subject=state, representation=target)
    )
    handle = _provider_handle(state, geometry, condition_type)
    if condition_type == "no_flux":
        if not isinstance(output, NumericalFlux):
            raise TypeError("no_flux transport must emit a NumericalFlux port")
        provider = LowLevelNoFlux(
            handle=handle,
            output=output,
            dependencies=dependencies,
        )
    else:
        if not isinstance(output, GhostState):
            raise TypeError("%s transport must emit a GhostState port" % condition_type)
        outputs = (output,)
        if characteristic is not None:
            if condition_type != "inflow":
                raise ValueError("characteristic no-inflow is defined only for Inflow")
            from pops.mesh.boundaries import DirectionalTransport

            provider = DirectionalTransport(
                handle=handle,
                outputs=outputs,
                dependencies=dependencies,
            )
        elif condition_type == "inflow":
            provider = LowLevelInflow(
                handle=handle,
                outputs=outputs,
                dependencies=dependencies,
            )
        elif condition_type == "outflow":
            provider = LowLevelOutflow(
                handle=handle,
                outputs=outputs,
                dependencies=dependencies,
            )
        elif condition_type == "slip_wall":
            provider = GhostFormula(
                handle=handle,
                outputs=outputs,
                dependencies=dependencies,
            )
        else:
            raise KeyError(condition_type)
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
    values: tuple[Expr | ScalarExpr, ...]
    representation: Representation | None
    converter: Handle | None
    characteristic: Handle | None

    def __init__(
        self,
        *,
        state: Any,
        value: Any,
        representation: Representation | None = None,
        converter: Any = None,
        characteristic: Any = None,
    ) -> None:
        checked_state = _state(state, where="Inflow.state")
        if representation is not None and not isinstance(representation, Representation):
            raise TypeError("Inflow.representation must be a typed Representation or None")
        raw_values = value if isinstance(value, tuple) else (value,)
        if not raw_values:
            raise ValueError("Inflow.value must prescribe at least one state component")
        analytic = any(isinstance(row, ScalarExpr) for row in raw_values)
        if analytic:
            from pops.analytic import constant as analytic_constant

            checked_values = []
            for index, row in enumerate(raw_values):
                if isinstance(row, ScalarExpr):
                    checked_values.append(row)
                elif isinstance(row, Expr):
                    raise TypeError(
                        "Inflow.value cannot mix PoPS Expr and analytic ScalarExpr values; "
                        "use pops.analytic.param(...) for parameters"
                    )
                else:
                    try:
                        checked_values.append(analytic_constant(row))
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            "Inflow.value[%d] must be an analytic ScalarExpr or exact scalar"
                            % index
                        ) from exc
        else:
            checked_values = [
                _expression(row, where="Inflow.value[%d]" % index)
                for index, row in enumerate(raw_values)
            ]
        object.__setattr__(self, "state", checked_state)
        object.__setattr__(self, "values", tuple(checked_values))
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "converter", _converter(converter))
        if characteristic is not None:
            expected = model_characteristic_no_inflow(checked_state)
            if not isinstance(characteristic, Handle) or characteristic != expected:
                raise ValueError(
                    "Inflow.characteristic must be the exact "
                    "model_characteristic_no_inflow(state) provider"
                )
            if representation is not None or converter is not None:
                raise NotImplementedError(
                    "characteristic no-inflow currently requires a conservative reference state"
                )
            if analytic:
                raise NotImplementedError(
                    "characteristic no-inflow requires one finite fixed conservative reference"
                )
        object.__setattr__(self, "characteristic", characteristic)

    def declaration_references(self) -> tuple[Handle, ...]:
        converter = () if self.converter is None else (self.converter,)
        characteristic = () if self.characteristic is None else (self.characteristic,)
        return _unique_references(
            (self.state,),
            *(
                value.declaration_references()
                if isinstance(value, Expr)
                else value.parameter_handles()
                for value in self.values
            ),
            converter,
            characteristic,
        )

    def resolve_references(self, resolver: Any) -> Inflow:
        if not callable(resolver):
            raise TypeError("Inflow.resolve_references requires a callable resolver")
        resolved_state = resolver(self.state)
        if self.converter is None:
            converter = None
        elif self.converter == model_primitive_to_conservative(self.state):
            # This provider is derived from the authenticated state, not an independently
            # registered declaration. Re-derive its canonical identity after resolving the state.
            converter = model_primitive_to_conservative(resolved_state)
        else:
            converter = resolver(self.converter)
        characteristic = None
        if self.characteristic is not None:
            if self.characteristic != model_characteristic_no_inflow(self.state):
                raise ValueError("Inflow retained a forged characteristic provider")
            characteristic = model_characteristic_no_inflow(resolved_state)
        return type(self)(
            state=resolved_state,
            value=tuple(value.resolve_references(resolver) for value in self.values),
            representation=self.representation,
            converter=converter,
            characteristic=characteristic,
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
            "characteristic": (
                None if self.characteristic is None else self.characteristic.inspect()),
        }

    def resolve_condition(
        self,
        *,
        geometry: DomainBoundary,
        boundary: Any,
        requirement: BoundaryStencilRequirement,
    ) -> ResolvedTransportCondition:
        components = _state_components(self.state, where="Inflow")
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


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NoFlux:
    """Close one physical face after the Riemann solve.

    Ghost values use the prepared extrapolation law so reconstruction remains defined; the same
    immutable face row then zeroes the already evaluated numerical flux before divergence/reflux.
    """

    condition_type: ClassVar[str] = "no_flux"
    state: Handle
    values: tuple[Expr, ...]
    representation: Representation | None
    converter: Handle | None

    def __init__(self, *, state: Any) -> None:
        object.__setattr__(self, "state", _state(state, where="NoFlux.state"))
        object.__setattr__(self, "values", ())
        object.__setattr__(self, "representation", None)
        object.__setattr__(self, "converter", None)

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.state,)

    def resolve_references(self, resolver: Any) -> NoFlux:
        if not callable(resolver):
            raise TypeError("NoFlux.resolve_references requires a callable resolver")
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


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SlipWall:
    """Model-aware reflective wall: reverse the normal polar-vector component only."""

    condition_type: ClassVar[str] = "slip_wall"
    state: Handle
    values: tuple[Expr, ...]
    representation: Representation | None
    converter: Handle | None

    def __init__(self, *, state: Any) -> None:
        object.__setattr__(self, "state", _state(state, where="SlipWall.state"))
        object.__setattr__(self, "values", ())
        object.__setattr__(self, "representation", None)
        object.__setattr__(self, "converter", None)

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self.state,)

    def resolve_references(self, resolver: Any) -> SlipWall:
        if not callable(resolver):
            raise TypeError("SlipWall.resolve_references requires a callable resolver")
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
        from pops.physics.roles import ComponentRole, native_role_token

        components = _state_components(self.state, where="SlipWall")
        space = getattr(self.state, "space", None)
        roles = getattr(space, "roles", None)
        if not isinstance(roles, Mapping) or set(roles) != set(components):
            raise ValueError(
                "SlipWall requires one explicit typed physical role for every state component")
        tokens = {
            component: (
                native_role_token(role) if isinstance(role, ComponentRole) else role)
            for component, role in roles.items()
        }
        scalar = {"density", "energy", "pressure", "temperature", "scalar", "custom"}

        def valid_semantic(token: Any) -> bool:
            if not isinstance(token, str):
                return False
            family, separator, axis = token.partition(":")
            return token in scalar or (
                family in {"momentum", "velocity", "axial"}
                and separator == ":" and axis.isdecimal())

        if any(not valid_semantic(token) for token in tokens.values()):
            raise ValueError(
                "SlipWall requires one explicit typed physical role for every state component")
        normal_token = "momentum:%d" % geometry.axis.index
        normal_velocity = "velocity:%d" % geometry.axis.index
        normal = [
            component
            for component, token in tokens.items()
            if token in {normal_token, normal_velocity}
        ]
        if not normal:
            raise ValueError(
                "SlipWall on %s requires a declared normal polar-vector component"
                % geometry.name
            )
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
    frame_id: str
    conditions: tuple[ResolvedTransportCondition, ...]
    plan: Any

    def __post_init__(self) -> None:
        from pops.mesh.boundaries import ResolvedBoundaryPlan

        if not isinstance(self.domain_geometry_id, str) or not self.domain_geometry_id:
            raise TypeError("resolved transport domain identity must be non-empty text")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise TypeError("resolved transport frame identity must be non-empty text")
        if not isinstance(self.conditions, tuple) or not self.conditions \
                or any(not isinstance(row, ResolvedTransportCondition)
                       for row in self.conditions):
            raise TypeError("resolved transport conditions must be a non-empty tuple")
        if not isinstance(self.plan, ResolvedBoundaryPlan):
            raise TypeError("resolved transport plan must be a ResolvedBoundaryPlan")
        # Resolution is the public acceptance boundary for a transport descriptor.  Reusing the
        # executable contract here prevents a characteristic, representation, analytic, or
        # multi-state descriptor from surviving as inert metadata and failing only later during
        # compile/bind.  compile_boundary_data() and runtime_boundary_data() intentionally call the
        # same pure validator again so detached/tampered resolved values remain fail-closed.
        self._native_contract()

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "authority_type": "transport_boundary_set",
            "domain_geometry_id": self.domain_geometry_id,
            "frame_id": self.frame_id,
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

    def _native_contract(
        self,
    ) -> tuple[Handle, int, tuple[ResolvedTransportCondition, ...], int, int]:
        """Validate the complete executable shape of the built-in native provider.

        This is the sole acceptance contract used at numerical resolution, compile, and bind.
        """
        from pops.mesh.boundaries import (
            ClosureMode,
            IncomingMultiplicity,
            SignDependence,
            SonicPolicy,
        )

        states = {row.state for row in self.conditions}
        if len(states) != 1:
            raise NotImplementedError(
                "the installed native block provider requires one state per boundary plan"
            )
        state = next(iter(states))
        components = _state_components(state, where="resolved transport boundary")
        if not components:
            raise TypeError("resolved transport boundary state has no component manifest")
        ncomp = len(components)
        boundaries = self.plan.topology.boundaries
        if len(boundaries) not in (2, 4, 6):
            raise ValueError(
                "native transport topology must contain exactly 2*Dim Cartesian faces")
        dimension = len(boundaries) // 2
        axes = [row.geometry.axis.index for row in self.conditions]
        if any(isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2)
               for axis in axes):
            raise NotImplementedError(
                "the installed native transport boundary provider supports dimensions 1, 2, and 3"
            )
        face_rows: list[ResolvedTransportCondition | None] = [None] * (2 * dimension)
        analytic_plan_clocks: set[str] = set()
        has_analytic = False
        depth = 0
        for condition in self.conditions:
            geometry = condition.geometry
            face = 2 * geometry.axis.index + (0 if geometry.side.value == "lower" else 1)
            if face_rows[face] is not None:
                raise ValueError("native transport boundary contains overlapping face producers")
            face_rows[face] = condition
            depth = max(depth, condition.requirement.ghost_depth)
            dependencies = condition.provider.dependencies
            characteristic = dependencies.characteristic
            if characteristic.mode is not ClosureMode.NONE:
                if getattr(geometry.axis, "name", None) not in ("x", "y", "z"):
                    raise NotImplementedError(
                        "characteristic no-inflow requires Cartesian x/y/z faces; polar "
                        "geometry has no prepared metric ABI"
                    )
                expected = model_characteristic_no_inflow(state)
                exact_no_inflow = (
                    condition.condition_type == "inflow"
                    and characteristic.mode is ClosureMode.DIRECTIONAL
                    and characteristic.sign_dependence is SignDependence.SPATIAL
                    and characteristic.sonic is SonicPolicy.NEUTRAL
                    and characteristic.incoming is IncomingMultiplicity.MULTIPLE
                    and characteristic.characteristics == (expected,)
                    and dependencies.states == (state,)
                    and not dependencies.fields
                    and not dependencies.time
                )
                if not exact_no_inflow:
                    raise NotImplementedError(
                        "native characteristic boundary requires prepared model eigenstructure "
                        "through the exact "
                        "model_characteristic_no_inflow(state) contract; directional modes "
                        "cannot fall back to component-wise ghost filling"
                    )
            representation, _ = self._native_representation_contract(condition, state)
            if condition.condition_type == "inflow":
                if len(condition.values) != ncomp:
                    raise ValueError(
                        "native inflow must prescribe exactly %d state components" % ncomp
                    )
                analytic = all(isinstance(row, ScalarExpr) for row in condition.values)
                if analytic:
                    has_analytic = True
                    analytic_expressions = tuple(
                        cast(ScalarExpr, expression) for expression in condition.values
                    )
                    if representation not in {"conservative", "primitive"}:
                        raise NotImplementedError(
                            "analytic inflow requires conservative values or "
                            "model_primitive_to_conservative(state)"
                        )
                    if dependencies.fields:
                        raise NotImplementedError(
                            "analytic inflow cannot read discrete field storage; interior "
                            "conservative state is the typed Input-slot ABI"
                        )
                    clocks = {
                        clock.qualified_id
                        for expression in analytic_expressions
                        for clock in expression.time_clocks()
                    }
                    if len(clocks) > 1:
                        raise ValueError(
                            "one analytic inflow face cannot mix several logical Clocks"
                        )
                    analytic_plan_clocks.update(clocks)
                    for expression in analytic_expressions:
                        if expression.frame_id not in (None, self.frame_id):
                            raise ValueError("analytic inflow coordinate belongs to another frame")
                        if expression.input_references():
                            raise NotImplementedError(
                                "analytic inflow cannot read setup-program discrete inputs"
                            )
                else:
                    if any(isinstance(row, ScalarExpr) for row in condition.values):
                        raise TypeError(
                            "native inflow values must use one expression protocol per face"
                        )
                    if (
                        dependencies.characteristic.mode is ClosureMode.NONE
                        and (dependencies.states or dependencies.fields or dependencies.time)
                    ):
                        raise NotImplementedError(
                            "state/field/time-dependent PoPS Expr inflow requires a compiled "
                            "boundary component"
                        )
                    for expression in condition.values:
                        if (
                            _expression_data(expression, qualified=True).get("protocol")
                            != "pops.expr.key.v1"
                        ):
                            raise NotImplementedError("unsupported boundary expression protocol")
        if any(row is None for row in face_rows):
            raise ValueError("native transport boundary has incomplete physical-face coverage")
        if len(analytic_plan_clocks) > 1:
            raise ValueError("one prepared analytic boundary plan cannot mix several logical Clocks")
        if has_analytic:
            for identification in getattr(self.plan.topology, "periodic", ()):
                orientation = getattr(identification, "orientation", None)
                permutation = getattr(orientation, "permutation", ())
                signs = getattr(orientation, "signs", ())
                dimension_axes = tuple(range(len(permutation)))
                if permutation != dimension_axes or any(sign != 1 for sign in signs):
                    raise NotImplementedError(
                        "axis-permuted periodic coordinates require a prepared coordinate map"
                    )
        return state, ncomp, tuple(row for row in face_rows if row is not None), depth, dimension

    @staticmethod
    def _native_representation_contract(
        condition: ResolvedTransportCondition,
        state: Handle,
    ) -> tuple[str, str | None]:
        flow = condition.provider.dependencies.representation
        target_name = getattr(getattr(state, "space", None), "representation", None)
        if target_name != "conservative":
            raise NotImplementedError(
                "native transport boundaries require a conservative target state")
        target = _representation_handle(state, target_name)
        if flow.source == target and flow.target == target and flow.converter is None:
            return "conservative", None
        primitive = _representation_handle(state, "primitive")
        expected_converter = model_primitive_to_conservative(state)
        if (
            flow.source == primitive
            and flow.target == target
            and flow.converter == expected_converter
        ):
            if condition.condition_type != "inflow":
                raise NotImplementedError(
                    "model primitive-to-conservative boundary conversion is defined only for "
                    "fixed-state inflow data"
                )
            return "primitive", expected_converter.qualified_id
        raise NotImplementedError(
            "native transport boundary representation conversion requires the exact "
            "model_primitive_to_conservative(state) provider"
        )

    def compile_boundary_data(self) -> dict[str, Any]:
        """Return deterministic evidence that the authority has a total native lowering.

        RuntimeParam values intentionally remain unbound here.  Their expression protocol and
        dependency set are authenticated now; numeric evaluation happens exactly once at bind.
        """
        from pops.mesh.boundaries import ClosureMode

        state, ncomp, conditions, depth, _ = self._native_contract()
        return {
            "schema_version": 1,
            "authority_type": "prepared_boundary_plan_compile",
            "source_plan": self.plan.canonical_id,
            "frame_id": self.frame_id,
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
                    "type": (
                        "characteristic_no_inflow"
                        if row.provider.dependencies.characteristic.mode
                        is not ClosureMode.NONE
                        else {
                        "outflow": "foextrap",
                        "inflow": "dirichlet",
                        "no_flux": "no_flux",
                        "slip_wall": "slip_wall",
                        }[row.condition_type]
                    ),
                    "representation": self._native_representation_contract(
                        row, state)[0],
                    "converter": self._native_representation_contract(
                        row, state)[1],
                    "values": (
                        []
                        if row.condition_type in {"no_flux", "outflow"}
                        else [
                            (
                                _expression_data(expression, qualified=True)
                                if isinstance(expression, ScalarExpr)
                                else _expression_data(expression, qualified=True)["value"]
                            )
                            for expression in row.values
                        ]
                    ),
                }
                for row in conditions
            ],
        }

    def runtime_boundary_data(self, params: Any) -> dict[str, Any]:
        """Lower this resolved authority to the executable native v1 transport contract.

        The built-in provider intentionally supports only data that can be executed without a
        Python callback: outflow, scalar expressions closed over BindSchema parameters, and
        conservative or model-primitive analytic ``(coordinates, t, interior state, params)``
        inflow programs. Setup-program discrete inputs still fail here; Polar/EB remain refused.
        """
        from pops.model._bind_expression import eval_expression_key
        from pops.mesh.boundaries import ClosureMode
        from pops.runtime._analytic_expression_lowering import lower_analytic_components

        if not isinstance(params, Mapping):
            raise TypeError("runtime boundary lowering requires resolved BindSchema values")
        env: dict[str, Any] = {}
        for handle, value in params.items():
            if not isinstance(handle, ParamHandle) or not handle.is_resolved:
                raise TypeError("runtime boundary parameters must use canonical ParamHandle keys")
            env[handle.qualified_id] = value

        state, ncomp, conditions, depth, dimension = self._native_contract()
        face_rows: list[dict[str, Any] | None] = [None] * (2 * dimension)
        for condition in conditions:
            geometry = condition.geometry
            face = 2 * geometry.axis.index + (0 if geometry.side.value == "lower" else 1)
            if condition.condition_type in {"no_flux", "outflow", "slip_wall"}:
                values = [0.0] * ncomp
                face_type = {
                    "no_flux": "no_flux",
                    "outflow": "foextrap",
                    "slip_wall": "slip_wall",
                }[condition.condition_type]
            else:
                analytic_values = all(
                    isinstance(expression, ScalarExpr) for expression in condition.values
                )
                if analytic_values:
                    analytic_expressions = tuple(
                        cast(ScalarExpr, expression) for expression in condition.values
                    )
                    clocks = {
                        clock.qualified_id
                        for expression in analytic_expressions
                        for clock in expression.time_clocks()
                    }
                    clock_id = next(iter(clocks), None)
                    lowered = lower_analytic_components(
                        [expression.to_data() for expression in analytic_expressions],
                        frame_id=self.frame_id,
                        bindings=params,
                        time_clock_id=clock_id,
                    )
                    analytic_programs = [
                        {"opcodes": list(opcodes), "literals": list(literals)}
                        for opcodes, literals in lowered
                    ]
                    values = [0.0] * ncomp
                else:
                    clock_id = None
                    analytic_programs = []
                    values = []
                    for index, expression in enumerate(condition.values):
                        data = _expression_data(expression, qualified=True)
                        value = eval_expression_key(
                            data["value"], env,
                            where="transport boundary %s component %d" % (geometry.name, index),
                        )
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            raise TypeError(
                                "transport boundary values must lower to real scalars, got %r"
                                % value
                            )
                        values.append(float(value))
                face_type = (
                    "characteristic_no_inflow"
                    if condition.provider.dependencies.characteristic.mode
                    is not ClosureMode.NONE
                    else "dirichlet"
                )
            if condition.condition_type in {"no_flux", "outflow", "slip_wall"}:
                analytic_programs = []
                clock_id = None
            face_rows[face] = {
                "ordinal": face,
                "geometry": geometry.canonical_identity(),
                "producer": condition.provider.qualified_id,
                "type": face_type,
                "representation": self._native_representation_contract(
                    condition, state)[0],
                "converter": self._native_representation_contract(
                    condition, state)[1],
                "values": values,
                "analytic_programs": analytic_programs,
                "analytic_clock": clock_id,
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
            frame_id=context.frame.canonical_id,
            conditions=tuple(resolved_conditions),
            plan=plan,
        )


__all__ = [
    "BoundaryStencilRequirement",
    "Inflow",
    "model_characteristic_no_inflow",
    "model_primitive_to_conservative",
    "NoFlux",
    "Outflow",
    "ResolvedTransportBoundarySet",
    "ResolvedTransportCondition",
    "SlipWall",
    "TransportBoundarySet",
]
