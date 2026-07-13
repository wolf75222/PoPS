from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from pops.mesh.boundaries import (
    BoundaryDependencies,
    BoundaryHandle,
    BoundaryOrientation,
    BoundaryProvider,
    BoundaryProviderRegistry,
    BoundarySide,
    BoundaryTopology,
    CharacteristicClosure,
    ClosureMode,
    ConstraintResidual,
    DirectionalTransport,
    Dirichlet,
    ExteriorTrace,
    GhostFormula,
    GhostState,
    IncomingMultiplicity,
    Inflow,
    Mixed,
    Neumann,
    NoFlux,
    NumericalFlux,
    Outflow,
    PeriodicIdentification,
    PeriodicOrientation,
    RepresentationFlow,
    SignDependence,
    SonicPolicy,
)
from pops.model import Handle, OwnerKind, OwnerPath, ParamHandle


def _boundaries():
    owner = OwnerPath.case("main")
    return (
        BoundaryHandle("x_min", owner=owner,
                       orientation=BoundaryOrientation(0, BoundarySide.LOWER)),
        BoundaryHandle("x_max", owner=owner,
                       orientation=BoundaryOrientation(0, BoundarySide.UPPER)),
        BoundaryHandle("y_min", owner=owner,
                       orientation=BoundaryOrientation(1, BoundarySide.LOWER)),
        BoundaryHandle("y_max", owner=owner,
                       orientation=BoundaryOrientation(1, BoundarySide.UPPER)),
    )


def _topology():
    x_min, x_max, y_min, y_max = _boundaries()
    periodic = PeriodicIdentification(
        x_min, x_max, PeriodicOrientation((0, 1), (1, 1)))
    return BoundaryTopology(
        OwnerPath.case("main"), (x_min, x_max, y_min, y_max),
        (periodic,), (y_min, y_max,))


def _model_values():
    owner = OwnerPath.model("transport")
    return (
        Handle("U", kind="state", owner=owner),
        Handle("velocity", kind="field", owner=owner),
        Handle("characteristics", kind="field", owner=owner),
    )


def _runtime_values():
    owner = OwnerPath.case("main")
    return (
        Handle("t", kind="time", owner=owner),
        ParamHandle("direction", owner=owner, param_kind="runtime"),
    )


def _representations():
    owner = OwnerPath.shared("pops.representations")
    return (
        Handle("primitive", kind="representation", owner=owner),
        Handle("conservative", kind="representation", owner=owner),
    )


def _none_closure():
    return CharacteristicClosure(
        ClosureMode.NONE, SignDependence.FIXED, SonicPolicy.NEUTRAL,
        IncomingMultiplicity.SINGLE, ())


def _dependencies(representation=None):
    _, conservative = _representations()
    representation = representation or RepresentationFlow(
        conservative, conservative, None)
    return BoundaryDependencies(
        states=(), fields=(), time=(), runtime_params=(), representation=representation,
        characteristic=_none_closure())


def _provider_handle(name):
    return Handle(name, kind="boundary_provider", owner=OwnerPath.case("main"))


def _case_instance(case_name):
    return (OwnerPath.case(case_name)
            .child(OwnerKind.BLOCK, "transport")
            .instance_of(OwnerPath.model("transport")))


def test_boundary_handle_round_trip_and_orientation_are_canonical():
    lower = BoundaryHandle(
        "wall", owner=OwnerPath.case("main"),
        orientation=BoundaryOrientation(0, BoundarySide.LOWER))
    upper = BoundaryHandle(
        "wall", owner=OwnerPath.case("main"),
        orientation=BoundaryOrientation(0, BoundarySide.UPPER))
    assert lower != upper
    assert lower.orientation.outward_sign == -1
    assert upper.orientation.outward_sign == 1
    assert BoundaryHandle.from_canonical_identity(lower.canonical_identity()) == lower
    assert lower.canonical_identity()["orientation"]["side"] == "lower"

    with pytest.raises(TypeError, match="post-resolution"):
        BoundaryHandle(
            "wall", owner=OwnerPath.fresh(OwnerKind.CASE, "main"),
            orientation=lower.orientation)
    with pytest.raises(TypeError, match="never a string"):
        BoundaryHandle("wall", owner="main", orientation=lower.orientation)


def test_topology_serializes_explicit_periodic_identification_and_physical_partition():
    topology = _topology()
    data = topology.canonical_identity()
    assert len(data["periodic"]) == 1
    assert data["periodic"][0]["orientation"] == {
        "schema_version": 1, "permutation": [0, 1], "signs": [1, 1]}
    assert len(data["physical"]) == 2
    assert topology.is_periodic(topology.boundaries[0])
    assert not topology.is_periodic(topology.physical[0])
    assert json.loads(json.dumps(topology.inspect())) == topology.inspect()


def test_topology_fails_loud_on_missing_double_extra_and_periodic_physical():
    x_min, x_max, y_min, y_max = _boundaries()
    periodic = PeriodicIdentification(
        x_min, x_max, PeriodicOrientation((0, 1), (1, 1)))
    with pytest.raises(ValueError, match="missing boundary topology"):
        BoundaryTopology(OwnerPath.case("main"), (x_min, x_max, y_min), (periodic,), ())
    with pytest.raises(ValueError, match="double periodic"):
        BoundaryTopology(
            OwnerPath.case("main"), (x_min, x_max), (periodic, periodic), ())
    with pytest.raises(ValueError, match=r"periodic\+physical"):
        BoundaryTopology(
            OwnerPath.case("main"), (x_min, x_max), (periodic,), (x_min,))
    foreign = BoundaryHandle(
        "z", owner=OwnerPath.case("main"),
        orientation=BoundaryOrientation(0, BoundarySide.LOWER))
    with pytest.raises(ValueError, match="extra topology"):
        BoundaryTopology(
            OwnerPath.case("main"), (x_min, x_max), (periodic,), (foreign,))
    with pytest.raises(ValueError, match="axis mapping"):
        PeriodicIdentification(
            x_min, y_max, PeriodicOrientation((0, 1), (1, 1)))


def test_ports_are_typed_owner_qualified_and_representation_explicit():
    boundary = _topology().physical[0]
    state, field, _ = _model_values()
    _, conservative = _representations()
    ports = (
        GhostState(boundary, state, conservative),
        ExteriorTrace(boundary, field, conservative),
        NumericalFlux(boundary, state, conservative),
        ConstraintResidual(boundary, field, conservative),
    )
    assert [row.port_type for row in ports] == [
        "ghost_state", "exterior_trace", "numerical_flux", "constraint_residual"]
    assert len({row.canonical_id for row in ports}) == 4
    with pytest.raises(TypeError, match="Handle.kind"):
        GhostState(boundary, field, conservative)
    with pytest.raises(TypeError, match="never a string"):
        ExteriorTrace(boundary, "phi", conservative)


def test_primitive_to_conservative_conversion_never_happens_implicitly():
    primitive, conservative = _representations()
    with pytest.raises(ValueError, match="explicit provider"):
        RepresentationFlow(primitive, conservative, None)
    converter = Handle(
        "euler_primitive_to_conservative", kind="representation_conversion",
        owner=OwnerPath.shared("pops.representations"))
    flow = RepresentationFlow(primitive, conservative, converter)
    dependencies = _dependencies(flow)
    state, _, _ = _model_values()
    output = GhostState(_topology().physical[0], state, conservative)
    provider = GhostFormula(
        handle=_provider_handle("ghost_conversion"), outputs=(output,),
        dependencies=dependencies)
    assert provider.dependencies.representation.converter == converter
    assert provider.canonical_identity()["dependencies"]["representation"]["converter"] \
        == converter.canonical_identity()


def test_directional_transport_requires_runtime_and_spatial_sign_dependencies():
    state, velocity, characteristics = _model_values()
    time, direction = _runtime_values()
    _, conservative = _representations()
    flow = RepresentationFlow(conservative, conservative, None)
    closure = CharacteristicClosure(
        ClosureMode.DIRECTIONAL, SignDependence.RUNTIME_SPATIAL, SonicPolicy.NEUTRAL,
        IncomingMultiplicity.MULTIPLE, (characteristics, state))
    with pytest.raises(ValueError, match="RuntimeParam"):
        BoundaryDependencies(
            states=(state,), fields=(velocity,), time=(time,), runtime_params=(),
            representation=flow, characteristic=closure)
    with pytest.raises(ValueError, match="state/field"):
        BoundaryDependencies(
            states=(), fields=(), time=(time,), runtime_params=(direction,),
            representation=flow, characteristic=closure)

    dependencies = BoundaryDependencies(
        states=(state,), fields=(velocity,), time=(time,), runtime_params=(direction,),
        representation=flow, characteristic=closure)
    output = ExteriorTrace(_topology().physical[0], state, conservative)
    provider = DirectionalTransport(
        handle=_provider_handle("directional"), outputs=(output,),
        dependencies=dependencies)
    report = provider.inspect()["dependencies"]["characteristic"]
    assert report["sign_dependence"] == "runtime_spatial"
    assert report["sonic"] == "neutral"
    assert report["incoming"] == "multiple"
    assert len(report["characteristics"]) == 2
    assert BoundaryProviderRegistry(provider).resolve(_topology(), (output,)).bindings[0].provider \
        == provider

    non_directional = BoundaryDependencies(
        states=(state,), fields=(), time=(), runtime_params=(), representation=flow,
        characteristic=CharacteristicClosure(
            ClosureMode.CHARACTERISTIC, SignDependence.FIXED, SonicPolicy.NEUTRAL,
            IncomingMultiplicity.SINGLE, (characteristics,)))
    with pytest.raises(ValueError, match="directional characteristic"):
        DirectionalTransport(
            handle=_provider_handle("wrong_directional"), outputs=(output,),
            dependencies=non_directional)


def test_named_provider_factories_are_data_only_and_port_typed():
    boundary = _topology().physical[0]
    state, field, _ = _model_values()
    _, conservative = _representations()
    dependencies = _dependencies()
    trace_state = ExteriorTrace(boundary, state, conservative)
    trace_field = ExteriorTrace(boundary, field, conservative)
    ghost = GhostState(boundary, state, conservative)
    residual = ConstraintResidual(boundary, field, conservative)
    providers = (
        Inflow(handle=_provider_handle("inflow"), outputs=(trace_state,),
               dependencies=dependencies),
        Outflow(handle=_provider_handle("outflow"), outputs=(ghost,),
                dependencies=dependencies),
        GhostFormula(handle=_provider_handle("ghost"), outputs=(ghost,),
                     dependencies=dependencies),
        Dirichlet(handle=_provider_handle("dirichlet"), outputs=(trace_field,),
                  dependencies=dependencies),
        Neumann(handle=_provider_handle("neumann"), outputs=(residual,),
                dependencies=dependencies),
        Mixed(handle=_provider_handle("mixed"), outputs=(residual,),
              dependencies=dependencies),
    )
    assert all(type(row) is BoundaryProvider for row in providers)
    assert all(not hasattr(row, "callback") for row in providers)
    with pytest.raises(TypeError, match="typed ConstraintResidual"):
        Mixed(handle=_provider_handle("bad_mixed"), outputs=(ghost,),
              dependencies=dependencies)


def test_noflux_satisfies_numerical_flux_only():
    boundary = _topology().physical[0]
    state, _, _ = _model_values()
    _, conservative = _representations()
    flux = NumericalFlux(boundary, state, conservative)
    ghost = GhostState(boundary, state, conservative)
    provider = NoFlux(
        handle=_provider_handle("no_flux"), output=flux, dependencies=_dependencies())
    assert provider.outputs == (flux,)
    assert BoundaryProviderRegistry(provider).resolve(_topology(), (flux,)).bindings
    with pytest.raises(TypeError, match="NumericalFlux only"):
        NoFlux(handle=_provider_handle("bad_no_flux"), output=ghost,
               dependencies=_dependencies())
    with pytest.raises(ValueError, match="missing boundary provider"):
        BoundaryProviderRegistry().resolve(_topology(), (ghost,))


def test_resolution_diagnostics_cover_missing_double_extra_ambiguous_and_periodic_physical():
    topology = _topology()
    boundary = topology.physical[0]
    state, _, _ = _model_values()
    _, conservative = _representations()
    need = ExteriorTrace(boundary, state, conservative)
    first = Inflow(
        handle=_provider_handle("first"), outputs=(need,), dependencies=_dependencies())
    second = Inflow(
        handle=_provider_handle("second"), outputs=(need,), dependencies=_dependencies())

    with pytest.raises(ValueError, match="missing"):
        BoundaryProviderRegistry().resolve(topology, (need,))
    with pytest.raises(ValueError, match="double boundary need"):
        BoundaryProviderRegistry(first).resolve(topology, (need, need))
    with pytest.raises(ValueError, match="double boundary provider"):
        BoundaryProviderRegistry(first, first)
    with pytest.raises(ValueError, match="ambiguous"):
        BoundaryProviderRegistry(first, second).resolve(topology, (need,))
    with pytest.raises(ValueError, match="extra boundary provider outputs"):
        BoundaryProviderRegistry(first).resolve(topology, ())

    periodic_need = ExteriorTrace(topology.boundaries[0], state, conservative)
    with pytest.raises(ValueError, match=r"periodic\+physical"):
        BoundaryProviderRegistry().resolve(topology, (periodic_need,))


def test_resolution_rejects_foreign_case_subject_before_compile():
    topology = _topology()
    _, conservative = _representations()
    foreign_state = Handle("U", kind="state", owner=_case_instance("foreign"))
    need = ExteriorTrace(topology.physical[0], foreign_state, conservative)
    provider = Inflow(
        handle=_provider_handle("foreign_subject"), outputs=(need,),
        dependencies=_dependencies())

    with pytest.raises(ValueError, match=r"subject belongs to foreign Case 'foreign'"):
        BoundaryProviderRegistry(provider).resolve(topology, (need,))


def test_resolution_rejects_foreign_case_dependency_before_compile():
    topology = _topology()
    state, _, _ = _model_values()
    _, conservative = _representations()
    foreign_field = Handle(
        "velocity", kind="field", owner=_case_instance("foreign"))
    dependencies = BoundaryDependencies(
        states=(), fields=(foreign_field,), time=(), runtime_params=(),
        representation=RepresentationFlow(conservative, conservative, None),
        characteristic=_none_closure())
    need = ExteriorTrace(topology.physical[0], state, conservative)
    provider = Inflow(
        handle=_provider_handle("foreign_dependency"), outputs=(need,),
        dependencies=dependencies)

    with pytest.raises(ValueError, match=r"fields\[0\] belongs to foreign Case 'foreign'"):
        BoundaryProviderRegistry(provider).resolve(topology, (need,))


def test_every_semantic_field_is_immutable():
    topology = _topology()
    boundary = topology.physical[0]
    state, field, characteristics = _model_values()
    time, direction = _runtime_values()
    primitive, conservative = _representations()
    converter = Handle(
        "convert", kind="representation_conversion",
        owner=OwnerPath.shared("pops.representations"))
    flow = RepresentationFlow(primitive, conservative, converter)
    closure = CharacteristicClosure(
        ClosureMode.DIRECTIONAL, SignDependence.RUNTIME_SPATIAL, SonicPolicy.NEUTRAL,
        IncomingMultiplicity.MULTIPLE, (characteristics, state))
    dependencies = BoundaryDependencies(
        states=(state,), fields=(field,), time=(time,), runtime_params=(direction,),
        representation=flow, characteristic=closure)
    port = ExteriorTrace(boundary, state, conservative)
    provider = DirectionalTransport(
        handle=_provider_handle("immutable"), outputs=(port,), dependencies=dependencies)
    registry = BoundaryProviderRegistry(provider)
    plan = registry.resolve(topology, (port,))
    periodic = topology.periodic[0]
    cases = [
        (boundary.orientation, "axis", 0), (boundary.orientation, "side", BoundarySide.UPPER),
        (boundary, "local_id", "other"), (boundary, "owner_path", OwnerPath.case("other")),
        (boundary, "orientation", BoundaryOrientation(0, BoundarySide.UPPER)),
        (periodic.orientation, "permutation", (1, 0)),
        (periodic.orientation, "signs", (-1, 1)), (periodic, "source", boundary),
        (periodic, "target", boundary), (periodic, "orientation", periodic.orientation),
        (topology, "owner", OwnerPath.case("other")), (topology, "boundaries", ()),
        (topology, "periodic", ()), (topology, "physical", ()),
        (port, "boundary", topology.physical[1]), (port, "subject", field),
        (port, "representation", primitive), (flow, "source", conservative),
        (flow, "target", primitive), (flow, "converter", None),
        (closure, "mode", ClosureMode.NONE),
        (closure, "sign_dependence", SignDependence.FIXED),
        (closure, "sonic", SonicPolicy.ERROR),
        (closure, "incoming", IncomingMultiplicity.SINGLE),
        (closure, "characteristics", (state,)),
        (dependencies, "states", ()), (dependencies, "fields", ()),
        (dependencies, "time", ()), (dependencies, "runtime_params", ()),
        (dependencies, "representation", _dependencies().representation),
        (dependencies, "characteristic", _none_closure()),
        (provider, "handle", _provider_handle("other")), (provider, "outputs", ()),
        (provider, "dependencies", _dependencies()), (registry, "providers", ()),
        (plan.bindings[0], "need", port), (plan.bindings[0], "provider", provider),
        (plan, "topology", topology), (plan, "needs", ()), (plan, "bindings", ()),
    ]
    for value, field_name, replacement in cases:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, field_name, replacement)
