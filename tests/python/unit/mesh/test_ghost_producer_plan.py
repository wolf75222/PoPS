from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from pops.mesh.boundaries import (
    BoundaryDependencies,
    BoundaryHandle,
    BoundaryLinearizationContribution,
    BoundaryOrientation,
    BoundaryResidualContribution,
    BoundarySide,
    BoundaryTopology,
    CharacteristicClosure,
    ClosureMode,
    CoarseFineInterpolation,
    CornerCondition,
    CornerConstraint,
    CornerMode,
    CornerPolicy,
    Dirichlet,
    ExteriorTrace,
    GhostCoverageManifest,
    GhostDepthCapability,
    GhostDepthRequirement,
    GhostProducerPlan,
    GhostProducerRegistry,
    GhostProduction,
    GhostRegion,
    GhostStencilManifest,
    IncomingMultiplicity,
    InterfaceAffineMapping,
    InterfaceGhost,
    InterfacePermutation,
    InterfaceSide,
    MultiBlockInterface,
    NumericalClosure,
    PeriodicGhost,
    PeriodicIdentification,
    PeriodicOrientation,
    PhysicalGhost,
    RepresentationFlow,
    SameLevelHaloMPI,
    SignDependence,
    SonicPolicy,
)
from pops.model import Handle, OwnerKind, OwnerPath


SHARED = OwnerPath.shared("ghost.fixtures")
CASE = OwnerPath.case("main")


class _ExecutableBoundaryAuthority:
    def canonical_identity(self):
        return {"authority_type": "test_boundary_execution"}

    def compile_boundary_data(self):
        return {"schema_version": 1, "authority_type": "prepared_boundary_plan_compile"}

    def runtime_boundary_data(self, params):
        del params
        return {
            "schema_version": 1,
            "authority_type": "prepared_boundary_plan",
            "identity": "test-plan",
        }


def _h(name, kind, owner=SHARED):
    return Handle(name, kind=kind, owner=owner)


def _boundaries():
    return (
        BoundaryHandle("x_min", owner=CASE,
                       orientation=BoundaryOrientation(0, BoundarySide.LOWER)),
        BoundaryHandle("x_max", owner=CASE,
                       orientation=BoundaryOrientation(0, BoundarySide.UPPER)),
        BoundaryHandle("y_min", owner=CASE,
                       orientation=BoundaryOrientation(1, BoundarySide.LOWER)),
        BoundaryHandle("y_max", owner=CASE,
                       orientation=BoundaryOrientation(1, BoundarySide.UPPER)),
    )


def _topology():
    x_min, x_max, y_min, y_max = _boundaries()
    periodic = PeriodicIdentification(
        x_min, x_max, PeriodicOrientation((0, 1), (1, -1)))
    return BoundaryTopology(CASE, (x_min, x_max, y_min, y_max),
                            (periodic,), (y_min, y_max))


def _depth(required=(2, 2), available=(3, 3), suffix=""):
    stencil = GhostStencilManifest(
        _h("weno%s" % suffix, "stencil_manifest"), required)
    capability = GhostDepthCapability(
        _h("allocated%s" % suffix, "capability"),
        _h("layout%s" % suffix, "layout_manifest"), available)
    return GhostDepthRequirement(stencil, capability)


def _region(name, *, boundary=None, depth=None, layout=None):
    return GhostRegion(
        _h("U", "state", OwnerPath.model("transport")),
        layout or _h("mesh", "layout", CASE.child(OwnerKind.LAYOUT, "mesh")),
        _h(name, "ghost_region", CASE), depth or _depth(suffix="_%s" % name), boundary)


def _coverage(*regions):
    return GhostCoverageManifest(
        _h("coverage", "ghost_coverage_manifest", CASE),
        _h("layout", "layout_manifest"),
        _h("disc", "discretization_manifest"), tuple(regions))


def _producer_handle(name):
    return _h(name, "ghost_producer", CASE)


def _protocol(name):
    return _h(name, "ghost_producer_protocol")


def _none_dependencies():
    representation = _h("conservative", "representation")
    return BoundaryDependencies(
        states=(), fields=(), time=(), runtime_params=(),
        representation=RepresentationFlow(representation, representation, None),
        characteristic=CharacteristicClosure(
            ClosureMode.NONE, SignDependence.FIXED, SonicPolicy.NEUTRAL,
            IncomingMultiplicity.SINGLE, ()))


def _mpi_periodic_fixture():
    topology = _topology()
    halo = _region("same_level_halo")
    periodic_region = _region("periodic_x", boundary=topology.periodic[0].source)
    mpi = SameLevelHaloMPI(
        handle=_producer_handle("mpi_halo"), protocol=_protocol("same_level_halo"),
        mpi_capability=_h("mpi_neighbor_exchange", "capability"))
    periodic = PeriodicGhost(
        handle=_producer_handle("periodic"), protocol=_protocol("periodic"),
        identification=topology.periodic[0])
    residual = BoundaryResidualContribution(
        _h("halo_residual", "boundary_residual_contribution", CASE), halo, mpi.handle,
        _h("halo_residual_op", "residual_operator"))
    linearization = BoundaryLinearizationContribution(
        _h("halo_jacobian", "boundary_linearization_contribution", CASE), halo, mpi.handle,
        _h("halo_jacobian_op", "linearization_operator"))
    plan = GhostProducerRegistry(periodic, mpi).resolve(
        topology, _coverage(halo, periodic_region), (periodic_region, halo),
        (GhostProduction(periodic_region, periodic), GhostProduction(halo, mpi)),
        residual_contributions=(residual,), linearization_contributions=(linearization,))
    return plan, halo, periodic_region, mpi, periodic


def test_mpi_halo_and_oriented_periodic_are_canonical_plan_data():
    plan, halo, periodic_region, mpi, periodic = _mpi_periodic_fixture()
    report = plan.inspect()
    assert isinstance(plan, GhostProducerPlan)
    assert halo.depth.depth == (2, 2)
    assert mpi.capabilities[0].local_id == "mpi_neighbor_exchange"
    assert periodic.periodic[0].orientation.signs == (1, -1)
    assert {row.region for row in plan.productions} == {halo, periodic_region}
    assert report["residual_contributions"][0]["residual"]["kind"] == "residual_operator"
    assert report["linearization_contributions"][0]["linearization"]["kind"] \
        == "linearization_operator"
    assert plan.residual_contributions[0].inspect()["report_type"] \
        == "boundary_residual_contribution"
    assert plan.linearization_contributions[0].inspect()["report_type"] \
        == "boundary_linearization_contribution"
    assert json.loads(json.dumps(report)) == report

    reversed_plan = GhostProducerRegistry(mpi, periodic).resolve(
        plan.topology, plan.coverage, tuple(reversed(plan.regions)),
        tuple(reversed(plan.productions)),
        residual_contributions=plan.residual_contributions,
        linearization_contributions=plan.linearization_contributions)
    assert reversed_plan.canonical_id == plan.canonical_id

    executable = replace(plan, execution_authority=_ExecutableBoundaryAuthority())
    with pytest.raises(NotImplementedError, match="signed/permuted periodic"):
        executable.compile_boundary_data()


def test_producer_dependencies_are_a_total_acyclic_graph():
    topology = _topology()
    first_region = _region("dependency_first")
    second_region = _region("dependency_second")
    first_handle = _producer_handle("dependency_first")
    second_handle = _producer_handle("dependency_second")
    first = SameLevelHaloMPI(
        handle=first_handle, protocol=_protocol("dependency_first"),
        mpi_capability=_h("mpi_first", "capability"), dependencies=(second_handle,))
    second = SameLevelHaloMPI(
        handle=second_handle, protocol=_protocol("dependency_second"),
        mpi_capability=_h("mpi_second", "capability"), dependencies=(first_handle,))
    with pytest.raises(ValueError, match="dependency graph contains a cycle"):
        GhostProducerRegistry(first, second).resolve(
            topology, _coverage(first_region, second_region),
            (first_region, second_region),
            (GhostProduction(first_region, first), GhostProduction(second_region, second)),
        )

    missing_handle = _producer_handle("dependency_missing")
    missing = SameLevelHaloMPI(
        handle=first_handle, protocol=_protocol("dependency_missing"),
        mpi_capability=_h("mpi_missing", "capability"), dependencies=(missing_handle,))
    with pytest.raises(ValueError, match="depends on absent ghost producer"):
        GhostProducerRegistry(missing).resolve(
            topology, _coverage(first_region), (first_region,),
            (GhostProduction(first_region, missing),),
        )


def test_coverage_manifest_is_the_authority_for_empty_missing_and_extra_regions():
    topology = _topology()
    first = _region("first")
    second = _region("second")
    coverage = _coverage(first, second)
    producer = SameLevelHaloMPI(
        handle=_producer_handle("first"), protocol=_protocol("halo"),
        mpi_capability=_h("mpi", "capability"))

    with pytest.raises(ValueError, match="missing expected ghost regions"):
        GhostProducerRegistry().resolve(topology, coverage, (), ())
    with pytest.raises(ValueError, match="missing expected ghost regions"):
        GhostProducerRegistry(producer).resolve(
            topology, coverage, (first,), (GhostProduction(first, producer),))
    with pytest.raises(ValueError, match="extra ghost regions"):
        GhostProducerRegistry(producer).resolve(
            topology, _coverage(first), (first, second),
            (GhostProduction(first, producer), GhostProduction(second, producer)))


def test_missing_overlap_and_extra_producers_fail_without_order_priority():
    topology = _topology()
    first = _region("first")
    second = _region("second")
    coverage = _coverage(first, second)
    producer = SameLevelHaloMPI(
        handle=_producer_handle("halo"), protocol=_protocol("halo"),
        mpi_capability=_h("mpi", "capability"))
    with pytest.raises(ValueError, match="missing ghost producer"):
        GhostProducerRegistry(producer).resolve(
            topology, coverage, (first, second), (GhostProduction(first, producer),))
    with pytest.raises(ValueError, match="overlapping ghost regions"):
        GhostProducerRegistry(producer).resolve(
            topology, _coverage(first), (first, first),
            (GhostProduction(first, producer),))
    unused = SameLevelHaloMPI(
        handle=_producer_handle("unused"), protocol=_protocol("halo"),
        mpi_capability=_h("mpi_other", "capability"))
    with pytest.raises(ValueError, match="extra unused ghost producers"):
        GhostProducerRegistry(producer, unused).resolve(
            topology, _coverage(first), (first,), (GhostProduction(first, producer),))


def test_depth_is_derived_consumable_and_capability_checked():
    depth_two = _depth(required=(2, 3), available=(3, 3), suffix="_two")
    depth_one = _depth(required=(1, 3), available=(3, 3), suffix="_one")
    assert depth_two.depth == (2, 3)
    assert depth_two.canonical_identity() != depth_one.canonical_identity()
    assert _region("deep", depth=depth_two).canonical_id != \
        _region("deep", depth=depth_one).canonical_id
    with pytest.raises(TypeError):
        GhostDepthRequirement(depth_two.stencil, depth_two.capability, (9, 9))
    with pytest.raises(ValueError, match="insufficient"):
        _depth(required=(4, 2), available=(3, 3), suffix="_insufficient")
    with pytest.raises(ValueError, match="dimension is inconsistent"):
        _depth(required=(2, 2, 2), available=(3, 3), suffix="_dimension")


def _physical_provider(boundary, name):
    state = _h("U", "state", OwnerPath.model("transport"))
    representation = _h("conservative", "representation")
    output = ExteriorTrace(boundary, state, representation)
    return Dirichlet(
        handle=_h(name, "boundary_provider", CASE), outputs=(output,),
        dependencies=_none_dependencies())


def _interface(topology):
    left_boundary, right_boundary = topology.physical
    left = InterfaceSide(
        left_boundary, _h("left_layout", "layout", CASE),
        _h("fv", "discretization"), left_boundary.orientation,
        _h("left_projection", "interface_projection"))
    right = InterfaceSide(
        right_boundary, _h("right_layout", "layout", CASE),
        _h("dg", "discretization"), right_boundary.orientation,
        _h("right_projection", "interface_projection"))
    return MultiBlockInterface(
        _h("coupling", "multiblock_interface", CASE), left, right,
        _h("shared_flux", "conservative_flux", CASE),
        InterfacePermutation(_h("axis_permutation", "interface_permutation"), (0,)),
        InterfaceAffineMapping(_h("geometry_map", "interface_mapping")))


def test_all_explicit_producer_protocols_and_shared_interface_flux():
    topology = _topology()
    interface = _interface(topology)
    coarse = CoarseFineInterpolation(
        handle=_producer_handle("coarse_fine"), protocol=_protocol("coarse_fine"),
        interpolation=_h("conservative_prolongation", "interpolation"))
    physical = PhysicalGhost(
        handle=_producer_handle("physical"), protocol=_protocol("physical"),
        provider=_physical_provider(topology.physical[0], "wall"))
    coupling = InterfaceGhost(
        handle=_producer_handle("interface"), protocol=_protocol("interface"),
        interface=interface)
    closure = NumericalClosure(
        handle=_producer_handle("closure"), protocol=_protocol("closure"),
        closure=_h("corner_flux", "numerical_closure"))
    assert coarse.operators[0].kind == "interpolation"
    assert physical.boundary_providers[0].qualified_id.endswith("wall")
    assert closure.operators[0].kind == "numerical_closure"
    assert interface.left.layout != interface.right.layout
    assert interface.left.discretization != interface.right.discretization
    assert interface.left.orientation.outward_sign == -interface.right.orientation.outward_sign

    region = _region("interface", boundary=interface.left.boundary,
                     layout=interface.left.layout)
    plan = GhostProducerRegistry(coupling).resolve(
        topology, _coverage(region), (region,), (GhostProduction(region, coupling),),
        interfaces=(interface,))
    payload = plan.inspect()["interfaces"][0]
    assert payload["shared_conservative_flux"]["qualified_id"] \
        == interface.shared_conservative_flux.qualified_id
    assert payload["left"]["projection"] != payload["right"]["projection"]

    same_boundary = next(
        row for row in topology.boundaries
        if row != interface.left.boundary and
        row.orientation.outward_sign == interface.left.orientation.outward_sign)
    same_direction = InterfaceSide(
        same_boundary, _h("other_layout", "layout", CASE),
        _h("other_disc", "discretization"), same_boundary.orientation,
        _h("other_projection", "interface_projection"))
    with pytest.raises(ValueError, match="opposite orientations"):
        MultiBlockInterface(
            _h("bad", "multiblock_interface", CASE), interface.left, same_direction,
            _h("bad_flux", "conservative_flux", CASE),
            InterfacePermutation(_h("bad_permutation", "interface_permutation"), (0,)),
            InterfaceAffineMapping(_h("bad_mapping", "interface_mapping")))

    wrong_region = _region("wrong_wall", boundary=topology.physical[1])
    with pytest.raises(ValueError, match="physical ghost provider does not cover"):
        GhostProducerRegistry(physical).resolve(
            topology, _coverage(wrong_region), (wrong_region,),
            (GhostProduction(wrong_region, physical),))


def test_incompatible_dirichlet_corner_diagnostic_names_both_sources():
    topology = _topology()
    corner = _region("corner")
    first = _physical_provider(topology.physical[0], "wall_y_min")
    second = _physical_provider(topology.physical[1], "wall_y_max")
    constraints = (
        CornerConstraint(first, CornerCondition.DIRICHLET, _h("zero", "boundary_datum")),
        CornerConstraint(second, CornerCondition.DIRICHLET, _h("one", "boundary_datum")),
    )
    with pytest.raises(ValueError) as error:
        CornerPolicy(corner, constraints, CornerMode.ERROR)
    assert first.qualified_id in str(error.value)
    assert second.qualified_id in str(error.value)
    policy = CornerPolicy(
        corner, constraints, CornerMode.EXPLICIT_RESOLVER,
        _h("corner_reconcile", "corner_resolver"))
    assert policy.canonical_identity()["resolver"]["kind"] == "corner_resolver"


def test_plan_identity_objects_are_immutable():
    plan, region, _, producer, _ = _mpi_periodic_fixture()
    production = next(row for row in plan.productions if row.region == region)
    values = (
        (region.depth.stencil, "required_depth", (9, 9)),
        (region.depth.capability, "available_depth", (9, 9)),
        (region.depth, "depth", (9, 9)), (region, "selector", _h("other", "ghost_region")),
        (plan.coverage, "regions", ()), (producer, "protocol", _protocol("other")),
        (production, "region", plan.regions[1]), (plan, "regions", ()),
        (plan.residual_contributions[0], "residual", _h("other", "residual_operator")),
        (plan.linearization_contributions[0], "linearization",
         _h("other", "linearization_operator")),
    )
    for value, field, replacement in values:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, field, replacement)
