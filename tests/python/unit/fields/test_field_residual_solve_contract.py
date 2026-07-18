from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pops.fields import (
    Accepted,
    CompositeHierarchySolve,
    ConnectedComponentsManifest,
    DirichletContribution,
    FieldArtifactUnavailable,
    FieldContext,
    FieldDependencyCoverage,
    FieldInput,
    FieldOperatorDomain,
    FieldResidualContract,
    FieldResidualDependencies,
    FieldRestartContract,
    FieldHierarchyPolicy,
    FieldSolveCapabilities,
    FieldSolveResolver,
    FieldValidity,
    InferHierarchyFromLayout,
    LayoutBinding,
    LevelByLevelSolve,
    MixedContribution,
    NeumannContribution,
    NullspaceBasis,
    NullspaceBasisVector,
    NullspaceCompatibility,
    PeriodicContribution,
    PreconditionerBinding,
    Provisional,
    RHSCompatibilityEvidence,
    ResolvedHierarchyPolicy,
    SolveOutcome,
    SolveStatus,
)
from pops.fields.bcs import Mixed
from pops.identity import make_identity
from pops.mesh.boundaries import (
    BoundaryDependencies,
    BoundaryHandle,
    BoundaryOrientation,
    BoundaryProvider,
    BoundarySide,
    BoundaryTopology,
    CharacteristicClosure,
    ClosureMode,
    ConstraintResidual,
    IncomingMultiplicity,
    PeriodicIdentification,
    PeriodicOrientation,
    RepresentationFlow,
    SignDependence,
    SonicPolicy,
)
from pops.model import Handle, OwnerPath
from pops.time import Clock, TimePoint


CASE = OwnerPath.case("main")
MODEL = OwnerPath.model("field-model")
SHARED = OwnerPath.shared("field-contracts")


def _h(name, kind, owner=SHARED):
    return Handle(name, kind=kind, owner=owner)


def _topology():
    boundaries = tuple(
        BoundaryHandle(
            "%s_%s" % (axis_name, side.value), owner=CASE,
            orientation=BoundaryOrientation(axis, side))
        for axis, axis_name in enumerate(("x", "y", "z"))
        for side in (BoundarySide.LOWER, BoundarySide.UPPER)
    )
    by_name = {row.local_id: row for row in boundaries}
    periodic = PeriodicIdentification(
        by_name["x_lower"], by_name["x_upper"],
        PeriodicOrientation((0, 1, 2), (1, -1, 1)))
    physical = tuple(row for row in boundaries if not row.local_id.startswith("x_"))
    return BoundaryTopology(CASE, boundaries, (periodic,), physical)


def _point(step=3):
    return TimePoint(Clock("field", owner=CASE), step=step)


def _dependencies(*, states=(), fields=(), point=None):
    return FieldResidualDependencies(
        _h("phi", "field", MODEL), tuple(states), tuple(fields), point or _point(),
        (_h("t", "time", CASE),))


def _provider(region, name, dependencies):
    if isinstance(region, PeriodicIdentification):
        return _h("%s_provider" % name, "periodic_field_provider", CASE)
    boundary = region
    representation = _h("conservative", "representation")
    provider_dependencies = BoundaryDependencies(
        dependencies.states, dependencies.fields, dependencies.time_sources, (),
        RepresentationFlow(representation, representation, None),
        CharacteristicClosure(
            ClosureMode.NONE, SignDependence.FIXED, SonicPolicy.NEUTRAL,
            IncomingMultiplicity.SINGLE, ()))
    return BoundaryProvider(
        _h("%s_provider" % name, "boundary_provider", CASE),
        (ConstraintResidual(boundary, dependencies.iterate, representation),),
        provider_dependencies)


def _contribution(cls, region, name, dependencies):
    return cls(
        region, _provider(region, name, dependencies), dependencies,
        _h("%s_residual" % name, "field_boundary_residual"),
        _h("%s_jacobian" % name, "field_boundary_jacobian"),
        _h("%s_jvp" % name, "field_boundary_jvp"),
    )


def _residual_contract():
    topology = _topology()
    physical = {row.local_id: row for row in topology.physical}
    base = _dependencies(
        states=(_h("rho", "state", MODEL),),
        fields=(_h("epsilon", "field", MODEL),))
    contributions = (
        _contribution(
            DirichletContribution, physical["y_lower"], "dirichlet",
            _dependencies(fields=(_h("wall_value", "field", MODEL),))),
        _contribution(
            NeumannContribution, physical["y_upper"], "neumann",
            _dependencies(states=(_h("wall_flux", "state", MODEL),))),
        _contribution(
            MixedContribution, physical["z_lower"], "mixed",
            _dependencies(fields=(_h("robin_alpha", "field", MODEL),))),
        _contribution(
            PeriodicContribution, topology.periodic[0], "periodic",
            _dependencies(fields=(_h("periodic_map", "field", MODEL),))),
    )
    expected = FieldResidualDependencies.merged(
        base, *(row.dependencies for row in contributions))
    coverage = FieldDependencyCoverage(expected, expected, expected, expected)
    residual = _h("R_phi", "field_residual", CASE)
    jacobian = _h("J_phi", "field_jacobian", CASE)
    jvp = _h("Jv_phi", "field_jvp", CASE)
    restart = FieldRestartContract(
        _h("field_restart", "field_restart_contract", CASE), expected,
        (jvp, residual, jacobian))
    return FieldResidualContract(
        _h("field_residual_contract", "field_residual_contract", CASE),
        _h("elliptic", "field_operator", CASE), base.iterate, topology, base,
        contributions, residual, jacobian, jvp, coverage, restart)


def _hierarchy_policy(name="composite"):
    return ResolvedHierarchyPolicy(
        "pops.field-hierarchy.%s" % name,
        1,
        "pops.field-hierarchy.options.empty@1",
        {},
    )


def _capabilities(*, inferred="composite",
                  native=("residual", "jacobian", "jvp", "restart"),
                  boundaries=("dirichlet", "neumann", "mixed", "periodic")):
    return FieldSolveCapabilities(
        _h("native_fields", "field_solve_capability", CASE),
        _hierarchy_policy(inferred),
        tuple(native), tuple(boundaries))


def _layout(generation=2):
    return LayoutBinding(_h("mesh", "layout", CASE), generation)


def test_typed_boundary_contributions_cover_exact_iterate_state_field_and_time():
    contract = _residual_contract()
    assert {row.contribution_type for row in contract.boundaries} == {
        "dirichlet", "neumann", "mixed", "periodic"}
    expected = contract.coverage.residual
    assert expected.iterate == contract.unknown
    assert expected.point == _point()
    assert {row.local_id for row in expected.states} == {"rho", "wall_flux"}
    assert {row.local_id for row in expected.fields} == {
        "epsilon", "wall_value", "robin_alpha", "periodic_map"}
    assert {row.local_id for row in expected.time_sources} == {"t"}
    for name in ("residual", "jacobian", "jvp", "restart"):
        assert getattr(contract.coverage, name) == expected
    assert contract.restart.dependencies == expected
    assert set(contract.restart.payloads) == {
        contract.residual, contract.jacobian, contract.jvp}
    assert contract.inspect()["identity"] == contract.identity


def test_exact_timepoint_and_complete_coverage_fail_before_lowering():
    contract = _residual_contract()
    physical = next(
        row for row in contract.boundaries if row.contribution_type == "dirichlet")
    with pytest.raises(ValueError, match="omits provider fields dependencies"):
        replace(physical, dependencies=_dependencies())
    wrong_dependencies = replace(contract.boundaries[0].dependencies, point=_point(4))
    wrong_time = replace(contract.boundaries[0], dependencies=wrong_dependencies)
    with pytest.raises(ValueError, match="exact TimePoint"):
        replace(contract, boundaries=(wrong_time,) + contract.boundaries[1:])

    omitted = FieldResidualDependencies(
        contract.unknown, contract.dependencies.states, contract.dependencies.fields,
        contract.dependencies.point, contract.dependencies.time_sources)
    with pytest.raises(ValueError, match="dependency coverage omits"):
        replace(
            contract,
            coverage=FieldDependencyCoverage(
                omitted, contract.coverage.jacobian,
                contract.coverage.jvp, contract.coverage.restart),
        )
    with pytest.raises(ValueError, match="restart contract omits"):
        replace(contract, restart=replace(contract.restart, dependencies=omitted))


def test_mixed_boundary_descriptor_is_explicit_and_resolves_dependencies():
    alpha = _h("alpha", "field", MODEL)
    beta = _h("beta", "field", MODEL)
    value = _h("value", "field", MODEL)
    mixed = Mixed(alpha, beta, value)
    assert set(mixed.declaration_references()) == {alpha, beta, value}
    assert mixed.options()["bc"] == "mixed"
    with pytest.raises(ValueError, match="non-zero"):
        Mixed(0, 0)


def _nullspace(*, compatible=True):
    components = (_h("island_a", "connected_component", CASE),
                  _h("island_b", "connected_component", CASE))
    manifest = ConnectedComponentsManifest(
        _h("components", "connected_components_manifest", CASE), components)
    basis = NullspaceBasis(
        manifest,
        tuple(NullspaceBasisVector(
            _h("basis_%s" % row.local_id, "nullspace_basis", CASE), row)
              for row in components))
    evidence = tuple(RHSCompatibilityEvidence(
        row, compatible or index == 0,
        _h("rhs_%s" % row.local_id, "rhs_compatibility_witness", CASE))
                     for index, row in enumerate(components))
    return basis, evidence


def test_nullspace_has_one_basis_per_component_and_never_projects_rhs():
    basis, evidence = _nullspace()
    compatibility = NullspaceCompatibility(basis, evidence)
    assert len(compatibility.basis.vectors) == 2
    assert compatibility.to_data()["rhs_projection"] == "forbidden"
    with pytest.raises(ValueError, match="exactly every connected component"):
        NullspaceBasis(basis.manifest, basis.vectors[:1])

    basis, incompatible = _nullspace(compatible=False)
    with pytest.raises(ValueError, match="silent projection is forbidden"):
        NullspaceCompatibility(basis, incompatible)


def test_hierarchy_and_native_contracts_resolve_by_capabilities_before_codegen():
    residual = _residual_contract()
    capabilities = _capabilities()
    resolver = FieldSolveResolver(capabilities)
    assert resolver.resolve(
        residual, CompositeHierarchySolve(), _layout()
    ).hierarchy.policy_id == "pops.field-hierarchy.composite"
    assert resolver.resolve(
        residual, LevelByLevelSolve(), _layout()
    ).hierarchy.policy_id == "pops.field-hierarchy.level-local"
    assert resolver.resolve(
        residual, InferHierarchyFromLayout(), _layout()
    ).hierarchy.policy_id == "pops.field-hierarchy.composite"

    class ExternalCoupledHierarchy(FieldHierarchyPolicy):
        def options(self):
            return self.resolved_authority().authority()

        def resolved_authority(self):
            return ResolvedHierarchyPolicy(
                "tests.field-hierarchy.coupled-graph",
                3,
                "tests.field-hierarchy.coupled-graph.options@2",
                {"overlap": 2},
            )

    external = resolver.resolve(
        residual, ExternalCoupledHierarchy(), _layout()
    ).hierarchy
    assert external.authority() == {
        "policy_id": "tests.field-hierarchy.coupled-graph",
        "interface_version": 3,
        "option_schema": "tests.field-hierarchy.coupled-graph.options@2",
        "options": {"overlap": 2},
    }
    assert external.capability == capabilities.handle

    missing_jvp = FieldSolveResolver(
        _capabilities(native=("residual", "jacobian", "restart")))
    with pytest.raises(FieldArtifactUnavailable) as native_error:
        missing_jvp.resolve(residual, CompositeHierarchySolve(), _layout())
    assert native_error.value.report["code"] == "field.native_contract.unsupported"
    assert native_error.value.report["artifact_created"] is False


def test_preconditioner_domain_and_nullspace_gauge_are_authenticated_separately():
    residual = _residual_contract()
    layout = _layout()
    resolver = FieldSolveResolver(_capabilities())
    domain = FieldOperatorDomain(residual.identity, residual.unknown, layout)
    preconditioner = PreconditionerBinding(
        _h("mg", "preconditioner", CASE), domain)
    basis, evidence = _nullspace()
    nullspace = NullspaceCompatibility(basis, evidence)
    gauge = _h("mean_zero", "field_gauge", CASE)
    plan = resolver.resolve(
        residual, CompositeHierarchySolve(), layout, preconditioner=preconditioner,
        nullspace=nullspace, gauge=gauge)
    assert plan.preconditioner.domain == plan.domain
    assert plan.nullspace is nullspace
    assert plan.gauge == gauge

    wrong_domain = FieldOperatorDomain(residual.identity, residual.unknown, _layout(3))
    with pytest.raises(ValueError, match="preconditioner domain"):
        resolver.resolve(
            residual, CompositeHierarchySolve(), layout,
            preconditioner=PreconditionerBinding(
                _h("wrong_mg", "preconditioner", CASE), wrong_domain))
    with pytest.raises(ValueError, match="nullspace and gauge are separate"):
        resolver.resolve(
            residual, CompositeHierarchySolve(), layout, nullspace=nullspace)


def _context(plan, *, accepted=True):
    dependencies = plan.residual.coverage.residual
    inputs = tuple(FieldInput(
        reference, make_identity("field-input", {"reference": reference.qualified_id}))
                   for reference in dependencies.states + dependencies.fields +
                   dependencies.time_sources + dependencies.parameters)
    materialization = Accepted() if accepted else Provisional("solve-attempt")
    return FieldContext(
        plan.residual.operator, inputs, dependencies.point.clock, dependencies.point,
        plan.domain.layout, materialization,
        FieldValidity.valid_at(dependencies.point, plan.domain.layout))


def test_solve_outcome_never_publishes_failed_or_incompatible_accepted_context():
    plan = FieldSolveResolver(_capabilities()).resolve(
        _residual_contract(), CompositeHierarchySolve(), _layout())
    witness = _h("solve_report", "field_solve_witness", CASE)
    accepted = _context(plan)
    converged = SolveOutcome(
        plan, SolveStatus.CONVERGED, 6, witness, "tolerance reached", accepted)
    assert converged.publish() is accepted

    for status in (SolveStatus.NON_CONVERGED, SolveStatus.INCOMPATIBLE_RHS):
        with pytest.raises(ValueError, match="cannot publish Accepted"):
            SolveOutcome(plan, status, 6, witness, "failed", accepted)
    failed = SolveOutcome(
        plan, SolveStatus.NON_CONVERGED, 6, witness, "maximum iterations", _context(
            plan, accepted=False))
    with pytest.raises(RuntimeError, match="only a converged solve"):
        failed.publish()
    incompatible = SolveOutcome(
        plan, SolveStatus.INCOMPATIBLE_RHS, 0, witness,
        "RHS violates the authenticated nullspace compatibility condition",
        _context(plan, accepted=False))
    assert incompatible.to_data()["status"] == "incompatible_rhs"
    assert incompatible.to_data()["reason"].startswith("RHS violates")


def test_resolved_contracts_and_outcomes_are_immutable():
    residual = _residual_contract()
    resolver = FieldSolveResolver(_capabilities())
    plan = resolver.resolve(
        residual, CompositeHierarchySolve(), _layout())
    basis, evidence = _nullspace()
    compatibility = NullspaceCompatibility(basis, evidence)
    outcome = SolveOutcome(
        plan, SolveStatus.NON_CONVERGED, 1,
        _h("witness", "field_solve_witness", CASE), "stopped", _context(
            plan, accepted=False))
    values = (
        (residual.dependencies, "states", ()), (residual.boundaries[0], "provider", None),
        (residual.coverage, "residual", residual.dependencies),
        (residual.restart, "payloads", ()), (residual, "boundaries", ()),
        (basis, "vectors", ()), (compatibility, "evidence", ()),
        (plan.domain, "layout", _layout(9)), (plan.capabilities, "native_contracts", ()),
        (resolver, "capabilities", _capabilities()), (plan, "preconditioner", None),
        (outcome, "status", SolveStatus.CONVERGED),
    )
    for value, field, replacement in values:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, field, replacement)
