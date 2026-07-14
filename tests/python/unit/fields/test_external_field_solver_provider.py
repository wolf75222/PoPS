"""Typed external FieldTopology + FieldSolver authoring and resolve gates."""
from __future__ import annotations

import json

import pytest

from pops import interfaces
from pops.codegen._orchestration_compile import capture_field_plans
from pops.codegen.lowering_coverage import LoweringRejection
from pops.external import build_source_package_manifest, load
from pops.fields import (
    CellCenteredSecondOrder,
    ExternalFieldSolver,
    FieldDiscretization,
    FieldOutput,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet
from pops.layouts import Uniform
from pops.math import laplacian
from pops.model import ComponentManifest
from pops.physics import Model
from pops.problem import Case
from tests.python.support.layout_plan import cartesian_grid


def _component(tmp_path, *, name, interface, source_suffix=b"", dimension=2):
    root = tmp_path / name
    root.mkdir(parents=True)
    manifest = ComponentManifest(
        uri="pops://external.test/fields/%s" % name,
        component_type=interface.name,
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={"variants": [{
            "dimension": dimension,
            "scalar": "float64",
            "device": "cpu",
            "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = b"// resolve-only external field component\n" + source_suffix
    source_name = name + ".cpp"
    (root / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={name: manifest}, payloads={source_name: ("source", source)})
    package_path = root / (name + ".pops.json")
    package_path.write_text(json.dumps(package_data), encoding="utf-8")
    return load(package_path).require(name, interface=interface)()


def _case(solver):
    model = Model("external-field-solver-model")
    (rho,) = model.state("U", components=("rho",))
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential",
        unknown=unknown,
        equation=(-laplacian(unknown) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )
    case = Case("external-field-solver-case")
    case.block("material", model)
    case.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(
            AllPhysicalBoundaries(), Dirichlet(0.0)),),
        solver=solver,
    ))
    return case


def _provider(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology)
    solver = _component(
        tmp_path, name="solver", interface=interfaces.FieldSolver)
    return ExternalFieldSolver(
        topology=topology, solver=solver,
        relative_tolerance=1.0e-10, absolute_tolerance=1.0e-12,
        max_iterations=17,
    ), topology, solver


def test_external_field_solver_refuses_a_3d_only_pair_member(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology)
    solver = _component(
        tmp_path, name="solver-3d", interface=interfaces.FieldSolver, dimension=3)

    with pytest.raises(ValueError, match="2D float64 CPU"):
        ExternalFieldSolver(topology=topology, solver=solver)


def test_external_pair_survives_field_lowering_with_exact_component_authorities(tmp_path):
    provider, topology, solver = _provider(tmp_path)
    plan = capture_field_plans(
        _case(provider), lambda value: value, target="system",
        layout=Uniform(cartesian_grid(n=8, periodic=False)),
    )["potential"]

    external = plan.native_options["solver_provider"]
    assert external["provider_kind"] == "external_component_v1"
    assert external["topology"]["component_id"] == topology.component_manifest.component_id
    assert external["solver"]["component_id"] == solver.component_manifest.component_id
    assert external["request"] == {
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance": 1.0e-12,
        "max_iterations": 17,
    }
    assert plan.component_bindings() == (external["topology"], external["solver"])
    plan.require_component_inputs((topology, solver))


def test_external_pair_requires_both_exact_resolve_inputs(tmp_path):
    provider, topology, _solver = _provider(tmp_path)
    plan = capture_field_plans(
        _case(provider), lambda value: value, target="system",
        layout=Uniform(cartesian_grid(n=8, periodic=False)),
    )["potential"]

    with pytest.raises(ValueError, match="requires exact component"):
        plan.require_component_inputs((topology,))


def test_external_pair_rejects_same_manifest_from_another_source_package(tmp_path):
    provider, topology, _solver = _provider(tmp_path / "authored")
    substituted_solver = _component(
        tmp_path / "substitute", name="solver", interface=interfaces.FieldSolver,
        source_suffix=b"// different authenticated payload\n",
    )
    plan = capture_field_plans(
        _case(provider), lambda value: value, target="system",
        layout=Uniform(cartesian_grid(n=8, periodic=False)),
    )["potential"]

    with pytest.raises(ValueError, match="changed source package"):
        plan.require_component_inputs((topology, substituted_solver))


def test_external_field_solver_v2_refuses_amr_during_resolve(tmp_path):
    provider, _topology, _solver = _provider(tmp_path)

    with pytest.raises(LoweringRejection, match="hierarchy-aware") as error:
        capture_field_plans(
            _case(provider), lambda value: value, target="amr_system",
            layout=Uniform(cartesian_grid(n=8, periodic=False)),
        )
    assert error.value.gate == "field.solver.layout_incompatible"


def test_external_solver_and_topology_roles_are_not_interchangeable(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology)
    solver = _component(
        tmp_path, name="solver", interface=interfaces.FieldSolver)

    with pytest.raises(TypeError, match="solver must implement exact interface"):
        ExternalFieldSolver(topology=topology, solver=topology)
    with pytest.raises(TypeError, match="topology must implement exact interface"):
        ExternalFieldSolver(topology=solver, solver=solver)
