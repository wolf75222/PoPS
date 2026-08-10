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
    CompositeHierarchySolve,
    ExternalFieldSolver,
    FieldDiscretization,
    FieldOutput,
    LevelByLevelSolve,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet
from pops.layouts import Uniform
from pops.math import laplacian
from pops.model import ComponentManifest
from pops.physics import Model
from pops.problem import Case
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout


def _component(
    tmp_path, *, name, interface, source_suffix=b"", dimension=2,
    manifest_parameters=(), instance_parameters=None, features=(), device="cpu",
):
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
        parameters=manifest_parameters,
        target={"variants": [{
            "dimension": dimension,
            "scalar": "float64",
            "device": device,
            "features": list(features),
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
    factory = load(package_path).require(name, interface=interface)
    return factory(**({} if instance_parameters is None else instance_parameters))


def _case(solver, *, hierarchy_policy=None):
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
    options = {}
    if hierarchy_policy is not None:
        options["hierarchy_policy"] = hierarchy_policy
    case.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(
            AllPhysicalBoundaries(), Dirichlet(0.0)),),
        solver=solver,
        **options,
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


def test_external_field_solver_refuses_pair_without_a_common_ranked_dimension(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology)
    solver = _component(
        tmp_path, name="solver_3d", interface=interfaces.FieldSolver, dimension=3)

    with pytest.raises(ValueError, match="share no supported native dimension"):
        ExternalFieldSolver(topology=topology, solver=solver)


def test_external_field_solver_accepts_a_3d_pair_then_uses_the_domain_rank(tmp_path):
    topology = _component(
        tmp_path, name="topology_3d", interface=interfaces.FieldTopology, dimension=3)
    solver = _component(
        tmp_path, name="solver_3d", interface=interfaces.FieldSolver, dimension=3)
    provider = ExternalFieldSolver(topology=topology, solver=solver)

    assert provider.capabilities().to_dict()["host"] is True
    with pytest.raises(LoweringRejection, match="compatible Dim=2 float64 CPU targets"):
        capture_field_plans(
            _case(provider), lambda value: value, target="system",
            layout=Uniform(cartesian_grid(n=8, periodic=False)),
        )


def test_external_pair_survives_field_lowering_with_exact_component_authorities(tmp_path):
    provider, topology, solver = _provider(tmp_path)
    plan = capture_field_plans(
        _case(provider), lambda value: value, target="system",
        layout=Uniform(cartesian_grid(n=8, periodic=False)),
    )["potential"]

    from pops.fields._prepared_field_solver_registry import (
        prepared_field_solver_binding_from_data,
    )

    external = prepared_field_solver_binding_from_data(
        plan.native_options["solver_provider"]
    )
    assert external.provider["provider_id"] == "pops.fields.external-field-solver"
    assert external.provider["version"] == 2
    provider_authority = external.to_data()["provider"]
    assert provider_authority["use_policy"] == {
        "policy_id": "pops.fields.external-field-solver.use",
        "version": 5,
        "capabilities": {
            "provider_id": "pops.fields.external-field-solver",
            "provider_version": 2,
            "adapter_identity": ("pops.fields.external-field-solver.system-amr-host@3"),
            "targets": ["system", "amr_system"],
            "layout_kinds": ["uniform", "amr"],
            "max_levels": None,
            "refinement_ratio_policy": "hierarchy_exact_rank",
            "hierarchy_policies": [
                "pops.field-hierarchy.level-local",
                "pops.field-hierarchy.composite",
            ],
            "abi_patch_level_metadata": True,
            "hierarchy_materialization": True,
            "amr_provider_bridge": True,
            "binary_coarse_fine_coverage": True,
            "execution": "host-serial-or-declared-mpi-hierarchy-batch",
            "components": ["FieldTopology@2", "FieldSolver@2"],
        },
    }
    topology_binding, solver_binding = plan.component_bindings()
    assert topology_binding["component_id"] == topology.component_manifest.component_id
    assert solver_binding["component_id"] == solver.component_manifest.component_id
    assert external.resolution.native_contract["options"] == {
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance": 1.0e-12,
        "max_iterations": 17,
    }
    assert external.resolution.native_contract["schema_identity"] == (
        "pops.external.field-solver-request@2"
    )
    assert external.resolution.native_contract["provider_id"] == external.provider["provider_id"]
    assert external.resolution.topology_contract["hierarchy_policy"]["policy_id"] == (
        "pops.field-hierarchy.level-local"
    )
    assert provider.to_data()["provider"] == provider_authority
    capabilities = provider.capabilities().to_dict()
    assert capabilities["provider"] == provider_authority
    assert capabilities["adapter"] == provider_authority["use_policy"]["capabilities"]
    assert capabilities["supports_amr"] is True
    assert capabilities["max_levels"] is None
    assert capabilities["refinement_ratio_policy"] == "hierarchy_exact_rank"
    plan.require_component_inputs((topology, solver))

    # Artifact state is recursively immutable, but the Python/native boundary must receive an
    # ordinary detached carrier: external component parameters are serialized to JSON at install.
    native = plan.native_install_data()
    assert type(native) is dict
    assert type(
        native["solver_provider"]["resolution"]["component_bindings"][0]["parameters"]
    ) is dict
    json.dumps(native["solver_provider"], sort_keys=True, allow_nan=False)


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


def test_external_pair_canonicalizes_nested_parameters_without_weakening_identity(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology)
    options = {"levels": [1, 2], "policy": {"strict": True}}
    solver = _component(
        tmp_path, name="solver", interface=interfaces.FieldSolver,
        manifest_parameters=({"name": "options", "kind": "runtime"},),
        instance_parameters={"options": options},
    )
    provider = ExternalFieldSolver(topology=topology, solver=solver)
    plan = capture_field_plans(
        _case(provider), lambda value: value, target="system",
        layout=Uniform(cartesian_grid(n=8, periodic=False)),
    )["potential"]

    plan.require_component_inputs((topology, solver))
    assert plan.native_install_data()["solver_provider"]["resolution"][
        "component_bindings"
    ][1]["parameters"] == {
        "options": options,
    }

    substituted_solver = solver.component_type(
        options={"levels": [1, 2], "policy": {"strict": 1}})
    with pytest.raises(ValueError, match="parameters"):
        plan.require_component_inputs((topology, substituted_solver))


def test_external_field_solver_v2_resolves_one_composite_amr_hierarchy(tmp_path):
    provider, topology, solver = _provider(tmp_path)

    plan = capture_field_plans(
        _case(provider, hierarchy_policy=CompositeHierarchySolve()),
        lambda value: value,
        target="amr_system",
        layout=final_amr_layout(cartesian_grid(n=8, periodic=False), max_levels=3, ratio=2),
    )["potential"]

    assert plan.native_options["hierarchy_policy"]["policy_id"] == (
        "pops.field-hierarchy.composite"
    )
    layout = plan.native_options["solver_provider"]["facts"]["layout"]
    assert layout["kind"] == "amr"
    assert layout["levels"] == 3
    plan.require_component_inputs((topology, solver))


def test_external_field_solver_v2_refuses_level_local_amr(tmp_path):
    provider, _topology, _solver = _provider(tmp_path)

    with pytest.raises(LoweringRejection, match="supports only hierarchy policy") as error:
        capture_field_plans(
            _case(provider, hierarchy_policy=LevelByLevelSolve()),
            lambda value: value,
            target="amr_system",
            layout=final_amr_layout(cartesian_grid(n=8, periodic=False), max_levels=2, ratio=2),
        )
    assert error.value.gate == "field.solver.provider_incompatible"


def test_external_field_solver_v2_preserves_runtime_selected_amr_ratio(tmp_path):
    provider, topology, solver = _provider(tmp_path)

    plan = capture_field_plans(
        _case(provider, hierarchy_policy=CompositeHierarchySolve()),
        lambda value: value,
        target="amr_system",
        layout=final_amr_layout(
            cartesian_grid(n=8, periodic=False), max_levels=2, ratio=4
        ),
    )["potential"]

    assert plan.native_options["solver_provider"]["facts"]["layout"][
        "transition_ratios"
    ] == ((4, 4),)
    plan.require_component_inputs((topology, solver))


def test_external_field_solver_reports_mpi_only_when_both_host_variants_declare_it(tmp_path):
    topology = _component(
        tmp_path, name="topology_mpi", interface=interfaces.FieldTopology, features=("mpi",))
    solver = _component(
        tmp_path, name="solver_serial", interface=interfaces.FieldSolver)
    provider = ExternalFieldSolver(topology=topology, solver=solver)
    assert provider.capabilities().to_dict()["mpi"] is False
    assert provider.capabilities().to_dict()["component_pair_declares_mpi"] is False

    solver_mpi = _component(
        tmp_path, name="solver_mpi", interface=interfaces.FieldSolver, features=("mpi",))
    mpi_provider = ExternalFieldSolver(topology=topology, solver=solver_mpi)
    assert mpi_provider.capabilities().to_dict()["mpi"] is True
    assert mpi_provider.capabilities().to_dict()["gpu"] is False


def test_external_field_solver_refuses_unsupported_hierarchy_policy_at_resolve(tmp_path):
    provider, _topology, _solver = _provider(tmp_path)

    with pytest.raises(LoweringRejection, match="supports only hierarchy policy") as error:
        capture_field_plans(
            _case(provider, hierarchy_policy=CompositeHierarchySolve()),
            lambda value: value,
            target="system",
            layout=Uniform(cartesian_grid(n=8, periodic=False)),
        )
    assert error.value.gate == "field.solver.provider_incompatible"
    assert "pops.field-hierarchy.composite" in str(error.value)


def test_external_solver_and_topology_roles_are_not_interchangeable(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology)
    solver = _component(
        tmp_path, name="solver", interface=interfaces.FieldSolver)

    with pytest.raises(TypeError, match="solver must implement exact interface"):
        ExternalFieldSolver(topology=topology, solver=topology)
    with pytest.raises(TypeError, match="topology must implement exact interface"):
        ExternalFieldSolver(topology=solver, solver=solver)
