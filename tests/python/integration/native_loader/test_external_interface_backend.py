"""Collected architecture contract for the closed generated native-interface vocabulary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pops import interfaces
from pops import _generated_component_interfaces as generated
from pops.model import ComponentManifest


def test_all_required_native_families_are_generated_data_only_contracts():
    expected = {
        "numerical_flux", "ghost_boundary", "boundary_flux", "field_boundary_closure", "tagger",
        "clustering", "transfer", "reflux", "field_solver", "writer", "field_topology",
    }
    resolved = {name: interfaces.resolve(name) for name in expected}
    assert set(resolved) == expected
    assert len({value.abi_id for value in resolved.values()}) == len(expected)
    assert {name: value.abi_id for name, value in resolved.items()} == {
        "numerical_flux": 0,
        "ghost_boundary": 1,
        "field_boundary_closure": 2,
        "tagger": 3,
        "clustering": 4,
        "transfer": 5,
        "reflux": 6,
        "field_solver": 7,
        "writer": 8,
        "field_topology": 9,
        "boundary_flux": 10,
    }
    assert all(value.table_symbol == "pops_component_interface_v1"
               for value in resolved.values())
    assert all(value.operations for value in resolved.values())


def test_reflux_is_exact_generated_id_6_and_incomplete_conformer_is_refused():
    interface = interfaces.Reflux
    assert interface.abi_id == 6
    assert interface.uri == "pops://interfaces/reflux"
    assert interface.cpp_table == "PopsRefluxApiV1"
    assert interface.operations == ("apply_interface_batch",)

    incomplete_signature = interface.signature_declaration()
    incomplete_signature["operations"] = ()
    manifest = ComponentManifest(
        uri="pops://external.test/reflux/incomplete",
        component_type="reflux",
        version="1.0.0",
        facets=interface.facets,
        signature={"native_interface": incomplete_signature},
        interfaces=interface.manifest_declarations(),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    with pytest.raises(ValueError, match="does not carry the generated native interface identity"):
        interface.require_manifest(manifest)


def test_python_native_component_boundary_has_no_ffi_or_test_owned_backend():
    root = Path(__file__).resolve().parents[4]
    production = (
        root / "python/pops/interfaces.py",
        root / "python/pops/external/artifacts.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in production)
    assert "import ctypes" not in text
    assert "NativeInterfaceBackend" not in text
    assert "ProbeBinding" not in text


def test_field_topology_and_solver_share_the_topology_contract():
    topology = interfaces.FieldTopology
    solver = interfaces.FieldSolver
    assert topology.operations == ("prepare_topology",)
    assert solver.operations == ("solve",)
    assert topology.version == 2
    assert topology.cpp_table == "PopsFieldTopologyApiV2"
    assert solver.version == 2
    assert solver.cpp_table == "PopsFieldSolverApiV2"
    header = (Path(__file__).resolve().parents[4]
              / "include/pops/runtime/config/generated_component_abi.hpp").read_text(
                  encoding="utf-8")
    assert "PopsInt32ViewV1 component_labels" in header
    assert "const PopsTopologyLabelV2* labels" in header
    assert "typedef struct PopsTopologyLabelV2" in header
    assert "uint32_t struct_size" in header
    assert "PopsFieldGlobalTopologyV1 topology" in header
    assert "const char* source_layout_identity" in header
    assert "const char* materialized_layout_identity" in header
    assert "PopsFieldMaterialRepresentationV1 material_representation" in header
    assert "size_t local_patch_count" in header
    assert "PopsFieldSolverTopologyLabelV2" in header
    assert "size_t topology_label_count" in header
    assert "const char* topology_provenance" in header
    assert header.count("const char* topology_digest") >= 2
    assert "int32_t status" in header
    assert header.count("int32_t action") >= 2
    assert "PopsComponentActionV1 action" not in header
    assert "PopsSolveStatusV2 status" not in header
    assert "PopsSolveActionV2 action" not in header
    assert "double relative_residual" in header
    assert "double reference_residual_norm" in header
    assert "double residual_norm" in header
    assert "const char* reason" in header
    for retired in ("initial_residual", "final_residual"):
        assert retired not in header


def test_common_pod_abi_version_is_generated_and_catalog_authenticated():
    root = Path(__file__).resolve().parents[4]
    catalog = json.loads((root / "schemas/component_catalog.v2.json").read_text(
        encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(
        catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    header = (root / "include/pops/runtime/config/generated_component_abi.hpp").read_text(
        encoding="utf-8")

    assert catalog["native_common_abi_version"] == 1
    assert generated.NATIVE_COMPONENT_COMMON_ABI_VERSION == 1
    assert "#define POPS_COMPONENT_COMMON_ABI_V1 1u" in header
    assert generated.NATIVE_COMPONENT_CATALOG_SHA256 == digest
    assert digest in header


def test_boundary_handle_native_routes_are_generated_from_exact_interfaces():
    assert generated.NATIVE_COMPONENT_BOUNDARY_HANDLE_ROUTES == {
        "boundary_provider": ("ghost_boundary", "apply_region_batch"),
        "corner_resolver": ("ghost_boundary", "apply_region_batch"),
        "numerical_closure": ("ghost_boundary", "apply_region_batch"),
        "conservative_flux": ("numerical_flux", "evaluate_faces"),
        "boundary_flux_provider": ("boundary_flux", "transform_faces"),
        "residual_operator": ("field_boundary_closure", "residual"),
        "linearization_operator": ("field_boundary_closure", "jvp"),
    }
    for interface_name, operation in generated.NATIVE_COMPONENT_BOUNDARY_HANDLE_ROUTES.values():
        assert operation in generated.NATIVE_COMPONENT_INTERFACE_BY_NAME[
            interface_name]["operations"]


def test_uniform_runtime_exposes_the_complete_supported_external_boundary_route():
    """The first production cutover is deliberately the dependency-free GhostBoundary route.

    BoundaryFlux and field residual/JVP remain fail-closed at Python's pre-mutation installer
    lookup until their generated block workspaces have a typed publication seam.
    """
    root = Path(__file__).resolve().parents[4]
    binding = (root / "python/bindings/core/init/init_system.cpp").read_text(encoding="utf-8")
    system = (root / "include/pops/runtime/system.hpp").read_text(encoding="utf-8")
    runtime = (root / "python/pops/runtime/_runtime_authorities.py").read_text(encoding="utf-8")

    assert binding.count('"_install_ghost_boundary_component"') == 1
    assert "stage_prepared_ghost_boundary_component" in binding
    assert "stage_prepared_ghost_boundary_component" in system
    for unsupported in (
        '"_install_boundary_flux_component"',
        '"_install_field_boundary_residual_component"',
        '"_install_field_boundary_jvp_component"',
    ):
        assert unsupported not in binding
    assert (
        '"apply_region_batch": getattr(native, "_install_ghost_boundary_component", None)'
        in runtime
    )
    assert "Uniform native GhostBoundary requires one exact primary-state output" in runtime
