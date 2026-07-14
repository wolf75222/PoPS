"""Collected architecture contract for the closed generated native-interface vocabulary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pops import interfaces
from pops import _generated_component_interfaces as generated


def test_all_required_native_families_are_generated_data_only_contracts():
    expected = {
        "numerical_flux", "ghost_boundary", "field_boundary_closure", "tagger",
        "clustering", "transfer", "reflux", "field_solver", "writer", "field_topology",
    }
    resolved = {name: interfaces.resolve(name) for name in expected}
    assert set(resolved) == expected
    assert len({value.abi_id for value in resolved.values()}) == len(expected)
    assert all(value.table_symbol == "pops_component_interface_v1"
               for value in resolved.values())
    assert all(value.operations for value in resolved.values())


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
    header = (Path(__file__).resolve().parents[4]
              / "include/pops/runtime/config/generated_component_abi.hpp").read_text(
                  encoding="utf-8")
    assert "PopsInt32ViewV1 component_labels" in header
    assert "const PopsTopologyLabelV1* labels" in header
    assert header.count("const char* topology_digest") >= 2


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
        "residual_operator": ("field_boundary_closure", "residual"),
        "linearization_operator": ("field_boundary_closure", "jvp"),
    }
    for interface_name, operation in generated.NATIVE_COMPONENT_BOUNDARY_HANDLE_ROUTES.values():
        assert operation in generated.NATIVE_COMPONENT_INTERFACE_BY_NAME[
            interface_name]["operations"]
