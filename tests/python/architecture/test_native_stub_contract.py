"""Source-only gate for the supported direct ``pops._pops`` stub contract."""
from __future__ import annotations

import ast
import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STUB = REPO_ROOT / "python" / "pops" / "_pops.pyi"
CORE_BINDINGS = REPO_ROOT / "python" / "bindings" / "core" / "init" / "init_core.cpp"
AMR_BINDINGS = REPO_ROOT / "python" / "bindings" / "core" / "init" / "init_amr.cpp"
HDF5_BINDINGS = REPO_ROOT / "python" / "bindings" / "core" / "init" / "init_parallel_hdf5.cpp"

PUBLIC_METADATA = {
    "__version__", "__abi_version__", "__release_contract_sha256__", "__public_api_version__",
    "__semantic_ir_version__", "__normalization_version__", "__component_registry_version__",
    "__checkpoint_schema_version__", "__cxx_std__", "__cxx_compiler__", "__has_kokkos__",
    "__has_mpi__", "__has_parallel_hdf5__", "__mpi_contract__",
    "__aux_named_base__", "__aux_max_extra__",
    "__aux_base_comps__", "__aux_max_comps__", "__max_runtime_params__", "__aux_canonical__",
}
PUBLIC_CALLABLES = {
    "abi_key", "my_rank", "n_ranks", "mpi_world", "module_capabilities", "capability_report",
    "runtime_environment_report", "runtime_backend_manifest", "numerical_defaults_report",
    "fallback_diagnostics_report", "reset_fallback_diagnostics", "kokkos_is_initialized",
}
INTERNAL_BOOTSTRAP_TYPES = {"SystemConfig", "AmrSystemConfig", "ModelSpec", "System", "AmrSystem"}
SYSTEM_CONFIG_FIELDS = {
    "n": "int", "L": "float", "periodic": "bool", "geometry": "str", "nr": "int",
    "ntheta": "int", "r_min": "float", "r_max": "float", "theta_boxes": "int",
}
AMR_CONFIG_FIELDS = {
    "n": "int", "L": "float", "regrid_every": "int", "level_count": "int",
    "regrid_grow": "int", "regrid_margin": "int", "explicit_bootstrap": "bool",
    "periodic": "bool", "distribute_coarse": "bool", "coarse_max_grid": "int",
    "cluster_min_efficiency": "float", "cluster_min_box_size": "int", "cluster_max_box_size": "int",
}


def _tree() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))


def _literal_all(tree: ast.Module) -> tuple[str, ...]:
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )
    return tuple(ast.literal_eval(assignment.value))


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _annotated_fields(cls: ast.ClassDef) -> dict[str, str]:
    return {
        node.target.id: ast.unparse(node.annotation)
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _native_config_fields(path: pathlib.Path, class_name: str) -> set[str]:
    source = path.read_text(encoding="utf-8")
    marker = 'py::class_<%s>(m, "%s")' % (class_name, class_name)
    start = source.find(marker)
    assert start >= 0, "cannot find %s pybind declaration in %s" % (class_name, path)
    end = source.find(";\n\n", start)
    assert end >= 0, "cannot find end of %s pybind declaration in %s" % (class_name, path)
    return set(re.findall(r'\.def_readwrite\("([^"]+)"', source[start:end]))


def test_public_native_stub_surface_is_closed_and_backed_by_cpp():
    tree = _tree()
    public = set(_literal_all(tree))
    native_source = CORE_BINDINGS.read_text(encoding="utf-8")
    native_metadata_source = native_source + HDF5_BINDINGS.read_text(encoding="utf-8")
    native_functions = set(re.findall(r'm\.def\(\s*"([^"]+)"', native_source))
    native_metadata = set(re.findall(r'm\.attr\("([^"]+)"\)', native_metadata_source))

    assert public == {"StepAttemptRejected", *PUBLIC_METADATA, *PUBLIC_CALLABLES}
    assert PUBLIC_CALLABLES == native_functions
    assert PUBLIC_METADATA == native_metadata
    assert not public & INTERNAL_BOOTSTRAP_TYPES
    assert "_System" not in STUB.read_text(encoding="utf-8")
    assert "_AmrSystem" not in STUB.read_text(encoding="utf-8")


def test_bootstrap_types_match_the_native_config_pods_without_dynamic_escape_hatches():
    tree = _tree()
    classes = _classes(tree)

    assert INTERNAL_BOOTSTRAP_TYPES <= classes.keys()
    assert _annotated_fields(classes["SystemConfig"]) == SYSTEM_CONFIG_FIELDS
    assert _annotated_fields(classes["AmrSystemConfig"]) == AMR_CONFIG_FIELDS
    assert set(SYSTEM_CONFIG_FIELDS) == _native_config_fields(CORE_BINDINGS, "SystemConfig")
    assert set(AMR_CONFIG_FIELDS) == _native_config_fields(AMR_BINDINGS, "AmrSystemConfig")

    source = STUB.read_text(encoding="utf-8")
    assert "Any" not in source
    assert "__getattr__" not in source
