"""Final cut-over fences: one authoring surface and no importable legacy facade."""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import re
import subprocess

import pops
import pytest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "python" / "pops"
TESTS = ROOT / "tests" / "python"


_RETIRED_ROOT_ENGINE_NAMES = frozenset({
    "AmrRuntime", "AmrSystem", "AmrSystemConfig", "AuxHalo", "BackgroundDensity",
    "ChargeDensity", "ChargeDensitySource", "Collision", "CompiledSimulationArtifact",
    "CompositeModel", "CompositeRhs", "CompressibleFlux", "Dirichlet", "DivEpsGrad",
    "ElectricFieldFromPotential", "EllipticModel", "EllipticSolver", "ExB", "Explicit",
    "FiniteVolume", "FluidState", "GravityCoupling", "GravityForce", "IMEX", "IMEXRK",
    "Implicit", "Ionization", "IsothermalFlux", "MagneticLorentzForce", "ModelSpec",
    "Neumann", "NoSource", "PerformanceSummary", "Periodic", "PolarMesh", "PotentialForce",
    "PotentialMagneticForce", "Profile", "PythonFlux", "ReportTree", "Role", "Scalar",
    "SourceImplicit", "SourceImplicitBE", "Spatial", "System", "SystemConfig",
    "ThermalExchange",
})
_LEGACY_MODEL_ARGUMENTS = frozenset({"state", "transport", "source", "elliptic"})


def _tracked_python_tests() -> tuple[Path, ...]:
    relative_paths = subprocess.check_output(
        ["git", "ls-files", "tests/python"], cwd=ROOT, text=True
    ).splitlines()
    return tuple(
        ROOT / relative
        for relative in relative_paths
        if relative.endswith(".py")
        and not relative.endswith((" 2.py", " 3.py"))
        and (ROOT / relative).is_file()
    )


def _is_root_attribute(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pops"
        and node.attr == name
    )


def _legacy_root_engine_uses(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "pops"
            and node.attr in _RETIRED_ROOT_ENGINE_NAMES
        ):
            violations.append("%d: pops.%s" % (node.lineno, node.attr))
        elif isinstance(node, ast.ImportFrom) and node.module == "pops":
            for alias in node.names:
                if alias.name in _RETIRED_ROOT_ENGINE_NAMES:
                    violations.append("%d: from pops import %s" % (node.lineno, alias.name))
        elif isinstance(node, ast.Call) and _is_root_attribute(node.func, "Model"):
            keywords = {keyword.arg for keyword in node.keywords}
            if len(node.args) >= 4 or keywords & _LEGACY_MODEL_ARGUMENTS:
                violations.append("%d: legacy engine pops.Model(...) signature" % node.lineno)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for name in _RETIRED_ROOT_ENGINE_NAMES:
                if re.search(r"(?<![_A-Za-z0-9])pops\.%s\b" % re.escape(name), node.value):
                    violations.append("%d: literal pops.%s" % (node.lineno, name))
    return violations


def _removed_names() -> tuple[str, ...]:
    # Keep the retired spellings out of normative source scans performed by this test itself.
    return (
        "Pro" + "blem",
        "Runtime" + "Policies",
        "Output" + "Policy",
        "Checkpoint" + "Policy",
        "Sys" + "tem",
        "Amr" + "System",
        "Model" + "Spec",
    )


def _removed_amr_authorities() -> tuple[str, ...]:
    return (
        "Checkpoint" + "Policy",
        "AMR" + "Output",
        "All" + "Levels",
        "Coarse" + "Only",
        "Selected" + "Levels",
        "Priority" + "Order",
    )


def test_retired_root_and_runtime_exports_are_absent() -> None:
    from pops import runtime

    for name in _removed_names():
        assert name not in pops.__all__
        assert not hasattr(pops, name)
    for name in _removed_names()[-3:]:
        assert name not in runtime.__all__
        assert not hasattr(runtime, name)


def test_removed_public_modules_do_not_exist() -> None:
    for relative in (
        "case.py",
        "dsl.py",
        "output/policies.py",
        "output/runtime_policies.py",
        "runtime_policies.py",
        "mesh/cartesian.py",
    ):
        assert not (PACKAGE / relative).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.mesh.cartesian")


def test_case_has_one_registration_spelling_per_authority() -> None:
    case = pops.Case("canonical")
    assert hasattr(case, "block") and not hasattr(case, "add_block")
    assert hasattr(case, "field") and not hasattr(case, "add_field")
    assert hasattr(case, "program") and not hasattr(case, "time")
    assert hasattr(case, "consumers") and not hasattr(case, "output")


def test_amr_has_one_checkpoint_output_and_tagging_authority_path() -> None:
    from pops import amr as authoring_amr
    import pops.mesh as public_mesh
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.mesh.amr")
    mesh_amr = importlib.import_module("pops.mesh._amr")
    from pops.layouts import AMR

    assert "amr" not in public_mesh.__all__
    assert "Availability" not in mesh_amr.__all__
    assert "IgnoreAMRCriteria" in authoring_amr.__all__
    assert not hasattr(mesh_amr, "LEGACY_CONFIG_LEVELS")
    assert not (PACKAGE / "amr" / "resolution.py").exists()
    assert (PACKAGE / "amr" / "_resolution.py").is_file()
    assert not (PACKAGE / "mesh" / "amr").exists()
    assert (PACKAGE / "mesh" / "_amr").is_dir()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.amr.resolution")
    for internal in (
        "AMRLayoutResolver",
        "AMRResolutionContext",
        "AMRTaggingResolutionContext",
        "ResolvedAMRAuthorities",
        "ResolvedTaggingAuthority",
        "resolve_amr_authorities",
        "resolve_tagging",
    ):
        assert internal not in authoring_amr.__all__
        assert not hasattr(authoring_amr, internal)
    removed = _removed_amr_authorities()
    for name in removed[:5]:
        assert name not in mesh_amr.__all__
        assert not hasattr(mesh_amr, name)
    assert removed[5] not in authoring_amr.__all__
    assert not hasattr(authoring_amr, removed[5])

    layout_parameters = inspect.signature(AMR).parameters
    assert "checkpoint" not in layout_parameters
    assert "output" not in layout_parameters

    tagging_parameters = inspect.signature(authoring_amr.AMRTagging).parameters
    for name in ("hysteresis", "conflict_policy"):
        assert tagging_parameters[name].default is inspect.Parameter.empty
    for name in ("Hysteresis", "EqualityPolicy", "ConflictPolicy"):
        assert name in authoring_amr.__all__
        assert hasattr(authoring_amr, name)

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            PACKAGE / "amr" / "authoring.py",
            PACKAGE / "mesh" / "_amr" / "__init__.py",
            PACKAGE / "layouts" / "__init__.py",
        )
    )
    for name in removed:
        assert name not in source


def test_mesh_layouts_legacy_package_is_absent() -> None:
    assert not (PACKAGE / "mesh" / "layouts" / "__init__.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.mesh." + "layouts")


def test_program_has_one_runtime_branch_spelling() -> None:
    source = (PACKAGE / "time" / "program_authoring.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    authoring = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ProgramAuthoring")
    methods = {
        node.name for node in authoring.body if isinstance(node, ast.FunctionDef)}
    assert "branch" in methods
    assert "if_" not in methods


def test_time_presets_are_factories_for_ordinary_programs() -> None:
    from pops.lib import time

    assert callable(time.SSPRK2)
    assert not hasattr(time, "ssprk2")


def test_tracked_tests_never_consume_retired_root_engine_attributes() -> None:
    violations = []
    for path in _tracked_python_tests():
        for detail in _legacy_root_engine_uses(path):
            violations.append("%s:%s" % (path.relative_to(ROOT), detail))
    assert not violations, (
        "tests must import native descriptors from pops.runtime._engine_descriptors, native "
        "systems from pops.runtime._system, and domain values from their real owner:\n  "
        + "\n  ".join(violations)
    )
