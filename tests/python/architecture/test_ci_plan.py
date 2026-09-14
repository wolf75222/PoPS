"""Exercise CI selection only: no selected PoPS test or native build is executed."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ci_plan", ROOT / "scripts/ci_plan.py")
planner = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec.loader.exec_module(planner)
finally:
    sys.path.pop(0)


def test_diff_uses_merge_base_and_preserves_renames_and_deletions(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "CI fixture")
    git("config", "user.email", "ci@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(tmp_path / "no-hooks"))
    (tmp_path / "old name.py").write_text("unchanged content for rename detection\n")
    (tmp_path / "deleted.hpp").write_text("old\n")
    git("add", ".")
    git("commit", "-qm", "common")
    common = git("rev-parse", "HEAD")
    git("checkout", "-qb", "pr")
    git("mv", "old name.py", "new name.py")
    git("rm", "deleted.hpp")
    git("commit", "-qm", "PR changes")
    head = git("rev-parse", "HEAD")
    git("checkout", "--detach", "-q", common)
    (tmp_path / "base-only.cpp").write_text("unrelated later base change\n")
    git("add", ".")
    git("commit", "-qm", "base advanced")
    base = git("rev-parse", "HEAD")
    assert planner.changed_files(base, head, root=tmp_path) == [
        "deleted.hpp", "new name.py", "old name.py"]


@pytest.mark.parametrize("path", [
    "CMakeLists.txt", "python/CMakeLists.txt", "cmake/PoPSConfig.cmake.in",
    "include/pops/core/array.hpp", "include/pops/parallel/mpi.hpp",
    ".github/workflows/ci.yml", "scripts/ci_components.toml",
    "docs/tuto/scalar_advection/08_mpi_amr_explicit_ssprk2.py",
    "tests/python/support/new_helper.py", "new_unmapped_module/input.dat",
    "python/pops/removed_module.py",
])
def test_shared_build_unknown_and_deleted_changes_have_explicit_fallback(path):
    _, reasons = planner.classify([path])
    assert reasons, path


def test_missing_pr_history_emits_full_fallback_input(tmp_path):
    output = tmp_path / "changed.txt"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/ci_plan.py"), "changes",
        "--event-name", "pull_request", "--output-file", str(output),
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == [planner.UNKNOWN_DIFF]
    assert "full fallback" in result.stderr


def test_collective_solver_contracts_keep_mpi_but_local_operator_does_not():
    paths = ["include/pops/numerics/elliptic/interface/field_nullspace.hpp",
             "include/pops/numerics/elliptic/linear/krylov_method_provider.hpp",
             "include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp"]
    impact, reasons = planner.classify(paths)
    assert not reasons
    assert impact[paths[0]]["mpi"] and impact[paths[1]]["mpi"]
    assert not impact[paths[2]]["mpi"]


@pytest.mark.parametrize("event,options", [
    ("schedule", {}), ("pull_request", {"ci_full": True}),
    ("pull_request", {"ci_kokkos": True}), ("workflow_call", {"force_full": True}),
])
def test_scheduled_release_and_explicit_full_overrides(tmp_path, event, options):
    plan = planner.make_plan([], event_name=event, output_dir=tmp_path, **options)
    assert plan["outputs"]["full"]
    assert plan["outputs"]["mpi_required"] and plan["outputs"]["openmp_required"]
    assert plan["outputs"]["cpp_matrix"] == list(range(planner.CPP_SHARDS))


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ci-plans")
    cases = {
        "docs": ["README.md"],
        "python-test": ["tests/python/unit/runtime/test_capacity_limits.py"],
        "dim1-test": ["tests/python/integration/runtime/test_public_drift_diffusion_matrix.py"],
        "cpp-test": ["tests/cpp/unit/mesh/test_box.cpp"],
        "architecture-test": ["tests/python/architecture/test_ci_shard_binpack.py"],
        "operator": ["include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp"],
        "compile-cache": ["tests/python/integration/mpi/test_dsl_compile_cache.py"],
        "unknown-mixed": ["tests/python/unit/runtime/test_capacity_limits.py", "unmapped/input.dat"],
    }
    return {name: planner.make_plan(paths, event_name="pull_request", output_dir=directory / name)
            for name, paths in cases.items()}


@pytest.mark.parametrize("name,cpp,python,architecture", [
    ("docs", [], [], []),
    ("python-test", [], ["tests/python/unit/runtime/test_capacity_limits.py"], []),
    ("cpp-test", ["test_box"], [], []),
    ("architecture-test", [], [], ["tests/python/architecture/test_ci_shard_binpack.py"]),
])
def test_test_only_and_metadata_changes_do_not_run_unrelated_components(plans, name, cpp, python, architecture):
    plan = plans[name]
    assert plan["cpp"] == cpp
    assert plan["python"] == python
    assert plan["architecture"] == architecture
    assert not any(plan["outputs"][flag] for flag in (
        "full", "mpi_required", "openmp_required", "compile_cache_required", "cpp_prewarm_required"))


def test_leaf_operator_has_bounded_native_coverage_without_unrelated_lanes(plans):
    plan = plans["operator"]
    assert not plan["outputs"]["full"]
    assert plan["cpp"], "a native operator requires executable C++ coverage"
    assert len(plan["cpp"]) < 20
    assert "test_polar_tensor_elliptic_mms" in plan["cpp"]
    assert "test_polar_fluid_transport" not in plan["cpp"]
    assert not plan["python"]
    assert not plan["outputs"]["mpi_required"]
    assert not plan["outputs"]["openmp_required"]


def test_compile_cache_runs_only_in_its_dedicated_job(plans):
    plan = plans["compile-cache"]
    assert plan["outputs"]["compile_cache_required"]
    assert not plan["outputs"]["python_required"]
    assert plan["compile_cache"] == ["tests/python/integration/mpi/test_dsl_compile_cache.py"]
    assert not any(plan["python_shards"])
    assert not plan["outputs"]["mpi_required"], "the legacy directory name is not MPI ownership"
    assert plan["outputs"]["python_dimensions"] == [2]


def test_unknown_change_in_a_mixed_pr_keeps_full_required_coverage(plans):
    plan = plans["unknown-mixed"]
    assert plan["outputs"]["full"]
    assert all(plan["outputs"][flag] for flag in (
        "cpp_required", "python_required", "architecture_required", "mpi_required", "openmp_required"))
    assert plan["outputs"]["cpp_matrix"] == list(range(planner.CPP_SHARDS))
    assert plan["outputs"]["python_matrix"] == list(range(planner.PYTHON_SHARDS))
    assert plan["outputs"]["python_dimensions"] == [1, 2]
    assert any("unmapped-path" in reason for reason in plan["full_reasons"])


def test_dynamic_matrices_cover_each_selected_test_once(plans):
    for plan in plans.values():
        ordinary = [test for shard in plan["python_shards"] for test in shard]
        assert len(ordinary) == len(set(ordinary))
        assert set(ordinary).isdisjoint(plan["compile_cache"])
        assert set(ordinary) | set(plan["compile_cache"]) == set(plan["python"])
        nonempty = [index for index, shard in enumerate(plan["python_shards"]) if shard]
        assert plan["outputs"]["python_required"] == bool(nonempty)
        assert plan["outputs"]["python_matrix"] == (nonempty or [0])
    assert plans["python-test"]["outputs"]["python_dimensions"] == [2]
    assert plans["dim1-test"]["outputs"]["python_dimensions"] == [1]
