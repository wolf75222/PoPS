"""CI must preserve scientific geometry and isolate incompatible native artifacts."""
import json
from types import SimpleNamespace

import pytest

from scripts import ci_python_dimensions as runner


def test_selected_files_are_partitioned_exactly_by_the_explicit_contract():
    contract = json.loads(runner.CONTRACT.read_text())
    drift = "tests/python/integration/runtime/test_public_drift_diffusion_matrix.py"
    ordinary = "tests/python/integration/runtime/test_public_tensor_diffusion.py"
    assert runner.partition([ordinary, drift], contract) == {1: [drift], 2: [ordinary]}
    for path in contract["files"]:
        assert (runner.ROOT / path).is_file()
    with pytest.raises(ValueError, match="duplicated"):
        runner.partition([drift, drift], contract)
    with pytest.raises(ValueError, match="explicit integers"):
        runner.partition([drift], {"default": 2, "files": {drift: "1"}})


def test_groups_use_separate_authenticated_packages_and_retain_failures(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    monkeypatch.setenv("PYTHONPATH", "/stale/dim2/package")
    monkeypatch.setenv("POPS_NATIVE_VARIANTS_ROOT", "/stale/dim2/variants")
    for dimension in (1, 2):
        manifest = tmp_path / f"packages/dim{dimension}/pops/_native/variants.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")

    def run(command, *, cwd, env, check):
        calls.append((command, env))
        assert cwd == runner.ROOT and check is False
        assert "/stale/" not in env["PYTHONPATH"]
        assert "POPS_NATIVE_VARIANTS_ROOT" not in env
        # A failure in the first process must remain red, while the second still runs.
        return SimpleNamespace(returncode=1 if "pytest" in command and env["POPS_NATIVE_DIM"] == "1" else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.run_groups({1: ["line.py"], 2: ["plane.py"]},
                             tmp_path / "packages", tmp_path / "timings") == 1
    assert len(calls) == 4
    for offset, dimension, path in ((0, 1, "line.py"), (2, 2, "plane.py")):
        verify, environment = calls[offset]
        command, pytest_environment = calls[offset + 1]
        assert verify[-3:] == ["--expect-dim", str(dimension), "--expect-serial"]
        assert environment == pytest_environment
        assert environment["POPS_NATIVE_DIM"] == str(dimension)
        assert environment["POPS_INCLUDE"] == str(runner.ROOT / "include")
        assert environment["PYTHONPATH"].endswith(f"/packages/dim{dimension}")
        assert environment["POPS_CI_PYTEST_TIMINGS_DIR"].endswith(f"/timings/dim{dimension}")
        assert "pytest" in command and command[-1] == path
        assert (tmp_path / f"timings/dim{dimension}/selected.txt").read_text() == path + "\n"


def test_missing_or_unauthenticated_native_package_cannot_run_tests(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="Dim1"):
        runner.run_groups({1: ["line.py"]}, tmp_path, tmp_path / "timings")
    manifest = tmp_path / "dim1/pops/_native/variants.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    calls = []

    def reject(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", reject)
    assert runner.run_groups({1: ["line.py"]}, tmp_path, tmp_path / "timings") == 7
    assert len(calls) == 1 and "pytest" not in calls[0]


def test_ci_builds_and_downloads_each_declared_dimension():
    workflow = (runner.ROOT / ".github/workflows/ci.yml").read_text()
    mode = workflow.split("\n  set-mode:\n", 1)[1].split("\n  gate-cpp-prewarm:\n", 1)[0]
    prewarm = workflow.split("\n  gate-python-prewarm:\n", 1)[1].split("\n  gate-python-build:\n", 1)[0]
    build = workflow.split("\n  gate-python-build:\n", 1)[1].split("\n  gate-python:\n", 1)[0]
    shard = workflow.split("\n  gate-python:\n", 1)[1].split("\n  gate-python-compile-cache:\n", 1)[0]
    # Selection/full-plan coverage belongs to test_ci_plan; both producers must
    # consume its same published dimensions, including a plan selecting only Dim1.
    assert "python3 scripts/ci_plan.py plan" in mode
    assert 'python_dimensions: ${{ steps.decide.outputs.python_dimensions }}' in mode
    assert '--github-output "$GITHUB_OUTPUT"' in mode
    for producer in (prewarm, build):
        needs = next(line.strip() for line in producer.splitlines() if line.strip().startswith("needs:"))
        assert "set-mode" in needs
        assert "dimension: ${{ fromJSON(needs.set-mode.outputs.python_dimensions) }}" in producer
        assert "POPS_NATIVE_DIM: ${{ matrix.dimension }}" in producer
    assert "pops-module-dim${{ matrix.dimension }}-" in build
    assert next(line.strip() for line in prewarm.splitlines() if "key: pops-module-" in line) == next(
        line.strip() for line in build.splitlines() if "key: pops-module-" in line)
    assert "name: gate-python-prewarm-dim${{ matrix.dimension }}-${{ matrix.lane }}" in prewarm
    assert "pattern: gate-python-prewarm-dim${{ matrix.dimension }}-*" in build
    assert "name: gate-python-build-kokkos-py-dim${{ matrix.dimension }}" in build
    contract = json.loads(runner.CONTRACT.read_text())
    for dimension in {contract["default"], *contract["files"].values()}:
        download = shard.split(f"- name: Download Dim{dimension} Python module artifact", 1)[1].split("\n      - ", 1)[0]
        assert f"if: steps.test-plan.outputs.dim{dimension}_count != '0'" in download
        assert "uses: actions/download-artifact@" in download
        assert f"name: gate-python-build-kokkos-py-dim{dimension}" in download
        assert f"path: .pops-ci/python-packages/dim{dimension}" in download
    assert "scripts/ci_python_dimensions.py --selected-file" in shard
    assert 'POPS_NATIVE_DIM: "2"' not in shard
