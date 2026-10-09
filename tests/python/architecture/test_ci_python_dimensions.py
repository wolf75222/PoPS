"""CI must preserve scientific geometry and isolate incompatible native artifacts."""

import json
from types import SimpleNamespace

import pytest

from scripts import ci_python_dimensions as runner


def test_selected_files_are_partitioned_exactly_by_the_explicit_contract():
    contract = json.loads(runner.CONTRACT.read_text())
    drift = "tests/python/integration/runtime/test_public_drift_diffusion_matrix.py"
    ordinary = "tests/python/integration/runtime/test_public_tensor_diffusion.py"
    mixed = "tests/python/integration/runtime/test_api040_m27_mixed_linear_runtime.py"
    assert runner.partition([ordinary, drift, mixed], contract) == {
        1: [drift, mixed], 2: [ordinary]}
    assert contract["files"][mixed] == 1
    for path in contract["files"]:
        assert (runner.ROOT / path).is_file()
    with pytest.raises(ValueError, match="duplicated"):
        runner.partition([drift, drift], contract)
    with pytest.raises(ValueError, match="explicit integers"):
        runner.partition([drift], {"default": 2, "files": {drift: "1"}})


def artifact(tmp_path, dimension):
    p = tmp_path / "packages" / ("dim%d" % dimension)
    p.mkdir(parents=True)
    (p / "pops-0.4.0-cp312-cp312-linux_x86_64.whl").write_bytes(
        b"test subprocess route fixture, not a native wheel"
    )
    return p


def test_true_install_stages_unset_source_paths_and_preserve_groups(tmp_path, monkeypatch):
    for d in (1, 2, 3):
        artifact(tmp_path, d)
    monkeypatch.setenv("PYTHONPATH", "/stale/source")
    monkeypatch.setenv("POPS_INCLUDE", "/stale/include")
    calls = []

    def run(command, **kwargs):
        env = kwargs["env"]
        assert "PYTHONPATH" not in env and "POPS_INCLUDE" not in env
        calls.append((command, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert (
        runner.run_groups(
            {1: ["one.py"], 2: ["two.py"], 3: ["three.py"]},
            tmp_path / "packages",
            tmp_path / "timings",
            install_root=tmp_path / "installed",
        )
        == 0
    )
    assert len(calls) == 15
    for i, d in enumerate((1, 2, 3)):
        group = calls[i * 5 : i * 5 + 5]
        assert "venv" in group[0][0] and "--system-site-packages" in group[0][0]
        assert "--ignore-installed" in group[1][0] and "--prefix" in group[1][0]
        assert any("prove_installed_wheel.py" in arg for arg in group[2][0])
        assert group[3][0][-1] == "--expect-serial"
        assert group[4][0][group[4][0].index("-o") + 1] == "pythonpath="
        assert group[4][1]["POPS_NATIVE_DIM"] == str(d)
        assert (tmp_path / "timings" / ("dim%d" % d) / "selected.txt").read_text() == [
            "one.py\n",
            "two.py\n",
            "three.py\n",
        ][i]


def test_missing_duplicate_wheel_and_checkout_prefix_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.run_groups(
            {2: ["two.py"]}, tmp_path, tmp_path / "timings", install_root=tmp_path / "prefix"
        )
    with pytest.raises(ValueError, match="outsidecheckout|outside checkout"):
        runner.run_groups({}, tmp_path, tmp_path / "timings", install_root=runner.ROOT / "bad")
    p = artifact(tmp_path, 2)
    (p / "pops-duplicate.whl").write_bytes(b"duplicate")
    with pytest.raises(FileNotFoundError):
        runner.run_groups(
            {2: ["two.py"]},
            tmp_path / "packages",
            tmp_path / "timings",
            install_root=tmp_path / "prefix",
        )


def test_install_auth_failure_stops_pytest_and_stays_red(tmp_path, monkeypatch):
    artifact(tmp_path, 2)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=7 if any("prove_installed_wheel.py" in arg for arg in command) else 0
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert (
        runner.run_groups(
            {2: ["two.py"]},
            tmp_path / "packages",
            tmp_path / "timings",
            install_root=tmp_path / "installed",
        )
        == 7
    )
    assert len(calls) == 3
    assert not any("-c" in command for command in calls)


def test_ci_builds_and_downloads_each_declared_dimension():
    workflow = (runner.ROOT / ".github/workflows/ci.yml").read_text()
    mode = workflow.split("\n  set-mode:\n", 1)[1].split("\n  gate-cpp-prewarm:\n", 1)[0]
    prewarm = workflow.split("\n  gate-python-prewarm:\n", 1)[1].split(
        "\n  gate-python-build:\n", 1
    )[0]
    build = workflow.split("\n  gate-python-build:\n", 1)[1].split("\n  gate-python:\n", 1)[0]
    shard = workflow.split("\n  gate-python:\n", 1)[1].split("\n  gate-python-compile-cache:\n", 1)[
        0
    ]
    # Selection/full-plan coverage belongs to test_ci_plan; both producers must
    # consume its same published dimensions, including a plan selecting only Dim1.
    assert "python3 scripts/ci_plan.py plan" in mode
    assert "python_dimensions: ${{ steps.decide.outputs.python_dimensions }}" in mode
    assert '--github-output "$GITHUB_OUTPUT"' in mode
    for producer in (prewarm, build):
        needs = next(
            line.strip() for line in producer.splitlines() if line.strip().startswith("needs:")
        )
        assert "set-mode" in needs
        assert "dimension: ${{ fromJSON(needs.set-mode.outputs.python_dimensions) }}" in producer
        assert "POPS_NATIVE_DIM: ${{ matrix.dimension }}" in producer
    assert "pops-module-dim${{ matrix.dimension }}-" in build
    assert next(
        line.strip() for line in prewarm.splitlines() if "key: pops-module-" in line
    ) == next(line.strip() for line in build.splitlines() if "key: pops-module-" in line)
    assert "name: gate-python-prewarm-dim${{ matrix.dimension }}-${{ matrix.lane }}" in prewarm
    assert "pattern: gate-python-prewarm-dim${{ matrix.dimension }}-*" in build
    assert "name: gate-python-build-kokkos-py-dim${{ matrix.dimension }}" in build
    contract = json.loads(runner.CONTRACT.read_text())
    for dimension in {contract["default"], *contract["files"].values()}:
        download = shard.split(f"- name: Download Dim{dimension} Python module artifact", 1)[
            1
        ].split("\n      - ", 1)[0]
        assert f"if: steps.test-plan.outputs.dim{dimension}_count != '0'" in download
        assert "uses: actions/download-artifact@" in download
        assert f"name: gate-python-build-kokkos-py-dim{dimension}" in download
        assert f"path: .pops-ci/python-packages/dim{dimension}" in download
    assert "scripts/ci_python_dimensions.py --selected-file" in shard
    assert "scripts/ci_assemble_installed_wheel.py" in build
    assert 'POPS_NATIVE_DIM: "2"' not in shard


def test_installed_process_mode_never_adds_source_package_or_pythonpath():
    import ast

    tree = ast.parse((runner.ROOT / "tests/python/conftest.py").read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_process_pythonpath"
    )
    body = ast.get_source_segment((runner.ROOT / "tests/python/conftest.py").read_text(), fn)
    assert "POPS_CI_TEST_EXECUTION" in body and '"installed"' in body
    process = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PythonProcessItem"
    )
    process_text = ast.get_source_segment(
        (runner.ROOT / "tests/python/conftest.py").read_text(), process
    )
    assert 'env.pop("PYTHONPATH", None)' in process_text
    assert "sys.path[:0]=" in process_text


@pytest.mark.parametrize("failed_stage", ("wheel-proof", "pytest"))
def test_dim1_stage_failure_remains_red_and_dim2_still_executes(
    tmp_path, monkeypatch, failed_stage
):
    for dimension in (1, 2):
        artifact(tmp_path, dimension)
    calls = []

    def run(command, **kwargs):
        dimension = kwargs["env"]["POPS_NATIVE_DIM"]
        stage = (
            "wheel-proof"
            if any("prove_installed_wheel.py" in str(arg) for arg in command)
            else "pytest"
            if "-c" in command
            else "other"
        )
        calls.append((dimension, stage, command))
        return SimpleNamespace(returncode=7 if dimension == "1" and stage == failed_stage else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.run_groups(
        {1: ["one.py"], 2: ["two.py"]},
        tmp_path / "packages",
        tmp_path / "timings",
        install_root=tmp_path / "installed",
    )
    assert result == 7
    assert any(dimension == "2" and stage == "wheel-proof" for dimension, stage, _ in calls)
    assert any(dimension == "2" and stage == "pytest" for dimension, stage, _ in calls)
    if failed_stage == "wheel-proof":
        assert not any(dimension == "1" and stage == "pytest" for dimension, stage, _ in calls)
    else:
        assert any(dimension == "1" and stage == "pytest" for dimension, stage, _ in calls)
