"""A module artifact built on another runner image is rebuilt, never trusted or relaxed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
ACTION = ROOT / ".github/actions/native-module-for-runner/action.yml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "ci_native_build_runner", ROOT / "scripts/ci_native_build_runner.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load()
tree = runner._tree_module()


def _contract(roots: list[Path], compiler: Path) -> dict:
    size, digest = tree.file_digest(str(compiler))
    trees = []
    for root in roots:
        snapshot = tree.snapshot_include_tree(str(root))
        trees.append({key: snapshot[key] for key in ("root", "sha256", "file_count")})
    return {"schema_version": 1,
            "compiler": {"selector": str(compiler), "path": str(compiler),
                         "size": size, "sha256": digest},
            "include_trees": trees}


@pytest.fixture
def system(tmp_path):
    base = (tmp_path / "usr").resolve()
    (base / "include/c++/13").mkdir(parents=True)
    (base / "include/stdio.h").write_text("int printf();\n")
    (base / "include/zlib.h").write_text("#define ZLIB_VERSION \"1.3\"\n")
    (base / "include/c++/13/vector").write_text("namespace std {}\n")
    compiler = base / "g++"
    compiler.write_bytes(b"\x7fELF compiler")
    roots = [base / "include/c++/13", base / "include"]
    package = tmp_path / "artifact"
    package.mkdir()
    (package / "pops-0.3.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"wheel bytes")
    built_packages = {"zlib1g-dev:amd64": "1:1.3.dfsg-3.1ubuntu2", "libc6-dev:amd64": "2.39-0ubuntu8.6"}
    runner.record(2, package, package / runner.RECORD_NAME,
                  contract=_contract(roots, compiler), packages=built_packages,
                  environment={"ImageOS": "ubuntu24", "ImageVersion": "20261101.360.1"})

    def identity() -> Path:
        path = tmp_path / "identity.json"
        path.write_text(json.dumps({"identity": "x", "contract": _contract(roots, compiler)}))
        return path

    return {"base": base, "roots": roots, "compiler": compiler, "package": package,
            "identity": identity, "packages": built_packages}


def _check(system, **overrides):
    files = {"zlib1g-dev:amd64": [str(system["base"] / "include/zlib.h"),
                                  str(system["base"] / "share/doc/zlib")],
             "libc6-dev:amd64": [str(system["base"] / "include/c++/13/vector")]}
    arguments = {"packages": dict(system["packages"]),
                 "environment": {"ImageOS": "ubuntu24", "ImageVersion": "20261111.370.1"},
                 "files_of": lambda name: files.get(name, [])}
    arguments.update(overrides)
    return runner.check(2, system["package"], system["identity"](), **arguments)


def test_same_include_trees_keep_the_downloaded_module(system):
    report = _check(system)
    assert report["rebuild"] is False and report["differing_roots"] == []
    assert "package_changes" not in report
    assert "used as is" in runner.describe(report)[0]


def test_changed_header_requests_rebuild_and_names_images_and_packages(system):
    (system["base"] / "include/zlib.h").write_text("#define ZLIB_VERSION \"1.3.1\"\n")
    here = dict(system["packages"], **{"zlib1g-dev:amd64": "1:1.3.dfsg-3.1ubuntu2.1"})
    report = _check(system, packages=here)
    assert report["rebuild"] is True and report["compiler_changed"] is False
    assert [row["root"] for row in report["differing_roots"]] == [str(system["base"] / "include")]
    assert report["matching_roots"] == [str(system["base"] / "include/c++/13")]
    assert report["package_changes"] == [{
        "package": "zlib1g-dev:amd64", "built": "1:1.3.dfsg-3.1ubuntu2",
        "here": "1:1.3.dfsg-3.1ubuntu2.1",
        "include_files": [str(system["base"] / "include/zlib.h")]}]
    text = "\n".join(runner.describe(report))
    assert "ubuntu24 20261101.360.1" in text and "ubuntu24 20261111.370.1" in text
    assert "zlib1g-dev:amd64 1:1.3.dfsg-3.1ubuntu2 -> 1:1.3.dfsg-3.1ubuntu2.1" in text


def test_file_under_a_matching_deeper_root_does_not_explain_the_difference(system):
    base = system["base"]
    files = runner.files_in_differing_roots(
        [str(base / "include/zlib.h"), str(base / "include/c++/13/vector"), str(base / "lib/x")],
        [str(base / "include")], [str(base / "include/c++/13")])
    assert files == [str(base / "include/zlib.h")]


@pytest.mark.parametrize("change", ["add", "remove", "compiler", "root"])
def test_every_contract_difference_requests_rebuild(system, change):
    base = system["base"]
    if change == "add":
        (base / "include/new.h").write_text("")
    elif change == "remove":
        (base / "include/stdio.h").unlink()
    elif change == "compiler":
        system["compiler"].write_bytes(b"\x7fELF other compiler")
    else:
        # The receipt here no longer lists a built root: it is fingerprinted directly.
        (base / "include/c++/13/vector").write_text("changed")
        identity = system["identity"]()
        data = json.loads(identity.read_text())
        data["contract"]["include_trees"] = data["contract"]["include_trees"][1:]
        identity.write_text(json.dumps(data))
        report = runner.check(2, system["package"], identity, packages={},
                              environment={}, files_of=lambda name: [])
        assert report["rebuild"] is True
        assert [row["root"] for row in report["differing_roots"]] == [
            str(base / "include/c++/13"), str(base / "include")]
        return
    report = _check(system)
    assert report["rebuild"] is True
    assert report["compiler_changed"] is (change == "compiler")


def test_compiler_selector_must_still_resolve_to_the_built_binary(tmp_path):
    first, second = tmp_path / "g++-13", tmp_path / "g++-14"
    first.write_bytes(b"same compiler bytes")
    second.write_bytes(b"same compiler bytes")
    selector = tmp_path / "g++"
    selector.symlink_to(first)
    size, digest = tree.file_digest(str(first))
    compiler = {"selector": str(selector), "path": str(first.resolve()), "size": size,
                "sha256": digest}
    built = {"compiler": compiler, "include_trees": []}
    assert runner.compare(built, {"compiler": dict(compiler), "include_trees": []},
                          snapshot=lambda root: {})["compiler_changed"] is False
    selector.unlink()
    selector.symlink_to(second)
    # Same bytes, same recorded path on this runner: only the selector check sees it.
    assert runner.compare(built, {"compiler": dict(compiler), "include_trees": []},
                          snapshot=lambda root: {})["compiler_changed"] is True


def test_record_must_describe_the_wheel_beside_it(system):
    (system["package"] / "pops-0.3.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"other")
    with pytest.raises(ValueError, match="does not describe the wheel"):
        _check(system)
    (system["package"] / runner.RECORD_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="carries no"):
        _check(system)


def test_record_dimension_is_exact(system):
    with pytest.raises(ValueError, match="expected Dim=1"):
        runner.check(1, system["package"], system["identity"](), packages={}, environment={})


def test_cli_reports_rebuild_and_require_match_fails_by_name(system, tmp_path, capsys, monkeypatch):
    (system["base"] / "include/zlib.h").write_text("changed")
    monkeypatch.setattr(runner, "installed_packages", lambda: dict(system["packages"]))
    monkeypatch.setattr(runner, "package_files", lambda name: [])
    output, summary, receipt = tmp_path / "out", tmp_path / "summary.md", tmp_path / "r.json"
    common = ["check", "--dimension", "2", "--package-dir", str(system["package"]),
              "--identity", str(system["identity"]())]
    assert runner.main([*common, "--github-output", str(output), "--summary", str(summary),
                        "--receipt", str(receipt)]) == 0
    assert output.read_text() == "rebuild=true\n"
    assert "::warning title=Native module built on another runner image::" in capsys.readouterr().out
    assert "Dim2 native module rebuilt for this runner" in summary.read_text()
    assert json.loads(receipt.read_text())["rebuild"] is True
    assert runner.main([*common, "--require-match"]) == 1
    assert "::error title=Native module built on another runner image::" in capsys.readouterr().out


def _job(workflow: str, name: str) -> str:
    block = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-z][a-z0-9-]*:\n", block, maxsplit=1)[0]


def _steps(block: str) -> list[str]:
    return block.split("\n      - ")[1:]


def test_every_module_consumer_checks_the_contract_on_its_own_runner_before_tests():
    workflow = WORKFLOW.read_text()
    shard = _job(workflow, "gate-python")
    for dimension in (1, 2, 3):
        steps = _steps(shard)
        download = next(i for i, step in enumerate(steps)
                        if step.startswith(f"name: Download Dim{dimension} Python module artifact"))
        check = steps[download + 1]
        assert check.startswith(f"name: Dim{dimension} native module for this runner")
        assert f"if: steps.test-plan.outputs.dim{dimension}_count != '0'" in check
        assert "uses: ./.github/actions/native-module-for-runner" in check
        assert f"dimension: '{dimension}'" in check
        assert f"package-dir: .pops-ci/python-packages/dim{dimension}" in check
    cache = _job(workflow, "gate-python-compile-cache")
    steps = _steps(cache)
    download = next(i for i, step in enumerate(steps)
                    if step.startswith("name: Download Python module artifact"))
    assert "uses: ./.github/actions/native-module-for-runner" in steps[download + 1]
    assert "package-dir: .pops-ci/python-packages/dim2" in steps[download + 1]
    for block in (shard, cache):
        # The identity receipt the check reads is written by setup-kokkos, after apt.
        assert block.index("uses: ./.github/actions/setup-kokkos") < block.index(
            "uses: ./.github/actions/native-module-for-runner")
        assert block.index("uses: actions/setup-python@") < block.index(
            "uses: ./.github/actions/native-module-for-runner")
        assert block.rindex("uses: ./.github/actions/native-module-for-runner") < block.index(
            '--packages-root "$PWD/.pops-ci/python-packages"')
    assert workflow.count("uses: ./.github/actions/native-module-for-runner") == 4


def test_build_job_records_its_runner_inside_the_uploaded_artifact():
    build = _job(WORKFLOW.read_text(), "gate-python-build")
    record = build.split("- name: Record the runner that built Dim=${{ matrix.dimension }}", 1)[1]
    record, upload = record.split("- name: Upload Python module artifact", 1)
    assert build.index("scripts/ci_assemble_installed_wheel.py") < build.index(
        "scripts/ci_native_build_runner.py record")
    assert "PYTHONPATH: ${{ github.workspace }}/build-kokkos-py/python" in record
    assert '--wheel-dir "$RUNNER_TEMP/pops-ci-wheel"' in record
    assert '--output "$RUNNER_TEMP/pops-ci-wheel/native-build-runner.json"' in record
    assert "path: ${{ runner.temp }}/pops-ci-wheel/**" in upload


def test_rebuild_replays_the_build_job_recipe_and_proves_the_contract():
    workflow = WORKFLOW.read_text()
    build = _job(workflow, "gate-python-build")
    action = ACTION.read_text()
    configure = re.search(r"cmake --preset ci-kokkos-python (.*?)\n\s*mapfile", build, re.S)
    assert configure is not None
    flags = re.findall(r"-D[A-Z_]+=", configure.group(1))
    assert flags == re.findall(r"-D[A-Z_]+=", action.split("cmake --preset ci-kokkos-python", 1)[1]
                               .split("set +e", 1)[0])
    assert '-DCMAKE_CXX_FLAGS="-ffile-prefix-map=${{ github.workspace }}=."' in action
    # The ccache restore follows the build job's attempt-qualified save key.
    assert ("CCACHE_CACHE_KEY: python-build-dim${{ matrix.dimension }}-${{ github.ref }}-"
            "${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}") in build
    assert ("key: ccache-${{ runner.os }}-python-build-dim${{ inputs.dimension }}-${{ github.ref }}-"
            "${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}") in action
    # actions/cache versions an entry by its literal path: restore under the path the build job
    # saved (its CCACHE_DIR), then hand it to the dimension's own directory.
    assert "CCACHE_DIR: ${{ github.workspace }}/.ccache-py" in build
    assert "key: ccache-${{ runner.os }}-${{ env.CCACHE_CACHE_KEY }}" in build
    restore = action.split("- name: Restore Dim${{ inputs.dimension }} module build ccache", 1)[1]
    restore = restore.split("\n    - name: ", 1)[0]
    assert "path: ${{ github.workspace }}/.ccache-py\n" in restore
    aside = action.split("- name: Set aside this job's ccache", 1)[1].split("\n    - name: ", 1)[0]
    assert 'mv "${{ github.workspace }}/.ccache-py" "$aside"' in aside
    assert action.index("- name: Set aside this job's ccache") < action.index(
        "- name: Restore Dim${{ inputs.dimension }} module build ccache")
    rebuild = action.split("- name: Rebuild Dim${{ inputs.dimension }} module on this runner", 1)[1]
    assert rebuild.index('mv .ccache-py "$CCACHE_DIR"') < rebuild.index('mv "$aside" .ccache-py') \
        < rebuild.index("cmake --preset ci-kokkos-python")
    rebuild, proof = rebuild.split(
        "- name: Prove the rebuilt Dim${{ inputs.dimension }} module contract holds here", 1)
    for command in ("scripts/verify_installed_native.py --expect-dim \"$dim\" --expect-serial",
                    "scripts/ci_assemble_installed_wheel.py",
                    "scripts/ci_native_build_runner.py record",
                    'timeout --signal=TERM --kill-after=30s "${{ inputs.rebuild-minutes }}m"',
                    "rm -rf build-kokkos-py\n"):
        assert command in rebuild
    assert rebuild.index("scripts/verify_installed_native.py") < rebuild.index(
        "scripts/ci_assemble_installed_wheel.py") < rebuild.index("ci_native_build_runner.py record")
    assert "check --require-match" in proof
    # Nothing in the route may relax or bypass the include contract.
    for forbidden in ("POPS_SKIP", "include_contract = None", "--no-verify", "continue-on-error"):
        assert forbidden not in action
    for step in ("Check", "Set aside", "Restore", "Rebuild", "Prove"):
        assert f"- name: {step} " in action


def test_new_runner_check_inputs_force_a_full_python_selection():
    spec = importlib.util.spec_from_file_location("ci_select_tests_runner_check",
                                                  ROOT / "scripts/ci_select_tests.py")
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = selector
    spec.loader.exec_module(selector)
    assert "scripts/ci_native_build_runner.py" in selector.PYTHON_BROAD_FILES
    assert ".github/actions/native-module-for-runner/" in selector.PYTHON_BROAD_PREFIXES
