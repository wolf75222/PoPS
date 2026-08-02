"""Source-only contract checks for the final release gate (ADC-695)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("final_release_contract", SCRIPTS / "final_release_contract.py")
gate = _load("_final_release_gate_test", SCRIPTS / "run_final_gate.py")
preflight = _load("_release_preflight_test", SCRIPTS / "release_preflight.py")
installed = _load("_installed_wheel_proof_test", SCRIPTS / "prove_installed_wheel.py")
example_runner = _load("_installed_example_test", SCRIPTS / "run_installed_example.py")


def _release_contract(version: str = "0.3.0"):
    return type(
        "ReleaseContract",
        (),
        {
            "PACKAGE_VERSION": version,
            "SUPPORTED_MATRIX": {
                "wheels": (
                    {
                        "os": "macos",
                        "arch": "arm64",
                        "python": "cp312",
                        "backend": "Kokkos Serial",
                    },
                ),
            },
        },
    )


def _write_release_wheel(
    path: Path,
    *,
    version: str = "0.3.0",
    tag: str = "cp312-cp312-macosx_11_0_arm64",
    purelib: str = "false",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"pops-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.3\nName: PoPS\nVersion: {version}\n",
        )
        archive.writestr(
            f"pops-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: ADC-688 test\n"
            f"Root-Is-Purelib: {purelib}\n"
            f"Tag: {tag}\n",
        )


def _write_final_source_tree(root: Path) -> None:
    specification = root / contract.FINAL_SPECIFICATION
    specification.parent.mkdir(parents=True)
    specification.write_text("# Specification Technique Finale\n", encoding="utf-8")
    for example in contract.FINAL_EXAMPLES:
        path = root / example
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "--output-dir\n"
            + "\n".join(contract.REQUIRED_PROOF_MARKERS)
            + "\nif __name__ == \"__main__\":\n    pass\n",
            encoding="utf-8",
        )
    test_sources = {}
    for ledger in (
        contract.FINAL_EXAMPLE_ACCEPTANCE_TESTS,
        contract.FINAL_EXAMPLE_QUALIFICATION_TESTS,
    ):
        for example, nodeid in zip(contract.FINAL_EXAMPLES, ledger, strict=True):
            relative, function_name = nodeid.split("::", 1)
            entry = test_sources.setdefault(relative, [example.name, []])
            assert entry[0] == example.name
            entry[1].append(function_name)
    for relative, (example_name, function_names) in test_sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "EXAMPLE = %r\n\n%s\n"
            % (
                example_name,
                "\n\n".join(
                    "def %s():\n    pass" % function_name
                    for function_name in function_names
                ),
            ),
            encoding="utf-8",
        )


def test_final_release_source_contract_accepts_exact_canonical_set(tmp_path):
    _write_final_source_tree(tmp_path)

    assert contract.source_contract_errors(tmp_path) == []


def test_final_release_source_contract_refuses_missing_and_extra_examples(tmp_path):
    _write_final_source_tree(tmp_path)
    (tmp_path / contract.FINAL_EXAMPLES[-1]).unlink()
    extra = tmp_path / "examples/final/temporary.py"
    extra.write_text("pass\n", encoding="utf-8")

    errors = contract.source_contract_errors(tmp_path)

    assert any("final examples must be exactly" in error for error in errors)


def test_final_release_source_contract_requires_executable_restart_output_proof(tmp_path):
    _write_final_source_tree(tmp_path)
    path = tmp_path / contract.FINAL_EXAMPLES[0]
    path.write_text("if __name__ == \"__main__\":\n    pass\n", encoding="utf-8")

    errors = contract.source_contract_errors(tmp_path)

    assert any("--output-dir" in error for error in errors)
    assert any("lacks final proof markers" in error for error in errors)


def test_final_release_source_contract_requires_exact_mandatory_example_tests(tmp_path):
    _write_final_source_tree(tmp_path)
    nodeid = contract.FINAL_EXAMPLE_ACCEPTANCE_TESTS[-1]
    relative, _function_name = nodeid.split("::", 1)
    (tmp_path / relative).write_text(
        "EXAMPLE = 'wrong.py'\n"
        "@pytest.mark.skip(reason='optional')\n"
        "def renamed_test():\n"
        "    pass\n",
        encoding="utf-8",
    )

    errors = contract.source_contract_errors(tmp_path)

    assert any("must resolve exactly once" in error for error in errors)


def test_final_release_source_contract_refuses_duplicate_required_nodeids(
    monkeypatch, tmp_path
):
    _write_final_source_tree(tmp_path)
    monkeypatch.setattr(
        contract,
        "FINAL_EXAMPLE_QUALIFICATION_TESTS",
        (
            contract.FINAL_EXAMPLE_ACCEPTANCE_TESTS[0],
            *contract.FINAL_EXAMPLE_QUALIFICATION_TESTS[1:],
        ),
    )

    errors = contract.source_contract_errors(tmp_path)

    assert "final-example required test nodeids must be unique" in errors


@pytest.mark.parametrize(
    "injected",
    (
        "\nfrom unittest.mock import patch\n",
        "\npytestmark = pytest.mark.xfail(reason='optional')\n",
    ),
)
def test_final_release_source_contract_refuses_mocked_or_optional_required_tests(
    tmp_path, injected
):
    _write_final_source_tree(tmp_path)
    nodeid = contract.FINAL_EXAMPLE_ACCEPTANCE_TESTS[0]
    relative, _function_name = nodeid.split("::", 1)
    path = tmp_path / relative
    path.write_text(path.read_text(encoding="utf-8") + injected, encoding="utf-8")

    errors = contract.source_contract_errors(tmp_path)

    assert any(nodeid in error and "optional" in error for error in errors)


@pytest.mark.parametrize("module", ("pops.ir", "pops._ir"))
def test_final_release_source_contract_refuses_internal_or_transitional_imports(
    tmp_path, module
):
    _write_final_source_tree(tmp_path)
    path = tmp_path / contract.FINAL_EXAMPLES[0]
    path.write_text(
        path.read_text(encoding="utf-8") + f"\nfrom {module} import ValueExpr\n",
        encoding="utf-8",
    )

    errors = contract.source_contract_errors(tmp_path)

    assert any("transitional/internal authoring names" in error for error in errors)


def test_required_junit_lane_rejects_skips_xfails_failures_and_empty_reports(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite tests="1"><testcase name="ok"/></testsuite>', encoding="utf-8")
    assert gate._junit_summary(report)["tests"] == 1

    for child in ('<skipped type="pytest.xfail"/>', '<failure/>', '<error/>'):
        report.write_text(
            '<testsuite tests="1"><testcase name="bad">%s</testcase></testsuite>' % child,
            encoding="utf-8",
        )
        with pytest.raises(gate.FinalGateError):
            gate._junit_summary(report)

    report.write_text('<testsuite tests="0"/>', encoding="utf-8")
    with pytest.raises(gate.FinalGateError):
        gate._junit_summary(report)


def test_required_junit_lane_authenticates_exact_final_example_tests(tmp_path):
    cases = []
    for nodeid in contract.FINAL_EXAMPLE_REQUIRED_TESTS:
        relative, function_name = nodeid.split("::", 1)
        classname = str(Path(relative).with_suffix("")).replace("/", ".")
        cases.append(
            '<testcase classname="%s" name="%s"/>' % (classname, function_name)
        )
    report = tmp_path / "final-examples.xml"
    report.write_text(
        '<testsuite tests="%d">%s</testsuite>'
        % (len(cases), "".join(cases)),
        encoding="utf-8",
    )

    assert gate._require_junit_nodeids(
        report, contract.FINAL_EXAMPLE_REQUIRED_TESTS
    ) == list(contract.FINAL_EXAMPLE_REQUIRED_TESTS)

    report.write_text(
        '<testsuite tests="%d">%s</testsuite>'
        % (len(cases) - 1, "".join(cases[:-1])),
        encoding="utf-8",
    )
    with pytest.raises(gate.FinalGateError, match="appears 0 times"):
        gate._require_junit_nodeids(
            report, contract.FINAL_EXAMPLE_REQUIRED_TESTS
        )

    report.write_text(
        '<testsuite tests="%d">%s%s</testsuite>'
        % (len(cases) + 1, "".join(cases), cases[0]),
        encoding="utf-8",
    )
    with pytest.raises(gate.FinalGateError, match="appears 2 times"):
        gate._require_junit_nodeids(
            report, contract.FINAL_EXAMPLE_REQUIRED_TESTS
        )


def test_release_preflight_reauthenticates_junit_all_pass_and_exact_nodeids(tmp_path):
    cases = []
    for nodeid in contract.FINAL_EXAMPLE_REQUIRED_TESTS:
        relative, function_name = nodeid.split("::", 1)
        classname = str(Path(relative).with_suffix("")).replace("/", ".")
        cases.append(
            '<testcase classname="%s" name="%s"/>' % (classname, function_name)
        )
    report = tmp_path / "final-examples.xml"

    def write_report(rows):
        report.write_text(
            '<testsuite tests="%d">%s</testsuite>'
            % (len(rows), "".join(rows)),
            encoding="utf-8",
        )

    def lane(*, tests, failures=0, skips=0):
        return {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "tests": tests,
            "failures": failures,
            "skips_or_xfails": skips,
        }

    write_report(cases)
    preflight._junit_evidence(
        report,
        lane(tests=len(cases)),
        required_nodeids=contract.FINAL_EXAMPLE_REQUIRED_TESTS,
    )

    write_report(cases[:-1])
    with pytest.raises(preflight.PreflightError, match="appears 0 times"):
        preflight._junit_evidence(
            report,
            lane(tests=len(cases) - 1),
            required_nodeids=contract.FINAL_EXAMPLE_REQUIRED_TESTS,
        )

    write_report([*cases, cases[0]])
    with pytest.raises(preflight.PreflightError, match="appears 2 times"):
        preflight._junit_evidence(
            report,
            lane(tests=len(cases) + 1),
            required_nodeids=contract.FINAL_EXAMPLE_REQUIRED_TESTS,
        )

    failed = cases[0].replace("/>", "><failure/></testcase>")
    write_report([failed, *cases[1:]])
    with pytest.raises(preflight.PreflightError, match="not all-pass"):
        preflight._junit_evidence(
            report,
            lane(tests=len(cases), failures=1),
            required_nodeids=contract.FINAL_EXAMPLE_REQUIRED_TESTS,
        )

    xfailed = cases[0].replace(
        "/>", '><skipped type="pytest.xfail"/></testcase>'
    )
    write_report([xfailed, *cases[1:]])
    with pytest.raises(preflight.PreflightError, match="not all-pass"):
        preflight._junit_evidence(
            report,
            lane(tests=len(cases), skips=1),
            required_nodeids=contract.FINAL_EXAMPLE_REQUIRED_TESTS,
        )


def test_required_python_lane_makes_xpass_fatal():
    source = (SCRIPTS / "run_final_gate.py").read_text(encoding="utf-8")

    assert '"-o", "xfail_strict=true"' in source


def test_release_preflight_requires_the_exact_final_example_test_ledger():
    evidence = {
        "final_example_nodeids": list(contract.FINAL_EXAMPLE_REQUIRED_TESTS),
    }

    preflight._final_example_test_evidence(evidence)
    evidence["final_example_nodeids"] = evidence["final_example_nodeids"][:-1]
    with pytest.raises(preflight.PreflightError, match="test ledger drifted"):
        preflight._final_example_test_evidence(evidence)


def test_required_python_lane_rejects_script_style_hidden_skips():
    gate._require_no_hidden_skip("42 tests passed")
    with pytest.raises(gate.FinalGateError, match="hidden skip"):
        gate._require_no_hidden_skip("skip (native engine unavailable)\n1 passed")


def test_final_gate_pins_one_conda_environment_and_native_headers(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("POPS_CONDA_EXE", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    executable = tmp_path / "conda"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(gate.shutil, "which", lambda name: str(executable))
    monkeypatch.setenv("POPS_ENV_NAME", "pops-proof")
    command = gate._conda_command(["python", "-c", "import pops"])

    assert command[:6] == [
        str(executable.resolve()), "run", "--no-capture-output", "-n", "pops-proof",
        "/usr/bin/env",
    ]
    assert "POPS_INCLUDE=" + str((ROOT / "include").resolve()) in command
    assert "POPS_REQUIRE_NATIVE_TESTS=1" in command
    assert command[-3:] == ["python", "-c", "import pops"]
    assert "bash" not in command


def test_final_gate_honours_explicit_conda_executable(monkeypatch, tmp_path):
    executable = tmp_path / "conda"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("POPS_CONDA_EXE", str(executable))
    monkeypatch.delenv("CONDA_EXE", raising=False)

    command = gate._conda_command(["python", "-V"])

    assert command[0] == str(executable.resolve())
    assert command[-2:] == ["python", "-V"]


def test_artifact_reopen_requires_and_records_npz(tmp_path):
    (tmp_path / "state.h5").write_bytes(b"\x89HDF\r\n\x1a\ncontent")
    (tmp_path / "state.vtu").write_text("<VTKFile/>", encoding="utf-8")
    npz = tmp_path / "state.npz"
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("state.npy", b"payload")

    evidence, hdf5_paths, npz_paths = gate._reopen_outputs(
        tmp_path, example=Path("final.py"))

    assert set(evidence) == {"hdf5", "npz", "paraview"}
    assert hdf5_paths == (tmp_path / "state.h5",)
    assert npz_paths == (npz,)
    npz.unlink()
    with pytest.raises(gate.FinalGateError, match="HDF5, NPZ and ParaView"):
        gate._reopen_outputs(tmp_path, example=Path("final.py"))


def test_release_evidence_authenticates_the_exact_retained_wheel(tmp_path):
    wheel = tmp_path / "wheels" / "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl"
    _write_release_wheel(wheel)
    gates = {
        "official_build": {
            "evidence": {
                "wheel": {
                    "path": str(wheel.relative_to(tmp_path)),
                    "sha256": gate._sha256(wheel),
                    "size": wheel.stat().st_size,
                },
            },
        },
    }
    release = _release_contract()

    preflight._wheel_evidence(tmp_path, gates, release)
    gates["official_build"]["evidence"]["wheel"]["size"] += 1
    with pytest.raises(preflight.PreflightError, match="size drifted"):
        preflight._wheel_evidence(tmp_path, gates, release)


@pytest.mark.parametrize(
    ("filename", "tag", "purelib", "message"),
    (
        (
            "pops-0.3.0-cp311-cp311-macosx_11_0_arm64.whl",
            "cp311-cp311-macosx_11_0_arm64",
            "false",
            "Python/ABI tags",
        ),
        (
            "pops-0.3.0-cp312-cp312-macosx_11_0_universal2.whl",
            "cp312-cp312-macosx_11_0_universal2",
            "false",
            "platform tag",
        ),
        (
            "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl",
            "cp311-cp311-macosx_11_0_arm64",
            "false",
            "WHEEL metadata",
        ),
        (
            "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl",
            "cp312-cp312-macosx_11_0_arm64",
            "true",
            "WHEEL metadata",
        ),
        (
            "pops-0.3.0-1-cp312-cp312-macosx_11_0_arm64.whl",
            "cp312-cp312-macosx_11_0_arm64",
            "false",
            "build tag",
        ),
    ),
)
def test_release_evidence_refuses_wheel_lane_drift(
    tmp_path, filename, tag, purelib, message,
):
    wheel = tmp_path / "wheels" / filename
    _write_release_wheel(wheel, tag=tag, purelib=purelib)
    gates = {
        "official_build": {
            "evidence": {
                "wheel": {
                    "path": str(wheel.relative_to(tmp_path)),
                    "sha256": gate._sha256(wheel),
                    "size": wheel.stat().st_size,
                },
            },
        },
    }

    with pytest.raises(preflight.PreflightError, match=message):
        preflight._wheel_evidence(tmp_path, gates, _release_contract())


def test_installed_wheel_proof_requires_exact_native_member_and_direct_url(tmp_path):
    wheel = tmp_path / "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl"
    native_bytes = b"exact wheel extension"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pops/__init__.py", "__version__ = '0.3.0'\n")
        archive.writestr("pops/_pops.cpython-312-darwin.so", native_bytes)
        archive.writestr(
            "pops-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: PoPS\nVersion: 0.3.0\n",
        )
    package = tmp_path / "site-packages" / "pops" / "__init__.py"
    extension = package.parent / "_pops.cpython-312-darwin.so"
    distribution = package.parents[1]
    package.parent.mkdir(parents=True)
    package.write_text("__version__ = '0.3.0'\n", encoding="utf-8")
    extension.write_bytes(native_bytes)
    metadata = distribution / "pops-0.3.0.dist-info" / "METADATA"
    metadata.parent.mkdir()
    metadata.write_text(
        "Metadata-Version: 2.3\nName: PoPS\nVersion: 0.3.0\n",
        encoding="utf-8",
    )
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    direct_url = {
        "archive_info": {"hashes": {"sha256": wheel_sha256}},
        "url": wheel.as_uri(),
    }

    proof = installed.build_proof(
        wheel,
        package_file=package,
        native_extension=extension,
        distribution_root=distribution,
        python_executable=Path(sys.executable),
        installed_version="0.3.0",
        direct_url=direct_url,
    )

    assert proof["wheel_sha256"] == wheel_sha256
    assert proof["native_sha256"] == hashlib.sha256(native_bytes).hexdigest()
    assert proof["installed_member_count"] == 3
    extension.write_bytes(b"not the retained wheel")
    with pytest.raises(installed.InstalledWheelProofError, match="not byte-identical"):
        installed.build_proof(
            wheel,
            package_file=package,
            native_extension=extension,
            distribution_root=distribution,
            python_executable=Path(sys.executable),
            installed_version="0.3.0",
            direct_url=direct_url,
        )


def test_installed_wheel_resolver_never_imports_unsigned_native_extension() -> None:
    source = (SCRIPTS / "prove_installed_wheel.py").read_text(encoding="utf-8")
    resolver = source.split("def installed_wheel_proof(", 1)[1].split(
        "\ndef main(", 1
    )[0]

    assert "import pops" not in resolver
    assert "from pops import" not in resolver
    assert "_installed_distribution_paths(" in resolver


def test_release_preflight_authenticates_installed_wheel_proof_and_transcripts(tmp_path):
    wheel = tmp_path / "wheels" / "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl"
    wheel.parent.mkdir()
    native_member = "pops/_pops.cpython-312-darwin.so"
    native_bytes = b"exact wheel extension"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pops/__init__.py", "__version__ = '0.3.0'\n")
        archive.writestr(native_member, native_bytes)
        archive.writestr(
            "pops-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: PoPS\nVersion: 0.3.0\n",
        )
    runtime = {
        "python_executable": "/proof/bin/python",
        "pops_file": "/proof/site-packages/pops/__init__.py",
        "native_extension": "/proof/site-packages/pops/_pops.so",
        "native_sha256": "post-sign-runtime-digest",
    }
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        rows = [
            "%s\0%s\n"
            % (name, hashlib.sha256(archive.read(name)).hexdigest())
            for name in sorted(archive.namelist())
            if not name.endswith("/") and not name.endswith(".dist-info/RECORD")
        ]
    commands = []
    command_argvs = (
        [
            "/proof/conda",
            "run",
            "python",
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ],
        [
            "/proof/conda",
            "run",
            "python",
            "scripts/prove_installed_wheel.py",
            "--wheel",
            str(wheel),
        ],
    )
    for index, argv in enumerate(command_argvs, 1):
        log = tmp_path / "logs" / f"{index:02d}_installed_wheel.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(json.dumps({"ok": True}), encoding="utf-8")
        commands.append(
            {
                "argv": argv,
                "log": str(log.relative_to(tmp_path)),
                "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
        )
    gates = {
        "official_build": {
            "evidence": {
                "wheel": {
                    "path": str(wheel.relative_to(tmp_path)),
                    "sha256": wheel_sha256,
                    "size": wheel.stat().st_size,
                },
            },
        },
        "installed_wheel": {
            "commands": commands,
            "evidence": {
                "schema_version": 2,
                "python_executable": runtime["python_executable"],
                "distribution_root": "/proof/site-packages",
                "package_file": runtime["pops_file"],
                "native_extension": runtime["native_extension"],
                "native_member": native_member,
                "native_sha256": hashlib.sha256(native_bytes).hexdigest(),
                "installed_member_count": len(rows),
                "installed_tree_sha256": hashlib.sha256(
                    "".join(rows).encode("utf-8")
                ).hexdigest(),
                "proof_script_sha256": hashlib.sha256(
                    (SCRIPTS / "prove_installed_wheel.py").read_bytes()
                ).hexdigest(),
                "version": "0.3.0",
                "wheel_path": str(wheel),
                "wheel_sha256": wheel_sha256,
            },
        },
    }
    release = type("ReleaseContract", (), {"PACKAGE_VERSION": "0.3.0"})

    preflight._installed_wheel_evidence(tmp_path, gates, release, runtime)
    gates["installed_wheel"]["evidence"]["native_sha256"] = "0" * 64
    with pytest.raises(preflight.PreflightError, match="native member hash drifted"):
        preflight._installed_wheel_evidence(tmp_path, gates, release, runtime)


def test_installed_wheel_gate_precedes_codesign_and_conformance():
    gates = contract.REQUIRED_RELEASE_GATES

    assert gates.index("official_build") < gates.index("installed_wheel")
    assert gates.index("installed_wheel") < gates.index("codesign")
    assert gates.index("codesign") < gates.index("doctor")
    assert gates.index("codesign") < gates.index("native_conformance")


def test_release_preflight_binds_codesign_to_live_runtime(tmp_path):
    log = tmp_path / "logs" / "codesign.log"
    log.parent.mkdir()
    log.write_text('{"platform": "darwin"}\n', encoding="utf-8")
    runtime = {
        "python_executable": "/proof/bin/python",
        "pops_file": "/proof/site-packages/pops/__init__.py",
        "native_extension": "/proof/site-packages/pops/_pops.so",
        "native_sha256": "a" * 64,
    }
    gates = {
        "installed_wheel": {
            "evidence": {"native_sha256": runtime["native_sha256"]},
        },
        "codesign": {
            "commands": [
                {
                    "argv": [
                        "/proof/conda",
                        "run",
                        "python",
                        "scripts/codesign_pops_extensions.py",
                        "--json",
                    ],
                    "log": str(log.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            ],
            "evidence": {
                "schema_version": 1,
                "platform": "darwin",
                "extensions": [
                    {
                        "path": runtime["native_extension"],
                        "sha256": runtime["native_sha256"],
                        "signature": "adhoc",
                    }
                ],
            },
        },
    }

    preflight._codesign_evidence(tmp_path, gates, runtime)
    gates["codesign"]["evidence"]["extensions"][0]["sha256"] = "b" * 64
    with pytest.raises(preflight.PreflightError, match="live native extension"):
        preflight._codesign_evidence(tmp_path, gates, runtime)

    gates["codesign"]["evidence"]["extensions"][0]["sha256"] = runtime["native_sha256"]
    gates["installed_wheel"]["evidence"]["native_sha256"] = "b" * 64
    with pytest.raises(preflight.PreflightError, match="published wheel"):
        preflight._codesign_evidence(tmp_path, gates, runtime)


def test_installed_example_authenticates_native_bytes_before_execution(
    monkeypatch, tmp_path, capsys,
):
    package = tmp_path / "site-packages" / "pops" / "__init__.py"
    extension = package.parent / "_pops.so"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    extension.write_bytes(b"signed release runtime")
    native = types.ModuleType("pops._pops")
    native.__file__ = str(extension)
    native.__version__ = "1.0.0"
    pops = types.ModuleType("pops")
    pops.__file__ = str(package)
    pops.__version__ = "1.0.0"
    pops._pops = native
    monkeypatch.setitem(sys.modules, "pops", pops)
    monkeypatch.setitem(sys.modules, "pops._pops", native)
    digest = hashlib.sha256(extension.read_bytes()).hexdigest()

    assert example_runner.verify_installed_runtime(digest) == digest
    with pytest.raises(example_runner.InstalledExampleError, match="does not match"):
        example_runner.verify_installed_runtime("0" * 64)

    monkeypatch.setattr(example_runner, "ROOT", tmp_path)
    example = tmp_path / "example.py"
    example.write_text(
        "import sys\nprint('example_args=' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(example_runner, "verify_installed_runtime", lambda expected: expected)
    assert example_runner.main([
        "--runtime-sha256",
        digest,
        "--example",
        str(example),
        "--",
        "--output-dir",
        "/proof/output",
    ]) == 0
    output = capsys.readouterr().out
    assert example_runner.RUNTIME_MARKER + digest in output
    assert "example_args=--output-dir|/proof/output" in output


def test_final_gate_rejects_incomplete_non_darwin_or_rewritten_codesign_runtime():
    evidence = {
        "schema_version": 1,
        "platform": "darwin",
        "extensions": [
            {
                "path": "/proof/pops/_pops.so",
                "sha256": "a" * 64,
                "signature": "adhoc",
            }
        ],
    }

    assert gate._signed_runtime_sha256(
        evidence, retained_native_sha256="a" * 64
    ) == "a" * 64
    with pytest.raises(gate.FinalGateError, match="different from the validated runtime"):
        gate._signed_runtime_sha256(evidence, retained_native_sha256="b" * 64)
    evidence["platform"] = "linux"
    with pytest.raises(gate.FinalGateError, match="Darwin release proof"):
        gate._signed_runtime_sha256(evidence, retained_native_sha256="a" * 64)


def test_release_preflight_requires_exact_runtime_bound_example_commands(tmp_path):
    runtime = {
        "python_executable": "/proof/bin/python",
        "pops_file": "/proof/site-packages/pops/__init__.py",
        "native_extension": "/proof/site-packages/pops/_pops.so",
        "native_sha256": "c" * 64,
    }
    examples = {}
    reopened = {}
    restarted = {}
    commands = []
    for index, example in enumerate(contract.FINAL_EXAMPLES, 1):
        key = example.as_posix()
        output_root = tmp_path / "examples" / example.stem
        output_root.mkdir(parents=True)
        hdf5 = output_root / "state.h5"
        hdf5.write_bytes(b"\x89HDF\r\n\x1a\npayload")
        npz = output_root / "state.npz"
        with zipfile.ZipFile(npz, "w") as archive:
            archive.writestr("state.npy", b"payload")
        paraview = output_root / "state.vtu"
        paraview.write_text("<VTKFile/>", encoding="utf-8")
        checkpoint = output_root / "checkpoint.bin"
        checkpoint.write_bytes(b"restart")
        transcript = "\n".join(
            [
                example_runner.RUNTIME_MARKER + runtime["native_sha256"],
                *contract.REQUIRED_PROOF_MARKERS,
            ]
        ) + "\n"
        log = tmp_path / "logs" / f"{index:02d}_examples.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(transcript, encoding="utf-8")
        commands.append(
            {
                "argv": [
                    "/proof/conda",
                    "run",
                    "python",
                    "scripts/run_installed_example.py",
                    "--runtime-sha256",
                    runtime["native_sha256"],
                    "--example",
                    key,
                    "--",
                    "--output-dir",
                    str(output_root),
                ],
                "log": str(log.relative_to(tmp_path)),
                "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
        )
        examples[key] = {
            "source_sha256": hashlib.sha256((ROOT / example).read_bytes()).hexdigest(),
            "stdout_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "output_root": str(output_root.relative_to(tmp_path)),
            "runtime_sha256": runtime["native_sha256"],
        }
        reopened[key] = {
            "hdf5": [
                {
                    "path": hdf5.name,
                    "sha256": hashlib.sha256(hdf5.read_bytes()).hexdigest(),
                }
            ],
            "npz": [
                {
                    "path": npz.name,
                    "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                }
            ],
            "paraview": [
                {
                    "path": paraview.name,
                    "sha256": hashlib.sha256(paraview.read_bytes()).hexdigest(),
                }
            ],
        }
        restarted[key] = {
            "checkpoint": str(checkpoint),
            "tree_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "proof_markers": list(contract.REQUIRED_PROOF_MARKERS),
        }
    gates = {
        "examples": {"commands": commands, "evidence": {"examples": examples}},
        "artifact_reopen": {"evidence": {"examples": reopened}},
        "strict_restart": {"evidence": {"examples": restarted}},
    }

    preflight._examples_evidence(tmp_path, gates, runtime)
    commands[0]["argv"][commands[0]["argv"].index(runtime["native_sha256"])] = "d" * 64
    with pytest.raises(preflight.PreflightError, match="command drifted"):
        preflight._examples_evidence(tmp_path, gates, runtime)


def test_tag_release_cannot_race_or_bypass_supported_matrix_wheel_and_final_gate():
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    wheels = (ROOT / ".github" / "workflows" / "wheels.yml").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    build = (ROOT / "scripts" / "build_python.sh").read_text()

    assert "uses: ./.github/workflows/ci.yml" in release
    assert "force_full: true" in release
    assert "uses: ./.github/workflows/wheels.yml" in release
    assert "needs: [full-source-matrix, wheel, validate]" in release
    assert "run_final_gate.py --wheel" in release
    assert "release_preflight.py" in release
    assert 'gh release create "$GITHUB_REF_NAME" wheelhouse/*.whl' in release
    assert "workflow_call:" in ci
    assert "FORCE_FULL: ${{ inputs.force_full || false }}" in ci
    assert "workflow_call:" in wheels
    assert "gh release upload" not in wheels
    assert "--wheel-dir" in build
    assert "--force-reinstall --no-deps" in build
