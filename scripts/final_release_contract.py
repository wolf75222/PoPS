"""The non-negotiable source contract for a final PoPS release.

This module deliberately contains only identities which are reviewed with the
release process.  Both the executable gate and the release preflight import it,
so a new example cannot be silently added to one path but omitted from the
other.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib


FINAL_SPECIFICATION = Path("docs/design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md")
FINAL_EXAMPLES = (
    Path("examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"),
)
FINAL_EXAMPLE_ACCEPTANCE_TESTS = (
    "tests/python/integration/bindings/test_m1_scalar_advection_pipeline.py"
    "::test_scalar_advection_final_example_runs_outputs_and_bit_identical_restart",
    "tests/python/examples/final/test_multiphysics_core_example.py"
    "::test_example_script_runs_outputs_and_restart_without_mock_or_fallback",
    "tests/python/examples/final/test_imex_amr_final_example.py"
    "::test_example_runs_and_every_scientific_format_reopens",
    "tests/python/examples/final/test_hyqmom15_final_example.py"
    "::test_hyqmom15_example_runs_outputs_and_restarts_bit_identically",
)
FINAL_EXAMPLE_QUALIFICATION_TESTS = (
    "tests/python/examples/final/test_scalar_advection_final_example.py"
    "::test_target_has_one_authority_per_concern_and_no_legacy_path",
    "tests/python/examples/final/test_multiphysics_core_example.py"
    "::test_program_has_exact_field_context_and_transactional_implicit_join",
    "tests/python/examples/final/test_imex_amr_final_example.py"
    "::test_resolved_amr_lowering_report_covers_every_executed_authority",
    "tests/python/unit/moments/test_hyqmom15_final_contract.py"
    "::test_particle_number_diagnostic_integrates_m00_and_rejects_drift",
)
FINAL_EXAMPLE_REQUIRED_TESTS = (
    *FINAL_EXAMPLE_ACCEPTANCE_TESTS,
    *FINAL_EXAMPLE_QUALIFICATION_TESTS,
)
FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS = {
    FINAL_EXAMPLES[0]: {
        "hdf5": ("state/tracer",),
        "npz": (),
        "paraview": ("solution/tracer",),
    },
    FINAL_EXAMPLES[1]: {
        "hdf5": ("state/two_fluid",),
        "npz": (),
        "paraview": ("visualization/two_fluid",),
    },
    FINAL_EXAMPLES[2]: {
        "hdf5": ("hdf5/state",),
        "npz": ("npz/state",),
        "paraview": ("paraview/state",),
    },
    FINAL_EXAMPLES[3]: {
        "hdf5": ("state/hyqmom15",),
        "npz": (),
        "paraview": ("visualization/hyqmom15",),
    },
}
REQUIRED_PROOF_MARKERS = (
    "HDF5:",
    "ParaView:",
    "checkpoint:",
    "bit-identical restart:",
)
FORBIDDEN_FINAL_IMPORTS = (
    "pops.ir",
    "pops._ir",
    "pops.runtime.bricks",
    "pops.runtime.integrate",
    "CartesianMesh",
)
# The complete source suite is authenticated by the release workflow's ``full-source-matrix`` job.
# The exact published wheel repeats the closed M4 Python ledger plus the final-example ledger; it
# must not serialize the complete suite a second time under a short release timeout.
PYTHON_CONFORMANCE_MANIFEST = Path("tests/gates/m4_runtime_io.toml")
PYTHON_REQUIRED_SELECTION = "m4-runtime-io-pytest+final-example-ledger"
INSTALLED_COMPONENT_PACKAGE_NODEID = (
    "tests/python/integration/native_loader/test_external_component_package.py"
    "::test_source_component_executes_through_generic_native_loader_and_flux_consumer"
)
REQUIRED_RELEASE_GATES = (
    "official_build",
    "installed_wheel",
    "codesign",
    "doctor",
    "native_conformance",
    "python_conformance",
    "examples",
    "artifact_reopen",
    "strict_restart",
    "documentation",
    "generated_products",
    "diff",
)


def required_python_conformance_nodeids(root: Path) -> tuple[str, ...]:
    """Return the exact installed-wheel Python ledger for the final gate.

    MPI-only rows stay proved by ``full-source-matrix`` because the published wheel is Serial.
    The external component row is executed separately with checkout headers explicitly cleared,
    which is a strictly stronger installed-wheel proof than repeating it in the main lane.
    """

    path = root / PYTHON_CONFORMANCE_MANIFEST
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("cannot read final Python conformance manifest: %s" % exc) from exc
    if data.get("schema_version") != 1 or data.get("gate") != "m4-runtime-io":
        raise ValueError("final Python conformance manifest identity drifted")
    if data.get("deferred") != []:
        raise ValueError("final Python conformance manifest must be closed")
    checks = data.get("check")
    if not isinstance(checks, list) or not checks:
        raise ValueError("final Python conformance manifest has no executable checks")

    nodeids: list[str] = []
    manifest_nodeids: set[str] = set()
    for row in checks:
        if not isinstance(row, dict):
            raise ValueError("final Python conformance manifest contains a malformed row")
        if row.get("kind") != "pytest":
            continue
        nodeid = row.get("nodeid")
        if not isinstance(nodeid, str) or "::" not in nodeid:
            raise ValueError("final Python conformance manifest contains an invalid pytest nodeid")
        if nodeid in manifest_nodeids:
            raise ValueError("final Python conformance manifest contains duplicate pytest nodeids")
        manifest_nodeids.add(nodeid)
        if nodeid != INSTALLED_COMPONENT_PACKAGE_NODEID \
                and nodeid not in FINAL_EXAMPLE_REQUIRED_TESTS:
            nodeids.append(nodeid)
    nodeids.extend(FINAL_EXAMPLE_REQUIRED_TESTS)
    if len(nodeids) != len(set(nodeids)):
        raise ValueError("final Python conformance ledger contains duplicate nodeids")
    return tuple(nodeids)


def release_matrix_source_errors(root: Path) -> list[str]:
    """Return drift between the declared support matrix and its executable workflow proof.

    The workflow syntax is intentionally treated as a reviewed source contract. Adding a matrix
    lane without teaching this preflight where that lane is built must fail before an expensive
    release build starts; a prose claim or an unreferenced job is not release evidence.
    """

    errors: list[str] = []
    contract_path = root / "schemas" / "release_contract.v1.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        matrix = contract["supported_matrix"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return ["cannot read the supported release matrix: %s" % exc]
    if not isinstance(matrix, dict):
        return ["supported release matrix must be an object"]

    relative_sources = (
        Path("CMakeLists.txt"),
        Path(".github/actions/setup-kokkos/action.yml"),
        Path(".github/workflows/ci.yml"),
        Path(".github/workflows/wheels.yml"),
        Path(".github/workflows/release.yml"),
    )
    sources: dict[Path, str] = {}
    for relative in relative_sources:
        try:
            sources[relative] = (root / relative).read_text(encoding="utf-8")
        except OSError as exc:
            errors.append("release matrix proof source %s is unavailable: %s" % (relative, exc))

    def require(relative: Path, label: str, *markers: str) -> None:
        text = sources.get(relative)
        if text is None:
            return
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                "%s lacks executable workflow markers %s in %s" % (label, missing, relative)
            )

    language = matrix.get("language")
    if not isinstance(language, dict):
        errors.append("supported release language matrix is malformed")
    else:
        python_versions = language.get("python")
        if not isinstance(python_versions, list) or not python_versions:
            errors.append("supported release matrix declares no Python version")
        else:
            for version in python_versions:
                if not isinstance(version, str) or not version:
                    errors.append("supported release matrix contains an invalid Python version")
                    continue
                require(
                    Path(".github/workflows/ci.yml"),
                    "Python %s source lane" % version,
                    "python-version: '%s'" % version,
                )
        standard = language.get("cpp_standard")
        if type(standard) is not int or standard < 1:
            errors.append("supported release matrix contains an invalid C++ standard")
        else:
            require(
                Path("CMakeLists.txt"),
                "C++%d package contract" % standard,
                "set(POPS_CXX_STD %d)" % standard,
            )
            require(
                Path(".github/workflows/ci.yml"),
                "C++%d Linux source lanes" % standard,
                "-DCMAKE_CXX_STANDARD=%d" % standard,
            )
            require(
                Path(".github/workflows/wheels.yml"),
                "C++%d wheel lane" % standard,
                "-DCMAKE_CXX_STANDARD=%d" % standard,
            )
        compiler_proofs = {
            "GNU": (Path(".github/workflows/ci.yml"), "g++ -dumpfullversion"),
            "AppleClang": (Path(".github/workflows/wheels.yml"), "clang++ --version"),
        }
        compilers = language.get("compiler_families")
        if not isinstance(compilers, list) or not compilers:
            errors.append("supported release matrix declares no compiler family")
        else:
            for compiler in compilers:
                if not isinstance(compiler, str):
                    errors.append("supported release matrix contains an invalid compiler family")
                    continue
                proof = compiler_proofs.get(compiler)
                if proof is None:
                    errors.append("compiler family %r has no executable release proof" % compiler)
                else:
                    require(proof[0], "%s compiler lane" % compiler, proof[1])

    kokkos = matrix.get("kokkos")
    if not isinstance(kokkos, dict) or not isinstance(kokkos.get("version"), str):
        errors.append("supported Kokkos release matrix is malformed")
    else:
        version = kokkos["version"]
        require(
            Path("CMakeLists.txt"),
            "Kokkos package version",
            'POPS_KOKKOS_FETCH_VERSION "%s"' % version,
        )
        require(
            Path(".github/workflows/ci.yml"), "Kokkos CI version", "KOKKOS_VERSION: %s" % version
        )
        require(
            Path(".github/workflows/wheels.yml"),
            "Kokkos wheel version",
            "git clone --depth 1 -b %s " % version,
        )
        execution_space_proofs = {
            "Serial": (
                Path(".github/actions/setup-kokkos/action.yml"),
                "-DKokkos_ENABLE_SERIAL=ON",
            ),
            "OpenMP": (
                Path(".github/workflows/ci.yml"),
                "-DKokkos_ENABLE_OPENMP=ON",
            ),
        }
        spaces = kokkos.get("execution_spaces")
        if not isinstance(spaces, list) or not spaces:
            errors.append("supported Kokkos release matrix declares no execution space")
        else:
            for space in spaces:
                if not isinstance(space, str):
                    errors.append("supported release matrix contains an invalid execution space")
                    continue
                proof = execution_space_proofs.get(space)
                if proof is None:
                    errors.append(
                        "Kokkos execution space %r has no executable release proof" % space
                    )
                else:
                    require(proof[0], "Kokkos %s lane" % space, proof[1])

    distributed = matrix.get("distributed")
    if not isinstance(distributed, dict):
        errors.append("supported distributed release matrix is malformed")
    elif distributed.get("mpi_implementation") != "OpenMPI":
        errors.append(
            "MPI implementation %r has no executable release proof"
            % distributed.get("mpi_implementation")
        )
    else:
        require(
            Path(".github/workflows/ci.yml"),
            "OpenMPI source lane",
            "libopenmpi-dev",
            "openmpi-bin",
            "-DPOPS_USE_MPI=ON",
        )
        if distributed.get("execution_spaces") != ["Serial"]:
            errors.append("distributed execution spaces have no exact executable release proof")

    source_lane_proofs = {
        ("linux", "x86_64", "Kokkos Serial"): (
            Path(".github/workflows/ci.yml"),
            ("name: ubuntu-latest / Kokkos Serial (C++)",),
        ),
        ("linux", "x86_64", "Kokkos OpenMP"): (
            Path(".github/workflows/ci.yml"),
            ("name: ubuntu-latest / Kokkos (OpenMP", "-DKokkos_ENABLE_OPENMP=ON"),
        ),
        ("linux", "x86_64", "Kokkos Serial + OpenMPI"): (
            Path(".github/workflows/ci.yml"),
            ("name: ubuntu-24.04 / MPI + Kokkos Serial", "libopenmpi-dev", "-DPOPS_USE_MPI=ON"),
        ),
        ("macos", "arm64", "Kokkos Serial"): (
            Path(".github/workflows/wheels.yml"),
            ("runs-on: macos-14", "CIBW_ARCHS_MACOS: arm64", "-DKokkos_ENABLE_SERIAL=ON"),
        ),
    }
    source_builds = matrix.get("source_builds")
    if not isinstance(source_builds, list) or not source_builds:
        errors.append("supported release matrix declares no source-build lane")
    else:
        seen_source_lanes: set[tuple[str, str, str]] = set()
        for row in source_builds:
            if not isinstance(row, dict) or set(row) != {"os", "arch", "backend"}:
                errors.append("supported source-build lane is malformed")
                continue
            lane = (row["os"], row["arch"], row["backend"])
            if not all(isinstance(value, str) and value for value in lane):
                errors.append("supported source-build lane is malformed")
                continue
            if lane in seen_source_lanes:
                errors.append("supported source-build lane is duplicated: %r" % (lane,))
                continue
            seen_source_lanes.add(lane)
            proof = source_lane_proofs.get(lane)
            if proof is None:
                errors.append("source-build lane %r has no executable release proof" % (lane,))
            else:
                require(proof[0], "source-build lane %r" % (lane,), *proof[1])

    wheel_lane_proofs = {
        ("macos", "arm64", "cp312", "Kokkos Serial"): (
            "CIBW_BUILD: cp312-macosx_arm64",
            "CIBW_ARCHS_MACOS: arm64",
            "name: pops-macos-arm64-cp312",
        ),
    }
    wheels = matrix.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        errors.append("supported release matrix declares no wheel lane")
    else:
        seen_wheel_lanes: set[tuple[str, str, str, str]] = set()
        for row in wheels:
            if not isinstance(row, dict) or set(row) != {"os", "arch", "python", "backend"}:
                errors.append("supported wheel lane is malformed")
                continue
            lane = (row["os"], row["arch"], row["python"], row["backend"])
            if not all(isinstance(value, str) and value for value in lane):
                errors.append("supported wheel lane is malformed")
                continue
            if lane in seen_wheel_lanes:
                errors.append("supported wheel lane is duplicated: %r" % (lane,))
                continue
            seen_wheel_lanes.add(lane)
            proof = wheel_lane_proofs.get(lane)
            if proof is None:
                errors.append("wheel lane %r has no executable release proof" % (lane,))
                continue
            require(Path(".github/workflows/wheels.yml"), "wheel lane %r" % (lane,), *proof)
            require(
                Path(".github/workflows/release.yml"),
                "published wheel lane %r" % (lane,),
                'test "$(uname -m)" = arm64',
                'test "$(python -c \'import sys; print("cp%d%d" % sys.version_info[:2])\')" = cp312',
                "name: pops-macos-arm64-cp312",
                "run_final_gate.py --wheel",
            )

    require(
        Path(".github/workflows/release.yml"),
        "release publication dependency",
        "needs: [full-source-matrix, wheel, validate]",
    )
    return errors


def require_release_matrix_source_contract(root: Path) -> None:
    errors = release_matrix_source_errors(root)
    if errors:
        raise ValueError("; ".join(errors))


def source_contract_errors(root: Path) -> list[str]:
    """Return every deterministic final-source contract violation.

    This is intentionally source-only: it is used before starting a costly
    build and can be exercised in isolation by architecture tests.
    """

    errors: list[str] = []
    specification = root / FINAL_SPECIFICATION
    if not specification.is_file() or not specification.read_text(encoding="utf-8").strip():
        errors.append("missing canonical final specification: %s" % FINAL_SPECIFICATION)

    examples_dir = root / "examples" / "final"
    actual = (
        tuple(sorted(path.relative_to(root) for path in examples_dir.glob("*.py")))
        if examples_dir.is_dir()
        else ()
    )
    expected = tuple(sorted(FINAL_EXAMPLES))
    if actual != expected:
        errors.append("final examples must be exactly %s (found %s)" % (expected, actual))
    if set(FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS) != set(FINAL_EXAMPLES):
        errors.append("final scientific-output ledger must cover exactly the final examples")

    for relative in FINAL_EXAMPLES:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "--output-dir" not in text:
            errors.append("%s must accept an explicit --output-dir" % relative)
        if 'if __name__ == "__main__"' not in text:
            errors.append("%s must remain directly executable" % relative)
        missing = [marker for marker in REQUIRED_PROOF_MARKERS if marker not in text]
        if missing:
            errors.append("%s lacks final proof markers %s" % (relative, missing))
        forbidden = [name for name in FORBIDDEN_FINAL_IMPORTS if name in text]
        if forbidden:
            errors.append(
                "%s imports transitional/internal authoring names %s" % (relative, forbidden)
            )
        formats = FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS.get(relative)
        if not isinstance(formats, dict) or set(formats) != {"hdf5", "npz", "paraview"}:
            errors.append("%s has no exact scientific-output format ledger" % relative)
            continue
        for format_name, targets in formats.items():
            if not isinstance(targets, tuple) or any(
                not isinstance(target, str) or not target for target in targets
            ):
                errors.append(
                    "%s has a malformed %s scientific-output target ledger"
                    % (relative, format_name)
                )
                continue
            for target in targets:
                if 'target="%s"' % target not in text:
                    errors.append(
                        "%s lacks its exact %s scientific-output target %s"
                        % (relative, format_name, target)
                    )
    ledgers = (
        ("acceptance", FINAL_EXAMPLE_ACCEPTANCE_TESTS),
        ("qualification", FINAL_EXAMPLE_QUALIFICATION_TESTS),
    )
    required_nodeids = tuple(nodeid for _kind, ledger in ledgers for nodeid in ledger)
    if len(set(required_nodeids)) != len(required_nodeids):
        errors.append("final-example required test nodeids must be unique")
    for proof_kind, ledger in ledgers:
        if len(ledger) != len(FINAL_EXAMPLES):
            errors.append(
                "final examples and exact %s tests must have one-to-one coverage"
                % proof_kind
            )
        for example, nodeid in zip(FINAL_EXAMPLES, ledger, strict=False):
            relative, separator, function_name = nodeid.partition("::")
            if not separator or not relative or not function_name:
                errors.append("invalid final-example %s nodeid %r" % (proof_kind, nodeid))
                continue
            test_path = root / relative
            if not test_path.is_file():
                errors.append("missing final-example %s test: %s" % (proof_kind, nodeid))
                continue
            source = test_path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(test_path))
            except SyntaxError as exc:
                errors.append(
                    "cannot parse final-example %s test %s: %s"
                    % (proof_kind, nodeid, exc)
                )
                continue
            functions = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ]
            if len(functions) != 1:
                errors.append(
                    "final-example %s nodeid must resolve exactly once: %s"
                    % (proof_kind, nodeid)
                )
                continue
            function = functions[0]
            fixture_names = {
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            if fixture_names & {"mock", "mocker", "monkeypatch", "patch"}:
                errors.append("%s uses a mock fixture" % nodeid)
            forbidden_calls = []
            forbidden_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (
                    node.module or ""
                ).startswith(("unittest.mock", "pytest_mock")):
                    forbidden_imports.append(node.module or "")
                elif isinstance(node, ast.Import) and any(
                    alias.name.startswith(("unittest.mock", "pytest_mock"))
                    for alias in node.names
                ):
                    forbidden_imports.extend(alias.name for alias in node.names)
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                call = node.func
                parts = []
                while isinstance(call, ast.Attribute):
                    parts.append(call.attr)
                    call = call.value
                if isinstance(call, ast.Name):
                    parts.append(call.id)
                name = ".".join(reversed(parts))
                if name in {
                    "patch",
                    "pytest.importorskip",
                    "pytest.skip",
                    "pytest.xfail",
                } or name.startswith(("mock.", "mocker.", "unittest.mock.")):
                    forbidden_calls.append(name)
            decorators = []
            for decorator in function.decorator_list:
                text = ast.unparse(decorator)
                if "skip" in text or "xfail" in text:
                    decorators.append(text)
            module_markers = []
            for statement in tree.body:
                value = None
                targets = ()
                if isinstance(statement, ast.Assign):
                    value = statement.value
                    targets = statement.targets
                elif isinstance(statement, ast.AnnAssign):
                    value = statement.value
                    targets = (statement.target,)
                if value is not None and any(
                    isinstance(target, ast.Name) and target.id == "pytestmark"
                    for target in targets
                ):
                    text = ast.unparse(value)
                    if "skip" in text or "xfail" in text:
                        module_markers.append(text)
            if forbidden_calls or forbidden_imports or decorators or module_markers:
                errors.append(
                    "%s is optional: %s"
                    % (
                        nodeid,
                        sorted(
                            set(
                                (
                                    *forbidden_calls,
                                    *forbidden_imports,
                                    *decorators,
                                    *module_markers,
                                )
                            )
                        ),
                    )
                )
            if example.name not in source:
                errors.append("%s is not bound to %s" % (nodeid, example))
    return errors


def require_source_contract(root: Path) -> None:
    errors = source_contract_errors(root)
    if errors:
        raise ValueError("; ".join(errors))
