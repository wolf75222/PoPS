"""The non-negotiable source contract for a final PoPS release.

This module deliberately contains only identities which are reviewed with the
release process.  Both the executable gate and the release preflight import it,
so a new example cannot be silently added to one path but omitted from the
other.
"""

from __future__ import annotations

import json
from pathlib import Path


FINAL_SPECIFICATION = Path("docs/design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md")
FINAL_EXAMPLES = (
    Path("examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"),
)
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
# The published wheel matrix is CPU/Kokkos Serial without MPI or parallel HDF5. The full suite still
# runs; this supported-platform subset is repeated with a strict all-pass/no-hidden-skip policy.
PYTHON_REQUIRED_SELECTION = "not mpi and not hdf5"
REQUIRED_RELEASE_GATES = (
    "official_build",
    "doctor",
    "codesign",
    "native_conformance",
    "python_conformance",
    "examples",
    "artifact_reopen",
    "strict_restart",
    "documentation",
    "generated_products",
    "diff",
)


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
    return errors


def require_source_contract(root: Path) -> None:
    errors = source_contract_errors(root)
    if errors:
        raise ValueError("; ".join(errors))
