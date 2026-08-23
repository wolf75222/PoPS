#!/usr/bin/env python3
"""Validate and, only when requested, execute the complete sine-wave matrix.

The default mode is intentionally read-only: it authenticates the versioned
coverage matrix and prints its exact commands.  ``--execute`` is the explicit
campaign action; it refuses every existing case directory before starting and
then refuses a result without the named observed witnesses.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CASE_DIRECTORY = Path(__file__).resolve().parent
MATRIX_PATH = CASE_DIRECTORY / "matrix.v1.json"
CANONICAL_MATRIX_SHA256 = "cd9e25b08796eb71fa243dd76bbe7b9b634510f900c314e2d882e62427b7eb83"
REPOSITORY_ROOT = CASE_DIRECTORY.parents[3]
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build_python.sh"
MATRIX_SOURCE_AUTHORITY_SCHEMA = "pops.sine-wave.matrix-source-authority.v1"
BUILD_SOURCE_AUTHORITY_ROOTS = (
    "CMakeLists.txt",
    "cmake",
    "include",
    "src",
    "python",
    "pyproject.toml",
    "schemas",
    "scripts",
)
KNOWN_OBLIGATIONS = frozenset(
    {
        "block_face",
        "block_edge_3d",
        "block_corner_3d",
        "coarse_fine_interface",
        "periodic_boundary",
        "prescribed_mobile_regrid",
        "repeated_patch_crossing",
        "second_order_convergence",
    }
)
DEFERRED_CONVERGENCE_WITNESS = {
    "applicable": True,
    "observed": False,
    "deferred": {
        "scope": "matrix_complete",
        "qualifier": "_convergence_receipt",
        "obligation": "second_order_convergence",
    },
    "diagnostic": "matrix_complete_postprocess_l1_order",
}


def _is_deferred_convergence_witness(witness: object) -> bool:
    """Recognize the sole per-case form that defers convergence to COMPLETE."""
    return isinstance(witness, dict) and witness == DEFERRED_CONVERGENCE_WITNESS


VALID_MODES = {
    1: {"x", "diagonal"},
    2: {"x", "y", "diagonal"},
    3: {"x", "y", "z", "xy", "diagonal"},
}
VALID_LAYOUTS = {"uniform", "amr-frozen", "amr-mobile"}
EXPECTED_CASE_IDS = (
    "d1-face",
    "d1-diagonal",
    "d1-cf-sync",
    "d1-cf-subcycled",
    "d1-mobile-sync",
    "d1-mobile-subcycled",
    "d2-face",
    "d2-y-block8",
    "d2-y-block16",
    "d2-y",
    "d2-diagonal",
    "d2-cf-sync",
    "d2-cf-subcycled",
    "d2-mobile-sync",
    "d2-mobile-subcycled",
    "d3-face",
    "d3-y",
    "d3-z",
    "d3-edge",
    "d3-corner",
    "d3-cf-sync",
    "d3-cf-subcycled",
    "d3-mobile-sync",
    "d3-mobile-subcycled",
    "d2-mpi-np1",
    "d2-mpi-np2",
    "d2-mpi-np4",
    "d3-mpi-np8-corner",
    "conv-d1-n32",
    "conv-d1-n64",
    "conv-d1-n128",
    "conv-d2-n16",
    "conv-d2-n32",
    "conv-d2-n64",
    "conv-d3-n16",
    "conv-d3-n32",
    "conv-d3-n64",
)
SUBCYCLING_COMPARISON_PAIRS = (
    ("d1-cf-sync", "d1-cf-subcycled"),
    ("d2-cf-sync", "d2-cf-subcycled"),
    ("d3-cf-sync", "d3-cf-subcycled"),
    ("d1-mobile-sync", "d1-mobile-subcycled"),
    ("d2-mobile-sync", "d2-mobile-subcycled"),
    ("d3-mobile-sync", "d3-mobile-subcycled"),
)
MPI_INVARIANCE_CASE_IDS = ("d2-mpi-np1", "d2-mpi-np2", "d2-mpi-np4")
MPI_TOPOLOGY_CASE_IDS = (*MPI_INVARIANCE_CASE_IDS, "d3-mpi-np8-corner")
BLOCK_SIZE_COMPARISON_CASE_IDS = ("d2-y-block8", "d2-y-block16", "d2-y")
NATIVE_PER_VARIANT_ABI_FIELDS = frozenset({"dim", "mpi", "mpi_abi"})


def _mpi_launcher() -> str:
    """Return the launcher colocated with the Python that loaded the native wheel.

    The MPI ABI is a process boundary: selecting an arbitrary ``mpiexec`` from an inherited
    ``PATH`` can start a singleton world when it belongs to a different MPI distribution than the
    extension.  The production build script activates the Python environment that owns both the
    wheel and its MPI launcher, so use that launcher explicitly.  ROMEO may keep Python and MPI
    in distinct package prefixes; it must name the exact launcher through ``POPS_MPIEXEC``.
    """
    override = os.environ.get("POPS_MPIEXEC")
    if override:
        return str(Path(override).expanduser().absolute())
    return str(Path(sys.executable).absolute().parent / "mpiexec")


def _require_mpi_launcher() -> None:
    """Fail before reserving or running a numerical MPI matrix case if the launcher is absent."""
    launcher = Path(_mpi_launcher())
    if launcher.is_file() and os.access(launcher, os.X_OK):
        return
    raise RuntimeError(
        "MPI matrix execution requires an executable mpiexec beside sys.executable; "
        "set POPS_MPIEXEC when the MPI and Python prefixes differ; got Python=%s and launcher=%s"
        % (sys.executable, launcher)
    )


def _preflight_mpi_launcher(dimension: int, environment: dict[str, str]) -> None:
    """Authenticate an active two-rank native world before any distributed numerical case.

    This only selects the exact native extension and exercises its byte collective.  It cannot
    advance a simulation, but it catches a foreign launcher that would otherwise expose one
    singleton MPI world per process and make ``np=1`` a misleading false positive.
    """
    _require_mpi_launcher()
    probe = """
import json
import sys

from pops._native_selector import select_native_dimension
from pops.runtime_environment import runtime_environment_report

native = select_native_dimension(int(sys.argv[1]))
world = native.mpi_world()
facts = runtime_environment_report()
row = {
    "rank": int(world.rank),
    "size": int(world.size),
    "active": bool(world.active),
    "native_mpi": bool(native.__has_mpi__),
    "facts_rank": int(facts["mpi_rank"]),
    "facts_size": int(facts["mpi_ranks"]),
    "facts_active": facts["mpi_active"] is True,
    "facts_compiled": facts["mpi_compiled"] is True,
}
rows = tuple(
    json.loads(payload.decode("utf-8"))
    for payload in world.allgather_bytes(json.dumps(row, sort_keys=True).encode("utf-8"))
)
if len(rows) != 2 or any(
    candidate != {
        **candidate,
        "rank": index,
        "size": 2,
        "active": True,
        "native_mpi": True,
        "facts_rank": index,
        "facts_size": 2,
        "facts_active": True,
        "facts_compiled": True,
    }
    for index, candidate in enumerate(rows)
):
    raise RuntimeError("MPI launcher preflight did not authenticate one coherent two-rank world")
"""
    subprocess.run(
        [_mpi_launcher(), "-n", "2", sys.executable, "-c", probe, str(dimension)],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )


def _read_matrix(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("matrix must be one regular file: %s" % path)
    if _sha256(path) != CANONICAL_MATRIX_SHA256:
        raise ValueError(
            "matrix differs from the exact canonical 37-case scientific inventory: %s" % path
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "generator",
        "result_namespace",
        "verification_environment",
        "build_phases",
        "coverage_obligations",
        "convergence_series",
        "cases",
    }:
        raise ValueError("matrix has an unsupported top-level shape")
    return raw


def _validate_matrix(matrix: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if matrix["schema_version"] != "pops.sine-wave.matrix.v1":
        raise ValueError("matrix schema_version is unsupported")
    if matrix["generator"] != "generate_data.py":
        raise ValueError("matrix must name the linear scientific generator")
    if not isinstance(matrix["result_namespace"], str) or not matrix["result_namespace"]:
        raise ValueError("matrix result_namespace must be non-empty text")
    if matrix["verification_environment"] != {
        "OMP_NUM_THREADS": "2",
        "OMP_PROC_BIND": "false",
        "KOKKOS_NUM_THREADS": "2",
    }:
        raise ValueError("matrix must declare the exact deterministic verification environment")
    phases = matrix["build_phases"]
    if phases != [
        {"id": "dim1-nonmpi", "dimension": 1, "mpi": False},
        {"id": "dim2-nonmpi", "dimension": 2, "mpi": False},
        {"id": "dim3-nonmpi", "dimension": 3, "mpi": False},
        {"id": "dim2-mpi", "dimension": 2, "mpi": True},
        {"id": "dim3-mpi", "dimension": 3, "mpi": True},
    ]:
        raise ValueError("matrix must declare the exact executable native build phases")
    cases = matrix["cases"]
    obligations = matrix["coverage_obligations"]
    series = matrix["convergence_series"]
    if (
        not isinstance(cases, list)
        or not cases
        or not isinstance(obligations, dict)
        or not isinstance(series, dict)
    ):
        raise ValueError("matrix requires non-empty cases, obligations, and convergence series")
    if len(cases) != len(EXPECTED_CASE_IDS):
        raise ValueError("matrix must retain the exact 37-case verification inventory")
    expected_case_keys = {
        "id",
        "dimension",
        "resolution",
        "mode",
        "layout",
        "subcycling",
        "block_size",
        "mpi_ranks",
        "cycles",
        "time_snapshots",
        "obligations",
    }
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in cases:
        if (
            not isinstance(row, dict)
        or set(row) - (expected_case_keys | {"mpi", "mpi_topology"})
            or not expected_case_keys <= set(row)
        ):
            raise ValueError("matrix case has an unsupported shape")
        identifier = row["id"]
        dimension = row["dimension"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("matrix case ids must be unique non-empty text")
        identifiers.add(identifier)
        if type(dimension) is not int or dimension not in (1, 2, 3):
            raise ValueError("%s has invalid dimension" % identifier)
        if row["mode"] not in VALID_MODES[dimension] or row["layout"] not in VALID_LAYOUTS:
            raise ValueError("%s has an invalid dimension/mode/layout combination" % identifier)
        if row["subcycling"] not in {"synchronous", "subcycled"} or (
            row["layout"] == "uniform" and row["subcycling"] != "synchronous"
        ):
            raise ValueError("%s has invalid subcycling" % identifier)
        if not isinstance(row["resolution"], str) or not row["resolution"]:
            raise ValueError("%s must retain an exact generator resolution string" % identifier)
        if type(row["block_size"]) is not int or row["block_size"] < 2:
            raise ValueError("%s has invalid block_size" % identifier)
        if type(row["cycles"]) is not int or row["cycles"] < 1:
            raise ValueError("%s has invalid cycles" % identifier)
        if type(row["time_snapshots"]) is not int or row["time_snapshots"] < 9:
            raise ValueError("%s has invalid time_snapshots" % identifier)
        if type(row["mpi_ranks"]) is not int or row["mpi_ranks"] not in (1, 2, 4, 8):
            raise ValueError("%s has invalid mpi_ranks" % identifier)
        mpi = row.get("mpi", False)
        if type(mpi) is not bool or (row["mpi_ranks"] > 1 and not mpi):
            raise ValueError("%s has inconsistent MPI selection" % identifier)
        topology = row.get("mpi_topology")
        if not mpi and topology is not None:
            raise ValueError("%s gives a topology to a non-MPI case" % identifier)
        if mpi:
            if (
                not isinstance(topology, list)
                or len(topology) != dimension
                or any(type(value) is not int or value < 1 for value in topology)
                or math.prod(topology) != row["mpi_ranks"]
            ):
                raise ValueError("%s has no exact Cartesian MPI topology" % identifier)
            resolution = _resolved_resolution(row)
            if any(extent % partitions for extent, partitions in zip(resolution, topology, strict=True)):
                raise ValueError("%s MPI topology does not partition the exact resolution" % identifier)
        named = row["obligations"]
        if (
            not isinstance(named, list)
            or len(named) != len(set(named))
            or not set(named) <= KNOWN_OBLIGATIONS
        ):
            raise ValueError("%s has unsupported or duplicate obligations" % identifier)
        normalized.append({**row, "mpi": mpi})

    if tuple(row["id"] for row in normalized) != EXPECTED_CASE_IDS:
        raise ValueError("matrix must retain the ordered, exact 37-case verification inventory")

    dimensions = {row["dimension"] for row in normalized}
    layouts = {row["layout"] for row in normalized}
    block_sizes = {row["block_size"] for row in normalized}
    mpi_ranks = {row["mpi_ranks"] for row in normalized if row["mpi"]}
    if (
        dimensions != {1, 2, 3}
        or layouts != VALID_LAYOUTS
        or len(block_sizes) < 2
        or mpi_ranks != {1, 2, 4, 8}
    ):
        raise ValueError("matrix does not cover dimensions, layouts, block sizes, and MPI np=1/2/4/8")
    for dimension, required_modes in VALID_MODES.items():
        present = {row["mode"] for row in normalized if row["dimension"] == dimension}
        if not required_modes <= present:
            raise ValueError("matrix omits a required Dim%d propagation mode" % dimension)
    if {row["subcycling"] for row in normalized if row["layout"] != "uniform"} != {
        "synchronous",
        "subcycled",
    }:
        raise ValueError("matrix must cover AMR subcycling on and off")

    bindings = {name: set() for name in KNOWN_OBLIGATIONS}
    for row in normalized:
        for name in row["obligations"]:
            bindings[name].add(row["id"])
    if set(obligations) != KNOWN_OBLIGATIONS:
        raise ValueError("matrix must declare every required coverage obligation")
    for name, declaration in obligations.items():
        if not isinstance(declaration, dict) or set(declaration) != {"case_ids"}:
            raise ValueError("%s has an unsupported obligation declaration" % name)
        declared = declaration["case_ids"]
        if not isinstance(declared, list) or not declared or set(declared) != bindings[name]:
            raise ValueError("%s lacks an exact case-to-witness binding" % name)
    repeated_cases = [row for row in normalized if "repeated_patch_crossing" in row["obligations"]]
    if not repeated_cases or any(
        row["layout"] != "amr-frozen" or row["cycles"] < 3 or row["time_snapshots"] < 49
        for row in repeated_cases
    ):
        raise ValueError(
            "repeated_patch_crossing requires a static AMR patch, three periods, and 49 snapshots"
        )
    if any(
        row["layout"] != "amr-mobile"
        for row in normalized
        if "prescribed_mobile_regrid" in row["obligations"]
    ):
        raise ValueError("prescribed_mobile_regrid requires one prescribed mobile AMR patch")
    _validate_convergence_series(series, tuple(normalized), bindings["second_order_convergence"])
    _validate_comparison_contracts(tuple(normalized))
    return tuple(normalized)


def _phase_cases(
    matrix: dict[str, Any], cases: tuple[dict[str, Any], ...]
) -> tuple[tuple[dict[str, Any], tuple[dict[str, Any], ...]], ...]:
    phases: list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = []
    assigned: set[str] = set()
    for phase in matrix["build_phases"]:
        selected = tuple(
            case
            for case in cases
            if case["dimension"] == phase["dimension"] and case["mpi"] is phase["mpi"]
        )
        if not selected:
            raise ValueError("native build phase %s has no cases" % phase["id"])
        assigned.update(case["id"] for case in selected)
        phases.append((phase, selected))
    if assigned != {case["id"] for case in cases}:
        raise ValueError("matrix has a case outside its executable native build phases")
    return tuple(phases)


def _validate_convergence_series(
    series: dict[str, Any], cases: tuple[dict[str, Any], ...], obligated: set[str]
) -> None:
    if set(series) != {"dim1", "dim2", "dim3"}:
        raise ValueError("matrix must declare exact Dim1/Dim2/Dim3 convergence series")
    by_id = {case["id"]: case for case in cases}
    declared: set[str] = set()
    for dimension in (1, 2, 3):
        row = series["dim%d" % dimension]
        if not isinstance(row, dict) or set(row) != {
            "case_ids",
            "qualified_norm",
            "reported_norms",
            "minimum_order",
        }:
            raise ValueError("Dim%d convergence declaration has an unsupported shape" % dimension)
        identifiers = row["case_ids"]
        if not isinstance(identifiers, list) or len(identifiers) != 3 or len(set(identifiers)) != 3:
            raise ValueError("Dim%d convergence requires exactly three distinct cases" % dimension)
        selected = [by_id.get(identifier) for identifier in identifiers]
        if any(case is None for case in selected):
            raise ValueError("Dim%d convergence names an unknown case" % dimension)
        if any(
            case["dimension"] != dimension
            or case["layout"] != "uniform"
            or case["mode"] != "x"
            or case["mpi"]
            or case["cycles"] != 1
            for case in selected
        ):
            raise ValueError(
                "Dim%d convergence cases must be serial uniform x-advection at T=1" % dimension
            )
        resolutions = [int(case["resolution"]) for case in selected]
        if resolutions != sorted(resolutions) or len(set(resolutions)) != 3:
            raise ValueError("Dim%d convergence resolutions must strictly increase" % dimension)
        controls = {
            tuple(
                (name, json.dumps(case[name], sort_keys=True))
                for name in (
                    "dimension",
                    "mode",
                    "layout",
                    "subcycling",
                    "block_size",
                    "mpi_ranks",
                    "cycles",
                    "time_snapshots",
                    "mpi",
                )
            )
            for case in selected
        }
        if len(controls) != 1:
            raise ValueError(
                "Dim%d convergence cases must retain identical non-resolution controls" % dimension
            )
        if row["qualified_norm"] != "l1" or row["reported_norms"] != ["l1", "l2", "linf"]:
            raise ValueError("Dim%d convergence qualifies L1 and reports L1/L2/Linf" % dimension)
        if type(row["minimum_order"]) not in (int, float) or not 1.5 <= row["minimum_order"] < 2.0:
            raise ValueError(
                "Dim%d L1 order threshold must be a conservative asymptotic bound" % dimension
            )
        declared.update(identifiers)
    if declared != obligated:
        raise ValueError("second_order_convergence lacks an exact complete series witness")


def _same_configuration(
    left: dict[str, Any], right: dict[str, Any], *, ignored: set[str]
) -> bool:
    return all(
        left[name] == right[name]
        for name in (
            "dimension",
            "resolution",
            "mode",
            "layout",
            "subcycling",
            "block_size",
            "mpi_ranks",
            "cycles",
            "time_snapshots",
            "mpi",
        )
        if name not in ignored
    )


def _validate_comparison_contracts(cases: tuple[dict[str, Any], ...]) -> None:
    """Keep every published comparison a controlled one-variable experiment."""
    by_id = {case["id"]: case for case in cases}
    for left_id, right_id in SUBCYCLING_COMPARISON_PAIRS:
        left = by_id[left_id]
        right = by_id[right_id]
        if (
            left["subcycling"] != "synchronous"
            or right["subcycling"] != "subcycled"
            or not _same_configuration(left, right, ignored={"subcycling"})
        ):
            raise ValueError(
                "%s/%s must differ only by the subcycling selection" % (left_id, right_id)
            )
    mpi_cases = [by_id[identifier] for identifier in MPI_INVARIANCE_CASE_IDS]
    if [case["mpi_ranks"] for case in mpi_cases] != [1, 2, 4] or not all(
        case["mpi"] for case in mpi_cases
    ):
        raise ValueError("MPI invariance requires the exact np=1/2/4 MPI case triplet")
    reference = mpi_cases[0]
    if any(
        not _same_configuration(reference, candidate, ignored={"mpi_ranks", "mpi_topology"})
        for candidate in mpi_cases[1:]
    ):
        raise ValueError("MPI invariance cases must differ only by rank/decomposition identity")
    topology_cases = [by_id[identifier] for identifier in MPI_TOPOLOGY_CASE_IDS]
    expected_topologies = [[1, 1], [1, 2], [2, 2], [2, 2, 2]]
    if [case.get("mpi_topology") for case in topology_cases] != expected_topologies:
        raise ValueError("MPI cases must retain the exact 1x1, 1x2, 2x2, and 2x2x2 decompositions")
    corner = topology_cases[-1]
    if (
        corner["dimension"] != 3
        or corner["mode"] != "diagonal"
        or corner["layout"] != "uniform"
        or corner["block_size"] != 16
        or corner["resolution"] != "32"
        or "block_corner_3d" not in corner["obligations"]
    ):
        raise ValueError("the Dim3 MPI case must authenticate a diagonal 2x2x2 ownership corner")
    block_cases = [by_id[identifier] for identifier in BLOCK_SIZE_COMPARISON_CASE_IDS]
    if [case["block_size"] for case in block_cases] != [8, 16, 32] or any(
        not _same_configuration(block_cases[0], candidate, ignored={"block_size"})
        for candidate in block_cases[1:]
    ):
        raise ValueError("block-size study requires the exact controlled Dim2 8/16/32 triplet")


def _command(case: dict[str, Any], output: Path) -> list[str]:
    generator = CASE_DIRECTORY / "generate_data.py"
    command = [
        sys.executable,
        str(generator),
        "--dimension",
        str(case["dimension"]),
        "--resolution",
        case["resolution"],
        "--mode",
        case["mode"],
        "--layout",
        case["layout"],
        "--subcycling",
        case["subcycling"],
        "--block-size",
        str(case["block_size"]),
        "--cycles",
        str(case["cycles"]),
        "--time-snapshots",
        str(case["time_snapshots"]),
        "--output",
        str(output),
    ]
    for obligation in case["obligations"]:
        command.extend(("--obligation", obligation))
    if case["mpi"]:
        command.append("--mpi")
        command.extend(("--mpi-topology", ",".join(map(str, case["mpi_topology"]))))
        return [_mpi_launcher(), "-n", str(case["mpi_ranks"]), *command]
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    """Hash one JSON authority without depending on dictionary insertion order."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_source_authority(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract the common source authority while deliberately excluding native leaves.

    ``source_provenance`` includes the selected native extension, which must differ
    between Dim1, Dim2, Dim3, and MPI builds. A scientific matrix nevertheless
    needs one exact repository revision and tracked working-tree diff across every
    case. This receipt is therefore the non-native authority shared by all 37 case
    metadata files; native artifacts remain authenticated per pair.
    """
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("matrix result has no provenance")
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("matrix result has no source provenance")
    repository_sha = source.get("repository_sha")
    repository_dirty = source.get("repository_dirty")
    tracked_diff_sha256 = source.get("tracked_diff_sha256")
    files = source.get("files")
    build_tree = source.get("build_tree")
    if (
        not isinstance(repository_sha, str)
        or not repository_sha
        or type(repository_dirty) is not bool
        or not isinstance(tracked_diff_sha256, str)
        or len(tracked_diff_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tracked_diff_sha256)
        or not isinstance(files, dict)
        or not files
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for path, digest in files.items()
        )
        or not isinstance(build_tree, dict)
        or build_tree.get("schema_version") != "pops.sine-wave.build-source-tree.v1"
        or not isinstance(build_tree.get("roots"), list)
        or not all(isinstance(root, str) and root for root in build_tree["roots"])
        or not isinstance(build_tree.get("files"), dict)
        or not build_tree["files"]
        or not isinstance(build_tree.get("fingerprint"), str)
        or len(build_tree["fingerprint"]) != 64
        or any(character not in "0123456789abcdef" for character in build_tree["fingerprint"])
        or build_tree["fingerprint"]
        != _canonical_sha256(
            {key: value for key, value in build_tree.items() if key != "fingerprint"}
        )
    ):
        raise RuntimeError("matrix result has an invalid non-native source authority")
    authority: dict[str, Any] = {
        "schema_version": MATRIX_SOURCE_AUTHORITY_SCHEMA,
        "repository_sha": repository_sha,
        "repository_dirty": repository_dirty,
        "tracked_diff_sha256": tracked_diff_sha256,
        "files": files,
        "build_tree": build_tree,
    }
    authority["fingerprint"] = _canonical_sha256(authority)
    return authority


def _resolved_resolution(case: dict[str, Any]) -> list[int]:
    values = [int(value) for value in case["resolution"].split(",")]
    if len(values) == 1:
        values *= case["dimension"]
    if len(values) != case["dimension"]:
        raise RuntimeError("%s resolution does not match its dimension" % case["id"])
    return values


def _native_compatibility_identity(metadata: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the native build identity shared across dimensional/MPI leaves."""
    source = metadata.get("provenance", {}).get("source")
    if not isinstance(source, dict) or not isinstance(source.get("native"), dict):
        raise RuntimeError("%s has no authenticated selected native receipt" % case["id"])
    native = source["native"]
    if (
        native.get("dimension") != case["dimension"]
        or native.get("has_mpi") is not case["mpi"]
        or native.get("has_kokkos") is not True
        or not isinstance(native.get("version"), str)
        or not native["version"]
        or not isinstance(native.get("build_fingerprint"), str)
        or len(native["build_fingerprint"]) != 64
        or any(character not in "0123456789abcdef" for character in native["build_fingerprint"])
        or not isinstance(native.get("abi_key"), str)
        or not native["abi_key"]
    ):
        raise RuntimeError("%s native receipt disagrees with its required variant" % case["id"])
    tokens: dict[str, str] = {}
    for raw in native["abi_key"].split(";"):
        name, separator, value = raw.partition("=")
        if not separator or not name or not value or name in tokens:
            raise RuntimeError("%s native ABI key is malformed" % case["id"])
        tokens[name] = value
    required = {"compiler", "std", "headers", "kokkos", "stdlib", "dim", "mpi"}
    if not required <= set(tokens):
        raise RuntimeError("%s native ABI key omits compiler/header/Kokkos facts" % case["id"])
    if tokens["dim"] != str(case["dimension"]) or tokens["mpi"] != ("1" if case["mpi"] else "0"):
        raise RuntimeError("%s native ABI key disagrees with its dimension/MPI variant" % case["id"])
    return {
        "version": native["version"],
        "build_fingerprint": native["build_fingerprint"],
        "abi_common": sorted(
            (name, value) for name, value in tokens.items() if name not in NATIVE_PER_VARIANT_ABI_FIELDS
        ),
    }


def _verify_mpi_ownership(case: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Fail closed unless every MPI rank proves the prescribed spatial ownership."""
    runtime = metadata.get("provenance", {}).get("execution", {}).get("runtime")
    source = metadata.get("provenance", {}).get("source")
    coverage = metadata.get("coverage")
    receipt = coverage.get("mpi_topology") if isinstance(coverage, dict) else None
    if (
        metadata.get("mpi") is not True
        or not isinstance(runtime, dict)
        or runtime.get("mpi_compiled") is not True
        or runtime.get("mpi_active") is not True
        or runtime.get("mpi_ranks") != case["mpi_ranks"]
        or not isinstance(source, dict)
        or not isinstance(source.get("native"), dict)
        or source["native"].get("has_mpi") is not True
        or not isinstance(receipt, dict)
    ):
        raise RuntimeError("%s lacks active compiled MPI provenance" % case["id"])
    topology = case["mpi_topology"]
    owners = receipt.get("rank_ownership")
    coordinates = receipt.get("rank_coordinates")
    if (
        receipt.get("requested_ranks") != case["mpi_ranks"]
        or receipt.get("observed_ranks") != case["mpi_ranks"]
        or receipt.get("expected_spatial_decomposition") != topology
        or receipt.get("ownership_active") is not True
        or not isinstance(owners, list)
        or not isinstance(coordinates, list)
        or len(owners) != case["mpi_ranks"]
        or len(coordinates) != case["mpi_ranks"]
    ):
        raise RuntimeError("%s MPI ownership receipt has an invalid cardinality/topology" % case["id"])
    owner_ranks = [row.get("rank") for row in owners if isinstance(row, dict)]
    coordinate_ranks = [row.get("rank") for row in coordinates if isinstance(row, dict)]
    coordinate_values = [row.get("coordinate") for row in coordinates if isinstance(row, dict)]
    if (
        sorted(owner_ranks) != list(range(case["mpi_ranks"]))
        or sorted(coordinate_ranks) != list(range(case["mpi_ranks"]))
        or sorted(tuple(value) for value in coordinate_values if isinstance(value, list))
        != sorted(tuple(value) for value in itertools.product(*(range(n) for n in topology)))
        or any(not isinstance(row.get("local_boxes"), list) or not row["local_boxes"] for row in owners if isinstance(row, dict))
    ):
        raise RuntimeError("%s MPI ownership receipt lacks a bijective active rank decomposition" % case["id"])
    corner = receipt.get("inter_rank_corner_crossing")
    if case["id"] == "d3-mpi-np8-corner":
        if (
            not isinstance(corner, dict)
            or corner.get("observed") is not True
            or corner.get("corner_index") != [16, 16, 16]
            or corner.get("corner_coordinate") != [0.5, 0.5, 0.5]
            or corner.get("participating_ranks") != list(range(8))
            or corner.get("velocity") != [1.0, 1.0, 1.0]
            or not isinstance(corner.get("arrival_time"), (int, float))
            or not 0.0 < corner["arrival_time"] < 1.0
        ):
            raise RuntimeError("d3-mpi-np8-corner lacks an observed inter-rank ownership-corner crossing")
    elif corner is not None:
        raise RuntimeError("%s records an unexpected 3D corner witness" % case["id"])


def _matrix_output_root(output: Path, namespace: str) -> Path:
    """Anchor a relative campaign output at the authenticated repository root."""
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    return output.resolve(strict=False) / namespace


def _verify_output(
    case: dict[str, Any], output: Path, matrix_environment: dict[str, str]
) -> tuple[Path, Path, dict[str, Any]]:
    metadata_paths = tuple(output.glob("*.json"))
    data_paths = tuple(output.glob("*.npz"))
    if len(metadata_paths) != 1 or len(data_paths) != 1:
        raise RuntimeError("%s did not publish exactly one authenticated result pair" % case["id"])
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    for key in ("dimension", "resolution", "mode", "layout", "subcycling", "block_size", "cycles"):
        expected = _resolved_resolution(case) if key == "resolution" else case[key]
        if metadata.get(key) != expected:
            raise RuntimeError("%s metadata differs from its matrix %s" % (case["id"], key))
    if (
        metadata.get("time_snapshots") != case["time_snapshots"]
        or metadata.get("mpi_ranks") != case["mpi_ranks"]
        or metadata.get("mpi") is not case["mpi"]
    ):
        raise RuntimeError("%s metadata differs from its matrix time/MPI selection" % case["id"])
    execution = metadata.get("provenance", {}).get("execution", {})
    recorded_environment = execution.get("environment")
    if not isinstance(recorded_environment, dict) or any(
        recorded_environment.get(name) != value for name, value in matrix_environment.items()
    ):
        raise RuntimeError(
            "%s does not retain the deterministic verification environment" % case["id"]
        )
    runtime = execution.get("runtime")
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("kokkos_backend"), str)
        or not runtime["kokkos_backend"]
        or type(runtime.get("kokkos_concurrency")) is not int
        or runtime["kokkos_concurrency"] < 1
    ):
        raise RuntimeError(
            "%s did not publish a usable native backend/concurrency receipt" % case["id"]
        )
    source = metadata.get("provenance", {}).get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("build_tree"), dict)
        or source["build_tree"].get("roots") != list(BUILD_SOURCE_AUTHORITY_ROOTS)
    ):
        raise RuntimeError("%s did not publish the complete native build-source authority" % case["id"])
    _native_compatibility_identity(metadata, case)
    if case["mpi"]:
        _verify_mpi_ownership(case, metadata)
    elif runtime.get("mpi_active") is not False:
        raise RuntimeError("%s non-MPI result unexpectedly has an active MPI runtime" % case["id"])
    if metadata.get("data") != data_paths[0].name or metadata.get("data_sha256") != _sha256(
        data_paths[0]
    ):
        raise RuntimeError("%s result pair fails its published data digest" % case["id"])
    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict) or set(coverage.get("requested_obligations", ())) != set(
        case["obligations"]
    ):
        raise RuntimeError("%s did not retain its requested coverage obligations" % case["id"])
    witnesses = coverage.get("witnesses")
    if not isinstance(witnesses, dict):
        raise RuntimeError("%s published no coverage witnesses" % case["id"])
    for name in case["obligations"]:
        witness = witnesses.get(name)
        if name == "second_order_convergence":
            if not _is_deferred_convergence_witness(witness):
                raise RuntimeError(
                    "%s must publish the explicit deferred convergence receipt" % case["id"]
                )
            continue
        if not isinstance(witness, dict) or witness.get("observed") is not True:
            raise RuntimeError("%s has no observed witness for %s" % (case["id"], name))
    return data_paths[0], metadata_paths[0], metadata


def _convergence_receipt(
    matrix: dict[str, Any], metadata_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for name, declaration in matrix["convergence_series"].items():
        identifiers = declaration["case_ids"]
        rows = [metadata_by_id[identifier] for identifier in identifiers]
        for identifier, row in zip(identifiers, rows, strict=True):
            coverage = row.get("coverage")
            witnesses = coverage.get("witnesses") if isinstance(coverage, dict) else None
            if (
                not isinstance(coverage, dict)
                or "second_order_convergence"
                not in set(coverage.get("requested_obligations", ()))
                or not isinstance(witnesses, dict)
                or not _is_deferred_convergence_witness(witnesses.get("second_order_convergence"))
            ):
                raise RuntimeError(
                    "%s convergence input %s lacks the explicit deferred matrix receipt"
                    % (name, identifier)
                )
        source_fingerprints = {row.get("source_fingerprint") for row in rows}
        runtime_facts = {
            json.dumps(row["provenance"]["execution"]["runtime"], sort_keys=True) for row in rows
        }
        methods = {json.dumps(row["metrics"]["method"], sort_keys=True) for row in rows}
        if (
            len(source_fingerprints) != 1
            or None in source_fingerprints
            or len(runtime_facts) != 1
            or len(methods) != 1
        ):
            raise RuntimeError(
                "%s convergence inputs do not share source, backend, and method" % name
            )
        resolutions = [int(row["resolution"][0]) for row in rows]
        errors: dict[str, list[float]] = {}
        orders: dict[str, list[float]] = {}
        for norm in declaration["reported_norms"]:
            values = [float(row["metrics"]["errors"][norm]) for row in rows]
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise RuntimeError("%s convergence has invalid %s errors" % (name, norm))
            errors[norm] = values
            orders[norm] = [
                math.log(previous / current) / math.log(current_resolution / previous_resolution)
                for previous, current, previous_resolution, current_resolution in zip(
                    values[:-1], values[1:], resolutions[:-1], resolutions[1:], strict=True
                )
            ]
        qualified = orders[declaration["qualified_norm"]]
        if any(order < declaration["minimum_order"] for order in qualified):
            raise RuntimeError(
                "%s L1 order below conservative %g threshold" % (name, declaration["minimum_order"])
            )
        receipt[name] = {
            "case_ids": identifiers,
            "resolutions": resolutions,
            "qualified_norm": declaration["qualified_norm"],
            "minimum_order": declaration["minimum_order"],
            "errors": errors,
            "orders": orders,
        }
    return receipt


def _write_complete_manifest(
    *,
    matrix_path: Path,
    output_root: Path,
    cases: tuple[dict[str, Any], ...],
    results: dict[str, tuple[Path, Path, dict[str, Any]]],
    convergence: dict[str, Any],
) -> Path:
    complete = output_root / "COMPLETE.json"
    if complete.exists():
        raise FileExistsError("refusing to overwrite complete matrix manifest: %s" % complete)
    source_authority = _matrix_source_authority(results[cases[0]["id"]][2])
    native_presence = []
    for case in cases:
        source = results[case["id"]][2].get("provenance", {}).get("source")
        native_presence.append(isinstance(source, dict) and isinstance(source.get("native"), dict))
    if any(native_presence) and not all(native_presence):
        raise RuntimeError("complete matrix cannot mix native-receipted and unreceipted results")
    native_compatibility = None
    if all(native_presence):
        identities = {
            json.dumps(_native_compatibility_identity(results[case["id"]][2], case), sort_keys=True)
            for case in cases
        }
        if len(identities) != 1:
            raise RuntimeError(
                "complete matrix native variants disagree on compiler/toolchain/header/Kokkos identity"
            )
        native_compatibility = json.loads(identities.pop())
    entries = []
    for case in cases:
        data_path, metadata_path, metadata = results[case["id"]]
        case_source_authority = _matrix_source_authority(metadata)
        if case_source_authority != source_authority:
            raise RuntimeError(
                "complete matrix cannot mix repository revision, tracked diff, source files, or build tree"
            )
        entries.append(
            {
                "case_id": case["id"],
                "data": data_path.relative_to(output_root).as_posix(),
                "data_sha256": _sha256(data_path),
                "metadata": metadata_path.relative_to(output_root).as_posix(),
                "metadata_sha256": _sha256(metadata_path),
                "result_identity": metadata["result_identity"],
                "source_fingerprint": metadata["source_fingerprint"],
                "source_authority_fingerprint": case_source_authority["fingerprint"],
                "native_artifact": metadata["provenance"]["artifact"],
                "runtime": metadata["provenance"]["execution"]["runtime"],
            }
        )
    document = {
        "schema_version": "pops.sine-wave.matrix-complete.v1",
        "matrix": matrix_path.name,
        "matrix_sha256": _sha256(matrix_path),
        "generator": "generate_data.py",
        "generator_sha256": _sha256(CASE_DIRECTORY / "generate_data.py"),
        "driver": "run_matrix.py",
        "driver_sha256": _sha256(CASE_DIRECTORY / "run_matrix.py"),
        "support": "_case_support.py",
        "support_sha256": _sha256(CASE_DIRECTORY / "_case_support.py"),
        "build_script": "scripts/build_python.sh",
        "build_script_sha256": _sha256(BUILD_SCRIPT),
        "source_authority": source_authority,
        "native_compatibility": native_compatibility,
        "case_count": len(cases),
        "pairs": entries,
        "convergence": convergence,
    }
    with complete.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--output", type=Path, default=CASE_DIRECTORY / "results")
    parser.add_argument(
        "--execute", action="store_true", help="run the complete validated campaign"
    )
    args = parser.parse_args()
    if BUILD_SCRIPT.is_symlink() or not BUILD_SCRIPT.is_file():
        raise RuntimeError("matrix driver cannot authenticate the repository build script")
    matrix = _read_matrix(args.matrix)
    cases = _validate_matrix(matrix)
    phases = _phase_cases(matrix, cases)
    output_root = _matrix_output_root(args.output, matrix["result_namespace"])
    commands = [
        (case, output_root / case["id"], _command(case, output_root / case["id"])) for case in cases
    ]
    if not args.execute:
        print(
            json.dumps(
                {
                    "matrix": str(args.matrix),
                    "cases": len(cases),
                    "build_phases": [phase for phase, _ in phases],
                    "commands": [command for _, _, command in commands],
                },
                indent=2,
            )
        )
        return
    if any(case["mpi"] for case in cases):
        _require_mpi_launcher()
    existing = [output for _, output, _ in commands if output.exists()]
    if existing or (output_root / "COMPLETE.json").exists():
        collision = existing[0] if existing else output_root / "COMPLETE.json"
        raise FileExistsError("refusing to overwrite an existing matrix target: %s" % collision)
    output_root.mkdir(parents=True, exist_ok=True)
    for case, output, _ in commands:
        output.mkdir()
        reservation = output / ".matrix-reservation"
        with reservation.open("x", encoding="utf-8") as stream:
            stream.write(case["id"] + "\n")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(matrix["verification_environment"])
    results: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    by_id = {case["id"]: (output, command) for case, output, command in commands}
    for phase, phase_cases in phases:
        build_command = [
            "bash",
            str(BUILD_SCRIPT),
            "--dim",
            str(phase["dimension"]),
            "--clean",
        ]
        if phase["mpi"]:
            build_command.append("--mpi")
        subprocess.run(build_command, check=True, cwd=REPOSITORY_ROOT, env=environment)
        if phase["mpi"]:
            _preflight_mpi_launcher(phase["dimension"], environment)
        for case in phase_cases:
            output, command = by_id[case["id"]]
            subprocess.run(command, check=True, cwd=CASE_DIRECTORY, env=environment)
            results[case["id"]] = _verify_output(case, output, matrix["verification_environment"])
    convergence = _convergence_receipt(matrix, {key: value[2] for key, value in results.items()})
    complete = _write_complete_manifest(
        matrix_path=args.matrix,
        output_root=output_root,
        cases=cases,
        results=results,
        convergence=convergence,
    )
    print("complete: %s" % complete)


if __name__ == "__main__":
    main()
