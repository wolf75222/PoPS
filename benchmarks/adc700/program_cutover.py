#!/usr/bin/env python3
"""ADC-700 candidate campaign driver.

This is the candidate half of the out-of-CI ADC-700 campaign.  It deliberately keeps the
authoring and native boundaries visible:

``Model + Program -> validate -> resolve -> compile(MODULE) -> bind -> run``

The shared library is produced by the installed PoPS compiler with the caller-supplied NVCC/MPI/
Kokkos toolchain.  Binding is the only place where the generated module is installed; for an AMR
layout the public binding reaches the native :class:`AmrSystem`, whose ``install_program`` seam is
asserted through the installed program hash.  There is no Python numerical fallback and no local
claim of ROMEO execution: every missing accelerator, MPI, profiler counter, digest, or topology
fact is a hard failure.

The driver writes exactly one measurement record on rank zero.  The companion ``verify.py``
authenticates the complete JSONL/ABBA campaign after the batch job has collected both routes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

import pops
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    PatchLayout,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Density, Energy, Model, Momentum
from pops.projection import ConservativeCellAverage
from pops.runtime._profile import Profile
from pops.runtime_environment import runtime_environment_report
from pops.time import FixedDt, StagePoint, TimePoint


SCHEMA = "pops.adc700.program_cutover.measurement.v1"
BASELINE_REVISION = "db3d390f43dfb14f12e88db31a9b3e631ff50488"
DISPATCH_COMPLEXITY = "O(operations*levels), never O(cells)"
ACCELERATOR_TOKENS = ("cuda", "hip", "sycl")
GAMMA = 1.4
WHEEL_PROOF_SCHEMA_VERSION = 3
PROBE_RESOLUTIONS = (16, 32)
CAMPAIGN_HARNESSES = (
    "benchmarks/adc700/CMakeLists.txt",
    "benchmarks/adc700/compare.py",
    "benchmarks/adc700/program_cutover.cpp",
    "benchmarks/adc700/program_cutover.py",
    "benchmarks/adc700/archive_receipt.py",
    "benchmarks/adc700/verify.py",
    "benchmarks/adc700/with_gpu_identity.sh",
    "benchmarks/manifest.toml",
    "benchmarks/romeo/adc700_program_cutover.sbatch",
    "benchmarks/romeo/submit_adc700_program_cutover.sh",
    "scripts/build_python.sh",
    "scripts/conda_runtime.sh",
    "scripts/prove_installed_wheel.py",
    "scripts/preserve_native_variants.py",
    "scripts/write_native_variant_manifest.py",
    "scripts/codesign_pops_extensions.py",
    "scripts/verify_installed_native.py",
)


class CampaignError(RuntimeError):
    """An absent or unauthenticated campaign fact; never recover silently."""


def _finite(value: Any, *, where: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError("%s is not finite" % where)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise CampaignError("%s must be an absolute path" % where)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError("cannot read %s: %s" % (where, error)) from error
    if not isinstance(payload, dict):
        raise CampaignError("%s must contain one JSON object" % where)
    return payload


def _require_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError("%s must be non-empty text" % where)
    return value


def _require_text_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise CampaignError("%s must be a non-empty sequence" % where)
    result = tuple(_require_text(item, where="%s[]" % where) for item in value)
    if len(set(result)) != len(result):
        raise CampaignError("%s must not contain duplicates" % where)
    return result


def _optional_text_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise CampaignError("%s must be a sequence" % where)
    result = tuple(_require_text(item, where="%s[]" % where) for item in value)
    if len(set(result)) != len(result):
        raise CampaignError("%s must not contain duplicates" % where)
    return result


def _command_version(executable: Path, *, where: str) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CampaignError("cannot query %s version: %s" % (where, error)) from error
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or not output:
        raise CampaignError("%s --version did not return authenticated output" % where)
    return output


def _resolved_executable(value: Any, *, where: str) -> Path:
    text = _require_text(value, where=where)
    path = Path(text).expanduser()
    if path.is_symlink():
        raise CampaignError("%s must not be a symlink: %s" % (where, path))
    if not path.is_file():
        located = shutil.which(text)
        if located is None:
            raise CampaignError("%s is not an executable file: %s" % (where, text))
        path = Path(located)
    if path.is_symlink():
        raise CampaignError("%s must not resolve through a symlink: %s" % (where, path))
    path = path.resolve()
    if not os.access(path, os.X_OK):
        raise CampaignError("%s is not executable: %s" % (where, path))
    return path


def _file_provenance(path: Path, *, where: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError("%s must be a regular non-symlink file: %s" % (where, path))
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise CampaignError("%s must be a regular non-symlink file: %s" % (where, path))
    return {"path": str(path), "sha256": _sha256(path)}


def _campaign_harness_provenance() -> dict[str, dict[str, str]]:
    """Hash only the immutable campaign files from the archived candidate tree."""
    root = Path(__file__).resolve().parents[2]
    if not root.is_dir() or root.name != "candidate":
        raise CampaignError(
            "candidate driver must execute from the archived candidate tree, got %s" % root
        )
    harnesses: dict[str, dict[str, str]] = {}
    for relative in CAMPAIGN_HARNESSES:
        path = root / relative
        if path.is_symlink():
            raise CampaignError("campaign harness is a symlink: %s" % relative)
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise CampaignError("campaign harness escaped archived candidate tree: %s" % relative) from error
        harnesses[relative] = _file_provenance(path, where="campaign harness %s" % relative)
    return harnesses


def _archive_provenance(
    receipt_path: Path, *, revision: str, helper_path: Path
) -> dict[str, Any]:
    """Reverify the frozen candidate archive before importing/compiling a MODULE.

    The helper is deliberately copied to an immutable work-root path outside the candidate tree.
    This is required because the pinned pre-cutover baseline does not contain the ADC-700 helper;
    the candidate archive's helper bytes are still checked against that external copy before it is
    executed for either tree.
    """
    root = Path(__file__).resolve().parents[2]
    if root.name != "candidate" or not root.is_dir() or root.is_symlink():
        raise CampaignError("candidate driver must execute from an immutable candidate archive")
    archived_script = root / "benchmarks" / "adc700" / "archive_receipt.py"
    if archived_script.is_symlink() or not archived_script.is_file():
        raise CampaignError("archived candidate tree has no archive receipt helper")
    script = helper_path.expanduser()
    if not script.is_absolute() or script.is_symlink() or not script.is_file():
        raise CampaignError("external archive receipt helper must be an absolute regular file")
    script = script.resolve()
    try:
        script.relative_to(root)
    except ValueError:
        pass
    else:
        raise CampaignError("external archive receipt helper must be outside the candidate tree")
    if script.stat().st_mode & 0o222:
        raise CampaignError("external archive receipt helper must be immutable")
    archived_sha = _sha256(archived_script)
    if _sha256(script) != archived_sha:
        raise CampaignError("external archive receipt helper differs from candidate archive")
    receipt = receipt_path.expanduser()
    if not receipt.is_absolute() or receipt.is_symlink() or not receipt.is_file():
        raise CampaignError("candidate archive receipt must be an absolute regular file")
    receipt = receipt.resolve()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--root",
                str(root),
                "--role",
                "candidate",
                "--revision",
                revision,
                "--receipt",
                str(receipt),
                "--helper",
                str(script),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CampaignError("cannot reverify candidate archive receipt: %s" % error) from error
    if result.returncode != 0:
        raise CampaignError(
            "candidate archive receipt verification failed: %s" % result.stderr.strip()
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CampaignError("archive receipt helper did not return JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != "pops.adc700.archive_receipt.v1":
        raise CampaignError("candidate archive receipt helper returned an invalid payload")
    if payload.get("schema_version") != 1 or payload.get("role") != "candidate" \
            or payload.get("revision") != revision or payload.get("root_path") != str(root) \
            or payload.get("immutable") is not True:
        raise CampaignError("candidate archive receipt is not linked to this immutable SHA/tree")
    if type(payload.get("entry_count")) is not int or payload["entry_count"] <= 0:
        raise CampaignError("candidate archive receipt has no file manifest")
    tree_sha256 = _require_text(payload.get("tree_sha256"), where="archive tree digest")
    if len(tree_sha256) != 64 or any(char not in "0123456789abcdef" for char in tree_sha256.lower()):
        raise CampaignError("candidate archive tree digest is malformed")
    receipt_script = payload.get("receipt_script")
    if not isinstance(receipt_script, dict) or set(receipt_script) != {"path", "sha256"}:
        raise CampaignError("candidate archive receipt lacks helper provenance")
    if receipt_script["path"] != str(script) \
            or not isinstance(receipt_script["sha256"], str) \
            or len(receipt_script["sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in receipt_script["sha256"].lower()):
        raise CampaignError("candidate archive receipt helper provenance is invalid")
    if receipt_script["sha256"] != _sha256(script) or receipt_script["sha256"] != archived_sha:
        raise CampaignError("candidate archive receipt helper bytes are not candidate-authenticated")
    return {
        "path": str(receipt),
        "sha256": _sha256(receipt),
        "revision": revision,
        "tree_sha256": tree_sha256,
        "entry_count": payload.get("entry_count"),
        "script_sha256": _require_text(
            receipt_script.get("sha256"),
            where="archive receipt script digest",
        ),
    }


def _gpu_assignments(native: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate every rank's GPU UUID for this individual measurement run."""
    uuid = os.environ.get("POPS_ADC700_GPU_UUID", "").strip()
    if not uuid:
        raise CampaignError("each candidate run must carry POPS_ADC700_GPU_UUID")
    world = native.mpi_world()
    rows = world.allgather_bytes(uuid.encode("utf-8"))
    if len(rows) != 4:
        raise CampaignError("GPU identity allgather did not return four ranks")
    uuids = []
    for index, payload in enumerate(rows):
        try:
            item = payload.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise CampaignError("GPU UUID for rank %d is not UTF-8" % index) from error
        if not item or any(char in item for char in '"\\\n\r\t'):
            raise CampaignError("GPU UUID for rank %d is empty or JSON-unsafe" % index)
        uuids.append(item)
    if len(set(uuids)) != 4:
        raise CampaignError("candidate run did not authenticate four distinct GPU UUIDs")
    assignments = [{"rank": index, "uuid": item} for index, item in enumerate(uuids)]
    local_rank = int(native.my_rank())
    if local_rank not in range(4):
        raise CampaignError("native MPI rank is outside the four-rank world")
    local = {"rank": local_rank, "uuid": uuid}
    if assignments[local_rank] != local:
        raise CampaignError("rank-local GPU UUID disagrees with the MPI allgather")
    return assignments, local


def _require_runtime() -> tuple[dict[str, Any], Any]:
    """Require the exact 4-rank accelerator runtime used by the ROMEO job."""
    native = getattr(pops, "_pops", None)
    if native is None:
        try:
            from pops import _pops as native  # type: ignore[no-redef]
        except Exception as error:  # noqa: BLE001 - turn absence into a campaign failure
            raise CampaignError("the installed native _pops extension is unavailable") from error
    facts = dict(runtime_environment_report())
    if facts.get("mpi_compiled") is not True or facts.get("mpi_active") is not True:
        raise CampaignError("ADC-700 requires an active MPI-enabled native extension")
    ranks = facts.get("mpi_ranks")
    if type(ranks) is not int or ranks != 4:
        raise CampaignError("ADC-700 requires exactly four MPI ranks, got %r" % (ranks,))
    if int(native.n_ranks()) != 4:
        raise CampaignError("loaded native MPI world does not contain exactly four ranks")
    if facts.get("communicator") != "MPI_COMM_WORLD":
        raise CampaignError("ADC-700 requires the native MPI_COMM_WORLD communicator")
    rank = facts.get("mpi_rank")
    if type(rank) is not int or rank != int(native.my_rank()) or not 0 <= rank < 4:
        raise CampaignError("runtime MPI rank is not authenticated against MPI_COMM_WORLD")
    if getattr(native, "__has_mpi__", False) is not True:
        raise CampaignError("_pops was not built with MPI")
    if getattr(native, "__has_kokkos__", False) is not True:
        raise CampaignError("_pops was not built with Kokkos")
    native_dimension = getattr(native, "__native_dimension__", None)
    if type(native_dimension) is not int or facts.get("dimension") != native_dimension:
        raise CampaignError("runtime dimension is not authenticated against the loaded extension")
    execution_tokens = " ".join(
        str(facts.get(key, "")) for key in ("kokkos_backend", "kokkos_device", "kokkos_shared_space")
    ).lower()
    if not any(token in execution_tokens for token in ACCELERATOR_TOKENS):
        raise CampaignError("ADC-700 requires a CUDA/HIP/SYCL Kokkos execution space")
    backend = str(facts.get("kokkos_backend", ""))
    if not any(token in backend.lower() for token in ACCELERATOR_TOKENS):
        raise CampaignError("ADC-700 requires an authenticated accelerator backend name")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise CampaignError(
            "each rank must receive one CUDA_VISIBLE_DEVICES entry; global distinctness is "
            "authenticated by the batch inventory"
        )
    return facts, native


def _require_baseline_toolchain_attestation() -> None:
    """Require the sbatch/CMake baseline preflight before a candidate measurement."""
    if os.environ.get("POPS_ADC700_BASELINE_TOOLCHAIN_ATTESTED") != "1":
        raise CampaignError(
            "candidate measurement requires the baseline CMake toolchain-contract attestation"
        )


def _contract_file_rows(contract: dict[str, Any], *, paths_key: str, hashes_key: str,
                        where: str) -> list[dict[str, str]]:
    paths = _require_text_tuple(contract.get(paths_key), where="%s.%s" % (where, paths_key))
    hashes = _require_text_tuple(contract.get(hashes_key), where="%s.%s" % (where, hashes_key))
    if len(paths) != len(hashes):
        raise CampaignError("%s path/hash counts differ" % where)
    rows = []
    for path_text, expected in zip(paths, hashes, strict=True):
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
            raise CampaignError("%s contains a malformed SHA-256" % where)
        path = Path(path_text).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise CampaignError("%s references a missing file: %s" % (where, path))
        path = path.resolve()
        if path.is_symlink() or not path.is_file():
            raise CampaignError("%s references a non-regular file: %s" % (where, path))
        observed = _sha256(path)
        if observed != expected.lower():
            raise CampaignError("%s changed in place: %s" % (where, path))
        rows.append({"path": str(path), "sha256": observed})
    return rows


def _require_toolchain(cxx: str, include: Path, native: Any) -> tuple[str, dict[str, Any]]:
    include = include.resolve()
    if not include.is_dir():
        raise CampaignError("PoPS include root does not exist: %s" % include)
    resolved = _resolved_executable(cxx, where="ADC-700 NVCC wrapper")
    if "nvcc_wrapper" not in resolved.name.lower():
        raise CampaignError("ADC-700 candidate must compile with the Kokkos nvcc_wrapper: %s" % resolved)
    version = _command_version(resolved, where="NVCC wrapper")
    baked_cxx = _require_text(
        getattr(native, "__cxx_compiler__", ""), where="loaded extension compiler"
    )
    if Path(baked_cxx).expanduser().resolve() != resolved:
        raise CampaignError(
            "candidate NVCC wrapper differs from the compiler baked into _pops: %s != %s"
            % (resolved, baked_cxx)
        )
    native_std = getattr(native, "__cxx_std__", None)
    if type(native_std) is not int or native_std != 20:
        raise CampaignError("ADC-700 requires the exact C++20 native extension ABI")

    kokkos = getattr(native, "__kokkos_contract__", None)
    if not isinstance(kokkos, dict) or kokkos.get("schema_version") != 1:
        raise CampaignError("loaded extension has no authenticated Kokkos contract")
    kokkos_headers = _contract_file_rows(
        kokkos, paths_key="header_paths", hashes_key="header_sha256", where="Kokkos contract"
    )
    kokkos_includes = _require_text_tuple(kokkos.get("include_dirs"), where="Kokkos contract.include_dirs")
    if any(not Path(item).is_absolute() or not Path(item).is_dir() for item in kokkos_includes):
        raise CampaignError("Kokkos contract.include_dirs must name absolute directories")
    mpi = getattr(native, "__mpi_contract__", None)
    if not isinstance(mpi, dict) or mpi.get("schema_version") != 1:
        raise CampaignError("loaded extension has no authenticated MPI contract")
    mpi_headers = _contract_file_rows(
        mpi, paths_key="header_paths", hashes_key="header_sha256", where="MPI contract"
    )
    mpi_libraries = _contract_file_rows(
        mpi, paths_key="library_paths", hashes_key="library_sha256", where="MPI contract"
    )
    mpi_compiler = _resolved_executable(mpi.get("compiler"), where="MPI compiler")
    mpi_version = _command_version(mpi_compiler, where="MPI compiler")
    mpi_includes = _require_text_tuple(mpi.get("include_dirs"), where="MPI contract.include_dirs")
    if any(not Path(item).is_absolute() or not Path(item).is_dir() for item in mpi_includes):
        raise CampaignError("MPI contract.include_dirs must name absolute directories")
    compile_options = _optional_text_tuple(mpi.get("compile_options"), where="MPI contract.compile_options")
    compile_definitions = _optional_text_tuple(
        mpi.get("compile_definitions"), where="MPI contract.compile_definitions"
    )
    link_options = _optional_text_tuple(mpi.get("link_options"), where="MPI contract.link_options")
    link_libraries = _require_text_tuple(
        mpi.get("link_libraries"), where="MPI contract.link_libraries"
    )
    if tuple(link_libraries) != tuple(item["path"] for item in mpi_libraries):
        raise CampaignError("MPI link libraries do not match authenticated library files")
    loader = getattr(native, "__native_loader_contract__", None)
    if not isinstance(loader, dict) or loader.get("schema_version") != 1:
        raise CampaignError("loaded extension has no authenticated native-loader contract")
    loader_definitions = _require_text_tuple(
        loader.get("compile_definitions"), where="native-loader contract.compile_definitions"
    )
    # ``pops.compile`` obtains these exact flags from the central production loader helper.  Keep
    # them in the receipt so the pinned C++ route and every candidate row can be compared against
    # the same compiler/MPI/Kokkos/link contract, rather than merely comparing component labels.
    from pops.codegen.compile_link_flags import deterministic_program_link_flags
    from pops.codegen.toolchain import pops_loader_build_flags

    effective_cxx, native_compile_flags, native_link_flags = pops_loader_build_flags(str(resolved))
    effective_cxx = _resolved_executable(effective_cxx, where="effective native compiler")
    if effective_cxx != resolved:
        raise CampaignError(
            "central production loader selected a compiler different from the NVCC wrapper: %s != %s"
            % (effective_cxx, resolved)
        )
    compile_flags = _optional_text_tuple(
        native_compile_flags, where="production compile flags"
    )
    link_flags = _optional_text_tuple(
        deterministic_program_link_flags(native_link_flags), where="production link flags"
    )
    return str(resolved), {
        "nvcc_wrapper": {
            **_file_provenance(resolved, where="NVCC wrapper"),
            "version": version,
        },
        "mpi": {
            "schema_version": 1,
            "compiler": str(mpi_compiler),
            "compiler_version": mpi_version,
            "compiler_sha256": _sha256(mpi_compiler),
            "abi_sha256": _require_text(mpi.get("abi_sha256"), where="MPI contract.abi_sha256"),
            "standard": _require_text(mpi.get("standard"), where="MPI contract.standard"),
            "compile_options": list(compile_options),
            "compile_definitions": list(compile_definitions),
            "link_options": list(link_options),
            "link_libraries": list(link_libraries),
            "include_dirs": list(mpi_includes),
            "headers": mpi_headers,
            "libraries": mpi_libraries,
        },
        "kokkos": {
            "schema_version": 1,
            "abi_sha256": _require_text(kokkos.get("abi_sha256"), where="Kokkos contract.abi_sha256"),
            "include_dirs": list(kokkos_includes),
            "headers": kokkos_headers,
        },
        "native_loader": {
            "schema_version": 1,
            "compile_definitions": list(loader_definitions),
        },
        "requested": {
            "cxx": str(resolved),
            "include": str(include),
            "std": "c++%d" % native_std,
            "compile_flags": list(compile_flags),
            "link_flags": list(link_flags),
        },
    }


def _write_toolchain_receipt(path: Path, *, toolchain: dict[str, Any]) -> None:
    """Persist the exact candidate toolchain object consumed by the C++ oracle."""
    target = path.expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists():
        raise CampaignError("toolchain receipt output must be a new absolute regular path")
    root = Path(__file__).resolve().parents[2]
    try:
        target.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        raise CampaignError("toolchain receipt must be outside the archived candidate tree")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(toolchain, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    target.chmod(0o444)


def _merge_cmake_target_contract(
    path: Path, *, toolchain: dict[str, Any]
) -> dict[str, Any]:
    """Bind the candidate extension facts to the concrete CMake targets used by the oracle.

    The probe is generated by the archived ADC-700 CMake harness with the same compiler, MPI
    wrapper, and Kokkos root as the installed wheel.  It is not trusted as a label: every common
    field is compared to the extension's authenticated contract before the target-only options,
    definitions, and libraries are retained in the shared receipt.
    """
    probe = path.expanduser()
    if not probe.is_absolute() or probe.is_symlink() or not probe.is_file():
        raise CampaignError("CMake target contract must be an absolute regular file")
    probe = probe.resolve()
    if probe.stat().st_mode & 0o222:
        raise CampaignError("CMake target contract must be immutable")
    payload = _read_json(probe, where="CMake target contract")
    if set(payload) != {"nvcc_wrapper", "mpi", "kokkos"}:
        raise CampaignError("CMake target contract has missing/extra top-level fields")
    for component in ("nvcc_wrapper", "mpi", "kokkos"):
        if not isinstance(payload[component], dict):
            raise CampaignError("CMake target contract %s is not an object" % component)
    if payload["nvcc_wrapper"] != toolchain["nvcc_wrapper"]:
        raise CampaignError("CMake target NVCC wrapper differs from loaded extension")
    if payload["mpi"] != toolchain["mpi"]:
        raise CampaignError("CMake target MPI contract differs from loaded extension")
    kokkos = payload["kokkos"]
    required = {
        "schema_version", "abi_sha256", "include_dirs", "headers", "compile_options",
        "compile_definitions", "link_options", "link_libraries",
    }
    if set(kokkos) != required or kokkos.get("schema_version") != 1:
        raise CampaignError("CMake target Kokkos contract has missing/extra fields")
    current = toolchain["kokkos"]
    if any(kokkos.get(field) != current.get(field)
           for field in ("schema_version", "abi_sha256", "include_dirs", "headers")):
        raise CampaignError("CMake target Kokkos headers/ABI differ from loaded extension")
    for field in ("compile_options", "compile_definitions", "link_options", "link_libraries"):
        value = kokkos[field]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise CampaignError("CMake target Kokkos %s is malformed" % field)
        if len(set(value)) != len(value):
            raise CampaignError("CMake target Kokkos %s contains duplicates" % field)
    merged = dict(toolchain)
    merged["kokkos"] = dict(current)
    merged["kokkos"].update({
        "compile_options": list(kokkos["compile_options"]),
        "compile_definitions": list(kokkos["compile_definitions"]),
        "link_options": list(kokkos["link_options"]),
        "link_libraries": list(kokkos["link_libraries"]),
    })
    return merged


def _toolchain_receipt(path: Path, *, toolchain: dict[str, Any], revision: str) -> dict[str, str]:
    """Re-read the immutable receipt and require byte/object equality with runtime facts."""
    target = path.expanduser()
    if not target.is_absolute() or target.is_symlink() or not target.is_file():
        raise CampaignError("toolchain receipt must be an absolute regular file")
    target = target.resolve()
    root = Path(__file__).resolve().parents[2]
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise CampaignError("toolchain receipt must be outside the archived candidate tree")
    mode = target.stat().st_mode
    if mode & 0o222:
        raise CampaignError("toolchain receipt must be immutable")
    payload = _read_json(target, where="toolchain receipt")
    if payload != toolchain:
        raise CampaignError("toolchain receipt differs from the active extension contract")
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "revision": revision,
    }


def _emit_toolchain_receipt(config: argparse.Namespace) -> int:
    """Emit one shared toolchain object for the baseline C++ oracle and candidate MODULE."""
    facts, native = _require_runtime()
    include_arg = Path(config.include).expanduser()
    if include_arg.is_symlink() or not include_arg.is_absolute():
        raise CampaignError("candidate include root must be an absolute non-symlink path")
    include = include_arg.resolve()
    cxx, toolchain = _require_toolchain(config.cxx, include, native)
    if not config.toolchain_probe:
        raise CampaignError("--toolchain-probe is mandatory for toolchain-only mode")
    toolchain = _merge_cmake_target_contract(
        Path(config.toolchain_probe), toolchain=toolchain
    )
    extension = Path(_require_text(getattr(native, "__file__", ""), where="native extension"))
    wheel = _wheel_provenance(
        Path(config.wheel_proof).expanduser().resolve(), native=native, extension=extension.resolve()
    )
    # The receipt is a raw canonical toolchain object so the pinned C++ oracle can inject the
    # complete object into its JSONL record without carrying a second JSON parser.
    if wheel["dimension"] != 2 or not wheel["abi_key"]:
        raise CampaignError("toolchain receipt is not linked to the retained Dim=2 wheel")
    if int(native.my_rank()) == 0:
        _write_toolchain_receipt(Path(config.toolchain_receipt), toolchain=toolchain)
    return 0


def _mpi_barrier(native: Any) -> Any:
    world = native.mpi_world()
    world.barrier()
    return world


def _collective_max(native: Any, value: float) -> float:
    """Reduce timing with the extension's exact MPI world, not a second communicator."""
    world = native.mpi_world()
    rank = int(native.my_rank())
    rows = world.gather_bytes(repr(_finite(value, where="local timing")).encode(), root=0)
    if rank == 0:
        if rows is None or len(rows) != 4:
            raise CampaignError("MPI gather did not return four timing samples")
        maximum = max(float(row.decode()) for row in rows)
        payload = repr(_finite(maximum, where="collective timing")).encode()
    else:
        payload = b""
    return float(world.broadcast_bytes(payload, root=0).decode())


def _euler_initial(n: int) -> np.ndarray:
    axis = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    dx = x - 0.37
    dy = y - 0.41
    density = 1.0 + 0.35 * np.exp(-(dx * dx + dy * dy) / 0.008)
    pressure = 2.0 + 0.1 * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    velocity_x, velocity_y = 0.15, -0.07
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return np.ascontiguousarray(
        np.stack((density, density * velocity_x, density * velocity_y, energy))
    )


def _euler_case(n: int, dt: float) -> tuple[Any, Any, Any, Any, np.ndarray, int]:
    """Author the same 2-D Euler/Rusanov workload through the real AMR target.

    The pinned native oracle starts with a coarse-only hierarchy.  The candidate therefore uses an
    explicit two-level AMR authority whose frozen tagging threshold is above this workload's
    density range: ``AmrSystem`` is still the installed executor, while the measured level-zero
    topology and numerical path remain comparable to the oracle.  The refined/planned scalar probe
    below separately authenticates the level-one path.
    """
    frame = Rectangle(
        "adc700-euler-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("adc700-euler-model", frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "rho_v", "E"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(x_axis),
            "rho_v": Momentum(y_axis),
            "E": Energy(),
        },
    )
    rho, rhou, rhov, energy = state
    u = model.primitive("u", rhou / rho)
    v = model.primitive("v", rhov / rho)
    pressure = model.primitive(
        "p", (GAMMA - 1.0) * (energy - 0.5 * rho * (u * u + v * v))
    )
    sound = model.primitive("c", sqrt(GAMMA * pressure / rho))
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (rhou, rhou * u + pressure, rhou * v, (energy + pressure) * u),
            y_axis: (rhov, rhov * u, rhov * v + pressure, (energy + pressure) * v),
        },
    )
    model.wave_speeds(
        flux,
        frame=frame,
        values={x_axis: (u - sound, u + sound), y_axis: (v - sound, v + sound)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("adc700-euler-case")
    block = case.block("gas", model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = pops.Program("adc700-euler-program")
    temporal = program.state(instance)
    stage = StagePoint("adc700-euler-stage", {"main": TimePoint(program.clock, 0)})
    rhs = program.value("adc700-euler-rhs", rate(temporal.n), at=stage)
    next_value = program.value(
        "adc700-euler-next", temporal.n + program.dt * rhs, at=temporal.next.point
    )
    program.commit(temporal.next, next_value)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(
        InitialCondition(state=instance, value=BindArray(), projection=ConservativeCellAverage())
    )
    threshold = case.param(RuntimeParam("adc700-euler-refine-threshold", default=10.0))
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=n // 2),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(instance) > ValueExpr(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    return case, layout, instance, program, _euler_initial(n), len(program.ir_nodes(recursive=True))


def _scalar_case(
    n: int, dt: float, *, adaptive: bool, name: str
) -> tuple[Any, Any, Any, np.ndarray, int]:
    """Author a scalar probe; AMR uses a frozen, preplanned two-level hierarchy."""
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("%s-model" % name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.2 * rho,), y_axis: (-0.1 * rho,)},
        waves={x_axis: (0.2,), y_axis: (-0.1,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("%s-case" % name)
    block = case.block("probe", model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = pops.Program("%s-program" % name)
    temporal = program.state(instance)
    stage = StagePoint("%s-stage" % name, {"main": TimePoint(program.clock, 0)})
    rhs = program.value("%s-rhs" % name, rate(temporal.n), at=stage)
    next_value = program.value(
        "%s-next" % name, temporal.n + program.dt * rhs, at=temporal.next.point
    )
    program.commit(temporal.next, next_value)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(
        InitialCondition(state=instance, value=BindArray(), projection=ConservativeCellAverage())
    )
    values = np.full((1, n, n), 1.25, dtype=np.float64)
    if adaptive:
        threshold = case.param(RuntimeParam("%s-threshold" % name, default=1.05))
        transfer = AMRTransfer()
        transfer.state(instance, StateTransfer())
        layout = AMR(
            grid=CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=n // 2),
            tagging=AMRTagging(
                rules=(Tag(ValueExpr(instance) > ValueExpr(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid.frozen(),
            transfer=transfer,
            execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
        )
    else:
        layout = Uniform(
            CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes))
        )
    return case, layout, instance, values, len(program.ir_nodes(recursive=True))


def _compile_and_bind(
    case: Any,
    layout: Any,
    instance: Any,
    values: np.ndarray,
    *,
    include: Path,
    cxx: str,
) -> tuple[Any, Any]:
    validated = pops.validate(case)
    resolved = pops.resolve(
        validated,
        layout=layout,
        backend=Production(),
        compile_options={"include": str(include.resolve()), "cxx": cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    simulation = pops.bind(artifact, initial_values={instance: np.ascontiguousarray(values)})
    if simulation.installed_program_hash() != artifact.program_hash:
        raise CampaignError("installed Program hash differs from the compiled MODULE")
    return artifact, simulation


def _assert_native_executor(simulation: Any, *, adaptive: bool) -> Any:
    from pops.runtime._system import AmrSystem, System

    executor = getattr(simulation, "_executor", None)
    expected = AmrSystem if adaptive else System
    if not isinstance(executor, expected):
        raise CampaignError(
            "pops.bind selected %s instead of the authenticated %s runtime"
            % (type(executor).__name__, expected.__name__)
        )
    if adaptive and not callable(getattr(executor, "install_program", None)):
        raise CampaignError("AMR executor has no AmrSystem.install_program seam")
    return executor


def _profile_probe(
    simulation: Any,
    *,
    operations: int,
    expected_levels: int,
    dt: float,
    warmups: int,
    measured_steps: int,
    cell_count: int,
) -> dict[str, Any]:
    """Profile a prepared executor and authenticate dispatches independently of cell count.

    Binding and warmups are an explicit preparation phase.  They are profiled and retained as
    evidence, then the native counters are reset at the preparation boundary; only the subsequent
    measured window can satisfy the zero-allocation contract.  The caller repeats this exact probe at
    another resolution with the same Program IR and hierarchy depth.
    """
    # RuntimeInstance intentionally exposes no profiling/delegation method.  The typed profile
    # context therefore belongs to the exact native executor selected by ``pops.bind``; numerical
    # execution still goes through the public ``pops.run(simulation, ...)`` boundary below.
    executor = getattr(simulation, "_executor", None)
    profile_context = getattr(executor, "profile", None)
    if not callable(profile_context):
        raise CampaignError("bound runtime has no native System/AmrSystem profiling seam")
    reset_profile = getattr(executor, "reset_profiling", None)
    if not callable(reset_profile):
        raise CampaignError("bound runtime has no native profiling reset seam")
    preparation_summary = None
    if warmups > 0:
        with profile_context(Profile.Advanced()) as preparation_profile:
            pops.run(
                simulation,
                t_end=float((warmups + 1) * dt),
                max_steps=warmups,
                console=False,
            )
        preparation_summary = preparation_profile.summary()
    # This reset is the explicit allocation-free boundary: bind and all warmups have completed,
    # and their counters remain attached below rather than being silently discarded.
    reset_profile()
    _mpi_barrier(getattr(pops, "_pops"))
    with profile_context(Profile.Advanced()) as profile:
        pops.run(
            simulation,
            t_end=float(simulation.time() + measured_steps * dt),
            max_steps=measured_steps,
            console=False,
        )
    summary = profile.summary()
    counters = dict(summary.counters())
    memory = summary.by_memory()
    if not isinstance(memory, dict) or "scratch_allocs" not in memory:
        raise CampaignError("Advanced profile did not emit scratch_allocs")
    if "kernels" not in counters:
        raise CampaignError("Advanced profile did not emit kernel dispatch counters")
    if type(counters["kernels"]) is not int:
        raise CampaignError("Advanced profile kernel dispatch counter is not an integer")
    if type(memory["scratch_allocs"]) is not int:
        raise CampaignError("Advanced profile scratch allocation counter is not an integer")
    allocations = int(memory["scratch_allocs"])
    dispatches = int(counters["kernels"])
    levels = int(simulation.n_levels())
    if levels != expected_levels:
        raise CampaignError(
            "probe expected %d hierarchy level(s), observed %d" % (expected_levels, levels)
        )
    if allocations != 0:
        raise CampaignError("probe allocated scratch after preparation: %d" % allocations)
    if dispatches <= 0:
        raise CampaignError("probe emitted no kernel dispatches")
    if operations <= 0 or measured_steps <= 0:
        raise CampaignError("probe operation and step counts must be positive")
    # The constant is a fixed accounting allowance for boundary/reflux stages, not a cell-sized
    # fallback.  The verifier repeats this arithmetic and refuses records without the literal
    # O(operations*levels), never O(cells) contract.
    dispatch_bound = int(operations * levels * measured_steps * 4)
    if dispatches > dispatch_bound:
        raise CampaignError(
            "dispatch count exceeds O(operations*levels) bound: %d > %d"
            % (dispatches, dispatch_bound)
        )
    return {
        "status": "passed",
        "preparation": {
            "bind_complete": bool(simulation.installed_program_hash()),
            "bind_install_route": "AmrSystem.install_program" if expected_levels > 1 else "System.install_program",
            "scope": "bind+warmups",
            "warmups": int(warmups),
            "profile": (
                None if preparation_summary is None else preparation_summary.to_dict()
            ),
            "counters_before_reset": (
                {} if preparation_summary is None else dict(preparation_summary.counters())
            ),
            "reset_after_preparation": True,
        },
        "allocations_after_prepare": allocations,
        "dispatches": dispatches,
        "nodes_due": int(counters.get("nodes_due", -1)),
        "operations": int(operations),
        "levels": levels,
        "measured_steps": int(measured_steps),
        "dispatch_bound": dispatch_bound,
        "allocation_free": allocations == 0,
        "dispatch_complexity": DISPATCH_COMPLEXITY,
        "cell_count": int(cell_count),
        "fixed_operations": True,
        "fixed_levels": True,
        "cell_independent_dispatches": True,
        "profile": summary.to_dict(),
    }


def _wheel_provenance(proof_path: Path, *, native: Any, extension: Path) -> dict[str, Any]:
    proof_arg = proof_path.expanduser()
    if proof_arg.is_symlink() or not proof_arg.is_file():
        raise CampaignError("installed wheel proof is not a regular file")
    proof_path = proof_arg.resolve()
    if proof_path.is_symlink() or not proof_path.is_file():
        raise CampaignError("installed wheel proof is not a regular file")
    proof = _read_json(proof_path, where="installed wheel proof")
    if proof.get("schema_version") != WHEEL_PROOF_SCHEMA_VERSION:
        raise CampaignError("installed wheel proof has an unsupported schema version")
    if proof.get("expected_dimensions") != [2]:
        raise CampaignError("installed wheel proof must contain exactly Dim=2")
    variants = proof.get("native_variants")
    if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], dict):
        raise CampaignError("installed wheel proof must contain exactly one native variant")
    variant = variants[0]
    if set(variant) != {
        "dimension", "extension", "member", "sha256", "version", "abi_key", "has_mpi", "has_kokkos"
    }:
        raise CampaignError("installed wheel native variant has an invalid schema")
    if variant.get("dimension") != 2 or variant.get("has_mpi") is not True \
            or variant.get("has_kokkos") is not True:
        raise CampaignError("installed wheel proof does not authenticate Dim=2 MPI/Kokkos")
    wheel_arg = Path(_require_text(proof.get("wheel_path"), where="wheel proof.wheel_path")).expanduser()
    if wheel_arg.is_symlink() or not wheel_arg.is_file():
        raise CampaignError("installed wheel proof does not name a readable wheel")
    wheel_path = wheel_arg.resolve()
    if wheel_path.is_symlink() or not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise CampaignError("installed wheel proof does not name a readable wheel")
    wheel_sha = _require_text(proof.get("wheel_sha256"), where="wheel proof.wheel_sha256").lower()
    if len(wheel_sha) != 64 or _sha256(wheel_path) != wheel_sha:
        raise CampaignError("retained wheel bytes differ from installed wheel proof")
    variants = proof.get("native_variants")
    if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], dict):
        raise CampaignError("installed wheel proof must contain exactly one native variant")
    variant = variants[0]
    if variant.get("dimension") != 2 or variant.get("has_mpi") is not True \
            or variant.get("has_kokkos") is not True:
        raise CampaignError("installed wheel proof does not authenticate Dim=2 MPI/Kokkos")
    native_extension_arg = Path(
        _require_text(variant.get("extension"), where="wheel proof.native_variants[0].extension")
    ).expanduser()
    if native_extension_arg.is_symlink() or not native_extension_arg.is_file():
        raise CampaignError("wheel proof native extension is not a regular file")
    native_extension = native_extension_arg.resolve()
    extension = extension.resolve()
    if native_extension != extension:
        raise CampaignError("wheel proof extension differs from the imported _pops extension")
    extension_sha = _sha256(extension)
    if variant.get("sha256") != extension_sha:
        raise CampaignError("wheel proof native extension digest differs from imported bytes")
    wheel_abi = _require_text(variant.get("abi_key"), where="wheel proof native ABI key")
    native_abi = _require_text(getattr(native, "abi_key", lambda: "")(), where="native ABI key")
    if wheel_abi != native_abi:
        raise CampaignError("wheel proof ABI key differs from the imported extension ABI")
    package_arg = Path(
        _require_text(proof.get("package_file"), where="wheel proof.package_file")
    ).expanduser()
    if package_arg.is_symlink() or not package_arg.is_file():
        raise CampaignError("installed wheel proof package file is unavailable")
    package_file = package_arg.resolve()
    if package_file.is_symlink() or not package_file.is_file():
        raise CampaignError("installed wheel proof package file is unavailable")
    package_sha = _sha256(package_file)
    manifest_arg = Path(
        _require_text(proof.get("native_manifest"), where="wheel proof.native_manifest")
    ).expanduser()
    if manifest_arg.is_symlink() or not manifest_arg.is_file():
        raise CampaignError("installed wheel proof native manifest is unavailable")
    manifest_path = manifest_arg.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CampaignError("installed wheel proof native manifest is unavailable")
    manifest_sha = _sha256(manifest_path)
    expected_manifest_sha = _require_text(
        proof.get("native_manifest_sha256"), where="wheel proof.native_manifest_sha256"
    )
    if manifest_sha != expected_manifest_sha:
        raise CampaignError("installed wheel native manifest differs from wheel proof")
    if manifest_path != package_file.parent / "_native" / "variants.json":
        raise CampaignError("installed wheel native manifest is not under the pops package")
    if native_extension.parent != package_file.parent / "_native":
        raise CampaignError("installed wheel native extension is not under the pops package")
    if package_file.name != "__init__.py" or package_file.parent.name != "pops":
        raise CampaignError("installed wheel proof package_file must be pops/__init__.py")
    if variant.get("version") != proof.get("version"):
        raise CampaignError("installed wheel native variant version differs from package version")
    candidate_root = Path(__file__).resolve().parents[2]
    if candidate_root.name != "candidate" or not candidate_root.is_dir():
        raise CampaignError("candidate driver must execute from the archived candidate tree")
    for label, retained in (
        ("wheel proof", proof_path),
        ("wheel", wheel_path),
        ("installed package", package_file),
        ("native manifest", manifest_path),
        ("native extension", native_extension),
    ):
        try:
            retained.relative_to(candidate_root.resolve())
        except ValueError:
            continue
        raise CampaignError("%s must be retained outside the archived candidate tree" % label)
    proof_script = candidate_root / "scripts" / "prove_installed_wheel.py"
    proof_script_sha = _require_text(
        proof.get("proof_script_sha256"), where="wheel proof.proof_script_sha256"
    ).lower()
    observed_proof_script_sha = _file_provenance(
        proof_script, where="archived wheel proof script"
    )["sha256"]
    if len(proof_script_sha) != 64 or observed_proof_script_sha != proof_script_sha:
        raise CampaignError("installed wheel proof was not generated by the archived proof script")
    return {
        "proof_path": str(proof_path),
        "proof_sha256": _sha256(proof_path),
        "wheel_path": str(wheel_path),
        "wheel_sha256": wheel_sha,
        "installed_tree_sha256": _require_text(
            proof.get("installed_tree_sha256"), where="wheel proof.installed_tree_sha256"
        ),
        "package_file": str(package_file),
        "package_sha256": package_sha,
        "native_manifest": str(
            manifest_path
        ),
        "native_manifest_sha256": manifest_sha,
        "native_extension_path": str(native_extension),
        "native_extension_sha256": extension_sha,
        "dimension": 2,
        "abi_key": wheel_abi,
        "proof_script_sha256": proof_script_sha,
        "version": _require_text(proof.get("version"), where="wheel proof.version"),
    }


def _provenance(
    artifact: Any,
    facts: dict[str, Any],
    native: Any,
    *,
    toolchain: dict[str, Any],
    toolchain_receipt: dict[str, str],
    archive_receipt: dict[str, Any],
    wheel: dict[str, Any],
    harnesses: dict[str, dict[str, str]],
) -> dict[str, Any]:
    extension_arg = Path(getattr(native, "__file__", ""))
    module_arg = Path(artifact.so_path)
    if extension_arg.is_symlink() or module_arg.is_symlink():
        raise CampaignError("extension and generated MODULE paths must not be symlinks")
    extension = extension_arg.resolve()
    module = module_arg.resolve()
    if not extension.is_file() or not module.is_file():
        raise CampaignError("extension and generated MODULE paths must be regular files")
    dimension = getattr(native, "__native_dimension__", facts.get("dimension"))
    if type(dimension) is not int or dimension not in (1, 2, 3):
        raise CampaignError("native extension did not expose an authenticated dimension")
    if dimension != wheel["dimension"] or dimension != 2:
        raise CampaignError("loaded extension dimension differs from the retained Dim=2 wheel")
    artifact_identity = getattr(artifact, "artifact_identity", None)
    artifact_key = getattr(artifact_identity, "token", None)
    if not isinstance(artifact_key, str) or not artifact_key:
        artifact_key = getattr(artifact, "cache_key", None)
    if not isinstance(artifact_key, str) or not artifact_key:
        raise CampaignError("compiled MODULE did not expose an artifact key")
    abi = getattr(native, "abi_key", lambda: "")()
    if not isinstance(abi, str) or not abi:
        raise CampaignError("native extension did not expose its ABI key")
    mpi = {
        "compiled": facts.get("mpi_compiled"),
        "active": facts.get("mpi_active"),
        "ranks": facts.get("mpi_ranks"),
        "communicator": facts.get("communicator"),
    }
    if mpi["compiled"] is not True or mpi["active"] is not True or mpi["ranks"] != 4 \
            or mpi["communicator"] != "MPI_COMM_WORLD":
        raise CampaignError("extension provenance MPI facts are not exactly the 4-rank route")
    space = str(facts.get("kokkos_shared_space", ""))
    if not space or space.lower() == "unknown":
        raise CampaignError("extension provenance has no authenticated Kokkos space")
    artifact_abi = _require_text(getattr(artifact, "abi_key", None), where="artifact ABI key")
    if artifact_abi != wheel["abi_key"] or artifact_abi != abi:
        raise CampaignError("generated MODULE ABI differs from the installed wheel/extension ABI")
    artifact_key = _require_text(artifact_key, where="compiled artifact key")
    program_hash = _require_text(getattr(artifact, "program_hash", None), where="Program hash")
    artifact_cxx = Path(
        _require_text(getattr(artifact, "cxx", None), where="artifact compiler")
    ).expanduser().resolve()
    if artifact_cxx != Path(toolchain["nvcc_wrapper"]["path"]).resolve():
        raise CampaignError("generated MODULE compiler differs from the authenticated NVCC wrapper")
    artifact_std = _require_text(getattr(artifact, "std", None), where="artifact C++ standard")
    if artifact_std != "c++20":
        raise CampaignError("ADC-700 generated MODULE must use C++20 exactly")
    baked_std = getattr(native, "__cxx_std__", None)
    if type(baked_std) is not int or baked_std != 20:
        raise CampaignError("loaded extension has no authenticated C++20 standard")
    if str(baked_std) != str(artifact_std).removeprefix("c++"):
        raise CampaignError("generated MODULE C++ standard differs from the loaded extension")
    requested = toolchain.get("requested")
    if not isinstance(requested, dict):
        raise CampaignError("toolchain provenance has no requested compile options")
    if requested.get("cxx") != str(artifact_cxx) or requested.get("std") != artifact_std:
        raise CampaignError("requested compile options differ from the generated MODULE")
    if toolchain_receipt.get("revision") != archive_receipt.get("revision"):
        raise CampaignError("toolchain and archive receipts refer to different revisions")
    if not isinstance(harnesses, dict) or set(harnesses) != set(CAMPAIGN_HARNESSES):
        raise CampaignError("campaign harness provenance is incomplete")
    return {
        "extension_path": str(extension),
        "extension_sha256": _sha256(extension),
        "generated_module_path": str(module),
        "generated_module_sha256": _sha256(module),
        "dimension": dimension,
        "mpi": mpi,
        "mpi_compiled": mpi["compiled"],
        "mpi_active": mpi["active"],
        "mpi_ranks": mpi["ranks"],
        "kokkos_space": space,
        "kokkos_backend": facts.get("kokkos_backend"),
        "kokkos_device": facts.get("kokkos_device"),
        "abi_key": abi,
        "artifact_abi_key": artifact_abi,
        "artifact_key": artifact_key,
        "program_hash": program_hash,
        "toolchain": toolchain,
        "toolchain_receipt": toolchain_receipt,
        "archive_receipt": archive_receipt,
        "wheel": wheel,
        "harnesses": harnesses,
    }


def _box_token(box: Any) -> str:
    level = getattr(box, "level", 0)
    raw = getattr(box, "box", box)
    lo = getattr(raw, "lo", None)
    hi = getattr(raw, "hi", None)
    # The public AMR seam returns the compact tuple ``(level, lower_tuple, upper_tuple)``.
    # Canonicalise it to the pinned C++ oracle's ``level:lo...,hi...`` spelling instead of falling
    # back to ``repr`` (which is neither stable across Python versions nor comparable to C++).
    if isinstance(raw, (tuple, list)) and len(raw) == 3:
        tuple_level, tuple_lo, tuple_hi = raw
        if type(tuple_level) is not int or not isinstance(tuple_lo, (tuple, list)) \
                or not isinstance(tuple_hi, (tuple, list)):
            raise CampaignError("AMR patch box tuple has an invalid level/corner schema")
        level = tuple_level
        lo, hi = tuple_lo, tuple_hi
    if lo is None and isinstance(raw, (tuple, list)) and len(raw) == 2:
        lo, hi = raw
    if lo is None or hi is None:
        raise CampaignError("AMR patch box has no authenticated lower/upper corners")
    lo = tuple(lo)
    hi = tuple(hi)
    if not lo or len(lo) != len(hi):
        raise CampaignError("AMR patch box lower/upper corners have inconsistent dimensions")
    if any(type(value) is not int for value in (*lo, *hi)):
        raise CampaignError("AMR patch box corners must contain exact integers")
    return "%s:%s" % (int(level), ",".join(map(str, (*lo, *hi))))


def _topology(simulation: Any, *, n: int) -> dict[str, Any]:
    # ``System`` has one implicit uniform level and intentionally has no AMR patch-box query;
    # the native oracle's level-zero topology is therefore represented by the empty fine-patch
    # list.  Adaptive executors expose their authenticated patch boxes directly.
    try:
        raw_boxes = simulation.patch_boxes()
    except (AttributeError, NotImplementedError):
        raw_boxes = ()
    boxes = sorted(_box_token(box) for box in raw_boxes)
    executor = getattr(simulation, "_executor", None)
    local_provider = getattr(executor, "coarse_local_boxes", None)
    total_provider = getattr(executor, "coarse_total_boxes", None)
    if not callable(local_provider) or not callable(total_provider):
        raise CampaignError(
            "main AMR workload did not expose authenticated coarse box distribution"
        )
    coarse_local = int(local_provider())
    coarse_total = int(total_provider())
    if coarse_local <= 0 or coarse_total <= 0 or coarse_local > coarse_total:
        raise CampaignError("main AMR workload returned an invalid coarse box distribution")
    return {
        "levels": int(simulation.n_levels()),
        "patches": len(boxes),
        "boxes": ";".join(boxes),
        "distribute_coarse": True,
        "coarse_max_grid": int(n // 2),
        "coarse_local_boxes": coarse_local,
        "coarse_total_boxes": coarse_total,
    }


def _measurement(
    simulation: Any,
    artifact: Any,
    *,
    facts: dict[str, Any],
    native: Any,
    config: argparse.Namespace,
    provenance: dict[str, Any],
    probes: dict[str, Any],
    gpu_assignments: list[dict[str, Any]],
    gpu_local: dict[str, Any],
) -> dict[str, Any]:
    state = np.asarray(
        simulation.block_level_state_global("gas", 0), dtype=np.float64
    ).reshape((4, config.n, config.n))
    initial = _euler_initial(config.n)
    initial_mass = float(initial[0].sum() / (config.n * config.n))
    final_mass = _finite(simulation.integral("gas", 0), where="final mass")
    checksum = _finite(state.sum(), where="checksum")
    checksum_square = _finite(np.square(state).sum(), where="checksum square")
    maximum = _finite(np.max(np.abs(state)), where="maximum")
    mass_error = abs(final_mass - initial_mass)
    mass_tolerance = 1.0e-9 * max(1.0, abs(initial_mass))
    if not np.isfinite(state).all() or maximum <= 0.0 or mass_error > mass_tolerance:
        raise CampaignError("candidate numerical validation failed")
    return {
        "schema": SCHEMA,
        "route": "program_only",
        "revision": config.revision,
        # The native oracle emits ``Kokkos::DefaultExecutionSpace::name()``.  Use the
        # corresponding runtime-report backend (for example ``Cuda``), not the lower-case
        # device identity (``cuda``), so compare.py can authenticate one execution-space string
        # across both routes.
        "execution_space": str(facts.get("kokkos_backend") or facts.get("kokkos_device")),
        "mpi_ranks": int(facts["mpi_ranks"]),
        "mpi_communicator": str(facts.get("communicator", "")),
        "toolchain_build_attested": True,
        "execution_concurrency": int(facts.get("kokkos_concurrency", 0)),
        "real_bytes": int(facts.get("real_bytes", 0)),
        "parameters": {
            "n": config.n,
            "warmups": config.warmups,
            "measured_steps": config.steps,
            "dt": config.dt,
        },
        "topology": _topology(simulation, n=config.n),
        "toolchain": provenance["toolchain"],
        "toolchain_receipt": provenance["toolchain_receipt"],
        "gpu": gpu_local,
        "gpu_uuid": gpu_local["uuid"],
        "gpu_assignments": gpu_assignments,
        "timing": {
            "seconds": config.seconds,
            "per_step_seconds": config.seconds / float(config.steps),
            "rank_aggregation": "max",
            "device_fence": "runtime_program_boundary",
            "mpi_barrier": "before_and_after",
        },
        "signature": {
            "mass": final_mass,
            "initial_mass": initial_mass,
            "mass_error": mass_error,
            "checksum": checksum,
            "checksum_square": checksum_square,
            "maximum": maximum,
        },
        "validation": {"passed": True, "mass_tolerance": mass_tolerance},
        "extension": provenance,
        "probes": probes,
    }


def _probe_campaign(
    *,
    include: Path,
    cxx: str,
    dt: float,
    warmups: int,
    measured_steps: int,
    adaptive: bool,
    label: str,
) -> dict[str, Any]:
    """Run one probe at two resolutions while holding Program IR and levels fixed."""
    resolution_rows: list[dict[str, Any]] = []
    for resolution in PROBE_RESOLUTIONS:
        probe_case, probe_layout, probe_instance, values, probe_ops = _scalar_case(
            resolution, dt, adaptive=adaptive, name="adc700-%s-%d" % (label, resolution)
        )
        probe_artifact, probe_simulation = _compile_and_bind(
            probe_case,
            probe_layout,
            probe_instance,
            values,
            include=include,
            cxx=cxx,
        )
        _assert_native_executor(probe_simulation, adaptive=adaptive)
        expected_levels = 2 if adaptive else 1
        if adaptive and probe_simulation.n_levels() < expected_levels:
            raise CampaignError("AMR probe did not materialize a refined level")
        probe = _profile_probe(
            probe_simulation,
            operations=probe_ops,
            expected_levels=expected_levels,
            dt=dt,
            warmups=max(warmups, 1),
            measured_steps=max(1, min(measured_steps, 4)),
            cell_count=resolution * resolution,
        )
        if probe_simulation.installed_program_hash() != probe_artifact.program_hash:
            raise CampaignError("probe MODULE was not installed by System/AmrSystem")
        probe["resolution"] = int(resolution)
        resolution_rows.append(probe)
    if len({row["operations"] for row in resolution_rows}) != 1:
        raise CampaignError("probe Program operation count changed with resolution")
    if len({row["levels"] for row in resolution_rows}) != 1:
        raise CampaignError("probe hierarchy level count changed with resolution")
    if len({row["dispatches"] for row in resolution_rows}) != 1:
        raise CampaignError(
            "probe dispatch count depends on cells; O(operations*levels), never O(cells), is unproven"
        )
    result = dict(resolution_rows[0])
    result.update({
        "resolutions": resolution_rows,
        "resolution_count": len(resolution_rows),
        "cell_counts": [row["cell_count"] for row in resolution_rows],
        "dispatches_by_resolution": [row["dispatches"] for row in resolution_rows],
        "fixed_operations": True,
        "fixed_levels": True,
        "cell_independent_dispatches": True,
    })
    return result


def run_campaign(config: argparse.Namespace) -> dict[str, Any] | None:
    if config.route != "program_only":
        raise CampaignError("the Python driver is the candidate program_only route only")
    if config.revision == BASELINE_REVISION:
        raise CampaignError("candidate revision must not equal the pinned baseline revision")
    if config.n < 16 or config.n % 4:
        raise CampaignError("--n must be >=16 and divisible by four")
    if config.warmups <= 0 or config.steps <= 0 or config.dt <= 0.0:
        raise CampaignError("warmups, steps and dt must be positive")
    facts, native = _require_runtime()
    _require_baseline_toolchain_attestation()
    archive_receipt = _archive_provenance(
        Path(config.archive_receipt),
        revision=config.revision,
        helper_path=Path(config.archive_helper),
    )
    include_arg = Path(config.include).expanduser()
    if include_arg.is_symlink() or not include_arg.is_absolute():
        raise CampaignError("candidate include root must be an absolute non-symlink path")
    include = include_arg.resolve()
    cxx, toolchain = _require_toolchain(config.cxx, include, native)
    if not config.toolchain_probe:
        raise CampaignError("--toolchain-probe is mandatory for candidate runs")
    toolchain = _merge_cmake_target_contract(
        Path(config.toolchain_probe), toolchain=toolchain
    )
    toolchain_receipt = _toolchain_receipt(
        Path(config.toolchain_receipt), toolchain=toolchain, revision=config.revision
    )
    wheel_proof_path = Path(config.wheel_proof).expanduser().resolve()
    harnesses = _campaign_harness_provenance()
    case, layout, instance, program, initial, operations = _euler_case(config.n, config.dt)
    artifact, simulation = _compile_and_bind(
        case, layout, instance, initial, include=include, cxx=cxx
    )
    _assert_native_executor(simulation, adaptive=True)
    # Binding initializes the selected Kokkos execution space on builds that defer that lifecycle
    # edge.  Refresh the facts after binding so execution_concurrency and backend strings describe
    # the same initialized runtime as the native oracle, instead of a pre-bind zero/unknown report.
    facts, native = _require_runtime()
    extension_path = Path(getattr(native, "__file__", "")).resolve()
    wheel = _wheel_provenance(wheel_proof_path, native=native, extension=extension_path)
    provenance = _provenance(
        artifact,
        facts,
        native,
        toolchain=toolchain,
        toolchain_receipt=toolchain_receipt,
        archive_receipt=archive_receipt,
        wheel=wheel,
        harnesses=harnesses,
    )
    # Two independent scalar probes make the memory/dispatch claim explicit for both layout
    # authorities.  AMR is frozen after bind: any regrid or scratch allocation during the measured
    # window is therefore evidence of an invalid preparation boundary and fails closed.
    probes: dict[str, Any] = {}
    probes["uniform"] = _probe_campaign(
        include=include,
        cxx=cxx,
        dt=config.dt,
        warmups=config.warmups,
        measured_steps=config.steps,
        adaptive=False,
        label="uniform",
    )
    probes["amr_refined_planned"] = _probe_campaign(
        include=include,
        cxx=cxx,
        dt=config.dt,
        warmups=config.warmups,
        measured_steps=config.steps,
        adaptive=True,
        label="amr_refined_planned",
    )
    facts, native = _require_runtime()
    if config.warmups > 0:
        pops.run(
            simulation,
            t_end=float((config.warmups + 1) * config.dt),
            max_steps=config.warmups,
            console=False,
        )
    _mpi_barrier(native)
    begin = time.perf_counter()
    pops.run(
        simulation,
        t_end=float(simulation.time() + config.steps * config.dt),
        max_steps=config.steps,
        console=False,
    )
    _mpi_barrier(native)
    config.seconds = _collective_max(native, time.perf_counter() - begin)
    gpu_assignments, gpu_local = _gpu_assignments(native)
    if int(native.my_rank()) != 0:
        return None
    return _measurement(
        simulation,
        artifact,
        facts=facts,
        native=native,
        config=config,
        provenance=provenance,
        probes=probes,
        gpu_assignments=gpu_assignments,
        gpu_local=gpu_local,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="program_only")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--include", default=os.environ.get("POPS_ADC700_INCLUDE_ROOT", ""))
    parser.add_argument("--cxx", default=os.environ.get("POPS_ADC700_NVCC_WRAPPER", os.environ.get("CXX", "")))
    parser.add_argument("--wheel-proof", default=os.environ.get("POPS_ADC700_WHEEL_PROOF", ""))
    parser.add_argument("--archive-receipt", default=os.environ.get("POPS_ADC700_ARCHIVE_RECEIPT", ""))
    parser.add_argument("--archive-helper", default=os.environ.get("POPS_ADC700_ARCHIVE_HELPER", ""))
    parser.add_argument("--toolchain-receipt", default=os.environ.get("POPS_ADC700_TOOLCHAIN_RECEIPT", ""))
    parser.add_argument("--toolchain-probe", default=os.environ.get("POPS_ADC700_TOOLCHAIN_PROBE", ""))
    parser.add_argument("--toolchain-only", action="store_true")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    config = _parser().parse_args(argv)
    try:
        if not config.include or not config.cxx or not config.wheel_proof:
            raise CampaignError(
                "--include, --cxx and --wheel-proof are mandatory for the candidate MODULE"
            )
        if config.toolchain_only:
            if not config.toolchain_receipt:
                raise CampaignError("--toolchain-receipt is mandatory for toolchain-only mode")
            return _emit_toolchain_receipt(config)
        if not config.archive_receipt or not config.archive_helper or not config.toolchain_receipt \
                or not config.toolchain_probe:
            raise CampaignError(
                "--archive-receipt, --archive-helper, --toolchain-probe and --toolchain-receipt "
                "are mandatory for candidate runs"
            )
        row = run_campaign(config)
        if row is not None:
            # Keep ``schema`` first so the archived batch harness can select the one rank-zero
            # record without trusting incidental stdout from MPI/Python dependencies.
            print(json.dumps(row, separators=(",", ":"), allow_nan=False))
        return 0
    except (CampaignError, ValueError, TypeError, RuntimeError) as error:
        rank = 0
        try:
            rank = int(getattr(getattr(pops, "_pops", None), "my_rank", lambda: 0)())
        except Exception:  # noqa: BLE001 - preserve the primary campaign failure
            pass
        print("ADC-700 program campaign failed on rank %d: %s" % (rank, error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
