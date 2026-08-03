#!/usr/bin/env python3
"""Fail-closed preflight for installed-runtime ADC-757 hardware evidence.

The current benchmark vector kernels are not a PoPS runtime scenario.  This probe authenticates the
retained wheel and its live installation, runs ``pops.runtime.doctor.doctor()``, records the module
ABI, and then refuses closure until one exact four-mode runtime matrix plus native local-time and
AMR-migration receipts exists.  It never turns header presence into positive runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any


REFUSAL_SCHEMA = "pops.adc757.installed-runtime-refusal.v1"
RUNTIME_SCHEMA = "pops.adc757.installed-runtime-matrix.v1"
RUNTIME_MODES = ("serial", "threaded", "gpu", "gpu_mpi")


class RuntimeProbeError(RuntimeError):
    """The installed candidate cannot even support an authenticated refusal."""


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeProbeError(f"{where} must be an object")
    return value


def _outside(path: Path, root: Path, *, where: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise RuntimeProbeError(f"{where} resolved inside the source checkout: {resolved}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path, where: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), where)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeProbeError(f"cannot load {where}: {error}") from error


def _required_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeProbeError(f"{where} must be non-empty text")
    return value


def _installed_candidate(
    proof: dict[str, Any], *, source_root: Path, expected_revision: str
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    required = {
        "schema_version",
        "python_executable",
        "distribution_root",
        "package_file",
        "native_extension",
        "native_member",
        "native_sha256",
        "installed_member_count",
        "installed_tree_sha256",
        "proof_script_sha256",
        "version",
        "wheel_path",
        "wheel_sha256",
    }
    if set(proof) != required or proof["schema_version"] != 2:
        raise RuntimeProbeError("installed wheel proof does not have the exact v2 contract")
    package_file = _outside(
        Path(_required_text(proof["package_file"], "wheel proof package_file")),
        source_root,
        where="installed package",
    )
    native_extension = _outside(
        Path(_required_text(proof["native_extension"], "wheel proof native_extension")),
        source_root,
        where="installed native extension",
    )
    python_executable = _outside(
        Path(_required_text(proof["python_executable"], "wheel proof python_executable")),
        source_root,
        where="installed Python",
    )

    import pops
    from pops import _pops
    from pops.codegen import abi as pops_abi
    from pops.codegen import toolchain
    from pops.runtime.doctor import doctor

    if Path(pops.__file__).resolve() != package_file:
        raise RuntimeProbeError("the imported pops package differs from the retained wheel proof")
    if Path(_pops.__file__).resolve() != native_extension:
        raise RuntimeProbeError("the imported pops extension differs from the retained wheel proof")
    if Path(sys.executable).resolve() != python_executable:
        raise RuntimeProbeError("the live Python differs from the retained wheel proof")

    raw_checks = _object(doctor(verbose=False), "pops doctor result")
    normalized_checks: dict[str, dict[str, Any]] = {}
    for name, raw in sorted(raw_checks.items()):
        if not isinstance(raw, tuple) or len(raw) != 2 or not isinstance(raw[0], bool):
            raise RuntimeProbeError(f"pops doctor check {name!r} is malformed")
        normalized_checks[name] = {"passed": raw[0], "detail": repr(raw[1])}
    doctor_result = {
        "passed": bool(normalized_checks)
        and all(item["passed"] for item in normalized_checks.values()),
        "checks_sha256": _sha256_json(normalized_checks),
    }
    module_abi_key = _required_text(_pops.abi_key(), "installed module ABI key")
    include_root = _outside(
        Path(toolchain.pops_include()), source_root, where="installed PoPS include root"
    )
    baked_signature = pops_abi.module_header_signature()
    if not isinstance(baked_signature, str) or not baked_signature:
        raise RuntimeProbeError("installed module has no baked header signature")
    if toolchain.pops_header_signature(include_root) != baked_signature:
        raise RuntimeProbeError("installed headers differ from the module ABI signature")

    def header_support(relative: str, needles: tuple[str, ...]) -> bool:
        path = include_root / relative
        if not path.is_file():
            return False
        source = path.read_text(encoding="utf-8")
        return all(needle in source for needle in needles)

    detected_support = {
        "cell_local_time_commit_receipt_primitives": header_support(
            "pops/runtime/program/cell_temporal_partition_executor.hpp",
            ("PreparedBatchedCellTemporalExecutor", "prepare_commit_attempt"),
        ),
        "amr_rebalance_migration_primitives": header_support(
            "pops/runtime/amr/amr_runtime.hpp",
            ("decide_rebalance", "apply_rebalance_decision"),
        ),
    }
    installation = {
        "revision": expected_revision,
        "version": _required_text(proof["version"], "wheel proof version"),
        "wheel_name": Path(_required_text(proof["wheel_path"], "wheel proof path")).name,
        "wheel_sha256": _required_text(proof["wheel_sha256"], "wheel proof wheel_sha256"),
        "installed_tree_sha256": _required_text(
            proof["installed_tree_sha256"], "wheel proof installed_tree_sha256"
        ),
        "native_sha256": _required_text(proof["native_sha256"], "wheel proof native_sha256"),
        "package_file": str(package_file),
        "native_extension": str(native_extension),
        "python_executable": str(python_executable),
        "outside_source_checkout": True,
    }
    return installation, module_abi_key, {"doctor": doctor_result, "support": detected_support}


def refusal_payload(
    *,
    revision: str,
    installation: dict[str, Any],
    module_abi_key: str,
    doctor: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    if doctor["passed"] is not True:
        blockers.append(
            {
                "code": "pops_doctor_failed",
                "detail": "the exact installed candidate did not pass every pops.doctor check",
            }
        )
    if not support["cell_local_time_commit_receipt_primitives"]:
        blockers.append(
            {
                "code": "adc757g_local_time_runtime_unavailable",
                "detail": (
                    "the installed candidate lacks the accepted local-time publication primitives "
                    "required for a native runtime receipt"
                ),
            }
        )
    if not support["amr_rebalance_migration_primitives"]:
        blockers.append(
            {
                "code": "adc757c_amr_migration_runtime_unavailable",
                "detail": (
                    "the installed candidate lacks decide_rebalance/apply_rebalance_decision and "
                    "cannot prove real accepted-boundary ownership migration"
                ),
            }
        )
    blockers.append(
        {
            "code": "installed_runtime_matrix_receipts_unavailable",
            "detail": (
                "no authenticated serial/threaded/GPU/GPU+MPI same-scenario matrix with artifact, "
                "ABI, solution and C/G authority receipts was supplied; vector kernels are not a "
                "PoPS runtime proof"
            ),
        }
    )
    return {
        "schema": REFUSAL_SCHEMA,
        "status": "refused",
        "revision": revision,
        "installation": installation,
        "doctor": doctor,
        "module_abi_key": module_abi_key,
        "detected_support": support,
        "blockers": blockers,
    }


def _accept_external_matrix(
    raw: Any,
    *,
    revision: str,
    installation: dict[str, Any],
    module_abi_key: str,
    doctor: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, Any]:
    matrix = _object(raw, "installed runtime matrix")
    expected = {"schema", "status", "revision", "scenario_id", "modes", "authorities"}
    if set(matrix) != expected or matrix.get("schema") != RUNTIME_SCHEMA:
        raise RuntimeProbeError("installed runtime matrix has an unexpected contract")
    if matrix.get("status") != "passed" or matrix.get("revision") != revision:
        raise RuntimeProbeError("installed runtime matrix did not pass for the candidate revision")
    modes = matrix.get("modes")
    if (
        not isinstance(modes, list)
        or not all(isinstance(item, dict) for item in modes)
        or [item.get("id") for item in modes] != list(RUNTIME_MODES)
    ):
        raise RuntimeProbeError(f"installed runtime matrix requires ordered modes {RUNTIME_MODES}")
    if doctor.get("passed") is not True:
        raise RuntimeProbeError("the live exact wheel did not pass pops.doctor")
    unavailable = [name for name, available in support.items() if available is not True]
    if unavailable:
        raise RuntimeProbeError(
            "the live exact wheel lacks required C/G runtime primitives: " + ", ".join(unavailable)
        )
    verifier = runpy.run_path(str(Path(__file__).with_name("verify.py")))
    evidence_error = verifier["EvidenceError"]
    try:
        verifier["validate_installed_runtime"](matrix, expected_revision=revision)
    except evidence_error as error:
        raise RuntimeProbeError(f"installed runtime matrix was refused: {error}") from error

    gpu_mpi = _object(modes[-1], "installed runtime gpu_mpi mode")
    live_installation = _object(gpu_mpi.get("installation"), "gpu_mpi installation")
    expected_installation = {
        "wheel_name": installation["wheel_name"],
        "wheel_sha256": installation["wheel_sha256"],
        "installed_tree_sha256": installation["installed_tree_sha256"],
        "native_sha256": installation["native_sha256"],
        "package_file": installation["package_file"],
        "native_extension": installation["native_extension"],
        "python_executable": installation["python_executable"],
        "outside_source_checkout": True,
    }
    if live_installation != expected_installation:
        raise RuntimeProbeError("gpu_mpi runtime evidence belongs to another installed wheel")
    if gpu_mpi.get("doctor") != doctor:
        raise RuntimeProbeError("gpu_mpi runtime evidence belongs to another pops.doctor result")
    artifact = _object(gpu_mpi.get("artifact"), "gpu_mpi artifact")
    if artifact.get("module_abi_key") != module_abi_key:
        raise RuntimeProbeError("gpu_mpi runtime evidence belongs to another module ABI")
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-proof", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        proof = _load_json(args.wheel_proof, "installed wheel proof")
        installation, module_abi_key, audit = _installed_candidate(
            proof,
            source_root=args.source_root,
            expected_revision=args.expected_revision,
        )
        refusal = refusal_payload(
            revision=args.expected_revision,
            installation=installation,
            module_abi_key=module_abi_key,
            doctor=audit["doctor"],
            support=audit["support"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.runtime_evidence is not None:
            accepted = _accept_external_matrix(
                _load_json(args.runtime_evidence, "installed runtime matrix"),
                revision=args.expected_revision,
                installation=installation,
                module_abi_key=module_abi_key,
                doctor=audit["doctor"],
                support=audit["support"],
            )
            args.output.write_text(
                json.dumps(accepted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0
        args.output.write_text(
            json.dumps(refusal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (RuntimeProbeError, OSError, ValueError) as error:
        print(f"ADC-757 installed runtime preflight failed: {error}", file=sys.stderr)
        return 3
    for blocker in refusal["blockers"]:
        print(
            f"ADC-757 runtime evidence refused [{blocker['code']}]: {blocker['detail']}",
            file=sys.stderr,
        )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
