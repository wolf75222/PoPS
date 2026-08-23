"""Reusable fail-closed READY/GO gate for the public lifecycle profile target."""

from __future__ import annotations

import os
import time
from pathlib import Path

from profile_contract import PROFILE_SCHEMA, ProfileContractError, read_json, write_json_new


def ready_after_bind_warmup(*, ready: Path, nonce: str, provenance: dict[str, object]) -> None:
    """Publish readiness only once compilation, bind and warmup are finished."""
    required = {
        "campaign",
        "source",
        "python_package",
        "artifact",
        "program_artifact",
        "native",
        "build",
        "runtime",
        "host",
        "command",
    }
    if type(provenance) is not dict or set(provenance) != required:
        raise ProfileContractError(
            "READY provenance must authenticate campaign/source/python/artifact/native/build/runtime/host/command"
        )
    if any(type(provenance[name]) is not dict for name in required):
        raise ProfileContractError("READY provenance sections must be objects")
    write_json_new(
        ready,
        {
            "schema": PROFILE_SCHEMA,
            "phase": "ready_after_bind_warmup",
            "nonce": nonce,
            "pid": os.getpid(),
            "ready_unix_seconds": time.time(),
            "provenance": provenance,
        },
    )


def await_go(*, go: Path, nonce: str, timeout_seconds: float = 60.0) -> None:
    """Block without numerical work until the recorder proves its matching GO."""
    deadline = time.monotonic() + timeout_seconds
    while not go.exists():
        if time.monotonic() >= deadline:
            raise ProfileContractError("timed out waiting for matching profiling GO")
        time.sleep(0.02)
    receipt = read_json(go, "GO receipt")
    if receipt.get("schema") != PROFILE_SCHEMA or receipt.get("phase") != "go":
        raise ProfileContractError("profiling GO has an unsupported schema")
    if receipt.get("nonce") != nonce:
        raise ProfileContractError("profiling GO nonce differs from READY")


def completed_public_lifecycle(*, receipt: Path, nonce: str, returncode: int) -> None:
    """Publish the full-process result after its one acquired lifecycle ends."""
    write_json_new(
        receipt,
        {
            "schema": PROFILE_SCHEMA,
            "phase": "completed_public_lifecycle",
            "nonce": nonce,
            "pid": os.getpid(),
            "returncode": returncode,
            "completed_unix_seconds": time.time(),
        },
    )
