"""Strict content-addressed checkpoint envelope shared by Uniform and AMR runtimes."""
from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from typing import Any

from pops.identity import Identity, canonical_bytes, make_identity
from pops._manifest_protocol import strict_json_loads
from pops._generated_release_contract import (
    CHECKPOINT_ENVELOPE_SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
)
MANIFEST_KEY = "pops_checkpoint_manifest"
IDENTITY_KEY = "pops_restart_identity"


def _identity_json(value: Identity) -> dict[str, Any]:
    return {
        "domain": value.domain, "schema_version": value.schema_version,
        "algorithm": value.algorithm, "hexdigest": value.hexdigest,
    }


def _identity_from_json(value: Any) -> Identity:
    required = {"domain", "schema_version", "algorithm", "hexdigest"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise TypeError("checkpoint identity must contain exactly %s" % sorted(required))
    digest = value["hexdigest"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("checkpoint identity hexdigest must be 64 lowercase hexadecimal characters")
    try:
        raw = bytes.fromhex(digest)
    except ValueError:
        raise ValueError("checkpoint identity hexdigest is not hexadecimal") from None
    return Identity(value["domain"], value["schema_version"], value["algorithm"], raw)


def _array_evidence(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("checkpoint payload cannot contain object dtype")
    header = canonical_bytes({
        "protocol": "pops.array-evidence.v1",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    })
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": digest.hexdigest(),
    }


def _runtime_identities(owner: Any) -> tuple[Identity, Identity, Identity]:
    provider = getattr(owner, "_checkpoint_identities", None)
    if not callable(provider):
        raise TypeError(
            "checkpoint owner must implement the private exact-identity provider protocol")
    supplied = provider()
    if type(supplied) is not tuple or len(supplied) != 3:
        raise TypeError("checkpoint identity provider must return an exact three-value tuple")
    values = tuple(zip(supplied, ("semantic", "artifact", "bind"), strict=True))
    checked = []
    for value, domain in values:
        if type(value) is not Identity or value.domain != domain:
            raise RuntimeError("checkpoint requires the runtime's domain-%r identity" % domain)
        checked.append(Identity.from_data(value.to_data()))
    return tuple(checked)  # type: ignore[return-value]


def seal_checkpoint_payload(owner: Any, payload: dict[str, Any], *, runtime_kind: str) -> Identity:
    """Add the canonical manifest and restart token to an in-memory NPZ payload."""
    if MANIFEST_KEY in payload or IDENTITY_KEY in payload:
        raise ValueError("checkpoint payload already contains reserved identity keys")
    semantic, artifact, bind = _runtime_identities(owner)
    run = getattr(owner, "last_run_identity", None)
    if type(run) is not Identity or run.domain != "run":
        raise RuntimeError(
            "checkpoint requires a prior pops.run(sim, **controls) so its execution controls "
            "have a run identity")
    arrays = {name: _array_evidence(value) for name, value in sorted(payload.items())}
    base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "runtime_kind": runtime_kind,
        "semantic_identity": _identity_json(semantic),
        "artifact_identity": _identity_json(artifact),
        "bind_identity": _identity_json(bind),
        "run_identity": _identity_json(run),
        "clock": {
            "time": float(payload["t"]).hex(),
            "macro_step": int(payload["macro_step"]),
        },
        "arrays": arrays,
    }
    restart = make_identity("restart", base)
    manifest = dict(base, restart_identity=_identity_json(restart))
    payload[MANIFEST_KEY] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload[IDENTITY_KEY] = restart.token
    return restart


def _strict_json(text: Any) -> dict[str, Any]:
    result = strict_json_loads(str(text), where="checkpoint manifest JSON")
    if not isinstance(result, dict):
        raise TypeError("checkpoint manifest must decode to a mapping")
    return result


def authenticate_checkpoint_payload(owner: Any, payload: Any, *, runtime_kind: str) -> Identity:
    """Authenticate every checkpoint byte and all runtime identities before state mutation."""
    files = set(getattr(payload, "files", ()))
    if MANIFEST_KEY not in files or IDENTITY_KEY not in files:
        raise ValueError("checkpoint has no canonical manifest/restart identity; historical formats are refused")
    manifest = _strict_json(payload[MANIFEST_KEY])
    expected_keys = {
        "schema_version", "runtime_kind", "semantic_identity", "artifact_identity",
        "bind_identity", "run_identity", "clock", "arrays", "restart_identity",
    }
    if set(manifest) != expected_keys:
        raise ValueError("checkpoint manifest keys must be exactly %s" % sorted(expected_keys))
    version = manifest["schema_version"]
    if (isinstance(version, bool) or not isinstance(version, int)
            or version != CHECKPOINT_SCHEMA_VERSION):
        raise ValueError("unsupported checkpoint manifest schema_version %r" % manifest["schema_version"])
    if manifest["runtime_kind"] != runtime_kind:
        raise ValueError("checkpoint runtime kind %r cannot restart %r" % (
            manifest["runtime_kind"], runtime_kind))
    if not isinstance(manifest["arrays"], Mapping):
        raise TypeError("checkpoint manifest arrays must be a mapping")
    expected_files = set(manifest["arrays"]) | {MANIFEST_KEY, IDENTITY_KEY}
    if files != expected_files:
        raise ValueError("checkpoint NPZ keys differ from its exact manifest")

    semantic, artifact, bind = _runtime_identities(owner)
    for field, current, domain in (
        ("semantic_identity", semantic, "semantic"),
        ("artifact_identity", artifact, "artifact"),
        ("bind_identity", bind, "bind"),
    ):
        recorded = _identity_from_json(manifest[field])
        if recorded.domain != domain or recorded.token != current.token:
            raise ValueError("checkpoint %s does not match the bound runtime" % field)
    run = _identity_from_json(manifest["run_identity"])
    if run.domain != "run":
        raise ValueError("checkpoint run_identity has wrong domain")
    for name, evidence in manifest["arrays"].items():
        if evidence != _array_evidence(payload[name]):
            raise ValueError("checkpoint payload digest mismatch for %r" % name)
    from pops.runtime._engine_descriptors import abi_key
    if "abi_key" not in files or str(payload["abi_key"]) != str(abi_key()):
        raise ValueError("checkpoint ABI identity does not match the loaded runtime")

    base = {key: manifest[key] for key in expected_keys - {"restart_identity"}}
    restart = _identity_from_json(manifest["restart_identity"])
    expected = make_identity("restart", base)
    if restart.domain != "restart" or restart.token != expected.token:
        raise ValueError("checkpoint restart identity does not match its canonical manifest")
    if str(payload[IDENTITY_KEY]) != restart.token:
        raise ValueError("checkpoint restart identity token does not match its manifest")
    clock = manifest["clock"]
    if set(clock) != {"time", "macro_step"} \
            or float(payload["t"]) != float.fromhex(clock["time"]) \
            or int(payload["macro_step"]) != int(clock["macro_step"]):
        raise ValueError("checkpoint clock does not match its canonical manifest")
    return restart


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "IDENTITY_KEY", "MANIFEST_KEY",
    "authenticate_checkpoint_payload", "seal_checkpoint_payload",
]
