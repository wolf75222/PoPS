"""Strict content-addressed checkpoint envelope shared by Uniform and AMR runtimes."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from pops.identity import Identity, canonical_bytes, make_identity
from pops._manifest_protocol import strict_json_loads
from pops._generated_release_contract import (
    CHECKPOINT_ENVELOPE_SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
)
MANIFEST_KEY = "pops_checkpoint_manifest"
IDENTITY_KEY = "pops_restart_identity"


def _payload_files(payload: Any) -> set[str]:
    stored_files = getattr(payload, "files", None)
    if stored_files is None:
        keys = getattr(payload, "keys", None)
        if not callable(keys):
            raise TypeError("checkpoint payload exposes neither files nor keys()")
        stored_files = keys()
    elif callable(stored_files):
        stored_files = stored_files()
    if isinstance(stored_files, (str, bytes)) or not isinstance(stored_files, Iterable):
        raise TypeError("checkpoint payload files/keys() must return an iterable of names")
    files = set(stored_files)
    if any(not isinstance(name, str) for name in files):
        raise TypeError("checkpoint payload file names must be text")
    return files


def require_exact_payload_version(
    payload: Any,
    *,
    key: str,
    expected: int,
    runtime_kind: str,
) -> int:
    """Require one current integer payload version without coercing legacy values.

    The canonical envelope version authenticates the outer container. Uniform and AMR also carry
    a codec-specific payload version, which must remain an exact scalar integer: strings, floats,
    booleans, arrays, and missing values are migration inputs, never runtime compatibility routes.
    """
    if not isinstance(key, str) or not key:
        raise TypeError("checkpoint payload version key must be non-empty text")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise TypeError("expected checkpoint payload version must be a positive integer")
    if not isinstance(runtime_kind, str) or not runtime_kind:
        raise TypeError("checkpoint runtime kind must be non-empty text")

    files = _payload_files(payload)
    if key not in files:
        raise ValueError(
            "restart: strict %s checkpoint is missing payload version %r; "
            "historical checkpoints require offline migration" % (runtime_kind, key)
        )

    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "iu":
        raise TypeError(
            "restart: %s checkpoint payload version must be an exact integer scalar; "
            "historical checkpoints require offline migration" % runtime_kind
        )
    version = int(value.item())
    if version != expected:
        raise ValueError(
            "restart: %s checkpoint payload version %r unsupported; expected exactly %d; "
            "historical checkpoints require offline migration"
            % (runtime_kind, version, expected)
        )
    return version


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
    if array.size:
        digest.update(memoryview(cast(Any, array)).cast("B"))
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


def _seal_checkpoint_payload_with_identities(
    payload: dict[str, Any],
    *,
    runtime_kind: str,
    semantic: Identity,
    artifact: Identity,
    bind: Identity,
    run: Identity,
) -> Identity:
    """Seal an offline payload with explicit, already-authenticated lifecycle identities."""
    if MANIFEST_KEY in payload or IDENTITY_KEY in payload:
        raise ValueError("checkpoint payload already contains reserved identity keys")
    for value, domain in (
        (semantic, "semantic"),
        (artifact, "artifact"),
        (bind, "bind"),
        (run, "run"),
    ):
        if type(value) is not Identity or value.domain != domain:
            raise TypeError("checkpoint requires an exact domain-%r identity" % domain)
    if not isinstance(runtime_kind, str) or not runtime_kind:
        raise TypeError("checkpoint runtime kind must be non-empty text")
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


def seal_checkpoint_payload(owner: Any, payload: dict[str, Any], *, runtime_kind: str) -> Identity:
    """Add the canonical manifest and restart token to an in-memory NPZ payload."""
    semantic, artifact, bind = _runtime_identities(owner)
    run = getattr(owner, "last_run_identity", None)
    if type(run) is not Identity or run.domain != "run":
        raise RuntimeError(
            "checkpoint requires a prior pops.run(sim, **controls) so its execution controls "
            "have a run identity")
    return _seal_checkpoint_payload_with_identities(
        payload,
        runtime_kind=runtime_kind,
        semantic=semantic,
        artifact=artifact,
        bind=bind,
        run=run,
    )


def _strict_json(text: Any) -> dict[str, Any]:
    result = strict_json_loads(str(text), where="checkpoint manifest JSON")
    if not isinstance(result, dict):
        raise TypeError("checkpoint manifest must decode to a mapping")
    return result


def inspect_checkpoint_payload_integrity(
    payload: Any,
    *,
    runtime_kind: str,
) -> tuple[dict[str, Any], Identity]:
    """Authenticate an envelope and every payload digest without consulting a runtime.

    This is an offline integrity inspection, not a compatibility or migration decision. It proves
    only that a canonical checkpoint is internally complete and untampered.
    """
    files = _payload_files(payload)
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

    for field, domain in (
        ("semantic_identity", "semantic"),
        ("artifact_identity", "artifact"),
        ("bind_identity", "bind"),
    ):
        recorded = _identity_from_json(manifest[field])
        if recorded.domain != domain:
            raise ValueError("checkpoint %s has wrong domain" % field)
    run = _identity_from_json(manifest["run_identity"])
    if run.domain != "run":
        raise ValueError("checkpoint run_identity has wrong domain")
    for name, evidence in manifest["arrays"].items():
        if evidence != _array_evidence(payload[name]):
            raise ValueError("checkpoint payload digest mismatch for %r" % name)
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
    return manifest, restart


def authenticate_checkpoint_payload(owner: Any, payload: Any, *, runtime_kind: str) -> Identity:
    """Authenticate every checkpoint byte and all runtime identities before state mutation."""
    manifest, restart = inspect_checkpoint_payload_integrity(
        payload,
        runtime_kind=runtime_kind,
    )
    semantic, artifact, bind = _runtime_identities(owner)
    for field, current, domain in (
        ("semantic_identity", semantic, "semantic"),
        ("artifact_identity", artifact, "artifact"),
        ("bind_identity", bind, "bind"),
    ):
        recorded = _identity_from_json(manifest[field])
        if recorded.domain != domain or recorded.token != current.token:
            raise ValueError("checkpoint %s does not match the bound runtime" % field)
    from pops.runtime._engine_descriptors import abi_key
    files = _payload_files(payload)
    if "abi_key" not in files or str(payload["abi_key"]) != str(abi_key()):
        raise ValueError("checkpoint ABI identity does not match the loaded runtime")
    return restart


def checkpoint_run_identity(payload: Any) -> Identity:
    """Return the exact run identity carried by an authenticated checkpoint envelope."""
    files = set(
        getattr(
            payload,
            "files",
            payload.keys() if isinstance(payload, Mapping) else (),
        )
    )
    if MANIFEST_KEY not in files:
        raise ValueError("checkpoint has no canonical manifest")
    manifest = _strict_json(payload[MANIFEST_KEY])
    run = _identity_from_json(manifest.get("run_identity"))
    if run.domain != "run":
        raise ValueError("checkpoint run_identity has wrong domain")
    return run


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "IDENTITY_KEY", "MANIFEST_KEY",
    "authenticate_checkpoint_payload", "inspect_checkpoint_payload_integrity",
    "checkpoint_run_identity",
    "require_exact_payload_version",
    "seal_checkpoint_payload",
]
