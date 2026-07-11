"""Canonical execution-request identity derived from one bound simulation."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
import math

from pops._manifest_protocol import manifest_envelope, parse_manifest_envelope
from pops.identity import Identity, make_identity


RUN_MANIFEST_SCHEMA_VERSION = 2
_MANIFEST_KIND = "run"


def _finite_float(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % where)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % where)
    return result


def _strict_int(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % where)
    if value < minimum:
        raise ValueError("%s must be >= %d" % (where, minimum))
    return value


class RunManifest:
    """Immutable run controls and their domain-``run`` identity."""

    __slots__ = (
        "schema_version", "bind_identity", "start_time", "start_macro_step", "controls",
        "run_identity",
    )

    def __init__(self, *, bind_identity: Any, start_time: Any, start_macro_step: Any,
                 controls: Any) -> None:
        if type(bind_identity) is not Identity or bind_identity.domain != "bind":
            raise TypeError("RunManifest bind_identity must be a domain-'bind' Identity")
        if not isinstance(controls, Mapping):
            raise TypeError("RunManifest controls must be a mapping")
        exact = dict(controls)
        expected = {"t_end", "cfl", "max_steps", "output_mode"}
        if set(exact) != expected:
            raise ValueError("RunManifest controls keys must be exactly %s" % sorted(expected))
        object.__setattr__(self, "schema_version", RUN_MANIFEST_SCHEMA_VERSION)
        object.__setattr__(self, "bind_identity", Identity.from_data(bind_identity.to_data()))
        object.__setattr__(self, "start_time", _finite_float(start_time, where="start_time"))
        object.__setattr__(self, "start_macro_step", _strict_int(
            start_macro_step, where="start_macro_step"))
        output_mode = exact["output_mode"]
        if not isinstance(output_mode, str) or not output_mode:
            raise TypeError("RunManifest output_mode must be a non-empty string")
        object.__setattr__(self, "controls", MappingProxyType({
            "t_end": _finite_float(exact["t_end"], where="controls.t_end"),
            "cfl": _finite_float(exact["cfl"], where="controls.cfl"),
            "max_steps": _strict_int(exact["max_steps"], where="controls.max_steps"),
            "output_mode": output_mode,
        }))
        object.__setattr__(self, "run_identity", make_identity("run", self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bind_identity": self.bind_identity.to_data(),
            "start_time": self.start_time.hex(),
            "start_macro_step": self.start_macro_step,
            "controls": {
                "t_end": self.controls["t_end"].hex(),
                "cfl": self.controls["cfl"].hex(),
                "max_steps": self.controls["max_steps"],
                "output_mode": self.controls["output_mode"],
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "bind_identity": self.bind_identity.token,
            "start_time": self.start_time,
            "start_macro_step": self.start_macro_step,
            "controls": dict(self.controls),
        }
        payload["run_identity"] = self.run_identity.token
        return manifest_envelope(
            kind=_MANIFEST_KIND,
            schema_version=self.schema_version,
            payload=payload,
        )

    @classmethod
    def from_dict(cls, data: Any) -> RunManifest:
        keys = {
            "bind_identity", "start_time", "start_macro_step", "controls", "run_identity",
        }
        payload = parse_manifest_envelope(
            data,
            kind=_MANIFEST_KIND,
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            payload_keys=keys,
            where="RunManifest",
        )
        bind = Identity.from_token(payload["bind_identity"])
        result = cls(
            bind_identity=bind,
            start_time=payload["start_time"],
            start_macro_step=payload["start_macro_step"],
            controls=payload["controls"],
        )
        if payload["run_identity"] != result.run_identity.token or result.to_dict() != dict(data):
            raise ValueError("RunManifest identity or canonical payload mismatch")
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RunManifest is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RunManifest is immutable")


def begin_run(engine: Any, *, t_end: Any, cfl: Any, max_steps: Any,
              output_dir: Any) -> RunManifest:
    snapshot = getattr(engine, "bound_snapshot", None)
    if snapshot is None:
        raise RuntimeError("run requires a completed pops.bind transaction")
    bind_identity = getattr(snapshot, "bind_identity", None)
    if type(bind_identity) is not Identity or bind_identity.domain != "bind":
        raise RuntimeError("bound runtime carries no canonical bind identity")
    manifest = RunManifest(
        bind_identity=bind_identity,
        start_time=engine.time(),
        start_macro_step=engine.macro_step(),
        controls={
            "t_end": t_end, "cfl": cfl, "max_steps": max_steps,
            # A path is placement/provenance, not semantic execution identity.
            "output_mode": "explicit-root" if output_dir is not None else "current-directory",
        },
    )
    engine._last_run_manifest = manifest
    engine._last_run_identity = manifest.run_identity
    return manifest


__all__ = ["RUN_MANIFEST_SCHEMA_VERSION", "RunManifest", "begin_run"]
