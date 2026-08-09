"""Semantic embedded-boundary authority for strict Uniform checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from typing import Any


EMBEDDED_BOUNDARY_CONTRACT_KEY = "pops_embedded_boundary_contract"
EMBEDDED_BOUNDARY_CONTRACT_SCHEMA_VERSION = 1

_EB_REPORT_KEYS = {
    "enabled",
    "geometry_mode",
    "kappa_min",
    "face_open_eps",
    "cut_theta_min",
    "semantic_digest",
    "materialization_digest",
    "generation",
}
_SEMANTIC_DIGEST_PREFIX = "pops.prepared-eb-semantic.v1:sha256:"
_MATERIALIZATION_DIGEST_PREFIX = "pops.prepared-eb-geometry.v1:sha256:"


def _dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise TypeError("checkpoint embedded-boundary dimension must be exactly 1, 2, or 3")
    return value


def _finite_threshold(value: Any, *, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("checkpoint embedded-boundary %s must be a real scalar" % name)
    result = float(value)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        interval = "(0,1]" if positive else "[0,1]"
        raise ValueError("checkpoint embedded-boundary %s must lie in %s" % (name, interval))
    return result


def _digest(value: Any, *, prefix: str, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise TypeError("checkpoint embedded-boundary %s has an unsupported identity" % name)
    suffix = value[len(prefix) :]
    if len(suffix) != 64:
        raise ValueError("checkpoint embedded-boundary %s must contain one SHA-256" % name)
    try:
        bytes.fromhex(suffix)
    except ValueError:
        raise ValueError(
            "checkpoint embedded-boundary %s SHA-256 is not hexadecimal" % name
        ) from None
    return value


@dataclass(frozen=True, slots=True)
class CheckpointEmbeddedBoundaryContract:
    """One immutable semantic EB contract, independent of ranks and local materialization."""

    dimension: int
    enabled: bool
    mode: str
    kappa_min: float
    face_open_eps: float
    cut_theta_min: float
    semantic_digest: str
    identity: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from pops.identity import make_identity

        dimension = _dimension(self.dimension)
        if type(self.enabled) is not bool:
            raise TypeError("checkpoint embedded-boundary enabled must be an exact bool")
        if not isinstance(self.mode, str):
            raise TypeError("checkpoint embedded-boundary mode must be text")
        if self.enabled:
            if self.mode not in ("staircase", "cutcell"):
                raise ValueError("enabled checkpoint embedded boundary needs staircase or cutcell")
            kappa_min = _finite_threshold(self.kappa_min, name="kappa_min", positive=True)
            face_open_eps = _finite_threshold(
                self.face_open_eps, name="face_open_eps", positive=False
            )
            cut_theta_min = _finite_threshold(
                self.cut_theta_min, name="cut_theta_min", positive=True
            )
            semantic_digest = _digest(
                self.semantic_digest,
                prefix=_SEMANTIC_DIGEST_PREFIX,
                name="semantic digest",
            )
        else:
            if self.mode != "none" or self.semantic_digest:
                raise ValueError("disabled checkpoint embedded boundary must be canonical none")
            if any(
                float(value) != 0.0
                for value in (
                    self.kappa_min,
                    self.face_open_eps,
                    self.cut_theta_min,
                )
            ):
                raise ValueError("disabled checkpoint embedded-boundary thresholds must be zero")
            kappa_min = face_open_eps = cut_theta_min = 0.0
            semantic_digest = ""
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "kappa_min", kappa_min)
        object.__setattr__(self, "face_open_eps", face_open_eps)
        object.__setattr__(self, "cut_theta_min", cut_theta_min)
        object.__setattr__(self, "semantic_digest", semantic_digest)
        object.__setattr__(
            self,
            "identity",
            make_identity("checkpoint-embedded-boundary", self._payload()),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": EMBEDDED_BOUNDARY_CONTRACT_SCHEMA_VERSION,
            "dimension": self.dimension,
            "enabled": self.enabled,
            "mode": self.mode,
            "kappa_min": self.kappa_min.hex(),
            "face_open_eps": self.face_open_eps.hex(),
            "cut_theta_min": self.cut_theta_min.hex(),
            "semantic_digest": self.semantic_digest,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.token}

    @classmethod
    def from_data(cls, data: Any) -> CheckpointEmbeddedBoundaryContract:
        from pops.identity import Identity

        required = {
            "schema_version",
            "dimension",
            "enabled",
            "mode",
            "kappa_min",
            "face_open_eps",
            "cut_theta_min",
            "semantic_digest",
            "identity",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("checkpoint embedded-boundary contract has an unsupported exact schema")
        if type(data["schema_version"]) is not int:
            raise TypeError("checkpoint embedded-boundary schema_version must be an exact integer")
        if data["schema_version"] != EMBEDDED_BOUNDARY_CONTRACT_SCHEMA_VERSION:
            raise ValueError("checkpoint embedded-boundary contract schema version is unsupported")
        float_names = ("kappa_min", "face_open_eps", "cut_theta_min")
        if any(not isinstance(data[name], str) for name in float_names):
            raise TypeError("checkpoint embedded-boundary thresholds must use float.hex strings")
        try:
            thresholds = {name: float.fromhex(data[name]) for name in float_names}
        except ValueError:
            raise ValueError(
                "checkpoint embedded-boundary thresholds contain invalid float.hex data"
            ) from None
        result = cls(
            dimension=data["dimension"],
            enabled=data["enabled"],
            mode=data["mode"],
            semantic_digest=data["semantic_digest"],
            **thresholds,
        )
        if Identity.from_token(data["identity"]) != result.identity or result.to_data() != dict(
            data
        ):
            raise ValueError("checkpoint embedded-boundary contract does not authenticate payload")
        return result

    @classmethod
    def from_effective_report(cls, report: Any) -> CheckpointEmbeddedBoundaryContract:
        if not isinstance(report, Mapping):
            raise TypeError("native effective-options report must be a mapping")
        topology = report.get("topology")
        eb = report.get("eb")
        if not isinstance(topology, Mapping) or set(topology) != {"dimension", "periodicity"}:
            raise TypeError("native effective-options report lacks exact spatial topology")
        dimension = _dimension(topology["dimension"])
        if not isinstance(eb, Mapping) or set(eb) != _EB_REPORT_KEYS:
            raise TypeError("native effective-options report lacks the exact EB authority")
        enabled = eb["enabled"]
        if type(enabled) is not bool:
            raise TypeError("native effective-options EB enabled must be an exact bool")
        if enabled:
            _digest(
                eb["materialization_digest"],
                prefix=_MATERIALIZATION_DIGEST_PREFIX,
                name="materialization digest",
            )
            generation = eb["generation"]
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
                raise ValueError("native effective-options EB generation must be positive")
            return cls(
                dimension=dimension,
                enabled=True,
                mode=eb["geometry_mode"],
                kappa_min=eb["kappa_min"],
                face_open_eps=eb["face_open_eps"],
                cut_theta_min=eb["cut_theta_min"],
                semantic_digest=eb["semantic_digest"],
            )
        if eb["geometry_mode"] != "none" or eb["semantic_digest"] or eb["materialization_digest"]:
            raise ValueError("disabled native EB report is not canonical")
        if eb["generation"] != 0:
            raise ValueError("disabled native EB report must have generation zero")
        return cls(dimension, False, "none", 0.0, 0.0, 0.0, "")


def current_checkpoint_embedded_boundary_contract(owner: Any) -> CheckpointEmbeddedBoundaryContract:
    native = getattr(owner, "_s", None)
    report = getattr(native, "effective_options_report", None)
    if not callable(report):
        raise TypeError("native checkpoint engine lacks its embedded-boundary authority report")
    return CheckpointEmbeddedBoundaryContract.from_effective_report(report())


def add_checkpoint_embedded_boundary_contract(
    payload: dict[str, Any], contract: CheckpointEmbeddedBoundaryContract
) -> None:
    if type(contract) is not CheckpointEmbeddedBoundaryContract:
        raise TypeError("checkpoint payload requires an exact embedded-boundary contract")
    if EMBEDDED_BOUNDARY_CONTRACT_KEY in payload:
        raise ValueError("checkpoint payload contains a duplicate embedded-boundary authority")
    payload[EMBEDDED_BOUNDARY_CONTRACT_KEY] = json.dumps(
        contract.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def inspect_checkpoint_embedded_boundary_contract(
    payload: Any,
) -> CheckpointEmbeddedBoundaryContract:
    files = set(getattr(payload, "files", payload.keys() if isinstance(payload, Mapping) else ()))
    if EMBEDDED_BOUNDARY_CONTRACT_KEY not in files:
        raise ValueError("checkpoint lacks its semantic embedded-boundary contract")
    from pops._manifest_protocol import strict_json_loads

    data = strict_json_loads(
        str(payload[EMBEDDED_BOUNDARY_CONTRACT_KEY]),
        where="checkpoint embedded-boundary contract",
    )
    return CheckpointEmbeddedBoundaryContract.from_data(data)


def authenticate_checkpoint_embedded_boundary_contract(
    owner: Any, payload: Any
) -> CheckpointEmbeddedBoundaryContract:
    recorded = inspect_checkpoint_embedded_boundary_contract(payload)
    current = current_checkpoint_embedded_boundary_contract(owner)
    if recorded.dimension != current.dimension:
        raise ValueError("restart embedded-boundary dimension differs from the native runtime")
    if recorded.identity != current.identity:
        raise ValueError("restart embedded-boundary geometry differs from the bound runtime")
    return recorded


__all__ = [
    "CheckpointEmbeddedBoundaryContract",
    "EMBEDDED_BOUNDARY_CONTRACT_KEY",
    "EMBEDDED_BOUNDARY_CONTRACT_SCHEMA_VERSION",
    "add_checkpoint_embedded_boundary_contract",
    "authenticate_checkpoint_embedded_boundary_contract",
    "current_checkpoint_embedded_boundary_contract",
    "inspect_checkpoint_embedded_boundary_contract",
]
