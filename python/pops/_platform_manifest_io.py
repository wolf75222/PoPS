"""Strict data boundary for platform manifests without importing runtime types."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _exact_mapping(value: Any, keys: frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    actual = frozenset(value)
    if actual != keys:
        raise ValueError("%s fields mismatch (missing=%r, unknown=%r)" % (
            where, sorted(keys - actual), sorted(actual - keys)))
    return value


def proof_to_data(proof: Any) -> dict[str, Any]:
    return {"value": _thaw(proof.value), "evidence": proof.evidence}


def proof_from_data(cls: Any, data: Any) -> Any:
    value = _exact_mapping(data, frozenset({"value", "evidence"}), "CapabilityProof")
    evidence = value["evidence"]
    if evidence is not None and type(evidence) is not str:
        raise TypeError("CapabilityProof.evidence must be text or None")
    return cls(value["value"], evidence)


_PRECISION_FIELDS = ("storage", "compute", "accumulation", "reduction")


def precision_to_data(policy: Any) -> dict[str, Any]:
    return {name: getattr(policy, name).to_data() for name in _PRECISION_FIELDS}


def precision_from_data(cls: Any, proof_cls: Any, data: Any) -> Any:
    value = _exact_mapping(data, frozenset(_PRECISION_FIELDS), "PrecisionPolicy")
    return cls(**{name: proof_cls.from_data(value[name]) for name in _PRECISION_FIELDS})


_MANIFEST_FIELDS = frozenset({
    "schema_version", "backend", "target", "abi", "precision", "device",
    "memory_spaces", "communicator", "capabilities",
})


def manifest_to_data(value: Any, schema_version: int) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "backend": value.backend.to_data(), "target": value.target.to_data(),
        "abi": value.abi.to_data(), "precision": value.precision.to_data(),
        "device": value.device.to_data(), "memory_spaces": value.memory_spaces.to_data(),
        "communicator": value.communicator.to_data(),
        "capabilities": {key: proof.to_data() for key, proof in value.capabilities.items()},
    }


def manifest_from_data(cls: Any, proof_cls: Any, precision_cls: Any, data: Any,
                       *, schema_version: int, where: str) -> Any:
    value = _exact_mapping(data, _MANIFEST_FIELDS, where)
    actual_schema = value["schema_version"]
    if type(actual_schema) is not int or actual_schema != schema_version:
        raise ValueError("%s.schema_version must be exactly %d" % (where, schema_version))
    raw_capabilities = value["capabilities"]
    if not isinstance(raw_capabilities, Mapping):
        raise TypeError("%s.capabilities must be a mapping" % where)
    if any(type(name) is not str or not name for name in raw_capabilities):
        raise TypeError("%s.capabilities keys must be non-empty exact strings" % where)
    return cls(
        backend=proof_cls.from_data(value["backend"]),
        target=proof_cls.from_data(value["target"]),
        abi=proof_cls.from_data(value["abi"]),
        precision=precision_cls.from_data(value["precision"]),
        device=proof_cls.from_data(value["device"]),
        memory_spaces=proof_cls.from_data(value["memory_spaces"]),
        communicator=proof_cls.from_data(value["communicator"]),
        capabilities={name: proof_cls.from_data(proof)
                      for name, proof in raw_capabilities.items()},
    )
