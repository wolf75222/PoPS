"""Authenticated structural identity of one per-block compiled model."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any


IDENTITY_PROTOCOL = "pops.compiled-model-identity.v1"


def model_compile_identity(model: Any, *, module_hash: Any = None) -> Any:
    """Return the exact structural identity a ``model.compile()`` result must carry."""
    model_hash = _hash_from(model, "_model_hash")
    if module_hash is None:
        module_hash = getattr(model, "_compile_source_module_hash", None)
    if module_hash is None:
        module_hash = _hash_from(model, "module_hash", required=False)
    if module_hash is None:
        module = getattr(model, "module", None)
        module_hash = _hash_from(module, "module_hash", required=False)
    if model_hash is None and module_hash is None:
        raise TypeError(
            "pops.compile: install-model protocol requires a structural _model_hash() or "
            "Module.module_hash(); compile() cannot authenticate a name/repr-only model"
        )
    return compiled_model_identity(model_hash=model_hash, module_hash=module_hash)


def compiled_model_identity(*, model_hash: Any, module_hash: Any = None) -> Any:
    """Build the immutable wire-sized identity retained by :class:`CompiledModel`."""
    if model_hash is not None:
        _require_hash(model_hash, "model_hash")
    if module_hash is not None:
        _require_hash(module_hash, "module_hash")
    if model_hash is None and module_hash is None:
        raise ValueError("compiled model identity requires model_hash or module_hash")
    return MappingProxyType({
        "protocol": IDENTITY_PROTOCOL,
        "model_hash": model_hash,
        "module_hash": module_hash,
    })


def validate_compiled_model_identity(value: Any) -> Any:
    """Return a detached canonical identity or reject an opaque/tampered payload."""
    if not hasattr(value, "keys"):
        raise TypeError("CompiledModel.definition_identity must be a mapping")
    row = dict(value)
    expected = {"protocol", "model_hash", "module_hash"}
    if set(row) != expected:
        raise ValueError(
            "CompiledModel.definition_identity keys must be exactly %s" % sorted(expected))
    if row["protocol"] != IDENTITY_PROTOCOL:
        raise ValueError("CompiledModel.definition_identity has an unsupported protocol")
    return compiled_model_identity(
        model_hash=row["model_hash"], module_hash=row["module_hash"])


def authenticate_compiled_model(model: Any, compiled: Any, *, module_hash: Any = None) -> None:
    """Prove that ``compiled`` was produced from the exact structural input ``model``."""
    expected = model_compile_identity(model, module_hash=module_hash)
    actual = validate_compiled_model_identity(
        getattr(compiled, "definition_identity", None))
    if dict(actual) != dict(expected):
        raise ValueError(
            "pops.compile: model.compile() returned a CompiledModel for a different structural "
            "model (expected %r, got %r)" % (dict(expected), dict(actual))
        )
    if actual["model_hash"] is not None and compiled.model_hash != actual["model_hash"]:
        raise ValueError(
            "pops.compile: CompiledModel.model_hash disagrees with its authenticated "
            "definition_identity"
        )


def _hash_from(owner: Any, name: str, *, required: bool = False) -> Any:
    method = getattr(owner, name, None) if owner is not None else None
    if not callable(method):
        if required:
            raise TypeError("%s must expose %s()" % (type(owner).__name__, name))
        return None
    value = method()
    _require_hash(value, name)
    return value


def _require_hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError("%s must be a non-empty structural hash string" % name)


__all__ = [
    "authenticate_compiled_model", "compiled_model_identity", "model_compile_identity",
    "validate_compiled_model_identity",
]
