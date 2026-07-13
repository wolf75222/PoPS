"""Canonical identity of one fully materialised ``pops.bind`` transaction.

The bind manifest deliberately sits between a compiled artifact and an execution.  It carries the
two authenticated compile identities plus every effective runtime choice.  No authoring object,
``repr`` projection, pathname, or implicit spatial default participates in the identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import hashlib
from types import MappingProxyType
from typing import Any

from pops.identity import Identity, canonical_bytes, make_identity


SCHEMA_VERSION = 4


def _identity(value: Any, domain: str, *, where: str) -> Identity:
    if type(value) is not Identity:
        raise TypeError("%s must be a pops.identity.Identity" % where)
    if value.domain != domain:
        raise ValueError("%s must have domain %r, got %r" % (where, domain, value.domain))
    # Round-trip through the strict wire protocol so a forged subclass/container never enters a bind.
    return Identity.from_data(value.to_data())


def _freeze(value: Any, *, where: str) -> Any:
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("%s keys must be non-empty strings" % where)
            out[key] = _freeze(item, where="%s.%s" % (where, key))
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, where=where) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("%s contains non-canonical value %s" % (where, type(value).__name__))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _data(value: Any, *, where: str) -> Any:
    """Strict JSON projection for an already-resolved descriptor/value."""
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, Fraction):
        return {"rational": [value.numerator, value.denominator]}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Enum):
        return {"enum": "%s.%s.%s" % (
            type(value).__module__, type(value).__qualname__, value.name)}
    if isinstance(value, type):
        return {"symbol": "%s.%s" % (value.__module__, value.__qualname__)}
    if type(value) is Identity:
        return {"identity": value.token}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return {key: _data(item, where=where) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_data(item, where=where) for item in value]
    canonical_identity = getattr(value, "canonical_identity", None)
    if callable(canonical_identity):
        return _data(canonical_identity(), where=where)
    for hook_name in ("to_data", "to_manifest", "consumer_data"):
        hook = getattr(value, hook_name, None)
        if callable(hook):
            return _data(hook(), where=where)
    options = getattr(value, "options", None)
    if callable(options):
        options = options()
    if isinstance(options, Mapping):
        return {
            "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
            "options": _data(options, where=where),
        }
    raise TypeError(
        "%s cannot enter bind identity: %s has no canonical data protocol"
        % (where, type(value).__name__))


def _array_evidence(value: Any, *, where: str) -> dict[str, Any]:
    """Content-addressed evidence for an initial-state or auxiliary array."""
    import numpy as np

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("%s must not use object dtype" % where)
    contiguous = np.ascontiguousarray(array)
    header = canonical_bytes({
        "protocol": "pops.array-evidence.v1",
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    })
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "content_sha256": digest.hexdigest(),
    }


def _input_evidence(values: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("%s must be a mapping" % where)
    if any(not isinstance(name, str) or not name for name in values):
        raise TypeError("%s keys must be non-empty strings" % where)
    return {
        name: _array_evidence(value, where="%s[%r]" % (where, name))
        for name, value in sorted(values.items())
    }


class BoundSnapshot:
    """Deeply immutable, exact bind manifest with a domain-``bind`` identity."""

    __slots__ = (
        "schema_version", "semantic_identity", "artifact_identity", "layout", "blocks",
        "solvers", "cadence", "params", "aux_evidence", "initial_evidence", "outputs",
        "diagnostics", "bind_schema_identity", "execution_context", "bind_identity",
    )

    def __init__(self, *, semantic_identity: Any, artifact_identity: Any, layout: Any,
                 blocks: Any, solvers: Any, cadence: Any, params: Any, aux_evidence: Any,
                 initial_evidence: Any, outputs: Any, diagnostics: Any,
                 bind_schema_identity: Any, execution_context: Any = None) -> None:
        semantic = _identity(semantic_identity, "semantic", where="semantic_identity")
        artifact = _identity(artifact_identity, "artifact", where="artifact_identity")
        schema_id = _identity(bind_schema_identity, "bind-schema", where="bind_schema_identity")
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "semantic_identity", semantic)
        object.__setattr__(self, "artifact_identity", artifact)
        for name, value in (
            ("layout", layout), ("blocks", list(blocks)), ("solvers", solvers),
            ("cadence", cadence), ("params", list(params)), ("aux_evidence", aux_evidence),
            ("initial_evidence", initial_evidence), ("outputs", list(outputs)),
            ("diagnostics", list(diagnostics)),
            ("execution_context", execution_context),
        ):
            object.__setattr__(self, name, _freeze(_data(value, where=name), where=name))
        object.__setattr__(self, "bind_schema_identity", schema_id)
        payload = self._identity_payload()
        object.__setattr__(self, "bind_identity", make_identity("bind", payload))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_identity": self.semantic_identity.to_data(),
            "artifact_identity": self.artifact_identity.to_data(),
            "layout": _thaw(self.layout),
            "blocks": _thaw(self.blocks),
            "solvers": _thaw(self.solvers),
            "cadence": _thaw(self.cadence),
            "params": _thaw(self.params),
            "aux_evidence": _thaw(self.aux_evidence),
            "initial_evidence": _thaw(self.initial_evidence),
            "outputs": _thaw(self.outputs),
            "diagnostics": _thaw(self.diagnostics),
            "bind_schema_identity": self.bind_schema_identity.to_data(),
            "execution_context": _thaw(self.execution_context),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._identity_payload()
        for key in ("semantic_identity", "artifact_identity", "bind_schema_identity"):
            identity = getattr(self, key)
            result[key] = {
                "domain": identity.domain, "schema_version": identity.schema_version,
                "algorithm": identity.algorithm, "hexdigest": identity.hexdigest,
            }
        result["bind_identity"] = {
            "domain": self.bind_identity.domain,
            "schema_version": self.bind_identity.schema_version,
            "algorithm": self.bind_identity.algorithm,
            "hexdigest": self.bind_identity.hexdigest,
        }
        return result

    def block_names(self) -> list[str]:
        return [row["name"] for row in self.blocks]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BoundSnapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BoundSnapshot is immutable")


def _require_compiled_identities(compiled: Any) -> tuple[Identity, Identity]:
    if compiled is None:
        raise TypeError("pops.bind requires a compiled artifact carrying canonical identities")
    return (
        _identity(getattr(compiled, "semantic_identity", None), "semantic",
                  where="compiled.semantic_identity"),
        _identity(getattr(compiled, "artifact_identity", None), "artifact",
                  where="compiled.artifact_identity"),
    )


def _schema_identity(params: Any) -> Identity:
    schema = getattr(params, "schema", None)
    rows = getattr(params, "rows", None)
    if schema is None or not callable(rows):
        raise TypeError("pops.bind requires ResolvedBindings with an authenticated BindSchema")
    return make_identity("bind-schema", schema.to_dict())


def _block_rows(engine: Any, instances: Any) -> list[dict[str, Any]]:
    rows = []
    for name, spec in (instances or {}).items():
        model = spec["model"]
        lower = getattr(engine, "_lower_spatial", None)
        if callable(lower):
            spatial = lower(spec.get("spatial"))
        else:
            # AMR.add_equation consumes Spatial directly and resolves its sole default explicitly.
            from pops.runtime.bricks import Spatial
            supplied = spec.get("spatial")
            spatial = supplied if supplied is not None else Spatial()
            if not isinstance(spatial, Spatial):
                raise TypeError(
                    "bound block %r spatial selection did not lower to runtime Spatial" % name)
        model_identity = getattr(model, "definition_identity", None)
        if model_identity is None:
            raise TypeError("bound block %r model carries no definition_identity" % name)
        rows.append({
            "name": name,
            "definition_identity": _data(model_identity, where="block.definition_identity"),
            "spatial": _data(spatial, where="block.spatial"),
            "evolve": bool(spec.get("evolve", True)),
        })
    return rows


def _cadence_data(cadence: Any) -> dict[str, Any]:
    if cadence is None:
        return {"kind": "engine-default", "substeps": 1, "stride": 1, "cfl": "default"}
    return {
        "kind": "compiled-time", "substeps": int(cadence.substeps),
        "stride": int(cadence.stride), "cfl": cadence.cfl,
    }


def _build_snapshot(engine: Any, compiled: Any, instances: Any, solvers: Any, cadence: Any,
                    aux: Any, params: Any, *, layout: str) -> BoundSnapshot:
    semantic, artifact = _require_compiled_identities(compiled)
    rows = params.rows()
    return BoundSnapshot(
        semantic_identity=semantic,
        artifact_identity=artifact,
        layout={"kind": layout},
        blocks=_block_rows(engine, instances),
        solvers={name: _data(value, where="solver[%r]" % name)
                 for name, value in sorted((solvers or {}).items())},
        cadence=_cadence_data(cadence),
        params=rows,
        aux_evidence=_input_evidence(aux or {}, where="aux"),
        initial_evidence=_input_evidence(
            {name: spec["initial"] for name, spec in (instances or {}).items()
             if "initial" in spec}, where="initial_state"),
        # Exact consumers are authenticated by the compiled artifact and owned by RuntimeInstance.
        # The native BoundSnapshot deliberately carries no second policy registry.
        outputs=(),
        diagnostics=(),
        bind_schema_identity=_schema_identity(params),
        execution_context=getattr(engine, "_execution_context", None),
    )


def build_uniform_snapshot(engine: Any, compiled: Any, resolved_models: Any, instances: Any,
                           solvers: Any, cadence: Any, aux: Any, params: Any) -> BoundSnapshot:
    effective = {
        name: dict(spec, model=resolved_models.get(name, spec["model"]))
        for name, spec in (instances or {}).items()
    }
    return _build_snapshot(
        engine, compiled, effective, solvers, cadence, aux, params, layout="uniform")


def build_amr_snapshot(engine: Any, compiled: Any, instances: Any, solvers: Any,
                       cadence: Any, aux: Any, params: Any) -> BoundSnapshot:
    return _build_snapshot(engine, compiled, instances, solvers, cadence, aux, params, layout="amr")


__all__ = ["BoundSnapshot", "SCHEMA_VERSION", "build_uniform_snapshot", "build_amr_snapshot"]
