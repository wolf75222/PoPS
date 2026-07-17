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
from typing import Any, cast

from pops.identity import Identity, canonical_bytes, make_identity


SCHEMA_VERSION = 7


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
    if isinstance(value, bytes):
        # Identity.to_data() deliberately carries its SHA-256 digest as canonical bytes. Bound
        # snapshots also expose a JSON view, so retain the exact value through an explicit tagged
        # lowercase-hex projection instead of rejecting a valid ExecutionContext identity.
        return {"bytes_hex": value.hex()}
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
    digest.update(memoryview(cast(Any, contiguous)).cast("B"))
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
        "field_plans", "step_transaction", "params", "aux_evidence", "initial_evidence",
        "bind_schema_identity", "execution_context", "bind_identity",
    )

    def __init__(self, *, semantic_identity: Any, artifact_identity: Any, layout: Any,
                 blocks: Any, field_plans: Any, step_transaction: Any, params: Any, aux_evidence: Any,
                 initial_evidence: Any, bind_schema_identity: Any,
                 execution_context: Any = None) -> None:
        semantic = _identity(semantic_identity, "semantic", where="semantic_identity")
        artifact = _identity(artifact_identity, "artifact", where="artifact_identity")
        schema_id = _identity(bind_schema_identity, "bind-schema", where="bind_schema_identity")
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "semantic_identity", semantic)
        object.__setattr__(self, "artifact_identity", artifact)
        for name, value in (
            ("layout", layout), ("blocks", list(blocks)), ("field_plans", field_plans),
            ("step_transaction", step_transaction), ("params", list(params)),
            ("aux_evidence", aux_evidence),
            ("initial_evidence", initial_evidence),
            ("execution_context", execution_context),
        ):
            object.__setattr__(self, name, _freeze(_data(value, where=name), where=name))
        object.__setattr__(self, "bind_schema_identity", schema_id)
        object.__setattr__(self, "bind_identity", make_identity("bind", self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_identity": self.semantic_identity.to_data(),
            "artifact_identity": self.artifact_identity.to_data(),
            "layout": _thaw(self.layout),
            "blocks": _thaw(self.blocks),
            "field_plans": _thaw(self.field_plans),
            "step_transaction": _thaw(self.step_transaction),
            "params": _thaw(self.params),
            "aux_evidence": _thaw(self.aux_evidence),
            "initial_evidence": _thaw(self.initial_evidence),
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


class MultiLayoutBoundSnapshot:
    """Immutable bind authority for a partitioned set of native Systems."""

    __slots__ = (
        "schema_version", "semantic_identity", "artifact_identity", "layout", "blocks",
        "field_plans", "step_transaction", "params", "aux_evidence", "initial_evidence",
        "bind_schema_identity", "execution_context", "bind_identity",
    )

    def __init__(self, install_plan: Any, child_snapshots: Any) -> None:
        from pops.codegen._plans import require_install_plan

        install_plan = require_install_plan(install_plan)
        snapshots = tuple(child_snapshots)
        if not snapshots:
            raise ValueError("MultiLayoutBoundSnapshot requires native child snapshots")
        if any(type(snapshot) is not BoundSnapshot for snapshot in snapshots):
            raise TypeError(
                "MultiLayoutBoundSnapshot children must be exact low-level BoundSnapshot values")
        artifact = install_plan.artifact
        if any(snapshot.semantic_identity != artifact.semantic_identity
               or snapshot.artifact_identity != artifact.artifact_identity
               for snapshot in snapshots):
            raise ValueError("multi-layout child snapshot changed the compiled artifact identity")
        child_names = tuple(
            name for snapshot in snapshots for name in snapshot.block_names())
        expected_names = tuple(install_plan.instances)
        if len(child_names) != len(set(child_names)) or set(child_names) != set(expected_names):
            raise ValueError(
                "multi-layout child snapshots must cover every InstallPlan instance exactly once")
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "semantic_identity", artifact.semantic_identity)
        object.__setattr__(self, "artifact_identity", artifact.artifact_identity)
        object.__setattr__(self, "layout", _freeze(
            _data(artifact.layout_plan.inspect(), where="layout"), where="layout"))
        by_name = {
            row["name"]: row
            for snapshot in snapshots for row in snapshot.to_dict()["blocks"]
        }
        ordered = tuple(by_name[block.name] for block in artifact.blocks)
        object.__setattr__(self, "blocks", _freeze(ordered, where="blocks"))
        object.__setattr__(self, "field_plans", _freeze({}, where="field_plans"))
        transactions = tuple(snapshot.to_dict()["step_transaction"] for snapshot in snapshots)
        if any(value != transactions[0] for value in transactions[1:]):
            raise ValueError("per-layout Program transaction contracts are not identical")
        object.__setattr__(
            self, "step_transaction", _freeze(transactions[0], where="step_transaction"))
        object.__setattr__(self, "params", _freeze(
            _data(install_plan.params.rows(), where="params"), where="params"))
        object.__setattr__(self, "aux_evidence", _freeze(
            _input_evidence(install_plan.aux, where="aux"), where="aux_evidence"))
        object.__setattr__(self, "initial_evidence", _freeze(
            _input_evidence(install_plan.bind_inputs.initial_state, where="initial_state"),
            where="initial_evidence"))
        object.__setattr__(
            self, "bind_schema_identity", make_identity(
                "bind-schema", artifact.bind_schema.to_dict()))
        object.__setattr__(self, "execution_context", _freeze(
            _data(install_plan.execution_context, where="execution_context"),
            where="execution_context"))
        object.__setattr__(self, "bind_identity", install_plan.bind_identity)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "semantic_identity": self.semantic_identity.to_data(),
            "artifact_identity": self.artifact_identity.to_data(),
            "layout": _thaw(self.layout),
            "blocks": _thaw(self.blocks),
            "field_plans": _thaw(self.field_plans),
            "step_transaction": _thaw(self.step_transaction),
            "params": _thaw(self.params),
            "aux_evidence": _thaw(self.aux_evidence),
            "initial_evidence": _thaw(self.initial_evidence),
            "bind_schema_identity": self.bind_schema_identity.to_data(),
            "execution_context": _thaw(self.execution_context),
            "bind_identity": self.bind_identity.to_data(),
        }
        return result

    def block_names(self) -> list[str]:
        return [row["name"] for row in self.blocks]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("MultiLayoutBoundSnapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MultiLayoutBoundSnapshot is immutable")


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
        if not callable(lower):
            raise TypeError("bound engine does not implement the private spatial-lowering protocol")
        spatial = lower(spec.get("spatial"))
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


def _transaction_data(compiled: Any) -> Any:
    component = getattr(compiled, "program", None)
    program = getattr(component, "program", component)
    plan = program.transaction_plan() if program is not None else None
    return None if plan is None else plan.to_data()


def _require_exact_install_inputs(engine: Any, compiled: Any, instances: Any,
                                  field_plans: Any, aux: Any, params: Any,
                                  install_plan: Any) -> Any:
    """Authenticate the exact final-plan values consumed by one native installation.

    Canonically equivalent copies are deliberately rejected.  The final ``bind`` identity belongs
    to one exact :class:`InstallPlan`; accepting aliases here would let the native runtime consume a
    different object graph while the snapshot claimed the plan's authenticated identity.
    """
    from pops.codegen._plans import require_install_plan

    plan = require_install_plan(install_plan)
    expected = (
        ("compiled artifact", compiled, plan.artifact),
        ("instances", instances, plan.instances),
        ("params", params, plan.params),
        ("aux", aux, plan.aux),
        ("field plans", field_plans, plan.artifact.plan.field_plans),
        ("execution context", getattr(engine, "_execution_context", None),
         plan.execution_context),
    )
    for label, actual, authoritative in expected:
        if actual is not authoritative:
            raise ValueError(
                "bound snapshot %s must be the exact value from the InstallPlan" % label)
    return plan


def _build_snapshot(engine: Any, compiled: Any, instances: Any, field_plans: Any,
                    aux: Any, params: Any, *, layout: str,
                    install_plan: Any = None) -> BoundSnapshot:
    plan = None
    if install_plan is not None:
        plan = _require_exact_install_inputs(
            engine, compiled, instances, field_plans, aux, params, install_plan)
        expected_layout = "uniform" if plan.target == "system" else "amr"
        if layout != expected_layout:
            raise ValueError("bound snapshot layout changed the InstallPlan target")
        # Derive the recorded values from the authenticated plan after proving that these are the
        # exact objects consumed by the native install.  No caller-supplied alias enters the final
        # snapshot authority.
        compiled = plan.artifact
        instances = plan.instances
        field_plans = plan.artifact.plan.field_plans
        aux = plan.aux
        params = plan.params
    semantic, artifact = _require_compiled_identities(compiled)
    rows = params.rows()
    snapshot = BoundSnapshot(
        semantic_identity=semantic,
        artifact_identity=artifact,
        layout={"kind": layout},
        blocks=_block_rows(engine, instances),
        field_plans={name: _data(value, where="field_plan[%r]" % name)
                     for name, value in sorted((field_plans or {}).items())},
        step_transaction=_transaction_data(compiled),
        params=rows,
        aux_evidence=_input_evidence(aux or {}, where="aux"),
        initial_evidence=_input_evidence(
            {name: spec["initial"] for name, spec in (instances or {}).items()
             if "initial" in spec}, where="initial_state"),
        bind_schema_identity=_schema_identity(params),
        execution_context=(
            plan.execution_context if plan is not None
            else getattr(engine, "_execution_context", None)
        ),
    )
    if plan is not None:
        # ``plan`` has just passed require_install_plan() and the exact-input proof above.  This is
        # the only route that may replace the autonomous low-level snapshot hash with a final bind
        # authority; BoundSnapshot itself accepts no bare/spoofable Identity.
        object.__setattr__(snapshot, "bind_identity", plan.bind_identity)
    return snapshot


def build_uniform_snapshot(engine: Any, compiled: Any, resolved_models: Any, instances: Any,
                           field_plans: Any, aux: Any, params: Any,
                           install_plan: Any = None) -> BoundSnapshot:
    if install_plan is not None:
        plan = _require_exact_install_inputs(
            engine, compiled, instances, field_plans, aux, params, install_plan)
        if tuple(resolved_models) != tuple(plan.instances) or any(
            resolved_models[name] is not plan.instances[name]["model"]
            for name in plan.instances
        ):
            raise ValueError(
                "bound snapshot resolved models must be the exact InstallPlan instance models")
        return _build_snapshot(
            engine, compiled, instances, field_plans, aux, params, layout="uniform",
            install_plan=plan)
    effective = {
        name: dict(spec, model=resolved_models.get(name, spec["model"]))
        for name, spec in (instances or {}).items()
    }
    return _build_snapshot(
        engine, compiled, effective, field_plans, aux, params, layout="uniform",
        install_plan=install_plan)


def build_amr_snapshot(engine: Any, compiled: Any, instances: Any, field_plans: Any,
                       aux: Any, params: Any, install_plan: Any = None) -> BoundSnapshot:
    return _build_snapshot(
        engine, compiled, instances, field_plans, aux, params, layout="amr",
        install_plan=install_plan)


__all__ = [
    "BoundSnapshot", "MultiLayoutBoundSnapshot", "SCHEMA_VERSION",
    "build_uniform_snapshot", "build_amr_snapshot",
]
