"""Canonical Program IR v2 serialization and hashing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pops.ir.literals import scalar_data
from pops.model.handles import Handle
from pops.time.references import handle_data
from pops.time.values import ProgramValue, _Affine, _affine_ids


def _serialize_schedule(schedule: Any) -> dict[str, Any]:
    from pops.time.schedule import AMRLevel, Event, Every, When
    domain = {"type": type(schedule.domain).__name__, "clock": schedule.clock.to_data(),
              "at": _json_ready(schedule.domain.at)}
    if type(schedule.domain) is AMRLevel:
        domain["level"] = schedule.domain.level
    elif type(schedule.domain) is Event:
        domain["event"] = schedule.domain.event.to_data()
    trigger = {"type": type(schedule.trigger).__name__}
    if type(schedule.trigger) is Every:
        trigger["n"] = schedule.trigger.n
    elif type(schedule.trigger) is When:
        condition = schedule.trigger.condition
        if isinstance(condition, ProgramValue):
            trigger["condition"] = {"program_value_id": condition.id}
        elif callable(condition):
            trigger["condition"] = {"unsupported_python_callable": {
                "module": getattr(condition, "__module__", type(condition).__module__),
                "qualname": getattr(condition, "__qualname__", type(condition).__qualname__),
            }}
        else:
            trigger["condition"] = _json_ready(condition)
    return {"schema_version": 1, "domain": domain, "trigger": trigger,
            "off": type(schedule.off).__name__ if schedule.off is not None else None}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Handle):
        return {"handle": handle_data(value)}
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return _json_ready(hook())
    if isinstance(value, Mapping):
        if all(isinstance(key, str) and key for key in value):
            return {key: _json_ready(item) for key, item in value.items()}
        entries = [[_json_ready(key), _json_ready(item)] for key, item in value.items()]
        entries.sort(key=lambda item: json.dumps(
            item[0], sort_keys=True, separators=(",", ":")))
        return {"mapping_entries": entries}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_ready(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":")))
    return value


def _serialize_field_context(context: Any) -> dict[str, Any]:
    from pops.time.field_context import FieldReadProvenance
    if isinstance(context, FieldReadProvenance):
        return {"reads": [_serialize_field_context(item) for item in context.contexts]}
    return {
        "field": _json_ready(context.field),
        "stage_sources": [[_json_ready(item[0]), item[1]] for item in context.stage_sources],
        "outputs": list(context.outputs),
    }


class _ProgramSerialization:
    """Mixin owning the canonical external form of a Program graph."""

    @staticmethod
    def _serialize_node(value: Any, *, include_provenance: bool = True) -> dict[str, Any]:
        attrs = dict(value.attrs)
        if "schedule" in attrs:
            attrs["schedule"] = _serialize_schedule(attrs["schedule"])
        if value.op == "scalar_op":
            attrs["operands"] = [
                (kind, scalar_data(item) if kind == "c" else item)
                for kind, item in attrs["operands"]]
        elif value.op == "compare" and "rhs" in attrs:
            attrs["rhs"] = scalar_data(attrs["rhs"])
        elif value.op == "cell_compare":
            attrs["value"] = scalar_data(attrs["value"])
        if value.op == "while":
            attrs["cond_block"] = [
                _ProgramSerialization._serialize_node(
                    node, include_provenance=include_provenance) for node in attrs["cond_block"]]
            attrs["body_block"] = [
                _ProgramSerialization._serialize_node(
                    node, include_provenance=include_provenance) for node in attrs["body_block"]]
            attrs["cond"], attrs["body"] = attrs["cond"].id, attrs["body"].id
        elif value.op == "range":
            attrs["body_block"] = [
                _ProgramSerialization._serialize_node(
                    node, include_provenance=include_provenance) for node in attrs["body_block"]]
            attrs["body"] = attrs["body"].id
        elif value.op == "branch":
            for arm in ("true", "false"):
                attrs[arm + "_block"] = [
                    _ProgramSerialization._serialize_node(
                        node, include_provenance=include_provenance)
                    for node in attrs[arm + "_block"]]
                attrs[arm + "_result"] = attrs[arm + "_result"].id
        elif value.op == "matrix_free_operator":
            attrs["apply_block"] = ([
                _ProgramSerialization._serialize_node(
                    node, include_provenance=include_provenance) for node in attrs["apply_block"]]
                if attrs.get("apply_block") else None)
            for key in ("apply_result", "apply_in", "apply_out"):
                ref = attrs.get(key)
                attrs[key] = (_affine_ids(ref) if isinstance(ref, _Affine)
                              else (ref.id if isinstance(ref, ProgramValue) else None))
        elif value.op == "solve_local_nonlinear":
            attrs["residual_block"] = [
                _ProgramSerialization._serialize_node(
                    node, include_provenance=include_provenance) for node in attrs["residual_block"]]
            for key in ("residual", "iterate", "guess"):
                attrs[key] = attrs[key].id
        node = {"id": value.id, "name": value.name, "vtype": value.vtype, "op": value.op,
                "block": handle_data(value.block) if value.block is not None else None,
                "state": handle_data(value.state_ref) if value.state_ref is not None else None,
                "point": _json_ready(value.point),
                "inputs": [item.id for item in value.inputs], "attrs": _json_ready(attrs)}
        if value.space is not None:
            node["space"] = _json_ready(value.space)
        # A local operator's context is a validation-only authoring witness. The solve node already
        # carries the explicit fields input that determines runtime semantics, so serializing the
        # witness would make P.call(L, fields) hash differently from the equivalent typed
        # P.linear_source(L) + solve_local_linear(..., fields=fields) route.
        if value.field_context is not None and value.vtype != "operator":
            node["field_context"] = _json_ready(_serialize_field_context(value.field_context))
        if include_provenance:
            node["provenance"] = value.provenance.to_data()
        return node

    def _serialize(self, *, include_provenance: bool = True) -> dict[str, Any]:
        if not isinstance(include_provenance, bool):
            raise TypeError("Program._serialize include_provenance must be bool")
        order = self._block_indices()
        result = {
            "name": self.name,
            "version": 4,
            "clock": self.clock.to_data(),
            "nodes": [self._serialize_node(
                value, include_provenance=include_provenance) for value in self._values],
            "commits": [
                {
                    "state": handle_data(state_ref),
                    "block": handle_data(state_ref.block_ref),
                    "value": value.id,
                }
                for state_ref, value in sorted(
                    self._commits.items(), key=lambda item: item[0].qualified_id)
            ],
            "block_order": [handle_data(block) for block in sorted(order, key=order.get)],
        }
        transaction = self.transaction_plan()
        if transaction is not None:
            result["step_transaction"] = transaction.to_data()
        if self._histories:
            result["histories"] = [
                {
                    "name": name,
                    "lag": lag,
                    "ncomp": getattr(self, "_histories_ncomp", {}).get(name),
                    "state": (handle_data(self._history_state_refs[name])
                              if name in self._history_state_refs else None),
                }
                for name, lag in sorted(self._histories.items())
            ]
        persistence = getattr(self, "_history_persistence", {})
        if persistence:
            result["history_persistence"] = [
                {
                    "name": name,
                    "depth": depth,
                    "policy": _json_ready(policy.to_manifest()),
                }
                for name, (depth, policy) in sorted(persistence.items())
            ]
        if self._dt_bound is not None:
            block, value = self._dt_bound
            result["dt_bound"] = {
                "nodes": [self._serialize_node(
                    node, include_provenance=include_provenance) for node in block],
                "result": value.id}
        return result

    def _ir_hash(self) -> str:
        blob = json.dumps(
            self._serialize(include_provenance=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _semantic_serialize(self) -> dict[str, Any]:
        """Scientific IR projection shared by manual programs and library constructors."""
        from pops.identity.semantic import program_semantic_data
        return program_semantic_data(self)

    def _block_indices(self) -> dict[Any, int]:
        order = {}
        for value in self._values:
            if value.op == "state" and value.block not in order:
                order[value.block] = len(order)
        return order


__all__ = ["_ProgramSerialization"]
