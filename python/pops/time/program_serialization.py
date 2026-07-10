"""Canonical Program IR v2 serialization and hashing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pops.ir.literals import scalar_data
from pops.time.values import ProgramValue, _Affine, _affine_ids


def _serialize_schedule(schedule: Any) -> dict[str, Any]:
    params = {}
    for name, item in schedule.params.items():
        if isinstance(item, ProgramValue):
            params[name] = {"program_value_id": item.id}
        elif callable(item):
            # A Python callable is deliberately not lowerable (ADC-458), but authoring/inspection
            # and hashing must still fail deterministically at the explicit lowerability gate rather
            # than inside json.dumps with an opaque TypeError.
            params[name] = {
                "unsupported_python_callable": {
                    "module": getattr(item, "__module__", type(item).__module__),
                    "qualname": getattr(item, "__qualname__", type(item).__qualname__),
                }
            }
        elif name == "dt" and item is not None:
            params[name] = {"scalar": scalar_data(item)}
        else:
            params[name] = _json_ready(item)
    return {"kind": schedule.kind, "policy": schedule.policy, "params": params}


def _json_ready(value: Any) -> Any:
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return _json_ready(hook())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
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
        "field_problem": context.field_problem,
        "stage_sources": [list(item) for item in context.stage_sources],
        "outputs": list(context.outputs),
    }


class _ProgramSerialization:
    """Mixin owning the canonical external form of a Program graph."""

    @staticmethod
    def _serialize_node(value: Any) -> dict[str, Any]:
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
                _ProgramSerialization._serialize_node(node) for node in attrs["cond_block"]]
            attrs["body_block"] = [
                _ProgramSerialization._serialize_node(node) for node in attrs["body_block"]]
            attrs["cond"], attrs["body"] = attrs["cond"].id, attrs["body"].id
        elif value.op in ("range", "if"):
            attrs["body_block"] = [
                _ProgramSerialization._serialize_node(node) for node in attrs["body_block"]]
            attrs["body"] = attrs["body"].id
        elif value.op == "matrix_free_operator":
            attrs["apply_block"] = ([
                _ProgramSerialization._serialize_node(node) for node in attrs["apply_block"]]
                if attrs.get("apply_block") else None)
            for key in ("apply_result", "apply_in", "apply_out"):
                ref = attrs.get(key)
                attrs[key] = (_affine_ids(ref) if isinstance(ref, _Affine)
                              else (ref.id if isinstance(ref, ProgramValue) else None))
        elif value.op == "solve_local_nonlinear":
            attrs["residual_block"] = [
                _ProgramSerialization._serialize_node(node) for node in attrs["residual_block"]]
            for key in ("residual", "iterate", "guess"):
                attrs[key] = attrs[key].id
        node = {"id": value.id, "name": value.name, "vtype": value.vtype, "op": value.op,
                "block": value.block,
                "inputs": [item.id for item in value.inputs], "attrs": _json_ready(attrs)}
        if value.space is not None:
            node["space"] = _json_ready(value.space)
        # A local operator's context is a validation-only authoring witness. The solve node already
        # carries the explicit fields input that determines runtime semantics, so serializing the
        # witness would make P.call(L, fields) hash differently from the equivalent typed
        # P.linear_source(L) + solve_local_linear(..., fields=fields) route.
        if value.field_context is not None and value.vtype != "operator":
            node["field_context"] = _json_ready(_serialize_field_context(value.field_context))
        return node

    def _serialize(self) -> dict[str, Any]:
        order = self._block_indices()
        result = {
            "name": self.name,
            "version": 2,
            "nodes": [self._serialize_node(value) for value in self._values],
            "commits": sorted((block, state.id) for block, state in self._commits.items()),
            "block_order": sorted(order, key=order.get),
        }
        if self._histories:
            result["histories"] = [
                {
                    "name": name,
                    "lag": lag,
                    "ncomp": getattr(self, "_histories_ncomp", {}).get(name),
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
                "nodes": [self._serialize_node(node) for node in block], "result": value.id}
        return result

    def _ir_hash(self) -> str:
        blob = json.dumps(self._serialize(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _block_indices(self) -> dict[str, int]:
        order = {}
        for value in self._values:
            if value.op == "state" and value.block not in order:
                order[value.block] = len(order)
        return order


__all__ = ["_ProgramSerialization"]
