"""Deterministic SSA-to-C++ evidence; source correspondence never proves execution."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
from pops._structural_data import canonical_json, json_containers, transform_tree
import re
from typing import TypedDict


class _EmissionScope(TypedDict):
    program_hash: str
    program: object
    nodes: dict[int, dict[str, object]]
    bindings: dict[int, dict[str, object]]
    unbound: set[int]


_ACTIVE: ContextVar[_EmissionScope | None] = ContextVar("pops_program_source_map", default=None)
_PREFIX = "// pops-source-map@1 "


def _plain(value):
    from collections.abc import Mapping
    def describe(value):
        if isinstance(value, Mapping):
            return tuple(value.values()), lambda items: dict(zip(map(str, value), items, strict=True))
        if isinstance(value, (tuple, list)):
            return value, list
        if value is None or type(value) in (str, bool, int):
            return None, value
        raise TypeError("source map metadata must be finite canonical data")
    return transform_tree(value, describe)


def _json(value):
    return canonical_json(value, allow_nan=False)


def _mark(kind, data):
    encoded = base64.b64encode(_json(data).encode()).decode("ascii")
    return _PREFIX + kind + " " + encoded


def _node(value):
    result = {"ssa": value.id, "op": value.op, "name": value.name,
              "inputs": [item.id for item in value.inputs]}
    for key in ("realization_use_v02", "realization_continuation_v14", "scientific_graph_v14"):
        if key in value.attrs:
            result[key] = _plain(value.attrs[key])
    return result


def _bindings(value):
    """Keep only literal C++ bindings returned by the emitter, not Python objects."""
    def describe(value):
        if type(value) is str:
            return None, value
        if isinstance(value, dict):
            entries = [(str(key), item) for key, item in value.items() if type(key) in (str, int)]
            return tuple(item for _, item in entries), \
                lambda items: dict(zip((key for key, _ in entries), items, strict=True))
        if isinstance(value, (tuple, list)):
            return value, list
        return None, None
    return transform_tree(value, describe)


def _remember(scope, environment):
    for key in scope["unbound"].intersection(environment):
        binding = _bindings(environment[key])
        if binding is not None:
            scope["bindings"][key] = {_json(binding): binding}
            scope["unbound"].remove(key)


@contextmanager
def operation(value, lines, prelude=None, *, environment=None, channel="body", binding_role=None):
    """Bracket the actual statements appended by one lowering, including nested regions."""
    scope = _ACTIVE.get()
    if scope is None:
        yield
        return
    metadata = _node(value)
    metadata["program_hash"] = scope["program_hash"]
    metadata["channel"] = channel
    if binding_role is not None:
        metadata["binding_role"] = binding_role
    if prelude is lines:
        prelude = None
    lines.append(_mark("begin", metadata))
    if prelude is not None:
        prelude.append(_mark("begin", dict(metadata, channel="install")))
    try:
        yield
    finally:
        if environment is not None:
            _remember(scope, environment)
        binding = (_bindings(environment.get(value.id)) if environment is not None else None)
        end = {"ssa": value.id, "op": value.op, "binding": binding}
        lines.append(_mark("end", end))
        if prelude is not None:
            prelude.append(_mark("end", dict(end, binding=None)))


@contextmanager
def operation_group(values, lines, *, environment, channel):
    """Map one actually coalesced native statement to every exact output SSA."""
    scope = _ACTIVE.get()
    if scope is None:
        yield
        return
    values = tuple(values)
    if not values:
        raise ValueError("source-map operation group cannot be empty")
    nodes = [dict(_node(value), program_hash=scope["program_hash"], channel=channel)
             for value in values]
    lines.append(_mark("group_begin", {"nodes": nodes}))
    try:
        yield
    finally:
        _remember(scope, environment)
        lines.append(_mark("group_end", {"nodes": [
            {"ssa": value.id, "op": value.op, "binding": _bindings(environment.get(value.id))}
            for value in values]}))


def emission_program_hash(program):
    """Reuse identity only inside this exact synchronous emission invocation.

    No cache is retained on the authoring Program. Separate invocations and
    different Programs always obtain their own current structural identity.
    """
    scope = _ACTIVE.get()
    if scope is not None and scope["program"] is program:
        return scope["program_hash"]
    return program._ir_hash()


def emission_scope(function):
    """One deterministic mapping scope for a complete emitted translation unit."""
    @wraps(function)
    def emit(program, *args, **kwargs):
        from .program_lowerability import all_ops
        nodes = {value.id: _node(value) for value in all_ops(program)}
        if program._dt_bound is not None:
            for value in program._dt_bound[0]:
                nodes[value.id] = _node(value)
                for child in program._subblock_value_refs(value):
                    nodes[child.id] = _node(child)
        scope: _EmissionScope = {"program": program, "program_hash": program._ir_hash(), "nodes": nodes, "bindings": {},
                                 "unbound": set(nodes)}
        token = _ACTIVE.set(scope)
        try:
            source = function(program, *args, **kwargs)
            inventory = {"program_hash": scope["program_hash"], "nodes": [
                dict(nodes[key], cpp_bindings=list(scope["bindings"].get(key, {}).values()))
                for key in sorted(nodes)]}
            source += "\n" + _mark("inventory", inventory) + "\n"
            evidence = source_map(source)
            missing = [row for row in evidence["unlocated"]
                       if row["status"] == "not_emitted_or_unmapped"]
            if missing:
                raise ValueError("Program source map has unaccounted SSA: " + ", ".join(
                    "%s:%s" % (row["ssa"], row["op"]) for row in missing))
            uses = [row for row in evidence["operations"] if "realization_use_v02" in row]
            if any(not row["effective_ranges"] for row in uses if row["channel"] != "install"):
                raise ValueError("Program source map cannot locate an effective Realize output binding")
            return source
        finally:
            _ACTIVE.reset(token)
    return emit


def _ranges(numbers):
    result = []
    for number in numbers:
        if result and result[-1][1] + 1 == number:
            result[-1][1] = number
        else:
            result.append([number, number])
    return result


def source_map(source):
    """Inspect exact retained C++; ranges are one-based, inclusive, and directly owned.

    Nested operations own their own lines. Parent ranges only contain parent
    statements, not a broad enclosure that claims the child's implementation.
    No-instruction aliases retain actual emitter bindings and have empty ranges.
    Any unaccounted SSA is exposed explicitly instead of declaring full coverage.
    """
    if type(source) is not str:
        raise TypeError("Program source map requires the exact generated C++ text")
    rows, stack, inventories = [], [], []
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith(_PREFIX):
            fields = stripped[len(_PREFIX):].split(" ", 1)
            if len(fields) != 2:
                raise ValueError("malformed Program source-map marker")
            kind, payload = fields
            try:
                data = json_containers(base64.b64decode(payload, validate=True))
            except Exception as error:
                raise ValueError("invalid Program source-map marker data") from error
            if kind == "begin":
                row = dict(data, emission=len(rows), _lines=[], cpp_binding=None,
                           lexical_regions=[parent["emission"] for parent in stack])
                rows.append(row)
                stack.append(row)
            elif kind == "group_begin":
                if not data["nodes"]:
                    raise ValueError("empty Program source-map emission group")
                shared_lines = []
                members = []
                identity = len(rows)
                for node in data["nodes"]:
                    row = dict(node, emission=len(rows), _lines=shared_lines, cpp_binding=None,
                               lexical_regions=[parent["emission"] for parent in stack],
                               shared_emission_group=identity,
                               shared_ssa=[item["ssa"] for item in data["nodes"]])
                    rows.append(row)
                    members.append(row)
                members[0]["_group_members"] = members
                stack.append(members[0])
            elif kind == "group_end":
                members = stack[-1].get("_group_members") if stack else None
                if not members or [(r["ssa"], r["op"]) for r in members] != [
                        (r["ssa"], r["op"]) for r in data["nodes"]]:
                    raise ValueError("unbalanced Program source-map emission group")
                stack.pop().pop("_group_members")
                for row, binding in zip(members, data["nodes"], strict=True):
                    row["cpp_binding"] = binding["binding"]
            elif kind == "end":
                if not stack or "_group_members" in stack[-1] or (stack[-1]["ssa"], stack[-1]["op"]) != (data["ssa"], data["op"]):
                    raise ValueError("unbalanced Program source-map regions")
                stack.pop()["cpp_binding"] = data["binding"]
            elif kind == "inventory":
                if stack:
                    raise ValueError("Program source inventory is inside an open region")
                inventories.append(data)
            else:
                raise ValueError("unknown Program source-map marker")
        elif stack and stripped and not stripped.startswith("//"):
            stack[-1]["_lines"].append(number)
    if stack:
        raise ValueError("unterminated Program source-map region")
    lines = source.splitlines()
    for row in rows:
        numbers = row.pop("_lines")
        row["ranges"] = _ranges(numbers)
        row["status"] = "emitted" if numbers else "no_direct_statements"
        # Identifier bindings are references supplied by the emitter. Do not
        # manufacture a declaration symbol from a member access or expression.
        binding = row["cpp_binding"]
        row["identifier_binding"] = binding if (type(binding) is str and
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", binding) and
            any(re.search(r"\b" + re.escape(binding) + r"\b", lines[n - 1]) for n in numbers)) else None
    origins = {}
    for row in rows:
        if row["ranges"] and row["cpp_binding"] is not None:
            origins.setdefault((row["program_hash"], _json(row["cpp_binding"])), []).append(row)
    for row in rows:
        related = ([] if row["ranges"] or row["cpp_binding"] is None else
                   [origin for origin in origins.get((row["program_hash"], _json(row["cpp_binding"])), [])
                    if row["lexical_regions"][:len(origin["lexical_regions"])] == origin["lexical_regions"]])
        row["binding_origins"] = [item["emission"] for item in related]
        row["effective_ranges"] = row["ranges"] or [span for item in related for span in item["ranges"]]
        row["location_basis"] = ("direct_statements" if row["ranges"] else
            "identical_emitter_binding" if related else "no_direct_statements")
    mapped = {(row["program_hash"], row["ssa"]) for row in rows}
    unlocated = []
    for inventory in inventories:
        for node in inventory["nodes"]:
            if (inventory["program_hash"], node["ssa"]) not in mapped:
                unlocated.append(dict(node, program_hash=inventory["program_hash"], ranges=[],
                    status="not_emitted_or_unmapped",
                    justification="no exact emitter region was recorded; an environment binding alone cannot locate an operation"))
    result = {"contract": "pops.program-source-map@1", "evidence": "source_correspondence_only",
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "line_count": len(lines), "operations": rows, "unlocated": unlocated,
              "inventories": inventories,
              "complete": bool(inventories) and not any(
                  row["status"] == "not_emitted_or_unmapped" for row in unlocated)}
    result["map_sha256"] = hashlib.sha256(_json(result).encode()).hexdigest()
    return result


__all__ = ["source_map"]
