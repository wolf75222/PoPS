"""Intern typed device regions while preserving each authored invocation.

Canonical names are assigned as the closed IR is lowered. No emitted scientific
C++ is rewritten, and the host staging/publication protocol is left in place.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
from typing import Any

from pops._structural_data import canonical_json

_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("pops_device_kernels", default=None)


def kernel_scope(function):
    @wraps(function)
    def emit(*args, **kwargs):
        token = _ACTIVE.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    return emit


class DeviceKernel:
    """Explicit typed captures plus one alpha-renamed, checked expression body."""

    def __init__(self):
        self.fields: list[tuple[str, str, str, Any]] = []
        self._captures: dict[tuple[str, str], str] = {}
        self._locals: dict[int, str] = {}
        self.values: dict[int, Any] = {}
        self.bindings: dict[int, Any] = {}

    def capture(self, cpp_type, host_expression, *, identity=None):
        key = (cpp_type, host_expression)
        if key not in self._captures:
            member = "capture_%d" % len(self.fields)
            self._captures[key] = member
            self.fields.append((cpp_type, member, host_expression, identity))
        return self._captures[key]

    def local(self, value):
        self._locals.setdefault(value.id, "value_%d" % len(self._locals))
        return self._locals[value.id]

    @contextmanager
    def operation(self, value, environment):
        self.values[value.id] = value
        try:
            yield
        finally:
            if value.id in environment:
                self.bindings[value.id] = environment[value.id]

    def invocation(self, body):
        registry = _ACTIVE.get()
        if registry is None:
            raise RuntimeError("device kernel emission requires a translation-unit scope")
        # This is the deterministic lowering of the IR, not a text substitution:
        # literals, checked failures, lazy branches and ordered operations remain.
        body = tuple(body)
        contract = canonical_json(
            {
                "schema": "pops.typed-device-region@1",
                "fields": [(kind, name, identity) for kind, name, _, identity in self.fields],
                "body": body,
            }
        )
        identity = hashlib.sha256(contract.encode()).hexdigest()
        name = "pops_device_kernel_" + identity
        if identity not in registry:
            registry[identity] = {
                "name": name,
                "contract": contract,
                "fields": tuple(self.fields),
                "body": body,
                "values": {},
                "bindings": {},
                "uses": 0,
            }
        record = registry[identity]
        if record["contract"] != contract:
            raise ValueError("device kernel identity collision")
        record["values"].update(self.values)
        record["bindings"].update(self.bindings)
        record["uses"] += 1
        return name + "{" + ", ".join(field[2] for field in self.fields) + "}"


def device_environment(kernel, captures, environment, signature):
    """Map already staged host bindings to explicitly typed device members."""
    from .program_emit_finite_kokkos import _tree, _path, _flat, _TYPES, _NS

    def domain(value):
        return {key: kernel.capture("std::int64_t", token) for key, token in value.items()}

    result = {}
    for value in captures:
        sig = signature(value)
        source = environment[value.id]
        if _flat(sig):
            result[value.id] = _tree(
                sig,
                lambda leaf, path, source=source: kernel.capture(
                    _TYPES[leaf["vtype"]], _path(source, path)
                ),
            )
        elif sig["vtype"] == "index_set":
            result[value.id] = domain(source)
        elif sig["vtype"] == "index":
            result[value.id] = {
                "domain": domain(source["domain"]),
                "offset": kernel.capture("std::int64_t", source["offset"]),
            }
        else:
            result[value.id] = {
                "domain": domain(source["domain"]),
                "columns": _tree(
                    sig["element"],
                    lambda leaf, path, source=source: kernel.capture(
                        _NS
                        + "View<"
                        + ("unsigned char" if leaf["vtype"] == "bool" else _TYPES[leaf["vtype"]])
                        + ">",
                        _path(source["columns"], path),
                    ),
                ),
            }
    return result


def kernel_definitions():
    from .program_source_map import operation_group

    registry = _ACTIVE.get()
    if registry is None:
        raise RuntimeError("device kernel definitions require a translation-unit scope")
    lines = []
    for identity, record in registry.items():
        lines.append(
            "// pops-interned-device-kernel@1 %s invocations=%d" % (identity, record["uses"])
        )
        lines.append("struct %s {" % record["name"])
        lines.extend("  %s %s;" % (kind, member) for kind, member, _, _ in record["fields"])
        lines.append("  KOKKOS_FUNCTION void operator()(const std::int64_t index) const {")
        values = tuple(record["values"].values())
        if values:
            with operation_group(
                values, lines, environment=record["bindings"], channel="interned_kokkos_body"
            ):
                lines.extend("    " + line for line in record["body"])
        else:
            lines.extend("    " + line for line in record["body"])
        lines.extend(("  }", "};"))
    return "\n".join(lines) + "\n"
