"""Structural helpers for :mod:`pops.model.module`.

Kept separate so ``module.py`` stays a compact Module front-end. Everything in
this file is inert Python IR bookkeeping: no runtime, codegen, or ``_pops`` use.
"""

import json

from .handles import OperatorHandle
from .operators import Operator
from .spaces import FieldSpace, StateSpace


def body_identity(body):
    """A stable JSON identity identifying an already-captured operator IR body."""
    return json.dumps(canonical_body_identity(body), sort_keys=True, separators=(",", ":"))


def canonical_body_identity(body):
    """Canonical, side-effect-free body identity."""
    if body is None:
        return None
    if isinstance(body, (bool, int, float, str)):
        return body
    if isinstance(body, (list, tuple)):
        return [canonical_body_identity(v) for v in body]
    if isinstance(body, dict):
        return {
            str(k): canonical_body_identity(v)
            for k, v in sorted(body.items(), key=lambda kv: str(kv[0]))
        }
    if callable(body):
        raise TypeError("Module operator bodies must be captured IR, not Python callables")
    if hasattr(body, "_key") and callable(body._key):
        return {"type": type(body).__name__, "key": canonical_body_identity(body._key())}
    if hasattr(body, "to_dict") and callable(body.to_dict):
        return {"type": type(body).__name__, "dict": canonical_body_identity(body.to_dict())}
    if hasattr(body, "as_dict") and callable(body.as_dict):
        return {"type": type(body).__name__, "dict": canonical_body_identity(body.as_dict())}
    attrs = {}
    if hasattr(body, "__dict__"):
        attrs.update(vars(body))
    slots = getattr(body, "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot in slots:
        if slot.startswith("_") or not hasattr(body, slot):
            continue
        attrs[slot] = getattr(body, slot)
    if attrs:
        return {"type": type(body).__name__, "attrs": canonical_body_identity(attrs)}
    raise TypeError(
        "Module operator body %s is not structurally serializable for module_hash; "
        "use inert IR primitives/containers instead of relying on repr()."
        % type(body).__name__)


def symbolic_args(inputs):
    """Symbolic arguments used to execute a module.operator decorator once."""
    return tuple(symbolic_arg(space) for space in inputs)


def symbolic_arg(space):
    if isinstance(space, StateSpace):
        return SpaceArg(space, "cons")
    if isinstance(space, FieldSpace):
        return SpaceArg(space, "aux")
    return space


class SpaceArg:
    """Small symbolic view over a Space's components for decorator-time IR capture."""

    def __init__(self, space, var_kind):
        from pops.ir.expr import Var
        self.space = space
        self.name = space.name
        self.components = tuple(space.components)
        self._vars = {c: Var(c, var_kind) for c in self.components}
        self._ordered = tuple(self._vars[c] for c in self.components)

    def __iter__(self):
        return iter(self._ordered)

    def __len__(self):
        return len(self._ordered)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ordered[key]
        return self._vars[key]

    def __getattr__(self, name):
        try:
            return self._vars[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self):
        return "SymbolicSpaceArg(%r, components=%r)" % (self.name, list(self.components))


def space_record(space):
    record = {
        "name": space.name,
        "kind": space.kind,
        "components": list(space.components),
        "layout": space.layout,
    }
    if hasattr(space, "roles"):
        record["roles"] = dict(space.roles)
    if hasattr(space, "storage"):
        record["storage"] = space.storage
    return record


def param_record(param):
    return {
        "name": param.name,
        "default": param.default,
        "dtype": param.dtype,
        "kind": param.kind,
    }


def metadata_record(record):
    return {k: (repr(v) if k == "expression" else v) for k, v in record.items()}


def operator_record(registry, op):
    return {
        "id": registry.id_of(op.name),
        "name": op.name,
        "kind": op.kind,
        "signature": repr(op.signature),
        "requirements": dict(op.requirements),
        "capabilities": dict(op.capabilities),
        "lowering": dict(op.lowering),
        "handle": repr(op.handle()),
        "body": body_identity(op.body),
    }


def normalize_source_selectors(sources, *, who):
    """Normalize typed source selectors for ``Module.rate_operator``."""
    if sources is None:
        return None
    out = []
    for src in sources:
        if isinstance(src, str):
            if src == "default":
                out.append(src)
                continue
            raise TypeError(
                "%s: sources must contain typed source operators/handles, not the string %r; "
                "keep the object returned by Module.operator(..., kind='local_source')" % (who, src))
        if isinstance(src, Operator):
            if src.kind != "local_source":
                raise TypeError("%s: source operator %r has kind %r, expected 'local_source'"
                                % (who, src.name, src.kind))
            out.append(src.name)
            continue
        if isinstance(src, OperatorHandle):
            if src.kind not in (None, "local_source"):
                raise TypeError("%s: source handle %r has kind %r, expected 'local_source'"
                                % (who, src.name, src.kind))
            out.append(src.name)
            continue
        raise TypeError("%s: sources must contain typed source operators/handles, got %r"
                        % (who, type(src).__name__))
    return out
