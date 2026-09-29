"""Versioned vector-row stencil authoring over exact declared state handles."""

from __future__ import annotations

import re

from pops._ir.expr import Const, Expr, Var
from pops._ir.vector_expr import VectorExpr
from pops._ir.visitors import _children
from pops.descriptors import BrickDescriptor
from pops.model.hash_data import canonical_hash_data

from .user import _NATIVE_INT_MIN, _NATIVE_INT_MAX, _capture_identities, _source_identity

_NATIVE_ID = "pops.generated.reconstruction.joint-stencil/v2"


def _width(state):
    from pops.model import StateHandle
    from pops.physics.board_handles import StateHandle as BoardStateHandle

    if not isinstance(state, (StateHandle, BoardStateHandle)):
        raise TypeError("joint reconstruction requires exact StateHandle inputs")
    return len(state.components if isinstance(state, BoardStateHandle) else state.space.components)


class _Sample:
    def __init__(self, states):
        self.states = states
        self.nodes = {}

    def __call__(self, offset, state=None):
        if type(offset) is not int:
            raise TypeError("reconstruction sample offset must be an exact integer constant")
        if not _NATIVE_INT_MIN <= offset <= _NATIVE_INT_MAX:
            raise ValueError("reconstruction sample offset is outside the native int32 range")
        state = self.states[0] if state is None else state
        try:
            row = self.states.index(state)
        except ValueError as exc:
            raise ValueError("joint reconstruction sample reads an undeclared state") from exc
        spelling = "m%d" % -offset if offset < 0 else "p%d" % offset
        for component in range(_width(state)):
            key = offset, row, component
            if key not in self.nodes:
                self.nodes[key] = Var(
                    "pops_recon_joint_%s_r%d_c%d" % (spelling, row, component),
                    "reconstruction_sample",
                )
        return VectorExpr(self.nodes[offset, row, component] for component in range(_width(state)))


def _reads(roots, counts, *, allowed=None):
    from pops._ir.native_call import NativeCall
    from pops._ir.values import RuntimeParamRef

    reads, captures, seen = set(), {}, set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, RuntimeParamRef):
            for name, qid in _capture_identities(node):
                if captures.setdefault(name, qid) != qid:
                    raise ValueError("joint reconstruction has colliding runtime parameter names")
            continue
        if isinstance(node, NativeCall):
            raise NotImplementedError(
                "user reconstruction native calls need a declared fallible status contract"
            )
        if isinstance(node, Var):
            match = re.fullmatch(r"pops_recon_joint_([mp])(\d+)_r(\d+)_c(\d+)", node.name)
            if (
                node.kind != "reconstruction_sample"
                or match is None
                or (allowed is not None and id(node) not in allowed)
            ):
                raise ValueError("joint reconstruction body contains a foreign variable")
            offset = int(match[2]) * (-1 if match[1] == "m" else 1)
            row, component = int(match[3]), int(match[4])
            if (
                not _NATIVE_INT_MIN <= offset <= _NATIVE_INT_MAX
                or row >= len(counts)
                or component >= counts[row]
            ):
                raise ValueError("joint reconstruction sample is outside its declared shape")
            reads.add((offset, row, component))
        children = _children(node)
        if not children and not isinstance(node, (Const, Var)):
            raise ValueError("joint reconstruction has an unsupported scalar leaf")
        pending.extend(children)
    return tuple(sorted(reads)), tuple(sorted(captures.items()))


def _contract(roots, states, order, *, allowed=None):
    counts = tuple(_width(state) for state in states)
    if not all(counts) or len(set(states)) != len(states):
        raise ValueError("joint reconstruction requires distinct nonempty state inputs")
    reads, captures = _reads(roots, counts, allowed=allowed)
    offsets = tuple(sorted({read[0] for read in reads}))
    minimum, maximum = min(offsets, default=0), max(offsets, default=0)
    depth = max(1, 1 - minimum, maximum + 1)
    if depth > _NATIVE_INT_MAX or maximum - minimum + 1 > _NATIVE_INT_MAX:
        raise ValueError("user reconstruction stencil cannot fit native int32 ghost/count metadata")
    payload = {
        "schema_version": 2,
        "kind": "joint_source_stencil",
        "state_names": tuple(state.inspect()["local_id"] for state in states),
        "component_counts": counts,
        "formal_order": order,
        "body": canonical_hash_data(roots, where="joint reconstruction body"),
        "sample_reads": reads,
        "runtime_captures": captures,
    }
    return {
        "state": states[0],
        "sampling": states[1:],
        "component_counts": counts,
        "formal_order": order,
        "ghost_depth": depth,
        "stencil_min_offset": minimum,
        "stencil_max_offset": maximum,
        "sample_offsets": offsets,
        "sample_reads": reads,
        "runtime_captures": captures,
        "source_identity": _source_identity(payload),
        "body": payload["body"],
    }


def author_joint(body, *, state, sampling, formal_order, name):
    if not isinstance(sampling, tuple):
        raise TypeError("joint reconstruction sampling must be an ordered tuple of StateHandle")
    states = (state, *sampling)
    counts = tuple(_width(item) for item in states)
    sample = _Sample(states)
    output = body(sample)
    if not isinstance(output, tuple) or len(output) != counts[0]:
        raise TypeError("joint reconstruction body must return one Expr per output state component")
    roots = tuple(item if isinstance(item, Expr) else Const(item) for item in output)
    options = _contract(
        roots, states, formal_order, allowed={id(node) for node in sample.nodes.values()}
    )
    descriptor = BrickDescriptor(
        name,
        "generated",
        category="reconstruction",
        native_id=_NATIVE_ID,
        scheme="source_stencil",
        options=options,
        requirements={"ghost_depth": options["ghost_depth"], "source_compiled": True},
        capabilities={"vector_row": True, "oriented_sample": True},
        expression=roots,
    )
    # Source capabilities survive shallow descriptor qualification. They are not serialized as
    # opaque IDs: resolved numerical identity already carries the complete canonical handles.
    descriptor._joint_state_sources = states
    return descriptor


def authenticate_joint(value):
    options = value.options
    roots = value.expression
    if (
        value.native_id != _NATIVE_ID
        or not isinstance(roots, tuple)
        or any(not isinstance(item, Expr) for item in roots)
    ):
        raise ValueError("joint reconstruction body shape changed")
    try:
        states = (options["state"], *options["sampling"])
        sources = getattr(value, "_joint_state_sources", ())
        if len(sources) != len(states):
            raise ValueError("joint reconstruction lost its exact source state authority")
        for state, source in zip(states, sources, strict=True):
            declared = state.declaration_ref or state
            original = source.declaration_ref or source
            # Equal authoring handles need no premature model freeze. Once an option is
            # qualified, compare canonical declarations rather than presentation names.
            if declared != original and declared._resolved() != original._resolved():
                raise ValueError("joint reconstruction state authority changed after authoring")
            if source.is_instance and (
                not state.is_instance or state.block_ref._resolved() != source.block_ref._resolved()
            ):
                raise ValueError("joint reconstruction block instance changed after authoring")
        order = options["formal_order"]
        if (
            type(order) is not int
            or not 1 <= order <= _NATIVE_INT_MAX
            or len(roots) != _width(states[0])
        ):
            raise ValueError("joint reconstruction order/output shape changed")
        expected = _contract(roots, states, order)
    except KeyError as exc:
        raise ValueError("joint reconstruction source contract is incomplete") from exc
    if set(options) != set(expected) or any(
        canonical_hash_data(options[key]) != canonical_hash_data(expected[key])
        for key in expected
        if key not in ("state", "sampling")
    ):
        raise ValueError("joint reconstruction source contract changed after authoring")
    if value.requirements != {
        "ghost_depth": expected["ghost_depth"],
        "source_compiled": True,
    } or value.capabilities != {"vector_row": True, "oriented_sample": True}:
        raise ValueError("joint reconstruction source capabilities changed")
    return value
