"""Source-authored, device-compiled scalar stencil reconstruction.

The Python body runs once during authoring.  Native face loops use only the
captured immutable expression and its statically known sample offsets.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pops._ir.expr import Const, Expr, Var
from pops._ir.visitors import _children
from pops.descriptors import BrickDescriptor
from pops.model.hash_data import canonical_hash_data


_NATIVE_ID = "pops.generated.reconstruction.stencil/v1"
_SCHEME = "source_stencil"
# The native sampler and its stencil metadata use signed 32-bit `int` in the
# published ABI. This is a representability bound, not a method-order ceiling.
_NATIVE_INT_MIN = -(1 << 31)
_NATIVE_INT_MAX = (1 << 31) - 1


class _Sample:
    def __init__(self) -> None:
        self.nodes: dict[int, Var] = {}

    def __call__(self, offset: int) -> Var:
        if type(offset) is not int:
            raise TypeError("reconstruction sample offset must be an exact integer constant")
        if offset < _NATIVE_INT_MIN or offset > _NATIVE_INT_MAX:
            raise ValueError("reconstruction sample offset is outside the native int32 range")
        if offset not in self.nodes:
            spelling = "m%d" % -offset if offset < 0 else "p%d" % offset
            self.nodes[offset] = Var("pops_recon_sample_%s" % spelling, "reconstruction_sample")
        return self.nodes[offset]


def _body_data(expression: Expr, nodes: dict[int, Var]) -> tuple[Any, tuple[int, ...]]:
    from pops._ir.native_call import NativeCall
    from pops._ir.values import RuntimeParamRef

    allowed = {id(node): offset for offset, node in nodes.items()}
    used: set[int] = set()
    pending = [expression]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, RuntimeParamRef):
            raise NotImplementedError(
                "user reconstruction runtime parameters need an exact model/block capture route")
        if isinstance(node, NativeCall):
            raise NotImplementedError(
                "user reconstruction native calls need a declared fallible status contract")
        if isinstance(node, Var) and id(node) not in allowed:
            raise ValueError("user reconstruction body reads a variable outside sample(offset)")
        if isinstance(node, Var):
            used.add(allowed[id(node)])
        children = _children(node)
        if not children and not isinstance(node, (Const, Var)):
            raise NotImplementedError(
                "user reconstruction body has a leaf without native scalar lowering")
        pending.extend(children)
    return canonical_hash_data(expression, where="user reconstruction body"), tuple(sorted(used))


def _identity_data(*, body_data: Any, order: int, offsets: tuple[int, ...],
                   minimum: int, maximum: int) -> dict[str, Any]:
    return {"schema_version": 1, "kind": _SCHEME, "body": body_data,
            "formal_order": order, "sample_offsets": offsets,
            "stencil_min_offset": minimum,
            "stencil_max_offset": maximum}


def _source_identity(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _retained_offsets(expression: Expr) -> tuple[int, ...]:
    """Re-check every live variable against the exact source-sampler protocol."""
    from pops._ir.native_call import NativeCall
    from pops._ir.values import RuntimeParamRef

    offsets: set[int] = set()
    pending = [expression]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (NativeCall, RuntimeParamRef)):
            raise ValueError("user reconstruction body has an unsupported native capture")
        if isinstance(node, Var):
            match = re.fullmatch(r"pops_recon_sample_([mp])(\d+)", node.name)
            if node.kind != "reconstruction_sample" or match is None:
                raise ValueError("user reconstruction body contains a foreign variable")
            offset = int(match.group(2)) * (-1 if match.group(1) == "m" else 1)
            if offset < _NATIVE_INT_MIN or offset > _NATIVE_INT_MAX:
                raise ValueError("user reconstruction body has an offset outside native int32")
            offsets.add(offset)
        children = _children(node)
        if not children and not isinstance(node, (Const, Var)):
            raise ValueError("user reconstruction body has an unsupported scalar leaf")
        pending.extend(children)
    return tuple(sorted(offsets))


def User(body: Any, *, formal_order: int, name: str = "user") -> BrickDescriptor:
    """Author one scalar trace formula from ``sample(integer_offset)``.

    The formula is applied independently to each component and each oriented face.
    ``formal_order`` is an authored assertion, not an order proof.  Captured Python
    scalars are evaluated now and become exact literals in the frozen expression;
    runtime parameters and cross-component/characteristic recipes require a
    separate typed capture protocol and are rejected here.
    """
    if not callable(body):
        raise TypeError("reconstruction.User(body=) requires a callable symbolic body")
    if type(formal_order) is not int or formal_order < 1:
        raise ValueError("reconstruction.User formal_order must be a positive exact integer")
    if formal_order > _NATIVE_INT_MAX:
        raise ValueError("reconstruction.User formal_order exceeds native int32 metadata")
    if not isinstance(name, str) or not name:
        raise ValueError("reconstruction.User name must be a non-empty string")
    sample = _Sample()
    expression = body(sample)
    if not isinstance(expression, Expr):
        if isinstance(expression, (tuple, list)):
            raise TypeError("reconstruction.User is scalar per component; vector bodies are unsupported")
        expression = Const(expression)
    body_data, offsets = _body_data(expression, sample.nodes)
    minimum = min(offsets, default=0)
    maximum = max(offsets, default=0)
    ghost_depth = max(1, 1 - minimum, maximum + 1)
    if ghost_depth > _NATIVE_INT_MAX or maximum - minimum + 1 > _NATIVE_INT_MAX:
        raise ValueError("user reconstruction stencil cannot fit native int32 ghost/count metadata")
    identity_data = _identity_data(body_data=body_data, order=formal_order, offsets=offsets,
                                   minimum=minimum, maximum=maximum)
    identity = _source_identity(identity_data)
    return BrickDescriptor(
        name, "generated", category="reconstruction", native_id=_NATIVE_ID,
        scheme=_SCHEME,
        options={"formal_order": formal_order, "ghost_depth": ghost_depth,
                 "stencil_min_offset": minimum, "stencil_max_offset": maximum,
                 "sample_offsets": offsets, "source_identity": identity, "body": body_data},
        requirements={"ghost_depth": ghost_depth, "source_compiled": True},
        capabilities={"componentwise_scalar": True, "oriented_sample": True},
        expression=expression,
    )


def authenticated_user_reconstruction(value: Any) -> BrickDescriptor:
    """Check the retained expression against its immutable package-source claim."""
    if type(value) is not BrickDescriptor or value.category != "reconstruction" or \
            value.brick_type != "generated" or value.native_id != _NATIVE_ID or \
            value.scheme != _SCHEME or not isinstance(value.expression, Expr):
        raise TypeError("user reconstruction requires an exact source-authored descriptor")
    options = value.options
    if set(options) != {"formal_order", "ghost_depth", "stencil_min_offset",
                        "stencil_max_offset", "sample_offsets", "source_identity", "body"}:
        raise ValueError("user reconstruction source contract is incomplete")
    order = options["formal_order"]
    minimum = options["stencil_min_offset"]
    maximum = options["stencil_max_offset"]
    offsets = options["sample_offsets"]
    if type(order) is not int or order < 1 or order > _NATIVE_INT_MAX or \
            any(type(n) is not int for n in (minimum, maximum)) \
            or minimum > maximum or minimum < _NATIVE_INT_MIN or maximum > _NATIVE_INT_MAX \
            or not isinstance(offsets, (tuple, list)) or \
            any(type(offset) is not int for offset in offsets) or \
            tuple(offsets) != tuple(sorted(set(offsets))) or \
            minimum != min(offsets, default=0) or maximum != max(offsets, default=0):
        raise ValueError("user reconstruction stencil/order contract is invalid")
    depth = max(1, 1 - minimum, maximum + 1)
    if depth > _NATIVE_INT_MAX or maximum - minimum + 1 > _NATIVE_INT_MAX:
        raise ValueError("user reconstruction stencil cannot fit native int32 ghost/count metadata")
    if options["ghost_depth"] != depth:
        raise ValueError("user reconstruction ghost contract differs from its stencil")
    if value.requirements != {"ghost_depth": depth, "source_compiled": True} or \
            value.capabilities != {"componentwise_scalar": True, "oriented_sample": True}:
        raise ValueError("user reconstruction source capabilities changed after authoring")
    observed_body = canonical_hash_data(value.expression, where="user reconstruction body")
    # Case.freeze recursively replaces mutable lists/dicts with immutable
    # counterparts. Compare their canonical JSON values, not container types.
    if observed_body != canonical_hash_data(options["body"],
                                             where="frozen user reconstruction body"):
        raise ValueError("user reconstruction body changed after authoring")
    if _retained_offsets(value.expression) != tuple(offsets):
        raise ValueError("user reconstruction stencil differs from its body")
    expected = _source_identity(_identity_data(
        body_data=observed_body, order=order, offsets=tuple(offsets),
        minimum=minimum, maximum=maximum))
    if expected != options["source_identity"]:
        raise ValueError("user reconstruction source identity changed after authoring")
    return value


__all__ = ["User", "authenticated_user_reconstruction"]
