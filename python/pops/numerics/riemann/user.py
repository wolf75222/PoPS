"""Source-authored conservative face flux, compiled once into the native package."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pops._ir.expr import Const, Expr, Var
from pops._ir.visitors import _children
from pops._ir.vector_expr import VectorExpr
from pops.descriptors import BrickDescriptor
from pops.model import StateHandle
from pops.physics.board_handles import StateHandle as BoardStateHandle
from pops.model.hash_data import canonical_hash_data


_NATIVE_ID = "pops.generated.riemann.face/v2"
_SCHEME = "source_face"
_CAPABILITIES = ("physical_flux", "provider_pack", "stability_bound")


def _state_width(state: Any) -> int:
    if not isinstance(state, (StateHandle, BoardStateHandle)):
        raise TypeError("riemann.User(state=) requires an exact model StateHandle")
    return len(state.components if isinstance(state, BoardStateHandle) else state.space.components)


def _variables(width: int) -> dict[str, VectorExpr]:
    return {kind: VectorExpr(Var("pops_face_%s_%d" % (kind, index), "face_input")
                             for index in range(width))
            for kind in ("left", "right", "flux_left", "flux_right")}


def _capture_identities(roots: tuple[Expr, ...]) -> tuple[tuple[str, str], ...]:
    from pops._ir.values import RuntimeParamRef
    from pops.model import ParamHandle

    found: dict[str, str] = {}
    pending, seen = list(roots), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, RuntimeParamRef):
            handle = node.handle
            if not isinstance(handle, ParamHandle) or handle.param_kind != "runtime":
                raise TypeError("user face runtime read needs model.value(RuntimeParam handle)")
            qid = handle.qualified_id
            prior = found.setdefault(node.name, qid)
            if prior != qid:
                raise ValueError("user face has colliding runtime parameter names")
        pending.extend(_children(node))
    return tuple(sorted(found.items()))


def _check_body(roots: tuple[Expr, ...], variables: dict[str, VectorExpr],
                speed: Var, *, exact_objects: bool) -> tuple[tuple[str, str], ...]:
    from pops._ir.native_call import NativeCall
    from pops._ir.values import RuntimeParamRef

    allowed = {id(node) for row in variables.values() for node in row} | {id(speed)}
    expected_names = {node.name for row in variables.values() for node in row} | {speed.name}
    pending, seen = list(roots), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, NativeCall):
            raise NotImplementedError("user face native calls need a declared fallible status contract")
        if isinstance(node, RuntimeParamRef):
            continue
        if isinstance(node, Var):
            if node.kind != "face_input" or node.name not in expected_names or \
                    (exact_objects and id(node) not in allowed):
                raise ValueError("user face body contains a foreign variable")
        children = _children(node)
        if not children and not isinstance(node, (Const, Var)):
            raise ValueError("user face body has an unsupported scalar leaf")
        pending.extend(children)
    return _capture_identities(roots)


def _source_identity(body: Any, stability: Any, *, width: int, state_name: str,
                     captures: tuple[tuple[str, str], ...]) -> str:
    data = {"schema_version": 2, "kind": _SCHEME, "body": body,
            "stability": stability,
            "width": width, "state": state_name, "runtime_captures": captures}
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def User(body: Any, *, state: StateHandle, stability: Any = None,
         name: str = "user_face") -> BrickDescriptor:
    """Author ``face(left,right,flux_left,flux_right,speed) -> tuple[Expr,...]``.

    Every state/flux argument is an indexable tuple with the declared state width, including
    width one. ``speed`` is the physical normal spectral-radius majorant. Python runs only now;
    the frozen expression executes inside the native face kernel. ``stability`` has the
    same arguments and returns the author's finite, nonnegative numerical face bound.
    """
    if not callable(body):
        raise TypeError("riemann.User(body=) requires a callable symbolic face body")
    if not callable(stability):
        raise TypeError("riemann.User(body=) requires a callable stability= numerical bound")
    width = _state_width(state)
    if width < 1:
        raise ValueError("riemann.User state has no components")
    if not isinstance(name, str) or not name:
        raise ValueError("riemann.User name must be non-empty text")
    variables = _variables(width)
    speed = Var("pops_face_speed", "face_input")
    result = body(variables["left"], variables["right"], variables["flux_left"],
                  variables["flux_right"], speed)
    if not isinstance(result, tuple) or len(result) != width:
        raise TypeError("riemann.User body must return a tuple of one Expr per state component")
    roots = tuple(item if isinstance(item, Expr) else Const(item) for item in result)
    stability_result = stability(variables["left"], variables["right"],
                                 variables["flux_left"], variables["flux_right"], speed)
    if isinstance(stability_result, (tuple, list, VectorExpr)):
        raise TypeError("riemann.User stability must return one scalar Expr")
    stability_root = (stability_result if isinstance(stability_result, Expr)
                      else Const(stability_result))
    captures = _check_body(roots + (stability_root,), variables, speed, exact_objects=True)
    body_data = canonical_hash_data(roots, where="user face body")
    stability_data = canonical_hash_data(stability_root, where="user face stability")
    # The declaration fingerprint and block owner path become canonical only
    # after Case.freeze/resolve.  Keep the stable local state name in the source
    # identity; the exact resolved StateHandle is checked against FiniteVolume
    # variables when installing the native policy.
    state_name = state.inspect()["local_id"]
    identity = _source_identity(body_data, stability_data, width=width, state_name=state_name,
                                captures=captures)
    return BrickDescriptor(
        name, "generated", category="riemann", native_id=_NATIVE_ID, scheme=_SCHEME,
        options={"state": state, "state_name": state_name, "width": width,
                 "runtime_captures": captures, "source_identity": identity,
                 "body": body_data, "stability": stability_data},
        requirements={"capabilities": _CAPABILITIES, "source_compiled": True},
        capabilities={"vector_face": True, "conservative_shared_face": True},
        expression=roots + (stability_root,),
    )


def authenticated_user_face(value: Any) -> BrickDescriptor:
    if type(value) is not BrickDescriptor or value.category != "riemann" or \
            value.brick_type != "generated" or value.native_id != _NATIVE_ID or \
            value.scheme != _SCHEME:
        raise TypeError("user face requires an exact source-authored descriptor")
    options = value.options
    if set(options) != {"state", "state_name", "width", "runtime_captures",
                        "source_identity", "body", "stability"}:
        raise ValueError("user face source contract is incomplete")
    state, width = options["state"], options["width"]
    if type(width) is not int or width != _state_width(state) or width < 1 or \
            state.inspect()["local_id"] != options["state_name"]:
        raise ValueError("user face state/shape authority changed")
    roots = value.expression
    if not isinstance(roots, tuple) or len(roots) != width + 1 or \
            any(not isinstance(item, Expr) for item in roots):
        raise ValueError("user face body shape changed")
    variables = _variables(width)
    speed = Var("pops_face_speed", "face_input")
    captures = _check_body(roots, variables, speed, exact_objects=False)
    if tuple(tuple(item) for item in options["runtime_captures"]) != captures:
        raise ValueError("user face runtime captures changed")
    body_data = canonical_hash_data(roots[:width], where="user face body")
    if body_data != \
            canonical_hash_data(options["body"], where="frozen user face body"):
        raise ValueError("user face body changed after authoring")
    stability_data = canonical_hash_data(roots[width], where="user face stability")
    if stability_data != canonical_hash_data(options["stability"],
                                             where="frozen user face stability"):
        raise ValueError("user face stability changed after authoring")
    expected = _source_identity(body_data, stability_data,
                                width=width, state_name=options["state_name"],
                                captures=captures)
    if expected != options["source_identity"]:
        raise ValueError("user face source identity changed after authoring")
    if value.requirements != {"capabilities": _CAPABILITIES, "source_compiled": True} or \
            value.capabilities != {"vector_face": True, "conservative_shared_face": True}:
        raise ValueError("user face capabilities changed after authoring")
    return value


__all__ = ["User", "authenticated_user_face"]
