"""Source-authored conservative face flux, compiled once into the native package."""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

from pops.model.scalar_contract import (
    Const, Expr, Var, VectorExpr, NativeCall, RuntimeParamRef,
    scalar_children as _children, scalar_expression, state_component_count,
)
from pops.descriptors import BrickDescriptor
from pops.model import StateHandle
from pops.model.hash_data import canonical_hash_data


_NATIVE_ID = "pops.generated.riemann.face/v2"
_SCHEME = "source_face"
_CAPABILITIES = ("physical_flux", "provider_pack", "stability_bound")
# The model capability of a face that reads the signed wave speeds (R9).
_WAVE_SPEEDS = "wave_speeds"
# The five positional face inputs, in order, then the names of the face inputs passed by name.
_POSITIONAL_INPUTS = ("the left trace uL", "the right trace uR", "the left flux fL",
                      "the right flux fR", "the speed s")
_NAMED_INPUTS = ("n", "speeds")


def _state_width(state: Any) -> int:
    try:
        return state_component_count(state)
    except TypeError as error:
        raise TypeError("riemann.User(state=) requires an exact model StateHandle") from error


def _variables(width: int) -> dict[str, VectorExpr]:
    return {kind: VectorExpr(Var("pops_face_%s_%d" % (kind, index), "face_input")
                             for index in range(width))
            for kind in ("left", "right", "flux_left", "flux_right")}


def _named_inputs() -> dict[str, VectorExpr]:
    """The face inputs that a body or a stability bound receives only when it names them.

    ``n`` is the canonical normal of the face, with the three components of the native face
    context (1 on the axis of the face, 0 on the others). ``speeds`` is the signed interval
    (sL, sR): the slowest and the fastest wave speeds that the model declares along n (R9),
    united over the two traces.
    """
    return {"n": VectorExpr(Var("pops_face_normal_%d" % axis, "face_input") for axis in range(3)),
            "speeds": VectorExpr(Var("pops_face_%s_speed" % bound, "face_input")
                                 for bound in ("slowest", "fastest"))}


def _requested_inputs(function: Any) -> tuple[str, ...]:
    """The named face inputs that ``function`` takes as parameters after the five face inputs.

    One of the five positional face inputs named ``n`` or ``speeds`` is refused: it would receive
    a trace, a flux or the speed ``s``, not the input of that name.
    """
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):  # no inspectable signature: the five face inputs only
        return ()
    names, positional = [], 0
    for parameter in parameters:
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
            if positional <= len(_POSITIONAL_INPUTS):
                if parameter.name in _NAMED_INPUTS:
                    raise TypeError(
                        "riemann.User: face input %d is %s, not the input %r; take n and speeds "
                        "as parameters after the five face inputs (uL, uR, fL, fR, s)"
                        % (positional, _POSITIONAL_INPUTS[positional - 1], parameter.name))
                continue
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY) and \
                parameter.name in _NAMED_INPUTS:
            names.append(parameter.name)
    return tuple(names)


def _evaluate(function: Any, variables: dict[str, VectorExpr], speed: Var,
              named: dict[str, VectorExpr]) -> Any:
    return function(variables["left"], variables["right"], variables["flux_left"],
                    variables["flux_right"], speed,
                    **{name: named[name] for name in _requested_inputs(function)})


def _requirements(roots: tuple[Expr, ...]) -> dict[str, Any]:
    """The model capabilities of a face; the signed wave speeds only when it reads them."""
    speeds = {node.name for node in _named_inputs()["speeds"]}
    pending, seen, reads = list(roots), set(), False
    while pending and not reads:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        reads = isinstance(node, Var) and node.kind == "face_input" and node.name in speeds
        pending.extend(_children(node))
    return {"capabilities": _CAPABILITIES + ((_WAVE_SPEEDS,) if reads else ()),
            "source_compiled": True}


def _capture_identities(roots: tuple[Expr, ...]) -> tuple[tuple[str, str], ...]:
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
                speed: Var, named: dict[str, VectorExpr], *,
                exact_objects: bool) -> tuple[tuple[str, str], ...]:

    inputs = (*variables.values(), *named.values())
    allowed = {id(node) for row in inputs for node in row} | {id(speed)}
    expected_names = {node.name for row in inputs for node in row} | {speed.name}
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

    Two more face inputs are passed by name to a body or a bound that takes a parameter of that
    name after the five above: ``n``, the canonical normal of the face (three components, 1 on
    the axis of the face), and ``speeds``, the pair (sL, sR) of the slowest and the fastest wave
    speeds that the model declares along n (R9), united over the two traces. The kernel
    evaluates an input only when an expression reads it. A face that reads ``speeds`` requires
    the model capability ``wave_speeds``, and a case whose model declares none is refused when
    it is validated, before any code is generated.
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
    variables, named = _variables(width), _named_inputs()
    speed = Var("pops_face_speed", "face_input")
    result = _evaluate(body, variables, speed, named)
    if not isinstance(result, tuple) or len(result) != width:
        raise TypeError("riemann.User body must return a tuple of one Expr per state component")
    roots = tuple(scalar_expression(item) for item in result)
    stability_result = _evaluate(stability, variables, speed, named)
    if isinstance(stability_result, (tuple, list, VectorExpr)):
        raise TypeError("riemann.User stability must return one scalar Expr")
    stability_root = scalar_expression(stability_result)
    captures = _check_body(roots + (stability_root,), variables, speed, named,
                           exact_objects=True)
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
        requirements=_requirements(roots + (stability_root,)),
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
    captures = _check_body(roots, variables, speed, _named_inputs(), exact_objects=False)
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
    if value.requirements != _requirements(roots) or \
            value.capabilities != {"vector_face": True, "conservative_shared_face": True}:
        raise ValueError("user face capabilities changed after authoring")
    return value


def reads_signed_wave_speeds(value: Any) -> bool:
    """Whether an authenticated face reads the signed wave speeds (sL, sR) of its model."""
    return _WAVE_SPEEDS in authenticated_user_face(value).requirements["capabilities"]


def require_declared_wave_speeds(value: Any, model: Any, *, where: str) -> None:
    """Refuse a face that reads the signed wave speeds of a model that declares none (R9).

    ``model`` is the authoring Model of the case or the model a package is lowered from; the
    signed pair is declared by ``Model.wave_speeds(...)``, ``wave_speeds_from_jacobian()`` or the
    pressure-derived eigenvalues. Without it the native face has no speeds to give.
    """
    if not reads_signed_wave_speeds(value):
        return
    from .waves import provider_of

    provider = provider_of(model)
    if provider is None or not provider.signed_pair:
        raise ValueError(
            "%s: riemann.User face %r reads the signed wave speeds (speeds=), but model %r "
            "declares none; declare them with Model.wave_speeds(flux, frame=..., values=...) or "
            "Model.wave_speeds_from_jacobian(), or write a face that does not read them"
            % (where, value.name, getattr(model, "name", "?")))


__all__ = ["User", "authenticated_user_face", "reads_signed_wave_speeds",
           "require_declared_wave_speeds"]
