"""Exact same-invocation reuse for pure physical field tuple solves.

The cache is an emission-local alias proof. Runtime solved buffers never survive as
reuse inputs across a new step, restart, or layout generation. One exception crosses a
rejected attempt (spec 0.3 T7, N8 lot 6): the solve of a fixed trial site of a traced
method, whose dependency set excludes the proposal, keeps its solution for the later
attempts of the same accepted step (``program_emit_solve._attempt_reuse``); such an
attempt counts a reuse, not a solve.

Across the stages of one invocation (spec 0.3 T7, N8 lot « reuse across stages »), a request of a
traced method whose data do not depend on the time at which it is solved is a time-independent
request (``_time_independent_request``). It reads and writes a cache of its own,
``traced_field_reuse_cache``, whose key leaves out the point and names each input by its root
through structural copies (``structural_copy``). Between two such requests, every top-level
operation must have region 0 and be on ``_SAFE`` or on ``_STAGE_RATES``, the rates of a stage,
which write no valid cell of a state and no solved buffer. So the request of a later stage at a
copy of ``B.state`` aliases the solve of an earlier stage. Every other request keeps the cache
with the point, ``field_reuse_cache``, and its barriers, unchanged. A b6 Program carries no trial
site, so it never reaches the traced cache. The selected profiles, which are traced, send their
stage-0 request there; no other request shares its key, so their C++ is b6's (the parity tests).
The traced cache does not cross ``local_transform``, which the b6 cache crosses with its versions.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pops.time._program.serialization import _json_ready
from pops.identity import canonical_bytes

# Every omitted operation is an effect barrier. In particular in-place projection,
# field publication, callback evaluation and control flow cannot preserve this proof.
_SAFE = frozenset({"state", "scalar_field", "linear_combine", "matrix_free_operator",
    "field_problem_load", "field_problem_coefficients", "field_component", "field_gradient", "solve_linear",
    "solve_outcome", "solve_outcome_component", "reduce", "record_scalar", "scalar_op", "compare"})
# The cache of the time-independent requests of traced methods, and the point of its keys.
_TRACED_CACHE = ("traced_field_reuse_cache",)
_TIMELESS = "time-independent"
# The rates of a stage that the traced cache crosses besides _SAFE, with everything each one
# writes (design note of the lot, section 2.6). None writes the valid cells of a state or a solved
# buffer, which are all that a time-independent entry reads. A later lot adds an operation here
# with its write set and a test; every operation that is not listed is a barrier.
_STAGE_RATES = frozenset({
    # rate and named-flux scratches, the ghost cells of its state, the auxiliary provider storage
    # and the stage time; a path-conservative rhs (M15, M17) stays a barrier
    "rhs",
    # the block's auxiliary provider storage and the stage time; it reads the solved buffer
    "field_publication",
})
# The third rate of a stage, a stencil flux (``validate_stencil_flux``), writes its rate scratch,
# the ghost cells of its state, its own prepared inputs and the stage time (``_crosses``).
# The closed grammar of the data of a time-independent request: the literal kinds of
# ``decode_expression_literal`` and the arithmetic of ``encode_field_expression``.
_LITERALS = frozenset({"integer", "rational", "decimal", "binary64"})
_BINARY = frozenset({"add", "sub", "mul", "div", "pow"})
_UNARY = frozenset({"neg", "sqrt", "abs"})


def observe_operation(value: Any, var: Any, *, model: Any = None) -> None:
    if value.op == "local_transform" and value.region == 0:
        # This existing operation writes fresh state storage and consumes its domain
        # guard before the next operation. Only direct copies of exact native input
        # coordinates preserve component versions; opaque arithmetic never does.
        from pops.codegen.program_emit_kernels import _model_impl
        from pops._ir.expr import Var
        impl = _model_impl(model) if model is not None else None
        declaration = getattr(impl, "_local_transforms", {}).get(value.attrs["transform"])
        if impl is not None and declaration is not None and len(value.inputs) == 1:
            expressions = declaration["expressions"]
            names = tuple(impl.cons_names)
            if len(expressions) == len(names):
                versions = var.setdefault(("field_component_versions",), {})
                source = value.inputs[0]
                for component, expression in enumerate(expressions):
                    if type(expression) is Var and expression.kind == "cons" \
                            and expression.name == names[component]:
                        versions[(value.id, component)] = versions.get(
                            (source.id, component), (source.id, component))
                return
    if value.op not in _SAFE or value.region != 0:
        var[("field_reuse_cache",)] = {}


def _key(value: Any, var: Any, *, target: str) -> tuple[Any, ...] | None:
    if target != "system" or value.region != 0 or value.op != "solve_linear":
        return None
    rhs = value.inputs[1]
    request = value.attrs.get("solve_request")
    if rhs.op != "field_problem_load" or request is None:
        return None
    from pops.time._program.solve_request import validate_solve_request_node
    validate_solve_request_node(value.prog, value)
    from pops.fields._program_problem import validate_field_apply
    apply = value.inputs[0].attrs["apply_result"]
    validate_field_apply(apply)
    coefficient = apply.inputs[2]
    versions = var.get(("field_component_versions",), {})

    def frozen_expression(expression: Any, inputs: Any) -> Any:
        if expression[0] == "input":
            _, ordinal, component, identity = expression
            source = inputs[ordinal]
            return ("input", _json_ready(identity), component,
                    versions.get((source.id, component), (source.id, component)))
        if expression[0] == "literal":
            return _json_ready(expression)
        return (expression[0], *(frozen_expression(child, inputs) for child in expression[1:]))

    def expressions(node: Any) -> bytes:
        return canonical_bytes(tuple(frozen_expression(row, node.inputs)
                                     for row in node.attrs["expressions"]))

    field = _json_ready(rhs.attrs["field_handle"])
    return (field["qualified_id"], rhs.attrs["field_problem_identity"], value.point,
            expressions(rhs), expressions(coefficient),
            canonical_bytes(_json_ready(apply.attrs["reaction"])),
            apply.attrs["physical_boundary"], request["solver_identity"],
            request["initialization_identity"])


def structural_copy(value: Any) -> Any:
    """The input of ``value`` when ``value`` is a structural copy of it, else None: a
    ``linear_combine`` of region 0 with one input of its block, state reference and value type,
    whose one coefficient is the polynomial 1, exponent 0 of ``p.dt`` and the integer 1 or
    ``Fraction(1)`` (no binary64 weight)."""
    from pops.time.value_metadata import CoeffPolynomial

    if value.op != "linear_combine" or value.region != 0 or len(value.inputs) != 1:
        return None
    coeffs = value.attrs.get("coeffs")
    if not isinstance(coeffs, tuple) or len(coeffs) != 1 or type(coeffs[0]) is not CoeffPolynomial:
        return None
    items = tuple(coeffs[0].items())
    if len(items) != 1 or items[0][0] != 0 or type(items[0][1]) not in (int, Fraction) \
            or items[0][1] != 1:
        return None
    (source,) = value.inputs
    if source.block != value.block or source.vtype != value.vtype \
            or getattr(source, "state_ref", None) != getattr(value, "state_ref", None):
        return None
    return source


def _root(value: Any) -> Any:
    """``value`` once every structural copy is followed back to its source."""
    source = structural_copy(value)
    while source is not None:
        value, source = source, structural_copy(source)
    return value


def _closed(expression: Any, inputs: Any) -> bool:
    """Whether a row of a field load or coefficient is in the closed grammar of a
    time-independent request: inputs whose root is the ``state`` read, constant literals and
    arithmetic. Every other kind is refused, ``temporal_tau`` and ``unknown`` included."""
    kind = expression[0]
    if kind == "input":
        return len(expression) == 4 and _root(inputs[expression[1]]).op == "state"
    if kind == "literal":
        return len(expression) == 2 and isinstance(expression[1], Mapping) \
            and expression[1].get("kind") in _LITERALS
    if kind in _BINARY:
        return len(expression) == 3 and all(_closed(item, inputs) for item in expression[1:])
    if kind in _UNARY:
        return len(expression) == 2 and _closed(expression[1], inputs)
    return False


def _time_independent_request(value: Any, *, target: str) -> bool:
    """Whether the solve ``value`` is a time-independent request (design note of the lot, section
    2.4). No attribute is added: each condition is computed here from the nodes, at emission.

    The preconditions of the b6 key hold (a uniform system solve of a field load, region 0), and
    (i) the outcome that consumes the solve carries a trial site: a request of a traced method;
    (ii) every row of its load and coefficients is in the closed grammar (``_closed``);
    (iii) its boundary is one that the native field route admits, which carries no data;
    (iv) its reaction is literal;
    (v) it has no initial guess, so the solve starts from zero.
    Such a request reads ``B.state`` with coefficient 1, a fixed site (T38): a retryable one is
    refused as a defect of the elaboration.
    """
    if target != "system" or value.region != 0 or value.op != "solve_linear":
        return False
    load = value.inputs[1]
    request = value.attrs.get("solve_request")
    if load.op != "field_problem_load" or request is None:
        return False
    from pops.codegen.program_emit_solve import _trial_site
    site = _trial_site(value.prog, value)
    if site is None:
        return False
    apply = value.inputs[0].attrs["apply_result"]
    coefficient = apply.inputs[2]
    if not all(_closed(row, node.inputs) for node in (load, coefficient)
               for row in node.attrs["expressions"]):
        return False
    # (iii): a boundary that the native field route admits carries no data (periodic or
    # homogeneous); the route's own validator decides it, so that this module names no law.
    from pops.fields._program_problem import validate_field_apply
    from pops.fields.problem import FieldProblemError
    try:
        validate_field_apply(apply)
    except (FieldProblemError, TypeError, ValueError):
        # The emission of the apply refuses it with the same validator; the b6 path reports it.
        return False
    reaction = apply.attrs.get("reaction")
    if not isinstance(reaction, tuple) or not all(
            isinstance(item, Mapping) and item.get("kind") in _LITERALS for item in reaction):
        return False
    if value.attrs.get("has_guess") or request.get("seed") is not None:
        return False
    if site["retry_can_change"]:
        raise ValueError("the time-independent field request %r is a retryable trial site; a "
                         "request at B.state is a fixed site (T38)" % site["site"])
    return True


def _crosses(value: Any) -> bool:
    """Whether a time-independent entry stays valid across the top-level operation ``value``."""
    if value.region != 0:
        return False
    if value.op == "rhs":
        return not value.attrs.get("path_conservative", False)
    if value.op in _SAFE or value.op in _STAGE_RATES:
        return True
    from pops.time._program.stencil_flux import validate_stencil_flux
    try:
        validate_stencil_flux(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _no_barrier(program: Any, first: Any, second: Any) -> bool:
    """Whether ``first`` and ``second`` are top-level operations of ``program``, in this order,
    and every top-level operation between them crosses (``_crosses``). The proof reads the IR, not
    the emission: some operations never reach ``observe_operation`` (rhs groups, structured
    control, and the families that ``_emit_op_impl`` lowers before it)."""
    values = program._values
    positions = [index for index, item in enumerate(values) if item is first or item is second]
    if len(positions) != 2 or values[positions[0]] is not first:
        return False
    return all(_crosses(item) for item in values[positions[0] + 1:positions[1]])


def _traced_key(value: Any, var: Any, *, target: str) -> tuple[Any, ...]:
    """The key of a time-independent request: b6's, checked by ``_key``, with the point replaced
    by ``_TIMELESS`` and each input named by its root and component."""
    key = _key(value, var, target=target)
    if key is None:  # a time-independent request has b6's key (review P6)
        raise ValueError("a time-independent field request has no b6 key")
    load = value.inputs[1]
    coefficient = value.inputs[0].attrs["apply_result"].inputs[2]

    def frozen_expression(expression: Any, inputs: Any) -> Any:
        if expression[0] == "input":
            _, ordinal, component, identity = expression
            return ("input", _json_ready(identity), component,
                    (_root(inputs[ordinal]).id, component))
        if expression[0] == "literal":
            return _json_ready(expression)
        return (expression[0], *(frozen_expression(child, inputs) for child in expression[1:]))

    def expressions(node: Any) -> bytes:
        return canonical_bytes(tuple(frozen_expression(row, node.inputs)
                                     for row in node.attrs["expressions"]))

    return key[:2] + (_TIMELESS, expressions(load), expressions(coefficient)) + key[5:]


def _counters(value: Any, var: Any, lines: Any) -> tuple[str, str, str]:
    field = _json_ready(value.inputs[1].attrs["field_handle"])["qualified_id"]
    counters = var.setdefault(("field_counters",), {})
    if field not in counters:
        index = len(counters)
        owner = "field_counters_%d" % index
        # Map continuations run after their creating stack frame has returned. This
        # diagnostic effect belongs to one invocation, so capture its owning Host
        # storage by value and mutate the common payload, rather than a scalar copy
        # or a reference to the vanished frame. Prepare all owners at invocation
        # entry, before any solve callback or suspended transport. Each allocation
        # joins the existing execution-lane failure vote; retries allocate zeros.
        var.setdefault(("field_counter_preparation",), []).extend((
            "std::shared_ptr<std::array<pops::Real, 2>> %s;" % owner,
            "std::exception_ptr %s_error;" % owner,
            "try { %s = std::make_shared<std::array<pops::Real, 2>>(" % owner
            + "std::array<pops::Real, 2>{pops::Real(0), pops::Real(0)}); }",
            "catch (...) { %s_error = std::current_exception(); }" % owner,
            "pops::collectively_rethrow_exception(%s_error, ctx.prepared_execution_lane(), " % owner
            + '"Program field diagnostic preparation");',
        ))
        solve, reuse = "(*%s)[0]" % owner, "(*%s)[1]" % owner
        counters[field] = (solve, reuse)
    solve, reuse = counters[field]
    return field, solve, reuse


def _record(field: str, solve: str, reuse: str, lines: Any) -> None:
    lines.append("ctx.record_scalar(%s, %s);" % (json.dumps("field.solves/" + field), solve))
    lines.append("ctx.record_scalar(%s, %s);" % (json.dumps("field.reuses/" + field), reuse))


def reuse_field_solve(value: Any, var: Any, lines: Any, *, target: str) -> bool:
    if _time_independent_request(value, target=target):
        entry = var.setdefault(_TRACED_CACHE, {}).get(_traced_key(value, var, target=target))
        if entry is None or not _no_barrier(value.prog, entry[1], value):
            return False
        prior = entry[0]
    else:
        key = _key(value, var, target=target)
        if key is None:
            return False
        cache = var.setdefault(("field_reuse_cache",), {})
        prior = cache.get(key)
        if prior is None:
            return False
    field, solves, reuses = _counters(value, var, lines)
    # The earlier emitted solve guard dominates this alias, so only an explicitly
    # accepted solution can reach it. Failed candidates are never readable here.
    var[value.id] = prior
    lines.append("++%s;" % reuses)
    _record(field, solves, reuses, lines)
    return True


def publish_field_solve(value: Any, var: Any, lines: Any, *, target: str,
                        reused: str | None = None) -> None:
    """Count the solve of ``value`` and publish its alias. ``reused`` is the flag of a solve that
    a retry of its step keeps (T7 across attempts): that attempt counts a reuse."""
    timeless = _time_independent_request(value, target=target)
    key = (_traced_key if timeless else _key)(value, var, target=target)
    if key is None:
        return
    field, solves, reuses = _counters(value, var, lines)
    if reused:
        lines.append("if (%s) ++%s; else ++%s;" % (reused, reuses, solves))
    else:
        lines.append("++%s;" % solves)
    _record(field, solves, reuses, lines)
    if timeless:
        var.setdefault(_TRACED_CACHE, {})[key] = (var[value.id], value)
    else:
        var.setdefault(("field_reuse_cache",), {})[key] = var[value.id]
