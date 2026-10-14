"""Lower typed products and bounded regions through genuine PoPS field/context operations."""
from __future__ import annotations

import json

from pops.time._program.structured_control import value_signature
from .program_emit_finite_arrays import FINITE_OPS, finite_cpp_type


STRUCTURED_OPS = frozenset((
    "product", "product_get", "region_argument", "region_index", "int64_constant",
    "int64_cast", "bool_constant", "int64_op", "int64_compare", "structured_fold",
    "structured_iterate", "iteration_success", "iteration_is_ok", "iteration_payload",
    "result_match",
    "result_ok", "result_error", "capture_result", "result_success", "result_is_ok",
    "result_payload",
    "variant_construct", "variant_is", "variant_payload", "variant_unreachable",
    "variant_require_success",
))

# A checked execution-effect profile, not a model/solver catalog. These operations have either
# no recoverable failure or lane-uniform native status handling. Extend only after checking the
# emitter's failure rendezvous, never simply because an operation is numerically useful.
from pops.time._program.versioned_values import VERSIONED_OPS
from pops.time._program.distributed_arrays import GLOBAL_OPS as _DISTRIBUTED, local_effects
_REGION_SAFE_OPS = STRUCTURED_OPS | FINITE_OPS | (VERSIONED_OPS - {'continuation_open', 'continuation_publish'}) | frozenset((
    "state", "linear_combine", "pointwise_expression", "scalar_op", "compare", "reduce",
    "cfl", "hmin", "branch", "scalar_field", "where", "cell_compare",
    "field_component", "vector_field", "requested_dt",
    "scalar_isfinite", "scalar_expression", "provenance_bind",
))


def _check_region_effects(block):
    for value in block:
        if value.op in _DISTRIBUTED:
            local_effects(value.attrs.get("body_block", ()))
            continue
        if value.op == "laplacian" and value.inputs and "spatial_port_storage" in value.inputs[0].attrs:
            from pops.time._program.spatial_interaction import validate_computed_laplacian
            validate_computed_laplacian(value)
        elif value.op not in _REGION_SAFE_OPS:
            raise NotImplementedError(
                "structured region operation %r has no checked lane-uniform failure contract"
                % value.op)
        for key in ("cond_block", "body_block", "true_block", "false_block"):
            _check_region_effects(value.attrs.get(key, ()))


def emit_region_laplacian(value, var, lines, *, owner, target):
    """Pure Uniform stencil with authenticated pre-halo failures and finite result consensus.

    The native operation fills only scratch ghosts and writes provisional scratch. No accepted
    exchange, history, or AMR flux-expression sidecar is manufactured or silently discarded.
    """
    from pops.time._program.spatial_interaction import validate_computed_laplacian
    output, source = validate_computed_laplacian(value)
    if target != "system":
        raise NotImplementedError("region computed Laplacian requires the qualified Uniform periodic route")
    prefix = "region_laplacian_%d" % value.id
    out, inp = var[output.id], var[source.id]
    lines += ["{", "auto& %s_out = %s;" % (prefix, out),
              "auto& %s_in = %s;" % (prefix, inp),
              "const auto& %s_lane = ctx.prepared_execution_lane();" % prefix,
              "std::exception_ptr %s_error;" % prefix,
              "try {", "  ctx.require_cartesian_generated_operator(%d, \"region computed Laplacian\");" % owner,
              "  if (&%s_out == &%s_in || %s_in.ncomp() < 1 ||" % (prefix, prefix, prefix),
              "      %s_out.ncomp() != %s_in.ncomp() || %s_out.layout() != %s_in.layout() ||" % (prefix, prefix, prefix, prefix),
              "      %s_out.distribution() != %s_in.distribution() ||" % (prefix, prefix),
              "      %s_out.local_rank() != %s_in.local_rank() || %s_out.local_size() != %s_in.local_size())" % (prefix, prefix, prefix, prefix),
              '    throw std::invalid_argument("region Laplacian requires distinct, identical componentwise stencil storage");',
              "  for (int axis = 0; axis < pops::kNativeDimension; ++axis)",
              "    if (%s_in.ghosts()[axis] < 1)" % prefix,
              '      throw std::invalid_argument("region Laplacian requires one input ghost per axis");',
              "} catch (...) { %s_error = std::current_exception(); }" % prefix,
              'pops::collectively_rethrow_exception(%s_error, %s_lane, "region Laplacian preflight");' % (prefix, prefix),
              "auto %s_boundary = ctx.prepare_mesh_boundary_session(%s_in, %s_lane);" % (prefix, prefix, prefix),
              "try {",
              "  for (int axis = 0; axis < pops::kNativeDimension; ++axis)",
              "    for (auto side : {pops::BoundarySide::lower, pops::BoundarySide::upper})",
              "      if (!%s_boundary->topology().is_periodic(pops::Face<pops::kNativeDimension>{axis, side}))" % prefix,
              '        throw std::invalid_argument("region Laplacian requires explicit periodic boundaries; physical boundary port not implemented");',
              "} catch (...) { %s_error = std::current_exception(); }" % prefix,
              'pops::collectively_rethrow_exception(%s_error, %s_lane, "region Laplacian boundary contract");' % (prefix, prefix),
              "try { ctx.laplacian(%s_out, %s_in, *%s_boundary); }" % (prefix, prefix, prefix),
              "catch (...) { %s_error = std::current_exception(); }" % prefix,
              "try { Kokkos::fence(); } catch (...) { if (!%s_error) %s_error = std::current_exception(); }" % (prefix, prefix),
              'pops::collectively_rethrow_exception(%s_error, %s_lane, "region Laplacian evaluation");' % (prefix, prefix),
              "pops::Real %s_invalid = 0;" % prefix, "try {",
              "  for (std::size_t local = 0; local < %s_out.local_size(); ++local) {" % prefix,
              "    const auto values = std::as_const(%s_out).fab(local).view();" % prefix,
              "    const int components = %s_out.ncomp();" % prefix,
              "    const auto bad = pops::for_each_cell_reduce_max(%s_out.box(local), [=] POPS_HD(const pops::Index<pops::kNativeDimension>& cell) {" % prefix,
              "      for (int component = 0; component < components; ++component)",
              "        if (!Kokkos::isfinite(values(cell, component))) return pops::Real(1);",
              "      return pops::Real(0);", "    });",
              "    %s_invalid = std::max(%s_invalid, bad);" % (prefix, prefix), "  }",
              "} catch (...) { %s_error = std::current_exception(); }" % prefix,
              "try { Kokkos::fence(); } catch (...) { if (!%s_error) %s_error = std::current_exception(); }" % (prefix, prefix),
              'pops::collectively_rethrow_exception(%s_error, %s_lane, "region Laplacian finite scan");' % (prefix, prefix),
              "if (pops::all_reduce_max(%s_invalid, %s_lane) != pops::Real(0))" % (prefix, prefix),
              '  throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::kInvalidEvaluation, "region_laplacian", "nonfinite provisional stencil output");', "}"]
    var[value.id] = var[output.id]


def _leaves(signature, tokens, path=()):
    if signature["vtype"] == "product":
        for key, child in signature["members"]:
            yield from _leaves(child, tokens[key] if tokens is not None else None, path + (key,))
    else:
        yield path, signature, tokens


def _repack(signature, leaves, path=()):
    if signature["vtype"] == "product":
        return {key: _repack(child, leaves, path + (key,))
                for key, child in signature["members"]}
    return leaves[path]


def _cpp_type(kind):
    return {"scalar": "pops::Real", "int64": "std::int64_t", "bool": "bool"}[kind]


class _DeferredField(str):
    """A field slot whose geometry is allocated only when its sum arm becomes active."""
    node_id: int
    slot: int

    def __new__(cls, token: str, node_id: int, slot: int):
        value = super().__new__(cls, token)
        value.node_id, value.slot = node_id, slot
        return value


def _storage_width(signature) -> int:
    if signature["vtype"] in ("product", "variant"):
        return sum(_storage_width(child) for _, child in signature["members"])
    return 1


def _variant_tag(signature, tag):
    return tuple(key for key, _ in signature["members"]).index(tag)


def _variant_copy(signature, destination, source, lines, *, references):
    _uniform_int(source["tag"], lines, "Variant discriminant differs across execution lane")
    lines.append("%s = %s;" % (destination["tag"], source["tag"]))
    lines.append("switch (%s) {" % source["tag"])
    for index, (tag, payload) in enumerate(signature["members"]):
        lines.append("case %d: {" % index)
        body = []
        (_assign_output if references else _copy)(payload, destination["payloads"][tag],
                                                 source["payloads"][tag], body)
        lines.extend("  " + line for line in body)
        lines.append("  break; }")
    lines.append('default: throw std::logic_error("invalid native Variant discriminant");')
    lines.append("}")


def _copy(signature, destination, source, lines):
    if signature["vtype"] == "product":
        for key, child in signature["members"]:
            _copy(child, destination[key], source[key], lines)
        return
    if signature["vtype"] == "variant":
        _variant_copy(signature, destination, source, lines, references=False)
        return
    source_leaves = dict((path, token) for path, _, token in _leaves(signature, source))
    body = []
    has_field = False
    for path, leaf, target in _leaves(signature, destination):
        value = source_leaves[path]
        if leaf["vtype"] == "variant":
            _variant_copy(leaf, target, value, body, references=False)
        elif leaf["vtype"] in ("state", "rhs", "scalar_field"):
            if isinstance(target, _DeferredField):
                pointer = target[2:-1]
                # Allocation takes place only for the active payload, with the exact source
                # geometry. Every rank converges before any later field-copy collective.
                body += ["%s = &ctx.scratch_state(%d, %d, %s);"
                         % (pointer, target.node_id, target.slot, value)]
            # Field copies must not evaluate 0 * uninitialized scratch contents.
            body.append("pops::PureFieldAlgebra::copy(%s, %s);" % (target, value))
            has_field = True
        else:
            body.append("%s = %s;" % (target, value))
    if not has_field:
        lines.extend(body)
        return
    # Copies are rank-local operations. Fatal host/device failures must converge before the
    # next loop condition/reduction; they are deliberately not recoverable mathematical Failure.
    lines += ["{", "std::exception_ptr structured_copy_error;", "try {"]
    lines.extend("  " + line for line in body)
    lines += ["} catch (...) { structured_copy_error = std::current_exception(); }",
              "try { Kokkos::fence(); } catch (...) { if (!structured_copy_error) structured_copy_error = std::current_exception(); }",
              'pops::collectively_rethrow_exception(structured_copy_error, ctx.prepared_execution_lane(), "structured control field copy");', "}"]


def _slots(signature, sources, node_id, prefix, lines, *, slot_offset=0, pointers=False,
           deferred=False):
    if signature["vtype"] == "variant":
        tag_token = "%s_tag_%d" % (prefix, node_id)
        lines.append("std::int64_t %s = -1;" % tag_token)
        payloads = {}
        offset = slot_offset
        for index, (tag, child) in enumerate(signature["members"]):
            payloads[tag] = _slots(child, None, node_id, "%s_arm%d" % (prefix, index), lines,
                                  slot_offset=offset, pointers=True,
                                  deferred=deferred or not pointers)
            offset += _storage_width(child)
        return {"tag": tag_token, "payloads": payloads}
    if signature["vtype"] == "result":
        code = "%s_status_%d" % (prefix, node_id)
        origin = "%s_origin_%d" % (prefix, node_id)
        lines.append("std::int64_t %s = 0;" % code)
        lines.append("std::int64_t %s = 0;" % origin)
        return {"code": code, "origin": origin, "value": _slots(signature["success"], None, node_id,
                                               prefix, lines, pointers=True)}
    leaves = {}
    offset = slot_offset
    for n, (path, leaf, source) in enumerate(_leaves(signature, sources)):
        token = "%s_%d_%d" % (prefix, node_id, n)
        if leaf["vtype"] == "variant":
            leaves[path] = _slots(leaf, source, node_id, token, lines,
                                  slot_offset=offset, pointers=pointers, deferred=deferred)
        elif leaf["vtype"] in ("state", "rhs", "scalar_field"):
            if pointers:
                lines.append("pops::MultiFab<pops::kNativeDimension>* %s = nullptr;" % token)
                leaves[path] = (_DeferredField("(*%s)" % token, node_id, offset) if deferred
                                else "(*%s)" % token)
            else:
                lines += ["pops::MultiFab<pops::kNativeDimension>* %s_storage = nullptr;" % token,
                          "std::exception_ptr %s_error;" % token,
                          "try { %s_storage = &ctx.scratch_state(%d, %d, %s); }"
                          % (token, node_id, offset, source),
                          "catch (...) { %s_error = std::current_exception(); }" % token,
                          "try { Kokkos::fence(); } catch (...) { if (!%s_error) %s_error = std::current_exception(); }" % (token, token),
                          'pops::collectively_rethrow_exception(%s_error, ctx.prepared_execution_lane(), "structured control field allocation");' % token,
                          "auto& %s = *%s_storage;" % (token, token)]
                leaves[path] = token
        else:
            lines.append("%s %s{};" % (finite_cpp_type(leaf), token))
            leaves[path] = token
        offset += _storage_width(leaf)
    return _repack(signature, leaves)


def _assign_output(signature, output, source, lines):
    """Publish references/scalars only along the selected success path."""
    if signature["vtype"] == "variant":
        _variant_copy(signature, output, source, lines, references=True)
        return
    if signature["vtype"] == "result":
        lines.append("%s = %s;" % (output["code"], source["code"]))
        lines.append("%s = %s;" % (output["origin"], source["origin"]))
        lines.append("if (%s == 0) {" % output["code"])
        body = []
        _assign_output(signature["success"], output["value"], source["value"], body)
        lines.extend("  " + line for line in body)
        lines.append("}")
        return
    sources = {path: token for path, _, token in _leaves(signature, source)}
    for path, leaf, out in _leaves(signature, output):
        token = sources[path]
        if leaf["vtype"] == "variant":
            _variant_copy(leaf, out, token, lines, references=True)
        elif leaf["vtype"] in ("state", "rhs", "scalar_field"):
            lines.append("%s = &%s;" % (out[2:-1], token))
        else:
            lines.append("%s = %s;" % (out, token))


def _failure_type(value, var, lines):
    key = ("structured_failure_type",)
    if key not in var:
        var[key] = "structured_failure_%d" % value.id
        lines.append("struct %s { std::int64_t code; std::int64_t origin; };" % var[key])
    return var[key]


def _require_success(code, origin, var, lines, message):
    _uniform_int(code, lines, "Result code differs across execution lane")
    if var.get(("failure_capture",)):
        lines.append("if (%s != 0) throw %s{%s, %s};"
                     % (code, var[("structured_failure_type",)], code, origin))
    else:
        # The legacy transaction exception has a uint32 reason ABI. Preserve the complete typed
        # cause in its diagnostic rather than narrowing arbitrary signed user Int64 codes.
        lines.append("if (pops::all_reduce_max((%s != 0) ? 1L : 0L, "
                     "ctx.prepared_execution_lane()) != 0) {" % code)
        lines.append("  throw pops::runtime::program::StepAttemptRejected("
                     "pops::SolveStatus::kInvalidEvaluation, \"structured_control\", "
                     "std::string(%s) + \"; origin=\" + std::to_string(%s) + "
                     "\"; code=\" + std::to_string(%s));" % (json.dumps(message), origin, code))
        lines.append("}")


def _reject_if(condition, lines, message):
    lines.append("if (pops::all_reduce_max((%s) ? 1L : 0L, ctx.prepared_execution_lane()) != 0) {"
                 % condition)
    lines.append("  throw pops::runtime::program::StepAttemptRejected("
                 "pops::SolveStatus::kInvalidEvaluation, \"structured_control\", %s);"
                 % json.dumps(message))
    lines.append("}")


def _uniform_int(token, lines, message):
    # The native reduction ABI overloads Real and long; int64_t may be long long (macOS).
    # Preserve every signed Int64 bit and refuse platforms with a narrower native integer lane.
    lines.append("static_assert(std::numeric_limits<long>::digits >= "
                 "std::numeric_limits<std::int64_t>::digits, "
                 "\"native collective long must represent every Int64 value\");")
    lines.append("if (pops::all_reduce_min(static_cast<long>(%s), ctx.prepared_execution_lane()) != "
                 "pops::all_reduce_max(static_cast<long>(%s), ctx.prepared_execution_lane())) {" % (token, token))
    lines.append("  throw pops::runtime::program::StepAttemptRejected("
                 "pops::SolveStatus::kInvalidEvaluation, \"structured_control\", %s);"
                 % json.dumps(message))
    lines.append("}")


def _emit_region(program, block, sub, model, lines, prelude, block_idx, field_plans, target):
    from pops.codegen.program_emit_ops import _emit_op
    sub[("structured_control",)] = True
    for value in block:
        # Each operation must receive its own block's prototype, not a single product-wide base.
        candidates = [candidate for candidate in program._values
                      if candidate.op == "state" and candidate.block == value.block]
        base = next((candidate for candidate in candidates
                     if value.state_ref is not None and candidate.state_ref == value.state_ref),
                    candidates[0] if value.state_ref is None and candidates else None)
        _emit_op(program, value, base, frozenset(), sub, model, lines, prelude,
                 block_idx=block_idx, field_plans=field_plans, target=target)


def _emit_loop(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    _check_region_effects(value.attrs.get("cond_block", ()))
    _check_region_effects(value.attrs["body_block"])
    initial = value.inputs[0]
    signature = value_signature(initial)
    current = _slots(signature, var[initial.id], value.id, "carry", lines)
    pending = _slots(signature, var[initial.id], value.id, "next", lines,
                     slot_offset=_storage_width(signature))
    _copy(signature, current, var[initial.id], lines)
    count = "iteration_%d" % value.id
    lines.append("std::int64_t %s = 0;" % count)
    if value.op == "structured_fold":
        bound, budget = (var[item.id] for item in value.inputs[1:])
        _uniform_int(bound, lines, "fold count differs across execution lane")
        _uniform_int(budget, lines, "fold capacity differs across execution lane")
        _reject_if("%s < 0 || %s < 0 || %s > %s" % (bound, budget, bound, budget),
                   lines, "fold count outside declared capacity")
        lines.append("for (; %s < %s; ++%s) {" % (count, bound, count))
        sub = dict(var)
        sub[value.attrs["body_argument"].id] = current
        sub[value.attrs["body_index"].id] = count
        body = []
        _emit_region(program, value.attrs["body_block"], sub, model, body, prelude,
                     block_idx, field_plans, target)
        _copy(signature, pending, sub[value.attrs["body"].id], body)
        _copy(signature, current, pending, body)
        lines.extend("  " + line for line in body)
        lines.append("}")
        var[value.id] = current
        return

    budget = var[value.inputs[1].id]
    status = "iteration_status_%d" % value.id
    origin = "iteration_origin_%d" % value.id
    failure_type = _failure_type(value, var, lines)
    lines.append("std::int64_t %s = 0;" % status)
    lines.append("std::int64_t %s = 0;" % origin)
    typed_errors = value.attrs.get("error_signature")
    error_slots = (_slots(typed_errors, None, value.id, "iteration_error", lines, pointers=True)
                   if typed_errors is not None else None)
    exit_label = "typed_iteration_done_%d" % value.id
    lines.append("try {")
    body = []
    _uniform_int(budget, body, "iterate budget differs across execution lane")
    body.append("if (%s < 0) { %s = 2; %s = 2; } else {" % (budget, status, origin))
    body.append("  for (;;) {")
    sub = dict(var)
    sub[("failure_capture",)] = True
    if typed_errors is not None:
        sub[("typed_failure_capture",)] = (typed_errors, error_slots, status, 1, exit_label)
    sub[value.attrs["cond_argument"].id] = current
    inner = []
    _emit_region(program, value.attrs["cond_block"], sub, model, inner, prelude,
                 block_idx, field_plans, target)
    stop = "iteration_stop_%d" % value.id
    inner.append("const long %s = (%s) ? 1L : 0L;" % (stop, sub[value.attrs["cond"].id]))
    _uniform_int(stop, inner, "iterate stop differs across execution lane")
    inner.append("if (%s != 0) break;" % stop)
    inner.append("if (%s == %s) { %s = 2; %s = 1; break; }" % (count, budget, origin, status))
    sub[value.attrs["body_argument"].id] = current
    _emit_region(program, value.attrs["body_block"], sub, model, inner, prelude,
                 block_idx, field_plans, target)
    next_value = sub[value.attrs["body"].id]
    if value.attrs.get("body_is_variant_result"):
        step_signature = value_signature(value.attrs["body"])
        _uniform_int(next_value["tag"], inner, "iteration Result differs across execution lane")
        inner.append("if (%s != %d) {" % (next_value["tag"], _variant_tag(step_signature, "ok")))
        failure_body = []
        _propagate_typed_error(dict(step_signature["members"])["err"],
                               next_value["payloads"]["err"], sub, failure_body, block_idx, target)
        inner.extend("  " + line for line in failure_body)
        inner.append("}")
        next_value = next_value["payloads"]["ok"]
    if value.attrs.get("body_is_result"):
        inner.append("if (%s != 0) { %s = %s; %s = %s; break; }"
                     % (next_value["code"], origin, next_value["origin"], status, next_value["code"]))
        next_value = next_value["value"]
    _copy(signature, pending, next_value, inner)
    _copy(signature, current, pending, inner)
    inner.append("++%s;" % count)
    body.extend("    " + line for line in inner)
    body += ["  }", "}"]
    lines.extend("  " + line for line in body)
    lines.append("} catch (const %s& failure) {" % failure_type)
    lines.append("  %s = failure.code;" % status)
    lines.append("  %s = failure.origin;" % origin)
    lines.append("} catch (const pops::runtime::program::StepAttemptRejected&) {")
    lines.append("  %s = 3;" % status)
    lines.append("  %s = 2;" % origin)
    lines.append("}")
    if typed_errors is not None:
        assert error_slots is not None
        lines.append("%s: ;" % exit_label)
        lines.append("if (%s != 0 && %s < 0) {" % (status, error_slots["tag"]))
        lines.append("  %s = %d;" % (error_slots["tag"], _variant_tag(typed_errors, "runtime")))
        lines.append("  %s = %s;" % (error_slots["payloads"]["runtime"]["code"], status))
        lines.append("  %s = %s;" % (error_slots["payloads"]["runtime"]["origin"], origin))
        lines.append("}")
        signature = value_signature(value)
        var[value.id] = {"tag": "(%s == 0 ? %d : %d)" % (status, _variant_tag(signature, "ok"), _variant_tag(signature, "err")),
                         "payloads": {"ok": {"state": current, "iterations": count}, "err": error_slots}}
    else:
        var[value.id] = {"state": current, "iterations": count, "code": status, "origin": origin}


def _emit_capture(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    if value.vtype == "variant":
        return _emit_typed_capture(program, value, var, model, lines, prelude,
                                   block_idx, field_plans, target)
    _check_region_effects(value.attrs["body_block"])
    signature = value.attrs["success_signature"]
    output = _slots(signature, None, value.id, "captured", lines, pointers=True)
    status = "capture_status_%d" % value.id
    origin = "capture_origin_%d" % value.id
    lines.append("std::int64_t %s = 0;" % status)
    lines.append("std::int64_t %s = 0;" % origin)
    failure_type = _failure_type(value, var, lines)
    lines.append("try {")
    sub = dict(var)
    sub[("failure_capture",)] = True
    body = []
    _emit_region(program, value.attrs["body_block"], sub, model, body, prelude,
                 block_idx, field_plans, target)
    source = sub[value.attrs["body"].id]
    if value.attrs.get("body_is_result"):
        _assign_output({"vtype": "result", "success": signature},
                       {"code": status, "origin": origin, "value": output}, source, body)
    else:
        _assign_output(signature, output, source, body)
    lines.extend("  " + line for line in body)
    lines.append("} catch (const %s& failure) {" % failure_type)
    lines.append("  %s = failure.code;" % status)
    lines.append("  %s = failure.origin;" % origin)
    lines.append("} catch (const pops::runtime::program::StepAttemptRejected&) {")
    lines.append("  %s = 3;" % status)
    lines.append("  %s = 2;" % origin)
    lines.append("}")
    var[value.id] = {"value": output, "code": status, "origin": origin}


def _inject_error(destination_signature, destination, source_signature, source, lines):
    """Inject a complete error into an inferred union without discarding payload leaves."""
    if source_signature["vtype"] == "variant":
        _uniform_int(source["tag"], lines, "error Variant differs across execution lane")
        lines.append("switch (%s) {" % source["tag"])
        for index, (tag, payload) in enumerate(source_signature["members"]):
            lines.append("case %d: {" % index)
            lines.append("  %s = %d;" % (destination["tag"], _variant_tag(destination_signature, tag)))
            body = []
            _assign_output(payload, destination["payloads"][tag], source["payloads"][tag], body)
            lines.extend("  " + line for line in body)
            lines.append("  break; }")
        lines.append('default: throw std::logic_error("invalid typed error discriminant");')
        lines.append("}")
    else:
        lines.append("%s = %d;" % (destination["tag"], _variant_tag(destination_signature, "user")))
        _assign_output(source_signature, destination["payloads"]["user"], source, lines)


def _propagate_typed_error(signature, tokens, var, lines, block_idx, target):
    capture = var.get(("typed_failure_capture",))
    if capture is None:
        from .program_emit_typed_failure import emit_terminal_failure
        emit_terminal_failure(signature, tokens, lines, block_idx, target)
        return
    error_signature, error_slots, outer_tag, outer_index, label = capture
    _inject_error(error_signature, error_slots, signature, tokens, lines)
    lines.append("%s = %d;" % (outer_tag, outer_index))
    lines.append("goto %s;" % label)


def _outline_capture_region(lines, start, node_id):
    """Keep one complete recovery frontier in one native function boundary."""
    region_lines = lines[start:]
    del lines[start:]
    lines += ["// pops-outlined-capture begin %d" % node_id,
              "#if defined(__GNUC__) || defined(__clang__)",
              "[&]() __attribute__((noinline)) {",
              "#else", "[&]() {", "#endif"]
    lines.extend("  " + line for line in region_lines)
    lines += ["}();", "// pops-outlined-capture end %d" % node_id]


def _emit_typed_capture(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    _check_region_effects(value.attrs["body_block"])
    signature = value_signature(value)
    success = value.attrs["success_signature"]
    errors = value.attrs["error_signature"]
    output = _slots(signature, None, value.id, "typed_capture", lines, pointers=True)
    ok_index, err_index = _variant_tag(signature, "ok"), _variant_tag(signature, "err")
    label = "typed_capture_done_%d" % value.id
    failure_type = _failure_type(value, var, lines)
    sub = dict(var)
    sub[("failure_capture",)] = True
    sub[("typed_failure_capture",)] = (errors, output["payloads"]["err"], output["tag"], err_index, label)
    # The complete recovery boundary is outlined. Its output slots and shared
    # exception type remain in the caller; all control-transfer labels and the
    # scoped continuation authority stay inside this immediately invoked lambda.
    # Nested captures establish their own typed_failure_capture target, so no
    # recoverable failure may jump across the generated function boundary.
    outlined_start = len(lines)
    scope = "continuation_scope_%d" % value.id if value.attrs.get("continuation_scope") else None
    if scope:
        lines.append("std::optional<pops::runtime::program::ContinuationAuthority<pops::kNativeDimension>::Scope> %s;" % scope)
    lines.append("try {")
    body = []
    if scope:
        body += ['%s.emplace(ctx.continuation_local_phase([&] { return ctx.continuation_authority().begin_scope(); }, "continuation child scope"));' % scope,
                 "ctx.continuation_control_consensus();"]
    _emit_region(program, value.attrs["body_block"], sub, model, body, prelude,
                 block_idx, field_plans, target)
    source_value = value.attrs["body"]
    source = sub[source_value.id]
    if value.attrs.get("body_is_variant_result"):
        source_signature = value_signature(source_value)
        source_errors = dict(source_signature["members"])["err"]
        _uniform_int(source["tag"], body, "Result discriminant differs across execution lane")
        body.append("if (%s == %d) {" % (source["tag"], _variant_tag(source_signature, "err")))
        error_body = []
        _propagate_typed_error(source_errors, source["payloads"]["err"], sub, error_body, block_idx, target)
        body.extend("  " + line for line in error_body)
        body.append("}")
        source = source["payloads"]["ok"]
    elif value.attrs.get("body_is_result"):
        _require_success(source["code"], source["origin"], sub, body, "captured legacy Result failed")
        source = source["value"]
    if scope:
        if success["vtype"] != "continuation_accepted":
            raise ValueError("continuation scope must return the complete accepted provisional")
        from .program_emit_versioned import _checked, capability_cpp_type
        promoted = "continuation_promoted_%d" % value.id
        _checked(body, promoted, capability_cpp_type(success), [],
                 "ctx.continuation_authority().promote(std::move(*%s),%s)" % (scope, source),
                 message="accepted child promotion failed")
        body.append("ctx.continuation_control_consensus();")
        source = promoted
    _assign_output(success, output["payloads"]["ok"], source, body)
    body.append("%s = %d;" % (output["tag"], ok_index))
    lines.extend("  " + line for line in body)
    runtime_slots = output["payloads"]["err"]["payloads"]["runtime"]
    def runtime_failure(code, origin):
        lines.extend(("  %s = %d;" % (output["tag"], err_index),
                      "  %s = %d;" % (output["payloads"]["err"]["tag"], _variant_tag(errors, "runtime")),
                      "  %s = %s;" % (runtime_slots["code"], code),
                      "  %s = %s;" % (runtime_slots["origin"], origin)))
    lines.append("} catch (const %s& failure) {" % failure_type)
    runtime_failure("failure.code", "failure.origin")
    lines.append("} catch (const pops::runtime::program::StepAttemptRejected&) {")
    runtime_failure("3", "2")
    lines.append("}")
    lines.append("%s: ;" % label)
    if scope:
        lines.append("%s.reset();" % scope)
        lines.append("ctx.continuation_control_consensus();")
    _outline_capture_region(lines, outlined_start, value.id)
    var[value.id] = output


def emit_structured_branch(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    is_match = value.op == "result_match"
    signature = value_signature(value)
    output = _slots(signature, None, value.id, "matched", lines, pointers=True)
    if is_match:
        result = var[value.inputs[0].id]
        condition = "%s == 0" % result["code"]
    else:
        condition = var[value.inputs[0].id]
    condition_token = "branch_choice_%d" % value.id
    lines.append("const long %s = (%s) ? 1L : 0L;" % (condition_token, condition))
    _uniform_int(condition_token, lines, "structured branch differs across execution lane")
    lines.append("if (%s != 0) {" % condition_token)
    for variant, prefix in (("ok", "true"), ("err", "false")):
        if variant == "err":
            lines.append("} else {")
        sub = dict(var)
        if is_match:
            sub[("iteration_variant", value.inputs[0].id)] = variant
        elif value.inputs[0].op == "variant_is" and variant == "ok":
            predicate = value.inputs[0]
            sub[("variant_guard", predicate.inputs[0].id)] = predicate.attrs["tag"]
        body = []
        _emit_region(program, value.attrs[prefix + "_block"], sub, model, body, prelude,
                     block_idx, field_plans, target)
        source = sub[value.attrs[prefix + "_result"].id]
        _assign_output(signature, output, source, body)
        lines.extend("  " + line for line in body)
    lines.append("}")
    var[value.id] = output


def emit_structured_op(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    op = value.op
    if op == "variant_construct":
        signature = value_signature(value)
        tag = value.attrs["tag"]
        output = _slots(signature, None, value.id, "variant", lines, pointers=True)
        _assign_output(dict(signature["members"])[tag], output["payloads"][tag],
                       var[value.inputs[0].id], lines)
        lines.append("%s = %d;" % (output["tag"], _variant_tag(signature, tag)))
        var[value.id] = output
    elif op == "variant_is":
        source = var[value.inputs[0].id]
        index = _variant_tag(value_signature(value.inputs[0]), value.attrs["tag"])
        var[value.id] = "(%s == %d)" % (source["tag"], index)
    elif op == "variant_payload":
        if var.get(("variant_guard", value.inputs[0].id)) != value.attrs["tag"]:
            raise ValueError("Variant payload escaped its exact guarded match arm")
        var[value.id] = var[value.inputs[0].id]["payloads"][value.attrs["tag"]]
    elif op == "variant_unreachable":
        _reject_if("true", lines, "invalid native Variant discriminant")
        var[value.id] = _slots(value_signature(value), None, value.id, "unreachable", lines, pointers=True)
    elif op == "variant_require_success":
        source_value = value.inputs[0]
        signature = value_signature(source_value)
        source = var[source_value.id]
        _uniform_int(source["tag"], lines, "Result discriminant differs across execution lane")
        lines.append("if (%s != %d) {" % (source["tag"], _variant_tag(signature, "ok")))
        body = []
        _propagate_typed_error(dict(signature["members"])["err"], source["payloads"]["err"], var, body, block_idx, target)
        lines.extend("  " + line for line in body)
        lines.append("}")
        var[value.id] = source["payloads"]["ok"]
    elif op == "product":
        var[value.id] = {key: var[item.id] for key, item in zip(value.attrs["fields"], value.inputs)}
    elif op == "product_get":
        var[value.id] = var[value.inputs[0].id][value.attrs["field"]]
    elif op in ("region_argument", "region_index"):
        if value.id not in var:
            raise ValueError("structured region argument has no native binding")
    elif op in ("int64_constant", "bool_constant"):
        token = "integer_%d" % value.id if op == "int64_constant" else "boolean_%d" % value.id
        literal = value.attrs["value"]
        if op == "int64_constant":
            text = ("std::numeric_limits<std::int64_t>::min()" if literal == -(1 << 63)
                    else "std::int64_t(%d)" % literal)
        else:
            text = "true" if literal else "false"
        lines.append("const %s %s = %s;" % (_cpp_type(value.vtype), token, text))
        var[value.id] = token
    elif op == "int64_cast":
        source = var[value.inputs[0].id]
        _reject_if("!std::isfinite(%s) || %s < -9223372036854775808.0 || "
                   "%s >= 9223372036854775808.0 || std::trunc(%s) != %s"
                   % (source, source, source, source, source), lines, "inexact/out-of-range Int64 cast")
        var[value.id] = "integer_%d" % value.id
        lines.append("const std::int64_t %s = static_cast<std::int64_t>(%s);" % (var[value.id], source))
    elif op == "int64_compare":
        a, b = (var[item.id] for item in value.inputs)
        var[value.id] = "(%s %s %s)" % (a, value.attrs["comparison"], b)
    elif op == "int64_op":
        a, b = (var[item.id] for item in value.inputs)
        minimum, maximum = "std::numeric_limits<std::int64_t>::min()", "std::numeric_limits<std::int64_t>::max()"
        operation = value.attrs["operation"]
        if operation == "add":
            valid = "(%s > 0 ? %s <= %s-%s : %s >= %s-%s)" % (b, a, maximum, b, a, minimum, b)
            expression = "%s + %s" % (a, b)
        elif operation == "sub":
            valid = "(%s > 0 ? %s >= %s+%s : %s <= %s+%s)" % (b, a, minimum, b, a, maximum, b)
            expression = "%s - %s" % (a, b)
        elif operation == "mul":
            valid = ("(%s == 0 || %s == 0 || (%s > 0 ? "
                     "(%s > 0 ? %s <= %s/%s : %s >= %s/%s) : "
                     "(%s > 0 ? %s >= %s/%s : %s >= %s/%s)))"
                     % (a, b, a, b, a, maximum, b, b, minimum, a,
                        b, a, minimum, b, a, maximum, b))
            expression = "%s * %s" % (a, b)
        else:
            valid = "(%s != 0 && !(%s == %s && %s == -1))" % (b, a, minimum, b)
            remainder = "(%s %% %s)" % (a, b)
            adjust = "(%s != 0 && ((%s < 0) != (%s < 0)))" % (remainder, remainder, b)
            expression = ("(%s / %s) - (%s ? 1 : 0)" % (a, b, adjust) if operation == "floordiv"
                          else "%s + (%s ? %s : 0)" % (remainder, adjust, b))
        _reject_if("!(%s)" % valid, lines, "Int64 arithmetic overflow or invalid division")
        var[value.id] = "integer_%d" % value.id
        lines.append("const std::int64_t %s = %s;" % (var[value.id], expression))
    elif op in ("structured_fold", "structured_iterate"):
        _emit_loop(program, value, var, model, lines, prelude, block_idx, field_plans, target)
    elif op == "result_ok":
        var[value.id] = {"value": var[value.inputs[0].id], "code": "std::int64_t(0)",
                         "origin": "std::int64_t(0)"}
    elif op == "result_error":
        code = var[value.inputs[1].id]
        _uniform_int(code, lines, "Result code differs across execution lane")
        _reject_if("%s == 0" % code, lines, "failure code must be nonzero")
        var[value.id] = {"value": var[value.inputs[0].id], "code": code, "origin": "std::int64_t(1)"}
    elif op == "capture_result":
        _emit_capture(program, value, var, model, lines, prelude, block_idx, field_plans, target)
    elif op in ("iteration_is_ok", "result_is_ok"):
        var[value.id] = "(%s == 0)" % var[value.inputs[0].id]["code"]
    elif op in ("iteration_success", "iteration_payload", "result_success", "result_payload"):
        result = var[value.inputs[0].id]
        variant = value.attrs.get("variant", "ok")
        if op in ("iteration_success", "result_success"):
            _require_success(result["code"], result["origin"], var, lines,
                             "iteration failed; successful state unavailable" if op == "iteration_success"
                             else "Result failed; successful value unavailable")
        elif var.get(("iteration_variant", value.inputs[0].id)) != variant:
            raise ValueError("iteration payload escaped its guarded match region")
        if op.startswith("result_"):
            var[value.id] = result["value"] if variant == "ok" else {"code": result["code"],
                                                                     "origin": result["origin"]}
        else:
            var[value.id] = ({"state": result["state"], "iterations": result["iterations"]}
                         if variant == "ok" else
                         {"code": result["code"], "origin": result["origin"],
                          "iterations": result["iterations"]})
    elif op == "result_match":
        emit_structured_branch(program, value, var, model, lines, prelude, block_idx, field_plans, target)
    else:
        raise ValueError("unknown structured control operation %r" % op)
