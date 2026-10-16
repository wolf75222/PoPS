"""Close the actual generated transformation before issuing/acquitting a Field.

There is one equation kernel. The native helpers only validate its captured
reads and own metadata/storage; they do not evaluate a second numerical model.
"""
from __future__ import annotations

from typing import Any

from pops._ir.primitive_expansion import expand_primitive_recipes
from pops.codegen.cpp_strings import cpp_string_expression

from .program_emit_kernels import _model_impl
from .program_emit_model_kernels import _emit_local_transform_kernel, _provider_binding
from .program_emit_versioned import schema_cpp
from .program_transform_read_plan import transform_read_plan


def emit_pending_presence(source: str, stem: str) -> tuple[list[str], str]:
    """Inspect the exact selected Field and agree before a collective branch."""
    present = stem + "_pending"
    ns = "pops::runtime::program::"
    return [
        "bool %s = false;" % present,
        "%sfield_read_detail::phase(ctx.prepared_execution_lane(), [&] {" % ns,
        "  %s = bool(ctx.field_representation_obligation(%s));" % (present, source),
        '}, "Field representation lookup rejected");',
        "if (pops::all_reduce_min(long(%s), ctx.prepared_execution_lane()) !=" % present,
        "    pops::all_reduce_max(long(%s), ctx.prepared_execution_lane()))" % present,
        '  throw std::invalid_argument("Field representation presence differs between ranks");',
    ], present


def emit_physical_input_guards(value: Any, variables: Any) -> list[str]:
    """A numerical producer may not silently erase an input representation.

    Diagnostics have no Field result, and structural control/transport retains
    the original metadata. The named transform has its executed discharge path.
    Identity affine copies are handled separately after their normal equation.
    """
    if value.vtype not in {"state", "rhs", "scalar_field"} and value.op != "project":
        return []
    if value.op in {"local_transform", "synchronize", "solve_outcome_component", "store_history"}:
        return []
    if (value.op == "linear_combine" and len(value.inputs) == 1
            and dict(value.attrs["coeffs"][0]) == {0: 1}):
        return []
    inputs = [item for item in value.inputs
              if item.vtype in {"state", "rhs", "scalar_field"} and item.id in variables]
    if not inputs:
        return []
    lines = ["pops::runtime::program::field_read_detail::phase(ctx.prepared_execution_lane(), [&] {"]
    lines += ["  ctx.require_discharged_field_representation(%s);" % variables[item.id]
              for item in inputs]
    lines += ['}, "numerical producer cannot discard a Field representation");']
    return lines


def emit_closed_transform(
    *, model: Any, name: str, source: str, output: str, status: str, mask: str,
    owner: int, signature: Any, block_indices: Any, provider_plans: Any, consumer_qid: str,
    stem: str, capacity: int, target: str, issue: bool, production_id: int | None = None,
) -> tuple[list[str], str]:
    """Emit a real LocalTransform then return an owned raw/physical Field.

    ``issue`` is selected by the authenticated implicit residual producer. On
    issue, the result is the raw coordinate and the transformed output is only
    the same-equation witness. On discharge, it is the actual physical result.
    The caller must preserve the returned shared owner across control regions.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("closed transformation needs an explicit positive workspace")
    if target not in {"system", "amr_system"}:
        raise ValueError("closed transformation requires the real native execution context")
    if issue and (type(production_id) is not int or production_id < 0):
        raise ValueError("representation issuance requires its compiled producer SSA")
    impl = _model_impl(model)
    declaration = impl._local_transforms[name]
    roots = expand_primitive_recipes(
        (*declaration["expressions"], declaration["valid_if"]), impl.prim_defs)
    binding = _provider_binding(impl, roots, provider_plans, consumer_qid)
    plan = transform_read_plan(impl, roots, binding)
    owner_var = stem + "_owner"
    fields = "pops::MultiFab<pops::kNativeDimension>"
    ns = "pops::runtime::program::"
    schema = cpp_string_expression(plan.descriptor)
    native_schema = schema_cpp(signature, block_indices)
    if signature.get("vtype") not in {"state", "rhs", "scalar_field"}:
        raise ValueError("closed transformation requires one typed Field")
    lines = [
        "std::shared_ptr<%s> %s;" % (fields, owner_var),
        "{",
        "  const auto& transform_lane_ = ctx.prepared_execution_lane();",
        "  constexpr std::uint64_t transform_capacity_ = UINT64_C(%d);" % capacity,
        "  std::optional<%sContinuationType> transform_type_owner_;" % ns,
        "  %sfield_read_detail::Budget transform_budget_(transform_capacity_);" % ns,
        "  std::string transform_support_;",
        "  %sfield_read_detail::phase(transform_lane_, [&] {" % ns,
        # The shallow schema factory owns at most three Type values and two
        # copies of each literal. Its complete emitted form is a conservative
        # bound for those string bytes, computed at compilation, not from data.
        "    transform_budget_.charge(3 * sizeof(%sContinuationType));" % ns,
        "    transform_budget_.charge(UINT64_C(%d));" % (2 * len(native_schema.encode("utf-8"))),
        "    transform_type_owner_.emplace(%s);" % native_schema,
        "    transform_support_ = ctx.field_transform_support(%s, *transform_type_owner_, transform_budget_.remaining(), %d);" % (source, production_id if issue else -1),
        '  }, "transform source authority preflight failed");',
        "  std::vector<std::string> transform_provider_authorities_;",
        "  std::vector<std::string_view> transform_authorities_;",
        "  const std::string_view transform_schema_ = %s;" % schema,
        "  const auto& transform_type_ = *transform_type_owner_;",
        "  const auto& transform_input_ = %s;" % source,
        "  auto& transform_output_ = %s;" % output,
        "  %sfield_read_detail::phase(transform_lane_, [&] {" % ns,
        "    transform_budget_.charge(transform_support_.size());",
        "    transform_budget_.charge(transform_schema_.size());",
        "    transform_budget_.charge(sizeof(transform_authorities_) + sizeof(transform_provider_authorities_));",
        "    transform_budget_.charge(UINT64_C(%d) * sizeof(std::string_view));" % plan.count,
        "    transform_budget_.charge(UINT64_C(%d) * sizeof(std::string));" % plan.provider_count,
        "    transform_authorities_.reserve(%d);" % plan.count,
        "    transform_provider_authorities_.resize(%d);" % plan.provider_count,
        *["    transform_authorities_.push_back(%s);" % cpp_string_expression(authority)
          for authority in plan.authorities],
        "    for (const auto authority : transform_authorities_) transform_budget_.charge(authority.size());",
        '  }, "transform source workspace preflight failed");',
        "  auto transform_capture_ = %sCapturedTransformReads<pops::kNativeDimension>::prepare(" % ns,
        "      %s, %d, transform_lane_, transform_budget_.remaining());" % (source, plan.count),
    ]
    lines += ["  " + line for line in _emit_local_transform_kernel(
        model, name, "transform_input_", "transform_output_", status, mask, owner,
        provider_plans=provider_plans, consumer_qid=consumer_qid,
        capture_var="transform_capture_.inputs.journal",
        coordinate_capture_var="transform_capture_.coordinates.journal")]
    reduce = "pointwise_level_status_max" if target == "amr_system" else "pointwise_status_max"
    lines += [
        "  const auto transform_status_ = ctx.%s(%d, %s, %s, transform_lane_);" % (reduce, owner, status, mask),
        "  if (transform_status_ != pops::Real(0))",
        "    throw %sStepAttemptRejected(pops::SolveStatus::kInvalidEvaluation," % ns,
        '        "local_transform", "closed transformation rejected its actual evaluation");',
        "  %sfield_read_detail::phase(transform_lane_, [&] {" % ns,
        "    %sfield_read_detail::Budget provider_workspace_(transform_budget_.remaining());" % ns,
        "    provider_workspace_.charge(sizeof(transform_capture_));",
        "    provider_workspace_.charge(transform_capture_.inputs.retained_bytes);",
        "    provider_workspace_.charge(transform_capture_.coordinates.retained_bytes);",
        *[line for slot in range(plan.provider_count) for line in (
            "    transform_provider_authorities_[%d] = ctx.field_provider_read_authority(%s, %d, provider_workspace_.remaining());" % (slot, cpp_string_expression(consumer_qid), slot),
            "    provider_workspace_.charge(transform_provider_authorities_[%d].size());" % slot,
            "    transform_budget_.charge(transform_provider_authorities_[%d].size());" % slot,
            "    transform_authorities_[%d] = transform_provider_authorities_[%d];" % (len(plan.parameter_indices)+slot, slot),
        )],
        '  }, "executed provider publication authority rejected");',
        "  auto transform_closed_ = transform_capture_.close(%s, transform_schema_," % source,
        '      "pops.transform.coordinates.v1", transform_support_, transform_authorities_,',
        "      transform_lane_, transform_budget_.remaining());",
        "  %sfield_read_detail::phase(transform_lane_, [&] {" % ns,
        "    transform_budget_.charge(sizeof(transform_capture_));",
        "    transform_budget_.charge(transform_capture_.inputs.retained_bytes);",
        "    transform_budget_.charge(transform_capture_.coordinates.retained_bytes);",
        "    transform_budget_.charge(sizeof(transform_closed_));",
        "    transform_budget_.charge(transform_closed_.inputs.size());",
        "    transform_budget_.charge(transform_closed_.coordinates.size());",
        "    transform_budget_.charge(sizeof(%sFieldRepresentationObligation));" % ns,
        "    auto transform_obligation_ = std::make_shared<const %sFieldRepresentationObligation>(" % ns,
        "        std::move(transform_closed_.inputs), std::move(transform_closed_.coordinates));",
    ]
    if issue:
        lines += ["    %s = ctx.own_field_representation(%s, transform_type_," % (owner_var, source),
                  "        std::move(transform_obligation_), transform_budget_.remaining(), %d);" % production_id]
    else:
        lines += ["    %s = ctx.own_discharged_field_representation(%s, %s, transform_type_," % (owner_var, output, source),
                  "        transform_obligation_, transform_budget_.remaining());"]
    lines += ['  }, "closed transformation publication rejected");', "}"]
    return lines, owner_var
