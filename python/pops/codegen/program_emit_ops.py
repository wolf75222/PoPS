"""pops.codegen.program_emit_ops : the per-op SSA -> C++ dispatcher.

Extracted verbatim from ``pops.codegen.program_codegen`` (Spec-4 file-size budget).
``_emit_op`` lowers a SINGLE SSA op to C++ (appending to the line list, recording its
token), shared by the top-level body walk (``program_emit_control._emit_body``) and the
control-flow sub-blocks; it dispatches to the model kernels, the matrix-free / generic
condensed-implicit emitters, control flow and the schedule wrap
(``program_emit_{model_kernels,solve,condensed,control,schedule}``).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pops.identity.scalar import scalar_cpp
from pops.time.references import block_name
from pops.codegen.program_emit_kernels import (
    _PROFILE_SKIP_OPS,
    _coeff_cpp,
    _coeff_metadata_cpp,
    _deref,
    _emit_cell_compare_kernel,
    _emit_where_kernel,
    _model_impl,
    _named_fluxes,
    program_provider_consumer_qid,
)
from pops.codegen.program_emit_model_kernels import (
    _emit_apply_kernel,
    _emit_coupled_rate_kernel,
    _emit_flux_kernel,
    _emit_local_transform_kernel,
    _emit_solve_local_linear_kernel,
    _emit_solve_local_nonlinear_kernel,
    _emit_solve_coupled_implicit_kernel,
    _emit_source_kernel,
)
from pops.codegen.program_emit_condensed import emit_condensed_op
from pops.codegen.program_emit_control import (
    _coupled_rate_components,
    _emit_branch,
    _emit_range,
    _emit_subcycle,
    _emit_while,
)
from pops.codegen.program_emit_solve import (
    _consumed_solve_action,
    _append_solve_report_guard,
    _emit_matrix_free_operator,
    _emit_solve_linear,
)
from pops.codegen.program_emit_schedule import _emit_schedule_wrap
from pops.codegen.program_persistent_plan import persistent_slot_token
from pops.codegen.program_emit_field_routes import field_point_cpp, resolved_field_route


def _required_block_index(block_idx: Any, block: Any, where: str) -> int:
    """Return an explicitly declared runtime block index, never an index-0 fallback."""
    if not isinstance(block_idx, Mapping):
        raise ValueError(
            "%s: runtime block routing is unavailable; lowering requires Program._block_indices()"
            % where)
    if block is None:
        raise ValueError("%s: a block-qualified Program value is required" % where)
    try:
        index = block_idx[block]
    except KeyError:
        raise ValueError(
            "%s: block %r is not declared in the Program block-index map"
            % (where, block)) from None
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("%s: invalid runtime block index %r" % (where, index))
    return index


def _value_owner_block(value: Any) -> Any:
    """Return one declared block identity carried by ``value``, or None."""
    block = getattr(value, "block", None)
    if block is not None:
        return block
    state_ref = getattr(value, "state_ref", None)
    return getattr(state_ref, "block_ref", None)


def _unique_dataflow_owner_block(value: Any, *, where: str) -> Any:
    """Return the single authenticated block on ``value``'s producer dataflow.

    Top-level generated Cartesian ops may themselves be unqualified. Their owner is
    the unique BlockHandle on the op or its producer SSA, never a default index.
    """
    seen: set[int] = set()
    owners: list[Any] = []

    def visit(node: Any) -> None:
        ident = getattr(node, "id", None)
        key = ident if isinstance(ident, int) else id(node)
        if key in seen:
            return
        seen.add(key)
        block = _value_owner_block(node)
        if block is not None and block not in owners:
            owners.append(block)
        for item in getattr(node, "inputs", ()) or ():
            visit(item)

    visit(value)
    if len(owners) == 1:
        return owners[0]
    if not owners:
        raise ValueError(
            "%s: generated Cartesian operator has no unique authenticated owner block"
            % where)
    raise ValueError(
        "%s: generated Cartesian operator has conflicting owner blocks %s"
        % (where, sorted(block_name(item) for item in owners)))


def _cartesian_generated_owner_index(block_idx: Any, value: Any, *, where: str) -> int:
    """Map a generated Cartesian op to its unique runtime owner index."""
    return _required_block_index(
        block_idx, _unique_dataflow_owner_block(value, where=where), where)


def _unique_resource_owner_block(program: Any, value: Any, *, where: str) -> Any:
    """Resolve the one authenticated block owner for an unqualified resource occurrence.

    Some top-level field buffers (notably condensed RHS storage) are deliberately authoring-
    unqualified.  Their owner is still recoverable from the executable dataflow: walk both the
    producer inputs and the reachable consumers, collecting only explicit ``ProgramValue.block`` /
    ``state_ref.block_ref`` identities.  A dead consumer is excluded by the same reachability pass
    that drives emission, and an absent or conflicting owner is a hard lowering error.  In
    particular, this helper never turns an ownerless occurrence into runtime block zero.
    """
    if program is None:
        raise ValueError("%s requires an authenticated Program owner block" % where)
    from pops.codegen.program_persistent_plan import (
        _embedded_program_values,
        _reachable_program_occurrences,
    )

    occurrences = tuple(_reachable_program_occurrences(program))
    by_object = {id(candidate): candidate for candidate, _path in occurrences}
    if id(value) not in by_object:
        raise ValueError(
            "%s references a Program resource occurrence outside the reachable executable graph"
            % where
        )

    users: dict[int, list[Any]] = {}
    for candidate in by_object.values():
        if candidate is value:
            continue
        refs = _embedded_program_values(
            (getattr(candidate, "inputs", ()), getattr(candidate, "attrs", {}))
        )
        for reference in refs:
            if id(reference) in by_object:
                users.setdefault(id(reference), []).append(candidate)

    owners: list[Any] = []
    seen: set[int] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        owner = _value_owner_block(current)
        if owner is not None and owner not in owners:
            owners.append(owner)
        for reference in _embedded_program_values(getattr(current, "inputs", ())):
            if id(reference) in by_object:
                pending.append(reference)
        pending.extend(users.get(marker, ()))

    if len(owners) == 1:
        return owners[0]
    if not owners:
        raise ValueError(
            "%s: resource occurrence %r has no unique authenticated owner block"
            % (where, getattr(value, "name", "<?>"))
        )
    raise ValueError(
        "%s: resource occurrence %r has conflicting owner blocks %s"
        % (where, getattr(value, "name", "<?>"),
           sorted(block_name(item) for item in owners))
    )


def _append_resource_preparation(
    prelude: Any,
    var: Any,
    *,
    kind: str,
    slot: str,
    subslot: int,
    program_block: int,
    ncomp: int | None = None,
    ghost_depth: int | None = None,
) -> None:
    """Emit one install-time preparation call for a dense resource occurrence.

    ``ProgramResourcePlan`` owns the slot contract; the generated call only carries the compact
    slot, its scratch subslot and the exact program block used to obtain the detached prototype.
    The emission-local ``var`` table is also the deduplication authority because nested control-flow
    walkers copy it before re-entering this dispatcher.  A missing prelude is a codegen misuse: it
    would otherwise move a first allocation into the step body.
    """
    key = ("program_resource_preparation", kind, slot, int(subslot))
    contract = (int(program_block), ncomp, ghost_depth)
    if key in var:
        if var[key] != contract:
            raise ValueError(
                "Program resource %s slot %s subslot %d has conflicting preparation contracts"
                % (kind, slot, int(subslot))
            )
        return
    if prelude is None:
        raise NotImplementedError(
            "Program resource preparation requires an install-time prelude; refusing step-local "
            "scratch allocation"
        )
    var[key] = contract
    if kind == "state":
        prelude.append(
            "ctx.prepare_state_scratch(%s, %d, %d);"
            % (slot, int(subslot), int(program_block))
        )
    elif kind == "rhs":
        prelude.append(
            "ctx.prepare_rhs_scratch(%s, %d, %d);"
            % (slot, int(subslot), int(program_block))
        )
    elif kind == "scalar":
        if ncomp is None or ghost_depth is None:
            raise ValueError("scalar Program resource preparation requires ncomp and ghost depth")
        prelude.append(
            "ctx.prepare_scalar_scratch(%s, %d, %d, %d, %d);"
            % (slot, int(subslot), int(program_block), int(ncomp), int(ghost_depth))
        )
    else:
        raise ValueError("unsupported Program resource preparation kind %r" % kind)


def _append_transaction_authority_declaration(
    prelude: Any,
    var: Any,
    *,
    kind: str,
    identity: str,
) -> None:
    """Declare one finite ProgramRuntimeState authority during candidate preparation.

    The declaration is emitted from the same dispatcher branch as the corresponding hot sink.  It
    therefore follows the executable lowering (including structured regions and post-sync bodies)
    instead of rescanning authoring nodes that a dead enclosing operator may have made inert.  The
    registry is a mutable value deliberately shared by shallow-copied structured-region token maps,
    so sibling branches deduplicate one install-wide authority without leaking their C++ aliases.
    """
    if prelude is None:
        raise NotImplementedError(
            "Program transaction authority declaration requires an install-time prelude"
        )
    if not isinstance(identity, str) or not identity:
        raise ValueError("Program transaction authority identity must be non-empty text")
    declarations_key = ("program_transaction_authority_declarations",)
    declarations = var.get(declarations_key)
    if declarations is None:
        declarations = set()
        var[declarations_key] = declarations
    if not isinstance(declarations, set):
        raise TypeError("Program transaction authority declaration registry is invalid")
    declaration = (kind, identity)
    if declaration in declarations:
        return
    if kind == "diagnostic":
        prelude.append("ctx.declare_diagnostic(%s);" % json.dumps(identity))
    elif kind == "balance_route":
        prelude.append("ctx.declare_balance_route(%s);" % json.dumps(identity))
    elif kind == "step_projection":
        prelude.append("ctx.declare_step_projection(%s);" % json.dumps(identity))
    else:
        raise ValueError("unsupported Program transaction authority kind %r" % kind)
    declarations.add(declaration)


def _append_generated_field_route_preparation(
    prelude: Any,
    var: Any,
    *,
    slot: str,
    field: str,
    program_blocks: tuple[int, ...],
) -> None:
    """Emit one install-time dense field-route registration.

    ``solve_fields_from_blocks_at`` used to key its hot route cache by the SSA value id.  That id is
    an authoring identity, not the bind-sealed resource slot, and two equivalent occurrences could
    therefore create an untracked runtime map entry.  Route preparation is now explicit and
    install-time: the dense slot, qualified provider identity, and exact Program-block subset are
    registered once before the first step.  The emission-local ``var`` table is shared by nested
    control walkers, so duplicate visits either coalesce or fail on a changed contract.
    """
    if prelude is None:
        raise NotImplementedError(
            "generated field-route preparation requires an install-time prelude; refusing a "
            "step-local route registration"
        )
    if not isinstance(field, str) or not field:
        raise ValueError("generated field-route preparation requires a non-empty provider identity")
    if any(isinstance(block, bool) or not isinstance(block, int) or block < 0
           for block in program_blocks):
        raise ValueError("generated field-route preparation requires exact non-negative blocks")
    if len(program_blocks) != len(set(program_blocks)):
        raise ValueError("generated field-route preparation contains duplicate Program blocks")
    key = ("program_generated_field_route", slot)
    contract = (field, program_blocks)
    if key in var:
        if var[key] != contract:
            raise ValueError(
                "Program resource slot %s has conflicting generated field-route contracts" % slot
            )
        return
    var[key] = contract
    prelude.append(
        "ctx.prepare_generated_field_route(%s, %s, {%s});"
        % (slot, json.dumps(field), ", ".join(str(block) for block in program_blocks))
    )


def _canonical_metadata_int(value: Any, *, where: str) -> int:
    """Decode one graph-canonical integer without accepting an approximate numeric cast."""
    if isinstance(value, bool):
        raise TypeError("%s must be an exact integer" % where)
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping) and set(value) == {"scalar"}:
        scalar = value["scalar"]
        if isinstance(scalar, Mapping) and scalar.get("kind") == "integer" \
                and isinstance(scalar.get("value"), str):
            try:
                return int(scalar["value"])
            except ValueError:
                pass
    raise TypeError("%s must be an exact graph-canonical integer" % where)


def _append_pointwise_solve_report(
        program: Any, solve: Any, status: str, lines: list[str], *,
        label: str, stem: str, active_mask: str | None = None,
        block: int | None = None) -> None:
    """Reduce typed per-cell status and consume one collective ``SolveReport``.

    Pointwise kernels own the report they create, so they author its failure disposition before
    constructing the outcome. Prepared/provider solves retain their native action authority.
    """
    action_kind, action_statuses = _consumed_solve_action(program, solve)

    def failure_action(status_name: str) -> str:
        if action_kind == "reject_attempt" and status_name in action_statuses:
            return "pops::SolveAction::kRejectAttempt"
        return "pops::SolveAction::kFailRun"

    code = "%s_code_%d" % (stem, solve.id)
    report = "%s_report_%d" % (stem, solve.id)
    lane = "%s_lane" % report
    lines.append(
        "const pops::ExecutionLane& %s = ctx.prepared_execution_lane();" % lane
    )
    if active_mask is None:
        reduction = "pops::all_reduce_max(pops::reduce_max_local(%s, 0), %s)" % (
            status,
            lane,
        )
    else:
        if block is None:
            raise ValueError("pointwise masked solve reduction requires a runtime block index")
        reduction = "ctx.pointwise_status_max(%d, %s, %s, %s)" % (
            block,
            status,
            active_mask,
            lane,
        )
    lines.append("const int %s = static_cast<int>(%s);" % (code, reduction))
    lines.append("pops::SolveReport %s;" % report)
    lines.append("if (%s == 0) %s.mark_solved();" % (code, report))
    lines.append(
        "else if (%s == 1) %s.mark_failed(pops::SolveStatus::kIterationLimit, %s);"
        % (code, report, failure_action("iteration_limit")))
    lines.append(
        "else if (%s == 2) %s.mark_failed(pops::SolveStatus::kSingular, %s);"
        % (code, report, failure_action("singular")))
    lines.append(
        "else %s.mark_failed(pops::SolveStatus::kInvalidEvaluation, %s);"
        % (report, failure_action("invalid_evaluation")))
    outcome = "%s_outcome_%d" % (stem, solve.id)
    lines.append(
        "pops::SolveOutcome %s = pops::SolveOutcome::collective_lane("
        "std::move(%s), %s);" % (outcome, report, lane)
    )
    _append_solve_report_guard(program, solve, outcome, lines, label=label)


def _append_local_nonlinear_report(
    program: Any, solve: Any, status: str, report: str, lines: list[str]
) -> str:
    """Reduce one prepared local solve and return its collective ``SolveOutcome`` token."""
    action_kind, _ = _consumed_solve_action(program, solve)
    failure_action = (
        "pops::SolveAction::kRejectAttempt"
        if action_kind == "reject_attempt"
        else "pops::SolveAction::kFailRun"
    )
    lane = "%s_lane" % report
    priority = "%s_priority" % report
    lines.append(
        "const pops::ExecutionLane& %s = ctx.prepared_execution_lane();" % lane
    )
    lines.append(
        "const int %s = static_cast<int>(pops::all_reduce_max("
        "pops::reduce_max_local(%s, 10), %s));" % (priority, status, lane)
    )
    reduced = {
        "status": "%s_status" % report,
    }
    lines.append(
        "const int %s = pops::local_nonlinear_status_code("
        "pops::local_nonlinear_status_from_priority(%s));" % (reduced["status"], priority)
    )
    fields = (
        ("iterations", 1, "int"),
        ("evaluations", 2, "int"),
        ("reference_residual", 3, "real"),
        ("residual", 4, "real"),
        ("step", 5, "real"),
        ("condition", 6, "real"),
        ("safeguard_steps", 7, "int"),
    )
    for suffix, component, kind in fields:
        token = "%s_%s" % (report, suffix)
        reduced[suffix] = token
        expression = "pops::all_reduce_max(pops::reduce_max_local(%s, %d), %s)" % (
            status,
            component,
            lane,
        )
        if kind == "int":
            lines.append("const int %s = static_cast<int>(%s);" % (token, expression))
        else:
            lines.append("const pops::Real %s = %s;" % (token, expression))
    location = "%s_failure_location" % report
    reported_failure = "%s_reported_failure" % report
    failed_count = "%s_failed_count" % report
    lines += [
        "const pops::Real %s = pops::all_reduce_sum("
        "pops::reduce_sum_local(%s, 9), %s);" % (failed_count, status, lane),
        "pops::LocalNonlinearFailureLocation<pops::kNativeDimension> %s;" % location,
        "if (%s > pops::Real(0))" % failed_count,
        "  %s = pops::collective_first_local_nonlinear_failure(%s, %s, 10, 8, %s);"
        % (location, status, priority, lane),
        "if (%s > pops::Real(0) && (!%s.found || %s.priority != %s))"
        % (failed_count, location, location, priority),
        "  throw std::runtime_error("
        '"local nonlinear collective status/location precedence mismatch");',
        "const pops::SolveFailureLocation %s = %s.found ? "
        "pops::SolveFailureLocation::from<pops::kNativeDimension>(%s.index, %s.component) : "
        "pops::SolveFailureLocation{};"
        % (reported_failure, location, location, location),
    ]
    lines.append(
        "pops::SolveReport %s = pops::local_nonlinear_solve_report("
        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        % (
            report,
            reduced["status"],
            reduced["iterations"],
            reduced["evaluations"],
            reduced["reference_residual"],
            reduced["residual"],
            reduced["step"],
            reduced["condition"],
            reduced["safeguard_steps"],
            reported_failure,
            failure_action,
        )
    )
    outcome = report.replace("_report_", "_outcome_", 1)
    if outcome == report:
        outcome = "%s_outcome" % report
    lines.append(
        "pops::SolveOutcome %s = pops::SolveOutcome::collective_lane("
        "std::move(%s), %s);" % (outcome, report, lane)
    )
    return outcome


def _emit_op(program: Any, v: Any, base: Any, committed_ids: Any, var: Any, model: Any, lines: Any,
             prelude: Any = None, block_idx: Any = None, target: Any = "system",
             field_plans: Any = None,
             has_shared_interface_implicit_jacvec: bool = False) -> None:
    """Lower a SINGLE op to C++, appending to @p lines and recording its C++ token in @p var. Shared
    by the top-level walk and the while sub-blocks (a while body re-runs this per op each pass), so
    reductions / compares / linear_combine all lower identically inside the loop. @p base is the
    block-state value of THIS op's block (its C++ var is the loop variable inside a while sub-block);
    @p committed_ids is the set of committed value ids (empty inside a sub-block: a body combine is
    never a commit). @p prelude collects INSTALL-TIME lines (persistent scratch + apply lambdas) for
    the matrix-free Krylov ops; None inside a sub-block (those ops only appear at the top level for
    now). @p block_idx maps exact ``BlockHandle`` identities to runtime indices (ADC-426). Missing
    routing is always an error; even a one-block Program reaches index 0 through an explicit map."""
    bidx = (_required_block_index(block_idx, v.block, "emit op %r" % v.name)
            if v.block is not None else None)
    from pops.codegen.program_models import model_for_node
    node_model = model_for_node(model, v) if model is not None and (
        v.block is not None or v.attrs.get("operator_handle") is not None) else model
    provider_plans = var.get(("program_provider_plans",))
    # PER-NODE PROFILING (ADC-459): bracket this op's emitted C++ with a steady_clock pair
    # recorded under "node:<v.name>" (shown by sim.profile_report next to the coarse phases). A
    # now() + ctx.profile_record pair (NOT a RAII ProfileScope { }) keeps the emitted declarations
    # at step-body scope -- later nodes read them (e.g. r2 / acc3). Additive, ~free when profiling
    # is off (record early-returns), changes no numerics; ops emitting no statement (pure inline
    # token: cfl / compare) are skipped by the len guard below. _start marks this op's first line.
    _profile_start = len(lines)

    # Resource ids are authoring identities and are deliberately not passed to
    # the generated runtime. Resolve the sealed dense slot once per emitted
    # occurrence; generated calls contain only this integer.
    resource_program_block = bidx

    def _resource_slot() -> str:
        return persistent_slot_token(program, v, target=target)

    def _prepare_resource(
        kind: str,
        prototype: Any,
        *,
        subslot: int = 0,
        ncomp: int | None = None,
        ghost_depth: int | None = None,
        owner: Any = None,
    ) -> None:
        nonlocal resource_program_block
        resource_owner = owner
        if resource_owner is None:
            resource_owner = v.block if v.block is not None else _value_owner_block(prototype)
        if resource_owner is None:
            resource_owner = _unique_resource_owner_block(
                program,
                v,
                where="prepare %s resource %r" % (kind, v.name),
            )
        program_block = _required_block_index(
            block_idx, resource_owner, "prepare %s resource %r" % (kind, v.name)
        )
        if resource_program_block is not None and resource_program_block != program_block:
            raise ValueError(
                "resource %r has conflicting owner blocks %d and %d"
                % (v.name, int(resource_program_block), int(program_block))
            )
        if resource_program_block is None:
            resource_program_block = program_block
        _append_resource_preparation(
            prelude,
            var,
            kind=kind,
            slot=_resource_slot(),
            subslot=subslot,
            program_block=program_block,
            ncomp=ncomp,
            ghost_depth=ghost_depth,
        )

    if v.op == "post_synchronization":
        var[v.id] = "/* post_synchronization */"
    elif v.op == "state":
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.state(%d);" % (var[v.id], bidx))
    elif v.op == "synchronize":
        (source,) = v.inputs
        relation = v.attrs.get("relation")
        from pops.time._schedule.synchronization import SampleAndHold, relation_data

        expected = relation_data(SampleAndHold())
        point = v.point.time if hasattr(v.point, "time") else v.point
        if relation == expected:
            lines.append(
                "ctx.synchronize_sample_and_hold(%s, %s, %d, %s);"
                % (json.dumps(source.clock.qualified_id), json.dumps(v.clock.qualified_id),
                   int(point.step), scalar_cpp(point.offset)))
            var[v.id] = var[source.id]
        elif isinstance(relation, Mapping) \
                and relation.get("kind") == "history_interpolation":
            capability = relation.get("interpolation")
            if not isinstance(capability, Mapping) or set(capability) != {
                    "kind", "schema_version", "minimum_samples"} \
                    or capability.get("kind") != "linear" \
                    or _canonical_metadata_int(
                        capability["schema_version"],
                        where="LinearInterpolation schema_version",
                    ) != 1 \
                    or _canonical_metadata_int(
                        capability["minimum_samples"],
                        where="LinearInterpolation minimum_samples",
                    ) != 2:
                raise NotImplementedError(
                    "history synchronization capability %r has no native lowering; "
                    "supported capability: LinearInterpolation()" % capability)
            if source.op != "history":
                raise ValueError(
                    "LinearInterpolation native lowering requires one retained history value")
            contract = relation["provider"]["contract"]
            depth = _canonical_metadata_int(
                contract["depth"], where="LinearInterpolation history depth")
            temporal = program.temporal_manifest()
            ticks = {
                row["id"]: int(row["ticks_per_macro"]) for row in temporal["clocks"]
            }
            coordinate = (
                (Fraction(point.step) + Fraction(point.offset.to_python()))
                * Fraction(ticks[source.clock.qualified_id], ticks[v.clock.qualified_id])
            )
            if coordinate < -depth or coordinate > 0:
                raise ValueError(
                    "LinearInterpolation target %s lies outside retained history [-%d, 0]"
                    % (coordinate, depth))
            var[v.id] = "u%d" % v.id
            _prepare_resource("state", var[source.id])
            lines.append(
                "pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                % (var[v.id], _resource_slot(), var[source.id]))
            lines.append(
                "ctx.interpolate_history_linear(%s, %s, %d, %d, %s, %s, %d, %s);"
                % (
                    var[v.id],
                    json.dumps(source.attrs["history"]),
                    depth,
                    bidx,
                    json.dumps(source.clock.qualified_id),
                    json.dumps(v.clock.qualified_id),
                    int(point.step),
                    scalar_cpp(point.offset),
                )
            )
        else:
            raise NotImplementedError(
                "synchronization provider %r has no native lowering; supported provider: "
                "SampleAndHold() or LinearInterpolation()" % relation)
    elif v.op == "solve_fields":
        # Per-stage field solve: the callable Case field operator re-solves phi from THIS
        # stage's explicit state (the shared aux is re-filled before the stage's RHS reads it; the
        # first stage state == U^n == the context's current state). Multi-block:
        # solve_fields_from_state_at(point, provider, idx, U_stage) is a genuinely COUPLED solve --
        # the Poisson RHS is Sum_s elliptic_rhs_s(U_s), block idx at its exact active level/stage
        # state, every other block contributing its live state into the shared phi/aux.
        (state_in,) = v.inputs  # solve_fields inputs = (state,)
        field_ref = v.attrs.get("field")
        if field_ref is None:
            raise ValueError("solve_fields node has no exact field identity")
        field, _ = resolved_field_route(field_ref, field_plans)
        lines += field_point_cpp(program, v, field)
        boundary_point = "field_boundary_point_%d" % v.id
        lines.append(
            "const auto %s = ctx.boundary_evaluation_point(%d);"
            % (boundary_point, v.id)
        )
        report = "field_report_%d" % v.id
        solve_stmt = (
            "pops::SolveOutcome %s = "
            "ctx.solve_fields_from_state_at(%s, %s, %d, %s);"
            % (report, boundary_point, json.dumps(field), bidx, var[state_in.id])
        )
        lines.append(solve_stmt)
        _append_solve_report_guard(program, v, report, lines, label="field_solve")
        var[v.id] = var[state_in.id]
    elif v.op == "solve_fields_from_blocks":
        # Coupled multi-block field solve (ADC-457): a SIMULTANEOUS solve, EVERY listed block at
        # its OWN stage state -- the Poisson RHS is Sum_s elliptic_rhs_s(U_s) over all coupled
        # blocks, not a single-target override. Lowers through the context-owned workspace seam: the
        # generated initializer_list is stack/static request data (no per-step host vector), and the
        # context remaps each exact Program block to its installed native block. An input whose block
        # was never declared via T.state fails loud at emit.
        if not isinstance(block_idx, Mapping):
            raise ValueError(
                "solve_fields_from_blocks: runtime block routing is unavailable")
        bmap = block_idx
        overrides = []
        program_blocks = []
        for st in v.inputs:  # inputs = the N state values, slotted by their own block index
            index = _required_block_index(
                bmap, st.block, "solve_fields_from_blocks input node %r" % st.id)
            program_blocks.append(index)
            overrides.append("{%d, &%s}" % (index, var[st.id]))
        field_ref = v.attrs.get("field")
        if field_ref is None:
            raise ValueError("solve_fields_from_blocks node has no exact field identity")
        field, _ = resolved_field_route(field_ref, field_plans)
        if program is None:
            raise NotImplementedError(
                "solve_fields_from_blocks requires an authenticated Program resource plan"
            )
        resource_slot = _resource_slot()
        _append_generated_field_route_preparation(
            prelude,
            var,
            slot=resource_slot,
            field=field,
            program_blocks=tuple(program_blocks),
        )
        lines += field_point_cpp(program, v, field)
        boundary_point = "field_boundary_point_%d" % v.id
        lines.append(
            "const auto %s = ctx.boundary_evaluation_point(%d);"
            % (boundary_point, v.id)
        )
        report = "field_report_%d" % v.id
        lines.append(
            "pops::SolveOutcome %s = ctx.solve_fields_from_blocks_at(%s, %s, {%s});"
            % (
                report,
                boundary_point,
                resource_slot,
                ", ".join(overrides),
            )
        )
        _append_solve_report_guard(program, v, report, lines, label="field_solve")
        # solve_fields_from_blocks returns a FieldContext (the shared aux); its var aliases the first
        # listed state so a downstream rhs(state, fields) reads the refreshed shared aux like any
        # solve_fields result (the FieldContext carries no readable buffer of its own).
        var[v.id] = var[v.inputs[0].id]
    elif v.op == "coupled_rate":
        # A coupled rate (collisions / ionization, Spec 3 criterion 27, ADC-457): ONE multi-state
        # for_each_cell kernel fills the per-block rate scratch of EVERY participating block at
        # once -- the component formulas reference cons vars from MULTIPLE input states, so the
        # blocks cannot be lowered as independent single-block rates. Allocate one rate scratch per
        # block (shaped like that block's state, via rhs_scratch_like), emit the shared kernel that
        # binds each input state's exact-ranked FieldView + conservative names and writes all block
        # scratches, and record
        # each block's scratch name so the coupled_rate_out for that block aliases it. All input
        # states are co-located (same ba/dm as the System aux), so a single shared loop is sound
        # (the same co-distribution every aux-reading kernel relies on; see _kernel_open).
        components = _coupled_rate_components(program, v, model)
        by_block = {s.block: s for s in v.inputs}
        if target == "system":
            for block in components:
                index = _required_block_index(
                    block_idx, block, "preflight coupled rate %r" % v.name)
                lines.append(
                    "ctx.require_cartesian_generated_operator(%d, %s);"
                    % (index, json.dumps("coupled_rate")))
        scratch = {}
        for subslot, blk in enumerate(components):   # bundle / expr block order
            scratch[blk] = "cr%d_%s" % (v.id, block_name(blk))
            _prepare_resource("rhs", var[by_block[blk].id], subslot=subslot, owner=blk)
            lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%s, %d, %s);"
                         % (scratch[blk], _resource_slot(), subslot, var[by_block[blk].id]))
        lines += _emit_coupled_rate_kernel(components, by_block, var, scratch)
        # Per-block names live in this emission's local token table. Codegen is a pure read of the
        # Program: repeated emission never writes scratch metadata back into frozen authoring state.
        var.update({("coupled_scratch", v.id, blk): scratch[blk] for blk in scratch})
        var[v.id] = scratch[next(iter(scratch))]     # a stable alias (the bundle has no single value)
    elif v.op == "coupled_rate_out":
        # Pure projection of one block out of the coupled bundle: its var aliases that block's rate
        # scratch (filled by the coupled_rate kernel above). Emits nothing -- like the FieldContext
        # alias of solve_fields_from_blocks. The producing coupled_rate is the node's sole input.
        (coupled_in,) = v.inputs
        var[v.id] = var[("coupled_scratch", coupled_in.id, v.attrs["out_block"])]
    elif v.op == "solve_coupled_implicit":
        components = _coupled_rate_components(program, v, model)
        by_block = {state.block: state for state in v.inputs}
        if target == "system":
            for block in components:
                index = _required_block_index(
                    block_idx, block, "preflight coupled implicit solve %r" % v.name)
                lines.append(
                    "ctx.require_cartesian_generated_operator(%d, %s);"
                    % (index, json.dumps("solve_coupled_implicit")))
        scratch = {}
        for subslot, block in enumerate(components):
            scratch[block] = "ci%d_%s" % (v.id, block_name(block))
            _prepare_resource("state", var[by_block[block].id], subslot=subslot, owner=block)
            lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, %d, %s);"
                         % (scratch[block], _resource_slot(), subslot, var[by_block[block].id]))
        status = "ci_status_%d" % v.id
        prototype_block = next(iter(components))
        prototype = var[by_block[prototype_block].id]
        _prepare_resource("scalar", prototype, ncomp=11, ghost_depth=0, owner=prototype_block)
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scalar_scratch(%s, 0, %s, 11, 0);"
                     % (status, _resource_slot(), prototype))
        lines += _emit_solve_coupled_implicit_kernel(
            components, by_block, var, scratch, status,
            controls=v.attrs, coefficient=v.attrs["coefficient"])
        report = "ci_report_%d" % v.id
        outcome = _append_local_nonlinear_report(program, v, status, report, lines)
        _append_solve_report_guard(
            program, v, outcome, lines, label="coupled_implicit")
        var.update({("coupled_solution", v.id, block): token
                    for block, token in scratch.items()})
        var[v.id] = scratch[next(iter(scratch))]
    elif v.op == "history":
        # Read the SYSTEM-OWNED history slot (a MultiFab&, ADC-406a): lag steps back. The reference
        # is bound to a C++ name the affine combine then reads like any other state/RHS term. An
        # explicit-ncomp read (ADC-427: the read-first 1-component cross-step carry) lowers to the
        # ZERO COLD-START variant -- its very first read (before any store) returns the zero-filled
        # slot, the declared step-0 value -- while the default multistep read keeps the fail-loud
        # ctx.history byte-identical (a store-first scheme reading before its store is a config error).
        var[v.id] = "h%d" % v.id
        if "ncomp" in v.attrs:
            if target == "amr_system" and bidx is not None:
                lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.history_zero_start(%s, %d, %d, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"]),
                                int(v.attrs["ncomp"]), bidx))
            else:
                lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.history_zero_start(%s, %d, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"]),
                                int(v.attrs["ncomp"])))
        else:
            if target == "amr_system":
                lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.history(%s, %d, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]),
                                int(v.attrs["lag"]), bidx))
            else:
                lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.history(%s, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"])))
    elif v.op == "store_history":
        # Side-effect: copy the value into the current slot of the history (the cold-start fill on
        # the first store happens System-side). store_history is a State-typed node but carries no
        # readable value -- nothing combines it. Its var maps to the stored value (a harmless alias).
        (value_in,) = v.inputs
        if target == "amr_system" and bidx is not None:
            lines.append("ctx.store_history(%s, %s, %d);"
                         % (json.dumps(v.attrs["history"]), var[value_in.id], bidx))
        else:
            lines.append("ctx.store_history(%s, %s);"
                         % (json.dumps(v.attrs["history"]), var[value_in.id]))
        var[v.id] = var[value_in.id]
    elif v.op == "fill_boundary":
        # Side effect on the field's ghosts (the valid cells are untouched). The result aliases the
        # input field (any subsequent op reading it sees the same C++ MultiFab, now with filled
        # halos). Forwards to ctx.fill_boundary (the shared transport-BC ghost exchange).
        (x,) = v.inputs
        lines.append("ctx.fill_boundary(%s);" % var[x.id])
        var[v.id] = var[x.id]
    elif v.op == "project":
        # In-place positivity projection of the state (the block's own project closure). The result
        # aliases the input state. Forwards to ctx.apply_projection(idx, state) (ADC-426: the op's
        # own block, so each block runs its own projection).
        (state_in,) = v.inputs
        step_projection = v.attrs.get("step_projection")
        if step_projection is not None:
            if not isinstance(step_projection, str) or not step_projection:
                raise TypeError("project step_projection must be a non-empty string")
        lines.append("ctx.apply_projection(%d, %s);" % (bidx, var[state_in.id]))
        if step_projection is not None:
            _append_transaction_authority_declaration(
                prelude,
                var,
                kind="step_projection",
                identity=step_projection,
            )
            lines.append("ctx.note_step_projection(%s);" % json.dumps(step_projection))
        var[v.id] = var[state_in.id]
    elif v.op == "local_transform":
        if prelude is None:
            raise NotImplementedError(
                "local_transform requires an install-time resource scope")
        state_in = v.inputs[0]
        var[v.id] = "u%d" % v.id
        status = "transform_status_field_%d" % v.id
        status_resource = "transform_status_resource_%d" % v.id
        active_mask = "transform_active_mask_%d" % v.id
        _prepare_resource("state", state_in)
        lines.append(
            "pops::MultiFab<pops::kNativeDimension>& %s = "
            "ctx.scratch_state(%s, 0, ctx.state(%d));"
            % (var[v.id], _resource_slot(), bidx))
        _prepare_resource("scalar", state_in, ncomp=1, ghost_depth=0)
        prelude.append(
            "auto* %s = &ctx.scalar_scratch(%s, 0, ctx.state(%d), 1, 0);"
            % (status_resource, _resource_slot(), bidx))
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = *%s;" % (status, status_resource))
        lines.append(
            "const pops::MultiFab<pops::kNativeDimension>* %s = ctx.pointwise_active_mask(%d, %s);"
            % (active_mask, bidx, var[v.id]))
        lines += _emit_local_transform_kernel(
            node_model, v.attrs["transform"], var[state_in.id], var[v.id], status,
            active_mask, bidx,
            provider_plans=provider_plans,
            consumer_qid=program_provider_consumer_qid(node_model, v.id, v.block),
        )
        reduced = "transform_failed_%d" % v.id
        lines.append(
            "const pops::Real %s = ctx.pointwise_status_max("
            "%d, %s, %s, ctx.prepared_execution_lane());"
            % (reduced, bidx, status, active_mask)
        )
        lines.append("if (%s != pops::Real(0)) {" % reduced)
        lines.append(
            "  throw pops::runtime::program::StepAttemptRejected("
            "pops::SolveStatus::kInvalidEvaluation, \"local_transform\", %s);"
            % json.dumps("transform '%s' rejected a non-finite or out-of-domain state"
                         % v.attrs["transform"]))
        lines.append("}")
    elif v.op == "cell_compare":
        # A PER-CELL threshold (spec op 17, ADC-418): mask(i,j,0) = field(i,j,0) <cmp> value ? 1 : 0,
        # a fresh 1-component scalar_field. Lowered to a for_each_cell select kernel (the mask the
        # `where` op selects on); no aux / model needed -- it reads component 0 of the input field.
        (field_in,) = v.inputs
        var[v.id] = "m%d" % v.id
        if target == "system":
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps("cell_compare")))
        _prepare_resource("scalar", var[field_in.id], ncomp=1, ghost_depth=1)
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scalar_scratch(%s, 0, %s, 1, 1);"
                     % (var[v.id], _resource_slot(), var[field_in.id]))
        lines += _emit_cell_compare_kernel(var[field_in.id], var[v.id], v.attrs["cmp"],
                                           v.attrs["value"])
    elif v.op == "where":
        # A PER-CELL conditional select (spec op 17, ADC-418): out(i,j,c) = mask ? a(i,j,c) :
        # b(i,j,c), COMPONENT-WISE. A fresh scratch the same shape as `a` (its vtype / ncomp); the
        # ternary is decided per cell inside the kernel (NOT the scalar lazy ``branch`` op).
        mask_in, a_in, b_in = v.inputs
        var[v.id] = "w%d" % v.id
        if target == "system":
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps("where")))
        _prepare_resource("state", var[a_in.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[a_in.id]))
        lines += _emit_where_kernel(var[mask_in.id], var[a_in.id], var[b_in.id], var[v.id])
    elif v.op == "record_scalar":
        # Store the (already-computed) Scalar into the System diagnostics map under its name. A
        # side-effecting op; its var maps to the recorded scalar (a harmless alias). The scalar input
        # is a 'reduce' result emitted earlier in the body (a const pops::Real local).
        (scalar_in,) = v.inputs
        _append_transaction_authority_declaration(
            prelude,
            var,
            kind="diagnostic",
            identity=v.attrs["diagnostic"],
        )
        lines.append("ctx.record_scalar(%s, %s);"
                     % (json.dumps(v.attrs["diagnostic"]), var[scalar_in.id]))
        var[v.id] = var[scalar_in.id]
    elif v.op == "record_balance_term":
        # Dedicated, non-bindable sink for a validated Program.record_balance term. Ordinary
        # record_scalar names cannot enter the reserved native attempt mailbox.
        from pops.codegen.program_balance_due import balance_record_due_expression

        (scalar_in,) = v.inputs
        _append_transaction_authority_declaration(
            prelude,
            var,
            kind="balance_route",
            identity=v.attrs["route"],
        )
        due = balance_record_due_expression(var, v.id)
        if due != "false":
            lines.append(
                "if (%s) { ctx.record_balance_term(%s, %s, %s); }"
                % (
                    due,
                    json.dumps(v.attrs["route"]),
                    json.dumps(v.attrs["term"]),
                    var[scalar_in.id],
                )
            )
        var[v.id] = var[scalar_in.id]
    elif v.op == "rhs":
        state_in = v.inputs[0]  # rhs inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        named_fluxes = _named_fluxes(v)
        requested = v.attrs.get("sources")
        named = [s for s in (requested or []) if s != "default"]
        if target == "system" and (named_fluxes is not None or named):
            operation = "named_flux" if named_fluxes is not None else "named_source"
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps(operation)))
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = "
                     "ctx.rhs_scratch(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[state_in.id]))
        _prepare_resource("rhs", var[state_in.id])
        named_source_subslot = 3
        want_flux = v.attrs.get("flux", True)
        # ADC-425 routing (spec criterion 17): the default/composite source is folded in iff the
        # caller did NOT exclude it -- i.e. sources is None (the legacy default) OR "default" is in
        # the explicit list. An EMPTY list [] (or a list of only named sources) excludes it -> flux
        # only. None and [] are recorded distinctly in the IR, so this is unambiguous.
        want_default_source = requested is None or "default" in requested
        if hasattr(v.point, "offset"):
            stage_point = v.point
        else:
            try:
                stage_point = v.point.time
            except ValueError:
                # A conservative flux belongs to the explicit partition of an ARK stage.  Its
                # implicit coordinate may differ and must never be silently substituted here.
                stage_point = v.point.time_for("explicit")
        stage = Fraction(stage_point.step) + Fraction(stage_point.offset.to_python())
        lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
        if not want_flux:
            # SOURCE-ONLY (ADC-430): flux=False -- NO -div F base (the rhs_scratch starts at zero).
            # The default/composite source is added iff requested (the same want_default_source
            # routing as flux=True): "default" present (or None) -> ctx.source_default_into (S only,
            # the exact mirror of neg_div_flux_default_into); excluded -> R stays the zeroed scratch.
            # The named source_terms below axpy on top either way -- so flux=False,sources=["default"]
            # is the default source only; flux=False,sources=["s"] is just s; flux=False,sources=[]
            # is the zero RHS. Named fluxes are rejected upstream (no flux base to divide). This is
            # the fix: before ADC-430 a flux=False stage still emitted the -div F base (it ignored the
            # flux attr), double-adding the flux on any non-zero-flux model in a Lie/Strang split.
            if want_default_source:
                lines.append("ctx.source_default_into(%d, %s, %s);"
                             % (bidx, var[state_in.id], var[v.id]))
        elif named_fluxes is None:
            if want_default_source:
                # R <- -div F + default/composite source (ctx.rhs_into) for THIS op's block (ADC-426
                # bidx), the historical path: sources is None (legacy) or "default" is requested.
                lines.append("ctx.rhs_into(%d, %s, %s, %d);"
                             % (bidx, var[state_in.id], var[v.id], int(v.id)))
            else:
                # FLUX-ONLY (ADC-425): "default" is NOT among the requested sources (the empty list
                # [] or a named-only list) -> R <- -div F(U) WITHOUT the model's default source
                # (ctx.neg_div_flux_default_into), for THIS op's block (bidx). The named source_terms
                # below are then axpy'd on top -- sources=[] is flux only, ["a","b"] is flux + a + b.
                lines.append("ctx.neg_div_flux_default_into(%d, %s, %s, %d);"
                             % (bidx, var[state_in.id], var[v.id], int(v.id)))
        else:
            # NAMED fluxes (ADC-419): R <- -div(sum of selected named fluxes). Evaluate the SUM of
            # the flux expressions into one exact-ranked scratch field per authored x[/y[/z] axis.
            # The retained block package first fills state/aux halos; one generic centered stencil
            # then accumulates every axis. There is no 2D fx/fy execution route.
            impl = _model_impl(node_model)
            axes = tuple(impl._flux_terms[named_fluxes[0]])
            flux_vars = {axis: "%s_f%s" % (var[v.id], axis) for axis in axes}
            lines.append("ctx.prepare_generated_state(%d, %s, %d);"
                         % (bidx, var[state_in.id], int(v.id)))
            for axis_index, axis in enumerate(axes):
                _prepare_resource("rhs", var[state_in.id], subslot=axis_index + 1)
                lines.append(
                    "pops::MultiFab<pops::kNativeDimension>& %s = "
                    "ctx.rhs_scratch(%s, %d, %s);"
                    % (flux_vars[axis], _resource_slot(), axis_index + 1, var[state_in.id])
                )
            named_source_subslot = 1 + len(axes)
        plan_exprs = []
        if named_fluxes is not None:
            for flux_name in named_fluxes:
                for axis_terms in impl._flux_terms[flux_name].values():
                    plan_exprs.extend(axis_terms)
        for source_name in named:
            plan_exprs.extend(_model_impl(node_model)._source_terms[source_name])
        consumer_qid = (
            program_provider_consumer_qid(node_model, v.id, v.block)
            if plan_exprs or named_fluxes is not None or named
            else None
        )
        if named_fluxes is not None:
            if consumer_qid is None:
                raise ValueError(
                    "named-flux kernel requires an explicit Program consumer qid"
                )
            # The named-flux kernel and named sources below are one Program node:
            # their shared plan is the first-use union, not an operator plan.
            lines += _emit_flux_kernel(
                node_model, named_fluxes, var[state_in.id], flux_vars, bidx,
                provider_plans=provider_plans, consumer_qid=consumer_qid,
                plan_exprs=plan_exprs,
            )
            lines.append(
                "ctx.neg_div_named_flux_into(%d, %s, %s, {%s}, %d);"
                % (
                    bidx,
                    var[state_in.id],
                    var[v.id],
                    ", ".join("&%s" % flux_vars[axis] for axis in axes),
                    int(v.id),
                )
            )
        for source_subslot, s in enumerate(named, start=named_source_subslot):
            # R += S_s(U, aux): assemble the named source into a scratch (same per-cell kernel as
            # the standalone 'source' op) and axpy it onto R.
            ssrc = "%s_%s" % (var[v.id], s)
            _prepare_resource("rhs", var[state_in.id], subslot=source_subslot)
            lines.append("pops::MultiFab<pops::kNativeDimension>& %s = "
                         "ctx.rhs_scratch(%s, %d, %s);"
                         % (ssrc, _resource_slot(), source_subslot, var[state_in.id]))
            if consumer_qid is None:
                raise ValueError(
                    "named-source kernel requires an explicit Program consumer qid"
                )
            lines += _emit_source_kernel(
                node_model, s, var[state_in.id], ssrc, bidx,
                provider_plans=provider_plans, consumer_qid=consumer_qid,
                plan_exprs=plan_exprs,
            )
            lines.append("ctx.axpy(%s, static_cast<pops::Real>(1), %s);" % (var[v.id], ssrc))
    elif v.op == "implicit_source":
        state_in = v.inputs[0]
        var[v.id] = "r%d" % v.id
        _prepare_resource("rhs", var[state_in.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[state_in.id]))
        lines.append("ctx.source_default_into(%d, %s, %s);"
                     % (bidx, var[state_in.id], var[v.id]))
        keep = ",".join(str(int(index)) for index in v.attrs["keep_components"])
        lines.append("ctx.apply_source_mask(%s, {%s});" % (var[v.id], keep))
    elif v.op == "solve_implicit_source":
        state_in = v.inputs[0]
        var[v.id] = "u%d" % v.id
        _prepare_resource("state", var[state_in.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[state_in.id]))
        lines.append(
            "ctx.lincomb(%s, static_cast<pops::Real>(1), %s, static_cast<pops::Real>(0), %s);"
            % (var[v.id], var[state_in.id], var[state_in.id]))
        outcome = "implicit_source_outcome_%d" % v.id
        report = "implicit_source_report_%d" % v.id
        lines.append(
            "pops::SolveOutcome %s = ctx.solve_source_default(%d, %s, static_cast<pops::Real>(dt), "
            "ctx.block_newton_options(%d));"
            % (outcome, bidx, var[v.id], bidx))
        lines.append(
            "const pops::SolveReport %s = pops::consume_solve_outcome(std::move(%s));"
            % (report, outcome))
        lines.append("ctx.publish_newton_report(%d, %s);" % (bidx, report))
    elif v.op == "source":
        state_in = v.inputs[0]  # source inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        if target == "system":
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps("named_source")))
        _prepare_resource("rhs", var[state_in.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[state_in.id]))
        lines += _emit_source_kernel(
            node_model, v.attrs["source"], var[state_in.id], var[v.id], bidx,
            provider_plans=provider_plans,
            consumer_qid=program_provider_consumer_qid(node_model, v.id, v.block),
        )
    elif v.op == "apply":
        state_in = v.inputs[0]  # apply inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        if target == "system":
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps("linear_source_apply")))
        _prepare_resource("rhs", var[state_in.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[state_in.id]))
        lines += _emit_apply_kernel(node_model, v.attrs["linear_source"], var[state_in.id], var[v.id],
                                    bidx, provider_plans=provider_plans,
                                    consumer_qid=program_provider_consumer_qid(node_model, v.id, v.block))
    elif v.op == "solve_local_linear":
        rhs_in = v.inputs[0]  # solve inputs = (rhs_state, op_value[, fields]); rhs first
        var[v.id] = "u%d" % v.id
        status = "local_solve_status_%d" % v.id
        if target == "system":
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps("solve_local_linear")))
        _prepare_resource("state", var[base.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[base.id]))
        _prepare_resource("scalar", var[v.id], ncomp=1, ghost_depth=0)
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scalar_scratch(%s, 0, %s, 1, 0);"
                     % (status, _resource_slot(), var[v.id]))
        lines += _emit_solve_local_linear_kernel(
            node_model, v.attrs["linear_source"], v.attrs["a_coeff"],
            var[rhs_in.id], var[v.id], status, bidx,
            provider_plans=provider_plans,
            consumer_qid=program_provider_consumer_qid(node_model, v.id, v.block),
        )
        _append_pointwise_solve_report(
            program, v, status, lines, label="local_linear", stem="local_solve")
    elif v.op == "solve_local_nonlinear":
        # Generated code supplies the residual and controls; the single prepared provider owns the
        # nonlinear algorithm and keeps the candidate private until the report is consumed.
        guess_in = v.inputs[0]  # solve inputs = (initial_guess,)
        var[v.id] = "u%d" % v.id
        status = "ln_status_%d" % v.id
        _prepare_resource("state", var[base.id])
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                     % (var[v.id], _resource_slot(), var[base.id]))
        _prepare_resource("scalar", var[base.id], subslot=1, ncomp=11, ghost_depth=0)
        lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scalar_scratch(%s, 1, %s, 11, 0);"
                     % (status, _resource_slot(), var[base.id]))
        active_mask = "local_solve_active_mask_%d" % v.id
        lines.append(
            "const pops::MultiFab<pops::kNativeDimension>* %s = ctx.pointwise_active_mask(%d, %s);"
            % (active_mask, bidx, status))
        lines += _emit_solve_local_nonlinear_kernel(
            node_model, v, var[guess_in.id], var[v.id], status, active_mask, bidx,
            provider_plans=provider_plans,
            consumer_qid=program_provider_consumer_qid(node_model, v.id, v.block),
        )
        report = "ln_report_%d" % v.id
        outcome = _append_local_nonlinear_report(program, v, status, report, lines)
        _append_solve_report_guard(
            program, v, outcome, lines, label="local_nonlinear")
    elif v.op == "scalar_field":
        # A step-body scratch scalar field (e.g. the explicit-flux buffer the RHS assembly fills):
        # a plan-slotted scratch reference reused every step. Inside an apply sub-block the
        # scalar_field is handled by _emit_matrix_free_operator instead (this branch is the
        # top-level / step-body path -- prelude is not None there).
        if prelude is None:
            raise NotImplementedError(
                "scalar_field is only lowerable at the top level / step body or inside a "
                "matrix_free_operator apply sub-block, not inside a control-flow (if/while/range) body")
        sp = "sf%d" % v.id
        ncomp = int(v.attrs.get("ncomp", 1))
        owner = _unique_resource_owner_block(
            program,
            v,
            where="prepare scalar_field resource %r" % v.name,
        )
        owner_index = _required_block_index(
            block_idx, owner, "prepare scalar_field resource %r" % v.name
        )
        _prepare_resource(
            "scalar",
            v,
            ncomp=ncomp,
            ghost_depth=1,
            owner=owner,
        )
        var[v.id] = sp
        lines.append(
            "pops::MultiFab<pops::kNativeDimension>& %s = "
            "ctx.scalar_scratch(%s, 0, ctx.state(%d), %d, 1);"
            % (sp, _resource_slot(), owner_index, ncomp)
        )
    elif v.op == "vector_field":
        if prelude is None:
            raise NotImplementedError(
                "vector_field is only lowerable at the top level / step body or inside a "
                "matrix_free_operator apply sub-block")
        sp = "sf%d" % v.id
        ncomp = int(v.attrs["ncomp"])
        owner = _unique_resource_owner_block(
            program,
            v,
            where="prepare vector_field resource %r" % v.name,
        )
        owner_index = _required_block_index(
            block_idx, owner, "prepare vector_field resource %r" % v.name
        )
        _prepare_resource(
            "scalar",
            v,
            ncomp=ncomp,
            ghost_depth=1,
            owner=owner,
        )
        var[v.id] = sp
        lines.append(
            "pops::MultiFab<pops::kNativeDimension>& %s = "
            "ctx.scalar_scratch(%s, 0, ctx.state(%d), %d, 1);"
            % (sp, _resource_slot(), owner_index, ncomp)
        )
        sources = ", ".join("&%s" % _deref(var[value.id]) for value in v.inputs)
        lines.append(
            "ctx.pack_vector(%s, std::array<const pops::MultiFab<"
            "pops::kNativeDimension>*, pops::kNativeDimension>{%s});"
            % (sp, sources))
    elif v.op == "laplacian":
        # Step-body bare Laplacian (e.g. Lap phi^n for the condensed RHS). Inside an apply sub-block
        # this op is handled by _emit_matrix_free_operator; here it is the top-level path.
        o, i = v.inputs
        if target == "system":
            owner = _cartesian_generated_owner_index(
                block_idx, v, where="preflight laplacian %r" % v.name)
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (owner, json.dumps("laplacian")))
        lines.append("ctx.laplacian(%s, %s);" % (_deref(var[o.id]), _deref(var[i.id])))
        var[v.id] = var[o.id]
    elif v.op == "gradient":
        o, p = v.inputs
        if target == "system":
            owner = _cartesian_generated_owner_index(
                block_idx, v, where="preflight gradient %r" % v.name)
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (owner, json.dumps("gradient")))
        lines.append("ctx.gradient(%s, %s);" % (_deref(var[o.id]), _deref(var[p.id])))
        var[v.id] = var[o.id]
    elif v.op == "divergence":
        o, flux = v.inputs
        if target == "system":
            owner = _cartesian_generated_owner_index(
                block_idx, v, where="preflight divergence %r" % v.name)
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (owner, json.dumps("divergence")))
        lines.append("ctx.divergence(%s, %s);"
                     % (_deref(var[o.id]), _deref(var[flux.id])))
        var[v.id] = var[o.id]
    elif v.op in ("condensed_coeffs", "condensed_rhs", "condensed_reconstruct", "condensed_energy"):
        # GENERIC condensed-implicit solve (ADC-637): the tensor coefficient A = I + c*rho*M^{-1} bundle,
        # the fused RHS -Lap(phi^n) - g*div(M^{-1}(m)), the velocity reconstruction and the kinetic-energy
        # increment, emitted INLINE via pops::detail::block_inverse<Dim> from an authored J (M = I -
        # th_dt*J) on a momentum subset -- no coupling/schur call. The thin dispatch lives in
        # program_emit_condensed to keep this router (and its budget) small; condensed_coeffs allocates
        # its persistent row-major tensor field there.
        if target == "system" and v.op == "condensed_rhs":
            lines.append(
                "ctx.require_cartesian_generated_operator(%d, %s);"
                % (bidx, json.dumps("condensed_rhs")))
        emit_condensed_op(
            v, var, node_model, lines, prelude,
            provider_plans=provider_plans,
            consumer_qid=program_provider_consumer_qid(node_model, v.id, v.block),
            program_block=_required_block_index(
                block_idx, v.block, "condensed op %r" % v.name
            ),
            target=target,
        )
    elif v.op == "matrix_free_operator":
        # Install-time: emit the apply lambda `apply_A{id}` into the prelude. Its persistent scratch
        # (the scalar_field ops of the apply sub-block) are shared_ptr fields, captured by value so
        # they outlive the install call and are reused across every Krylov iteration (alloc-once).
        # The lambda is itself captured by the step closure ([=]) and passed to pops::*_solve. An
        # rhs_jacvec apply (ADC-431) also captures persistent jac_uk / jac_r0 scratch the lambda
        # dereferences; the step body refreshes them from the live iterate / rhs(U^k) here (@p lines).
        _emit_matrix_free_operator(
            program, v, var, prelude, lines, field_plans=field_plans, target=target,
            has_shared_interface_implicit_jacvec=(
                has_shared_interface_implicit_jacvec
            ))
    elif v.op in ("apply_in", "apply_out", "apply_laplacian_coeff"):
        # The lambda in/out placeholders and the coefficiented apply matvec only appear INSIDE a
        # matrix_free_operator apply sub-block (lowered by _emit_matrix_free_operator); they never
        # lower standalone at the top level.
        raise NotImplementedError(
            "emit_cpp_program: op '%s' (value '%s') is only lowerable inside a matrix_free_operator "
            "apply sub-block" % (v.op, v.name))
    elif v.op == "solve_linear":
        _emit_solve_linear(program, v, base, var, prelude, lines, target=target)
    elif v.op in ("solve_outcome", "solve_outcome_component"):
        # Python graph/authoring requires an explicit consumed outcome before a solve result can feed
        # effects. The solve-producing operation has already constructed and guarded its native
        # SolveReport using the exact action attached to this outcome. These projections are therefore
        # zero-cost aliases of a scratch that is reachable only after the guard accepted it.
        (source,) = v.inputs
        if v.op == "solve_outcome_component" and "out_block" in v.attrs:
            solve = source.inputs[0]
            var[v.id] = var[("coupled_solution", solve.id, v.attrs["out_block"])]
        else:
            var[v.id] = var[source.id]
    elif v.op == "reduce":
        # A collective owner/block/layout/lane-authenticated raw active-domain algebra
        # reduction -> a C++ scalar. Inactive cells are excluded; there is no kappa/volume
        # weighting. Physical weighted integrals stay System services. Route every
        # operation through ProgramExecutionServices with the exact Program block owner. All ranks
        # execute the same context call, including ranks that own no box.
        var[v.id] = "s%d" % v.id
        kind = v.attrs["kind"]
        owner = _required_block_index(block_idx, v.block, "reduce value %r" % v.name)
        if kind == "norm2":
            (u,) = v.inputs
            reduction = "ctx.norm2(%d, %s)" % (owner, var[u.id])
        elif kind == "norm_inf":
            (u,) = v.inputs
            reduction = "ctx.norm_inf(%d, %s)" % (owner, var[u.id])
        elif kind in ("sum", "max", "min", "abs_sum"):
            (u,) = v.inputs
            comp = int(v.attrs.get("comp", 0))
            context_op = {
                "sum": "sum_component",
                "max": "max_component",
                "min": "min_component",
                "abs_sum": "abs_sum_component",
            }[kind]
            reduction = "ctx.%s(%d, %s, %d)" % (
                context_op,
                owner,
                var[u.id],
                comp,
            )
        else:  # dot
            a, b = v.inputs
            reduction = "ctx.dot(%d, %s, %s)" % (
                owner,
                var[a.id],
                var[b.id],
            )
        from pops.codegen.program_balance_due import balance_value_due_expression

        due = balance_value_due_expression(var, v.id)
        if due is not None:
            reduction = "(%s) ? (%s) : pops::Real(0)" % (due, reduction)
        lines.append("const pops::Real %s = %s;" % (var[v.id], reduction))
    elif v.op == "cfl":
        # The candidate dt-bound's runtime cfl argument. It is
        # NOT a statement; its token is the bound parameter name (spec s18 / ADC-417).
        var[v.id] = "cfl"
    elif v.op == "hmin":
        # MIN physical cell size (ctx.hmin(), = the native CFL's hmin). A scalar local (spec s18).
        var[v.id] = "s%d" % v.id
        lines.append("const pops::Real %s = ctx.hmin();" % var[v.id])
    elif v.op == "max_wave_speed":
        # Max |wave speed| of the block on the state (ctx.max_wave_speed(idx, u)): the SAME per-block
        # reduction the native CFL reads, REUSED (spec s18). A collective reduction -> a scalar local.
        # ADC-426: the wave speed of the input state's OWN block (idx of u.block).
        (u,) = v.inputs
        var[v.id] = "s%d" % v.id
        lines.append("const pops::Real %s = ctx.max_wave_speed(%d, %s);"
                     % (var[v.id], _required_block_index(
                         block_idx, u.block, "max_wave_speed input"), var[u.id]))
    elif v.op == "scalar_op":
        # Scalar arithmetic (add/sub/mul/div) over scalar locals / literal constants -> a new scalar
        # local. Used by the dt_bound expression cfl * hmin / max_wave_speed (spec s18).
        var[v.id] = "s%d" % v.id
        toks = []
        for kind, val in v.attrs["operands"]:
            if kind == "v":
                toks.append(var[v.inputs[val].id])
            else:  # a literal constant
                toks.append(scalar_cpp(val))
        cppop = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[v.attrs["fn"]]
        expression = "(%s %s %s)" % (toks[0], cppop, toks[1])
        from pops.codegen.program_balance_due import balance_value_due_expression

        due = balance_value_due_expression(var, v.id)
        if due is not None:
            expression = "(%s) ? (%s) : pops::Real(0)" % (due, expression)
        lines.append("const pops::Real %s = %s;" % (var[v.id], expression))
    elif v.op == "compare":
        # A predicate over scalars -> an inline boolean C++ expression (no statement of its own; the
        # while op embeds it directly in `if (!(<expr>)) break;`).
        lhs = v.inputs[0]
        if len(v.inputs) == 2:  # scalar vs scalar
            rhs_tok = var[v.inputs[1].id]
        else:  # scalar vs float tolerance
            rhs_tok = scalar_cpp(v.attrs["rhs"])
        var[v.id] = "(%s %s %s)" % (var[lhs.id], v.attrs["cmp"], rhs_tok)
        var[("when_predicate", v.id)] = var[v.id]  # emission-local schedule predicate token
    elif v.op == "acceptance_guard":
        value, condition = v.inputs
        action = v.attrs["action"]
        from pops.time.solve_outcome import FailRun, RejectAttempt
        message = "acceptance guard %r failed" % v.attrs["guard"]
        lines.append("if (!(%s)) {" % var[condition.id])
        if type(action) is RejectAttempt:
            lines.append(
                "  throw pops::runtime::program::StepAttemptRejected("
                "pops::SolveStatus::kInvalidEvaluation, \"guard\", %s);"
                % json.dumps(message))
        elif type(action) is FailRun:
            lines.append("  throw std::runtime_error(%s);" % json.dumps(message))
        else:  # guarded by Program.guard; defense in depth for a forged IR
            raise TypeError("acceptance_guard has an unsupported terminal action")
        lines.append("}")
        var[v.id] = var[value.id]
    elif v.op == "while":
        _emit_while(
            program, v, base, var, model, lines, prelude, block_idx, field_plans,
            target=target)
    elif v.op == "range":
        _emit_range(
            program, v, base, var, model, lines, prelude, block_idx, field_plans,
            target=target)
    elif v.op == "subcycle":
        _emit_subcycle(
            program, v, base, var, model, lines, prelude, block_idx, field_plans, target=target)
    elif v.op == "branch":
        _emit_branch(
            program, v, base, var, model, lines, prelude, block_idx, field_plans,
            target=target)
    elif v.op == "linear_combine":
        terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
        if v.id in committed_ids:
            # Commit: block state <- c_base * base + sum(non-base coeff * term), in place.
            c_base = {0: 0}
            acc = "acc%d" % v.id
            _prepare_resource("state", var[base.id])
            lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                         % (acc, _resource_slot(), var[base.id]))
            for inp, coeff in terms:
                if inp.id == base.id:
                    c_base = coeff
                else:
                    lines.append("ctx.axpy(%s, %s, %s, dt, %s);"
                                 % (acc, _coeff_cpp(coeff), var[inp.id],
                                    _coeff_metadata_cpp(coeff)))
            lines.append(
                "ctx.lincomb(%s, %s, %s, static_cast<pops::Real>(1), %s, dt, %s, {{0, 1, 1}});"
                % (var[base.id], _coeff_cpp(c_base), var[base.id], acc,
                   _coeff_metadata_cpp(c_base)))
            var[v.id] = var[base.id]  # the commit wrote the block state in place (no final copy)
        else:
            var[v.id] = "u%d" % v.id  # an intermediate stage state (scratch, zero-initialized)
            # A scalar_field combine (ADC-427: the phi^{n+1} extrapolation) has no block, so it has no
            # base block-state to shape the scratch: template it on the FIRST scalar input instead (a
            # 1-component field, same (ba, dm)). A State combine shapes it on the block base as before.
            template = var[terms[0][0].id] if v.vtype == "scalar_field" else var[base.id]
            _prepare_resource("state", template)
            lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.scratch_state(%s, 0, %s);"
                         % (var[v.id], _resource_slot(), template))
            for inp, coeff in terms:
                lines.append("ctx.axpy(%s, %s, %s, dt, %s);"
                             % (var[v.id], _coeff_cpp(coeff), var[inp.id],
                                _coeff_metadata_cpp(coeff)))
    # UNIFIED SCHEDULER (ADC-458, Spec 3 sections 17-18): if this op carries a non-always schedule,
    # wrap the statements it just emitted (lines[_profile_start:]) in the due-test guard + policy
    # branch. Done HERE, after the op lowered itself, so EVERY schedulable node (field solve, rhs,
    # source, linear_combine, where, ...) reuses the one general mechanism -- no per-op special
    # case. The wrap nests INSIDE the per-node profiling pair below (the profiler times the guarded
    # block as the node's cost). An always() schedule (or no schedule) leaves the lines untouched.
    _emit_schedule_wrap(
        program,
        v,
        var,
        lines,
        _profile_start,
        target=target,
        prelude=prelude,
        program_block=resource_program_block,
    )
    # PER-NODE PROFILING (ADC-459): if this op emitted at least one statement, bracket those
    # statements with the steady_clock pair (see the note at the top of _emit_op). A ProfileScope is
    # named "node:<v.name>"; profile_record(name, _pt) accumulates now() - _pt into the System
    # Profiler. Inserted only when lines grew (a pure inline-token op emits nothing and is skipped).
    # The pure reference-binding ops (state / history bind a MultiFab&; hmin reads a cached scalar)
    # do no per-step numerical work, so they are not wrapped -- the report keeps the meaningful
    # work nodes (rhs / solve_fields / linear_combine / source / apply / reductions / loops).
    if v.op not in _PROFILE_SKIP_OPS and len(lines) > _profile_start:
        node_name = json.dumps("node:%s" % v.name)
        pt = "_pt%d" % v.id  # unique per node id (no redefinition at body scope or in a loop pass)
        lines.insert(_profile_start,
                     "const auto %s = std::chrono::steady_clock::now();  // ProfileScope %s"
                     % (pt, node_name))
        lines.append("ctx.profile_record(%s, %s);" % (node_name, pt))
