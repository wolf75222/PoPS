"""pops.codegen.program_emit_solve : matrix-free Krylov op emitters.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  These leaf emitters (called from
``program_emit_ops._emit_op`` for the matrix_free_operator / solve_linear ops) build
install-time apply lambdas + the Krylov solve calls; they never recurse back into the op
dispatcher.  They reuse the shared primitives in ``program_emit_kernels``.
"""
from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pops.identity.scalar import exact_cpp_int, scalar_cpp
from pops.fields._prepared_nullspace_registry import (
    prepared_nullspace_contracts_from_attrs,
)
from pops.solvers._prepared_preconditioner_registry import (
    prepared_preconditioner_provider_from_attrs,
)
from pops.solvers.krylov._prepared_method_registry import (
    prepared_krylov_method_provider_from_attrs,
)
from pops.solvers.providers import (
    PreparedHierarchySolverEmitRequest,
    prepared_hierarchy_solver_provider_from_attrs,
)
from pops.time.points import StagePoint, TimePoint

from pops.codegen.program_emit_kernels import (
    _apply_in_arg,
    _coeff_cpp,
    _emit_field_combine,
)
from pops.codegen.krylov_contract import (
    validated_krylov_footprint,
    validated_prepared_problem_contract,
)


def _program_nodes(program: Any) -> Any:
    """Iterate top-level and nested ProgramValue nodes without importing lowerability helpers."""
    def walk(value: Any) -> Any:
        yield value
        for key in ("cond_block", "body_block", "apply_block", "residual_block",
                    "true_block", "false_block"):
            block = value.attrs.get(key)
            if isinstance(block, (list, tuple)):
                for nested in block:
                    yield from walk(nested)

    for value in program._values:
        yield from walk(value)


_SOLVE_STATUS_CPP = {
    "singular": "pops::SolveStatus::kSingular",
    "breakdown": "pops::SolveStatus::kBreakdown",
    "iteration_limit": "pops::SolveStatus::kIterationLimit",
    "invalid_evaluation": "pops::SolveStatus::kInvalidEvaluation",
    "capability_failure": "pops::SolveStatus::kCapabilityFailure",
    "invalid_input": "pops::SolveStatus::kInvalidInput",
    "incompatible_rhs": "pops::SolveStatus::kIncompatibleRhs",
}


def _consumed_solve_action(program: Any, solve: Any) -> tuple[str, tuple[str, ...]]:
    """Return the complete canonical action attached to the unique outcome consumer."""
    matches = []
    for node in _program_nodes(program):
        if node.op != "solve_outcome" or len(node.inputs) != 1 or node.inputs[0] is not solve:
            continue
        action = node.attrs.get("action")
        kind = str(getattr(action, "kind", ""))
        statuses = getattr(action, "statuses", None)
        if (
            kind not in ("fail_run", "reject_attempt")
            or not isinstance(statuses, tuple)
            or not statuses
            or any(type(status) is not str or status not in _SOLVE_STATUS_CPP
                   for status in statuses)
            or len(set(statuses)) != len(statuses)
        ):
            raise ValueError("solve outcome action is not canonical")
        matches.append((kind, statuses))
    if len(matches) != 1:
        raise ValueError(
            "solve %r must have exactly one explicit outcome.consume(action=FailRun(...) or "
            "RejectAttempt(...)); found %d" % (solve.name, len(matches)))
    return matches[0]


def _validate_matrix_free_contract(v: Any, model: Any) -> None:
    """Validate matrix-free facts that need either the final node or physical model metadata."""
    if v.op == "rhs_jacvec":
        if len(v.inputs) != 4:
            raise ValueError("rhs_jacvec IR requires out, direction, iterate, and r0 inputs")
        iterate, r0 = v.inputs[2], v.inputs[3]
        named = [source for source in (v.attrs.get("sources") or ()) if source != "default"]
        if not isinstance(v.attrs.get("field_coupled"), bool):
            raise ValueError("rhs_jacvec IR requires an explicit boolean field_coupled attribute")
        if v.attrs.get("flux") is not True or named:
            raise NotImplementedError(
                "rhs_jacvec lowers only the default flux with sources=[] or ['default']; "
                "got flux=%r, named_sources=%r" % (v.attrs.get("flux"), named))
        if getattr(r0, "op", None) != "rhs" or len(getattr(r0, "inputs", ())) < 1:
            raise ValueError("rhs_jacvec r0 must be an exact precomputed rhs(iterate) IR node")
        if r0.inputs[0] is not iterate:
            raise ValueError("rhs_jacvec r0 must be computed from the exact frozen iterate")
        if r0.block != iterate.block or r0.point != iterate.point:
            raise ValueError(
                "rhs_jacvec r0 and iterate must share one exact block and temporal point")
        expected_sources = v.attrs.get("sources")
        actual_sources = r0.attrs.get("sources")
        if actual_sources is not None:
            actual_sources = list(actual_sources)
        if expected_sources is not None:
            expected_sources = list(expected_sources)
        if (r0.attrs.get("flux") is not True or actual_sources != expected_sources
                or r0.attrs.get("fluxes") not in (None, (), [])):
            raise ValueError(
                "rhs_jacvec r0 must use the exact same default-flux/default-source selection "
                "as the Jacobian-vector product and no named flux")
        context = getattr(r0, "field_context", None)
        if v.attrs["field_coupled"]:
            field = getattr(context, "field", None)
            stage_sources = tuple(getattr(context, "stage_sources", ()))
            if field is None or stage_sources != ((iterate.block, iterate.id),):
                raise ValueError(
                    "rhs_jacvec field coupling requires one unambiguous field context solved "
                    "only from the frozen iterate")
        elif context is not None:
            raise ValueError(
                "rhs_jacvec field_coupled=False requires an r0 with no field-solve provenance")
        return
    if v.op != "solve_linear":
        return
    rhs = v.inputs[1]
    if rhs.vtype != "state" or rhs.space is not None or model is None:
        return
    impl = getattr(model, "_m", model)
    model_ncomp = getattr(impl, "n_cons", None)
    if model_ncomp is None:
        model_ncomp = getattr(impl, "n_vars", None)
    if model_ncomp is None:
        names = getattr(impl, "cons_names", None)
        if names is None:  # opaque native ModelSpec: no truthful Python-side component metadata
            return
        model_ncomp = len(names)
    declared = int(v.attrs["ncomp"])
    if declared != int(model_ncomp):
        raise ValueError(
            "solve_linear: untyped State rhs uses operator ncomp=%d but the physical model "
            "declares n_cons=%d" % (declared, int(model_ncomp)))


def _rhs_stage_fraction(value: Any) -> Fraction:
    """Return the exact explicit residual coordinate carried by one RHS-like IR value.

    A partitioned stage may expose distinct explicit and implicit coordinates.  Conservative RHS
    evaluation belongs to the explicit partition, exactly as in the top-level RHS emitter.  This
    helper deliberately accepts only the typed temporal IR: a missing/opaque point is a codegen
    error, never a reason to invent stage zero for a matrix-free callback that will outlive the
    authoring scope.
    """
    point = getattr(value, "point", None)
    if type(point) is TimePoint:
        stage_point = point
    elif type(point) is StagePoint:
        try:
            stage_point = point.time
        except ValueError:
            try:
                stage_point = point.time_for("explicit")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "rhs_jacvec r0 requires an exact explicit StagePoint coordinate"
                ) from exc
    else:
        raise ValueError(
            "rhs_jacvec r0 requires an exact TimePoint or StagePoint in the Program IR")
    try:
        return Fraction(stage_point.step) + Fraction(stage_point.offset.to_python())
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            "rhs_jacvec r0 carries no exact stage fraction") from exc


def _solve_stage_fraction(value: Any) -> Fraction:
    """Return the exact solve evaluation coordinate, preferring the implicit partition."""
    point = getattr(value, "point", None)
    if type(point) is TimePoint:
        time_point = point
    elif type(point) is StagePoint:
        try:
            time_point = point.time
        except ValueError:
            for partition in ("implicit", "explicit"):
                try:
                    time_point = point.time_for(partition)
                    break
                except (KeyError, TypeError, ValueError):
                    continue
            else:
                raise ValueError("solve_linear carries no exact implicit stage coordinate")
    else:
        raise ValueError("solve_linear requires an exact TimePoint or StagePoint")
    try:
        return Fraction(time_point.step) + Fraction(time_point.offset.to_python())
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("solve_linear carries no exact stage fraction") from exc


def _rhs_jacvec_field_slot(r0: Any, field_plans: Any) -> str:
    """Resolve the exact FieldContext captured by r0 to its installed native provider slot."""
    context = getattr(r0, "field_context", None)
    field = getattr(context, "field", None)
    if field is None:
        raise ValueError("field-coupled rhs_jacvec r0 has no exact FieldContext identity")
    if len(getattr(r0, "inputs", ())) != 2:
        raise ValueError(
            "field-coupled rhs_jacvec r0 must consume exactly its iterate and field solve")
    fields = r0.inputs[1]
    if (getattr(fields, "vtype", None) != "fields"
            or getattr(fields, "field_context", None) != context):
        raise ValueError(
            "field-coupled rhs_jacvec r0 field input disagrees with its FieldContext")
    from pops.codegen.program_emit_field_routes import resolved_field_route
    slot, _ = resolved_field_route(field, field_plans)
    if not isinstance(slot, str) or not slot:
        raise ValueError("field-coupled rhs_jacvec resolved an invalid native provider slot")
    return slot


def _emit_matrix_free_operator(program: Any, v: Any, var: Any, prelude: Any,
                               lines: Any = None, *, field_plans: Any = None) -> None:
    """Lower a matrix_free_operator to an authenticated factory of C++ execution sessions. Each
    session owns a fresh ``ApplyFn`` and deep-copied scratch snapshot; its body re-emits the apply
    sub-block:

      - each ``scalar_field`` scratch -> a template field refreshed before preparation, then a
        workspace-private deep copy reused across every Krylov iteration (alloc-once per session);
      - ``laplacian(o, i)`` -> ``ctx.laplacian(*o, i)`` (i const_cast when it is the lambda's ``in``,
        which is logically read-only -- the fill only writes ghosts, as in test_generic_krylov);
      - ``rhs_jacvec(out, in, iterate, r0, ...)`` (ADC-431) -> a finite-difference Jacobian-vector
        product over the core residual plus the exact prepared-boundary JVP.  The lambda captures one
        shared ``BoundaryEvaluationPoint`` refreshed from r0's exact stage in the step body, freezing
        that point even if later operators advance the shared context stage. Boundary-only scratch is
        allocated once and only when that block has an installed boundary linearization;
      - the apply RESULT (the affine the body returned, e.g. ``in - alpha*Lap(in)``) is written into
        ``out`` via the same accumulate-then-lincomb idiom as a linear_combine commit.

    The factory captures ``[ctx_owner, <snapshot templates>]``; each returned apply callback owns
    private mutable fields/scalars while retaining the installed context authority. @p
    lines is the mandatory step-body scope used to resolve the live fields captured by an optional
    rhs_jacvec.  The resulting refresh statements and both live-dt pointees are attached to the
    operator token and emitted at each solve site, immediately before native preparation.  A
    matrix-free operator cannot be lowered in a control-flow-local scope because its install-time
    ApplyFn would otherwise have no authenticated evaluation-lifetime source for those values."""
    apply_id = v.id
    lam = "apply_A%d" % apply_id
    var[apply_id] = lam
    in_sf = v.attrs["apply_in"]
    out_sf = v.attrs["apply_out"]
    block = v.attrs["apply_block"]
    result = v.attrs["apply_result"]
    # Sub-scope token map: the lambda params + persistent scratch. `in` is the const lambda param;
    # `out` is the (non-const) lambda param the result is written into.
    sub = {in_sf.id: "in", out_sf.id: "out"}
    # 1) Evaluation templates. The step body refreshes these before problem.prepare(); each problem
    #    or workspace session factory invocation deep-copies them into private mutable state.
    scratch = [w for w in block if w.op == "scalar_field"]
    captures = ["ctx_owner"]
    session_fields = []
    session_optional_fields = []
    session_scalars = []
    session_points = []
    session_direct = ["ctx_owner"]
    session_dynamic = []
    if lines is None:
        raise NotImplementedError(
            "matrix-free operators require a top-level prepared evaluation scope")
    prepare_refresh = []
    operator_dt = "operator_dt%d" % apply_id
    prelude.append(
        "auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));"
        % operator_dt)
    captures.append(operator_dt)
    session_scalars.append(operator_dt)
    for w in scratch:
        sp = "sf%d_%d" % (apply_id, w.id)
        sub[w.id] = sp
        ncomp = int(w.attrs.get("ncomp", 1))  # >1 for a gradient buffer consumed by divergence
        prelude.append(
            "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
            % (sp, ncomp))
        captures.append(sp)
        session_fields.append(sp)
    # The affine result-write accumulator: one PERSISTENT shared_ptr (alloc-once, like the scratch),
    # zeroed and reused every matvec instead of allocated per call -- so the apply lambda allocates
    # NOTHING per Krylov iteration (the runtime r/p/Ap scratch in generic_krylov.hpp is likewise
    # alloc-once). _emit_field_combine writes the affine into `out` through it. It carries the
    # operator's component count so the axpy / lincomb cover ALL components (a vector / state apply).
    op_ncomp = int(v.attrs["ncomp"])
    acc_sp = "acc%d" % apply_id
    prelude.append(
        "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
        % (acc_sp, op_ncomp))
    captures.append(acc_sp)
    session_fields.append(acc_sp)
    # The ApplyFn is constructed at install time, outside ``ctx.install([=](double dt) {...})``,
    # while affine apply bodies evaluate exact dt-polynomial coefficients and pass the current dt
    # to the conservative axpy/lincomb ledger.  Carry the live step value through one persistent
    # scalar, exactly like the rhs_jacvec coefficient capture below.  Reusing a single value is safe:
    # a ProgramContext invokes one matrix-free ApplyFn synchronously within its owning step.
    apply_dt = "apply_dt%d" % apply_id
    prelude.append(
        "auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));" % apply_dt)
    captures.append(apply_dt)
    session_scalars.append(apply_dt)
    var[("operator_dt_captures", apply_id)] = (operator_dt, apply_dt)
    # A coefficiented apply (apply_laplacian_coeff) reads an OUTER condensed_coeffs bundle (assembled in
    # the step body, before the operator): capture its four coefficient shared_ptrs (already
    # allocated in the prelude by emit_condensed_op) so the lambda can dereference them.
    frozen_coefficients = {}
    freeze_pairs = []
    for w in block:
        if w.op == "apply_laplacian_coeff":
            coeffs = w.inputs[2]
            for sp in var[coeffs.id]:
                if sp in frozen_coefficients:
                    continue
                frozen = "frozen_A%d_%d" % (apply_id, len(frozen_coefficients))
                prelude.append(
                    "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(1, 1));"
                    % frozen)
                frozen_coefficients[sp] = frozen
                freeze_pairs.append((sp, frozen))
                captures.append(frozen)
                session_fields.append(frozen)
    freeze_name = "freeze_A%d" % apply_id
    if freeze_pairs:
        freeze_captures = []
        for live, frozen in freeze_pairs:
            freeze_captures.extend((live, frozen))
        prelude.append("pops::PreparedResourceFn %s = [%s]() {" %
                       (freeze_name, ", ".join(freeze_captures)))
        for live, frozen in freeze_pairs:
            # Tensor face/cross stencils read coefficient neighbours.  The live condensed fields
            # have already completed their typed ghost production; freezing only valid cells would
            # silently replace multibox/interface coefficients by stale or zero halo values.
            prelude.append(
                "  pops::PureFieldAlgebra::copy_allocated(*%s, *%s);" % (frozen, live))
        prelude.append("};")
        var[("operator_freeze", apply_id)] = freeze_name
    else:
        var[("operator_freeze", apply_id)] = "pops::PreparedResourceFn{}"
    # An rhs_jacvec apply (ADC-431, implicit-flux BDF) needs the FROZEN Newton iterate U^k and its
    # precomputed rhs(U^k) inside the lambda. They are step-body locals that CHANGE each Newton
    # iteration, so -- like schur_coeffs -- they become PERSISTENT shared_ptr scratch (jac_uk / jac_r0)
    # captured by value (shared pointee), refreshed from the live iterate / r0 in the step body BEFORE
    # the solve. Plus a perturbed-state scratch (jac_up) and a perturbed-rhs scratch (jac_rp) the
    # lambda fills per matvec. All carry the operator's component count (= the block n_cons).  The
    # exact BoundaryEvaluationPoint is a shared pointee because it must remain frozen at r0's stage
    # while other operator nodes may advance the shared context to a later stage.
    jac_ops = [w for w in block if w.op == "rhs_jacvec"]
    jac_scratch = {}
    # jacvec op id -> (uk, r0, up, rp, r0_core, boundary_work, point, has_boundary,
    #                  field_slot, cdt, block_idx) names/provenance
    for w in jac_ops:
        _validate_matrix_free_contract(w, None)
        iterate_in, r0_in = w.inputs[2], w.inputs[3]
        indices = program._block_indices()
        if iterate_in.block not in indices:
            raise ValueError(
                "rhs_jacvec iterate block %r has no declared Program state" % iterate_in.block)
        block_idx = indices[iterate_in.block]
        ng_state = "ctx.state(%d).n_grow()" % block_idx
        uk = "jac_uk%d_%d" % (apply_id, w.id)
        r0 = "jac_r0%d_%d" % (apply_id, w.id)
        up = "jac_up%d_%d" % (apply_id, w.id)
        rp = "jac_rp%d_%d" % (apply_id, w.id)
        for sp in (uk, r0, up, rp):
            prelude.append(
                "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, %s));"
                % (sp, op_ncomp, ng_state))
            captures.append(sp)
            session_fields.append(sp)
        point = "jac_point%d_%d" % (apply_id, w.id)
        prelude.append(
            "auto %s = std::make_shared<"
            "pops::runtime::multiblock::BoundaryEvaluationPoint>();" % point)
        captures.append(point)
        session_points.append(point)
        has_boundary = "jac_has_boundary%d_%d" % (apply_id, w.id)
        prelude.append(
            "const bool %s = ctx.has_boundary_linearization(%d);"
            % (has_boundary, block_idx))
        captures.append(has_boundary)
        session_direct.append(has_boundary)
        # Krylov invokes this ApplyFn sequentially.  Reuse one boundary buffer first for C(U^k) in
        # the step-body refresh, then for C'(U^k)v in each matvec.  Both conditional allocations are
        # skipped entirely for the ordinary no-boundary-linearization path.
        r0_core = "jac_r0_core%d_%d" % (apply_id, w.id)
        boundary_work = "jac_boundary_work%d_%d" % (apply_id, w.id)
        for sp in (r0_core, boundary_work):
            prelude.append(
                "auto %s = %s ? std::make_shared<pops::MultiFab>("
                "ctx.alloc_scalar_field(%d, %s)) : std::shared_ptr<pops::MultiFab>{};"
                % (sp, has_boundary, op_ncomp, ng_state))
            captures.append(sp)
            session_optional_fields.append(sp)
        field_slot = None
        if w.attrs["field_coupled"]:
            field_slot = "jac_field_slot%d_%d" % (apply_id, w.id)
            resolved_slot = _rhs_jacvec_field_slot(r0_in, field_plans)
            prelude.append(
                "const std::string %s = %s;" % (field_slot, json.dumps(resolved_slot)))
            captures.append(field_slot)
            session_direct.append(field_slot)
        # The BDF coefficient c*dt depends on the step's dt (the step-closure parameter), which the
        # install-time lambda cannot see; carry it through a captured shared_ptr<Real> the step body
        # sets to its dt value before the solve (the same persistent-scratch idiom as jac_uk).
        cdt = "jac_cdt%d_%d" % (apply_id, w.id)
        prelude.append("auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));" % cdt)
        captures.append(cdt)
        session_scalars.append(cdt)
        metric_scratch = "jac_metric_scratch%d_%d" % (apply_id, w.id)
        session_dynamic.append(
            (metric_scratch,
             "std::make_shared<std::vector<double>>("
             "ctx_owner->program_resource_vector_distribution()."
             "reduction_scratch_value_count("
             "pops::detail::PreparedFieldAlgebra::kRobustDotPayloadWidth), 0.0)"))
        jac_scratch[w.id] = (
            uk, r0, up, rp, r0_core, boundary_work, point, has_boundary,
            field_slot, cdt, block_idx, metric_scratch)
        # Step body: first restore the exact StagePoint of r0 and snapshot it into the shared point;
        # then refresh the frozen U^k / rhs(U^k) / dt captures.  Prepared boundary residuals are
        # removed from the frozen base so the finite difference covers only the core residual; their
        # derivative is supplied separately by boundary_jvp_into_at in the ApplyFn.
        stage = _rhs_stage_fraction(r0_in)
        prepare_refresh.append(
            "ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
        prepare_refresh.append(
            "*%s = ctx.boundary_evaluation_point(%d);" % (point, int(r0_in.id)))
        prepare_refresh.append(
            "pops::PureFieldAlgebra::copy(*%s, %s);" % (uk, var[iterate_in.id]))
        prepare_refresh.append(
            "pops::PureFieldAlgebra::copy(*%s, %s);" % (r0, var[r0_in.id]))
        prepare_refresh.append("*%s = %s;" % (cdt, _coeff_cpp(w.attrs["c_dt"])))
    boundary_sessions = {}
    for block_idx in sorted({entry[-2] for entry in jac_scratch.values()}):
        prototype_entry = next(
            entry for entry in jac_scratch.values() if entry[-2] == block_idx)
        prototype = prototype_entry[2]
        preparation_point = prototype_entry[6]
        name = "operator_boundary_session%d_%d" % (apply_id, block_idx)
        session_dynamic.append(
            (name,
             "ctx_owner->prepare_block_boundary_session(%d, *session_%s, "
             "*session_%s, lane)" % (block_idx, prototype, preparation_point)))
        boundary_sessions[block_idx] = name
    stencil_ops = {
        "laplacian", "gradient", "divergence", "apply_laplacian_coeff",
    }
    has_stencil = any(w.op in stencil_ops for w in block)
    stencil_boundary = None
    stencil_point = None
    if has_stencil and len(jac_ops) == 1:
        jac_entry = jac_scratch[jac_ops[0].id]
        stencil_boundary = boundary_sessions[jac_entry[-2]]
        stencil_point = jac_entry[6]
    elif has_stencil:
        stencil_boundary = "operator_mesh_boundary_session%d" % apply_id
        session_dynamic.append(
            (stencil_boundary,
             "ctx_owner->prepare_mesh_boundary_session(*session_%s, lane)" % acc_sp))
    var[("operator_prepare_refresh", apply_id)] = tuple(prepare_refresh)
    # 2) The lambda body: the laplacian / gradient ops + the result write into `out`.
    body = ["const pops::Real dt = *%s;" % apply_dt]
    for w in block:
        if w.op in ("scalar_field", "apply_in", "apply_out"):
            continue  # scratch shared_ptr / lambda params: already bound in `sub`, nothing to emit
        if w.op == "laplacian":
            o, i = w.inputs
            sub[w.id] = sub[o.id]
            point_arg = ", *%s" % stencil_point if stencil_point else ""
            body.append("ctx.laplacian(*%s, %s, *%s%s);"
                        % (sub[o.id], _apply_in_arg(sub, i), stencil_boundary, point_arg))
        elif w.op == "gradient":
            o, p = w.inputs
            sub[w.id] = sub[o.id]
            point_arg = ", *%s" % stencil_point if stencil_point else ""
            body.append("ctx.gradient(*%s, %s, *%s%s);"
                        % (sub[o.id], _apply_in_arg(sub, p), stencil_boundary, point_arg))
        elif w.op == "divergence":
            o, fx, fy = w.inputs
            sub[w.id] = sub[o.id]
            point_arg = ", *%s" % stencil_point if stencil_point else ""
            body.append("ctx.divergence(*%s, %s, %s, *%s%s);"
                        % (sub[o.id], _apply_in_arg(sub, fx), _apply_in_arg(sub, fy),
                           stencil_boundary, point_arg))
        elif w.op == "apply_laplacian_coeff":
            # out = div(A grad in), A the coefficient tensor of a condensed_coeffs bundle (ADC-637): the
            # SAME two steps the retired brick wrapper did, emitted INLINE through Schur-free seams --
            # ctx.fill_boundary(in) (the transport-BC ghost fill) then the pops::apply_laplacian
            # coefficient floor -- so a generated .so compiles without coupling/schur/** and the operator
            # arithmetic is bit-identical (eps_x/eps_y/a_xy/a_yx are the captured coeff fields).
            o, i, coeffs = w.inputs
            ex, ey, axy, ayx = (
                frozen_coefficients[name] for name in var[coeffs.id])
            sub[w.id] = sub[o.id]
            point_arg = ", *%s" % stencil_point if stencil_point else ""
            body.append("ctx.tensor_laplacian(*%s, %s, *%s, *%s, *%s, *%s, "
                        "*%s%s);"
                        % (sub[o.id], _apply_in_arg(sub, i), ex, ey, axy, ayx,
                           stencil_boundary, point_arg))
        elif w.op == "rhs_jacvec":
            # out = J(U^k) in = in - (c*dt/h)(rhs(U^k + h*in) - rhs(U^k)), the finite-difference
            # Jacobian-vector product of the implicit-flux BDF residual (ADC-431). h is a relatively
            # scaled FD step (Brown-Saad / WP: h = eps*(1+||U^k||)/||in||, eps the relative step). The
            # captured jac_uk / jac_r0 hold U^k and rhs(U^k) (refreshed in the step body); jac_up /
            # jac_rp are per-matvec scratch; jac_cdt holds c*dt. The op writes directly into `out`.
            o, i = w.inputs[0], w.inputs[1]
            (uk, r0, up, rp, r0_core, boundary_work, point, has_boundary,
             field_slot, cdt, block_idx, metric_scratch) = jac_scratch[w.id]
            boundary_session = boundary_sessions[block_idx]
            in_arg = _apply_in_arg(sub, i)        # the Krylov vector v (the lambda's const `in`)
            out_tok = sub[o.id]                   # the apply out buffer (== "out")
            eps = scalar_cpp(w.attrs["eps"])
            sub[w.id] = out_tok
            want_default = w.attrs.get("sources")
            want_default = want_default is None or "default" in want_default
            flux_only = "false" if want_default else "true"
            body.append("{")
            # FD step norms use PureFieldAlgebra::dot over the complete prepared vector
            # distribution: the same reduction contract as the Krylov residual norm.
            body.append("  const pops::Real jvn = std::sqrt("
                        "pops::detail::PreparedFieldAlgebra::dot("
                        "%s, %s, ctx.program_resource_vector_distribution(), "
                        "*%s, *execution_lane));" % (in_arg, in_arg, metric_scratch))
            body.append("  const pops::Real jukn = std::sqrt("
                        "pops::detail::PreparedFieldAlgebra::dot("
                        "*%s, *%s, ctx.program_resource_vector_distribution(), "
                        "*%s, *execution_lane));" % (uk, uk, metric_scratch))
            body.append("  const pops::Real jh = jvn > pops::Real(0) ? "
                        "static_cast<pops::Real>(%s) * (pops::Real(1) + jukn) / jvn "
                        ": static_cast<pops::Real>(%s);" % (eps, eps))
            # U^k + h*v -> jac_up; solve fields from that SAME perturbed state before evaluating rhs.
            # This includes elliptic dependence in Jv instead of reusing stale U^n/U^k fields.
            body.append("  pops::PureFieldAlgebra::lincomb(*%s, pops::Real(1), *%s, jh, %s);"
                        % (up, uk, in_arg))
            if w.attrs["field_coupled"]:
                body.append("  ctx.evaluate_with_field_state_at("
                            "*%s, %s, %d, *%s, *%s, [&]() {"
                            % (point, field_slot, block_idx, up, uk))
                body.append("    ctx.rhs_core_into_at(*%s, %d, *%s, *%s, %s, *%s);"
                            % (point, block_idx, up, rp, flux_only, boundary_session))
                body.append("  });")
            else:
                body.append("  ctx.rhs_core_into_at(*%s, %d, *%s, *%s, %s, *%s);"
                            % (point, block_idx, up, rp, flux_only, boundary_session))
            # out = v - (c*dt/h)(Rcore(U^k + h*v) - Rcore(U^k)).  The boundary contribution uses its
            # exact JVP contract below, avoiding an invalid finite difference of ghost/action effects.
            body.append("  const pops::Real jc = *%s / jh;" % cdt)
            body.append("  pops::PureFieldAlgebra::lincomb(%s, pops::Real(1), %s, -jc, *%s);"
                        % (out_tok, in_arg, rp))
            body.append("  if (%s) {" % has_boundary)
            body.append("    pops::PureFieldAlgebra::axpy(%s, jc, *%s);" % (out_tok, r0_core))
            body.append("    pops::PureFieldAlgebra::zero_valid(*%s);" % boundary_work)
            body.append("    ctx.boundary_jvp_into_at(*%s, %d, *%s, %s, *%s, *%s);"
                        % (point, block_idx, uk, in_arg, boundary_work, boundary_session))
            body.append("    pops::PureFieldAlgebra::axpy(%s, -*%s, *%s);"
                        % (out_tok, cdt, boundary_work))
            body.append("  } else {")
            body.append("    pops::PureFieldAlgebra::axpy(%s, jc, *%s);" % (out_tok, r0))
            body.append("  }")
            body.append("}")
        else:
            raise NotImplementedError(
                "emit_cpp_program: op '%s' is not lowerable inside a matrix_free_operator apply "
                "(supported: scalar_field, laplacian, gradient, divergence, apply_laplacian_coeff, "
                "rhs_jacvec)" % w.op)
    body += _emit_field_combine(
        result, "out", sub, acc_sp, dt_symbol="(*%s)" % operator_dt)
    factory = "make_apply_A%d_session" % apply_id
    prelude.append(
        "pops::PreparedAffineOperatorSessionFactory %s = "
        "[%s](const pops::ExecutionLane& lane) {"
        % (factory, ", ".join(captures)))
    session_capture_initializers = []
    template_capture_initializers = []
    session_refresh = []
    for name in session_fields:
        local = "session_%s" % name
        template = "template_%s" % name
        prelude.append("  auto %s = std::make_shared<pops::MultiFab>(*%s);" % (local, name))
        prelude.append("  %s->detach_communication_caches();" % local)
        template_capture_initializers.append("%s = %s" % (template, name))
        session_capture_initializers.append("%s = %s" % (name, local))
        session_refresh.append(
            "pops::PureFieldAlgebra::copy_allocated(*%s, *%s);" % (name, template))
    for name in session_optional_fields:
        local = "session_%s" % name
        template = "template_%s" % name
        prelude.append(
            "  auto %s = %s ? std::make_shared<pops::MultiFab>(*%s) "
            ": std::shared_ptr<pops::MultiFab>{};" % (local, name, name))
        prelude.append("  if (%s) %s->detach_communication_caches();" % (local, local))
        template_capture_initializers.append("%s = %s" % (template, name))
        session_capture_initializers.append("%s = %s" % (name, local))
        session_refresh.append(
            "if (%s && %s) pops::PureFieldAlgebra::copy_allocated(*%s, *%s);"
            % (name, template, name, template))
    for name in session_scalars:
        local = "session_%s" % name
        template = "template_%s" % name
        prelude.append("  auto %s = std::make_shared<pops::Real>(*%s);" % (local, name))
        template_capture_initializers.append("%s = %s" % (template, name))
        session_capture_initializers.append("%s = %s" % (name, local))
        session_refresh.append("*%s = *%s;" % (name, template))
    for name in session_points:
        local = "session_%s" % name
        template = "template_%s" % name
        prelude.append(
            "  auto %s = std::make_shared<"
            "pops::runtime::multiblock::BoundaryEvaluationPoint>(*%s);" % (local, name))
        template_capture_initializers.append("%s = %s" % (template, name))
        session_capture_initializers.append("%s = %s" % (name, local))
        session_refresh.append("*%s = *%s;" % (name, template))
    for name, expression in session_dynamic:
        local = "session_%s" % name
        prelude.append("  auto %s = %s;" % (local, expression))
        session_capture_initializers.append("%s = %s" % (name, local))
    allocation_terms = ["std::size_t{%d}" % len(session_fields)]
    allocation_terms.extend(
        "(%s ? std::size_t{1} : std::size_t{0})" % name
        for name in session_optional_fields
    )
    prelude.append(
        "  const std::size_t session_field_count = %s;"
        % " + ".join(allocation_terms)
    )
    prepare_captures = (
        session_direct + template_capture_initializers + session_capture_initializers)
    prelude.append("  pops::PreparedResourceFn prepare = [%s]() {"
                   % ", ".join(prepare_captures))
    prelude.append("    auto& ctx = *ctx_owner;")
    prelude += ["    " + statement for statement in session_refresh]
    for w in jac_ops:
        (uk, r0, _up, _rp, r0_core, boundary_work, point, has_boundary,
         _field_slot, _cdt, block_idx, _metric_scratch) = jac_scratch[w.id]
        boundary_session = boundary_sessions[block_idx]
        prelude.append("    if (%s) {" % has_boundary)
        prelude.append("      pops::PureFieldAlgebra::copy(*%s, *%s);" % (r0_core, r0))
        prelude.append("      pops::PureFieldAlgebra::zero_valid(*%s);" % boundary_work)
        prelude.append(
            "      ctx.boundary_residual_into_at(*%s, %d, *%s, *%s, *%s);"
            % (point, block_idx, uk, boundary_work, boundary_session))
        prelude.append(
            "      pops::PureFieldAlgebra::axpy(*%s, static_cast<pops::Real>(-1), *%s);"
            % (r0_core, boundary_work))
        prelude.append("    }")
    prelude.append("  };")
    # Apply sees only its private session state.  The outer template snapshots are refresh inputs
    # for prepare() and must not bloat every hot matvec closure.
    apply_captures = session_direct + session_capture_initializers + ["execution_lane = &lane"]
    prelude.append("  pops::ApplyFn apply = [%s](pops::MultiFab& out, const pops::MultiFab& in) {"
                   % ", ".join(apply_captures))
    prelude.append("    auto& ctx = *ctx_owner;")
    prelude += ["    " + ln for ln in body]
    prelude.append("  };")
    prelude.append(
        "  return pops::PreparedAffineOperatorSessionCallbacks{"
        "std::move(prepare), std::move(apply), "
        "[session_field_count]() { return session_field_count; }};")
    prelude.append("};")
    exact_parameters = "%s:%d" % (program._ir_hash(), apply_id)
    exclusive_context = any(
        w.op == "rhs_jacvec" and bool(w.attrs["field_coupled"]) for w in block)
    concurrency = (
        "pops::PreparedOperatorConcurrency::Exclusive"
        if exclusive_context
        else "pops::PreparedOperatorConcurrency::Independent"
    )
    prelude.append(
        "auto %s = pops::PreparedAffineOperatorProvider::trusted_extension("
        "{\"pops.codegen.matrix-free-operator\", 1}, %s, %s, %s);"
        % (lam, json.dumps(exact_parameters), factory, concurrency))


def _prepared_preconditioner(
        v: Any, prelude: Any, prototype: str,
        vector_distribution_expr: str) -> str:
    """Dispatch through the exact authenticated provider carried by the solve IR.

    The dispatcher knows no provider class or provider name.  A compiler plugin registers one
    immutable provider carrying both metadata and emitter; unknown or scheme-mismatched identities
    fail before any C++ is emitted.  The plugin must supply its referenced native headers/C++ too.
    """
    provider = prepared_preconditioner_provider_from_attrs(v.attrs)
    return provider.emit(v, prelude, prototype, vector_distribution_expr)


def _validated_direct_solve_components(v: Any, operator: Any) -> int:
    """Authenticate the operator shape needed by a provider-owned direct solve.

    A direct hierarchy provider owns its native storage and therefore has no Krylov footprint. The
    component count still has two independent authorities (operator declaration and solve node), and
    both must agree before native emission.
    """
    operator_attrs = getattr(operator, "attrs", None)
    if getattr(operator, "op", None) != "matrix_free_operator" or not isinstance(
        operator_attrs, Mapping
    ):
        raise ValueError("direct hierarchy solve requires an authenticated matrix_free_operator")
    operator_components = exact_cpp_int(
        operator_attrs.get("ncomp"),
        where="direct hierarchy operator component count",
        minimum=1,
    )
    solve_components = exact_cpp_int(
        v.attrs.get("ncomp"), where="direct hierarchy solve component count", minimum=1
    )
    if solve_components != operator_components:
        raise ValueError("direct hierarchy solve component count disagrees with its operator")
    return solve_components


def _emit_solve_linear(program: Any, v: Any, base: Any, var: Any, prelude: Any,
                       lines: Any, target: Any = "system") -> None:
    """Lower solve_linear to a call into the runtime's matrix-free Krylov loop. The solution field
    ``sf_sol{id}`` is a PERSISTENT shared_ptr (prelude, captured by the step closure); the step body
    seeds the initial guess (zero, or a copy of the supplied guess), then calls the runtime context's
    typed ``solve_prepared_linear`` seam with its authenticated problem and persistent workspace.
    The SolveReport is checked before the token is published: solved writes may continue,
    while non-converged / singular / breakdown / invalid-evaluation reports fail the run instead of
    letting a partial iterate masquerade as a solved value. The trip count is still decided C++-side,
    inside the loop -- invisible to the IR. The result token is the solution field, dereferenced for the
    final copy back into the block state at commit.

    Uniform and level-scoped AMR solves use the generic context seam. A prepared hierarchy provider
    owns the refined native emission and declares its exact flat Krylov fallback contract."""
    op_value = v.inputs[0]
    rhs_in = v.inputs[1]
    guess_in = v.inputs[2] if v.attrs["has_guess"] else None
    lam = var[op_value.id]  # the apply lambda (already emitted into the prelude)
    direct_hierarchy_phase = bool(var.get(("direct_hierarchy_solve", v.id), False))
    hierarchy_provider = None
    if v.attrs.get("scope") == "hierarchy" or "hierarchy_solver_provider" in v.attrs:
        if target != "amr_system" or v.attrs.get("scope") != "hierarchy":
            raise ValueError("a prepared hierarchy solver requires target='amr_system'")
        hierarchy_provider = prepared_hierarchy_solver_provider_from_attrs(v.attrs)
        hierarchy_provider.validate_node(v, target=target)
    if direct_hierarchy_phase and hierarchy_provider is None:
        raise ValueError("a direct hierarchy phase requires an authenticated provider")
    direct_provider_execution = hierarchy_provider is not None and (
        direct_hierarchy_phase
        or not hierarchy_provider.flat_execution.uses_prepared_krylov_fallback
    )
    uses_prepared_krylov = not direct_provider_execution
    sol_sp = "sf_sol%d" % v.id
    # The solution carries the operator's component count: a vector / state solve writes an ncomp
    # iterate (the Krylov scratch r/p/Ap is co-allocated from it, so the whole loop is ncomp-wide).
    if uses_prepared_krylov:
        footprint = validated_krylov_footprint(v.attrs, operator=op_value)
        problem_contract = validated_prepared_problem_contract(v.attrs, operator=op_value)
        op_ncomp = footprint["components"]
        input_ghosts = footprint["input_ghosts"]
        prelude.append(
            "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, %d));"
            % (sol_sp, op_ncomp, input_ghosts))
    else:
        footprint = None
        problem_contract = None
        op_ncomp = _validated_direct_solve_components(v, op_value)
    # On a refined AMR hierarchy the mathematical solution is one field per level.  The persistent
    # level-0 scratch remains the actual solve argument, while every downstream consumer resolves the
    # published field through the context's current-level seam.  Flat AMR returns the scratch itself.
    if direct_provider_execution:
        var[v.id] = "ctx.hierarchy_solution()"
    else:
        var[v.id] = ("ctx.linear_solution(*%s)" % sol_sp
                     if target == "amr_system" and v.attrs.get("scope") == "hierarchy"
                     else "(*%s)" % sol_sp)
    # Initial guess: zero (default) or a copy of the guess field.
    if uses_prepared_krylov:
        if guess_in is None:
            lines.append("pops::PureFieldAlgebra::zero_valid(*%s);" % sol_sp)
        else:
            lines.append("pops::PureFieldAlgebra::copy(*%s, %s);"
                         % (sol_sp, var[guess_in.id]))
    tol = "static_cast<pops::Real>(%s)" % scalar_cpp(v.attrs["tol"])
    max_iter = int(v.attrs["max_iter"])
    rhs_tok = var[rhs_in.id]
    kr = "kr%d" % v.id
    action_kind, action_statuses = _consumed_solve_action(program, v)

    def _append_report_guard() -> None:
        lines.append("if (!%s.solved_value_available()) {" % kr)
        if action_kind == "reject_attempt":
            selected = " || ".join(
                "%s.status == %s" % (kr, _SOLVE_STATUS_CPP[status])
                for status in action_statuses)
            lines.append("  if (%s) {" % selected)
            lines.append("    throw pops::runtime::program::StepAttemptRejected("
                         "%s.status, \"solve\", std::string(\"solve_linear failed: \") + "
                         "%s.status_name());" % (kr, kr))
            lines.append("  }")
        lines.append("  throw std::runtime_error(std::string(\"solve_linear failed: \") + "
                     "%s.status_name() + \" action=fail_run\");" % kr)
        lines.append("}")

    abs_tol = "static_cast<pops::Real>(%s)" % scalar_cpp(v.attrs["abs_tol"])
    hierarchy_emission = None
    if hierarchy_provider is not None:
        hierarchy_emission = hierarchy_provider.emit(
            PreparedHierarchySolverEmitRequest(
                node=v,
                target=target,
                report_name=kr,
                solution_name=var[v.id] if direct_provider_execution else sol_sp,
                components=op_ncomp,
                block_index=int(v.attrs["hierarchy_block_index"]),
                relative_tolerance_cpp=tol,
                absolute_tolerance_cpp=abs_tol,
                max_iterations=max_iter,
            )
        )
        prelude.extend(hierarchy_emission.configure)

    if direct_provider_execution:
        if hierarchy_emission is None:
            raise ValueError("a direct hierarchy phase has no native provider emission")
        lines.extend(hierarchy_emission.solve)
        _append_report_guard()
        return

    if footprint is None or problem_contract is None:
        raise RuntimeError("prepared Krylov emission has no authenticated contract")

    method_expr = prepared_krylov_method_provider_from_attrs(v.attrs).emit_cpp(v)

    properties = problem_contract["operator_properties"]
    if properties == {
        "symmetric": True,
        "positive_definite": True,
        "positive_definite_on_nullspace_complement": False,
    }:
        properties_expr = "pops::LinearOperatorProperties::symmetric_positive_definite()"
    elif properties == {
        "symmetric": True,
        "positive_definite": False,
        "positive_definite_on_nullspace_complement": True,
    }:
        properties_expr = (
            "pops::LinearOperatorProperties::"
            "symmetric_positive_definite_on_nullspace_complement()"
        )
    elif properties == {
        "symmetric": True,
        "positive_definite": False,
        "positive_definite_on_nullspace_complement": False,
    }:
        properties_expr = "pops::LinearOperatorProperties::symmetric()"
    elif properties == {
        "symmetric": False,
        "positive_definite": False,
        "positive_definite_on_nullspace_complement": False,
    }:
        properties_expr = "pops::LinearOperatorProperties::general()"
    else:
        raise ValueError("solve_linear operator properties are incoherent or unauthenticated")
    footprint_name = "krylov_footprint%d" % v.id
    prelude.append(
        "const pops::KrylovFootprint %s{%d, %d, %s};"
        % (footprint_name, op_ncomp, input_ghosts,
           "true" if footprint["preconditioned"] else "false"))

    authority_material = json.dumps({
        "program": program._ir_hash(),
        "operator": op_value.id,
        "solve": v.id,
        "solver": v.attrs.get("solver_identity"),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    authority_digest = hashlib.sha256(authority_material).digest()
    authority = [
        int.from_bytes(authority_digest[offset:offset + 8], "big")
        for offset in range(0, 32, 8)
    ]
    var.setdefault(("compiled_program_operator_authorities",), []).append(tuple(authority))
    resource_digest = hashlib.sha256(
        b"prepared-resources:" + authority_material).digest()
    resources = [
        int.from_bytes(resource_digest[offset:offset + 8], "big")
        for offset in range(0, 32, 8)
    ]
    authority_cpp = ", ".join("UINT64_C(%d)" % word for word in authority)
    resources_cpp = ", ".join("UINT64_C(%d)" % word for word in resources)
    snapshot_name = "operator_snapshot%d" % v.id
    prelude.append(
        "auto %s = std::make_shared<pops::OperatorEvaluationSnapshot>();" % snapshot_name)
    nullspace_provider, nullspace_contracts = prepared_nullspace_contracts_from_attrs(
        v.attrs
    )
    nullspace_policy_expr = nullspace_provider.emit(
        node=v, prelude=prelude, contracts=nullspace_contracts
    )
    vector_distribution_expr = "ctx.program_resource_vector_distribution()"
    preconditioner_expr = _prepared_preconditioner(
        v, prelude, sol_sp, vector_distribution_expr)
    problem_name = "prepared_problem%d" % v.id
    freeze_expr = var.get(("operator_freeze", op_value.id))
    if not isinstance(freeze_expr, str):
        raise ValueError("matrix-free operator has no prepared resource contract")
    vector_distribution_arg = ", " + vector_distribution_expr
    prelude.append(
        "auto %s = std::make_shared<pops::PreparedAffineLinearProblem>("
        "*%s, %s, %s, %s, %s, %s, "
        "[ctx_owner, %s]() { "
        "return ctx_owner->probe_operator_evaluation({%s}, %s->topology, {%s}, %s->revision); }, "
        "%s, ctx.authenticated_program_apply_token({%s})%s);"
        % (problem_name, sol_sp, lam, preconditioner_expr, properties_expr, footprint_name,
           nullspace_policy_expr,
           snapshot_name, authority_cpp, snapshot_name, resources_cpp, snapshot_name,
           freeze_expr, authority_cpp, vector_distribution_arg))
    workspace_name = "krylov_workspace%d" % v.id
    prelude.append(
        "auto %s = std::make_shared<pops::KrylovWorkspace>("
        "*%s, %s, %s%s);"
        % (workspace_name, sol_sp, method_expr, footprint_name,
           vector_distribution_arg))
    controls_name = "krylov_controls%d" % v.id
    prelude.append(
        "const pops::KrylovControls %s{%s, %s, %s, %d};"
        % (controls_name, method_expr, tol, abs_tol, max_iter))

    prepare_refresh = var.get(("operator_prepare_refresh", op_value.id))
    dt_captures = var.get(("operator_dt_captures", op_value.id))
    if not isinstance(prepare_refresh, tuple) or not isinstance(dt_captures, tuple) \
            or not dt_captures:
        raise ValueError("matrix-free operator has no per-solve evaluation refresh contract")
    lines.extend(prepare_refresh)
    solve_stage = _solve_stage_fraction(v)
    lines.append("ctx.set_stage_time(%d, %d);" %
                 (solve_stage.numerator, solve_stage.denominator))
    for capture in dt_captures:
        lines.append("*%s = static_cast<pops::Real>(dt);" % capture)
    lines.append(
        "*%s = ctx.operator_evaluation_snapshot("
        "{%s}, *%s, {%s});"
        % (snapshot_name, authority_cpp, sol_sp, resources_cpp))
    lines.append("%s->prepare(*%s);" % (problem_name, snapshot_name))
    lines.append("%s->bind(*%s);" % (workspace_name, problem_name))
    lines.append(
        "pops::SolveReport %s = ctx.solve_prepared_linear("
        "*%s, *%s, *%s, %s, %s);"
        % (kr, problem_name, workspace_name, sol_sp, rhs_tok, controls_name))
    _append_report_guard()
