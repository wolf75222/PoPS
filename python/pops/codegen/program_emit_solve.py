"""pops.codegen.program_emit_solve : matrix-free Krylov op emitters.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  These leaf emitters (called from
``program_emit_ops._emit_op`` for the matrix_free_operator / solve_linear ops) build
install-time apply lambdas + the Krylov solve calls; they never recurse back into the op
dispatcher.  They reuse the shared primitives in ``program_emit_kernels``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pops._ir.literals import scalar_cpp
from pops.time.points import StagePoint, TimePoint

from pops.codegen.program_emit_kernels import (
    _apply_in_arg,
    _coeff_cpp,
    _emit_field_combine,
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


def _consumed_solve_action(program: Any, solve: Any) -> str:
    """Return the explicit action kind attached to the unique solve_outcome consumer."""
    matches = []
    for node in _program_nodes(program):
        if node.op != "solve_outcome" or len(node.inputs) != 1 or node.inputs[0] is not solve:
            continue
        action = node.attrs.get("action")
        matches.append(str(getattr(action, "kind", "")))
    if len(matches) != 1 or matches[0] not in ("fail_run", "reject_attempt"):
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
    """Lower a matrix_free_operator to an INSTALL-TIME C++ apply lambda ``apply_A{id}`` (appended to
    @p prelude). The lambda has the pops::ApplyFn signature ``(pops::MultiFab& out, const pops::MultiFab&
    in)``; its body re-emits the apply sub-block:

      - each ``scalar_field`` scratch -> a PERSISTENT shared_ptr field (declared in the prelude
        BEFORE the lambda, captured by value), reused across every Krylov iteration (alloc-once);
      - ``laplacian(o, i)`` -> ``ctx.laplacian(*o, i)`` (i const_cast when it is the lambda's ``in``,
        which is logically read-only -- the fill only writes ghosts, as in test_generic_krylov);
      - ``rhs_jacvec(out, in, iterate, r0, ...)`` (ADC-431) -> a finite-difference Jacobian-vector
        product over the core residual plus the exact prepared-boundary JVP.  The lambda captures one
        shared ``BoundaryEvaluationPoint`` refreshed from r0's exact stage in the step body, so its
        install-time ProgramContext copy can never reconstruct stale time.  Boundary-only scratch is
        allocated once and only when that block has an installed boundary linearization;
      - the apply RESULT (the affine the body returned, e.g. ``in - alpha*Lap(in)``) is written into
        ``out`` via the same accumulate-then-lincomb idiom as a linear_combine commit.

    The lambda captures ``[ctx, <scratch shared_ptrs>]``; the step closure captures it by value. @p
    lines is the mandatory step-body line list: it refreshes the current ``dt`` before every solve,
    and also carries the rhs_jacvec scratch refresh when that optional operation is present.  A
    matrix-free operator cannot be lowered in a control-flow-local scope because its install-time
    ApplyFn would otherwise have no authenticated step-lifetime source for those values."""
    if lines is None:
        raise NotImplementedError(
            "matrix_free_operator is only lowerable at the top level / step body")
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
    # 1) Persistent scratch (the scalar_field ops): one shared_ptr per scratch, declared before the
    #    lambda so it is in scope to capture. Collected first so the capture list is known.
    scratch = [w for w in block if w.op == "scalar_field"]
    captures = ["ctx"]
    for w in scratch:
        sp = "sf%d_%d" % (apply_id, w.id)
        sub[w.id] = sp
        ncomp = int(w.attrs.get("ncomp", 1))  # >1 for a gradient buffer consumed by divergence
        prelude.append(
            "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
            % (sp, ncomp))
        captures.append(sp)
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
    # The ApplyFn is constructed at install time, outside ``ctx.install([=](double dt) {...})``,
    # while affine apply bodies evaluate exact dt-polynomial coefficients and pass the current dt
    # to the conservative axpy/lincomb ledger.  Carry the live step value through one persistent
    # scalar, exactly like the rhs_jacvec coefficient capture below.  Reusing a single value is safe:
    # a ProgramContext invokes one matrix-free ApplyFn synchronously within its owning step.
    apply_dt = "apply_dt%d" % apply_id
    prelude.append(
        "auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));" % apply_dt)
    captures.append(apply_dt)
    lines.append("*%s = static_cast<pops::Real>(dt);" % apply_dt)
    # A coefficiented apply (apply_laplacian_coeff) reads an OUTER condensed_coeffs bundle (assembled in
    # the step body, before the operator): capture its four coefficient shared_ptrs (already
    # allocated in the prelude by emit_condensed_op) so the lambda can dereference them.
    for w in block:
        if w.op == "apply_laplacian_coeff":
            coeffs = w.inputs[2]
            for sp in var[coeffs.id]:
                if sp not in captures:
                    captures.append(sp)
    # An rhs_jacvec apply (ADC-431, implicit-flux BDF) needs the FROZEN Newton iterate U^k and its
    # precomputed rhs(U^k) inside the lambda. They are step-body locals that CHANGE each Newton
    # iteration, so -- like schur_coeffs -- they become PERSISTENT shared_ptr scratch (jac_uk / jac_r0)
    # captured by value (shared pointee), refreshed from the live iterate / r0 in the step body BEFORE
    # the solve. Plus a perturbed-state scratch (jac_up) and a perturbed-rhs scratch (jac_rp) the
    # lambda fills per matvec. All carry the operator's component count (= the block n_cons).  The
    # exact BoundaryEvaluationPoint is a shared pointee because the ApplyFn itself is constructed
    # before begin_step; rebuilding it from the lambda's captured ctx would observe stale time.
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
        point = "jac_point%d_%d" % (apply_id, w.id)
        prelude.append(
            "auto %s = std::make_shared<"
            "pops::runtime::multiblock::BoundaryEvaluationPoint>();" % point)
        captures.append(point)
        has_boundary = "jac_has_boundary%d_%d" % (apply_id, w.id)
        prelude.append(
            "const bool %s = ctx.has_boundary_linearization(%d);"
            % (has_boundary, block_idx))
        captures.append(has_boundary)
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
        field_slot = None
        if w.attrs["field_coupled"]:
            field_slot = "jac_field_slot%d_%d" % (apply_id, w.id)
            resolved_slot = _rhs_jacvec_field_slot(r0_in, field_plans)
            prelude.append(
                "const std::string %s = %s;" % (field_slot, json.dumps(resolved_slot)))
            captures.append(field_slot)
        # The BDF coefficient c*dt depends on the step's dt (the step-closure parameter), which the
        # install-time lambda cannot see; carry it through a captured shared_ptr<Real> the step body
        # sets to its dt value before the solve (the same persistent-scratch idiom as jac_uk).
        cdt = "jac_cdt%d_%d" % (apply_id, w.id)
        prelude.append("auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));" % cdt)
        captures.append(cdt)
        jac_scratch[w.id] = (
            uk, r0, up, rp, r0_core, boundary_work, point, has_boundary,
            field_slot, cdt, block_idx)
        # Step body: first restore the exact StagePoint of r0 and snapshot it into the shared point;
        # then refresh the frozen U^k / rhs(U^k) / dt captures.  Prepared boundary residuals are
        # removed from the frozen base so the finite difference covers only the core residual; their
        # derivative is supplied separately by boundary_jvp_into_at in the ApplyFn.
        stage = _rhs_stage_fraction(r0_in)
        lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
        lines.append("*%s = ctx.boundary_evaluation_point(%d);" % (point, int(r0_in.id)))
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), %s);"
                     % (uk, uk, var[iterate_in.id]))
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), %s);"
                     % (r0, r0, var[r0_in.id]))
        lines.append("if (%s) {" % has_boundary)
        lines.append("  ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, "
                     "static_cast<pops::Real>(1), *%s);" % (r0_core, r0_core, r0))
        lines.append("  %s->set_val(static_cast<pops::Real>(0));" % boundary_work)
        lines.append("  ctx.boundary_residual_into_at(*%s, %d, *%s, *%s);"
                     % (point, block_idx, uk, boundary_work))
        lines.append("  ctx.axpy(*%s, static_cast<pops::Real>(-1), *%s);"
                     % (r0_core, boundary_work))
        lines.append("}")
        lines.append("*%s = %s;" % (cdt, _coeff_cpp(w.attrs["c_dt"])))
    # 2) The lambda body: the laplacian / gradient ops + the result write into `out`.
    body = ["const pops::Real dt = *%s;" % apply_dt]
    for w in block:
        if w.op in ("scalar_field", "apply_in", "apply_out"):
            continue  # scratch shared_ptr / lambda params: already bound in `sub`, nothing to emit
        if w.op == "laplacian":
            o, i = w.inputs
            sub[w.id] = sub[o.id]
            body.append("ctx.laplacian(*%s, %s);" % (sub[o.id], _apply_in_arg(sub, i)))
        elif w.op == "gradient":
            o, p = w.inputs
            sub[w.id] = sub[o.id]
            body.append("ctx.gradient(*%s, %s);" % (sub[o.id], _apply_in_arg(sub, p)))
        elif w.op == "divergence":
            o, fx, fy = w.inputs
            sub[w.id] = sub[o.id]
            body.append("ctx.divergence(*%s, %s, %s);"
                        % (sub[o.id], _apply_in_arg(sub, fx), _apply_in_arg(sub, fy)))
        elif w.op == "apply_laplacian_coeff":
            # out = div(A grad in), A the coefficient tensor of a condensed_coeffs bundle (ADC-637): the
            # SAME two steps the retired brick wrapper did, emitted INLINE through Schur-free seams --
            # ctx.fill_boundary(in) (the transport-BC ghost fill) then the pops::apply_laplacian
            # coefficient floor -- so a generated .so compiles without coupling/schur/** and the operator
            # arithmetic is bit-identical (eps_x/eps_y/a_xy/a_yx are the captured coeff fields).
            o, i, coeffs = w.inputs
            ex, ey, axy, ayx = var[coeffs.id]
            sub[w.id] = sub[o.id]
            body.append("ctx.fill_boundary(%s);" % _apply_in_arg(sub, i))
            body.append("pops::apply_laplacian(%s, ctx.geom(), *%s, nullptr, %s.get(), "
                        "nullptr, %s.get(), %s.get(), %s.get());"
                        % (_apply_in_arg(sub, i), sub[o.id], ex, ey, axy, ayx))
        elif w.op == "rhs_jacvec":
            # out = J(U^k) in = in - (c*dt/h)(rhs(U^k + h*in) - rhs(U^k)), the finite-difference
            # Jacobian-vector product of the implicit-flux BDF residual (ADC-431). h is a relatively
            # scaled FD step (Brown-Saad / WP: h = eps*(1+||U^k||)/||in||, eps the relative step). The
            # captured jac_uk / jac_r0 hold U^k and rhs(U^k) (refreshed in the step body); jac_up /
            # jac_rp are per-matvec scratch; jac_cdt holds c*dt. The op writes directly into `out`.
            o, i = w.inputs[0], w.inputs[1]
            (uk, r0, up, rp, r0_core, boundary_work, point, has_boundary,
             field_slot, cdt, block_idx) = jac_scratch[w.id]
            in_arg = _apply_in_arg(sub, i)        # the Krylov vector v (the lambda's const `in`)
            out_tok = sub[o.id]                   # the apply out buffer (== "out")
            eps = scalar_cpp(w.attrs["eps"])
            sub[w.id] = out_tok
            want_default = w.attrs.get("sources")
            want_default = want_default is None or "default" in want_default
            flux_only = "false" if want_default else "true"
            body.append("{")
            # FD step norms via krylov_dot (all components when ncomp>1, component 0 otherwise --
            # the SAME reduction the Krylov loop uses for its residual norm).
            body.append("  const pops::Real jvn = std::sqrt(pops::detail::krylov_dot(%s, %s));"
                        % (in_arg, in_arg))
            body.append("  const pops::Real jukn = std::sqrt(pops::detail::krylov_dot(*%s, *%s));"
                        % (uk, uk))
            body.append("  const pops::Real jh = jvn > pops::Real(0) ? "
                        "static_cast<pops::Real>(%s) * (pops::Real(1) + jukn) / jvn "
                        ": static_cast<pops::Real>(%s);" % (eps, eps))
            # U^k + h*v -> jac_up; solve fields from that SAME perturbed state before evaluating rhs.
            # This includes elliptic dependence in Jv instead of reusing stale U^n/U^k fields.
            body.append("  ctx.lincomb(*%s, pops::Real(1), *%s, jh, %s);" % (up, uk, in_arg))
            if w.attrs["field_coupled"]:
                body.append("  ctx.solve_fields_from_state_at(*%s, %s, %d, *%s);"
                            % (point, field_slot, block_idx, up))
            body.append("  ctx.rhs_core_into_at(*%s, %d, *%s, *%s, %s);"
                        % (point, block_idx, up, rp, flux_only))
            if w.attrs["field_coupled"]:
                # The perturbed RHS solve mutates the installed provider.  Restore the exact frozen
                # iterate before any boundary JVP (and before returning from ApplyFn), so a Krylov
                # matvec cannot leak U^k+h*v field state into the next callback or outer solve.
                body.append("  ctx.solve_fields_from_state_at(*%s, %s, %d, *%s);"
                            % (point, field_slot, block_idx, uk))
            # out = v - (c*dt/h)(Rcore(U^k + h*v) - Rcore(U^k)).  The boundary contribution uses its
            # exact JVP contract below, avoiding an invalid finite difference of ghost/action effects.
            body.append("  const pops::Real jc = *%s / jh;" % cdt)
            body.append("  ctx.lincomb(%s, pops::Real(1), %s, -jc, *%s);" % (out_tok, in_arg, rp))
            body.append("  if (%s) {" % has_boundary)
            body.append("    ctx.axpy(%s, jc, *%s);" % (out_tok, r0_core))
            body.append("    %s->set_val(static_cast<pops::Real>(0));" % boundary_work)
            body.append("    ctx.boundary_jvp_into_at(*%s, %d, *%s, %s, *%s);"
                        % (point, block_idx, uk, in_arg, boundary_work))
            body.append("    ctx.axpy(%s, -*%s, *%s);" % (out_tok, cdt, boundary_work))
            body.append("  } else {")
            body.append("    ctx.axpy(%s, jc, *%s);" % (out_tok, r0))
            body.append("  }")
            body.append("}")
        else:
            raise NotImplementedError(
                "emit_cpp_program: op '%s' is not lowerable inside a matrix_free_operator apply "
                "(supported: scalar_field, laplacian, gradient, divergence, apply_laplacian_coeff, "
                "rhs_jacvec)" % w.op)
    body += _emit_field_combine(result, "out", sub, acc_sp)
    prelude.append("pops::ApplyFn %s = [%s](pops::MultiFab& out, const pops::MultiFab& in) {"
                   % (lam, ", ".join(captures)))
    prelude += ["  " + ln for ln in body]
    prelude.append("};")


def _precond_applyfn(v: Any, prelude: Any) -> str:
    """Return the C++ expression for the preconditioner ApplyFn of a solve_linear node @p v, emitting any
    real callback into @p prelude (install-time, captured by the step closure -- alloc-once, like the
    operator apply lambda).

      - ``"identity"`` -> ``pops::ApplyFn{}`` (an EMPTY std::function = unpreconditioned; the historical
        path, byte-identical);
      - ``"geometric_mg"`` -> a named ``pops::ApplyFn precond_mg{id}`` lambda capturing a persistent
        ``pops::runtime::program::GeometricMgPreconditioner`` (the V-cycle cache, coeff_elliptic_ops.hpp)
        whose ``apply(ctx, out, in)`` runs ONE V-cycle of the already-wired pops::GeometricMG (the
        field-solve multigrid), no new numerical kernel (ADC-516).

    A scheme other than these two never reaches here: the Python layer
    (pops.time._program.solve.solve_linear) lowers only identity / geometric_mg for gmres / bicgstab and
    rejects every other preconditioner upstream."""
    scheme = v.attrs.get("preconditioner", "identity")
    if scheme == "identity":
        return "pops::ApplyFn{}"
    if scheme == "geometric_mg":
        name = "precond_mg%d" % v.id
        # The GeometricMG V-cycle cache lives on a PERSISTENT GeometricMgPreconditioner (re-homed to the
        # Schur-free coeff_elliptic_ops.hpp, ADC-637; it moved off ProgramContext with the Schur/Lorentz
        # module split). Allocate ONE (alloc-once, like the matrix-free scratch), capture it by shared_ptr
        # into the ApplyFn lambda: the MG is built once on the first apply and reused across every Krylov
        # iteration / step. ctx is captured by value too (it forwards the seam ops the apply reuses).
        pc = "precond_mg_state%d" % v.id
        # ADC-644: a configured GeometricMG preconditioner carries V-cycle-shape knobs
        # (pre/post/bottom sweeps, min_coarse, n_vcycles). When absent (a default GeometricMG()) emit
        # the no-arg ctor -- byte-identical to the pre-644 source. When present, emit the explicit ctor
        # in the fixed positional order (nu1, nu2, nbottom, min_coarse, n_vcycles), each defaulting to
        # the native kMG* value so an omitted knob keeps its historical default.
        opts = v.attrs.get("precond_options")
        if not opts:
            ctor_args = ""
        else:
            nu1 = int(opts.get("pre_sweeps", 2))
            nu2 = int(opts.get("post_sweeps", 2))
            nbottom = int(opts.get("bottom_sweeps", 50))
            min_coarse = int(opts.get("min_coarse", 2))
            n_vcycles = int(opts.get("n_vcycles", 1))
            ctor_args = "%d, %d, %d, %d, %d" % (nu1, nu2, nbottom, min_coarse, n_vcycles)
        prelude.append(
            "auto %s = std::make_shared<pops::runtime::program::GeometricMgPreconditioner>(%s);"
            % (pc, ctor_args))
        prelude.append(
            "pops::ApplyFn %s = [ctx, %s](pops::MultiFab& out, const pops::MultiFab& in) {"
            % (name, pc))
        prelude.append("  %s->apply(ctx, out, in);" % pc)
        prelude.append("};")
        return name
    raise NotImplementedError(
        "emit_cpp_program: preconditioner scheme '%s' is not lowerable (supported: identity, "
        "geometric_mg)" % scheme)


def _composite_tensor_fac_options(
        v: Any) -> tuple[int, Any, Any, int | None, Any, Any, int | None, bool | None]:
    """Authenticate the complete direct-solver identity carried by a hierarchy solve node."""
    identity = v.attrs.get("hierarchy_solver_identity")
    expected_identity = {"schema_version", "solver_id", "capabilities", "options"}
    if not isinstance(identity, Mapping) or set(identity) != expected_identity:
        raise TypeError(
            "CompositeTensorFAC hierarchy solve requires an exact canonical solver identity")
    if identity["schema_version"] != 1:
        raise ValueError("CompositeTensorFAC hierarchy solve uses an unsupported identity schema")
    if identity["solver_id"] != "composite_tensor_fac" \
            or v.attrs.get("hierarchy_solver") != identity["solver_id"]:
        raise ValueError("CompositeTensorFAC hierarchy solve solver identity is unauthenticated")
    capabilities = identity["capabilities"]
    if (not isinstance(capabilities, (list, tuple))
            or tuple(capabilities) != (
                "amr_hierarchy", "flat_bicgstab", "scalar", "tensor_elliptic")):
        raise ValueError("CompositeTensorFAC hierarchy solve capabilities are unauthenticated")
    options = identity["options"]
    expected_options = {
        "max_iter", "rel_tol", "abs_tol", "fine_sweeps", "coarse_rel_tol", "coarse_abs_tol",
        "coarse_cycles", "verbose"}
    if not isinstance(options, Mapping) or set(options) != expected_options:
        raise TypeError(
            "CompositeTensorFAC options must contain exactly max_iter, rel_tol, abs_tol, fine_sweeps, "
            "coarse_rel_tol, coarse_abs_tol, coarse_cycles and verbose")
    max_iter = options["max_iter"]
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("CompositeTensorFAC max_iter must be a positive int")
    fine_sweeps, coarse_cycles, verbose = (
        options["fine_sweeps"], options["coarse_cycles"], options["verbose"])
    if fine_sweeps is not None and (
            isinstance(fine_sweeps, bool) or not isinstance(fine_sweeps, int)
            or fine_sweeps < 1):
        raise ValueError("CompositeTensorFAC fine_sweeps must be a positive int or None")
    if coarse_cycles is not None and (
            isinstance(coarse_cycles, bool) or not isinstance(coarse_cycles, int)
            or coarse_cycles < 1):
        raise ValueError("CompositeTensorFAC coarse_cycles must be a positive int or None")
    if verbose is not None and type(verbose) is not bool:
        raise TypeError("CompositeTensorFAC verbose must be a Python bool or None")
    from pops.model._bind_schema_data import literal_value
    rel_tol = literal_value(options["rel_tol"], where="CompositeTensorFAC rel_tol")
    if isinstance(rel_tol, bool) or not 0 < rel_tol < 1:
        raise ValueError("CompositeTensorFAC rel_tol must be in (0, 1)")
    abs_tol = literal_value(options["abs_tol"], where="CompositeTensorFAC abs_tol")
    if isinstance(abs_tol, bool) or abs_tol < 0:
        raise ValueError("CompositeTensorFAC abs_tol must be >= 0")
    coarse_rel_tol = options["coarse_rel_tol"]
    if coarse_rel_tol is not None:
        coarse_rel_tol = literal_value(
            coarse_rel_tol, where="CompositeTensorFAC coarse_rel_tol")
        if isinstance(coarse_rel_tol, bool) or not 0 < coarse_rel_tol < 1:
            raise ValueError("CompositeTensorFAC coarse_rel_tol must be in (0, 1) or None")
    coarse_abs_tol = options["coarse_abs_tol"]
    if coarse_abs_tol is not None:
        coarse_abs_tol = literal_value(
            coarse_abs_tol, where="CompositeTensorFAC coarse_abs_tol")
        if isinstance(coarse_abs_tol, bool) or coarse_abs_tol < 0:
            raise ValueError("CompositeTensorFAC coarse_abs_tol must be >= 0 or None")
    from pops._ir.literals import scalar_data
    emitted_tol_data = scalar_data(v.attrs["tol"])
    emitted_max_iter = v.attrs.get("max_iter")
    if (emitted_tol_data != options["rel_tol"] or isinstance(emitted_max_iter, bool)
            or not isinstance(emitted_max_iter, int) or emitted_max_iter != max_iter):
        raise ValueError(
            "CompositeTensorFAC emitted convergence controls disagree with solver identity")
    if v.attrs.get("method") != "bicgstab" or v.attrs.get("preconditioner") != "identity" \
            or v.attrs.get("restart") is not None:
        raise ValueError("CompositeTensorFAC flat branch must be exact unpreconditioned BiCGStab")
    emitted_ncomp = v.attrs.get("ncomp")
    if isinstance(emitted_ncomp, bool) or not isinstance(emitted_ncomp, int) \
            or emitted_ncomp != 1:
        raise ValueError("CompositeTensorFAC supports exactly ncomp=1")
    block_index = v.attrs.get("hierarchy_block_index")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("CompositeTensorFAC requires an authenticated hierarchy block index")
    return (max_iter, rel_tol, abs_tol, fine_sweeps, coarse_rel_tol, coarse_abs_tol,
            coarse_cycles, verbose)


def _emit_solve_linear(program: Any, v: Any, base: Any, var: Any, prelude: Any,
                       lines: Any, target: Any = "system") -> None:
    """Lower solve_linear to a call into the runtime's matrix-free Krylov loop. The solution field
    ``sf_sol{id}`` is a PERSISTENT shared_ptr (prelude, captured by the step closure); the step body
    seeds the initial guess (zero, or a copy of the supplied guess), then calls the runtime context's
    generic ``solve_linear_matfree`` seam with the operator's apply lambda.
    The SolveReport is checked before the token is published: solved writes may continue,
    while non-converged / singular / breakdown / invalid-evaluation reports fail the run instead of
    letting a partial iterate masquerade as a solved value. The trip count is still decided C++-side,
    inside the loop -- invisible to the IR. The result token is the solution field, dereferenced for the
    final copy back into the block state at commit.

    Uniform and level-scoped AMR solves use the generic context seam. A direct hierarchy solve emits
    the flat Krylov call in the flat topology body and the authenticated composite-FAC call in the
    refined solve-once phase."""
    op_value = v.inputs[0]
    rhs_in = v.inputs[1]
    guess_in = v.inputs[2] if v.attrs["has_guess"] else None
    lam = var[op_value.id]  # the apply lambda (already emitted into the prelude)
    hierarchy_solver = v.attrs.get("hierarchy_solver")
    direct_refined = bool(var.get(("direct_hierarchy_solve", v.id), False))
    fac_options = None
    if hierarchy_solver is not None:
        if target != "amr_system" or v.attrs.get("scope") != "hierarchy" \
                or hierarchy_solver != "composite_tensor_fac":
            raise ValueError(
                "CompositeTensorFAC lowers only as a direct hierarchy solver for target='amr_system'")
        fac_options = _composite_tensor_fac_options(v)
    sol_sp = "sf_sol%d" % v.id
    # The solution carries the operator's component count: a vector / state solve writes an ncomp
    # iterate (the Krylov scratch r/p/Ap is co-allocated from it, so the whole loop is ncomp-wide).
    op_ncomp = int(v.attrs.get("ncomp", 1))
    prelude.append(
        "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
        % (sol_sp, op_ncomp))
    if fac_options is not None:
        _, _, _, fine_sweeps, coarse_rel_tol, coarse_abs_tol, coarse_cycles, verbose = fac_options
        prelude.append(
            "ctx.configure_composite_tensor_fac(%d, %d, %d, static_cast<pops::Real>(%s), "
            "static_cast<pops::Real>(%s), %d, %s);"
            % (int(v.attrs["hierarchy_block_index"]), op_ncomp,
               0 if fine_sweeps is None else fine_sweeps,
               scalar_cpp(0 if coarse_rel_tol is None else coarse_rel_tol),
               scalar_cpp(0 if coarse_abs_tol is None else coarse_abs_tol),
               0 if coarse_cycles is None else coarse_cycles,
               -1 if verbose is None else int(verbose)))
    # On a refined AMR hierarchy the mathematical solution is one field per level.  The persistent
    # level-0 scratch remains the actual solve argument, while every downstream consumer resolves the
    # published field through the context's current-level seam.  Flat AMR returns the scratch itself.
    var[v.id] = ("ctx.linear_solution(*%s)" % sol_sp
                 if target == "amr_system" and v.attrs.get("scope") == "hierarchy"
                 else "(*%s)" % sol_sp)
    # Initial guess: zero (default) or a copy of the guess field.
    if not direct_refined:
        if guess_in is None:
            lines.append("%s->set_val(static_cast<pops::Real>(0));" % sol_sp)
        else:
            lines.append(
                "ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, "
                "static_cast<pops::Real>(1), %s);"
                % (sol_sp, sol_sp, var[guess_in.id]))
    tol = "static_cast<pops::Real>(%s)" % scalar_cpp(v.attrs["tol"])
    max_iter = int(v.attrs["max_iter"])
    rhs_tok = var[rhs_in.id]
    method = v.attrs["method"]
    kr = "kr%d" % v.id
    action_kind = _consumed_solve_action(program, v)

    def _append_report_guard() -> None:
        lines.append("if (!%s.solved_value_available()) {" % kr)
        if action_kind == "reject_attempt":
            lines.append("  throw pops::runtime::program::StepAttemptRejected("
                         "%s.status, \"solve\", std::string(\"solve_linear failed: \") + "
                         "%s.status_name());" % (kr, kr))
        else:
            lines.append("  throw std::runtime_error(std::string(\"solve_linear failed: \") + "
                         "%s.status_name() + \" action=fail_run\");" % kr)
        lines.append("}")

    # The preconditioner ApplyFn passed to bicgstab / gmres: an EMPTY pops::ApplyFn{} for the identity
    # (unpreconditioned), or a real M^{-1} callback for a non-identity scheme. _precond_applyfn emits the
    # real callback into the prelude (alloc-once, like the operator apply) and returns the C++ expression
    # that names it; identity returns the empty-ApplyFn token. (CG / Richardson have no precond parameter;
    # the Python layer rejects a non-identity precond for them, so they never reach this branch.)
    precond_arg = _precond_applyfn(v, prelude)
    restart = int(v.attrs["restart"]) if method == "gmres" else 0
    # CG / Richardson carry no preconditioner parameter, so pass the empty ApplyFn{}; the context
    # ignores it. Method id: 0 = cg, 1 = bicgstab, 2 = gmres, 3 = richardson.
    method_id = {"cg": 0, "bicgstab": 1, "gmres": 2, "richardson": 3}[method]
    precond_expr = precond_arg if method in ("bicgstab", "gmres") else "pops::ApplyFn{}"
    omega = v.attrs.get("omega")
    omega_tok = ("static_cast<pops::Real>(1)" if omega is None
                 else "static_cast<pops::Real>(%s)" % scalar_cpp(omega))
    if direct_refined:
        if fac_options is None:
            raise ValueError("a direct hierarchy phase requires CompositeTensorFAC")
        fac_abs_tol = "static_cast<pops::Real>(%s)" % scalar_cpp(fac_options[2])
        lines.append(
            "pops::SolveReport %s = ctx.solve_composite_tensor_fac(%d, %d, %s, %s, %d);"
            % (kr, int(v.attrs["hierarchy_block_index"]), op_ncomp, tol, fac_abs_tol, max_iter))
    else:
        lines.append(
            "pops::SolveReport %s = ctx.solve_linear_matfree(*%s, %s, %s, %s, %d, %s, "
            "%d, %d, %s);"
            % (kr, sol_sp, rhs_tok, lam, precond_expr, method_id, tol, max_iter, restart,
               omega_tok))
    _append_report_guard()
