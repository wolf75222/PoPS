"""pops.codegen.program_emit_solve : matrix-free Krylov op emitters.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  These leaf emitters (called from
``program_emit_ops._emit_op`` for the matrix_free_operator / solve_linear ops) build
install-time apply lambdas + the Krylov solve calls; they never recurse back into the op
dispatcher.  They reuse the shared primitives in ``program_emit_kernels``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.ir.literals import scalar_cpp

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
        named = [source for source in (v.attrs.get("sources") or ()) if source != "default"]
        if not isinstance(v.attrs.get("field_coupled"), bool):
            raise ValueError("rhs_jacvec IR requires an explicit boolean field_coupled attribute")
        if v.attrs.get("flux") is not True or named:
            raise NotImplementedError(
                "rhs_jacvec lowers only the default flux with sources=[] or ['default']; "
                "got flux=%r, named_sources=%r" % (v.attrs.get("flux"), named))
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


def _emit_matrix_free_operator(program: Any, v: Any, var: Any, prelude: Any,
                               lines: Any = None) -> None:
    """Lower a matrix_free_operator to an INSTALL-TIME C++ apply lambda ``apply_A{id}`` (appended to
    @p prelude). The lambda has the pops::ApplyFn signature ``(pops::MultiFab& out, const pops::MultiFab&
    in)``; its body re-emits the apply sub-block:

      - each ``scalar_field`` scratch -> a PERSISTENT shared_ptr field (declared in the prelude
        BEFORE the lambda, captured by value), reused across every Krylov iteration (alloc-once);
      - ``laplacian(o, i)`` -> ``ctx.laplacian(*o, i)`` (i const_cast when it is the lambda's ``in``,
        which is logically read-only -- the fill only writes ghosts, as in test_generic_krylov);
      - ``rhs_jacvec(out, in, iterate, r0, ...)`` (ADC-431) -> a finite-difference Jacobian-vector
        product ``out = in - (c*dt/eps)(rhs(U^k + eps*in) - rhs(U^k))`` calling ``ctx.rhs_into`` (or
        ``neg_div_flux_default_into``) on PERSISTENT jac_uk / jac_r0 scratch the lambda captures; the
        step body refreshes that scratch from the live iterate / rhs(U^k) (@p lines, see below);
      - the apply RESULT (the affine the body returned, e.g. ``in - alpha*Lap(in)``) is written into
        ``out`` via the same accumulate-then-lincomb idiom as a linear_combine commit.

    The lambda captures ``[ctx, <scratch shared_ptrs>]``; the step closure captures it by value. @p
    lines is the step-body line list (for the rhs_jacvec scratch refresh); None when the operator has
    no jacvec op (the historical matrix-free path, prelude only)."""
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
    # lambda fills per matvec. All carry the operator's component count (= the block n_cons).
    jac_ops = [w for w in block if w.op == "rhs_jacvec"]
    if jac_ops and lines is None:
        raise NotImplementedError(
            "rhs_jacvec is only lowerable in a top-level / step-body matrix-free solve, not inside a "
            "control-flow (if/while/range) body (the Newton outer loop must be a static_range unroll)")
    jac_scratch = {}  # jacvec op id -> (uk, r0, up, rp, cdt, block_idx) names/provenance
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
        # The BDF coefficient c*dt depends on the step's dt (the step-closure parameter), which the
        # install-time lambda cannot see; carry it through a captured shared_ptr<Real> the step body
        # sets to its dt value before the solve (the same persistent-scratch idiom as jac_uk).
        cdt = "jac_cdt%d_%d" % (apply_id, w.id)
        prelude.append("auto %s = std::make_shared<pops::Real>(static_cast<pops::Real>(0));" % cdt)
        captures.append(cdt)
        jac_scratch[w.id] = (uk, r0, up, rp, cdt, block_idx)
        # Step body: refresh the FROZEN captures from this iteration's live iterate / rhs(U^k) / dt.
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), %s);"
                     % (uk, uk, var[iterate_in.id]))
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), %s);"
                     % (r0, r0, var[r0_in.id]))
        lines.append("*%s = %s;" % (cdt, _coeff_cpp(w.attrs["c_dt"])))
    # 2) The lambda body: the laplacian / gradient ops + the result write into `out`.
    body = []
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
            uk, r0, up, rp, cdt, block_idx = jac_scratch[w.id]
            in_arg = _apply_in_arg(sub, i)        # the Krylov vector v (the lambda's const `in`)
            out_tok = sub[o.id]                   # the apply out buffer (== "out")
            eps = scalar_cpp(w.attrs["eps"])
            sub[w.id] = out_tok
            want_default = w.attrs.get("sources")
            want_default = want_default is None or "default" in want_default
            rhs_call = ("ctx.rhs_into(%d, *%%s, *%%s);" % block_idx
                        if (w.attrs["flux"] and want_default)
                        else "ctx.neg_div_flux_default_into(%d, *%%s, *%%s);" % block_idx) % (up, rp)
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
                body.append("  ctx.solve_fields_from_state(%d, *%s);" % (block_idx, up))
            body.append("  %s" % rhs_call)
            # out = v - (c*dt/h)(rhs(U^k + h*v) - rhs(U^k)): lincomb then axpy back the frozen rhs(U^k).
            body.append("  const pops::Real jc = *%s / jh;" % cdt)
            body.append("  ctx.lincomb(%s, pops::Real(1), %s, -jc, *%s);" % (out_tok, in_arg, rp))
            body.append("  ctx.axpy(%s, jc, *%s);" % (out_tok, r0))
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
    (pops.time.program_solve.solve_linear) lowers only identity / geometric_mg for gmres / bicgstab and
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
        v: Any) -> tuple[int | None, Any, int | None, bool | None]:
    """Authenticate the complete provider identity carried by a hierarchy solve node."""
    identity = v.attrs.get("hierarchy_provider_identity")
    expected_identity = {"schema_version", "provider_id", "capabilities", "options"}
    if not isinstance(identity, Mapping) or set(identity) != expected_identity:
        raise TypeError(
            "CompositeTensorFAC hierarchy solve requires an exact canonical provider identity")
    if identity["schema_version"] != 1:
        raise ValueError("CompositeTensorFAC hierarchy solve uses an unsupported identity schema")
    if identity["provider_id"] != "composite_tensor_fac" \
            or v.attrs.get("hierarchy_provider") != identity["provider_id"]:
        raise ValueError("CompositeTensorFAC hierarchy solve provider identity is unauthenticated")
    capabilities = identity["capabilities"]
    if (not isinstance(capabilities, (list, tuple))
            or tuple(capabilities) != ("amr_hierarchy", "tensor_elliptic")):
        raise ValueError("CompositeTensorFAC hierarchy solve capabilities are unauthenticated")
    options = identity["options"]
    expected_options = {"fine_sweeps", "coarse_rel_tol", "coarse_cycles", "verbose"}
    if not isinstance(options, Mapping) or set(options) != expected_options:
        raise TypeError(
            "CompositeTensorFAC options must contain exactly fine_sweeps, coarse_rel_tol, "
            "coarse_cycles and verbose")
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
    coarse_rel_tol = options["coarse_rel_tol"]
    if coarse_rel_tol is not None:
        coarse_rel_tol = literal_value(
            coarse_rel_tol, where="CompositeTensorFAC coarse_rel_tol")
        if isinstance(coarse_rel_tol, bool) or not 0 < coarse_rel_tol < 1:
            raise ValueError("CompositeTensorFAC coarse_rel_tol must be in (0, 1) or None")
    return fine_sweeps, coarse_rel_tol, coarse_cycles, verbose


def _emit_solve_linear(program: Any, v: Any, base: Any, var: Any, prelude: Any,
                       lines: Any, target: Any = "system") -> None:
    """Lower solve_linear to a call into the runtime's matrix-free Krylov loop. The solution field
    ``sf_sol{id}`` is a PERSISTENT shared_ptr (prelude, captured by the step closure); the step body
    seeds the initial guess (zero, or a copy of the supplied guess), then calls the runtime context's
    generic ``solve_linear_matfree`` seam with the operator's apply lambda.
    The SolveReport/KrylovResult is checked before the token is published: solved writes may continue,
    while non-converged / singular / breakdown / invalid-evaluation reports fail the run instead of
    letting a partial iterate masquerade as a solved value. The trip count is still decided C++-side,
    inside the loop -- invisible to the IR. The result token is the solution field, dereferenced for the
    final copy back into the block state at commit.

    Both uniform and AMR targets use the same context seam. ProgramContext selects the geometry-aware
    uniform provider; AmrProgramContext selects the flat or composite hierarchy provider. The emitted
    Program therefore never branches on a concrete solver/runtime class."""
    op_value = v.inputs[0]
    rhs_in = v.inputs[1]
    guess_in = v.inputs[2] if v.attrs["has_guess"] else None
    lam = var[op_value.id]  # the apply lambda (already emitted into the prelude)
    sol_sp = "sf_sol%d" % v.id
    # The solution carries the operator's component count: a vector / state solve writes an ncomp
    # iterate (the Krylov scratch r/p/Ap is co-allocated from it, so the whole loop is ncomp-wide).
    op_ncomp = int(v.attrs.get("ncomp", 1))
    prelude.append(
        "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
        % (sol_sp, op_ncomp))
    # On a refined AMR hierarchy the mathematical solution is one field per level.  The persistent
    # level-0 scratch remains the actual solve argument, while every downstream consumer resolves the
    # published field through the context's current-level seam.  Flat AMR returns the scratch itself.
    var[v.id] = ("ctx.linear_solution(*%s)" % sol_sp
                 if target == "amr_system" and v.attrs.get("scope") == "hierarchy"
                 else "(*%s)" % sol_sp)
    # Initial guess: zero (default) or a copy of the guess field.
    if guess_in is None:
        lines.append("%s->set_val(static_cast<pops::Real>(0));" % sol_sp)
    else:
        lines.append("ctx.lincomb(*%s, static_cast<pops::Real>(0), *%s, static_cast<pops::Real>(1), "
                     "%s);" % (sol_sp, sol_sp, var[guess_in.id]))
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
    if (target == "amr_system" and v.attrs.get("scope") == "hierarchy"
            and v.attrs.get("hierarchy_provider") == "composite_tensor_fac"):
        fine_sweeps, coarse_rel_tol, coarse_cycles, verbose = (
            _composite_tensor_fac_options(v))
        lines.append(
            "ctx.configure_composite_tensor_fac(%d, static_cast<pops::Real>(%s), %d, %s);"
            % (0 if fine_sweeps is None else fine_sweeps,
               scalar_cpp(0 if coarse_rel_tol is None else coarse_rel_tol),
               0 if coarse_cycles is None else coarse_cycles,
               -1 if verbose is None else int(verbose)))
    lines.append("pops::SolveReport %s = ctx.solve_linear_matfree(*%s, %s, %s, %s, %d, %s, "
                 "%d, %d, %s);"
                 % (kr, sol_sp, rhs_tok, lam, precond_expr, method_id, tol, max_iter, restart,
                    omega_tok))
    _append_report_guard()
