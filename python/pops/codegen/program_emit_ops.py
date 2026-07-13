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

from pops.ir.literals import scalar_cpp
from pops.time.references import block_name
from pops.codegen.program_emit_kernels import (
    _PROFILE_SKIP_OPS,
    _coeff_cpp,
    _deref,
    _emit_cell_compare_kernel,
    _emit_where_kernel,
    _named_fluxes,
)
from pops.codegen.program_emit_model_kernels import (
    _emit_apply_kernel,
    _emit_coupled_rate_kernel,
    _emit_flux_kernel,
    _emit_solve_local_linear_kernel,
    _emit_solve_local_nonlinear_kernel,
    _emit_source_kernel,
)
from pops.codegen.program_emit_condensed import emit_condensed_op
from pops.codegen.program_emit_control import (
    _coupled_rate_components,
    _emit_branch,
    _emit_range,
    _emit_while,
)
from pops.codegen.program_emit_solve import (
    _emit_matrix_free_operator,
    _emit_solve_linear,
)
from pops.codegen.program_emit_schedule import _emit_schedule_wrap
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


def _emit_op(program: Any, v: Any, base: Any, committed_ids: Any, var: Any, model: Any, lines: Any,
             prelude: Any = None, block_idx: Any = None, target: Any = "system",
             field_plans: Any = None) -> None:
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
    # PER-NODE PROFILING (ADC-459): bracket this op's emitted C++ with a steady_clock pair
    # recorded under "node:<v.name>" (shown by sim.profile_report next to the coarse phases). A
    # now() + ctx.profile_record pair (NOT a RAII ProfileScope { }) keeps the emitted declarations
    # at step-body scope -- later nodes read them (e.g. r2 / acc3). Additive, ~free when profiling
    # is off (record early-returns), changes no numerics; ops emitting no statement (pure inline
    # token: cfl / compare) are skipped by the len guard below. _start marks this op's first line.
    _profile_start = len(lines)
    if v.op == "state":
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab& %s = ctx.state(%d);" % (var[v.id], bidx))
    elif v.op == "solve_fields":
        # Per-stage field solve (ADC-409): P.solve_fields(state=...) re-solves phi from THIS
        # stage's explicit state (the shared aux is re-filled before the stage's RHS reads it; the
        # first stage state == U^n == the old ctx.solve_fields()). Multi-block (ADC-426):
        # solve_fields_from_state(idx, U_stage) is a genuinely COUPLED solve -- the Poisson RHS is
        # Sum_s elliptic_rhs_s(U_s), block idx at its stage state, every other block contributing
        # its live state into the shared phi/aux.
        (state_in,) = v.inputs  # solve_fields inputs = (state,)
        field_ref = v.attrs.get("field")
        if field_ref is None:
            raise ValueError("solve_fields node has no exact field identity")
        field, _ = resolved_field_route(field_ref, field_plans)
        lines += field_point_cpp(program, v, field)
        solve_stmt = ('ctx.solve_fields_from_state(%s, %d, %s);'
                      % (json.dumps(field), bidx, var[state_in.id]))
        lines.append(solve_stmt)
    elif v.op == "solve_fields_from_blocks":
        # Coupled multi-block field solve (ADC-457): a SIMULTANEOUS solve, EVERY listed block at
        # its OWN stage state -- the Poisson RHS is Sum_s elliptic_rhs_s(U_s) over all coupled
        # blocks, not a single-target override. Lowers to ctx.solve_fields_from_blocks(u_stages),
        # a vector indexed BY BLOCK INDEX (size == ctx.n_blocks(); nullptr = the block's live
        # state) -- the multi-species seam (IR commit_many: no operator observes a partial group).
        # Inputs slot at their OWN block index (not list position), so a reordered list still
        # solves right; an input whose block was never declared via T.state fails loud at emit.
        if not isinstance(block_idx, Mapping):
            raise ValueError(
                "solve_fields_from_blocks: runtime block routing is unavailable")
        bmap = block_idx
        vec = "u_stages_%d" % v.id
        lines.append("std::vector<const pops::MultiFab*> %s(ctx.n_blocks(), nullptr);" % vec)
        for st in v.inputs:  # inputs = the N state values, slotted by their own block index
            index = _required_block_index(
                bmap, st.block, "solve_fields_from_blocks input node %r" % st.id)
            lines.append("%s[%d] = &%s;" % (vec, index, var[st.id]))
        field_ref = v.attrs.get("field")
        if field_ref is None:
            raise ValueError("solve_fields_from_blocks node has no exact field identity")
        field, _ = resolved_field_route(field_ref, field_plans)
        lines += field_point_cpp(program, v, field)
        lines.append("ctx.solve_fields_from_blocks(%s, %s);" % (json.dumps(field), vec))
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
        # binds each input state's Array4 + cons names and writes all block scratches, and record
        # each block's scratch name so the coupled_rate_out for that block aliases it. All input
        # states are co-located (same ba/dm as the System aux), so a single shared loop is sound
        # (the same co-distribution every aux-reading kernel relies on; see _kernel_open).
        components = _coupled_rate_components(program, v)
        by_block = {s.block: s for s in v.inputs}
        scratch = {}
        for blk in components:                       # bundle / expr block order
            scratch[blk] = "cr%d_%s" % (v.id, block_name(blk))
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (scratch[blk], var[by_block[blk].id]))
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
                lines.append("pops::MultiFab& %s = ctx.history_zero_start(%s, %d, %d, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"]),
                                int(v.attrs["ncomp"]), bidx))
            else:
                lines.append("pops::MultiFab& %s = ctx.history_zero_start(%s, %d, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]), int(v.attrs["lag"]),
                                int(v.attrs["ncomp"])))
        else:
            if target == "amr_system":
                lines.append("pops::MultiFab& %s = ctx.history(%s, %d, %d);"
                             % (var[v.id], json.dumps(v.attrs["history"]),
                                int(v.attrs["lag"]), bidx))
            else:
                lines.append("pops::MultiFab& %s = ctx.history(%s, %d);"
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
        lines.append("ctx.apply_projection(%d, %s);" % (bidx, var[state_in.id]))
        var[v.id] = var[state_in.id]
    elif v.op == "cell_compare":
        # A PER-CELL threshold (spec op 17, ADC-418): mask(i,j,0) = field(i,j,0) <cmp> value ? 1 : 0,
        # a fresh 1-component scalar_field. Lowered to a for_each_cell select kernel (the mask the
        # `where` op selects on); no aux / model needed -- it reads component 0 of the input field.
        (field_in,) = v.inputs
        var[v.id] = "m%d" % v.id
        lines.append("pops::MultiFab %s = ctx.alloc_scalar_field(1, 1);" % var[v.id])
        lines += _emit_cell_compare_kernel(var[field_in.id], var[v.id], v.attrs["cmp"],
                                           v.attrs["value"])
    elif v.op == "where":
        # A PER-CELL conditional select (spec op 17, ADC-418): out(i,j,c) = mask ? a(i,j,c) :
        # b(i,j,c), COMPONENT-WISE. A fresh scratch the same shape as `a` (its vtype / ncomp); the
        # ternary is decided per cell inside the kernel (NOT the scalar lazy ``branch`` op).
        mask_in, a_in, b_in = v.inputs
        var[v.id] = "w%d" % v.id
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (var[v.id], var[a_in.id]))
        lines += _emit_where_kernel(var[mask_in.id], var[a_in.id], var[b_in.id], var[v.id])
    elif v.op == "record_scalar":
        # Store the (already-computed) Scalar into the System diagnostics map under its name. A
        # side-effecting op; its var maps to the recorded scalar (a harmless alias). The scalar input
        # is a 'reduce' result emitted earlier in the body (a const pops::Real local).
        (scalar_in,) = v.inputs
        lines.append("ctx.record_scalar(%s, %s);"
                     % (json.dumps(v.attrs["diagnostic"]), var[scalar_in.id]))
        var[v.id] = var[scalar_in.id]
    elif v.op == "rhs":
        state_in = v.inputs[0]  # rhs inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[v.id], var[state_in.id]))
        named_fluxes = _named_fluxes(v)
        requested = v.attrs.get("sources")
        want_flux = v.attrs.get("flux", True)
        # ADC-425 routing (spec criterion 17): the default/composite source is folded in iff the
        # caller did NOT exclude it -- i.e. sources is None (the legacy default) OR "default" is in
        # the explicit list. An EMPTY list [] (or a list of only named sources) excludes it -> flux
        # only. None and [] are recorded distinctly in the IR, so this is unambiguous.
        want_default_source = requested is None or "default" in requested
        if target == "amr_system":
            if hasattr(v.point, "offset"):
                stage_point = v.point
            else:
                try:
                    stage_point = v.point.time
                except ValueError:
                    # A conservative flux belongs to the explicit partition of an ARK stage.  Its
                    # implicit coordinate may differ and must never be silently substituted here.
                    stage_point = v.point.time_for("explicit")
            stage = Fraction(stage_point.offset.to_python())
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
            # the flux expressions per direction into two n_cons scratch fields (fx / fy) by a
            # per-cell kernel, then take the negated centered FV divergence into R. Linear in the
            # named pieces -> splitting the physical flux into named pieces that sum to it gives the
            # SAME -div (to round-off). Distinct stencil from rhs_into (centered FV vs Riemann), so
            # this path is NEVER mixed with the default (guarded by _named_fluxes).
            fx = "%s_fx" % var[v.id]
            fy = "%s_fy" % var[v.id]
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fx, var[state_in.id]))
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);" % (fy, var[state_in.id]))
            lines += _emit_flux_kernel(node_model, named_fluxes, var[state_in.id], fx, fy, bidx)
            lines.append("ctx.neg_div_flux_into(%s, %s, %s);" % (var[v.id], fx, fy))
        named = [s for s in (v.attrs.get("sources") or []) if s != "default"]
        for s in named:
            # R += S_s(U, aux): assemble the named source into a scratch (same per-cell kernel as
            # the standalone 'source' op) and axpy it onto R.
            ssrc = "%s_%s" % (var[v.id], s)
            lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                         % (ssrc, var[state_in.id]))
            lines += _emit_source_kernel(node_model, s, var[state_in.id], ssrc, bidx)
            lines.append("ctx.axpy(%s, static_cast<pops::Real>(1), %s);" % (var[v.id], ssrc))
    elif v.op == "source":
        state_in = v.inputs[0]  # source inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[v.id], var[state_in.id]))
        lines += _emit_source_kernel(
            node_model, v.attrs["source"], var[state_in.id], var[v.id], bidx)
    elif v.op == "apply":
        state_in = v.inputs[0]  # apply inputs = (state[, fields]); the state is first
        var[v.id] = "r%d" % v.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[v.id], var[state_in.id]))
        lines += _emit_apply_kernel(node_model, v.attrs["linear_source"], var[state_in.id], var[v.id],
                                    bidx)
    elif v.op == "solve_local_linear":
        rhs_in = v.inputs[0]  # solve inputs = (rhs_state, op_value[, fields]); rhs first
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);"
                     % (var[v.id], var[base.id]))
        lines += _emit_solve_local_linear_kernel(
            node_model, v.attrs["linear_source"], v.attrs["a_coeff"], var[rhs_in.id], var[v.id], bidx)
    elif v.op == "solve_local_nonlinear":
        # Per-cell Newton (spec op 10): solve residual(U) = 0 from the initial guess U0, cell by
        # cell, with an in-kernel FD Jacobian + the SAME stack dense inverse solve_local_linear
        # uses. The output is a fresh scratch state; the guess input seeds the iterate.
        guess_in = v.inputs[0]  # solve inputs = (initial_guess,)
        var[v.id] = "u%d" % v.id
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);"
                     % (var[v.id], var[base.id]))
        lines += _emit_solve_local_nonlinear_kernel(
            node_model, v, var[guess_in.id], var[v.id], bidx)
    elif v.op == "scalar_field":
        # A step-body scratch scalar field (e.g. the explicit-flux buffer the RHS assembly fills):
        # a persistent shared_ptr (prelude, alloc-once) reused every step. Inside an apply sub-block
        # the scalar_field is handled by _emit_matrix_free_operator instead (this branch is the
        # top-level / step-body path -- prelude is not None there).
        if prelude is None:
            raise NotImplementedError(
                "scalar_field is only lowerable at the top level / step body or inside a "
                "matrix_free_operator apply sub-block, not inside a control-flow (if/while/range) body")
        sp = "sf%d" % v.id
        var[v.id] = "(*%s)" % sp
        ncomp = int(v.attrs.get("ncomp", 1))
        prelude.append("auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(%d, 1));"
                       % (sp, ncomp))
    elif v.op == "laplacian":
        # Step-body bare Laplacian (e.g. Lap phi^n for the condensed RHS). Inside an apply sub-block
        # this op is handled by _emit_matrix_free_operator; here it is the top-level path.
        o, i = v.inputs
        lines.append("ctx.laplacian(%s, %s);" % (_deref(var[o.id]), _deref(var[i.id])))
        var[v.id] = var[o.id]
    elif v.op == "gradient":
        o, p = v.inputs
        lines.append("ctx.gradient(%s, %s);" % (_deref(var[o.id]), _deref(var[p.id])))
        var[v.id] = var[o.id]
    elif v.op == "divergence":
        o, fx, fy = v.inputs
        lines.append("ctx.divergence(%s, %s, %s);"
                     % (_deref(var[o.id]), _deref(var[fx.id]), _deref(var[fy.id])))
        var[v.id] = var[o.id]
    elif v.op in ("condensed_coeffs", "condensed_rhs", "condensed_reconstruct", "condensed_energy"):
        # GENERIC condensed-implicit solve (ADC-637): the tensor coefficient A = I + c*rho*M^{-1} bundle,
        # the fused RHS -Lap(phi^n) - g*div(M^{-1}(m)), the velocity reconstruction and the kinetic-energy
        # increment, emitted INLINE via pops::detail::block_inverse<2> from an authored J (M = I -
        # th_dt*J) on a momentum subset -- no coupling/schur call. The thin dispatch lives in
        # program_emit_condensed to keep this router (and its budget) small; condensed_coeffs allocates
        # its four persistent coeff fields there.
        emit_condensed_op(v, var, node_model, lines, prelude)
    elif v.op == "matrix_free_operator":
        # Install-time: emit the apply lambda `apply_A{id}` into the prelude. Its persistent scratch
        # (the scalar_field ops of the apply sub-block) are shared_ptr fields, captured by value so
        # they outlive the install call and are reused across every Krylov iteration (alloc-once).
        # The lambda is itself captured by the step closure ([=]) and passed to pops::*_solve. An
        # rhs_jacvec apply (ADC-431) also captures persistent jac_uk / jac_r0 scratch the lambda
        # dereferences; the step body refreshes them from the live iterate / rhs(U^k) here (@p lines).
        _emit_matrix_free_operator(program, v, var, prelude, lines)
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
        # effects. Runtime lowering keeps the existing Krylov call as the value-producing operation;
        # these nodes are zero-cost aliases that preserve that explicit contract in the IR.
        (source,) = v.inputs
        var[v.id] = var[source.id]
    elif v.op == "reduce":
        # A collective all_reduce -> a C++ scalar. norm2 = sqrt(dot(u, u)); dot(a, b) directly;
        # sum/max/min (over a component) via the matching pops reduction. All MUST run on every rank
        # (the reductions are collective all_reduce); they sit at the top of the loop body.
        var[v.id] = "s%d" % v.id
        kind = v.attrs["kind"]
        if kind == "norm2":
            (u,) = v.inputs
            lines.append("const pops::Real %s = std::sqrt(pops::dot(%s, %s));"
                         % (var[v.id], var[u.id], var[u.id]))
        elif kind == "norm_inf":
            (u,) = v.inputs
            lines.append("const pops::Real %s = pops::norm_inf(%s);" % (var[v.id], var[u.id]))
        elif kind in ("sum", "max", "min", "abs_sum"):
            # abs_sum -> pops::reduce_abs_sum (the L1 reduction; P.norm1 / Norm(L1)); the reduce_<kind>
            # naming matches the free-function name exactly, like sum/max/min.
            (u,) = v.inputs
            comp = int(v.attrs.get("comp", 0))
            lines.append("const pops::Real %s = pops::reduce_%s(%s, %d);"
                         % (var[v.id], kind, var[u.id], comp))
        else:  # dot
            a, b = v.inputs
            lines.append("const pops::Real %s = pops::dot(%s, %s);"
                         % (var[v.id], var[a.id], var[b.id]))
    elif v.op == "cfl":
        # The dt_bound's runtime cfl argument -- the C++ parameter of pops_program_dt_bound. It is
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
        lines.append("const pops::Real %s = (%s %s %s);"
                     % (var[v.id], toks[0], cppop, toks[1]))
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
    elif v.op == "while":
        _emit_while(program, v, base, var, model, lines, block_idx, field_plans)
    elif v.op == "range":
        _emit_range(program, v, base, var, model, lines, block_idx, field_plans)
    elif v.op == "branch":
        _emit_branch(program, v, base, var, model, lines, block_idx, field_plans)
    elif v.op == "linear_combine":
        terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
        if v.id in committed_ids:
            # Commit: block state <- c_base * base + sum(non-base coeff * term), in place.
            c_base = {0: 0}
            acc = "acc%d" % v.id
            lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (acc, var[base.id]))
            for inp, coeff in terms:
                if inp.id == base.id:
                    c_base = coeff
                else:
                    lines.append("ctx.axpy(%s, %s, %s);" % (acc, _coeff_cpp(coeff), var[inp.id]))
            lines.append("ctx.lincomb(%s, %s, %s, static_cast<pops::Real>(1), %s);"
                         % (var[base.id], _coeff_cpp(c_base), var[base.id], acc))
            var[v.id] = var[base.id]  # the commit wrote the block state in place (no final copy)
        else:
            var[v.id] = "u%d" % v.id  # an intermediate stage state (scratch, zero-initialized)
            # A scalar_field combine (ADC-427: the phi^{n+1} extrapolation) has no block, so it has no
            # base block-state to shape the scratch: template it on the FIRST scalar input instead (a
            # 1-component field, same (ba, dm)). A State combine shapes it on the block base as before.
            template = var[terms[0][0].id] if v.vtype == "scalar_field" else var[base.id]
            lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (var[v.id], template))
            for inp, coeff in terms:
                lines.append("ctx.axpy(%s, %s, %s);" % (var[v.id], _coeff_cpp(coeff), var[inp.id]))
    # UNIFIED SCHEDULER (ADC-458, Spec 3 sections 17-18): if this op carries a non-always schedule,
    # wrap the statements it just emitted (lines[_profile_start:]) in the due-test guard + policy
    # branch. Done HERE, after the op lowered itself, so EVERY schedulable node (field solve, rhs,
    # source, linear_combine, where, ...) reuses the one general mechanism -- no per-op special
    # case. The wrap nests INSIDE the per-node profiling pair below (the profiler times the guarded
    # block as the node's cost). An always() schedule (or no schedule) leaves the lines untouched.
    _emit_schedule_wrap(program, v, var, lines, _profile_start)
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
