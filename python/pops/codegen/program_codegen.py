"""pops.codegen.program_codegen -- C++ emission for a pops.time.Program.

FREE FUNCTIONS taking the ``program`` (and an optional physical ``model``), mirroring
``pops.codegen.module_codegen`` for models. ``emit_cpp_program(program, model=None)`` lowers
the Program SSA IR to the C++ source of a ``problem.so`` (the stable .so ABI installed by
``System::install_program``); the ``_emit_*`` / ``_check_*`` helpers and the per-cell kernel
emitters are the lowering machinery. ``pops.time.Program.emit_cpp_program`` is a thin
delegator into this module (lazy import), keeping ``pops.time`` free of any codegen edge.

This is the THIN public module of the program emitter.  The lowering machinery is split
across sibling modules so each file fits the Spec-4 size budget, and every name re-imported
below so the public surface of ``pops.codegen.program_codegen`` is unchanged:

  - ``program_emit_kernels``       -- op tables, text helpers, model-free per-cell kernels,
                                      the ``_PROGRAM_CPP_TEMPLATE``;
  - ``program_emit_model_kernels`` -- the model-coefficient per-cell kernels;
  - ``program_emit_solve``         -- the matrix-free Krylov + condensed-Schur emitters;
  - ``program_emit_schedule``      -- the unified scheduler wrap (ADC-458);
  - ``program_emit_control``       -- the body walk + control-flow (while/range/if) emitters;
  - ``program_emit_ops``           -- the per-op ``_emit_op`` dispatcher;
  - ``program_emit_amr``           -- the AMR install-entry emitter (target='amr_system', ADC-508).
  - ``program_metadata``           -- complete owner-qualified GeneratedModule metadata.

The orchestration (``emit_cpp_program`` + the block-name / dt-bound emitters + the lowerability
checks) stays here.

The emission-only tables (_MODEL_OPS / _ALLOWED_OPS / _PROFILE_SKIP_OPS / _AUX_OUTPUT_OPS)
are module-level constants in ``program_emit_kernels``, re-exported here.
"""
from __future__ import annotations

from typing import Any
from pops.time.references import block_name

import json  # noqa: F401  (kept for any external reference to program_codegen.json)

# Re-export every moved name so the public surface of this module is unchanged.
from pops.codegen.program_emit_kernels import (  # noqa: F401
    _ALLOWED_OPS,
    _AUX_OUTPUT_OPS,
    _MODEL_OPS,
    _PROFILE_SKIP_OPS,
    _PROGRAM_CPP_TEMPLATE,
    _block_inverse_include,
    _coeff_elliptic_include,
    ProgramValue,
    _apply_in_arg,
    _aux_comp,
    _cell_locals,
    _coeff_cpp,
    _has_runtime_param,
    _deref,
    _emit_cell_compare_kernel,
    _emit_field_combine,
    _emit_where_kernel,
    _kernel_close,
    _kernel_open,
    _model_impl,
    _named_fluxes,
    _to_affine,
)
from pops.codegen.program_emit_model_kernels import (  # noqa: F401
    _emit_apply_kernel,
    _emit_coupled_rate_kernel,
    _emit_flux_kernel,
    _emit_residual_eval,
    _emit_solve_local_linear_kernel,
    _emit_solve_local_nonlinear_kernel,
    _emit_source_kernel,
    _linear_source_rows,
    _residual_term_exprs,
)
from pops.codegen.program_emit_solve import (  # noqa: F401
    _emit_matrix_free_operator,
    _emit_solve_linear,
    _validate_matrix_free_contract,
)
from pops.codegen.program_emit_schedule import (  # noqa: F401
    _emit_schedule_wrap,
    _schedule_due_test,
    _split_output_decl,
)
from pops.codegen.program_emit_control import (  # noqa: F401
    _coupled_rate_components,
    _emit_body,
    _emit_if,
    _emit_range,
    _emit_while,
    _walk_expr,
)
from pops.codegen.program_emit_ops import _emit_op  # noqa: F401
from pops.codegen.program_emit_params import (  # noqa: F401
    emit_program_params as _emit_program_params,
    program_param_entries as _program_param_entries,
)
from pops.codegen.program_emit_amr import _emit_amr_install  # noqa: F401
from pops.codegen.program_metadata import emit_module_metadata as _emit_module_metadata
from pops.codegen.compile_emit import _emit_route_manifest  # noqa: F401 (ADC-599 embedded manifest)
from pops.codegen.program_lowerability import (
    all_ops as _all_ops,
    check_model_owner_dispatch as _check_model_owner_dispatch,
    check_schedules_lowerable as _check_schedules_lowerable,
)
from pops.time.program_space_resolution import resolve_program_spaces
from pops.codegen.program_models import ProgramModelGraph, model_for_node


# --- Program -> C++ lowering (free functions taking `program`) ------------------------------
# --- C++ codegen (Phase 2c-ii / Phase 4b): lower the IR to a problem.so source ---
def emit_cpp_program(
    program: Any,
    model: Any = None,
    target: str = "system",
    *,
    model_graph: Any = None,
) -> str:
    """Generate the C++ source of a problem.so implementing this Program (codegen).

    Exports the stable .so ABI -- ``pops_program_abi_key`` (the ``POPS_ABI_KEY_LITERAL``
    preprocessor literal, NOT the interposable inline), ``pops_program_name``, ``pops_program_hash``,
    ``pops_install_program`` -- and installs the macro step as a closure built from `ProgramContext`
    primitives only (no MultiFab / flux / solver reimplementation). It is the source the C++ loader
    (`System::install_program`) compiles, dlopens, and runs.

    @p target selects the install entry the .so exports. ``"system"`` (default) emits only
    ``pops_install_program`` (the single-level ``System`` macro-step closure). ``"amr_system"``
    (epic ADC-511 / ADC-508, Spec 6) ALSO emits ``pops_install_program_amr``, the entry
    ``AmrSystem::install_program`` resolves: it wraps the ``AmrSystem`` in an ``AmrProgramContext``
    and installs the per-level Lie/Strang macro-step. The driver (``AmrProgramContext``) lands with
    the Kokkos-gated AMR runtime seam; until then the AMR install entry FAILS LOUD at install (it is
    ABI-complete so the whole loader path is wired, but it never produces a silently-wrong run).

    Lowers the Program by a topological walk of the SSA IR: each block's current state is its base
    (``ctx.state(idx)``); ``solve_fields()`` runs the elliptic solve; each RHS becomes a
    scratch + ``rhs_into``; each intermediate ``linear_combine`` becomes a zero scratch accumulated
    with ``axpy``; the committed combine writes the block state via ``lincomb``. Forward Euler,
    SSPRK2/SSPRK3 and RK4 all lower this way -- no per-scheme class.

    Multi-block (ADC-426): N typed ``T.state(block, U)`` declarations + N ``T.commit``
    are lowered -- each op routes to its own block's runtime index (``_block_indices``, in the order
    the blocks are first declared via ``T.state``). The .so also exports its block NAMES in that
    order (``pops_program_block_count`` / ``pops_program_block_name``); ``System::install_program``
    binds them to the instantiated System blocks BY NAME (Spec 3 criterion 23, ADC-457), so the
    System blocks (``sim.add_equation`` / ``sim.add_block``) may be added in ANY order -- a Program
    block whose name has no instantiated System block fails loud (``Program requires block instance
    '<name>', but simulation did not instantiate it``). A block declared but never committed is a
    READ-ONLY block (allowed; e.g. a passive field whose charge couples the others through the shared
    Poisson). A commit of a block no ``T.state`` declares is rejected. A single-block Program lowers
    byte-identically (its one block is index 0; an order-matching multi-block Program too -- the
    name map is the identity).

    Phase-4b also lowers the SPLIT-SOURCE / LOCAL-LINEAR ops -- ``source`` (a named ``m.source_term``
    evaluated per cell), ``apply`` (LU for a named ``m.linear_source``) and ``solve_local_linear``
    ((I -/+ a*L) U = rhs solved cell by cell via a dense per-cell inverse) -- but ONLY when the
    physical ``model`` (the ``pops.dsl`` model whose ``source_term`` / ``linear_source`` they name)
    is provided: the codegen reads the model's symbolic coefficients to emit the per-cell kernels.
    Without ``model`` those ops raise NotImplementedError (the Program cannot be lowered in
    isolation); ``model=None`` still lowers FE / SSPRK / RK4 (no model needed). A ``rhs`` routes its
    base on its ``flux`` flag and whether ``"default"`` is among the requested ``sources`` (ADC-425 /
    ADC-430, spec criterion 17 -- flux and sources are explicit, never summed implicitly). With
    ``flux=True``: ``"default"`` present -> ``ctx.rhs_into`` (= ``-div F`` + the model's
    default/composite source, the historical path); ``"default"`` absent (incl. the empty list
    ``[]``) -> ``ctx.neg_div_flux_default_into`` (= ``-div F`` only, NO default source). With
    ``flux=False`` (SOURCE-ONLY, ADC-430): NO ``-div F`` base -- ``"default"`` present (or ``None``)
    -> ``ctx.source_default_into`` (= S only, the exact mirror); ``"default"`` absent -> the zeroed
    scratch (the named sources, if any, are the whole RHS). Each NAMED source (``sources=[...]``
    beyond ``"default"``) then lowers with a model: the same per-cell ``m.source_term`` kernel as the
    standalone ``source`` op, accumulated onto ``R`` via ``axpy``. So ``flux=True,sources=[]`` is flux
    only, ``flux=True,sources=["default"]`` is flux + default source (unchanged),
    ``flux=False,sources=["default"]`` is the default source only, ``flux=False,sources=["s"]`` is
    just ``s`` -- the named ones never double-count the default (it is folded in iff "default" was
    listed). More than one block now lowers (ADC-426): each op routes to its block's runtime index
    (``_block_indices``, in T.state declaration order) and control flow (while/range/if) inside a
    block lowers per block; a SIMULTANEOUS multi-target coupled field solve
    (``solve_fields_from_blocks([Ua, Ub])``) lowers to ``ctx.solve_fields_from_blocks`` (see below).

    Each ``solve_fields(state=...)`` op lowers to ``ctx.solve_fields_from_state(idx, <stage state>)``
    (ADC-409): the elliptic fields are re-solved -- and the shared aux re-filled -- from THAT stage's
    state, not the block's current state. So a field-coupled multi-stage scheme (Poisson feedback
    into the flux) is exact: stage k's RHS reads phi solved from stage k's own state. For the first
    stage the stage state is U^n, so this is identical to the historical ``solve_fields()``; for an
    uncoupled model the field solve is inert either way. This is already a COUPLED multi-block solve:
    the system Poisson RHS is ``Sum_s elliptic_rhs_s(U_s)`` (``assemble_poisson_rhs``), so block
    ``idx`` reads its stage state while every OTHER block contributes its LIVE state into the one
    shared phi/aux. A per-block ``P.solve_fields(state=Ub)`` therefore sees all blocks' charge. A
    SIMULTANEOUS multi-target override (several blocks at their stage states in ONE solve) lowers to
    ``ctx.solve_fields_from_blocks(<vec>)`` (Spec 3 criterion 24, ADC-457): the RHS is
    ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state at once
    (``assemble_poisson_rhs_from_blocks``), each slotted at its block index (nullptr = the block's
    live state) -- the coupled multi-species field solve."""
    if model is not None and model_graph is not None:
        raise TypeError("emit_cpp_program received competing model and model_graph authorities")
    if model_graph is not None and type(model_graph) is not ProgramModelGraph:
        raise TypeError("model_graph must be an exact ProgramModelGraph")
    authority = model_graph if model_graph is not None else model
    program = resolve_program_spaces(program, authority)
    if target not in ("system", "amr_system"):
        raise ValueError("emit_cpp_program: target 'system' | 'amr_system' (got %r)" % (target,))
    program.validate()
    _check_lowerable(program, authority)
    prelude, body = _emit_body(program, authority, target=target)
    # Optional dt bound (spec s18 / ADC-417): emit the SECOND ABI pair -- pops_program_has_dt_bound()
    # (true iff a bound was set) and pops_program_dt_bound(ProgramContext*, cfl) (the lowered scalar
    # expression). Without a bound, has_dt_bound() returns false and the dt_bound function returns a
    # +inf sentinel (never reached: the loader stores the closure only when has_dt_bound() is true).
    has_dt_bound, dt_bound_body = _emit_dt_bound(program, authority)
    return _PROGRAM_CPP_TEMPLATE.format(
        name=json.dumps(program.name), hash=program._ir_hash(), prelude=prelude, body=body,
        has_dt_bound=has_dt_bound, dt_bound_body=dt_bound_body,
        module_metadata=_emit_module_metadata(program, authority),
        program_params=_emit_program_params(program, authority),
        block_names=_emit_block_names(program),
        route_manifest=_emit_route_manifest("pops_program_route_manifest"),
        coeff_elliptic_include=_coeff_elliptic_include(program),
        block_inverse_include=_block_inverse_include(program),
        amr_install=_emit_amr_install(program, target, prelude, body))

def _emit_block_names(program: Any) -> str:
    """C++ source of the NAME-based block-binding ABI the .so exports (Spec 3 criterion 23, ADC-457):
    ``pops_program_block_count()`` and ``pops_program_block_name(int)`` -- the Program's block names in
    ``_block_indices`` order (T.state declaration order, the order the step body's ``ctx.state(idx)``
    addresses). System::install_program reads them, matches each to the instantiated System block of
    that name, and stores the program-index -> system-index map (read by ProgramContext), so the
    System blocks may be added in ANY order vs the Program's T.state declarations -- a Program block
    whose name has no System block fails loud. The block names are also part of the IR identity (the
    block_order field of _serialize feeds the IR hash), so reordering T.state changes the hash."""
    order = program._block_indices()  # name -> index, declaration order
    names = sorted(order, key=order.get)
    cases = "".join(
        '    case %d: return %s;\n' % (order[block], json.dumps(block_name(block)))
        for block in names)
    return (
        "// NAME-based block binding (Spec 3 criterion 23, ADC-457): the Program's block names in\n"
        "// T.state declaration order. install_program matches each to a System block BY NAME (not\n"
        "// add-order) and builds the program-index -> system-index map ProgramContext resolves.\n"
        'extern "C" int pops_program_block_count() { return %d; }\n' % len(names) +
        'extern "C" const char* pops_program_block_name(int i) {\n'
        '  switch (i) {\n%s    default: return "";\n  }\n}\n' % cases)

def _emit_dt_bound(program: Any, model: Any = None) -> tuple:
    """Lower the optional dt bound (spec s18 / ADC-417) to ``(has_dt_bound, body)``: the bool literal
    pops_program_has_dt_bound returns and the C++ body of pops_program_dt_bound. No bound -> ("false",
    a +inf return that is never reached). The bound is a READ-ONLY scalar sub-program: it reuses the
    same per-op lowering (state -> ctx.state(idx), reductions, cfl/hmin/max_wave_speed, scalar_op) and
    returns the final scalar. ADC-426: a multi-block dt bound may read several blocks' states (e.g.
    the min over blocks of cfl*hmin/max_wave_speed), so each op resolves its OWN block index / base.
    No commit lives in a dt bound (empty committed_ids)."""
    if program._dt_bound is None:
        return "false", "    return std::numeric_limits<pops::Real>::infinity();"
    sub, result = program._dt_bound
    block_idx = program._block_indices()
    bases = {}
    for v in sub:
        if v.op == "state" and v.block not in bases:
            bases[v.block] = v
    var = {}
    lines = []
    for v in sub:
        _emit_op(program, v, bases.get(v.block), frozenset(), var, model, lines, None, block_idx)
    lines.append("return %s;" % var[result.id])
    body = "\n".join("    " + ln for ln in lines)
    return "true", body



def _check_lowerable(program: Any, model: Any = None) -> None:
    """Raise NotImplementedError if the IR uses a construct the current codegen cannot lower yet,
    naming the offending construct (never a silent mis-lowering). @p model: the physical model that
    declares the named sources / linear sources; required for the Phase-4b ops.

    Multi-block (ADC-426): N ``T.state`` blocks + N ``T.commit`` are supported -- each op routes to
    its block's index (``_block_indices``). Validation: a block is committed AT MOST once (enforced
    at ``commit`` time); a read-only block (declared via ``T.state`` but never committed) is allowed
    (e.g. a passive field whose charge couples the others); a commit of a block that was never
    declared by ``T.state`` is rejected (an unknown-block commit cannot route to an index)."""
    _check_model_owner_dispatch(program, model)
    blocks = program._block_indices()
    for state_ref in program._commits:
        block = state_ref.block_ref
        if block not in blocks:
            raise ValueError(
                "commit of unknown block %r: no T.state(block, U) declares it "
                "(declared blocks: %s)"
                % (block_name(block), sorted(block_name(item) for item in blocks)))
    _check_schedules_lowerable(program)
    for v in program._values:
        _check_op_lowerable(program, v, model)
    # Per-cell dense fallback bound for the local dense solves (mat_inverse<N> uses fixed stack
    # buffers): solve_local_linear (M = I - a*L) and solve_local_nonlinear (the Newton FD Jacobian).
    dense_ops = ("solve_local_linear", "solve_local_nonlinear")
    for value in (v for v in _all_ops(program) if v.op in dense_ops):
        if model is None:
            continue
        impl = _model_impl(model_for_node(model, value))
        n_cons = len(getattr(impl, "cons_names", []) or [])
        if n_cons > 8:
            raise ValueError(
                "local dense fallback for %r supports n_cons <= 8 (got %d)"
                % (value.name, n_cons))


# 'linear_source' is a pure NAME-reference SSA node (vtype 'operator'): it carries no runtime work
# (consumed by apply / solve_local_linear, which read the model coefficients), so it lowers to
# nothing -- always allowed, model or not. 'reduce' / 'compare' / 'while' are the ADC-404a control
# flow / reduction ops (lowered inline via pops::dot; no model needed). 'matrix_free_operator' /
# 'scalar_field' / 'laplacian' / 'gradient' / 'divergence' / 'solve_linear' are the ADC-405 / ADC-412
# matrix-free Krylov ops (the operator declaration carries an apply sub-block; solve_linear lowers to
# pops::*_solve; divergence is the centered FV divergence of a gradient field).

# Ops NOT wrapped in a per-node profile scope (ADC-459): they bind a reference or read a cached
# scalar and do no per-step numerical work, so timing them only adds always-zero noise to
# sim.profile_report(). Every other op that emits a statement is wrapped (rhs / solve_fields /
# linear_combine / source / apply / reductions / loops / Schur kernels / ...).

def _check_op_lowerable(program: Any, v: Any, model: Any) -> None:
    """Lowerability check for a single op (used for both the top-level walk and a while sub-block).
    Raises NotImplementedError / ValueError naming the offending construct (never a mis-lowering)."""
    node_model = (
        model_for_node(model, v)
        if model is not None
        and (v.block is not None or v.attrs.get("operator_handle") is not None)
        else model
    )
    _validate_matrix_free_contract(v, node_model)
    if v.op in _MODEL_OPS:
        if model is None:
            raise NotImplementedError(
                "emit_cpp_program cannot lower op '%s' (value '%s') without the physical model "
                "that declares its named source / linear source; pass model= "
                "(compile_problem threads it through)" % (v.op, v.name))
        if v.op == "solve_local_nonlinear":  # recurse: the residual sub-block ops must lower too
            for w in v.attrs["residual_block"]:
                _check_op_lowerable(program, w, model)
        return  # _emit_op lowers it from the model's symbolic coefficients
    if v.op not in _ALLOWED_OPS:
        raise NotImplementedError(
            "emit_cpp_program cannot lower op '%s' (value '%s') yet; supported ops are %s "
            "(+ %s with a model; nested control flow / Krylov are later phases)"
            % (v.op, v.name, sorted(_ALLOWED_OPS), sorted(_MODEL_OPS)))
    if v.op == "coupled_rate":
        # A coupled_rate (collisions / ionization, Spec 3 criterion 27) lowers to ONE multi-state
        # for_each_cell kernel (see _emit_coupled_rate_kernel). The lowering reaches the operator
        # body (its per-block component formulas) through the BOUND registry, and binds each input
        # state's cons names from that input's StateSpace -- so the operator must be bound and the
        # formulas must be cons-only (the MVP). Validate both here so a non-lowerable coupled_rate
        # fails loud naming ADC-457, never emits an undefined reference.
        _coupled_rate_components(program, v)
        return
    if v.op == "coupled_rate_out":
        # A pure projection of one block out of the coupled bundle: it emits nothing (its var
        # aliases that block's rate scratch). Lowerable iff its producing coupled_rate is (checked
        # when that node is walked); nothing to validate here.
        return
    if v.op in ("while", "range", "if"):  # recurse: the cond / body sub-blocks must lower too
        for key in ("cond_block", "body_block"):
            for w in v.attrs.get(key, []):
                _check_op_lowerable(program, w, model)
        return
    if v.op == "matrix_free_operator":  # recurse into the apply sub-block (set by set_apply)
        if v.attrs.get("apply_block") is None:
            raise ValueError(
                "matrix_free_operator '%s' has no apply; call P.set_apply before lowering"
                % v.name)
        for w in v.attrs["apply_block"]:
            _check_op_lowerable(program, w, model)
        return
    if v.op == "solve_fields":
        # A NAMED elliptic field (ADC-419/ADC-428) drives a SECOND elliptic solve into its own aux
        # channel. The runtime now hosts it (System::solve_fields_from_state(field, ...) via
        # ProgramContext); lowering needs the model so the field name can be validated against the
        # declared m.elliptic_field set (the codegen emits the named ctx call).
        field_ref = v.attrs.get("field")
        if field_ref is not None:
            from pops.time.references import field_name
            field = field_name(field_ref)
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower solve_fields with a named elliptic field "
                    "('%s') without the physical model that declares it (m.elliptic_field); pass "
                    "model= (compile_problem threads it through)" % field)
            if field not in _model_impl(node_model)._elliptic_fields:
                raise ValueError(
                    "unknown elliptic_field '%s' in solve_fields '%s'; declared: %s"
                    % (field, v.name, sorted(_model_impl(node_model)._elliptic_fields)))
        return
    if v.op == "rhs":
        named_fluxes = _named_fluxes(v)
        # ADC-430: flux=False is SOURCE-ONLY -- no -div F base. Named fluxes (a -div of selected
        # flux_terms) contradict "no flux": reject the combination loud rather than silently picking
        # one (request flux=True for named fluxes, or flux=False for a source-only stage).
        if not v.attrs.get("flux", True) and named_fluxes is not None:
            raise ValueError(
                "rhs '%s' sets flux=False (source-only) but also requests named fluxes %r; a "
                "source-only stage has no flux divergence -- drop fluxes= or set flux=True"
                % (v.name, named_fluxes))
        if named_fluxes is not None:  # NAMED fluxes (ADC-419): need the model's flux_term coeffs
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower rhs '%s' with named fluxes %r without the "
                    "physical model that declares them (m.flux_term); pass model= "
                    "(compile_problem threads it through)" % (v.name, named_fluxes))
            impl_f = _model_impl(node_model)
            ft = impl_f._flux_terms
            for f in named_fluxes:
                if f not in ft:
                    raise ValueError(
                        "unknown flux_term '%s' in rhs '%s'; declared flux_terms: %s"
                        % (f, v.name, sorted(ft)))
            # The named-flux path emits -div(selected fluxes) only (no ctx.rhs_into), so the model's
            # DEFAULT source would be silently dropped -- reject it (it must be requested as a named
            # source_term instead). The named sources below are still axpy'd on top.
            if getattr(impl_f, "_source", None):
                raise NotImplementedError(
                    "rhs with named fluxes %r needs a model whose default source is empty (no "
                    "m.source); rhs '%s' has a non-empty default source that the named-flux path "
                    "would drop (declare it as a source_term instead)" % (named_fluxes, v.name))
        extra = [s for s in (v.attrs.get("sources") or []) if s != "default"]
        if not extra:
            return
        # A named source in an rhs reads the model's symbolic source_term coefficients (same as the
        # standalone 'source' op): lowering needs the model.
        if model is None:
            raise NotImplementedError(
                "emit_cpp_program cannot lower rhs '%s' with named sources %r without the "
                "physical model that declares them (m.source_term); pass model= "
                "(compile_problem threads it through)" % (v.name, extra))
        impl = _model_impl(node_model)
        # ADC-425: the named sources are axpy'd on top of an EXPLICIT base. With "default" requested
        # the base is ctx.rhs_into (flux + the model's default/composite source); without it the base
        # is ctx.neg_div_flux_default_into (flux only). Either way the default source is folded in iff
        # the caller listed "default", so adding distinct named source_terms cannot double-count it --
        # the old "model default source must be empty" rejection is gone (the routing is now exact).
        for s in extra:
            if s not in impl._source_terms:
                raise ValueError(
                    "unknown source_term '%s' in rhs '%s'; declared source_terms: %s"
                    % (s, v.name, sorted(impl._source_terms)))
