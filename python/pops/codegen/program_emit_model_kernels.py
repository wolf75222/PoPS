"""pops.codegen.program_emit_model_kernels : the model-coefficient per-cell kernels.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  These Phase-4b helpers emit the body of a for_each_cell
kernel over the VALID cells of each local fab from a physical model's symbolic coefficients
(source_term / flux_term / linear_source).  They reuse the shared primitives in
``program_emit_kernels`` (``_kernel_open`` / ``_cell_locals`` / ``_coeff_cpp`` / ...).
"""
from __future__ import annotations

from typing import Any

from pops.identity.scalar import scalar_cpp
from pops.model.state_symbols import state_component_symbol
from pops.codegen.cpp_writer import _cse_emit

from pops.codegen.program_emit_kernels import (
    _aux_comp,  # noqa: F401
    _cell_locals,
    _coeff_cpp,
    _has_runtime_param,
    _kernel_close,
    _kernel_open,
    _model_impl,
)


def _emit_local_transform_kernel(
    model: Any, name: Any, state_var: Any, out_var: Any, status_var: Any,
    active_mask_var: Any, block_idx: Any = 0,
) -> list:
    """Lower one named pointwise State -> State map into a fail-closed device kernel."""

    impl = _model_impl(model)
    transforms = getattr(impl, "_local_transforms", {}) or {}
    if name not in transforms:
        raise NotImplementedError(
            "emit_cpp_program: local transform '%s' is not declared; declared: %s"
            % (name, sorted(transforms)))
    declaration = transforms[name]
    exprs = list(declaration["expressions"])
    valid_if = declaration["valid_if"]
    if len(exprs) != len(impl.cons_names):
        raise ValueError(
            "local transform '%s' has %d outputs for %d conservative components"
            % (name, len(exprs), len(impl.cons_names)))
    roots = exprs + [valid_if]
    impl.assign_runtime_indices()
    params_block = block_idx if _has_runtime_param(roots) else None
    body = _kernel_open(out_var, state_var, params_block)
    lambda_index = next(
        index for index, line in enumerate(body) if "pops::for_each_cell" in line)
    body[lambda_index:lambda_index] = [
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> statusA = "
        "%s.fab(li).view();" % status_var,
        "  const bool transform_has_active_mask_ = %s != nullptr;" % active_mask_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> "
        "transform_activeA_ = transform_has_active_mask_ "
        "? std::as_const(*%s).fab(li).view() "
        ": pops::FieldView<const pops::Real, pops::kNativeDimension>{};"
        % active_mask_var,
    ]
    body += [
        "    if (transform_has_active_mask_ && "
        "!(transform_activeA_(index, 0) >= pops::Real(0.5))) {",
    ]
    for component in range(len(exprs)):
        body.append(
            "      outA(index, %d) = %sA(index, %d);"
            % (component, state_var, component))
    body += [
        "      statusA(index, 0) = pops::Real(0);",
        "      return;",
        "    }",
    ]
    body += ["    " + line for line in _cell_locals(
        impl, roots, state_var, with_cons=True, with_prim=True)]
    temporaries, rendered, temporary_names = _cse_emit(
        roots, "pops::Real", "    ", materialize_all=True, return_names=True)
    body.append("    pops::Real transform_failed_ = pops::Real(0);")
    for declaration_line, temporary_name in zip(
        temporaries, temporary_names, strict=True,
    ):
        body.append(declaration_line)
        body.append(
            "    if (!Kokkos::isfinite(%s)) transform_failed_ = pops::Real(1);"
            % temporary_name)
    for component, expression in enumerate(rendered[:-1]):
        body.append(
            "    const pops::Real transformed_%d_ = %s;" % (component, expression))
    body.append("    const pops::Real transform_valid_ = %s;" % rendered[-1])
    for component in range(len(exprs)):
        body.append(
            "    if (!Kokkos::isfinite(%sA(index, %d))) transform_failed_ = pops::Real(1);"
            % (state_var, component))
    body.append(
        "    if (!Kokkos::isfinite(transform_valid_) || "
        "!(transform_valid_ != pops::Real(0))) transform_failed_ = pops::Real(1);")
    for component in range(len(exprs)):
        body.append(
            "    if (!Kokkos::isfinite(transformed_%d_)) transform_failed_ = pops::Real(1);"
            % component)
        body.append(
            "    outA(index, %d) = transformed_%d_;" % (component, component))
    body.append("    statusA(index, 0) = transform_failed_;")
    body += _kernel_close()
    return body


def _emit_source_kernel(model: Any, name: Any, state_var: Any, out_var: Any, block_idx: Any = 0) -> list:
    """Lower ``source`` (a named ``m.source_term``): outA(i,j,c) = S_c(U, prims, aux, params) per cell.

    @p block_idx (ADC-510): the PROGRAM block index whose RuntimeParams the kernel reads when a source
    expression references a canonical RuntimeParam read. The model's runtime
    indices are assigned here (idempotent, sorted-name order matching the .so metadata + the per-block
    ``ctx.program_params`` store), so a RuntimeParamRef lowers to ``params.get(<index>)`` and _kernel_open
    binds the ``params`` struct; a source reading no runtime param is byte-identical (params_block None)."""
    impl = _model_impl(model)
    if name not in impl._source_terms:
        raise NotImplementedError(
            "emit_cpp_program: source '%s' is not declared on the model (m.source_term); declared: %s"
            % (name, sorted(impl._source_terms)))
    exprs = impl._source_terms[name]
    impl.assign_runtime_indices()  # stable params.get(idx) indices BEFORE any to_cpp() (no-op if none)
    params_block = block_idx if _has_runtime_param(exprs) else None
    body = _kernel_open(out_var, state_var, params_block)
    body += ["    " + ln for ln in _cell_locals(impl, exprs, state_var, with_cons=True,
                                                 with_prim=True)]
    body += ["    outA(index, %d) = %s;" % (c, e.to_cpp()) for c, e in enumerate(exprs)]
    body += _kernel_close()
    return body


def _component_sources(
    referenced: set[str], by_block: Any, source_for_state: Any,
) -> dict[str, Any]:
    """Map exact symbolic coordinates to their generated per-cell source.

    Qualified coordinates are total for arbitrary overlapping StateSpaces. Bare
    coordinates are a convenience only when unique across every input space.
    """
    from collections import Counter

    states = [state for state in by_block.values() if state.space is not None]
    counts = Counter(
        component for state in states for component in state.space.components)
    ambiguous = sorted(
        component for component, count in counts.items()
        if count > 1 and component in referenced)
    if ambiguous:
        raise ValueError(
            "multi-state operator references ambiguous bare component(s) %s; obtain exact "
            "coordinates with module.state_symbols(state_space)" % ambiguous)
    sources = {}
    for state in states:
        for index, component in enumerate(state.space.components):
            source = source_for_state(state, index)
            qualified = state_component_symbol(state.space, component)
            if qualified in referenced:
                sources[qualified] = source
            if counts[component] == 1 and component in referenced:
                sources[component] = source
    missing = sorted(referenced - set(sources))
    if missing:
        raise ValueError(
            "multi-state operator references conservative symbol(s) %s that belong to no input "
            "StateSpace" % missing)
    return sources


def _emit_coupled_rate_kernel(components: Any, by_block: Any, var: Any, scratch: Any) -> list:
    """Lower a ``coupled_rate`` (Spec 3 criterion 27, ADC-457) to ONE multi-state for_each_cell kernel
    filling every participating block's rate scratch at once.

    @p components: ``{block: [Expr, ...]}`` -- the per-block component formulas (cons-only MVP).
    @p by_block:   ``{block: state Value}`` -- each block's input state (its StateSpace gives the cons
                   names + their component indices; its C++ token gives the ranked read FieldView).
    @p var:        the id -> C++ token map (the input states are already bound to ``ctx.state(idx)``).
    @p scratch:    ``{block: scratch var name}`` -- the per-block rate scratch (alloc'd by the caller).

    The component formulas reference cons vars from MULTIPLE input states, so the blocks share ONE loop
    (they cannot be independent single-block rates). The first block drives the loop; all inputs and
    scratches are co-located (same ba/dm as the System aux), so ``fab(li)`` is the same box on every
    rank -- the co-distribution every aux-reading kernel relies on (see _kernel_open). Each input
    state binds its OWN read handle (``<state token>A``); a referenced cons var binds from its state's
    FieldView at its component index. A cons NAME shared by two states' components AND referenced by a
    formula is ambiguous (no single source) -- rejected loud, never silently bound to one state."""
    blocks = list(components)
    driver = scratch[blocks[0]]                  # the block whose box / local_size drives the loop
    # Which cons vars does any formula reference, and from which state does each come?
    referenced = set()
    for comps in components.values():
        for e in comps:
            referenced |= e.deps()
    cons_source = _component_sources(
        referenced, by_block, lambda state, index: (var[state.id], index))

    def state_handle(token: Any) -> str:
        return "%sA" % token                     # read handle for an input state token (u0A / u1A)

    lines = ["for (int li = 0; li < %s.local_size(); ++li) {" % driver]
    # Bind a write handle per OUTPUT block scratch, then a read handle per DISTINCT input state that a
    # formula actually reads (incl. a read-only catalyst input that is not an output block), all inside
    # the per-fab loop and BEFORE for_each_cell so the device lambda captures them by value.
    for blk in blocks:
        lines.append(
            "  const pops::FieldView<pops::Real, pops::kNativeDimension> %sA = "
            "%s.fab(li).view();" % (scratch[blk], scratch[blk])
        )
    read_tokens = {src[0] for src in cons_source.values()}
    seen_states = []
    for st in by_block.values():                 # input order (v.inputs); deterministic
        tok = var[st.id]
        if tok in read_tokens and tok not in seen_states:
            seen_states.append(tok)
            lines.append(
                "  const pops::FieldView<const pops::Real, pops::kNativeDimension> %s = "
                "std::as_const(%s).fab(li).view();" % (state_handle(tok), tok)
            )
    lines.append(
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % driver
    )
    for c in sorted(cons_source):                # bind only the referenced cons (no unused locals)
        tok, idx = cons_source[c]
        lines.append("    const pops::Real %s = %s(index, %d);" % (c, state_handle(tok), idx))
    for blk in blocks:
        for comp, e in enumerate(components[blk]):
            lines.append("    %sA(index, %d) = %s;" % (scratch[blk], comp, e.to_cpp()))
    lines += ["  });", "}"]
    return lines


def _prepared_local_control_lines(attrs: Any, *, indent: str = "    ") -> list[str]:
    safeguard = {
        "exact": "pops::LocalSafeguardKind::kExactNewton",
        "damped": "pops::LocalSafeguardKind::kFixedDamping",
        "backtracking": "pops::LocalSafeguardKind::kBacktrackingLineSearch",
    }[attrs["safeguard"]]
    return [
        indent + "pops::PreparedLocalNonlinearControls controls_;",
        indent + "controls_.absolute_tolerance = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["tol"]),
        indent + "controls_.relative_tolerance = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["relative_tol"]),
        indent + "controls_.step_tolerance = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["step_tol"]),
        indent + "controls_.max_iterations = %d;" % int(attrs["max_iter"]),
        indent + "controls_.max_evaluations = %d;" % int(attrs["max_evaluations"]),
        indent + "controls_.finite_difference_step = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["fd_eps"]),
        indent + "controls_.safeguard = %s;" % safeguard,
        indent + "controls_.initial_step = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["damping"]),
        indent + "controls_.max_backtracks = %d;" % int(attrs["max_backtracks"]),
        indent + "controls_.minimum_step = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["minimum_step"]),
        indent + "controls_.armijo = static_cast<pops::Real>(%s);"
        % scalar_cpp(attrs["armijo"]),
    ]


def _emit_solve_coupled_implicit_kernel(components: Any, by_block: Any, var: Any,
                                        scratch: Any, status: str, *, controls: Any,
                                        coefficient: Any) -> list:
    """Emit one fail-closed prepared nonlinear solve over a coupled ``RateBundle``.

    Every output block is an unknown; additional signed inputs are frozen catalysts.  Results land
    only in fresh scratches.  The generated route contributes only a residual functor and controls;
    the unique prepared provider owns convergence, finite-difference Jacobians, pivoted factorization,
    budgets and diagnostics.
    """
    blocks = list(components)
    offsets = {}
    total = 0
    for block in blocks:
        offsets[block] = total
        total += len(components[block])
    referenced = {name for rows in components.values() for expr in rows for name in expr.deps()}
    sources = _component_sources(
        referenced,
        by_block,
        lambda state, index: (
            ("unknown", offsets[state.block] + index)
            if state.block in offsets else ("frozen", var[state.id], index)),
    )
    driver = scratch[blocks[0]]
    coefficient_cpp = scalar_cpp(coefficient)
    lines = ["for (int li = 0; li < %s.local_size(); ++li) {" % driver]
    for block in blocks:
        lines.append(
            "  const pops::FieldView<pops::Real, pops::kNativeDimension> %sA = "
            "%s.fab(li).view();" % (scratch[block], scratch[block])
        )
    lines.append(
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> %sA = "
        "%s.fab(li).view();" % (status, status)
    )
    seen = set()
    for state in by_block.values():
        token = var[state.id]
        if token not in seen:
            seen.add(token)
            lines.append(
                "  const pops::FieldView<const pops::Real, pops::kNativeDimension> %sA = "
                "std::as_const(%s).fab(li).view();" % (token, token)
            )
    lines.append(
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % driver
    )
    lines.append("    pops::Real G_[%d];" % total)
    for block in blocks:
        state = by_block[block]
        for index in range(len(components[block])):
            lines.append("    G_[%d] = %sA(index, %d);"
                         % (offsets[block] + index, var[state.id], index))
    lines.append(
        "    auto residual_eval = [&](const pops::Real (&Ueval)[%d], pops::Real (&rout)[%d]) {"
        % (total, total))
    for component in sorted(referenced):
        source = sources[component]
        if source[0] == "unknown":
            lines.append("      const pops::Real %s = Ueval[%d];" % (component, source[1]))
        else:
            lines.append("      const pops::Real %s = %sA(index, %d);"
                         % (component, source[1], source[2]))
    for block in blocks:
        for index, expr in enumerate(components[block]):
            slot = offsets[block] + index
            lines.append(
                "      rout[%d] = Ueval[%d] - G_[%d] - "
                "static_cast<pops::Real>(%s) * dt * (%s);"
                % (slot, slot, slot, coefficient_cpp, expr.to_cpp()))
    lines.append("    };")
    lines += _prepared_local_control_lines(controls)
    lines.append(
        "    const auto prepared_ = pops::prepare_local_nonlinear_problem<%d>("
        "residual_eval, pops::FiniteDifferenceLocalJacobian<%d>{}, "
        "pops::AcceptAllLocalCandidates<%d>{}, controls_);"
        % (total, total, total))
    lines.append(
        "    const pops::LocalNonlinearCellResult<%d> solved_ = "
        "pops::solve_prepared_local_nonlinear(prepared_, G_);" % total)
    for block in blocks:
        for index in range(len(components[block])):
            lines.append("    %sA(index, %d) = solved_.value[%d];"
                         % (scratch[block], index, offsets[block] + index))
    lines += [
        "    %sA(index, 0) = static_cast<pops::Real>("
        "pops::local_nonlinear_status_code(solved_.status));" % status,
        "    %sA(index, 1) = static_cast<pops::Real>(solved_.iterations);" % status,
        "    %sA(index, 2) = static_cast<pops::Real>(solved_.evaluations);" % status,
        "    %sA(index, 3) = solved_.reference_residual_norm;" % status,
        "    %sA(index, 4) = solved_.residual_norm;" % status,
        "    %sA(index, 5) = solved_.step_norm;" % status,
        "    %sA(index, 6) = solved_.condition_evidence;" % status,
        "    %sA(index, 7) = static_cast<pops::Real>(solved_.safeguard_steps);" % status,
        "    %sA(index, 10) = static_cast<pops::Real>("
        "pops::local_nonlinear_status_priority(solved_.status));" % status,
        "    if (!solved_.solved()) {",
        "      %sA(index, 8) = static_cast<pops::Real>(solved_.failing_component);" % status,
        "      %sA(index, 9) = pops::Real(1);" % status,
        "    } else {",
        "      %sA(index, 8) = pops::Real(0);" % status,
        "      %sA(index, 9) = pops::Real(0);" % status,
        "    }",
    ]
    lines += ["  });", "}"]
    return lines


def _emit_flux_kernel(
    model: Any,
    names: Any,
    state_var: Any,
    flux_vars: dict[str, str],
    block_idx: Any = 0,
) -> list:
    """Lower a named physical-flux sum over the model's exact ranked axis set.

    The authored model already owns a canonical ``x[/y[/z]]`` mapping.  The emitted kernel binds
    one exact-ranked field per authored axis and iterates once over ``CellIndex<kNativeDimension>``;
    no two-dimensional fallback or runtime dimension branch is generated.
    """
    impl = _model_impl(model)
    flux_terms = impl._flux_terms
    for name in names:
        if name not in flux_terms:
            raise NotImplementedError(
                "emit_cpp_program: flux '%s' is not declared on the model (m.flux_term); declared: %s"
                % (name, sorted(flux_terms)))
    axes = tuple(flux_terms[names[0]])
    if tuple(flux_vars) != axes:
        raise ValueError(
            "named-flux scratch axes %s differ from the model axes %s"
            % (tuple(flux_vars), axes)
        )
    for name in names[1:]:
        if tuple(flux_terms[name]) != axes:
            raise ValueError("named fluxes must share one exact ranked axis set")
    n = len(impl.cons_names)
    expressions = {
        axis: [flux_terms[names[0]][axis][component] for component in range(n)]
        for axis in axes
    }
    for name in names[1:]:
        for axis in axes:
            expressions[axis] = [
                expressions[axis][component] + flux_terms[name][axis][component]
                for component in range(n)
            ]
    impl.assign_runtime_indices()  # stable params.get(idx) indices BEFORE any to_cpp() (no-op if none)
    roots = [expression for axis in axes for expression in expressions[axis]]
    params_block = block_idx if _has_runtime_param(roots) else None
    first_axis = axes[0]
    body = _kernel_open(
        flux_vars[first_axis], state_var, params_block, ghost_depth=1
    )
    insertion = 3
    handles = {first_axis: "outA"}
    for axis in axes[1:]:
        handle = "flux_%sA" % axis
        handles[axis] = handle
        body.insert(
            insertion,
            "  const pops::FieldView<pops::Real, pops::kNativeDimension> %s = "
            "%s.fab(li).view();" % (handle, flux_vars[axis]),
        )
        insertion += 1
    body += [
        "    " + line
        for line in _cell_locals(
            impl, roots, state_var, with_cons=True, with_prim=True
        )
    ]
    for axis in axes:
        body += [
            "    %s(index, %d) = %s;" % (handles[axis], component, expression.to_cpp())
            for component, expression in enumerate(expressions[axis])
        ]
    body += _kernel_close()
    return body


def _emit_apply_kernel(model: Any, name: Any, state_var: Any, out_var: Any, block_idx: Any = 0) -> list:
    """Lower ``apply`` (a named ``m.linear_source`` L): outA(i,j,r) = sum_c L[r][c](aux, params) *
    U(i,j,c). @p block_idx (ADC-510): the PROGRAM block whose RuntimeParams the L coefficients read
    when one references a runtime parameter (L may depend on aux / const / params, never on U)."""
    impl = _model_impl(model)
    rows = _linear_source_rows(impl, name)
    n = len(rows)
    flat = [e for row in rows for e in row]
    impl.assign_runtime_indices()  # stable params.get(idx) indices BEFORE any to_cpp() (no-op if none)
    params_block = block_idx if _has_runtime_param(flat) else None
    body = _kernel_open(out_var, state_var, params_block)
    # L coefficients depend on aux / const / params only (linear_source invariant): cons/prim locals not needed.
    body += ["    " + ln for ln in _cell_locals(impl, flat, state_var, with_cons=False,
                                                 with_prim=False)]
    for r in range(n):
        terms = [
            "(%s) * %sA(index, %d)" % (rows[r][c].to_cpp(), state_var, c)
            for c in range(n)
        ]
        body.append("    outA(index, %d) = %s;" % (r, " + ".join(terms)))
    body += _kernel_close()
    return body


def _emit_solve_local_linear_kernel(model: Any, name: Any, a_coeff: Any, rhs_var: Any, out_var: Any,
                                    status_var: Any, block_idx: Any = 0) -> list:
    """Lower ``solve_local_linear``: per cell M = I - a*L (a = a_coeff(dt)), invert M (dense N x N
    via pops::detail::mat_inverse) and set outA(i,j,r) = sum_c Minv[r][c] * q(i,j,c), q = the rhs state.
    L's coefficients depend on aux / const / params only, so M is assembled from the aux / param locals +
    the literal a. @p block_idx (ADC-510): the PROGRAM block whose RuntimeParams an L coefficient reads."""
    impl = _model_impl(model)
    rows = _linear_source_rows(impl, name)
    n = len(rows)
    flat = [e for row in rows for e in row]
    a_cpp = _coeff_cpp(a_coeff)
    impl.assign_runtime_indices()  # stable params.get(idx) indices BEFORE any to_cpp() (no-op if none)
    params_block = block_idx if _has_runtime_param(flat) else None
    body = _kernel_open(out_var, rhs_var, params_block)
    lambda_index = next(
        index for index, line in enumerate(body) if "pops::for_each_cell" in line)
    body[lambda_index:lambda_index] = [
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> solve_statusA = "
        "%s.fab(li).view();" % status_var,
    ]
    body += ["    " + ln for ln in _cell_locals(impl, flat, rhs_var, with_cons=False,
                                                 with_prim=False)]
    body.append("    const pops::Real a_ = %s;" % a_cpp)
    body.append("    pops::Real M_[%d][%d];" % (n, n))
    body.append("    int solve_failure_ = 0;")
    for r in range(n):
        for c in range(n):
            ident = "pops::Real(1)" if r == c else "pops::Real(0)"
            body.append("    M_[%d][%d] = %s - a_ * (%s);" % (r, c, ident, rows[r][c].to_cpp()))
            body.append(
                "    if (!std::isfinite(M_[%d][%d])) solve_failure_ = 3;" % (r, c))
    for c in range(n):
        body.append(
            "    if (!std::isfinite(%sA(index, %d))) solve_failure_ = 3;"
            % (rhs_var, c))
    body.append("    pops::Real Minv_[%d][%d];" % (n, n))
    body.append(
        "    if (solve_failure_ == 0 && !pops::detail::mat_inverse<%d>(M_, Minv_)) "
        "solve_failure_ = 2;" % n)
    for r in range(n):
        terms = [
            "Minv_[%d][%d] * %sA(index, %d)" % (r, c, rhs_var, c)
            for c in range(n)
        ]
        body.append(
            "    pops::Real solved_%d_ = %sA(index, %d);" % (r, rhs_var, r))
        body.append(
            "    if (solve_failure_ == 0) solved_%d_ = %s;"
            % (r, " + ".join(terms)))
        body.append(
            "    if (!std::isfinite(solved_%d_)) solve_failure_ = 3;" % r)
        body.append("    outA(index, %d) = solved_%d_;" % (r, r))
    body.append(
        "    solve_statusA(index, 0) = static_cast<pops::Real>(solve_failure_);")
    body += _kernel_close()
    return body


def _residual_term_exprs(impl: Any, w: Any) -> list:
    """The per-component Expr list of one LOCAL residual sub-block op @p w, as a function of the bare
    conservative-variable names (which the Newton kernel binds to the iterate stack ``Ueval[c]``):

      - ``source`` (a named ``m.source_term``): S_c(U) -- the declared source expressions;
      - ``apply`` (a named ``m.linear_source`` L): (L U)_c = sum_k L[c][k] * <cons_k>.

    The iterate / guess State placeholders and ``linear_combine`` are handled by the affine walk in
    `_emit_residual_eval`, not here (they are not standalone-evaluable Exprs)."""
    from pops._ir.expr import Const, Var
    if w.op == "source":
        name = w.attrs["source"]
        if name not in impl._source_terms:
            raise NotImplementedError(
                "emit_cpp_program: residual source '%s' is not declared on the model (m.source_term); "
                "declared: %s" % (name, sorted(impl._source_terms)))
        return list(impl._source_terms[name])
    if w.op == "apply":
        rows = _linear_source_rows(impl, w.attrs["linear_source"])
        n = len(rows)
        # (L U)_r = sum_c L[r][c] * cons_c -- a per-component Expr in the cons names + aux.
        return [sum((rows[r][c] * Var(impl.cons_names[c], "cons") for c in range(n)),
                    Const(0.0)) for r in range(n)]
    raise NotImplementedError(
        "emit_cpp_program: residual op '%s' is not a per-cell Expr term (source / apply only)" % w.op)


def _emit_residual_eval(impl: Any, v: Any, n: Any) -> list:
    """Build the device residual-evaluation lambda body for ``solve_local_nonlinear``: lines computing
    ``rout[0..n-1] = r(Ueval)`` from the iterate stack ``Ueval`` (bound to the conservative names), the
    frozen guess stack ``Gval`` (the initial-guess State, read as a per-cell constant) and the captured
    aux locals. Mirrors the affine walk: each residual sub-block op is one of the iterate / guess State
    placeholders, a ``source`` / ``apply`` per-cell Expr term, or a ``linear_combine`` (an affine over
    earlier terms). The result is the affine the residual returned.

    @p v is the solve_local_nonlinear op; @p n the conservative count. Returns the lambda BODY lines
    (indented two spaces past the lambda header). The lambda captures the aux locals + ``Gval`` by ref."""
    block = v.attrs["residual_block"]
    iterate_id = v.attrs["iterate"].id
    guess_id = v.attrs["guess"].id
    # term id -> a list of n C++ expression strings (one per conservative component). The iterate is the
    # stack Ueval; the guess is the frozen Gval; source / apply lower to Exprs over the cons names.
    comps = {iterate_id: ["Ueval[%d]" % c for c in range(n)],
             guess_id: ["Gval[%d]" % c for c in range(n)]}
    lines = []
    for w in block:
        if w.op == "state":
            continue  # the iterate / guess placeholders: bound in `comps` above, nothing to emit
        if w.op in ("source", "apply"):
            exprs = _residual_term_exprs(impl, w)
            comps[w.id] = ["(%s)" % e.to_cpp() for e in exprs]
        elif w.op == "linear_combine":
            # An affine sum over earlier terms: comps[w] = sum_k coeff_k(dt) * comps[input_k].
            coeffs = w.attrs["coeffs"]  # aligned with w.inputs; each a dt-polynomial power->float dict
            for inp in w.inputs:
                if inp.id not in comps:  # an input outside the residual sub-block (validate() guards this)
                    raise NotImplementedError(
                        "emit_cpp_program: residual combine reads value '%s' which is not produced "
                        "inside the residual (only the iterate / guess and earlier residual ops are "
                        "available to a per-cell Newton kernel)" % inp.name)
            row = []
            for c in range(n):
                parts = []
                for inp, coeff in zip(w.inputs, coeffs, strict=True):
                    parts.append("%s * (%s)" % (_coeff_cpp(coeff), comps[inp.id][c]))
                row.append(" + ".join(parts) if parts else "static_cast<pops::Real>(0)")
            comps[w.id] = row
        else:  # builder guards _RESIDUAL_LOCAL_OPS; this is belt-and-suspenders
            raise NotImplementedError(
                "emit_cpp_program: residual op '%s' is not lowerable in a local Newton kernel" % w.op)
    result = comps[v.attrs["residual"].id]
    for c in range(n):
        lines.append("rout[%d] = %s;" % (c, result[c]))
    return lines


def _emit_solve_local_nonlinear_kernel(
    model: Any,
    v: Any,
    guess_var: Any,
    out_var: Any,
    status_var: Any,
    active_mask_var: Any,
    block_idx: Any = 0,
) -> list:
    """Lower ``solve_local_nonlinear`` to the unique prepared local nonlinear provider.

    Generated code contributes only the residual functor, immutable controls and a
    transaction-local candidate scratch. The provider owns convergence, finite-difference
    Jacobians, pivoted factorization, safeguards, budgets and typed diagnostics.
    """
    impl = _model_impl(model)
    n = len(impl.cons_names)
    fd_eps = v.attrs.get("fd_eps")
    term_exprs = []
    for w in v.attrs["residual_block"]:
        if w.op in ("source", "apply"):
            term_exprs += _residual_term_exprs(impl, w)
    impl.assign_runtime_indices()
    params_block = block_idx if _has_runtime_param(term_exprs) else None
    body = _kernel_open(out_var, guess_var, params_block)
    lambda_index = next(index for index, line in enumerate(body) if "pops::for_each_cell" in line)
    body[lambda_index:lambda_index] = [
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> solve_statusA = "
        "%s.fab(li).view();" % status_var,
        "  const bool nonlinear_has_active_mask_ = %s != nullptr;" % active_mask_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> "
        "nonlinear_activeA_ = nonlinear_has_active_mask_ "
        "? std::as_const(*%s).fab(li).view() "
        ": pops::FieldView<const pops::Real, pops::kNativeDimension>{};"
        % active_mask_var,
    ]
    body += [
        "    if (nonlinear_has_active_mask_ && "
        "!(nonlinear_activeA_(index, 0) >= pops::Real(0.5))) {",
    ]
    for component in range(n):
        body.append(
            "      outA(index, %d) = %sA(index, %d);"
            % (component, guess_var, component)
        )
    body += [
        "      for (int component_ = 0; component_ < 11; ++component_)",
        "        solve_statusA(index, component_) = pops::Real(0);",
        "      return;",
        "    }",
    ]
    body += [
        "    " + line
        for line in _cell_locals(impl, term_exprs, guess_var, with_cons=False, with_prim=False)
    ]
    body.append("    pops::Real Gval[%d];" % n)
    for component in range(n):
        body.append("    Gval[%d] = %sA(index, %d);" % (component, guess_var, component))
    body.append(
        "    auto residual_eval = "
        "[&](const pops::Real (&Ueval)[%d], pops::Real (&rout)[%d]) {" % (n, n)
    )
    for component, name in enumerate(impl.cons_names):
        body.append("      const pops::Real %s = Ueval[%d];" % (name, component))
    live = impl._live_prims(term_exprs) if term_exprs else set()
    for name, expr in impl.prim_defs.items():
        if name in live:
            body.append("      const pops::Real %s = %s;" % (name, expr.to_cpp()))
    body += ["      " + line for line in _emit_residual_eval(impl, v, n)]
    body.append("    };")
    attrs = dict(v.attrs)
    attrs["fd_eps"] = 1.0e-7 if fd_eps is None else fd_eps
    body += _prepared_local_control_lines(attrs)
    body.append(
        "    const auto prepared_ = pops::prepare_local_nonlinear_problem<%d>("
        "residual_eval, pops::FiniteDifferenceLocalJacobian<%d>{}, "
        "pops::AcceptAllLocalCandidates<%d>{}, controls_);" % (n, n, n)
    )
    body.append(
        "    const pops::LocalNonlinearCellResult<%d> solved_ = "
        "pops::solve_prepared_local_nonlinear(prepared_, Gval);" % n
    )
    for component in range(n):
        body.append("    outA(index, %d) = solved_.value[%d];" % (component, component))
    body += [
        "    solve_statusA(index, 0) = static_cast<pops::Real>("
        "pops::local_nonlinear_status_code(solved_.status));",
        "    solve_statusA(index, 1) = static_cast<pops::Real>(solved_.iterations);",
        "    solve_statusA(index, 2) = static_cast<pops::Real>(solved_.evaluations);",
        "    solve_statusA(index, 3) = solved_.reference_residual_norm;",
        "    solve_statusA(index, 4) = solved_.residual_norm;",
        "    solve_statusA(index, 5) = solved_.step_norm;",
        "    solve_statusA(index, 6) = solved_.condition_evidence;",
        "    solve_statusA(index, 7) = static_cast<pops::Real>(solved_.safeguard_steps);",
        "    solve_statusA(index, 10) = static_cast<pops::Real>("
        "pops::local_nonlinear_status_priority(solved_.status));",
        "    if (!solved_.solved()) {",
        "      solve_statusA(index, 8) = static_cast<pops::Real>(solved_.failing_component);",
        "      solve_statusA(index, 9) = pops::Real(1);",
        "    } else {",
        "      solve_statusA(index, 8) = pops::Real(0);",
        "      solve_statusA(index, 9) = pops::Real(0);",
        "    }",
    ]
    body += _kernel_close()
    return body


def _linear_source_rows(impl: Any, name: Any) -> Any:
    """The n_cons x n_cons matrix of Expr of a model linear source @p name (m.linear_source).
    @p impl is the HyperbolicModel."""
    if name not in impl._linear_sources:
        raise NotImplementedError(
            "emit_cpp_program: linear source '%s' is not declared on the model (m.linear_source); "
            "declared: %s" % (name, sorted(impl._linear_sources)))
    return impl._linear_sources[name]
