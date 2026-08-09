"""pops.codegen.program_emit_condensed : the GENERIC condensed-implicit-solve emitters (ADC-637).

The condensed-implicit pattern eliminates a per-cell block-linear source response ``M = I - theta*dt*J``
(J authored via ``m.local_linear_operator`` on a coupled momentum subset K) against a gradient-linear
elliptic coupling, yielding the tensor elliptic coefficient ``A = I + c*rho*M^{-1}``, a fused RHS and a
velocity reconstruction. These three emitters lower those stages to INLINE ``for_each_cell`` kernels
that compute ``M^{-1}`` once per cell with the closed-form ``pops::detail::block_inverse<N>`` intrinsic
(block_inverse.hpp) -- generic in J, with NO physics vocabulary and NO call into ``coupling/schur/**``.

They are the codegen counterpart of the hand-written Schur brick's per-cell kernels: for the
Lorentz linearization ``J = [[0, B_z], [-B_z, 0]]`` the emitted coefficient entries are bit-identical to
``SchurOperatorCoeffKernelC`` (block_inverse<2> == LorentzEliminator, proven in test_block_inverse), so
the retirement parity gate rests on the intrinsic, not on a pattern-match of "is this a rotation?".

The J entries are lowered by the SAME ``Expr.to_cpp()`` + ``_cell_locals`` machinery the model-kernel
emitters use (program_emit_model_kernels / program_emit_kernels). block_inverse is computed ONCE per
cell and reused for every coefficient / apply entry -- the fusion the brick had, preserved by
construction (one for_each_cell per kernel).

R2 (design section 12): the coefficient ``A = I + c*rho*M^{-1}`` reads ``rho`` (a conservative var) in
the OUTER factor only; the block ``M`` is J-only (aux / params, never U|_K). The coeff kernel therefore
binds ``rho`` from the state directly and NEVER feeds it into M -- exactly as SchurOperatorCoeffKernelC
splits ``cr = c*rho`` from ``M^{-1}``.

Naming: the field TOKENS passed in (state / phi / the coefficient shared_ptrs) may be dereferenced-
pointer expressions like ``(*sf4)`` -- valid as MultiFab lvalues (``.local_size()`` / ``.fab(li)`` /
``.box(li)``) but NOT as C++ identifier prefixes. Every NEW local declaration therefore uses a clean
``cond<uid>_<role>`` identifier, never a token as a name prefix.
"""
from __future__ import annotations

import json
from typing import Any

from pops.codegen.program_emit_kernels import _cell_locals, _coeff_cpp, _deref, _model_impl
from pops.codegen.program_emit_model_kernels import _linear_source_rows, _provider_binding


def emit_condensed_op(v: Any, var: Any, model: Any, lines: Any, prelude: Any, *,
                      provider_plans: Any, consumer_qid: str, program_block: int) -> None:
    """Dispatch a condensed_coeffs / condensed_rhs / condensed_reconstruct / condensed_energy op to its
    inline emitter (ADC-637), keeping program_emit_ops.py a thin router. Records the op's C++ token in
    @p var and appends its kernel to @p lines (the coefficient bundle also allocates one persistent
    row-major tensor field in @p prelude, captured by the apply lambda)."""
    if v.op == "condensed_coeffs":
        if prelude is None:
            raise NotImplementedError(
                "condensed_coeffs is only lowerable at the top level / step body, not inside a "
                "control-flow (if/while/range) body")
        (state_in,) = v.inputs
        dimension = len(v.attrs["subset"])
        tensor = "cond_tensor%d" % v.id
        prelude.append(
            "auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
            "ctx.alloc_scalar_field(%d, 1));" % (tensor, dimension * dimension)
        )
        var[v.id] = tensor
        lines += _emit_condensed_coeffs_kernel(
            v.id, model, v.attrs["linear_operator"], v.attrs["subset"], v.attrs["c"],
            v.attrs["th_dt"], v.attrs["c_rho"], "(*%s)" % tensor, var[state_in.id],
            provider_plans=provider_plans, consumer_qid=consumer_qid, program_block=program_block)
        # Coefficient halos: the tensor apply reads neighbouring cells, so the field needs
        # its ghosts filled after assembly. The ctx
        # fill_boundary seam (the transport BC) is bit-identical to the brick's coefficient BC on
        # periodic and zero-gradient (Foextrap) sides -- the whole sanctioned condensed envelope. A
        # Dirichlet transport side would differ (the brick forces Foextrap on the coefficients); lifting
        # that needs a ctx coefficient-BC seam, batched with the brick retirement (header change).
        lines.append("ctx.fill_boundary(*%s);" % tensor)
    elif v.op == "condensed_rhs":
        out_in, phi_in, state_in = v.inputs
        lines += _emit_condensed_rhs_kernel(
            v.id, model, v.attrs["linear_operator"], v.attrs["subset"], v.attrs["th_dt"],
            v.attrs["g"], var[out_in.id], var[phi_in.id], var[state_in.id],
            provider_plans=provider_plans, consumer_qid=consumer_qid, program_block=program_block)
        var[v.id] = var[out_in.id]
    elif v.op == "condensed_reconstruct":
        state_in, phi_in = v.inputs
        lines += _emit_condensed_reconstruct_kernel(
            v.id, model, v.attrs["linear_operator"], v.attrs["subset"], v.attrs["th_dt"],
            v.attrs["c_rho"], var[state_in.id], var[phi_in.id],
            provider_plans=provider_plans, consumer_qid=consumer_qid, program_block=program_block)
        var[v.id] = var[state_in.id]
    else:  # condensed_energy
        state_in, old_in = v.inputs
        lines += _emit_condensed_energy_kernel(
            v.id, v.attrs["c_rho"], v.attrs["subset"], v.attrs["c_E"],
            var[state_in.id], var[old_in.id])
        var[v.id] = var[state_in.id]


def _subset_block_rows(impl: Any, op_name: Any, subset: Any) -> Any:
    """The n x n submatrix (n = len(@p subset)) of the authored linear operator @p op_name restricted to
    the coupled components @p subset: ``J_K[r][c] = J[subset[r]][subset[c]]`` (Expr). J is authored via
    ``m.local_linear_operator``/``m.linear_source`` as the full n_cons x n_cons matrix; the coupled block
    is the momentum subset the condensed solve eliminates. Coefficients depend on aux / params only (the
    linear_source cons/prim-free invariant), so the block is constant in U|_K -- the eliminable class."""
    rows = _linear_source_rows(impl, op_name)  # n_cons x n_cons Expr matrix
    n_cons = len(rows)
    for c in subset:
        if not (0 <= c < n_cons):
            raise ValueError(
                "condensed emit: subset component %d is out of range for operator '%s' (n_cons=%d)"
                % (c, op_name, n_cons))
    return [[rows[r][c] for c in subset] for r in subset]


def _emit_block_M(body: Any, impl: Any, jblock: Any, th_dt_cpp: Any, indent: Any,
                  provider_binding: Any) -> Any:
    """Emit ``M = I - th_dt*J`` from the subset block @p jblock (n x n Expr) into the local ``M_[n][n]``,
    each line prefixed with @p indent. Binds the aux / param locals the J entries reference FIRST (via
    _cell_locals, cons/prim-free) and the th_dt_ scalar. Returns n. Shared by the coefficient path (which
    then inverts M with block_inverse) and the vector-apply paths (which call block_apply_inverse on M_).
    """
    n = len(jblock)
    flat = [e for row in jblock for e in row]
    # aux / param locals the J entries read (cons/prim-free by the linear_source invariant): bound once.
    # _cell_locals reads state only for cons/prim (both False here), so the state_var arg is unused.
    for ln in _cell_locals(
        impl, flat, "STATE_UNUSED", with_cons=False, with_prim=False,
        provider_binding=provider_binding,
    ):
        body.append(indent + ln)
    body.append("%sconst pops::Real th_dt_ = %s;" % (indent, th_dt_cpp))
    body.append("%spops::Real M_[%d][%d];" % (indent, n, n))
    for r in range(n):
        for c in range(n):
            ident = "pops::Real(1)" if r == c else "pops::Real(0)"
            body.append("%sM_[%d][%d] = %s - th_dt_ * (%s);"
                        % (indent, r, c, ident, jblock[r][c].to_cpp()))
    return n


def _emit_block_inverse(body: Any, impl: Any, jblock: Any, th_dt_cpp: Any, indent: Any,
                        provider_binding: Any) -> Any:
    """Emit ``M = I - th_dt*J`` (via _emit_block_M) and all entries ``Mi_ = M^{-1}`` via
    ``pops::detail::block_inverse<n>``, into @p body. Returns n so the caller reads Mi_[r][c]. This is the
    COEFFICIENT primitive: the tensor ``A = I + c*rho*M^{-1}`` reads the entries directly, and each
    block_inverse<2> entry is a DIRECT division (bit-identical to LorentzEliminator's binv_11..22). The
    VECTOR applies (flux, reconstruct) do NOT use this -- they call block_apply_inverse on M_ so the
    single reciprocal is factored out of the bracket (apply_Binv order, bit-exact); see _emit_apply_minv.
    block_inverse computes the inverse once per cell; the caller reuses it for the full tensor."""
    n = _emit_block_M(body, impl, jblock, th_dt_cpp, indent, provider_binding)
    body.append("%spops::Real Mi_[%d][%d];" % (indent, n, n))
    # block_inverse returns false on a singular M; we do not branch in the device kernel (no throw on
    # device). M = I - th_dt*J is invertible for a well-posed eliminable source (Lorentz: det = 1 + w^2
    # > 0); a singular authored block yields a non-finite result surfacing downstream, not a wrong one.
    body.append("%spops::detail::block_inverse<%d>(M_, Mi_);" % (indent, n))
    return n


def _emit_apply_minv(
    body: Any,
    input_components: list[str],
    output_names: list[str],
    indent: Any,
) -> None:
    """Emit ``out = M^{-1} input`` in the factored order via
    ``pops::detail::block_apply_inverse<n>`` on the local ``M_`` (emitted by _emit_block_M): one
    reciprocal ``1/det`` factored out of the adjugate-vector bracket. For the Lorentz block this is
    ``LorentzEliminator::apply_Binv`` bit-for-bit -- the flux / reconstruct parity the retirement gate
    rests on. Summing the pre-divided block_inverse entries would round differently (a per-step ULP
    drift). The input vector and the outputs are named C++ scalars; the block-inverse local M_ is reused
    for every apply in the same cell (the brick's fusion)."""
    n = len(input_components)
    if len(output_names) != n:
        raise ValueError("condensed inverse apply output rank differs from its input rank")
    body.append(
        "%spops::Real cond_v_[%d] = {%s};"
        % (indent, n, ", ".join(input_components))
    )
    body.append("%spops::Real cond_mv_[%d];" % (indent, n))
    body.append("%spops::detail::block_apply_inverse<%d>(M_, cond_v_, cond_mv_);" % (indent, n))
    for component, output_name in enumerate(output_names):
        body.append(
            "%sconst pops::Real %s = cond_mv_[%d];"
            % (indent, output_name, component)
        )


def _emit_condensed_coeffs_kernel(
    uid: Any,
    model: Any,
    jblock_op: Any,
    subset: Any,
    c_coeff: Any,
    th_dt: Any,
    c_rho: Any,
    tensor: Any,
    state_var: Any,
    *, provider_plans: Any, consumer_qid: str, program_block: int,
) -> list:
    """Emit ``A = I + c*rho*M^{-1}`` into one row-major ``Dim*Dim`` field.

    The subset rank is authenticated against the native layout before emission.  A single fused
    kernel therefore covers 1D, 2D and 3D by iterating matrix rows and columns generated from the
    authored subset; no tensor component is invented or ignored.
    """
    impl = _model_impl(model)
    dimension = len(subset)
    if dimension not in (1, 2, 3):
        raise ValueError("condensed coefficient tensor rank must be 1, 2, or 3")
    jblock = _subset_block_rows(impl, jblock_op, subset)
    provider_binding = _provider_binding(
        impl, [entry for row in jblock for entry in row], provider_plans, consumer_qid)
    c_cpp = _coeff_cpp(c_coeff)
    th_dt_cpp = _coeff_cpp(th_dt)
    tensor_write = "cond%s_tensorW" % uid
    body = [
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        'ctx.assembly_target(%s, "pops.tensor-elliptic.coefficients");'
        % (tensor_write, tensor),
        "for (int li = 0; li < %s.local_size(); ++li) {" % tensor_write,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> tensorA = "
        "%s.fab(li).view();" % tensor_write,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> stateA = "
        "std::as_const(%s).fab(li).view();" % state_var,
        "  const auto providers = ctx.template provider_values_view<%d>(%s, %d, li);"
        % (provider_binding["count"], json.dumps(provider_binding["qid"]), program_block),
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % tensor_write,
        "    const pops::Real rho = stateA(index, %d);" % int(c_rho),
    ]
    _emit_block_inverse(body, impl, jblock, th_dt_cpp, "    ", provider_binding)
    body.append("    const pops::Real cr = (%s) * rho;  // c*rho: the outer factor (R2: rho not in M)"
                % c_cpp)
    for row in range(dimension):
        for column in range(dimension):
            identity = "pops::Real(1)" if row == column else "pops::Real(0)"
            body.append(
                "    tensorA(index, %d) = %s + cr * Mi_[%d][%d];"
                % (row * dimension + column, identity, row, column)
            )
    body += ["  });", "}"]
    return body


def _emit_condensed_flux_kernel(body: Any, uid: Any, impl: Any, jblock: Any, th_dt_cpp: Any,
                                subset: Any, fx_var: Any, state_var: Any, provider_binding: Any,
                                program_block: int) -> None:
    """Emit ``F = M^{-1} momentum`` into one component per exact native axis."""
    body += [
        "for (int li = 0; li < %s.local_size(); ++li) {" % fx_var,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> fA = "
        "%s.fab(li).view();" % fx_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> stateA = "
        "std::as_const(%s).fab(li).view();" % state_var,
        "  const auto providers = ctx.template provider_values_view<%d>(%s, %d, li);"
        % (provider_binding["count"], json.dumps(provider_binding["qid"]), program_block),
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % fx_var,
    ]
    n = _emit_block_M(body, impl, jblock, th_dt_cpp, "    ", provider_binding)
    inputs = ["stateA(index, %d)" % int(component) for component in subset]
    outputs = ["cond_flux_%d_" % component for component in range(n)]
    _emit_apply_minv(body, inputs, outputs, "    ")
    for component, output in enumerate(outputs):
        body.append("    fA(index, %d) = %s;" % (component, output))
    body += ["  });", "}"]


def _emit_condensed_rhs_kernel(uid: Any, model: Any, jblock_op: Any, subset: Any, th_dt: Any,
                               g_coeff: Any, rhs_var: Any, phi_n_var: Any, state_var: Any,
                               *, provider_plans: Any, consumer_qid: str, program_block: int) -> list:
    """Emit ``rhs = -Lap(phi_n) - g*div(M^{-1} momentum)`` for the exact native rank."""
    impl = _model_impl(model)
    jblock = _subset_block_rows(impl, jblock_op, subset)
    provider_binding = _provider_binding(
        impl, [entry for row in jblock for entry in row], provider_plans, consumer_qid)
    th_dt_cpp = _coeff_cpp(th_dt)
    g_cpp = _coeff_cpp(g_coeff)
    rhs = _deref(rhs_var)
    lap = "cond%s_lap" % uid
    negl = "cond%s_neglap" % uid
    fx = "cond%s_flux" % uid
    flux_write = "cond%s_fluxW" % uid
    rhs_write = "cond%s_rhsW" % uid
    dimension = len(subset)
    body = [
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        "ctx.scalar_scratch(%d, 0, %s, 1, 0);" % (lap, uid, rhs),
        "ctx.laplacian(%s, %s);" % (lap, _deref(phi_n_var)),
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        "ctx.scalar_scratch(%d, 1, %s, 1, 0);" % (negl, uid, rhs),
        "for (int li = 0; li < %s.local_size(); ++li) {" % negl,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> nlA = "
        "%s.fab(li).view();" % negl,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> lapA = "
        "std::as_const(%s).fab(li).view();" % lap,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % negl,
        "    nlA(index, 0) = -lapA(index, 0);",
        "  });",
        "}",
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        "ctx.scalar_scratch(%d, 2, %s, %d, 1);" % (fx, uid, rhs, dimension),
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        'ctx.assembly_target(%s, "pops.tensor-elliptic.flux");'
        % (flux_write, fx),
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        'ctx.assembly_target(%s, "pops.tensor-elliptic.rhs");'
        % (rhs_write, rhs),
    ]
    _emit_condensed_flux_kernel(
        body, uid, impl, jblock, th_dt_cpp, subset, flux_write, state_var, provider_binding,
        program_block,
    )
    body.append("ctx.fill_boundary(%s);" % flux_write)
    body += [
        "const pops::Geometry<pops::kNativeDimension> cond%s_geometry = ctx.geometry();" % uid,
        "const pops::Real cond%s_g = %s;" % (uid, g_cpp),
        "for (int li = 0; li < %s.local_size(); ++li) {" % rhs_write,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> rhsA = "
        "%s.fab(li).view();" % rhs_write,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> nlA = "
        "std::as_const(%s).fab(li).view();" % negl,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> fA = "
        "std::as_const(%s).fab(li).view();" % flux_write,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % rhs_write,
        "    pops::Real divF = pops::Real(0);",
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      pops::CellIndex<pops::kNativeDimension> lower = index;",
        "      pops::CellIndex<pops::kNativeDimension> upper = index;",
        "      --lower[axis];",
        "      ++upper[axis];",
        "      divF += (fA(upper, axis) - fA(lower, axis)) / "
        "(pops::Real(2) * cond%s_geometry.spacing(axis));" % uid,
        "    }",
        "    rhsA(index, 0) = nlA(index, 0) - cond%s_g * divF;" % uid,
        "  });",
        "}",
    ]
    return body


def _emit_condensed_reconstruct_kernel(uid: Any, model: Any, jblock_op: Any, subset: Any,
                                       th_dt: Any, c_rho: Any, state_var: Any, phi_var: Any,
                                       *, provider_plans: Any, consumer_qid: str,
                                       program_block: int) -> list:
    """Emit the exact-ranked velocity reconstruction and write momentum in place."""
    impl = _model_impl(model)
    jblock = _subset_block_rows(impl, jblock_op, subset)
    provider_binding = _provider_binding(
        impl, [entry for row in jblock for entry in row], provider_plans, consumer_qid)
    th_dt_cpp = _coeff_cpp(th_dt)
    state = state_var
    phi = _deref(phi_var)
    phi_read = "cond%s_phiR" % uid
    dimension = len(subset)
    body = [
        "pops::MultiFab<pops::kNativeDimension>& %s = "
        'ctx.assembly_source(%s, "pops.tensor-elliptic.solution");'
        % (phi_read, phi),
        "ctx.fill_boundary(%s);" % phi_read,
        "const pops::Geometry<pops::kNativeDimension> cond%s_geometry = ctx.geometry();" % uid,
        "for (int li = 0; li < %s.local_size(); ++li) {" % state,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> stateA = "
        "%s.fab(li).view();" % state,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> phiA = "
        "std::as_const(%s).fab(li).view();" % phi_read,
        "  const auto providers = ctx.template provider_values_view<%d>(%s, %d, li);"
        % (provider_binding["count"], json.dumps(provider_binding["qid"]), program_block),
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % state,
        "    const pops::Real rho = stateA(index, %d);" % int(c_rho),
        "    const pops::Real inv_rho = rho != pops::Real(0) ? pops::Real(1) / rho : pops::Real(0);",
        "    const int cond_components_[%d] = {%s};" % (
            dimension, ", ".join(str(int(component)) for component in subset)),
        "    pops::Real cond_residual_[%d];" % dimension,
    ]
    _emit_block_M(body, impl, jblock, th_dt_cpp, "    ", provider_binding)
    body += [
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      pops::CellIndex<pops::kNativeDimension> lower = index;",
        "      pops::CellIndex<pops::kNativeDimension> upper = index;",
        "      --lower[axis];",
        "      ++upper[axis];",
        "      const pops::Real gradient = (phiA(upper, 0) - phiA(lower, 0)) / "
        "(pops::Real(2) * cond%s_geometry.spacing(axis));" % uid,
        "      cond_residual_[axis] = stateA(index, cond_components_[axis]) * inv_rho "
        "- th_dt_ * gradient;",
        "    }",
    ]
    outputs = ["cond_velocity_%d_" % component for component in range(dimension)]
    _emit_apply_minv(
        body,
        ["cond_residual_[%d]" % component for component in range(dimension)],
        outputs,
        "    ",
    )
    for component, output in enumerate(outputs):
        body.append(
            "    stateA(index, cond_components_[%d]) = rho * %s;"
            % (component, output)
        )
    body += [
        "  });",
        "}",
    ]
    return body


def _emit_condensed_energy_kernel(uid: Any, c_rho: Any, subset: Any, c_E: Any,
                                  state_var: Any, old_var: Any) -> list:
    """Emit the exact-ranked kinetic-energy increment in one fused kernel."""
    del uid
    dimension = len(subset)
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % state_var,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> stateA = "
        "%s.fab(li).view();" % state_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> oldA = "
        "std::as_const(%s).fab(li).view();" % old_var,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % state_var,
        "    const pops::Real rho = stateA(index, %d);" % int(c_rho),
        "    const pops::Real inv_rho = rho != pops::Real(0) ? pops::Real(1) / rho : pops::Real(0);",
        "    const int cond_components_[%d] = {%s};" % (
            dimension, ", ".join(str(int(component)) for component in subset)),
        "    pops::Real speed_new_squared = pops::Real(0);",
        "    pops::Real speed_old_squared = pops::Real(0);",
        "    for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "      const pops::Real velocity_new = "
        "stateA(index, cond_components_[axis]) * inv_rho;",
        "      const pops::Real velocity_old = "
        "oldA(index, cond_components_[axis]) * inv_rho;",
        "      speed_new_squared += velocity_new * velocity_new;",
        "      speed_old_squared += velocity_old * velocity_old;",
        "    }",
        "    stateA(index, %d) += pops::Real(0.5) * rho * "
        "(speed_new_squared - speed_old_squared);" % int(c_E),
        "  });",
        "}",
    ]
