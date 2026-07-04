"""pops.codegen.program_emit_condensed : the GENERIC condensed-implicit-solve emitters (ADC-637).

The condensed-implicit pattern eliminates a per-cell block-linear source response ``M = I - theta*dt*J``
(J authored via ``m.local_linear_operator`` on a coupled momentum subset K) against a gradient-linear
elliptic coupling, yielding the tensor elliptic coefficient ``A = I + c*rho*M^{-1}``, a fused RHS and a
velocity reconstruction. These three emitters lower those stages to INLINE ``for_each_cell`` kernels
that compute ``M^{-1}`` once per cell with the closed-form ``pops::detail::block_inverse<N>`` intrinsic
(block_inverse.hpp) -- generic in J, with NO physics vocabulary and NO call into ``coupling/schur/**``.

They are the codegen counterpart of the hand-written Schur brick's four per-cell kernels: for the
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

from typing import Any

from pops.codegen.program_emit_kernels import _cell_locals, _coeff_cpp, _deref, _model_impl
from pops.codegen.program_emit_model_kernels import _linear_source_rows


def emit_condensed_op(v: Any, var: Any, model: Any, lines: Any, prelude: Any) -> None:
    """Dispatch a condensed_coeffs / condensed_rhs / condensed_reconstruct op to its inline emitter
    (ADC-637), keeping program_emit_ops.py a thin router. Records the op's C++ token in @p var and
    appends its kernel to @p lines (the coefficient bundle also allocates four persistent coefficient
    shared_ptrs in @p prelude, alloc-once, captured by the apply lambda -- like schur_coeffs)."""
    if v.op == "condensed_coeffs":
        if prelude is None:
            raise NotImplementedError(
                "condensed_coeffs is only lowerable at the top level / step body, not inside a "
                "control-flow (if/while/range) body")
        (state_in,) = v.inputs
        ex, ey, axy, ayx = ("ceps_x%d" % v.id, "ceps_y%d" % v.id, "ca_xy%d" % v.id, "ca_yx%d" % v.id)
        for sp in (ex, ey, axy, ayx):
            prelude.append(
                "auto %s = std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field(1, 1));" % sp)
        var[v.id] = (ex, ey, axy, ayx)  # the bundle token: the four coefficient shared_ptr names
        lines += _emit_condensed_coeffs_kernel(
            v.id, model, v.attrs["linear_operator"], v.attrs["subset"], v.attrs["c"],
            v.attrs["th_dt"], v.attrs["c_rho"], "(*%s)" % ex, "(*%s)" % ey, "(*%s)" % axy,
            "(*%s)" % ayx, var[state_in.id])
        # Coefficient halos: the apply_laplacian face means read eps at i+-1, so the four fields need
        # their ghosts filled after assembly, exactly as the brick's assemble_schur_coeffs did. The ctx
        # fill_boundary seam (the transport BC) is bit-identical to the brick's coefficient BC on
        # periodic and zero-gradient (Foextrap) sides -- the whole sanctioned condensed envelope. A
        # Dirichlet transport side would differ (the brick forces Foextrap on the coefficients); lifting
        # that needs a ctx coefficient-BC seam, batched with the brick retirement (header change).
        for sp in (ex, ey, axy, ayx):
            lines.append("ctx.fill_boundary(*%s);" % sp)
    elif v.op == "condensed_rhs":
        out_in, phi_in, state_in = v.inputs
        lines += _emit_condensed_rhs_kernel(
            v.id, model, v.attrs["linear_operator"], v.attrs["subset"], v.attrs["th_dt"],
            v.attrs["g"], var[out_in.id], var[phi_in.id], var[state_in.id])
        var[v.id] = var[out_in.id]
    else:  # condensed_reconstruct
        state_in, phi_in = v.inputs
        lines += _emit_condensed_reconstruct_kernel(
            v.id, model, v.attrs["linear_operator"], v.attrs["subset"], v.attrs["th_dt"],
            v.attrs["c_rho"], var[state_in.id], var[phi_in.id])
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


def _emit_block_M(body: Any, impl: Any, jblock: Any, th_dt_cpp: Any, indent: Any) -> Any:
    """Emit ``M = I - th_dt*J`` from the subset block @p jblock (n x n Expr) into the local ``M_[n][n]``,
    each line prefixed with @p indent. Binds the aux / param locals the J entries reference FIRST (via
    _cell_locals, cons/prim-free) and the th_dt_ scalar. Returns n. Shared by the coefficient path (which
    then inverts M with block_inverse) and the vector-apply paths (which call block_apply_inverse on M_).
    """
    n = len(jblock)
    flat = [e for row in jblock for e in row]
    # aux / param locals the J entries read (cons/prim-free by the linear_source invariant): bound once.
    # _cell_locals reads state only for cons/prim (both False here), so the state_var arg is unused.
    for ln in _cell_locals(impl, flat, "STATE_UNUSED", with_cons=False, with_prim=False):
        body.append(indent + ln)
    body.append("%sconst pops::Real th_dt_ = %s;" % (indent, th_dt_cpp))
    body.append("%spops::Real M_[%d][%d];" % (indent, n, n))
    for r in range(n):
        for c in range(n):
            ident = "pops::Real(1)" if r == c else "pops::Real(0)"
            body.append("%sM_[%d][%d] = %s - th_dt_ * (%s);"
                        % (indent, r, c, ident, jblock[r][c].to_cpp()))
    return n


def _emit_block_inverse(body: Any, impl: Any, jblock: Any, th_dt_cpp: Any, indent: Any) -> Any:
    """Emit ``M = I - th_dt*J`` (via _emit_block_M) and its four ENTRIES ``Mi_ = M^{-1}`` via
    ``pops::detail::block_inverse<n>``, into @p body. Returns n so the caller reads Mi_[r][c]. This is the
    COEFFICIENT primitive: the tensor ``A = I + c*rho*M^{-1}`` reads the entries directly, and each
    block_inverse<2> entry is a DIRECT division (bit-identical to LorentzEliminator's binv_11..22). The
    VECTOR applies (flux, reconstruct) do NOT use this -- they call block_apply_inverse on M_ so the
    single reciprocal is factored out of the bracket (apply_Binv order, bit-exact); see _emit_apply_minv.
    block_inverse computes the inverse ONCE per cell; the caller reuses Mi_ for all four entries."""
    n = _emit_block_M(body, impl, jblock, th_dt_cpp, indent)
    body.append("%spops::Real Mi_[%d][%d];" % (indent, n, n))
    # block_inverse returns false on a singular M; we do not branch in the device kernel (no throw on
    # device). M = I - th_dt*J is invertible for a well-posed eliminable source (Lorentz: det = 1 + w^2
    # > 0); a singular authored block yields a non-finite result surfacing downstream, not a wrong one.
    body.append("%spops::detail::block_inverse<%d>(M_, Mi_);" % (indent, n))
    return n


def _emit_apply_minv(body: Any, n: Any, vx_cpp: Any, vy_cpp: Any, out_x: Any, out_y: Any,
                     indent: Any) -> None:
    """Emit ``(out_x, out_y) = M^{-1} . (vx, vy)`` in the FACTORED order via
    ``pops::detail::block_apply_inverse<n>`` on the local ``M_`` (emitted by _emit_block_M): one
    reciprocal ``1/det`` factored out of the adjugate-vector bracket. For the Lorentz block this is
    ``LorentzEliminator::apply_Binv`` bit-for-bit -- the flux / reconstruct parity the retirement gate
    rests on. Summing the pre-divided block_inverse entries would round differently (a per-step ULP
    drift). The input vector and the outputs are named C++ scalars; the block-inverse local M_ is reused
    for every apply in the same cell (the brick's fusion)."""
    body.append("%spops::Real cond_v_[%d] = {%s, %s};" % (indent, n, vx_cpp, vy_cpp))
    body.append("%spops::Real cond_mv_[%d];" % (indent, n))
    body.append("%spops::detail::block_apply_inverse<%d>(M_, cond_v_, cond_mv_);" % (indent, n))
    body.append("%sconst pops::Real %s = cond_mv_[0];" % (indent, out_x))
    body.append("%sconst pops::Real %s = cond_mv_[1];" % (indent, out_y))


def _emit_condensed_coeffs_kernel(uid: Any, model: Any, jblock_op: Any, subset: Any, c_coeff: Any,
                                  th_dt: Any, c_rho: Any, ex: Any, ey: Any, axy: Any,
                                  ayx: Any, state_var: Any) -> list:
    """Emit the coefficient assembly ``A = I + c*rho*M^{-1}`` into the four 1-component field TOKENS
    @p ex / @p ey / @p axy / @p ayx (eps_x, eps_y, a_xy, a_yx of the 2x2 elliptic tensor). ONE fused
    ``for_each_cell`` over the coupled 2D momentum subset: block_inverse<2> of ``M = I - th_dt*J`` once,
    then ``cr = c*rho`` (rho from the state, R2) times the four Mi entries. Mirrors, and is bit-identical
    to, SchurOperatorCoeffKernelC for the Lorentz J. @p subset must be a 2-component (2D) momentum block
    (the elliptic coefficient tensor apply_laplacian consumes exactly eps_x/eps_y/a_xy/a_yx)."""
    impl = _model_impl(model)
    if len(subset) != 2:
        raise ValueError(
            "condensed_coeffs: the subset is the spatial velocity block and the native core is "
            "dimension=2 (the eps_x/eps_y/a_xy/a_yx tensor), so it has exactly 2 components; got "
            "%d. Authoring validates this upstream (_condensed_subset); reaching here is an IR "
            "bypass." % len(subset))
    jblock = _subset_block_rows(impl, jblock_op, subset)
    c_cpp = _coeff_cpp(c_coeff)
    th_dt_cpp = _coeff_cpp(th_dt)
    aux = "cond%s_aux" % uid
    body = [
        "pops::MultiFab& %s = ctx.aux();" % aux,
        "for (int li = 0; li < %s.local_size(); ++li) {" % ex,
        "  const pops::Array4 exA = %s.fab(li).array();" % ex,
        "  const pops::Array4 eyA = %s.fab(li).array();" % ey,
        "  const pops::Array4 axyA = %s.fab(li).array();" % axy,
        "  const pops::Array4 ayxA = %s.fab(li).array();" % ayx,
        "  const pops::ConstArray4 stateA = %s.fab(li).const_array();" % state_var,
        "  const pops::ConstArray4 auxA = %s.fab(li).const_array();" % aux,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % ex,
        "    const pops::Real rho = stateA(i, j, %d);" % int(c_rho),
    ]
    _emit_block_inverse(body, impl, jblock, th_dt_cpp, "    ")
    body.append("    const pops::Real cr = (%s) * rho;  // c*rho: the outer factor (R2: rho not in M)"
                % c_cpp)
    body.append("    exA(i, j, 0) = pops::Real(1) + cr * Mi_[0][0];")
    body.append("    eyA(i, j, 0) = pops::Real(1) + cr * Mi_[1][1];")
    body.append("    axyA(i, j, 0) = cr * Mi_[0][1];")
    body.append("    ayxA(i, j, 0) = cr * Mi_[1][0];")
    body += ["  });", "}"]
    return body


def _emit_condensed_flux_kernel(body: Any, uid: Any, impl: Any, jblock: Any, th_dt_cpp: Any,
                                subset: Any, fx_var: Any, state_var: Any) -> None:
    """Emit the explicit flux ``F = M^{-1}(mx, my)`` (Fx in comp 0, Fy in comp 1) into the 2-component
    field @p fx_var, over the valid cells + 1 ghost. block_inverse<2> once per cell, then the matrix-
    vector apply of Mi to the momentum subset. Appended to @p body (the RHS kernel shares this loop)."""
    aux = "cond%s_flux_aux" % uid
    body += [
        "pops::MultiFab& %s = ctx.aux();" % aux,
        "for (int li = 0; li < %s.local_size(); ++li) {" % fx_var,
        "  const pops::Array4 fA = %s.fab(li).array();" % fx_var,
        "  const pops::ConstArray4 stateA = %s.fab(li).const_array();" % state_var,
        "  const pops::ConstArray4 auxA = %s.fab(li).const_array();" % aux,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % fx_var,
        "    const pops::Real mx_ = stateA(i, j, %d);" % int(subset[0]),
        "    const pops::Real my_ = stateA(i, j, %d);" % int(subset[1]),
    ]
    n = _emit_block_M(body, impl, jblock, th_dt_cpp, "    ")
    # F = M^{-1} (mx, my): the FACTORED matrix-vector apply (block_apply_inverse) -- one reciprocal out of
    # the bracket, bit-for-bit the brick's LorentzEliminator::apply_Binv (not the pre-divided entries).
    _emit_apply_minv(body, n, "mx_", "my_", "cond_fx_", "cond_fy_", "    ")
    body.append("    fA(i, j, 0) = cond_fx_;")
    body.append("    fA(i, j, 1) = cond_fy_;")
    body += ["  });", "}"]


def _emit_condensed_rhs_kernel(uid: Any, model: Any, jblock_op: Any, subset: Any, th_dt: Any,
                               g_coeff: Any, rhs_var: Any, phi_n_var: Any, state_var: Any) -> list:
    """Emit the fused RHS ``rhs = -Lap(phi_n) - g*div(F)``, F = M^{-1}(mx, my), into the 1-component
    field @p rhs_var. Sequence mirrors the native assemble_schur_rhs: fill phi_n ghosts, bare Laplacian
    (ctx.laplacian) negated, the explicit flux F (an inline block_inverse kernel), then the centered FV
    divergence fused with -Lap. @p rhs_var / @p phi_n_var / @p state_var are the C++ MultiFab tokens.
    The Lap / flux buffers are allocated on rhs's layout (transient, like the native assembler)."""
    impl = _model_impl(model)
    jblock = _subset_block_rows(impl, jblock_op, subset)
    th_dt_cpp = _coeff_cpp(th_dt)
    g_cpp = _coeff_cpp(g_coeff)
    rhs = _deref(rhs_var)
    lap = "cond%s_lap" % uid
    negl = "cond%s_neglap" % uid
    fx = "cond%s_flux" % uid
    body = [
        "ctx.fill_boundary(%s);" % _deref(phi_n_var),
        "pops::MultiFab %s = ctx.alloc_scalar_field(1, 0);" % lap,
        "ctx.laplacian(%s, %s);" % (lap, _deref(phi_n_var)),
        "pops::MultiFab %s = ctx.alloc_scalar_field(1, 0);" % negl,
        # -Lap phi^n: negate the bare Laplacian (one inline kernel, no coupling/schur NegateKernel).
        "for (int li = 0; li < %s.local_size(); ++li) {" % negl,
        "  const pops::Array4 nlA = %s.fab(li).array();" % negl,
        "  const pops::ConstArray4 lapA = %s.fab(li).const_array();" % lap,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % negl,
        "    nlA(i, j, 0) = -lapA(i, j, 0);",
        "  });",
        "}",
        "pops::MultiFab %s = ctx.alloc_scalar_field(2, 1);  // F = M^-1 (mx, my), 1 ghost for div" % fx,
    ]
    _emit_condensed_flux_kernel(body, uid, impl, jblock, th_dt_cpp, subset, fx, state_var)
    body.append("ctx.fill_boundary(%s);" % fx)
    # rhs = -Lap phi^n - g*div(F): centered FV divergence (Fx comp 0, Fy comp 1), fused with -Lap.
    body += [
        "const pops::Real cond%s_hx = pops::Real(1) / (pops::Real(2) * ctx.geom().dx());"
        % uid,
        "const pops::Real cond%s_hy = pops::Real(1) / (pops::Real(2) * ctx.geom().dy());"
        % uid,
        "const pops::Real cond%s_g = %s;" % (uid, g_cpp),
        "for (int li = 0; li < %s.local_size(); ++li) {" % rhs,
        "  const pops::Array4 rhsA = %s.fab(li).array();" % rhs,
        "  const pops::ConstArray4 nlA = %s.fab(li).const_array();" % negl,
        "  const pops::ConstArray4 fA = %s.fab(li).const_array();" % fx,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % rhs,
        "    const pops::Real divF = (fA(i + 1, j, 0) - fA(i - 1, j, 0)) * cond%s_hx + "
        "(fA(i, j + 1, 1) - fA(i, j - 1, 1)) * cond%s_hy;" % (uid, uid),
        "    rhsA(i, j, 0) = nlA(i, j, 0) - cond%s_g * divF;" % uid,
        "  });",
        "}",
    ]
    return body


def _emit_condensed_reconstruct_kernel(uid: Any, model: Any, jblock_op: Any, subset: Any,
                                       th_dt: Any, c_rho: Any, state_var: Any, phi_var: Any) -> list:
    """Emit the velocity reconstruction ``v^{n+theta} = M^{-1}(v^n - th_dt*grad phi)`` then
    ``mom = rho*v`` IN PLACE on @p state_var (mx/my overwritten, rho frozen), into ONE fused
    ``for_each_cell``. Fills phi ghosts, block_inverse<2> of M once (binding th_dt_ + the aux J locals),
    the centered gradient, the residual ``v^n - th_dt*grad phi``, the matrix-vector apply of Mi, and
    mom = rho*v. Mirrors the native SchurReconstructKernelC (the gradient coefficient is th_dt =
    theta*dt; the coupling alpha lives in the coefficient / RHS, not here)."""
    impl = _model_impl(model)
    jblock = _subset_block_rows(impl, jblock_op, subset)
    th_dt_cpp = _coeff_cpp(th_dt)
    state = state_var
    phi = _deref(phi_var)
    aux = "cond%s_aux" % uid
    body = [
        "ctx.fill_boundary(%s);" % phi,
        "pops::MultiFab& %s = ctx.aux();" % aux,
        "const pops::Real cond%s_hx = pops::Real(1) / (pops::Real(2) * ctx.geom().dx());"
        % uid,
        "const pops::Real cond%s_hy = pops::Real(1) / (pops::Real(2) * ctx.geom().dy());"
        % uid,
        "for (int li = 0; li < %s.local_size(); ++li) {" % state,
        "  const pops::Array4 stateA = %s.fab(li).array();" % state,
        "  const pops::ConstArray4 phiA = %s.fab(li).const_array();" % phi,
        "  const pops::ConstArray4 auxA = %s.fab(li).const_array();" % aux,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD(int i, int j) {" % state,
        "    const pops::Real rho = stateA(i, j, %d);" % int(c_rho),
        "    const pops::Real inv_rho = rho != pops::Real(0) ? pops::Real(1) / rho : pops::Real(0);",
        "    const pops::Real vx_ = stateA(i, j, %d) * inv_rho;  // v^n = (mx, my)/rho"
        % int(subset[0]),
        "    const pops::Real vy_ = stateA(i, j, %d) * inv_rho;" % int(subset[1]),
    ]
    # M = I - th_dt*J FIRST: _emit_block_M binds th_dt_ and the aux J locals the residual reads below.
    n = _emit_block_M(body, impl, jblock, th_dt_cpp, "    ")
    body += [
        "    const pops::Real gx_ = (phiA(i + 1, j, 0) - phiA(i - 1, j, 0)) * cond%s_hx;" % uid,
        "    const pops::Real gy_ = (phiA(i, j + 1, 0) - phiA(i, j - 1, 0)) * cond%s_hy;" % uid,
        "    const pops::Real rx_ = vx_ - th_dt_ * gx_;  // (v^n - theta dt grad phi)_x",
        "    const pops::Real ry_ = vy_ - th_dt_ * gy_;",
    ]
    # v^{n+theta} = M^{-1}(v^n - theta dt grad phi): the FACTORED apply (block_apply_inverse), bit-for-bit
    # the brick's apply_Binv(rx_, ry_) -- one reciprocal out of the bracket, not the pre-divided entries.
    _emit_apply_minv(body, n, "rx_", "ry_", "nx_", "ny_", "    ")
    body += [
        "    stateA(i, j, %d) = rho * nx_;  // mom = rho v^{n+theta}" % int(subset[0]),
        "    stateA(i, j, %d) = rho * ny_;" % int(subset[1]),
        "  });",
        "}",
    ]
    return body
