"""pops.codegen.module_emit_riemann : Riemann-capability blocks of emit_cpp_brick.

Extracted verbatim from ``pops.codegen.module_codegen.emit_cpp_brick`` so the brick
emitter fits the Spec-4 file-size budget.  Each helper builds the C++ lines for one
OPTIONAL Riemann capability and returns them as a list; ``emit_cpp_brick`` extends its
accumulator with the result.  Every directional hook consumes the same compile-time Axis and
exact x[/y[/z]] rank as the physical-flux emitter.

Helpers
-------
_emit_hllc           -- m.enable_hllc : contact_speed + hllc_star_state from roles
_emit_roe_roles      -- m.enable_roe : roe_dissipation |A_roe| dU from roles
_emit_roe_provided   -- m.roe_dissipation : user rows via left()/right()
_emit_roe_jacobian   -- m.roe_from_jacobian : d = |A| (UR-UL), A = dF/dU at Uavg
"""
from __future__ import annotations

from typing import Any

from pops._dense_spectral import is_exact_block_triangular
from pops.codegen.cpp_writer import _cpp_roe
from pops.codegen.module_emit_helpers import (
    _codegen_exprs,
    _live_prims,
    _prim_block,
    _ranked_axes,
    _roles_for,
)
from pops.identity.scalar import scalar_cpp


def has_characteristic_no_inflow_provider(model: Any) -> bool:
    """Whether the generated block can evaluate its characteristic Jacobian locally.

    Boundary kernels receive the conservative cell state and model value parameters, but no
    auxiliary field pack.  Refuse a Jacobian that transitively reads an auxiliary field instead of
    emitting a hook with an undeclared dependency or silently freezing that field.
    """
    jacobian = getattr(model, "_roe_jacobian", None)
    requirements = getattr(model, "_aux_requirements", None)
    if jacobian is None or not callable(requirements):
        return False
    axes = _ranked_axes(model)
    if any(axis not in jacobian for axis in axes):
        return False
    expressions = [
        expression
        for direction in axes
        for row in jacobian[direction]
        for expression in row
    ]
    return not bool(requirements(expressions).get("aux"))


def _certified_roe_blocks(model: Any, jacobians: Any) -> Any:
    """Return exact block-triangular certificates reusable by dense Roe, or ``None``.

    A wave-speed partition is only reused after proving the *Roe* Jacobian itself is structurally
    block triangular in both directions.  Sampling, user assertion, finite-difference Jacobians and
    incomplete partitions never qualify.
    """

    waves = getattr(model, "_ws_jacobian", None)
    if not waves or waves.get("eig") != "numeric":
        return None
    axes = _ranked_axes(model)
    blocks = waves.get("blocks")
    if not isinstance(blocks, dict) or tuple(blocks) != axes:
        return None
    if not all(
        is_exact_block_triangular(jacobians[direction], blocks[direction])
        for direction in axes
    ):
        return None
    return blocks


def _emit_real_spectrum_blocks(blocks: Any, *, indent: str, im_tol: str, max_iter: int) -> list:
    lines = []
    for block_index, block in enumerate(blocks):
        size = len(block)
        name = "roe_block_%d_" % block_index
        lines.append("%s{" % indent)
        lines.append("%s  pops::Real %s[%d][%d];" % (indent, name, size, size))
        for row, global_row in enumerate(block):
            for column, global_column in enumerate(block):
                lines.append(
                    "%s  %s[%d][%d] = A[%d][%d];"
                    % (indent, name, row, column, global_row, global_column)
                )
        lines.append(
            "%s  if (pops::real_spectrum(%s, static_cast<pops::Real>(%s), %d) "
            "!= pops::Spectrum::kReal) spectrum_real_ = false;"
            % (indent, name, im_tol, max_iter)
        )
        lines.append("%s}" % indent)
    return lines


def _unique_role_index(roles: list[str], role: str, *, capability: str) -> int:
    matches = [index for index, value in enumerate(roles) if value == role]
    if len(matches) != 1:
        raise ValueError(
            "%s: exactly one %s role is required; current roles %r"
            % (capability, role, roles)
        )
    return matches[0]


def _ranked_fluid_roles(model: Any, *, capability: str) -> tuple:
    """Resolve density/energy and one axis-qualified momentum per emitted axis."""
    axes = _ranked_axes(model)
    roles = _roles_for(model.cons_names, model.cons_roles)
    density = _unique_role_index(roles, "density", capability=capability)
    momenta = [
        _unique_role_index(roles, "momentum:%d" % axis, capability=capability)
        for axis in range(len(axes))
    ]
    energy_matches = [index for index, role in enumerate(roles) if role == "energy"]
    if len(energy_matches) > 1:
        raise ValueError(
            "%s: at most one energy role is supported; current roles %r"
            % (capability, roles)
        )
    energy = energy_matches[0] if energy_matches else -1
    return axes, roles, density, momenta, energy


def _axis_branch(ordinal: int, *, indent: str = "    ") -> str:
    keyword = "if" if ordinal == 0 else "else if"
    return "%s%s constexpr (Axis == %d) {" % (indent, keyword, ordinal)


def _axis_guard(capability: str, *, indent: str = "    ") -> str:
    return (
        '%sstatic_assert(Axis >= 0 && Axis < dimension, "%s axis is outside the emitted rank");'
        % (indent, capability)
    )


def _emit_hllc(model: Any, nc: Any) -> list:
    """Emit exact-ranked Toro contact and star-state hooks from physical roles."""
    out = []
    if "p" not in model.prim_defs:
        raise ValueError("enable_hllc: the primitive 'p' (pressure) must be declared "
                         "(m.primitive('p', ...)) -- contact_speed/star state depend on it")
    axes, _, iD, momenta, iE = _ranked_fluid_roles(model, capability="enable_hllc")
    out.append("  // CAPABILITY HLLC generee depuis les ROLES (enable_hllc) : algorithme")
    out.append("  // contact-resolving exact-ranked du coeur (HasHLLCStructure), aucun layout fige.")
    out.append("  template <int Axis>")
    out.append("  POPS_HD pops::Real contact_speed(const State& UL, const State& UR, "
               "pops::Real pL, pops::Real pR, pops::Real sL, pops::Real sR) const {")
    out.append(_axis_guard("HLLC contact"))
    for ordinal, momentum in enumerate(momenta):
        out.append(_axis_branch(ordinal))
        out.append("      constexpr int in_ = %d;" % momentum)
        out.append("      const pops::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
        out.append("      const pops::Real unL = UL[in_] / rL, unR = UR[in_] / rR;")
        out.append("      return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /")
        out.append("             (rL * (sL - unL) - rR * (sR - unR));")
        out.append("    }")
    out += ["  }", ""]
    out.append("  template <int Axis>")
    out.append("  POPS_HD State hllc_star_state(const State& U, pops::Real p, pops::Real s, "
               "pops::Real sStar) const {")
    out.append(_axis_guard("HLLC star-state"))
    for ordinal, momentum in enumerate(momenta):
        out.append(_axis_branch(ordinal))
        out.append("      constexpr int in_ = %d;" % momentum)
        out.append("      const pops::Real r = U[%d];" % iD)
        out.append("      const pops::Real un = U[in_] / r;")
        out.append("      const pops::Real fac = r * (s - un) / (s - sStar);")
        out.append("      State Us{};")
        out.append("      for (int c = 0; c < %d; ++c) Us[c] = fac * (U[c] / r);  "
                   "// defaut : advection passive" % nc)
        out.append("      Us[%d] = fac;" % iD)
        out.append("      Us[in_] = fac * sStar;")
        if iE >= 0:
            out.append("      Us[%d] = fac * (U[%d] / r + (sStar - un) * (sStar + p / "
                       "(r * (s - un))));" % (iE, iE))
        out += ["      return Us;", "    }"]
    out += ["  }", ""]
    return out


def _emit_roe_roles(model: Any, nc: Any) -> list:
    """Emit exact-ranked role-derived Roe dissipation for one to three axes."""
    from pops.numerics.riemann.providers import ENTROPY_HARTEN, RoeEntropyPolicy

    policy = getattr(model, "_roe_entropy_policy", None)
    if type(policy) is not RoeEntropyPolicy:
        raise ValueError("enable_roe: missing exact typed entropy policy")
    out = []
    if "p" not in model.prim_defs:
        raise ValueError("enable_roe: the primitive 'p' (pressure) must be declared "
                         "(m.primitive('p', ...)) -- the Roe linearization depends on it")
    axes, _, iD, momenta, iE = _ranked_fluid_roles(model, capability="enable_roe")
    fluid_components = {iD, *momenta}
    if iE >= 0:
        fluid_components.add(iE)
    passives = [component for component in range(nc) if component not in fluid_components]
    out.append("  // CAPABILITY ROE generee depuis les ROLES (enable_roe) : dissipation")
    out.append("  // |A_roe| dU exact-ranked (HasRoeDissipation), tangentielles iterees au build.")
    out.append("  template <int Axis>")
    out.append("  POPS_HD State roe_dissipation(const State& UL, const auto&, "
               "const State& UR, const auto&) const {")
    out.append(_axis_guard("Roe dissipation"))
    for normal_axis, normal_component in enumerate(momenta):
        tangential_axes = [axis for axis in range(len(axes)) if axis != normal_axis]
        out.append(_axis_branch(normal_axis))
        out.append("      constexpr int in_ = %d;" % normal_component)
        out.append("      const pops::Real rL = UL[%d], rR = UR[%d];" % (iD, iD))
        for axis, component in enumerate(momenta):
            out.append(
                "      const pops::Real u%dL = UL[%d] / rL, u%dR = UR[%d] / rR;"
                % (axis, component, axis, component)
            )
        out.append("      const pops::Real pL = pressure(UL), pR = pressure(UR);")
        out.append("      const pops::Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), "
                   "den = sqL + sqR;")
        for axis in range(len(axes)):
            out.append(
                "      const pops::Real u%d = (sqL * u%dL + sqR * u%dR) / den;"
                % (axis, axis, axis)
            )
        out.append("      const pops::Real rho = sqL * sqR;")
        q2 = " + ".join("u%d * u%d" % (axis, axis) for axis in range(len(axes)))
        q2_left = " + ".join("u%dL * u%dL" % (axis, axis) for axis in range(len(axes)))
        if iE >= 0:
            out.append("      // gaz parfait : H de Roe + gamma-1 deduit sur le rang exact")
            out.append("      const pops::Real HL = (UL[%d] + pL) / rL, HR = "
                       "(UR[%d] + pR) / rR;" % (iE, iE))
            out.append("      const pops::Real H = (sqL * HL + sqR * HR) / den;")
            out.append("      const pops::Real q2 = %s;" % q2)
            out.append("      const pops::Real gm1 = pL / (UL[%d] - pops::Real(0.5) * rL * "
                       "(%s));" % (iE, q2_left))
            out.append("      const pops::Real c2 = gm1 * (H - pops::Real(0.5) * q2);")
            out.append("      const pops::Real c = std::sqrt(c2);")
        else:
            out.append("      // sans Energy : c LOCAL = sqrt(p/rho) par cote, moyenne de Roe")
            out.append("      const pops::Real c = (sqL * std::sqrt(pL / rL) + sqR * "
                       "std::sqrt(pR / rR)) / den;")
            out.append("      const pops::Real c2 = c * c;")
        out.append(
            "      const pops::Real dr = rR - rL, dp = pR - pL, dun = u%dR - u%dL;"
            % (normal_axis, normal_axis)
        )
        out.append("      const pops::Real a1 = (dp - rho * c * dun) / "
                   "(pops::Real(2) * c2);")
        out.append("      const pops::Real a2 = dr - dp / c2;")
        for axis in tangential_axes:
            out.append(
                "      const pops::Real at%d = rho * (u%dR - u%dL);"
                % (axis, axis, axis)
            )
        out.append("      const pops::Real a5 = (dp + rho * c * dun) / "
                   "(pops::Real(2) * c2);")
        out.append("      // Politique d'entropie explicite du provider Roe (%s)." % policy.kind)
        if policy.kind == ENTROPY_HARTEN:
            out.append("      const pops::HartenEntropyFix entropy_fix{%s};"
                       % scalar_cpp(policy.delta))
        out.append("      const pops::Real l1r = u%d - c, l5r = u%d + c;"
                   % (normal_axis, normal_axis))
        if policy.kind == ENTROPY_HARTEN:
            out.append("      const pops::Real al1 = entropy_fix(l1r, c);")
        else:
            out.append("      const pops::Real al1 = l1r < 0 ? -l1r : l1r;")
        out.append("      const pops::Real al2 = u%d < 0 ? -u%d : u%d;"
                   % (normal_axis, normal_axis, normal_axis))
        if policy.kind == ENTROPY_HARTEN:
            out.append("      const pops::Real al5 = entropy_fix(l5r, c);")
        else:
            out.append("      const pops::Real al5 = l5r < 0 ? -l5r : l5r;")
        out.append("      State d{};")
        out.append("      d[%d] = al1 * a1 + al2 * a2 + al5 * a5;" % iD)
        out.append(
            "      d[in_] = al1 * a1 * (u%d - c) + al2 * a2 * u%d + "
            "al5 * a5 * (u%d + c);" % (normal_axis, normal_axis, normal_axis)
        )
        for axis in tangential_axes:
            out.append(
                "      d[%d] = al1 * a1 * u%d + al2 * (a2 * u%d + at%d) + "
                "al5 * a5 * u%d;"
                % (momenta[axis], axis, axis, axis, axis)
            )
        if iE >= 0:
            shear_energy = "".join(" + at%d * u%d" % (axis, axis) for axis in tangential_axes)
            out.append(
                "      d[%d] = al1 * a1 * (H - u%d * c) + al2 * "
                "(a2 * pops::Real(0.5) * q2%s) + al5 * a5 * (H + u%d * c);"
                % (iE, normal_axis, shear_energy, normal_axis)
            )
        for passive in passives:
            out.append("      {  // scalaire passif [%d] : onde entropique (phi = q/rho)"
                       % passive)
            out.append("        const pops::Real fL = UL[%d] / rL, fR = UR[%d] / rR;"
                       % (passive, passive))
            out.append("        const pops::Real ft = (sqL * fL + sqR * fR) / den;")
            out.append("        d[%d] = al1 * a1 * ft + al2 * "
                       "(a2 * ft + rho * (fR - fL)) + al5 * a5 * ft;" % passive)
            out.append("      }")
        out += ["      return d;", "    }"]
    out += ["  }", ""]
    return out


def _emit_roe_provided(model: Any, nc: Any) -> list:
    """CAPABILITY ROE PROVIDED (m.roe_dissipation): 'user' counterpart of enable_roe. The d_i
    rows come from the user (their eigenstructure), written with left()/right() of both states.
    We emit the SAME hook roe_dissipation<Axis>(UL, AL, UR, AR) as the roles path (trait
    HasRoeDissipation, the core does F = 1/2(FL+FR) - 1/2 d). left(e) -> e on the L_ locals
    (computed from UL), right(e) -> R_ locals (from UR). _roe_rows and _roe are exclusive
    (guard at declaration and in check())."""
    out = []
    axes = _ranked_axes(model)
    if tuple(model._roe_rows) != axes:
        raise ValueError(
            "provided Roe rows must cover the exact emitted axis set %s" % (axes,)
        )
    from pops._ir.visitors import _dependencies

    aux_dependencies = _dependencies(
        expression
        for axis in axes
        for expression in model._roe_rows[axis]
    )
    provider_components = tuple(
        name for name in model._provider_components if name in aux_dependencies
    )
    has_aux = bool(provider_components)  # bound only when this consumer actually reads them
    aL = "const auto& aL" if has_aux else "const auto&"
    aR = "const auto& aR" if has_aux else "const auto&"
    out.append("  // CAPABILITY ROE FOURNIE (m.roe_dissipation) : dissipation d ecrite par")
    out.append("  // l'utilisateur via left()/right() des deux etats ; hook HasRoeDissipation.")
    out.append("  template <int Axis>")
    out.append("  POPS_HD State roe_dissipation(const State& UL, %s, const State& UR, %s) "
               "const {" % (aL, aR))
    out.append(_axis_guard("provided Roe dissipation"))
    # locals of BOTH states: conservatives, primitives (def with prefix), then aux read.
    for side, U, av in (("L_", "UL", "aL"), ("R_", "UR", "aR")):
        out += ["    const pops::Real %s%s = %s[%d];" % (side, c, U, i)
                for i, c in enumerate(model.cons_names)]
        out += ["    const pops::Real %s%s = %s;" % (side, p, _cpp_roe(e, side))
                for p, e in model.prim_defs.items()]
        if has_aux:
            out += ["    const pops::Real %s%s = pops::provider_value<%d>(%s);"
                    % (side, n, model._physical_flux_consumer_slot(n), av)
                    for n in provider_components]
    out.append("    State d{};")
    for ordinal, axis in enumerate(axes):
        out.append(_axis_branch(ordinal))
        out += [
            "      d[%d] = %s;" % (i, _cpp_roe(expression, None))
            for i, expression in enumerate(model._roe_rows[axis])
        ]
        out.append("    }")
    out += ["    return d;", "  }", ""]
    return out


def _emit_roe_jacobian(model: Any, nc: Any, cse: Any) -> list:
    """CAPABILITY ROE FROM THE FLUX JACOBIAN (m.roe_from_jacobian): generic moment Roe. The hook
    builds A = dF_dir/dU at Uavg = 1/2(UL+UR) (cons locals bound to the mean, like the
    wave_speeds-from-jacobian path binds them to U), then d = |A| (UR-UL) via pops::roe_abs_apply
    (matrix-sign |A| = A sign(A); for a real-diagonalizable A this is R|Lambda|R^-1 exactly).
    When ``entropy_fix`` is configured, pops::roe_entropy_fix_apply applies the generic Harten
    function Phi_delta(A), including zero eigenvalues, and returns false for complex/non-converged
    spectra; the emitted NaN is then rejected by the native residual contract.  The unconfigured
    route uses the true matrix absolute value with a zero-mode projector for singular real
    Jacobians. Neither route substitutes another Riemann solver. Roles-free (no 'p', no
    Density/Momentum): the generic provider for a moment hierarchy. The core
    (HasRoeDissipation) does F = 1/2(FL+FR) - 1/2 d."""
    out = []
    axes = _ranked_axes(model)
    carried_axes = tuple(
        key for key in model._roe_jacobian if key != "entropy_fix"
    )
    if carried_axes != axes:
        raise ValueError(
            "Roe Jacobian must cover the exact emitted axis set %s" % (axes,)
        )
    jacobians = {axis: model._roe_jacobian[axis] for axis in axes}
    entropy_fix = model._roe_jacobian.get("entropy_fix")
    live = _live_prims(
        model,
        [expression for axis in axes for row in jacobians[axis] for expression in row],
    )
    out.append("  // CAPABILITY ROE depuis la JACOBIENNE (roe_from_jacobian) : d = |A| (UR-UL),")
    out.append("  // A = dF/dU a l'etat moyen Uavg = 1/2(UL+UR) ; fonction spectrale native")
    if entropy_fix is None:
        out.append("  // (matrix-sign + projecteur du noyau) ; complexe/non converge refuse.")
    else:
        out.append("  // Phi_delta(A), delta=%s ; spectre complexe/non converge refuse."
                   % scalar_cpp(entropy_fix))
    out.append("  template <int Axis>")
    out.append("  POPS_HD State roe_dissipation(const State& UL, const auto&, "
               "const State& UR, const auto&) const {")
    out.append(_axis_guard("Jacobian Roe dissipation"))
    # conservatives at the ARITHMETIC-MEAN interface state Uavg = 1/2 (UL + UR)
    out += ["    const pops::Real %s = pops::Real(0.5) * (UL[%d] + UR[%d]);" % (c, i, i)
            for i, c in enumerate(model.cons_names)]
    out += _prim_block(model, live)  # live primitives, evaluated at Uavg
    out.append("    pops::Real A[%d][%d];" % (nc, nc))
    for ordinal, axis in enumerate(axes):
        out.append(_axis_branch(ordinal))
        temporaries, expressions = _codegen_exprs(
            model,
            [jacobians[axis][i][j] for i in range(nc) for j in range(nc)],
            cse,
            indent="      ",
        )
        out += temporaries
        for i in range(nc):
            out += [
                "      A[%d][%d] = %s;" % (i, j, expressions[i * nc + j])
                for j in range(nc)
            ]
        out.append("    }")
    out.append("    pops::Real dU[%d], out[%d];" % (nc, nc))
    out += ["    dU[%d] = UR[%d] - UL[%d];" % (i, i, i) for i in range(nc)]
    out.append("    State d{};")
    # Eig knobs from wave_speeds_from_jacobian are also the Roe spectral-classification contract.
    # None selects the native roundoff floor; an explicit zero keeps the exact-zero predicate, and
    # an explicit positive value is an author-controlled relaxation.  A structurally proven
    # block-triangular partition certifies the full spectrum from its diagonal blocks; the matrix
    # function itself is still applied to the complete Jacobian.
    ws = getattr(model, "_ws_jacobian", None) or {}
    eig_max_iter = ws.get("eig_max_iter")
    im_tol = ws.get("im_tol")
    eig_max_iter_value = int(eig_max_iter) if eig_max_iter is not None else 100
    im_tol_cpp = scalar_cpp(im_tol) if im_tol is not None else "pops::kEigStrictImagTol"
    certified_blocks = _certified_roe_blocks(model, jacobians)
    if certified_blocks is not None:
        out.append("    bool spectrum_real_ = true;")
        for ordinal, axis in enumerate(axes):
            out.append(_axis_branch(ordinal))
            out += _emit_real_spectrum_blocks(
                certified_blocks[axis],
                indent="      ",
                im_tol=im_tol_cpp,
                max_iter=eig_max_iter_value,
            )
            out.append("    }")
    if eig_max_iter is not None or im_tol is not None:
        roe_args = ", 80, static_cast<pops::Real>(1e-13), static_cast<pops::Real>(%s), %d" % (
            im_tol_cpp,
            eig_max_iter_value)
    else:
        roe_args = ""
    if certified_blocks is not None and entropy_fix is None:
        apply_call = (
            "spectrum_real_ && pops::detail::roe_abs_apply_certified_real(A, dU, out)"
        )
    elif certified_blocks is not None:
        apply_call = (
            "spectrum_real_ && pops::detail::roe_entropy_fix_apply_certified_real("
            "A, dU, out, static_cast<pops::Real>(%s))" % scalar_cpp(entropy_fix)
        )
    elif entropy_fix is None:
        apply_call = "pops::roe_abs_apply(A, dU, out%s)" % roe_args
    else:
        apply_call = "pops::roe_entropy_fix_apply(A, dU, out, static_cast<pops::Real>(%s)%s)" % (
            scalar_cpp(entropy_fix), roe_args)
    out.append("    if (%s) {" % apply_call)
    out += ["      d[%d] = out[%d];" % (i, i) for i in range(nc)]
    out.append("    } else {  // complexe/non converge : refus par le contrat residual natif")
    out += ["      d[%d] = std::numeric_limits<pops::Real>::quiet_NaN();" % i
            for i in range(nc)]
    out.append("    }")
    out += ["    return d;", "  }", ""]
    if not has_characteristic_no_inflow_provider(model):
        return out
    out.append("  // Prepared characteristic no-inflow: the same complete model Jacobian, oriented")
    out.append("  // by the physical-face normal. Sonic modes are neutral; no model-specific fallback.")
    out.append("  static constexpr int characteristic_no_inflow_contract_version = 1;")
    out.append("  static constexpr int characteristic_no_inflow_dimension = dimension;")
    out.append("  static constexpr int characteristic_no_inflow_components = n_vars;")
    out.append("  static constexpr bool characteristic_no_inflow_conservative = true;")
    out.append("  POPS_HD bool characteristic_no_inflow(const State& interior, ")
    out.append("      const State& reference, int dir, int outward_sign, State& ghost) const {")
    out += ["    const pops::Real %s = interior[%d];" % (c, i)
            for i, c in enumerate(model.cons_names)]
    out += _prim_block(model, live)
    out.append("    pops::Real A[%d][%d];" % (nc, nc))
    for ordinal, axis in enumerate(axes):
        keyword = "if" if ordinal == 0 else "else if"
        out.append("    %s (dir == %d) {" % (keyword, ordinal))
        temporaries, expressions = _codegen_exprs(
            model,
            [jacobians[axis][i][j] for i in range(nc) for j in range(nc)],
            cse,
            indent="      ",
        )
        out += temporaries
        for i in range(nc):
            out += [
                "      A[%d][%d] = %s;" % (i, j, expressions[i * nc + j])
                for j in range(nc)
            ]
        out.append("    }")
    out.append("    else {")
    out.append("      return false;")
    out.append("    }")
    out.append("    pops::Real jump[%d], incoming[%d];" % (nc, nc))
    out += ["    jump[%d] = interior[%d] - reference[%d];" % (i, i, i)
            for i in range(nc)]
    out.append(
        "    if (!pops::characteristic_incoming_apply(A, jump, incoming, outward_sign, "
        "80, static_cast<pops::Real>(1e-13), static_cast<pops::Real>(%s), %d))"
        % (im_tol_cpp, eig_max_iter_value)
    )
    out.append("      return false;")
    for i in range(nc):
        out.append("    ghost[%d] = interior[%d] - pops::Real(2) * incoming[%d];" % (i, i, i))
        out.append("    if (!std::isfinite(ghost[%d])) return false;" % i)
    out += ["    return true;", "  }", ""]
    return out
