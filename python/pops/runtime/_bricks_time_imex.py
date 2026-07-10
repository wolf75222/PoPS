"""IMEX / source-implicit time policies + the implicit-mask helpers.

Split out of :mod:`pops.runtime._bricks_time` for the 500-line cap (ADC-550): the mask
normalization helpers ``_role_to_stable`` / ``_norm_implicit`` and the implicit-source time
policies ``IMEX`` / ``SourceImplicit`` / ``SourceImplicitBE`` / ``IMEXRK``. ``_bricks_time``
re-imports every name and ``pops.runtime.bricks`` re-exports them, so no public import path
changes. The split / Schur policies (``Split`` / ``Strang`` / ``CondensedSchur``) and ``Role``
stay in their own modules.
"""
from __future__ import annotations

from typing import Any

from pops.runtime._numeric import exact_real, positive_int, strict_bool
from pops.runtime.defaults import (
    NEWTON_DEFAULT_ABS_TOL,
    NEWTON_DEFAULT_DAMPING,
    NEWTON_DEFAULT_FAIL_POLICY,
    NEWTON_DEFAULT_FD_EPS,
    NEWTON_DEFAULT_MAX_ITERS,
    NEWTON_DEFAULT_REL_TOL,
)
from pops.runtime.routes import TIME_IMEX, TIME_IMEXRK_ARS222


def _cadence(label: str, substeps: Any, stride: Any) -> tuple[int, int]:
    return (
        positive_int(substeps, where=label + ".substeps"),
        positive_int(stride, where=label + ".stride"),
    )


def _newton_controls(
    label: str, max_iters: Any, rel_tol: Any, abs_tol: Any, fd_eps: Any,
    diagnostics: Any, damping: Any, fail_policy: Any,
) -> tuple[Any, ...]:
    if not isinstance(fail_policy, str) or fail_policy not in ("none", "warn", "throw"):
        raise ValueError(
            "%s.newton_fail_policy must be 'none'|'warn'|'throw' (got %r)"
            % (label, fail_policy))
    return (
        positive_int(max_iters, where=label + ".newton_max_iters"),
        exact_real(rel_tol, where=label + ".newton_rel_tol", minimum=0),
        exact_real(abs_tol, where=label + ".newton_abs_tol", minimum=0),
        exact_real(fd_eps, where=label + ".newton_fd_eps", minimum=0, minimum_open=True),
        strict_bool(diagnostics, where=label + ".newton_diagnostics"),
        exact_real(
            damping, where=label + ".newton_damping", minimum=0, minimum_open=True,
            maximum=1),
        fail_policy,
    )


def _role_to_stable(name: Any) -> Any:
    """Normalize a role name to the STABLE key expected by the C++ (role_from_name): lowercase
    snake_case ("momentum_x", "energy"). Tolerates the PascalCase variants of the C++ enum exposed in
    the target API (e.g. "MomentumX" -> "momentum_x", "Energy" -> "energy") by inserting a '_' before each
    internal uppercase letter before lowercasing. A name already in snake_case ("momentum_x") is unchanged."""
    s = str(name).strip()
    if not s:
        return s
    if s == s.lower():  # already snake_case / lowercase: unchanged
        return s
    out = [s[0].lower()]
    for ch in s[1:]:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _norm_implicit(label: Any, implicit_vars: Any, implicit_roles: Any) -> Any:
    """Normalize the implicit-mask lists (names / physical roles) into lists of strings.

    None -> [] (default: inactive mask, model default, bit-identical). A bare string is tolerated
    (e.g. implicit_vars="rho_u" -> ["rho_u"]). The roles are reduced to the STABLE C++ key (snake_case)
    via _role_to_stable -> "MomentumX" and "momentum_x" are equivalent. The mask lives on the TEMPORAL
    POLICY / block side (and NOT the model): the SAME model is reused with distinct implicit treatments.
    The RESOLUTION of names/roles -> indices and the validation (name/role absent from the block) lives
    on the C++ side (System::add_block), the only source of truth for the block names/roles."""
    def as_list(x: Any, what: Any) -> Any:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        try:
            out = [str(v) for v in x]
        except TypeError:
            raise ValueError("%s: %s must be a list of strings (received %r)" % (label, what, x))
        return out
    names = as_list(implicit_vars, "implicit_vars")
    roles = [_role_to_stable(r) for r in as_list(implicit_roles, "implicit_roles")]
    return names, roles


class IMEX:
    """IMEX: explicit transport (SSPRK) + stiff implicit source (backward-Euler, local Newton).

    PARTIAL treatment: only the SOURCE is implicit (backward-Euler, local cell Newton,
    via backward_euler_source / ImplicitSourceStepper on the C++ side). The TRANSPORT stays explicit
    (advanced by the SSPRK core). This is NOT a global implicit solver (flux + source + Poisson
    solved implicitly / Newton-Krylov) -- that work is a distinct future phase.

    - ``substeps=N``: substeps per macro-step (cf. Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. Explicit): the block is held
      while (macro_step + 1) % M != 0, then advances by an effective step M*dt at the end of the window. Between
      two catch-ups, its STALE state contributes to the system Poisson. Default 1 = every macro-step,
      bit-identical. Backend 'aot': stride > 1 rejected (cf. Explicit).
    - ``implicit_vars``: names of the conserved variables to treat IMPLICITLY in the source step;
      the others stay explicit (forward Euler). The mask is CARRIED BY THIS POLICY / the block,
      NOT by the model -> the SAME model is reused with different implicit treatments.
      Default [] (+ implicit_roles []) = model default (Model::is_implicit, or all implicit by
      default), BIT-IDENTICAL. Resolved on the C++ side against the block names (an absent name raises an error).
      E.g. pops.IMEX(implicit_vars=["rho_u", "rho_v"]).
    - ``implicit_roles``: same mask but by physical ROLE ("density", "momentum_x", "energy", ...)
      instead of the name (cf. System.variable_roles). Union with implicit_vars. E.g.
      pops.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"]).
    - ``newton_max_iters``: iteration budget of the local Newton (default 2 = historical constant).
    - ``newton_rel_tol`` / ``newton_abs_tol``: per-cell stopping criterion
      ||F||_inf <= abs_tol + rel_tol*||F0||_inf (0/0 = disabled, bit-identical historical loop).
    - ``newton_fd_eps``: step of the finite-difference Jacobian (default 1e-7 = historical).
    - ``newton_diagnostics``: enables the Newton report (sim.newton_report(name) -> dict
      {enabled, converged, max_residual, max_iters_used, n_failed}), aggregated over the last advance
      of the block. OPT-IN: default False = zero extra cost.

    NOMENCLATURE (audit 2026-06): the wired scheme is exactly ForwardEuler(transport without
    source) + local backward-Euler on the source ("SourceImplicitBE"). It is NOT an
    IMEX-RK / ARK family (no choice of Butcher tableau, ``method=`` of the explicit does not apply
    to the IMEX half-step); a true IMEXRK family would be a distinct future work.
    """

    kind = TIME_IMEX  # typed time route (ADC-584); str value stays the historical "imex"
    def __init__(self, substeps: int = 1, stride: int = 1, implicit_vars: Any = None, implicit_roles: Any = None,
                 newton_max_iters: Any = NEWTON_DEFAULT_MAX_ITERS,
                 newton_rel_tol: Any = NEWTON_DEFAULT_REL_TOL,
                 newton_abs_tol: Any = NEWTON_DEFAULT_ABS_TOL,
                 newton_fd_eps: Any = NEWTON_DEFAULT_FD_EPS,
                 newton_diagnostics: bool = False,
                 newton_damping: Any = NEWTON_DEFAULT_DAMPING,
                 newton_fail_policy: Any = NEWTON_DEFAULT_FAIL_POLICY) -> None:
        self.substeps, self.stride = _cadence("IMEX", substeps, stride)
        self.implicit_vars, self.implicit_roles = _norm_implicit("IMEX", implicit_vars, implicit_roles)
        (self.newton_max_iters, self.newton_rel_tol, self.newton_abs_tol,
         self.newton_fd_eps, self.newton_diagnostics, self.newton_damping,
         self.newton_fail_policy) = _newton_controls(
             "IMEX", newton_max_iters, newton_rel_tol, newton_abs_tol, newton_fd_eps,
             newton_diagnostics, newton_damping, newton_fail_policy)


class SourceImplicit:
    """Implicit treatment of the STIFF SOURCE (backward-Euler, local Newton), explicit transport.

    Clear name for the source-only IMEX scheme: only the SOURCE is treated implicitly
    (backward-Euler solved by local per-cell Newton, via backward_euler_source /
    ImplicitSourceStepper on the C++ side). TRANSPORT stays EXPLICIT (advanced by the SSPRK core).

    IMPORTANT -- this is NOT a global implicit PDE solver. A global implicit solver
    (flux + source + Poisson all implicit, Newton-Krylov or global Schur) is a distinct
    future effort. SourceImplicit = source-only IMEX (strictly equivalent to IMEX/pops.Implicit,
    bit-identical numerics).

    WHEN TO USE IT (SourceImplicit LOCAL vs pops.CondensedSchur GLOBAL) -- both mechanisms
    treat a stiff source implicitly, but at different scales:

    - SourceImplicit is LOCAL: the implicit part couples only the components of A SINGLE CELL
      (backward-Euler solved by per-cell Newton), there is NO spatial coupling between
      cells. Suited to purely local stiff terms (relaxation, reactions, friction).
    - pops.CondensedSchur (via pops.Split) is GLOBAL: it assembles and solves a tensor
      elliptic operator by Schur (Krylov BiCGStab) that COUPLES the whole domain. Suited to
      non-local stiff Lorentz / electrostatic coupling (e.g. magnetized Euler-Poisson from the
      Hoffart paper, arXiv:2510.11808). A local stiff source does NOT need Schur.

    - ``substeps=N``: substeps per macro-step (cf. Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. Explicit). Default 1.
    - ``implicit_vars`` / ``implicit_roles``: implicit mask by NAME or by physical ROLE of the
      conserved variables to treat implicitly in the source step (cf. IMEX). Mask CARRIED BY
      THIS POLICY / the block, not by the model. Defaults [] = model default, bit-identical.
    """

    kind = TIME_IMEX  # same C++ path as IMEX (ImplicitSourceStepper); typed route (ADC-584)

    def __init__(self, substeps: int = 1, stride: int = 1, implicit_vars: Any = None, implicit_roles: Any = None,
                 newton_max_iters: Any = NEWTON_DEFAULT_MAX_ITERS,
                 newton_rel_tol: Any = NEWTON_DEFAULT_REL_TOL,
                 newton_abs_tol: Any = NEWTON_DEFAULT_ABS_TOL,
                 newton_fd_eps: Any = NEWTON_DEFAULT_FD_EPS,
                 newton_diagnostics: bool = False,
                 newton_damping: Any = NEWTON_DEFAULT_DAMPING,
                 newton_fail_policy: Any = NEWTON_DEFAULT_FAIL_POLICY) -> None:
        self.substeps, self.stride = _cadence("SourceImplicit", substeps, stride)
        self.implicit_vars, self.implicit_roles = _norm_implicit(
            "SourceImplicit", implicit_vars, implicit_roles)
        (self.newton_max_iters, self.newton_rel_tol, self.newton_abs_tol,
         self.newton_fd_eps, self.newton_diagnostics, self.newton_damping,
         self.newton_fail_policy) = _newton_controls(
             "SourceImplicit", newton_max_iters, newton_rel_tol, newton_abs_tol,
             newton_fd_eps, newton_diagnostics, newton_damping, newton_fail_policy)


# PRECISE name of the scheme wired by IMEX / SourceImplicit (audit 2026-06): ForwardEuler transport
# without source + LOCAL backward-Euler on the source (per-cell Newton). STRICT alias of
# SourceImplicit (same object): to use when you want to name the hypothesis in a script.
SourceImplicitBE = SourceImplicit


class IMEXRK:
    """IMEX-RK family (Implicit-Explicit Runge-Kutta), ARS(2,2,2) scheme, ORDER 2.

    Ascher-Ruuth-Spiteri scheme (1997): the hyperbolic transport L = -div F is treated by the
    EXPLICIT tableau, the stiff source S by the IMPLICIT tableau (LOCAL per-cell backward-Euler,
    Newton, like pops.IMEX) -- but with coupled stages that raise the GLOBAL ORDER TO 2 (transport
    AND source), whereas pops.IMEX stays a ForwardEuler(transport) + backward-Euler(source) of order 1.

    Coefficients: gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma). Tableaus (stiffly accurate):
    explicit A_E = [[0,0,0],[gamma,0,0],[delta,1-delta,0]], b_E = [delta,1-delta,0];
    implicit A_I = [[0,0,0],[0,gamma,0],[0,1-gamma,gamma]], b_I = [0,1-gamma,gamma].

    DISTINCT FAMILY from pops.IMEX (kind="imexrk_ars222" != "imex"): the pops.IMEX default (local
    backward-Euler, order 1) is UNCHANGED / bit-identical. SCOPE: CARTESIAN System only -- AMR, the
    polar grid, compiled models (.so: prototype/aot/production) and the Strang/Schur splittings
    REJECT it explicitly (use pops.IMEX / pops.Explicit on those paths).

    - ``scheme``: "ars222" (only wired scheme; another name raises an explicit error).
    - ``substeps=N``: substeps per macro-step (cf. pops.Explicit). Default 1.
    - ``stride=M``: block cadence, hold-then-catch-up semantics (cf. pops.Explicit). Default 1.
    - ``newton_*``: SAME options as pops.IMEX (max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy/
      diagnostics) -- they parametrize BOTH implicit stage solves of the scheme. Defaults =
      historical constants (max_iters=2, fd_eps=1e-7), without extra cost.

    FULLY IMPLICIT SOURCE: unlike pops.IMEX, IMEXRK does NOT expose implicit_vars /
    implicit_roles (the ARS(2,2,2) stage-consistency relation assumes a homogeneous solve). A partial
    mask is rejected on the C++ side; for a partial per-component IMEX, use pops.IMEX.
    """

    kind = TIME_IMEXRK_ARS222  # typed time route (ADC-584)

    def __init__(self, scheme: str = "ars222", substeps: int = 1, stride: int = 1,
                 newton_max_iters: Any = NEWTON_DEFAULT_MAX_ITERS,
                 newton_rel_tol: Any = NEWTON_DEFAULT_REL_TOL,
                 newton_abs_tol: Any = NEWTON_DEFAULT_ABS_TOL,
                 newton_fd_eps: Any = NEWTON_DEFAULT_FD_EPS,
                 newton_diagnostics: bool = False,
                 newton_damping: Any = NEWTON_DEFAULT_DAMPING,
                 newton_fail_policy: Any = NEWTON_DEFAULT_FAIL_POLICY) -> None:
        if not isinstance(scheme, str) or scheme != "ars222":
            raise ValueError("IMEXRK: scheme 'ars222' (only wired IMEX-RK scheme; got %r)"
                             % (scheme,))
        self.scheme = scheme
        self.substeps, self.stride = _cadence("IMEXRK", substeps, stride)
        (self.newton_max_iters, self.newton_rel_tol, self.newton_abs_tol,
         self.newton_fd_eps, self.newton_diagnostics, self.newton_damping,
         self.newton_fail_policy) = _newton_controls(
             "IMEXRK", newton_max_iters, newton_rel_tol, newton_abs_tol, newton_fd_eps,
             newton_diagnostics, newton_damping, newton_fail_policy)


__all__ = ["_role_to_stable", "_norm_implicit", "IMEX", "SourceImplicit", "SourceImplicitBE",
           "IMEXRK"]
