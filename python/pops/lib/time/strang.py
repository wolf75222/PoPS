"""pops.lib.time.strang -- Strang and Lie splitting macros, and the condensed Schur source stage.

Exports: strang, lie, condensed_schur.
"""
from __future__ import annotations

from typing import Any

from ._helpers import program_macro


@program_macro
def strang(P: Any, block: Any, half_flow: Any, source: Any, *, commit: Any = True) -> Any:
    """Strang splitting macro H(dt/2); S(dt); H(dt/2), the macro form of pops.Strang (lowers to the SAME
    IR, no special class). @p half_flow and @p source are IR-building callables (prog, state, frac) ->
    state that advance the hyperbolic flow and the source by a fraction @p frac of dt. Returns the final
    state (committed when @p commit)."""
    U = P.state(block)
    U1 = half_flow(P, U, 0.5)
    U2 = source(P, U1, 1.0)
    U3 = half_flow(P, U2, 0.5)
    if commit:
        P.commit(block, U3)
    return U3


@program_macro
def lie(P: Any, block: Any, half_flow: Any, source: Any, *, commit: Any = True) -> Any:
    """Lie (Godunov) splitting macro H(dt); S(dt) -- the sequential first-order sibling of `strang`
    (ADC-423). @p half_flow and @p source are the SAME IR-building callables `strang` takes
    ``(prog, state, frac) -> state`` (each advances its sub-flow by a fraction @p frac of dt); Lie
    just composes them sequentially over the FULL step (H over dt, then S over dt) with no half-steps.
    Lowers to the SAME IR primitives as `strang` (no scheme-specific class). Returns the final state
    (committed when @p commit)."""
    U = P.state(block)
    U1 = half_flow(P, U, 1.0)
    U2 = source(P, U1, 1.0)
    if commit:
        P.commit(block, U2)
    return U2


@program_macro
def condensed_schur(P: Any, block: Any, *, alpha: Any, theta: Any = 1.0, c_rho: Any = 0,
                    c_mx: Any = 1, c_my: Any = 2, c_E: Any = None,
                    method: Any = None, tol: Any = 1e-10, max_iter: Any = 400,
                    linear_operator: Any = None, commit: Any = True) -> Any:
    """Condensed-implicit electrostatic-Lorentz SOURCE stage as a compiled Program (epic ADC-399,
    acceptance 32), authored ENTIRELY in the DSL and emitted to C++ with no Schur vocabulary (ADC-637).
    Mirrors the native ``pops.CondensedSchur`` (CondensedSchurSourceStepper) sequence:

      1. assemble the anisotropic tensor coefficient ``A = I + c*rho*M^{-1}`` (``P.condensed_coeffs``,
         ``M = I - theta*dt*J``, ``c = theta^2 dt^2 alpha``);
      2. assemble the fused RHS ``-Lap(phi^n) - theta*dt*alpha*div(M^{-1}(mx,my))``
         (``P.condensed_rhs``);
      3. solve ``-div(A grad phi^{n+theta}) = RHS`` matrix-free (``P.matrix_free_operator`` +
         ``P.apply_laplacian_coeff`` negated, ``P.solve_linear``), warm-started from phi^n;
      4. reconstruct ``v^{n+theta} = M^{-1}(v^n - theta*dt*grad phi)`` and write ``mom = rho*v``
         (``P.condensed_reconstruct``, the closed block inverse); rho stays frozen;
      5. (``theta < 1``) extrapolate the theta-stage state to ``n+1`` by the native factor ``1/theta``:
         ``U^{n+1} = U^n + (1/theta)(U^{n+theta} - U^n)`` (the affine algebra, see THETA below);
      6. (``c_E`` given) update the total energy ``E^{n+1} = E^n + (1/2)rho(|v^{n+1}|^2 - |v^n|^2)``
         (``P.condensed_energy``, the kinetic-energy increment).

    The per-cell block linearization ``J`` is the electrostatic-Lorentz rotation generator
    ``J = [[0, B_z], [-B_z, 0]]`` on the coupled momentum subset ``(c_mx, c_my)``. The compiling model
    MUST carry it -- author it with ``pops.lib.models.author_electrostatic_lorentz(m)`` (canonical name
    ``pops.lib.models.LORENTZ_J_NAME``, referenced by default). ``B_z`` enters through J's aux (canonical
    component 3), not a separate ``c_bz``. @p linear_operator overrides the operator name / handle.

    phi^n handling depends on theta. At ``theta == 1`` (the sanctioned backward-Euler electrostatic
    push) phi^n is a fresh ZERO scalar field each step: the ``-Lap(phi^n)`` RHS term vanishes and the
    solve warm starts from zero, so a step matches the native step from ``phi^n = 0`` and the historical
    IR / trajectory is byte-identical. At ``theta < 1`` phi^n is CARRIED across steps (ADC-427): last
    step's phi^{n+1} is this step's phi^n, kept in a lag-1, 1-component System history ring (the
    ncomp-aware ``register_history``), exactly as the native stepper carries it via ``ell_phi`` + the
    ``-Lap(phi^n)`` anchor. The ring's cold-start fill seeds phi^n = 0 at step 0, so the FIRST theta<1
    step still matches native. Every numerical kernel is emitted inline from the authored J (no stencil /
    block inverse reimplementation); the native ``pops.CondensedSchur`` stepper is untouched.

    THETA != 1 (ADC-427). The native stepper takes the implicit stage at ``n+theta`` and extrapolates
    phi AND the MOMENTUM (not rho) to ``n+1`` by the factor ``1/theta``. This macro lowers BOTH with the
    EXISTING affine algebra, no component-restricted IR op. State: ``condensed_reconstruct`` freezes rho
    (and energy), so ``rho^{n+theta} = rho^n`` and ``mom^{n+theta} = rho v^{n+theta}``,
    ``mom^n = rho v^n``. The plain STATE affine ``U^n + (1/theta)(U^{n+theta} - U^n)`` therefore leaves
    rho (and a yet-unwritten energy) untouched -- ``rho^{n+1} = (1-1/theta)rho^n + (1/theta)rho^n =
    rho^n`` -- and on the momentum it equals the native ``mom^{n+1} = mom^n + (1/theta)(mom^{n+theta} -
    mom^n) = rho(v^n + (1/theta)(v^{n+theta} - v^n))``. Phi: the SCALAR affine ``phi^n + (1/theta)(
    phi^{n+theta} - phi^n)`` (a ``linear_combine`` of 1-component fields) equals the native
    ``SchurExtrapolateScalarKernel``; the result is stored into the history ring so it is the NEXT step's
    phi^n. This carry is what lifts theta = 0.5 to second-order (Crank-Nicolson) temporal accuracy. It is
    GATED to ``theta != 1`` so the theta == 1 golden (the fresh-zero phi path) stays byte-identical;
    a user selects theta = 1 or theta = 0.5, never sweeps continuously through 1.

    @p alpha is the electrostatic coupling constant; @p theta the theta-scheme implicitness in ``(0, 1]``;
    @p c_rho / @p c_mx / @p c_my the conserved-variable components and @p c_E the OPTIONAL energy
    component (None = no energy update, like a rho/mx/my isothermal block). @p method (a TYPED
    pops.solvers.krylov descriptor; None defaults to BiCGStab()) / @p tol / @p max_iter configure the
    Krylov phi solve.

    NEAR-MATCH to native, not bit-exact: the native solve is BiCGStab + GeometricMG preconditioner while
    the Program solve is matrix-free BiCGStab WITHOUT a preconditioner -- the SAME operator and RHS, a
    different Krylov path (both converge to the same phi at tolerance).
    ``tests/python/unit/time/test_time_condensed_schur.py`` checks against an offline reference of the
    identical assemble / solve / reconstruct / extrapolate steps and documents the gap vs native
    (theta == 1 and theta == 0.5)."""
    if not (0.0 < float(theta) <= 1.0):
        raise ValueError("condensed_schur: theta must be in (0, 1] (got %r)" % (theta,))
    if c_E is not None and (isinstance(c_E, bool) or not isinstance(c_E, int) or c_E < 0):
        raise ValueError("condensed_schur: c_E must be None or a Python int >= 0 (got %r)" % (c_E,))
    if linear_operator is None:
        from pops.lib.models import LORENTZ_J_NAME
        linear_operator = LORENTZ_J_NAME
    subset = (c_mx, c_my)  # the coupled momentum block the condensed solve eliminates (2D core invariant)
    U = P.state(block)
    P.solve_fields(U)  # fill the shared aux (B_z at component 3) from the current state, like the native stage
    # phi^n. theta == 1 (the sanctioned backward-Euler push) keeps a fresh ZERO scalar field each step:
    # the -Lap(phi^n) RHS term vanishes and the solve warm starts from zero, so the historical IR /
    # trajectory is byte-identical. theta < 1 CARRIES phi^n across steps through a lag-1, 1-component
    # System history ring (ADC-427): last step's phi^{n+1} is this step's phi^n, exactly as the native
    # CondensedSchurSourceStepper carries it (via ell_phi + the -Lap(phi^n) anchor). The ring's
    # cold-start fill seeds phi^n = 0 at step 0, so the FIRST theta<1 step still matches native.
    carry = float(theta) != 1.0
    if carry:
        phi_n = P.history(block + ".schur_phi", lag=1, ncomp=1)  # scalar read; cold-start = 0
    else:
        phi_n = P.scalar_field(block + ".schur_phi_n")           # UNCHANGED: fresh zero each step
    c_coeff = (float(theta) * float(theta) * float(alpha)) * P.dt * P.dt  # c = theta^2 dt^2 alpha
    th_dt = float(theta) * P.dt  # theta dt
    g = (float(theta) * float(alpha)) * P.dt  # theta dt alpha (coefficient of the div(F) term)
    # The three per-cell stages carry the authored J + the coupled momentum subset; B_z enters through J's
    # aux, emitted inline via the closed-form block_inverse intrinsic with no Schur vocabulary.
    coeffs = P.condensed_coeffs(state=U, linear_operator=linear_operator, subset=subset,
                                c=c_coeff, th_dt=th_dt, c_rho=c_rho)
    rhs = P.scalar_field(block + ".schur_rhs")
    P.condensed_rhs(rhs, phi_n, U, linear_operator=linear_operator, subset=subset, th_dt=th_dt, g=g)
    A = P.matrix_free_operator(block + ".schur_op")

    def apply(P: Any, out: Any, x: Any) -> Any:  # out <- A(x) = -div((I + c rho M^{-1}) grad x) = -apply_laplacian_coeff(x)
        lap = P.scalar_field("schur_lap")
        P.apply_laplacian_coeff(lap, x, coeffs)
        return -1.0 * lap  # the condensed operator -div(A grad phi); the affine is the lowered result

    # Spec 5 sec.7: method is a TYPED pops.solvers.krylov descriptor (default BiCGStab(), the
    # native CondensedSchur solver). A bare string is rejected by P.solve_linear with a clear
    # message naming the typed alternative; None defaults here byte-identically to the old
    # "bicgstab" string.
    if method is None:
        from pops.solvers.krylov import BiCGStab
        method = BiCGStab(max_iter=max_iter)
    P.set_apply(A, apply)
    # theta < 1 warm-starts the Krylov solve from phi^n (the carried potential), like the native stepper;
    # theta == 1 keeps the zero warm start (initial_guess=None) so the IR is byte-identical.
    phi = P.solve_linear(operator=A, rhs=rhs, method=method, tol=tol, max_iter=max_iter,
                         initial_guess=phi_n if carry else None)
    # The reconstruction overwrites the MOMENTUM in place. theta == 1 with no energy keeps the historical
    # IR byte-identical (reconstruct directly on U). For theta < 1 OR an energy update we need U^n
    # (mom^n / E^n) AFTER the reconstruction, so reconstruct on a fresh COPY of U^n and keep U^n intact.
    needs_un = float(theta) != 1.0 or c_E is not None
    target = P.linear_combine(block + ".schur_un_copy", 1.0 * U) if needs_un else U
    out = P.condensed_reconstruct(state=target, phi=phi, linear_operator=linear_operator,
                                  subset=subset, th_dt=th_dt, c_rho=c_rho)
    # 5) theta-stage -> n+1 extrapolation (ADC-427). theta < 1 lowers U^n + (1/theta)(U^{n+theta} - U^n)
    # with the affine algebra (out is the theta-stage on the copy, U^n is the untouched original). rho is
    # frozen by the reconstruction, so this affine leaves rho (and the not-yet-written energy) at U^n.
    if float(theta) != 1.0:
        inv_theta = 1.0 / float(theta)
        out = P.linear_combine(block + ".schur_extrap", U + inv_theta * (out - U))
    # 6) energy role (ADC-427). E^{n+1} = E^n + (1/2)rho(|v^{n+1}|^2 - |v^n|^2): the kinetic-energy
    # increment from v^n (= mom^n/rho, read from U^n) to v^{n+1} (= mom^{n+1}/rho, in `out`). Skipped
    # for an isothermal rho/mx/my block (c_E is None). Emitted generically (no Schur kernel).
    if c_E is not None:
        out = P.condensed_energy(state=out, state_old=U, c_rho=c_rho, c_mx=c_mx, c_my=c_my, c_E=c_E)
    # PERSISTENT phi^n carry (ADC-427, theta < 1 only). Extrapolate phi to n+1 by the SAME 1/theta
    # factor as the state (the native SchurExtrapolateScalarKernel: phi^{n+1} = phi^n + (1/theta)(
    # phi^{n+theta} - phi^n)) and store it so the next step reads it as phi^n. A scalar-field affine
    # (all terms 1-component), lowered through the same axpy/lincomb idiom as a State combine. Every op
    # here is gated by `carry`, so theta == 1 emits none of it and stays byte-identical.
    if carry:
        inv_theta = 1.0 / float(theta)
        phi_np1 = P.linear_combine(block + ".schur_phi_np1", phi_n + inv_theta * (phi - phi_n))
        P.store_history(block + ".schur_phi", phi_np1)  # rotated to lag 1 for the next step
    if commit:
        P.commit(block, out)
    return out
