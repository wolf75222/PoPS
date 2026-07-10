"""Named inter-species coupling PRESETS lowering to the generic coupled source (ADC-595).

The named couplings ``Ionization`` / ``Collision`` / ``ThermalExchange`` used to be hard-coded C++
methods (``System::add_ionization`` / ``add_collision`` / ``add_thermal_exchange``), each freezing a
formula. They are now PRESETS: thin builders that emit the SAME formula as a
:class:`~pops.physics.multispecies.CoupledSource`, so a named coupling is just a well-known instance of
the one generic representation -- no new C++ method per coupling.

Each preset returns a ``CoupledSource`` PLUS a DECLARED conservation contract (``conserved`` /
``created`` roles), so the coupling's conservation is a machine-checked declaration, not magic
``add_pair`` behavior. The contract is validated symbolically by
``CoupledSource._verify_conservation`` (Python) and again at C++ registration
(``validate_coupling_contract``); ionization legally NET-sources density (declared ``created``).

Formula parity (matched line-for-line to the deleted C++ helpers): the source terms build the Expr in
the SAME associativity as the hand-written C++, so the compiled bytecode drives the same device stack
machine. The one unavoidable difference is the position of ``dt``: the kernel applies ``dt * S`` after
evaluating ``S`` (frozen-register additive split), whereas the helper folded ``dt`` INTO the product
(``dt * k * ...``); floating-point multiplication is non-associative, so ``dt * (k * ...)`` differs from
``(dt * k) * ...`` by at most one ULP per step. The presets are therefore NEAR-EXACT to the old helpers
(~1e-16 per step), not bit-exact -- documented in the CHANGELOG, same precedent as ``strang.py``.

Import-graph rule (Spec 4): pure ``pops.ir`` / ``pops.physics.multispecies`` + stdlib. No codegen /
``_pops`` import; the preset only BUILDS a CoupledSource, the install layer compiles and registers it.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from pops.physics._scalars import (
    exact_physics_scalar,
    native_real,
    scalar_data_view,
    subtract_exact_integer,
)
from pops.physics.multispecies import CoupledSource


# --- declared conservation contract carried alongside a preset's CoupledSource -------------------
class ContractedCoupling:
    """A preset's :class:`CoupledSource` PLUS its DECLARED conservation contract (ADC-595).

    ``conserved`` lists the roles the coupling conserves (its terms cancel over the participating
    blocks); ``created`` lists the roles it legally net-sources (e.g. ionization creating density).
    ``frequency`` is the declared coupling frequency mu (0 = no bound). The install layer compiles
    ``source`` and registers it as a typed coupling operator carrying this contract.
    """

    __slots__ = ("source", "conserved", "created", "frequency")

    def __init__(self, source: Any, conserved: Any = (), created: Any = (),
                 frequency: Any = 0) -> None:
        self.source = source
        self.conserved = list(conserved)
        self.created = list(created)
        self.frequency = exact_physics_scalar(
            frequency, where="ContractedCoupling.frequency")

    def __repr__(self) -> str:
        return (
            "ContractedCoupling(source=%r, conserved=%r, created=%r, frequency=%r)"
            % (self.source.name, self.conserved, self.created,
               scalar_data_view(self.frequency, where="ContractedCoupling.frequency"))
        )


def ionization_preset(electron: Any, ion: Any, neutral: Any, rate: Any,
                      name: str = "ionization") -> Any:
    """Ionization ``n_g -> n_i + n_e`` at rate ``r = k * n_e * n_g`` (ADC-595 preset).

    Reproduces ``System::add_ionization`` (deleted): one neutral disappears, one ion and one electron
    appear -- ``n_g -= dt*r``, ``n_i += dt*r``, ``n_e += dt*r`` -- so mass transfers from the neutral to
    the ion (``n_i + n_g`` conserved) while the electron/ion PAIR is CREATED. The rate product is built
    as ``(k * n_e) * n_g`` to match the C++ left-to-right grouping. Declared ``created`` in ``density``:
    the coupling legally net-sources (it is NOT conservative in density), so
    ``_verify_conservation`` must NOT reject it.

    Returns a :class:`ContractedCoupling` (source + declared ``created=['density']`` contract).
    """
    src = CoupledSource(name)
    # Only the electron and neutral densities are READ (the rate is k * n_e * n_g); the ion density is a
    # write target only, so it is NOT registered as an input (matching the C++ helper's register set).
    ne = src.block(electron).role("Density")
    ng = src.block(neutral).role("Density")
    k = src.param("k_ionization", rate)
    r = (k * ne) * ng  # r = k * n_e * n_g, grouped ((k*ne)*ng) as the C++ helper
    src.add(neutral, role="density", expr=-r)   # n_g -= dt*r
    src.add(ion, role="density", expr=r)        # n_i += dt*r
    src.add(electron, role="density", expr=r)   # n_e += dt*r
    # Ionization is NOT conservative in density (an e/i pair is created): declare it created so the
    # contract validator allows the net source instead of demanding cancellation.
    return ContractedCoupling(src, created=["density"])


def collision_preset(a: Any, b: Any, rate: Any, name: str = "collision") -> Any:
    """Inter-species friction ``F = k * (u_a - u_b)`` on the momentum (ADC-595 preset).

    Reproduces ``System::add_collision`` (deleted): the force is applied with OPPOSITE sign on each
    species -- ``ua.m -= dt*F``, ``ub.m += dt*F`` -- so the total momentum is CONSERVED. The velocity is
    read as ``m / rho`` and the exchanged value per axis is ``k * (mxa/da - mxb/db)``, built in the C++
    grouping ``k * ((mxa/da) - (mxb/db))``. ``add_pair(b, a, ...)`` emits ``+expr`` on @p b (which GAINS,
    matching ``ub += dt*F``) and ``-expr`` on @p a (which LOSES, matching ``ua -= dt*F``), so the two
    legs share the SAME subtree with opposite sign -- conservative by construction.

    Returns a :class:`ContractedCoupling` (source + declared ``conserved=['momentum_x','momentum_y']``).
    """
    src = CoupledSource(name)
    mxa = src.block(a).role("MomentumX")
    mya = src.block(a).role("MomentumY")
    da = src.block(a).role("Density")
    mxb = src.block(b).role("MomentumX")
    myb = src.block(b).role("MomentumY")
    db = src.block(b).role("Density")
    k = src.param("k_collision", rate)
    fx = k * ((mxa / da) - (mxb / db))  # F_x = k (u_xa - u_xb); C++ grouping preserved
    fy = k * ((mya / da) - (myb / db))  # F_y = k (u_ya - u_yb)
    # b GAINS +F (ub += dt*F), a LOSES -F (ua -= dt*F): add_pair(block_a=b, block_b=a) emits +expr on b
    # and -expr on a -- the exact signs of the C++ helper.
    src.add_pair(b, a, role="momentum_x", expr=fx)
    src.add_pair(b, a, role="momentum_y", expr=fy)
    return ContractedCoupling(src, conserved=["momentum_x", "momentum_y"])


def thermal_exchange_preset(a: Any, b: Any, rate: Any, gamma_a: Any, gamma_b: Any,
                            name: str = "thermal_exchange") -> Any:
    """Inter-species thermal exchange ``q = k * (T_a - T_b)`` on the energy (ADC-595 preset).

    Reproduces ``System::add_thermal_exchange`` (deleted): the heat flux is applied with OPPOSITE sign on
    each species -- ``ua.E -= dt*q``, ``ub.E += dt*q`` -- so the total energy is CONSERVED. The
    temperature is ``T = p / rho`` with the ideal-gas pressure ``p = (gamma-1)(E - 0.5 rho |u|^2)``; the
    ADIABATIC INDEX gamma is PER BLOCK (read from each block's descriptor by the install layer and
    inlined here as a ``.param()``, exactly like the helper's ``P->sp[ia].gamma``). The pressure closure
    and ``q = k * (pa/ra - pb/rb)`` are built in the C++ grouping. ``add_pair(b, a, ...)`` emits ``+q`` on
    @p b (GAINS, ``ub += dt*q``) and ``-q`` on @p a (LOSES, ``ua -= dt*q``): conservative by construction.

    Returns a :class:`ContractedCoupling` (source + declared ``conserved=['energy']``).
    """
    src = CoupledSource(name)
    ea = src.block(a).role("Energy")
    mxa = src.block(a).role("MomentumX")
    mya = src.block(a).role("MomentumY")
    da = src.block(a).role("Density")
    eb = src.block(b).role("Energy")
    mxb = src.block(b).role("MomentumX")
    myb = src.block(b).role("MomentumY")
    db = src.block(b).role("Density")
    k = src.param("k_thermal", rate)
    ga = exact_physics_scalar(gamma_a, where="thermal_exchange_preset.gamma_a")
    gb = exact_physics_scalar(gamma_b, where="thermal_exchange_preset.gamma_b")
    gam_a = src.param(
        "gamma_a", subtract_exact_integer(ga, 1, where="thermal_exchange_preset.gamma_a"))
    gam_b = src.param(
        "gamma_b", subtract_exact_integer(gb, 1, where="thermal_exchange_preset.gamma_b"))
    half = src.param("half", Fraction(1, 2))
    # p = (gamma-1) * (E - 0.5 * (mx*mx + my*my) / rho), grouped exactly as the C++ helper.
    pa = gam_a * (ea - half * (((mxa * mxa) + (mya * mya)) / da))
    pb = gam_b * (eb - half * (((mxb * mxb) + (myb * myb)) / db))
    q = k * ((pa / da) - (pb / db))  # q = k (T_a - T_b), T = p / rho
    # b GAINS +q (ub += dt*q), a LOSES -q (ua -= dt*q): add_pair(block_a=b, block_b=a).
    src.add_pair(b, a, role="energy", expr=q)
    return ContractedCoupling(src, conserved=["energy"])


def coupling_operator_args(compiled: Any, conserved: Any = (), created: Any = (),
                           frequency: Any = None) -> Any:
    """Positional args for ``System.add_coupling_operator`` from a compiled coupled source (ADC-595).

    Marshals the flat bytecode program of @p compiled PLUS the declared conservation contract
    (@p conserved / @p created roles) into the tuple the typed C++ entry consumes, so both the
    named-preset and the generic CompiledCoupledSource paths register identically. @p frequency
    overrides the compiled source's own declared frequency when a preset carries one (else the
    compiled value). Kept here (not in the install mixin) to hold the install module under its size cap.
    """
    freq = compiled.frequency if frequency is None else frequency
    # This helper is the explicit Python -> pybind/native ABI boundary.  The compiled descriptor
    # itself remains lossless; only the vectors handed to pops::Real are rounded to binary64 here.
    native_consts = [
        native_real(value, where="coupling_operator_args.constants[%d]" % index)
        for index, value in enumerate(compiled.consts)
    ]
    native_frequency = native_real(freq, where="coupling_operator_args.frequency")
    return (compiled.in_blocks, compiled.in_roles, native_consts, compiled.out_blocks,
            compiled.out_roles, compiled.prog_ops, compiled.prog_args, compiled.prog_lens,
            native_frequency, compiled.name, getattr(compiled, "freq_prog_ops", []),
            getattr(compiled, "freq_prog_args", []), list(conserved), list(created))


def lower_named_coupling(coupling: Any, gamma_of: Any) -> Any:
    """Lower a named coupling object to its :class:`ContractedCoupling` preset, or ``None`` (ADC-595).

    Dispatches ``pops.Ionization`` / ``Collision`` / ``ThermalExchange`` (duck-typed by their fields, to
    avoid importing the ``pops.runtime.bricks`` scheme classes here and creating a cycle) to the matching
    preset builder. ``ThermalExchange`` reads each block's adiabatic index via the @p gamma_of callback
    (``lambda name: sim.block_gamma(name)``), inlined as a per-block ``.param()``, exactly like the
    deleted ``System::add_thermal_exchange`` read ``P->sp[ia].gamma``. Returns ``None`` for any object
    that is not a named coupling (the caller then handles the generic CompiledCoupledSource path)."""
    kind = type(coupling).__name__
    if kind == "Ionization":
        return ionization_preset(coupling.electron, coupling.ion, coupling.neutral, coupling.rate)
    if kind == "Collision":
        return collision_preset(coupling.a, coupling.b, coupling.rate)
    if kind == "ThermalExchange":
        return thermal_exchange_preset(coupling.a, coupling.b, coupling.rate,
                                       gamma_of(coupling.a), gamma_of(coupling.b))
    return None
