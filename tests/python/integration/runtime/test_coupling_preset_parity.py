"""ADC-595: the named-coupling PRESETS reproduce the deleted C++ helpers, bit-identical.

The named couplings ``Ionization`` / ``Collision`` / ``ThermalExchange`` used to be hand-coded C++
methods (``System::add_ionization`` / ``add_collision`` / ``add_thermal_exchange``); they are now
PRESETS that lower to the generic coupled source (``pops.physics.coupling_presets``). This test pins the
numerical parity: the trajectory a preset produces must match the trajectory the OLD HELPER produced.

The reference trajectories below were CAPTURED from the pre-ADC-595 helpers (borrowed ``pops`` 0.3.0
build, before the helpers were deleted) on a SPATIALLY UNIFORM state where the hyperbolic transport and
the zero-charge Poisson force are EXACT no-ops (verified: rho_a stays 1.0 to the bit), so only the
coupling acts. They are embedded as literals with this provenance so the test survives the deletion of
the helpers in the same PR.

Parity result: the presets are BIT-IDENTICAL to the helpers on these representative states
(max abs err == 0 for Collision, Ionization AND ThermalExchange), because each preset builds its Expr in
the exact C++ associativity (including the ``(gamma-1)`` factor and the pressure closure order) and the
``add_pair`` sign convention matches the helper's ``ua -= dt*F ; ub += dt*F``. The one theoretical caveat
(documented in the CHANGELOG) is the position of ``dt``: the kernel applies ``dt * S`` after evaluating
``S`` whereas the helper folded ``dt`` INTO the product, so for NON-representative values a ~1 ULP drift
per step is possible; the tolerance below is bit-exact for the tested states, with a tiny epsilon guard.
"""
import numpy as np
import pytest
from pops.runtime._system import System  # ADC-545 advanced runtime seam

# The _bootstrap of a mismatched-interpreter extension raises ImportError (a subclass), so gate on it.
pops = pytest.importorskip("pops", exc_type=ImportError)
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._engine_descriptors import Periodic  # noqa: E402


N = 8
DT = 0.01
# Bit-exact on the tested uniform states; a 1e-13 guard absorbs a stray dt-folding ULP without hiding a
# real formula divergence (a wrong formula drifts far above 1e-13 within a few steps).
TOL = 1e-13

# --- reference trajectories captured from the deleted C++ helpers (pops 0.3.0) --------------------
# Collision(a, b, k=0.5): (u_a, v_a, u_b, v_b) at cell (0,0) after each step.
COLLISION_REF = [
    (0.496, 0.19950000000000001, -0.29799999999999999, 0.10025000000000001),
    (0.49203000000000002, 0.19900375000000001, -0.29601499999999997, 0.10049812500000001),
    (0.488089775, 0.198511221875, -0.29404488749999996, 0.10074438906250001),
    (0.4841791016875, 0.1980223877109375, -0.29208955084374999, 0.10098880614453126),
    (0.48029775842484373, 0.19753721980310546, -0.29014887921242188, 0.10123139009844728),
]
# Ionization(e, i, g, k=1.7): (rho_e, rho_i, rho_g) at cell (0,0) after each step. rho_i + rho_g is
# conserved (mass transfer) while rho_e grows (electron creation).
IONIZATION_REF = [
    (0.30509999999999998, 0.1051, 0.99490000000000001),
    (0.31026024783, 0.11026024783, 0.98973975216999999),
    (0.31548055514352297, 0.11548055514352294, 0.98451944485647702),
    (0.32076069974074251, 0.12076069974074249, 0.97923930025925743),
    (0.326100424954544, 0.12610042495454399, 0.97389957504545588),
]
# ThermalExchange(a, b, k=0.3), gamma_a=1.4, gamma_b=1.6667: (p_a, p_b) at cell (0,0) after each step.
THERMAL_REF = [
    (1.9981999999999998, 1.0030001500000001),
    (1.9964039600899997, 1.0059936995199925),
    (1.9946118715576038, 1.0089806630813636),
    (1.9928237257095833, 1.0119610551735514),
    (1.991039513871836, 1.0149348902541169),
]


def _uni(value):
    return np.full((N, N), value)


def _compressible(gamma):
    return engine.Model(state=engine.FluidState("compressible", gamma=gamma),
                      transport=engine.CompressibleFlux(),
                      source=engine.PotentialForce(charge=0.0),
                      elliptic=engine.ChargeDensity(charge=0.0))


def _isothermal():
    return engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                      transport=engine.IsothermalFlux(),
                      source=engine.PotentialForce(charge=0.0),
                      elliptic=engine.ChargeDensity(charge=0.0))


def test_collision_preset_matches_deleted_helper():
    sim = System(n=N, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    sim.add_equation("a", _compressible(1.4), spatial=engine.Spatial())
    sim.add_equation("b", _compressible(1.4), spatial=engine.Spatial())
    sim.set_primitive_state("a", rho=_uni(1.0), u=_uni(0.5), v=_uni(0.2), p=_uni(1.0))
    sim.set_primitive_state("b", rho=_uni(2.0), u=_uni(-0.3), v=_uni(0.1), p=_uni(1.0))
    sim.add_coupling(engine.Collision("a", "b", 0.5))  # lowered via the preset -> add_coupling_operator
    # The coupling is registered as a TYPED operator with the declared momentum contract.
    ops = sim.coupled_operators()
    assert len(ops) == 1 and ops[0]["conserved_roles"] == ["momentum_x", "momentum_y"], ops
    for step in range(len(COLLISION_REF)):
        sim.step(DT)
        pa, pb = sim.get_primitive_state("a"), sim.get_primitive_state("b")
        got = (np.asarray(pa["u"])[0, 0], np.asarray(pa["v"])[0, 0],
               np.asarray(pb["u"])[0, 0], np.asarray(pb["v"])[0, 0])
        for g, ref in zip(got, COLLISION_REF[step], strict=False):
            assert abs(g - ref) <= TOL, ("collision drift at step %d: got %.17g ref %.17g"
                                         % (step, g, ref))


def test_ionization_preset_matches_deleted_helper():
    sim = System(n=N, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    for name in ("e", "i", "g"):
        sim.add_equation(name, _isothermal(), spatial=engine.Spatial())
    sim.set_primitive_state("e", rho=_uni(0.3), u=_uni(0.0), v=_uni(0.0))
    sim.set_primitive_state("i", rho=_uni(0.1), u=_uni(0.0), v=_uni(0.0))
    sim.set_primitive_state("g", rho=_uni(1.0), u=_uni(0.0), v=_uni(0.0))
    sim.add_coupling(engine.Ionization("e", "i", "g", 1.7))
    # Ionization is a declared NET SOURCE in density (electron creation), not a conserved exchange.
    ops = sim.coupled_operators()
    assert len(ops) == 1 and ops[0]["created_roles"] == ["density"], ops
    assert ops[0]["conserved_roles"] == [], ops
    for step in range(len(IONIZATION_REF)):
        sim.step(DT)
        got = (np.asarray(sim.get_primitive_state("e")["rho"])[0, 0],
               np.asarray(sim.get_primitive_state("i")["rho"])[0, 0],
               np.asarray(sim.get_primitive_state("g")["rho"])[0, 0])
        for g, ref in zip(got, IONIZATION_REF[step], strict=False):
            assert abs(g - ref) <= TOL, ("ionization drift at step %d: got %.17g ref %.17g"
                                         % (step, g, ref))
        # rho_i + rho_g stays conserved (mass transfer) while rho_e grew (creation).
        assert abs((got[1] + got[2]) - 1.1) <= 1e-12, "ion mass transfer rho_i+rho_g conserved"


def test_thermal_exchange_preset_matches_deleted_helper():
    sim = System(n=N, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    sim.add_equation("a", _compressible(1.4), spatial=engine.Spatial())
    sim.add_equation("b", _compressible(1.6667), spatial=engine.Spatial())
    sim.set_primitive_state("a", rho=_uni(1.0), u=_uni(0.0), v=_uni(0.0), p=_uni(2.0))
    sim.set_primitive_state("b", rho=_uni(2.0), u=_uni(0.0), v=_uni(0.0), p=_uni(1.0))
    sim.add_coupling(engine.ThermalExchange("a", "b", 0.3))
    ops = sim.coupled_operators()
    assert len(ops) == 1 and ops[0]["conserved_roles"] == ["energy"], ops
    for step in range(len(THERMAL_REF)):
        sim.step(DT)
        pa = np.asarray(sim.get_primitive_state("a")["p"])[0, 0]
        pb = np.asarray(sim.get_primitive_state("b")["p"])[0, 0]
        for g, ref in zip((pa, pb), THERMAL_REF[step], strict=False):
            assert abs(g - ref) <= TOL, ("thermal drift at step %d: got %.17g ref %.17g"
                                         % (step, g, ref))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
