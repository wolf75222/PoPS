"""ADC-77 end-to-end: the FluidState(vacuum_floor=...) knob threads to the native isothermal model.

Builds two identical isothermal Systems that differ ONLY by vacuum_floor and shows:
  (1) on a QUASI-VACUUM state (rho << floor) with O(1) velocity, the bounded model velocity
      u = m/max(rho, floor) changes the trajectory -> the two runs differ (the knob is wired through
      FluidState -> ModelSpec -> IsothermalFlux);
  (2) on a NORMAL state (rho >> floor) the floor is inactive -> the two runs are BIT-IDENTICAL (the
      bound does not perturb normal runs -- the reason it is a SEPARATE knob from positivity_floor);
  (3) vacuum_floor < 0 is rejected at the python boundary.

The private ModelSpec adapter compiles the isothermal transport; NoSource isolates the velocity
floor. The positive cases explicitly use periodic transport, so every face has a valid ghost source.
The field's Dirichlet declaration does not supply a hyperbolic boundary. A separate nonperiodic
case without that physical transport contract must fail and roll back its attempted step.
"""
from tests.python.support.requirements import require_native_or_skip
from pops.numerics.variables import Conservative
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
from tests.python.support.explicit_program import install_forward_euler_program

import numpy as np

try:
    import pops.runtime._engine_descriptors as engine
    from pops.runtime._engine_descriptors import Dirichlet, Periodic
    from pops.runtime._system import System  # ADC-545 advanced runtime seam
except ImportError as e:
    require_native_or_skip('module pops absent (PYTHONPATH ?) : %s' % e)


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        raise AssertionError(label)


def build(n, L, vacuum_floor, rho_scale, *, periodicity):
    """Author the transport topology explicitly; a Poisson BC never fills transport ghosts."""
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # Keep the original sampled profiles and density scales. Transport topology is declared above;
    # these samples do not themselves constitute a physical ghost-cell boundary condition.
    rho = rho_scale * (1.0 + 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L))
    u = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    sim = System(n=n, L=L, periodicity=periodicity)
    periodic = all(periodicity)
    sim.set_poisson(bc=Periodic() if periodic else Dirichlet())
    sim.add_equation(
        "ions",
        model=engine.Model(
            state=engine.FluidState(kind="isothermal", cs2=1.0, vacuum_floor=vacuum_floor),
            transport=engine.IsothermalFlux(),
            source=engine.NoSource(),
            # Periodic Poisson requires an explicitly neutral load. This fixed background is
            # the initial mean, conserved by periodic transport; NoSource keeps phi out of the rate.
            elliptic=engine.BackgroundDensity(alpha=1.0, n0=float(rho.mean()) if periodic else 0.0),
        ),
        spatial=engine.Spatial(limiter=Minmod(), flux=Rusanov(), recon=Conservative()),
        time=engine.Explicit(),
    )
    sim.set_primitive_state("ions", rho=rho, u=u, v=v)
    install_forward_euler_program(sim)
    return sim


def run(n, L, vacuum_floor, rho_scale, nsteps, dt):
    """The full five-step transport oracle with an explicit periodic ghost source."""
    sim = build(n, L, vacuum_floor, rho_scale, periodicity=(True, True))
    for _ in range(nsteps):
        sim.step(dt)
    return np.array(sim.get_state("ions")).reshape(3, n, n)


def main():
    n, L = 24, 1.0
    dt = 1.0e-3
    nsteps = 5
    floor = 1.0e-2

    # (1) quasi-vacuum: rho ~ 1.5e-3 << floor -> bounded velocity changes the momentum flux.
    s_off = run(n, L, 0.0, 1.0e-3, nsteps, dt)
    s_on = run(n, L, floor, 1.0e-3, nsteps, dt)
    chk(np.all(np.isfinite(s_off)) and np.all(np.isfinite(s_on)), "(1) both runs finite")
    dmax_vac = float(np.max(np.abs(s_on - s_off)))
    chk(dmax_vac > 1e-9,
        "(1) vacuum_floor wired: bounded velocity changes the trajectory at rho<<floor (dmax=%.3e)"
        % dmax_vac)

    # (2) normal density: rho ~ 1.5 >> floor -> floor inactive -> bit-identical.
    t_off = run(n, L, 0.0, 1.0, nsteps, dt)
    t_on = run(n, L, floor, 1.0, nsteps, dt)
    dmax_norm = float(np.max(np.abs(t_on - t_off)))
    chk(dmax_norm == 0.0,
        "(2) floor inactive at rho>>floor: bit-identical to vacuum_floor=0 (dmax=%.3e)" % dmax_norm)

    # (3) validation at the python boundary.
    try:
        engine.FluidState(kind="isothermal", cs2=1.0, vacuum_floor=-1.0)
        chk(False, "(3) vacuum_floor < 0 rejected")
    except ValueError:
        chk(True, "(3) vacuum_floor < 0 rejected")

    # (4) The old nonperiodic setup declared only a field BC. Its invalid transport evaluation must
    # be rejected without publishing any state or clock, rather than being masked by a density floor.
    invalid = build(n, L, 0.0, 1.0e-3, periodicity=(False, False))
    before = np.array(invalid.get_state("ions"), copy=True)
    before_time, before_step = invalid.time(), invalid.macro_step()
    try:
        invalid.step(dt)
    except RuntimeError as exc:
        chk("status 1" in str(exc) or "status=1" in str(exc),
            "(4) missing physical transport BC fails through the native invalid-evaluation guard")
    else:
        chk(False, "(4) nonperiodic transport without a physical boundary must be refused")
    chk(np.array_equal(invalid.get_state("ions"), before),
        "(4) invalid physical boundary preserves the complete conservative state")
    chk((invalid.time(), invalid.macro_step()) == (before_time, before_step),
        "(4) invalid physical boundary publishes no time or macro-step")

    print("test_isothermal_vacuum_floor_system : tout est vert (4 verifications)")


if __name__ == "__main__":
    main()
