"""Pipeline Riemann capability-driven sur l'anneau isotherme.

CE QUE VERROUILLE CE TEST :
  T1 - DEFAUT BIT-IDENTIQUE : un run polaire isotherme avec riemann='rusanov' (le defaut) est
       reproductible (deux constructions identiques -> etat bit-identique). On ne PROUVE pas ici la
       non-regression vis-a-vis d'avant le patch (impossible dans un seul process), mais le patch ne
       touche PAS la branche rusanov de make_block_polar (ajout d'une branche 'hll' SEPAREE) : le
       defaut reste strictement l'historique.
  T2 - HLL tourne fini par sa feuille distincte.
  T3 - HLLC/Roe tournent finis et diffèrent de Rusanov : aucun alias/fallback silencieux.
  T4 - un transport ExB sans les capacités exactes refuse HLLC/Roe au lieu de changer de solveur.

Le fluide isotherme polaire expose model.wave_speeds (herite d'IsothermalFlux) : c'est la condition
du gate 'hll' (identique au cartesien block_builder.hpp). Un transport ExB SCALAIRE ne fournit pas
les capacités HLLC/Roe et le test de refus ci-dessous verrouille l'absence de fallback.
"""
import math

import numpy as np

from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov
from pops.numerics.variables import Conservative
import pops.runtime._engine_descriptors as engine
from pops.mesh import PolarMesh
from pops.runtime._engine_descriptors import Dirichlet
from pops.runtime._system import System  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import install_ssprk2_program

RMIN, RMAX = 0.3, 1.0


def iso_polar_model(cs2=1.0):
    """Fluide isotherme NATIF : sur l'anneau, le dispatch polaire (block_builder_polar) instancie
    IsothermalFluxPolar (roles Density / MomentumX (radial) / MomentumY (azimutal)). Second membre
    elliptique neutre (alpha=0) : on isole le TRANSPORT (le flux Riemann), pas le couplage Poisson."""
    return engine.Model(
        state=engine.FluidState(kind="isothermal", cs2=cs2),
        transport=engine.IsothermalFlux(),
        source=engine.NoSource(),
        elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0),
    )


def _annular_state(nr, nth):
    """Etat conservatif (3, ntheta, nr) = (rho, mom_r, mom_theta) lisse, > 0, avec une vitesse
    initiale module en theta (gradient azimutal -> le flux travaille). Layout set_state : composante
    lente, puis j=theta, puis i=r (numpy (ncomp, ntheta, nr) aplati en C-order), cf.
    test_polar_schur_via_system._initial_velocity_state."""
    dr = (RMAX - RMIN) / nr
    dth = 2.0 * math.pi / nth
    rho = np.empty((nth, nr), dtype=np.float64)
    mr = np.empty((nth, nr), dtype=np.float64)
    mth = np.empty((nth, nr), dtype=np.float64)
    for j in range(nth):
        th = (j + 0.5) * dth
        for i in range(nr):
            r = RMIN + (i + 0.5) * dr
            rr = (r - RMIN) / (RMAX - RMIN)
            h = math.sin(math.pi * rr)  # 0 aux bords radiaux (compatible paroi)
            rho[j, i] = 1.5 + 0.3 * math.cos(2.0 * th) * h
            vr = 0.5 * h * math.cos(2.0 * th)
            vth = -0.4 * h * math.sin(th)
            mr[j, i] = rho[j, i] * vr
            mth[j, i] = rho[j, i] * vth
    return np.stack([rho, mr, mth], axis=0)  # (3, ntheta, nr)


def _build(nr, nth, riemann, cs2=1.0):
    """System polaire isotherme avec le flux Riemann demande, etat initial pose, pret a stepper."""
    sim = System(mesh=PolarMesh(r_min=RMIN, r_max=RMAX, nr=nr, ntheta=nth))
    sim.set_poisson(rhs="charge_density", solver="polar", bc=Dirichlet())
    sim.add_equation(
        "ions",
        model=iso_polar_model(cs2=cs2),
        spatial=engine.Spatial(limiter=Minmod(), flux=riemann, recon=Conservative()),
        time=engine.Explicit(),
    )
    u0 = _annular_state(nr, nth)
    sim.set_density("ions", u0[0].ravel())     # pose rho (vitesse au repos)
    sim.set_state("ions", u0.ravel())          # injecte la vitesse initiale (mom_r, mom_theta)
    install_ssprk2_program(sim)
    return sim


def _state3(sim, nr, nth):
    return np.array(sim.get_state("ions")).reshape(3, nth, nr)


def _assert_scalar_rejected(flux, capability):
    sim = System(mesh=PolarMesh(r_min=RMIN, r_max=RMAX, nr=8, ntheta=8))
    sim.set_poisson(rhs="charge_density", solver="polar", bc=Dirichlet())
    model = engine.Model(
        state=engine.Scalar(),
        transport=engine.ExB(B0=1.0),
        source=engine.NoSource(),
        elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0),
    )
    try:
        sim.add_equation(
            "density",
            model=model,
            spatial=engine.Spatial(limiter=Minmod(), flux=flux, recon=Conservative()),
            time=engine.Explicit(),
        )
    except (RuntimeError, ValueError) as error:
        message = str(error)
        assert capability in message and "fallback" in message.lower(), message
        return
    raise AssertionError("scalar ExB accepted %r without %s" % (flux, capability))


def _run(sim, nr, nth, n_steps, dt):
    for _ in range(n_steps):
        sim.step(dt)
    return _state3(sim, nr, nth)


def test_polar_hll():
    nr, nth = 24, 24
    cs2 = 1.0
    h = min((RMAX - RMIN) / nr, RMIN * (2.0 * math.pi / nth))  # pas physique min polaire
    dt = 0.2 * h / math.sqrt(cs2)
    n_steps = 8

    # T1 : defaut rusanov reproductible (deux constructions identiques -> bit-identique).
    s_rus_a = _run(_build(nr, nth, Rusanov(), cs2), nr, nth, n_steps, dt)
    s_rus_b = _run(_build(nr, nth, Rusanov(), cs2), nr, nth, n_steps, dt)
    assert np.array_equal(s_rus_a, s_rus_b), "rusanov polaire : non reproductible (T1)"

    state = _run(_build(nr, nth, HLL(), cs2), nr, nth, n_steps, dt)
    assert np.all(np.isfinite(state)), "hll polaire : etat non fini (T2)"
    diff = float(np.max(np.abs(state - s_rus_a)))
    assert diff > 1e-8, "hll polaire est un alias/fallback Rusanov (diff=%.3e)" % diff


def test_polar_isothermal_hllc_and_roe_use_requested_provider():
    nr, nth = 24, 24
    cs2 = 1.0
    h = min((RMAX - RMIN) / nr, RMIN * (2.0 * math.pi / nth))
    dt = 0.2 * h / math.sqrt(cs2)
    reference = _run(_build(nr, nth, Rusanov(), cs2), nr, nth, 8, dt)
    for name, provider in (("hllc", HLLC()), ("roe", Roe())):
        state = _run(_build(nr, nth, provider, cs2), nr, nth, 8, dt)
        assert np.all(np.isfinite(state)), "%s polaire : etat non fini (T3)" % name
        diff = float(np.max(np.abs(state - reference)))
        assert diff > 1e-8, (
            "%s polaire ne differe pas de rusanov (diff=%.3e) : alias/fallback silencieux (T3)"
            % (name, diff)
        )


def test_polar_exb_refuses_missing_hllc_and_roe_capabilities():
    _assert_scalar_rejected(HLLC(), "HasHLLCStructure")
    _assert_scalar_rejected(Roe(), "HasRoeDissipation")


if __name__ == "__main__":
    test_polar_hll()
    test_polar_isothermal_hllc_and_roe_use_requested_provider()
    test_polar_exb_refuses_missing_hllc_and_roe_capabilities()
    print("test_polar_hll : OK (HLL/HLLC/Roe finis, distincts et capability-gated)")
