#!/usr/bin/env python3
"""ADC-515 (Spec 6 sec.20): the condensed-Schur source stage on a native AmrSystem.

The Strang / Split(Lie) column of the sec.20 matrix. ``AmrSystem.add_equation(time=pops.Strang(
source=pops.CondensedSchur(...)))`` routes the transport to a source-free explicit step and installs
the GLOBAL Schur-condensed source stage (``set_source_stage`` + ``set_time_scheme``), mirroring
``test_amr_schur_via_system``. Two verdicts:

  * MONO block (the capability is green): a native isothermal block + ``set_magnetic_field`` steps to
    a finite density / potential with mass frozen by the condensed stage, under both the 2nd-order
    Strang and the 1st-order Split (Lie) splitting.
  * MULTI block (a precise refusal): the global condensed stage is SINGLE-BLOCK only, so a second
    ``add_equation(source=CondensedSchur)`` block is refused at build with a stable message (pinned
    substrings, not the full sentence). The clean-``compile(layout=AMR)`` whole-system Schur Program
    is a SEPARATE route being implemented under ADC-634 (+ ADC-633 for the compiled hierarchy
    elliptic); it is a pending row in the sec.20 declarative matrix, not asserted here.

Runtime: ``importorskip('pops')`` skips on a bare box; the mono cells step a real Kokkos-Serial
engine on the CI runner. ``__main__`` runs pytest.
"""
import sys

import numpy as np
import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.runtime.system import AmrSystem  # noqa: E402  (ADC-545 advanced runtime seam)


def _iso_model(cs2=1.0, alpha=1.0):
    """A native ISOTHERMAL fluid block: Density / MomentumX / MomentumY roles, source-free transport.

    The roles are exactly the ones the Schur-condensed source stage requires; ``source=NoSource``
    keeps the transport source-free (the condensed stage carries the electrostatic / Lorentz source).
    Mirror of ``test_amr_schur_via_system.iso_model``.
    """
    return pops.Model(state=pops.FluidState(kind="isothermal", cs2=cs2),
                      transport=pops.IsothermalFlux(), source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=alpha, n0=0.0))


def _iso_state(n, L, rho=1.5):
    """A smooth isothermal conservative state (rho > 0, drift zero on the walls)."""
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = rho * np.ones((n, n))
    u = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    return np.stack([r, r * u, r * v])


def _build_schur(splitting, *, n=24, L=1.0):
    """A mono-block AmrSystem with the condensed-Schur source stage under @p splitting (strang|lie)."""
    sim = AmrSystem(n=n, L=L, periodic=False, regrid_every=0)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    sim.set_refinement(1e30)  # mono-level hierarchy: the global stage degenerates to the uniform one
    sim.set_magnetic_field(4.0 * np.ones((n, n)))
    cls = pops.Strang if splitting == "strang" else pops.Split
    sim.add_equation(
        "electrons", model=_iso_model(cs2=1.0, alpha=3.0), spatial=pops.Spatial(minmod=True),
        time=cls(hyperbolic=pops.Explicit(),
                 source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=1.0, alpha=3.0)))
    sim.set_conservative_state("electrons", _iso_state(n, L))
    return sim


@pytest.mark.parametrize("splitting", ["strang", "lie"])
def test_amr_condensed_schur_mono_runs(splitting):
    """AMR x {Strang, Split(Lie)}(source=CondensedSchur) x mono: the global condensed stage runs.

    Finite density / potential after a few steps and mass frozen by the condensed stage (the stiff
    source conserves mass; the strict per-stage parity is the C++ standalone test's job). Both the
    2nd-order Strang and the 1st-order Lie splitting are wired on AMR.
    """
    sim = _build_schur(splitting)
    m0 = sim.mass()
    for _ in range(5):
        sim.step(5.0e-4)
    rho = np.asarray(sim.density())
    phi = np.asarray(sim.potential())
    assert np.isfinite(rho).all(), "%s: density not finite" % splitting
    assert np.isfinite(phi).all(), "%s: potential not finite" % splitting
    assert abs(sim.mass() - m0) <= 1e-9 * max(abs(m0), 1e-30), \
        "%s: mass not conserved by the condensed stage" % splitting


def test_amr_condensed_schur_multiblock_is_refused():
    """AMR x multi-block condensed-Schur source stage: a precise SINGLE-BLOCK refusal.

    The global Schur-condensed source stage is wired for one ``add_equation`` block only; declaring a
    second CondensedSchur block is refused at build with a stable message. Pin only the stable
    substrings so a wording change fails in ONE place (the matrix's purpose), not the full sentence.
    """
    n, L = 16, 1.0
    sim = AmrSystem(n=n, L=L, periodic=False, regrid_every=0)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    sim.set_refinement(1e30)
    sim.set_magnetic_field(4.0 * np.ones((n, n)))
    schur = pops.Strang(hyperbolic=pops.Explicit(),
                        source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=1.0, alpha=3.0))
    sim.add_equation("e1", model=_iso_model(alpha=3.0), spatial=pops.Spatial(minmod=True), time=schur)
    # The global condensed stage is installed at add_equation; a SECOND CondensedSchur block trips the
    # single-block guard right here (before any step).
    with pytest.raises(RuntimeError) as excinfo:
        sim.add_equation("e2", model=_iso_model(alpha=3.0), spatial=pops.Spatial(minmod=True),
                         time=schur)
    msg = str(excinfo.value)
    for needle in ("set_source_stage", "SINGLE-BLOCK", ">= 2 blocks"):
        assert needle in msg, "multi-block Schur refusal missing %r; got %r" % (needle, msg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
