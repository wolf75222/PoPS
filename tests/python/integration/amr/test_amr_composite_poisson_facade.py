"""ADC-645: the composite-FAC AMR Poisson FIELD solve is facade-reachable.

Python-facade mirror of tests/cpp/integration/amr/test_amr_composite_poisson.cpp: the opt-in
``AmrSystem.set_poisson(composite=True, fac_*)`` route (lowered from
``GeometricMG(amr_composite=CompositeFAC(...))`` by pops.bind) replaces the Option A coarse solve +
gradient injection with the composite FAC elliptic solve on the single-block coupler.

Pins:
  (1) default OFF is byte-identical (composite=False leaves the Option A trajectory unchanged --
      np.array_equal against a never-configured system);
  (2) composite ON actually changes the solve (the fine aux gradient differs from Option A);
  (3) out-of-scope hierarchies refuse LOUD at build (multi-block; distributed coarse);
  (4) the descriptor lowering (GeometricMG(amr_composite=...) -> set_poisson kwargs) is exact.

Kokkos-gated runtime (self-skips without _pops); descriptor tier is pure Python.
"""
import numpy as np
import pytest

pops = pytest.importorskip("pops")
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._system import AmrSystem  # noqa: E402


def _model():
    return engine.Model(state=engine.Scalar(), transport=engine.ExB(B0=1.0),
                      source=engine.NoSource(), elliptic=engine.BackgroundDensity(alpha=1.0, n0=0.0))


def _sim(n=32, composite=None, **fac):
    sim = AmrSystem(n=n, L=1.0, periodic=True)
    sim.block("ne", model=_model(), spatial=engine.Spatial(minmod=True), time=engine.Explicit())
    # A REAL 2-level hierarchy: refine over the compact gaussian bump (one interior mono-box fine
    # patch after Berger-Rigoutsos clustering) -- the coupler's composite scope.
    sim.set_refinement(0.5)
    if composite is not None:
        sim.set_poisson(composite=composite, **fac)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = np.exp(-80.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    sim.set_density("ne", rho)
    return sim


def _run(sim, nsteps=3, dt=1e-3):
    for _ in range(nsteps):
        sim.step(dt)
    return np.array(sim.density("ne"), copy=True)


def test_composite_off_is_byte_identical():
    """(1) composite=False (explicit) == never-configured (Option A), np.array_equal."""
    base = _run(_sim())
    off = _run(_sim(composite=False))
    assert np.array_equal(base, off)


def test_composite_on_changes_the_field_solve():
    """(2) the composite FAC field solve actually replaces Option A (trajectory differs)."""
    base = _run(_sim())
    comp = _run(_sim(composite=True))
    assert np.all(np.isfinite(comp))
    assert not np.array_equal(base, comp), "composite=True must reach the coupler field solve"


def test_composite_out_of_scope_refuses_loud():
    """(3) multi-block refuses at build (the composite solve lives on the single-block coupler)."""
    sim = AmrSystem(n=16, L=1.0, periodic=True)
    sim.block("a", model=_model(), spatial=engine.Spatial(minmod=True), time=engine.Explicit())
    sim.block("b", model=_model(), spatial=engine.Spatial(minmod=True), time=engine.Explicit())
    sim.set_poisson(composite=True)
    sim.set_density("a", np.ones((16, 16)))
    with pytest.raises(RuntimeError, match="single-block"):
        sim.step(1e-3)  # lazy build -> the refusal fires here


def test_composite_fac_knobs_refuse_out_of_domain():
    sim = AmrSystem(n=16, L=1.0, periodic=True)
    sim.block("ne", model=_model(), spatial=engine.Spatial(minmod=True), time=engine.Explicit())
    with pytest.raises((RuntimeError, ValueError)):
        sim.set_poisson(composite=True, fac_rel_tol=2.0)
    with pytest.raises((RuntimeError, ValueError)):
        sim.set_poisson(composite=True, fac_abs_tol=-1.0)
    with pytest.raises((RuntimeError, ValueError)):
        sim.set_poisson(composite=True, fac_coarse_abs_tol=-1.0)
    with pytest.raises((RuntimeError, ValueError)):
        sim.set_poisson(composite=True, fac_max_iters=-1)


def test_descriptor_lowering_matches_set_poisson_kwargs():
    """(4) GeometricMG(amr_composite=CompositeFAC(...)) lowers to the exact set_poisson kwargs."""
    from pops.solvers.elliptic import GeometricMG
    from pops.solvers.options import CompositeFAC
    g = GeometricMG(
        amr_composite=CompositeFAC(max_iters=10, rel_tol=1e-8, abs_tol=1e-14, verbose=True))
    kw = g.amr_composite.set_poisson_kwargs()
    assert kw == {"composite": True, "fac_max_iters": 10, "fac_fine_sweeps": 0,
                  "fac_rel_tol": 1e-8, "fac_abs_tol": 1e-14,
                  "fac_coarse_rel_tol": 0.0, "fac_coarse_abs_tol": 0.0,
                  "fac_coarse_cycles": 0, "fac_verbose": True}
    # Default GeometricMG(): no composite lowering at all (byte-identical Option A).
    assert GeometricMG().amr_composite is None


def main():
    test_descriptor_lowering_matches_set_poisson_kwargs()
    test_composite_off_is_byte_identical()
    test_composite_on_changes_the_field_solve()
    test_composite_out_of_scope_refuses_loud()
    test_composite_fac_knobs_refuse_out_of_domain()
    print("OK  ADC-645 composite-FAC facade")


if __name__ == "__main__":
    main()
