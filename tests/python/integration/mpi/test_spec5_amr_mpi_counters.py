"""Spec 5 (sec.12.5, ADC-479 criterion 43): the AMR / MPI profiling counters are REAL.

The multi-block AMR runtime (the private engine view wired into ``AmrSystem`` when >= 2 blocks are
added) times its non-numeric phases into the facade-owned ``pops::runtime::program::Profiler`` --
``regrid`` (rebuild the patch hierarchy), ``fill_boundary`` (the coarse aux / phi ghost halo
exchange) and ``average_down`` (restrict fine onto coarse) -- plus integer counters (``regrid`` /
``fill_boundary`` per-run counts; under MPI np>1 also ``mpi_reductions`` / ``mpi_messages``). Before
this change NO C++ path emitted those scopes, so :meth:`PerformanceSummary.by_amr_mpi` always returned
the honest "unavailable" sentinel. This test builds a SMALL native multi-block ``AmrSystem`` (native
bricks, no DSL compile -- the real engine), enables profiling, runs enough macro-steps that a regrid
fires (``regrid_every=1`` + an energy bump so the union tags refine), then asserts:

  * ``profile_report()`` now contains the ``regrid`` / ``fill_boundary`` / ``average_down`` scopes
    with count > 0 ;
  * the typed ``PerformanceSummary.by_amr_mpi()`` view SURFACES them (no longer ``_Unavailable``).

The ``mpi_reductions`` / ``mpi_messages`` counters are np>1 only (the serial all_reduce / fill is an
identity that issues no collective / point-to-point round) -> validated on ROMEO, not on this
single-rank Mac ; here we only assert they are 0 (the honest serial value), never fabricated.

Pre-rebuild (an ``_pops`` that predates ``AmrSystem.enable_profiling`` or the engine scopes) the test
SKIPS cleanly: the binding / scope is simply absent and the typed view stays unavailable.
"""
from tests.python.support.requirements import require_native_or_skip
import sys

import numpy as np
import pytest
from pops.runtime._system import AmrSystem  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._engine_descriptors import Periodic  # noqa: E402

from pops.runtime._profile import PerformanceSummary, Profile  # noqa: E402


def _comp():
    """A pure compressible-Euler block (4 vars: rho, rho_u, rho_v, E), trivial background elliptic.

    alpha=0 -> Poisson RHS is zero (no periodic solvability constraint); the regrid tags on the
    conservative field. Native bricks only -- no DSL compiler required.
    """
    return engine.Model(state=engine.FluidState("compressible", gamma=1.4),
                      transport=engine.CompressibleFlux(), source=engine.NoSource(),
                      elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0))


def _state(n, rho, energy, bump_comp, bump_val, lo, hi):
    """Conservative state (rho, rho_u, rho_v, E), shape (4, n, n); a bump in [lo, hi)^2."""
    comps = [np.full((n, n), rho), np.zeros((n, n)), np.zeros((n, n)), np.full((n, n), energy)]
    comps[bump_comp][lo:hi, lo:hi] = bump_val
    return np.stack(comps)


def _built_multiblock(n=64, regrid_every=1):
    """A small built MULTI-block AmrSystem (>= 2 blocks -> AmrRuntime engine) with a refining bump.

    Two Euler blocks on the shared hierarchy; block 0 carries an energy bump in the bottom-left
    corner and the refinement tags on energy (role) so the union regrid forms a real fine patch.
    """
    sim = AmrSystem(n=n, L=1.0, periodicity=(True, True), regrid_every=regrid_every)
    sim.set_temporal_relations([2], [1], ["integral_only"])
    sim.add_equation("gas0", _comp(), time=engine.Explicit())
    sim.add_equation("gas1", _comp(), time=engine.Explicit())
    sim.set_poisson(bc=Periodic())
    sim.set_refinement(6.0, role="energy")  # tag where E > 6 -> the bottom-left bump refines
    sim.set_conservative_state("gas0", _state(n, 1.0, 2.0, bump_comp=3, bump_val=12.0, lo=4, hi=20))
    sim.set_conservative_state("gas1", _state(n, 1.0, 2.0, 0, 1.0, 0, 0))  # uniform background
    return sim


def _has_amr_profiling():
    """True iff this _pops exposes the AmrSystem profiling bindings (skip-guard, pre-rebuild)."""
    sim = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
    return all(hasattr(sim, m) for m in
               ("enable_profiling", "disable_profiling", "reset_profiling", "profile_report"))


@pytest.mark.skipif(not _has_amr_profiling(),
                    reason="this _pops predates AmrSystem profiling (criterion 43): rebuild required")
def test_amr_phase_scopes_emitted():
    """The engine emits regrid / fill_boundary / average_down scopes with count > 0 under profiling."""
    sim = _built_multiblock()
    sim.enable_profiling()
    # Enough macro-steps that the regrid_every=1 cadence fires past the first step (macro_step_ > 0).
    for _ in range(4):
        sim.step(1e-3)
    report = sim.profile_report()
    sim.disable_profiling()

    summary = PerformanceSummary(report, Profile.Basic())
    scopes = summary.scopes()

    # A regrid must have actually formed a fine patch (precondition: the engine ran the regrid path).
    assert any(b[0] >= 1 for b in sim.patch_boxes()), \
        "no fine patch formed -- the regrid did not fire, test precondition unmet"

    # The three AMR phase scopes are present with a nonzero count.
    for name in ("average_down", "fill_boundary", "regrid"):
        assert name in scopes, "missing AMR phase scope %r in report:\n%s" % (name, report)
        assert scopes[name]["count"] > 0, "AMR phase scope %r has count 0" % name

    # The per-run counters moved too (regrid completed at least once; fill_boundary every solve).
    counters = summary.counters()
    assert counters.get("regrid", 0) > 0, "regrid counter did not move"
    assert counters.get("fill_boundary", 0) > 0, "fill_boundary counter did not move"

    # mpi_reductions / mpi_messages are np>1 ONLY -> 0 on this single-rank run (honest, never faked).
    assert counters.get("mpi_reductions", 0) == 0, "serial run must issue no MPI reduction"
    assert counters.get("mpi_messages", 0) == 0, "serial run must issue no MPI message"


@pytest.mark.skipif(not _has_amr_profiling(),
                    reason="this _pops predates AmrSystem profiling (criterion 43): rebuild required")
def test_by_amr_mpi_now_available():
    """PerformanceSummary.by_amr_mpi() surfaces the phases (no longer the unavailable sentinel)."""
    sim = _built_multiblock()
    from pops.runtime._profile import Profile

    with sim.profile(Profile.Basic()) as prof:
        for _ in range(4):
            sim.step(1e-3)
    view = prof.summary().by_amr_mpi()

    # The view is a real dict now (not the _Unavailable sentinel); bool(_Unavailable) is False.
    assert view, "by_amr_mpi() is still unavailable -- the engine emitted no AMR scope"
    # It carries the AMR phase timings (regrid / fill_boundary / average_down) as timing dicts.
    for name in ("average_down", "fill_boundary", "regrid"):
        assert name in view, "by_amr_mpi() is missing %r; got keys %r" % (name, sorted(view))
        assert isinstance(view[name], dict) and view[name].get("count", 0) > 0, \
            "by_amr_mpi()[%r] is not a populated timing entry" % name


def test_serial_no_amr_run_stays_unavailable():
    """A System (no AMR engine) keeps by_amr_mpi() unavailable -- the honest no-AMR path is intact."""
    # A plain non-AMR run emits no AMR scope; the typed view must still declare itself unavailable
    # rather than fabricate a zero (regression guard for the host / non-AMR contract).
    summary = PerformanceSummary("Profiler report (total 0.0 s, 1 scopes)\n  step  count=1  "
                                 "total=0.0s  mean=0.0s  min=0.0s  max=0.0s\n", Profile.Basic())
    assert not summary.by_amr_mpi(), "non-AMR report must keep by_amr_mpi() unavailable"


def main():
    """__main__ guard: run the assertions directly (CI auto-discovers + runs tests/python/**/*.py)."""
    fails = 0
    if not _has_amr_profiling():
        require_native_or_skip('skip  test_spec5_amr_mpi_counters : _pops predates AmrSystem profiling (rebuild)')
        return 0
    for fn in (test_amr_phase_scopes_emitted, test_by_amr_mpi_now_available,
               test_serial_no_amr_run_stays_unavailable):
        try:
            fn()
            print("  [OK ] %s" % fn.__name__)
        except AssertionError as exc:
            print("  [XX ] %s : %s" % (fn.__name__, exc))
            fails += 1
    if fails == 0:
        print("test_spec5_amr_mpi_counters : OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
