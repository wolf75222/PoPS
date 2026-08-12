#!/usr/bin/env python3
"""ADC-515 (Spec 6 sec.20): the explicit-family time schemes run end-to-end on a native AmrSystem.

The AMR column of the sec.20 matrix for the explicit family: a REAL ``AmrSystem`` built from native
``engine.Model(...)`` bricks (no Kokkos ``.so`` compile) steps under Explicit / SSPRK3 and an
explicit source-free Forward-Euler spatial fixture, mono- AND multi-block.  The verdict is the one
the dedicated AMR suites use -- finite state after stepping, a live fine patch (refinement is not
inert) and per-block mass conserved to ~machine (reflux + average_down). This is the green half the
refusal cells (``test_amr_refusals``) sit next to; SSPRK3 mirrors ``test_amr_ssprk3`` and
Strang/CondensedSchur lives in ``test_amr_strang_condensed_schur`` -- here we pin the explicit family
as one focused file.

Runtime: ``importorskip('pops')`` skips the whole file on a bare box (this Mac has no ``_pops``);
on the Kokkos-Serial CI runner the cells step a real engine. ``__main__`` runs pytest.
"""

import sys

import numpy as np
import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._engine_descriptors import Periodic  # noqa: E402

from pops.runtime._system import AmrSystem  # noqa: E402  (ADC-545 advanced runtime seam)
from tests.python.support.explicit_program import (  # noqa: E402
    install_forward_euler_program,
    install_ssprk2_program,
    install_ssprk3_program,
)
from tests.python.support.amr_tagging import install_prepared_threshold_union  # noqa: E402


def _scalar_charge(q, B0=1.0):
    """A scalar E x B block with the explicit periodic background ``q * (rho - 1)``."""
    return engine.Model(
        engine.Scalar(),
        engine.ExB(),
        engine.NoSource(),
        engine.BackgroundDensity(alpha=q, n0=1.0),
    )


def _bump(n, amp):
    """A smooth density bump with a zero-mean offset (periodic-Poisson solvable), refinement-tagging."""
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    r = 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    return r + (1.0 - r.mean())


def _run_amr_explicit_family(time_brick, *, multi, n=32):
    """Build + step a native AmrSystem of scalar-charge block(s) under an explicit-family @p time_brick."""
    sim = AmrSystem(n=n, L=1.0, periodicity=(True, True), regrid_every=4)
    sim.set_temporal_relations([2], [1], ["integral_only"])
    sim.add_equation(
        "ions", _scalar_charge(+1.0), spatial=engine.Spatial(minmod=True), time=time_brick
    )
    if multi:
        sim.add_equation(
            "electrons",
            _scalar_charge(-1.0),
            spatial=engine.Spatial(minmod=True),
            time=time_brick,
        )
    sim.set_poisson(bc=Periodic())
    blocks = ("ions", "electrons") if multi else ("ions",)
    install_prepared_threshold_union(sim, ((block, "n", 1.05) for block in blocks))
    sim.set_density("ions", _bump(n, 0.40))
    if multi:
        sim.set_density("electrons", _bump(n, 0.20))
    if time_brick.kind == "ssprk3":
        install_ssprk3_program(sim)
    elif time_brick.kind == "euler":
        # This source-free fixture exercises only the spatial AMR facade; temporal semantics are
        # proved through a Program-authored Case elsewhere.
        install_forward_euler_program(sim)
    else:
        install_ssprk2_program(sim)
    blocks = ("ions", "electrons") if multi else ("ions",)
    m0 = {b: sim.mass(b) for b in blocks}
    sim.advance(0.002, 10)
    return sim, m0


def _assert_finite_and_conserved(sim, m0, label):
    """Shared run verdict: finite state, a live fine patch, per-block mass conserved to ~machine."""
    for b, m_start in m0.items():
        d = np.asarray(sim.density(b))
        assert np.isfinite(d).all(), "%s: block %r state not finite" % (label, b)
        drift = abs(sim.mass(b) - m_start) / (abs(m_start) + 1.0)
        assert drift < 1e-9, "%s: block %r mass not conserved (drift=%.2e)" % (label, b, drift)
    assert sim.n_patches() >= 1, "%s: no live fine patch (refinement inert)" % label


@pytest.mark.parametrize("multi", [False, True], ids=["mono", "multi"])
def test_amr_explicit_runs_finite_and_conserved(multi):
    """AMR x Explicit x {mono, multi}: native run, finite, live patch, per-block mass conserved."""
    sim, m0 = _run_amr_explicit_family(engine.Explicit(), multi=multi)
    _assert_finite_and_conserved(sim, m0, "AMR/Explicit/%s" % ("multi" if multi else "mono"))


@pytest.mark.parametrize("multi", [False, True], ids=["mono", "multi"])
def test_amr_ssprk3_runs_finite_and_conserved(multi):
    """AMR x SSPRK3 x {mono, multi}: native ``Explicit(ssprk3=True)`` (kind='ssprk3') runs + conserves.

    The dedicated ``test_amr_ssprk3`` proves SSPRK3 route exclusivity; this cell pins it as part of
    the sec.20 explicit-family column so the matrix has the SSPRK3 x block-count entries.
    """
    sim, m0 = _run_amr_explicit_family(engine.Explicit(ssprk3=True), multi=multi)
    _assert_finite_and_conserved(sim, m0, "AMR/SSPRK3/%s" % ("multi" if multi else "mono"))


def test_amr_source_free_forward_euler_fixture_runs_finite_and_conserved():
    """AMR x source-free Forward Euler x mono: an explicit spatial fixture runs and conserves.

    The fixture intentionally has ``NoSource`` and does not qualify any higher-order or implicit
    temporal semantics. Dedicated Program-authored suites cover those paths.
    """
    sim, m0 = _run_amr_explicit_family(engine.Explicit(method="euler"), multi=False)
    _assert_finite_and_conserved(sim, m0, "AMR/source-free-FE/mono")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
