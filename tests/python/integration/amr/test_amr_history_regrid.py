#!/usr/bin/env python3
"""ADC-631 (b): multistep history with ACTIVE regrid on a 2-level AMR hierarchy.

An AB2 Program on a 2-level AMR system with ``regrid_every>0``. Two precise assertions:

  (i)  NULL regrid (the active criterion tags the full positive domain, so every scheduled rebuild
       preserves the exact full-domain fine boxes) -> the trajectory equals a no-regrid-window run
       to round-off: a layout-identical rebuild must not remap either the history ring or its lagged
       interface-flux authority (the bitwise native invariant is locked by the C++
       test_amr_history_ring.RegridRemapKeepsSlotsConsistent case);
  (ii) REAL regrid (a moving scalar-advection front tags cells) -> the run is stable (finite, coarse mass
       conserved to round-off) and after the regrids EVERY prev(k) global buffer is defined on the NEW
       layout (its flat size == the current sum_k ncomp*nf_k*nf_k) -- the layout-consistency invariant.

Native prerequisites are expressed by pytest markers and fixtures. Any compile, bind or run failure
is a hard test failure rather than a self-skip.
"""
from pathlib import Path

import numpy as np
import pops
import pops.lib.time as lt
import pytest
from pops.lib.initial import BindArray
from pops.time import FailRun, FixedDt
from tests.python.integration._final_field_program import (
    resolve_periodic_field_program,
    scalar_advection_model,
)

ROOT = Path(__file__).resolve().parents[4]
N = 16
NSTEPS = 6
DT = 2.0e-3

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _euler_model(name):
    """Scalar advection moves the tagged blob while preserving one conservative density."""
    return scalar_advection_model(name)


def _ab2_plan(model, name, *, regrid_every, native_cxx):
    def program(state, rate, fields):
        result = lt.AdamsBashforth(
            state,
            rate=rate,
            fields=fields,
            order=2,
            solve_action=FailRun(),
        )
        result.step_strategy(FixedDt(DT))
        return result

    return resolve_periodic_field_program(
        model,
        program,
        name=name,
        block_name="blk",
        target="amr_system",
        n=N,
        regrid_every=regrid_every,
        initial_profile=BindArray(),
        cxx=native_cxx,
        include=str(ROOT / "include"),
    )


def _blob(amp=0.5, w=0.12):
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + amp * np.exp(
        -((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (w * w)
    )


def _build(regrid_every, refine_thr, u0, tag, native_cxx):
    model = _euler_model("rg_blk_%s" % tag)
    plan_name = "rg_ab2_%s" % tag
    plan = _ab2_plan(
        model,
        plan_name,
        regrid_every=(NSTEPS + 1 if regrid_every == 0 else regrid_every),
        native_cxx=native_cxx,
    )
    bindings = tuple(plan.initial_condition_plan.bindings)
    thresholds = tuple(
        slot.handle for slot in plan.bind_schema.runtime_slots
        if slot.handle.local_id == "%s_refine_threshold" % plan_name
    )
    if len(bindings) != 1 or len(thresholds) != 1:
        raise RuntimeError("resolved AMR history plan lost its initial/threshold bind authority")
    artifact = pops.compile(plan)
    runtime = pops.bind(
        artifact,
        params={thresholds[0]: refine_thr},
        initial_values={
            bindings[0].subject: np.ascontiguousarray(u0[None, ...], dtype=np.float64),
        },
    )
    return runtime


def _advance(runtime, steps):
    return pops.run(
        runtime,
        t_end=float(runtime.time()) + steps * DT,
        max_steps=steps,
    )


def _coarse_density(runtime):
    return np.asarray(
        runtime.block_level_state_global("blk", 0), dtype=np.float64,
    ).reshape(N, N)


def test_null_regrid_matches_no_regrid_to_roundoff(
    native_cxx, isolated_native_cache, kokkos_root,
):
    """(i) Full-domain tagging makes each scheduled rebuild topology-null.

    The exact invariant is structural: the dynamic run completes native regrids while its public
    ``patch_boxes`` stay equal to the bootstrap boxes; the comparison run has the same bootstrap
    hierarchy but no regrid inside this six-step window.  This does not claim that an empty tag set
    preserves a frozen seed.
    """
    del isolated_native_cache, kokkos_root
    u0 = _blob(amp=0.2)
    a = _build(regrid_every=2, refine_thr=0.0, u0=u0, tag="null_a", native_cxx=native_cxx)
    b = _build(regrid_every=0, refine_thr=0.0, u0=u0, tag="null_b", native_cxx=native_cxx)
    initial_boxes = tuple(a.patch_boxes())
    dynamic_regrids_before = a.amr.explain_regrid().regrid_count
    comparison_regrids_before = b.amr.explain_regrid().regrid_count
    assert int(a.n_levels()) == 2
    assert initial_boxes == tuple(b.patch_boxes())
    _advance(a, NSTEPS)
    _advance(b, NSTEPS)
    dynamic_regrids_after = a.amr.explain_regrid().regrid_count
    comparison_regrids_after = b.amr.explain_regrid().regrid_count
    assert dynamic_regrids_after > dynamic_regrids_before
    assert comparison_regrids_after == comparison_regrids_before
    assert tuple(a.patch_boxes()) == initial_boxes == tuple(b.patch_boxes())
    da = float(np.abs(_coarse_density(a) - _coarse_density(b)).max())
    assert da < 1e-9, "null-regrid trajectory mismatch: max|d| = %.3e" % da
    ra = {h: [np.asarray(a.history_global(h, k)).ravel()
              for k in range(int(a.history_depth(h)))] for h in a.history_names()}
    assert ra
    assert all(np.all(np.isfinite(x)) for slots in ra.values() for x in slots)


def test_real_regrid_stable_and_layout_consistent(
    native_cxx, isolated_native_cache, kokkos_root,
):
    """(ii) A real regrid (dispersing blob tags cells) -> the run stays STABLE (finite) on a genuinely
    two-level hierarchy, CONSERVES the total mass to ROUND-OFF across the regrids, and every prev(k) buffer
    is defined on the NEW hierarchy (flat size == sum_k ncomp*nf_k*nf_k).

    ROUND-OFF conservation (ADC-639): the synchronous Program driver now couples fine->coarse by
    average_down THEN conservative REFLUX at the coarse-fine interface (amr_program_context.hpp::
    couple_levels + amr_program_reflux.hpp). The per-level effective flux is captured through the AB2
    Program's own linear combination (1.5 R_n - 0.5 R_{n-1}, the flux ledger + the persistent per-ring
    strip that carries R_{n-1}'s flux across steps), so the coarse cell's flux at the interface is
    corrected by exactly (fine effective flux - coarse effective flux). The total mass is therefore
    conserved to round-off on a genuinely MULTILEVEL run -- matching the native reflux -- INCLUDING across
    an in-window regrid (the deferred-rotate + slot-0 resync keeps the multistep ring consistent with the
    refluxed live state, ADC-631 x ADC-639, acceptance e). This is the tracked concession: the tolerance
    was 2e-4 (average_down-only v1) and is tightened to 1e-8 with the reflux."""
    del isolated_native_cache, kokkos_root
    u0 = _blob(amp=0.5)
    a = _build(regrid_every=2, refine_thr=1.2, u0=u0, tag="real", native_cxx=native_cxx)
    boxes_before = tuple(a.patch_boxes())
    regrids_before = a.amr.explain_regrid().regrid_count
    assert int(a.n_levels()) == 2
    m0 = a.integral("blk", levels=(0,))
    _advance(a, NSTEPS)
    assert a.amr.explain_regrid().regrid_count > regrids_before
    assert tuple(a.patch_boxes()) != boxes_before
    rho = _coarse_density(a)
    assert np.all(np.isfinite(rho))
    mass_drift = abs(a.integral("blk", levels=(0,)) - m0)
    assert mass_drift < 1e-8, "AB2 + reflux mass drift = %.2e" % mass_drift
    nlev = int(a.n_levels())
    names = list(a.history_names())
    assert names
    ok = True
    for h in names:
        ncomp = int(a.history_ncomp(h))
        expected = sum(ncomp * (N << k) * (N << k) for k in range(nlev))
        for k in range(int(a.history_depth(h))):
            buf = np.asarray(a.history_global(h, k), dtype=np.float64).ravel()
            if buf.size != expected or not np.all(np.isfinite(buf)):
                ok = False
                pytest.fail(
                    "ring %s slot %d size %d != expected %d (or non-finite)"
                    % (h, k, buf.size, expected)
                )
    assert nlev >= 2 and ok
