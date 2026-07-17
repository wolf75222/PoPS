#!/usr/bin/env python3
"""ADC-639: conservative reflux for a whole-system compiled Program on a genuinely two-level AMR
hierarchy.

The synchronous per-level Program driver (AmrProgramContext) advances every level with the same dt, then
couples fine->coarse by average_down THEN conservative REFLUX at the coarse-fine interface. The per-level
effective flux is captured through the Program's OWN linear combination (the flux ledger,
amr_program_context.hpp) and routed through the native route_reflux at level sync (amr_program_reflux.hpp).
So on a genuinely MULTILEVEL run the total conserved quantity is conserved across the C/F interface to
ROUND-OFF, matching the native reflux -- while the coarse-only / flat Program stays bit-identical (locked
by test_amr_program_parity).

Acceptances (design-639 section 5):
  (a) a 2-level SSPRK2 Program conserves the total mass to < 1e-8 over several steps including a real
      regrid;
  (c) a 2-level MIDPOINT (RK2) Program -- a DIFFERENT combine through the same seam -- also conserves to
      < 1e-8 (validates that the ledger tracks the Program's ACTUAL stage weights Feff = F1, not a
      hard-coded RK), and its trajectory DIFFERS from SSPRK2.

Needs a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) to build the .so; the compiled-.so dlopen + the
per-level run is validatable on Kokkos CPU (Serial/OpenMP) locally. Native prerequisites are expressed
by pytest markers and fixtures; any compile, bind or run error is a hard failure.
"""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.lib.initial import BindArray
from pops.time import FailRun, FixedDt
from pops.time._methods.tableau import RungeKuttaTableau
from tests.python.integration._final_field_program import (
    resolve_periodic_field_program,
    scalar_burgers_model,
)

ROOT = Path(__file__).resolve().parents[4]
N = 16
NSTEPS = 6
DT = 1.0e-3

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _euler_model(name):
    """Nonlinear scalar transport discriminates RK2 schemes at the reflux interface."""
    return scalar_burgers_model(name)


def _ssprk2_program(model, name, native_cxx):
    """Canonical SSPRK2 (Heun): U1 = U + dt R(U); U <<= 0.5 U + 0.5 (U1 + dt R(U1))."""
    def program(state, rate, fields):
        result = libtime.SSPRK2(
            state,
            rate=rate,
            fields=fields,
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
        regrid_every=2,
        initial_profile=BindArray(),
        cxx=native_cxx,
        include=str(ROOT / "include"),
    )


def _midpoint_program(model, name, native_cxx):
    """Midpoint RK2: U1 = U + 0.5 dt R(U); U <<= U + dt R(U1). Effective flux Feff = F1 (the 2nd stage
    only) -- proves the ledger tracks the Program's actual weights, not a hard-coded scheme."""
    midpoint = RungeKuttaTableau(
        A=[[], [Fraction(1, 2)]], b=[0, 1], c=[0, Fraction(1, 2)], name="midpoint")
    def program(state, rate, fields):
        result = libtime.RungeKutta(
            state,
            rate=rate,
            fields=fields,
            tableau=midpoint,
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
        regrid_every=2,
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


def _run(program_fn, tag, native_cxx, refine_thr=1.2, u0=None, nsteps=NSTEPS):
    """Bind ``program_fn`` on a genuine two-level public AMR runtime and advance it."""
    if u0 is None:
        u0 = _blob(amp=0.5)
    model = _euler_model("rfx_blk_%s" % tag)
    plan_name = "rfx_prog_%s" % tag
    plan = program_fn(model, plan_name, native_cxx)
    bindings = tuple(plan.initial_condition_plan.bindings)
    thresholds = tuple(
        slot.handle for slot in plan.bind_schema.runtime_slots
        if slot.handle.local_id == "%s_refine_threshold" % plan_name
    )
    if len(bindings) != 1 or len(thresholds) != 1:
        raise RuntimeError("resolved reflux plan lost its initial/threshold bind authority")
    artifact = pops.compile(plan)
    runtime = pops.bind(
        artifact,
        params={thresholds[0]: refine_thr},
        initial_values={
            bindings[0].subject: np.ascontiguousarray(u0[None, ...], dtype=np.float64),
        },
    )
    boxes_before = tuple(runtime.patch_boxes())
    regrids_before = runtime.amr.explain_regrid().regrid_count
    assert int(runtime.n_levels()) == 2
    m0 = runtime.integral("blk", levels=(0,))
    pops.run(
        runtime,
        t_end=float(runtime.time()) + nsteps * DT,
        max_steps=nsteps,
    )
    assert runtime.amr.explain_regrid().regrid_count > regrids_before
    assert tuple(runtime.patch_boxes()) != boxes_before
    rho = np.asarray(
        runtime.block_level_state_global("blk", 0), dtype=np.float64,
    ).reshape(N, N)
    return m0, runtime.integral("blk", levels=(0,)), rho


def test_multilevel_ssprk2_conserves_to_roundoff(
    native_cxx, isolated_native_cache, kokkos_root,
):
    """(a) a genuinely 2-level SSPRK2 Program conserves the total mass to < 1e-8 over 6 steps including a
    real regrid -- the conservative reflux at the C/F interface, matching the native path."""
    del isolated_native_cache, kokkos_root
    m0, mf, rho = _run(_ssprk2_program, "ss", native_cxx)
    assert np.all(np.isfinite(rho)) and float(rho.min()) > 0.0
    assert abs(mf - m0) < 1e-8, "SSPRK2 + reflux mass drift = %.3e" % abs(mf - m0)


def test_multilevel_midpoint_conserves_and_differs(
    native_cxx, isolated_native_cache, kokkos_root,
):
    """(c) a 2-level MIDPOINT Program conserves to < 1e-8 (validates Feff = F1: the ledger tracks the
    Program's actual stage weights, not a hard-coded RK), and its trajectory DIFFERS from SSPRK2."""
    del isolated_native_cache, kokkos_root
    m0, mf, mid_rho = _run(_midpoint_program, "mid", native_cxx)
    assert np.all(np.isfinite(mid_rho))
    assert abs(mf - m0) < 1e-8, "midpoint + reflux mass drift = %.3e" % abs(mf - m0)
    _, _, ss_rho = _run(_ssprk2_program, "mid_ss", native_cxx)
    diff = float(np.abs(mid_rho - ss_rho).max())
    assert diff > 1e-12, "midpoint and SSPRK2 trajectories match: max|diff| = %.3e" % diff
