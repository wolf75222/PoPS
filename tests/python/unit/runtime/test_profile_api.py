#!/usr/bin/env python3
"""Spec 5 sec.12.5 (criteria 41-44): the TYPED profiling surface.

``pops.Profile`` (Profile.Basic() / Profile.Advanced(), not profile="advanced") + the
``PerformanceSummary`` wrapper (printable __str__, to_dict / to_json, by_program_node /
by_native_brick / by_solver / by_memory) + the ``System.profile(...)`` context manager that
enables the native profiler on __enter__ and disables it on __exit__.

Two kinds of check:

* PURE shape / parsing / env -- no engine, no _pops. These run in any interpreter and exercise the
  typed objects + the report parser against the EXACT native report format
  (profiler.hpp report()). They do NOT fake the engine: they parse a literal sample of what the C++
  Profiler emits, which is the contract this wrapper is built to read.
* ENGINE -- a real native System under the context manager: asserts enable/disable, the
  off-by-default contract, and that a stepped block's report flows into a PerformanceSummary. The
  promised installed _pops / numpy stack is required; absence fails the gate and is never faked.

The heavy per-brick / scheduler / memory counters are Kokkos-gated (compiled .so step on ROMEO); the
typed views DECLARE those measures unavailable here rather than fabricating them -- asserted below.
"""
import json
import os

import pytest

from pops.runtime.profile import (PerformanceSummary, Profile, _parse_report,
                                   _Unavailable)
from pops.runtime._system import System  # ADC-545 advanced runtime seam


# A literal sample of the native report (profiler.hpp report()): two coarse phases, a per-node scope,
# and the trailing counters line. This is the format the wrapper parses -- not a fake engine.
_SAMPLE_REPORT = (
    "Profiler report (total 0.010849 s, 3 scopes)\n"
    "  step  count=2  total=0.007229s  mean=0.003614s  min=0.003575s  max=0.003654s\n"
    "  field_solve  count=1  total=0.003621s  mean=0.003621s  min=0.003621s  max=0.003621s\n"
    "  node:solve_fields1  count=1  total=0.001000s  mean=0.001000s  min=0.001000s  max=0.001000s\n"
    "counters:  steps=2  kernels=3\n"
)


# ---- (A) Profile typed level ----------------------------------------------------------------
def test_profile_basic_advanced_are_typed_objects():
    basic = Profile.Basic()
    adv = Profile.Advanced()
    assert isinstance(basic, Profile) and isinstance(adv, Profile)
    assert basic.level == "basic" and adv.level == "advanced"
    assert adv.advanced is True and basic.advanced is False
    assert Profile.Basic() == Profile.Basic() and Profile.Basic() != Profile.Advanced()
    assert repr(Profile.Basic()) == "Profile.Basic()"


def test_profile_rejects_bad_level():
    with pytest.raises(ValueError):
        Profile("turbo")


def test_profile_from_env_maps_pops_profile():
    saved = os.environ.get("POPS_PROFILE")
    try:
        os.environ.pop("POPS_PROFILE", None)
        assert Profile.from_env() is None
        assert Profile.from_env(default=Profile.Basic()) == Profile.Basic()
        os.environ["POPS_PROFILE"] = "off"
        assert Profile.from_env() is None
        os.environ["POPS_PROFILE"] = "advanced"
        assert Profile.from_env() == Profile.Advanced()
        os.environ["POPS_PROFILE"] = "basic"
        assert Profile.from_env() == Profile.Basic()
        os.environ["POPS_PROFILE"] = "1"
        assert Profile.from_env() == Profile.Basic()
    finally:
        if saved is None:
            os.environ.pop("POPS_PROFILE", None)
        else:
            os.environ["POPS_PROFILE"] = saved


# ---- (B) report parser ----------------------------------------------------------------------
def test_parse_report_extracts_scopes_and_counters():
    parsed = _parse_report(_SAMPLE_REPORT)
    assert abs(parsed["total_s"] - 0.010849) < 1e-9
    assert set(parsed["scopes"]) == {"step", "field_solve", "node:solve_fields1"}
    assert parsed["scopes"]["step"]["count"] == 2
    assert abs(parsed["scopes"]["step"]["total_s"] - 0.007229) < 1e-9
    assert parsed["counters"] == {"steps": 2, "kernels": 3}


def test_parse_empty_report_is_safe():
    parsed = _parse_report("")
    assert parsed["scopes"] == {} and parsed["counters"] == {} and parsed["total_s"] == 0.0
    assert _parse_report(None)["scopes"] == {}


def test_summary_accepts_structured_snapshot_without_text_parsing():
    snapshot = {
        "schema_version": 1,
        "enabled": False,
        "total_s": 0.012,
        "scopes": [
            {"name": "step", "count": 2, "total_s": 0.010, "mean_s": 0.005,
             "min_s": 0.004, "max_s": 0.006},
        ],
        "counters": [
            {"name": "kernels", "value": 3},
        ],
    }
    summ = PerformanceSummary(snapshot, Profile.Basic())
    assert summ.source == "snapshot"
    assert summ.raw_report == ""
    assert summ.scopes()["step"]["count"] == 2
    assert summ.counters()["kernels"] == 3
    d = summ.to_dict()
    assert d["source"] == "snapshot"
    assert d["schema_version"] == 1


# ---- (C) PerformanceSummary typed views -----------------------------------------------------
def test_summary_views_read_the_native_tables():
    summ = PerformanceSummary(_SAMPLE_REPORT, Profile.Advanced())
    # by_program_node: the node:<name> scope, bare name.
    nodes = summ.by_program_node()
    assert "solve_fields1" in nodes and nodes["solve_fields1"]["count"] == 1
    # by_solver: the coarse field_solve phase + the solve_fields node.
    solver = summ.by_solver()
    assert "field_solve" in solver and "solve_fields1" in solver
    # counters / scopes / total surfaced.
    assert summ.counters()["kernels"] == 3
    assert abs(summ.total_s() - 0.010849) < 1e-9


def test_summary_declares_unavailable_measures_honestly():
    # by_native_brick: the native runtime has no per-brick scope -> declared unavailable, NOT faked.
    summ = PerformanceSummary(_SAMPLE_REPORT, Profile.Advanced())
    brick = summ.by_native_brick()
    assert isinstance(brick, _Unavailable) and bool(brick) is False
    assert brick.available is False
    # by_memory: the sample has no scratch counters (host path) -> unavailable, not a faked 0.
    mem = summ.by_memory()
    assert isinstance(mem, _Unavailable) and bool(mem) is False


def test_summary_by_memory_reads_counters_when_present():
    report = _SAMPLE_REPORT.replace(
        "counters:  steps=2  kernels=3\n",
        "counters:  steps=2  kernels=3  scratch_allocs=4  scratch_peak_bytes=2048\n")
    mem = PerformanceSummary(report).by_memory()
    assert mem == {"scratch_allocs": 4, "scratch_peak_bytes": 2048}


def test_summary_by_amr_mpi_declares_unavailable_on_host_report():
    # The host sample has no regrid / halo / reflux scopes and no MPI counters: the AMR/MPI dimension
    # is declared unavailable (criterion 43), NOT a faked zero -- exactly like by_native_brick/by_memory.
    view = PerformanceSummary(_SAMPLE_REPORT, Profile.Advanced()).by_amr_mpi()
    assert isinstance(view, _Unavailable) and bool(view) is False
    assert view.available is False
    assert "regrid" in view.reason and "halo" in view.reason  # the reason is honest + specific


def test_summary_by_amr_mpi_surfaces_amr_scopes_and_mpi_counters():
    # A synthetic distributed-AMR report: a regrid + halo_exchange timing scope and an mpi_reductions
    # counter, built the same way the by_elliptic / by_memory tests synthesise their reports.
    report = (
        "Profiler report (total 0.020000 s, 5 scopes)\n"
        "  step  count=2  total=0.007229s  mean=0.003614s  min=0.003575s  max=0.003654s\n"
        "  field_solve  count=1  total=0.003621s  mean=0.003621s  min=0.003621s  max=0.003621s\n"
        "  regrid  count=3  total=0.001500s  mean=0.000500s  min=0.000400s  max=0.000700s\n"
        "  halo_exchange  count=8  total=0.002000s  mean=0.000250s  min=0.000200s  max=0.000300s\n"
        "  average_down  count=4  total=0.000800s  mean=0.000200s  min=0.000150s  max=0.000250s\n"
        "counters:  steps=2  kernels=3  mpi_reductions=12  reflux=4\n")
    view = PerformanceSummary(report, Profile.Advanced()).by_amr_mpi()
    assert not isinstance(view, _Unavailable) and bool(view) is True
    # Timing scopes surface as full timing dicts.
    assert "regrid" in view and view["regrid"]["count"] == 3
    assert "halo_exchange" in view and abs(view["halo_exchange"]["total_s"] - 0.002000) < 1e-9
    assert "average_down" in view
    # MPI / reflux counters surface as ints.
    assert view["mpi_reductions"] == 12 and view["reflux"] == 4
    # Unrelated scopes / counters are NOT pulled into the AMR/MPI bucket.
    assert "step" not in view and "field_solve" not in view and "kernels" not in view


def test_summary_is_printable_and_serialisable():
    summ = PerformanceSummary(_SAMPLE_REPORT, Profile.Basic())
    text = str(summ)
    assert "PerformanceSummary" in text and "step" in text and "kernels=3" in text
    # to_dict carries the level, scopes, counters, and the typed views (with availability).
    d = summ.to_dict()
    assert d["profile"] == "basic"
    assert d["counters"]["kernels"] == 3
    assert d["views"]["by_native_brick"]["available"] is False
    assert d["views"]["by_amr_mpi"]["available"] is False  # no AMR scopes in the host sample
    assert "solve_fields1" in d["views"]["by_program_node"]
    # to_json round-trips and can write to a path.
    parsed = json.loads(summ.to_json())
    assert parsed["counters"]["steps"] == 2


def test_summary_to_json_writes_path(tmp_path):
    out = tmp_path / "profile.json"
    PerformanceSummary(_SAMPLE_REPORT).to_json(str(out))
    assert out.is_file()
    assert json.loads(out.read_text())["total_s"] > 0.0


def test_empty_summary_has_no_data_message():
    summ = PerformanceSummary("")
    assert "no profiling data" in str(summ)
    assert summ.scopes() == {} and summ.counters() == {}


# ---- (D) ENGINE: the context manager over a real native System ------------------------------
def _make_stepped_system():
    """A real native System with one isothermal block, ready to step. Returns (pops, sim, np)."""
    import numpy as np

    import pops
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov

    n = 16
    sim = System(n=n, L=1.0, periodic=True)
    sim.block(
        "gas",
        pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                   transport=pops.IsothermalFlux(), source=pops.NoSource(),
                   elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
        time=pops.Explicit())
    rho = np.ones((n, n), dtype=float)
    sim.set_state("gas", np.stack([rho, 0.1 * rho, 0.0 * rho]))
    return pops, sim, np


def test_engine_exports_typed_surface():
    import pops

    assert hasattr(pops, "Profile") and hasattr(pops, "PerformanceSummary")
    assert pops.Profile.Basic().level == "basic"


def test_engine_context_manager_enables_then_disables():
    pops, sim, _ = _make_stepped_system()
    # off by default -- a plain System never enabled.
    assert sim.is_profiling() is False
    with sim.profile(pops.Profile.Basic()) as prof:
        assert sim.is_profiling() is True, "context manager enables on __enter__"
        sim.step(1e-3)
        sim.step(1e-3)
        summary_inside = prof.summary()
        assert isinstance(summary_inside, pops.PerformanceSummary)
    # disabled on __exit__.
    assert sim.is_profiling() is False, "context manager disables on __exit__"
    summary = prof.summary()
    assert isinstance(summary, pops.PerformanceSummary)
    assert summary.counters().get("steps") == 2
    assert "step" in summary.scopes()


def test_engine_off_by_default_contract():
    """A plain run (no with sim.profile()) records NOTHING: profiling stays disabled."""
    pops, sim, _ = _make_stepped_system()
    sim.step(1e-3)
    assert sim.is_profiling() is False
    summ = PerformanceSummary(sim.profile_report())
    # the native report on a never-enabled profiler carries no counters.
    assert summ.counters() == {}, "off-by-default: no heavy timers without an explicit profile()"


def test_engine_profile_rejects_non_profile_arg():
    pops, sim, _ = _make_stepped_system()
    with pytest.raises(TypeError):
        sim.profile("advanced")


def test_engine_profile_no_arg_uses_env_default():
    pops, sim, _ = _make_stepped_system()
    saved = os.environ.get("POPS_PROFILE")
    try:
        os.environ["POPS_PROFILE"] = "advanced"
        with sim.profile() as prof:
            sim.step(1e-3)
        assert prof.summary().profile.level == "advanced"
    finally:
        if saved is None:
            os.environ.pop("POPS_PROFILE", None)
        else:
            os.environ["POPS_PROFILE"] = saved
