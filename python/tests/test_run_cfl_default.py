"""Spec 5 C7 (#379 follow-up): System.run(cfl=None) defaults to the pinned program cadence CFL.

PR #379 wired CompiledProgramCadence(cfl=X) -> System._program_cadence_cfl (via _install_cadence) so that a
bare sim.run(t_end) advances at X instead of silently falling back to the 0.4 default. The pin was
host-tested (test_unified_install.test_install_cadence_routing), but the run() DEFAULTING itself --
the actual consumer of the pin -- had no host test (the CHANGELOG claimed one). This file closes that
gap: it pins the cadence CFL and asserts run(cfl=None) forwards the pinned value to step_cfl, while
run(cfl=<value>) overrides it.

run() is pure-Python sugar (`while time() < t_end: step_cfl(cfl)`), so the assertion needs no built
engine: step_cfl / time are delegated to _pops and are shadowed here by instance attributes (which
take precedence over the System __getattr__ delegation). Constructing the System and CompiledProgramCadence
still needs pops importable, so the test self-skips cleanly when _pops is absent.
"""
import sys

try:
    import pops
    from pops import time as adctime
    from pops.runtime._compiled_cadence import CompiledProgramCadence
except Exception as exc:  # noqa: BLE001
    print("skip test_run_cfl_default (pops unavailable: %s)" % exc)
    sys.exit(0)


def _instrument(sim):
    """Shadow step_cfl / time on @p sim: capture the cfl run() passes and stop the loop after one step.

    Instance attributes take precedence over the System __getattr__ delegation to _pops, so run()
    calls these instead of the engine. time() returns 0.0 until the first step, then a large value so
    `while time() < t_end` exits after exactly one step_cfl call.
    """
    captured = {"cfl": None, "calls": 0}

    def fake_step_cfl(cfl):
        captured["cfl"] = cfl
        captured["calls"] += 1

    def fake_time():
        return 0.0 if captured["calls"] == 0 else 1.0e30

    sim.step_cfl = fake_step_cfl
    sim.time = fake_time
    return captured


def _pin_cadence_cfl(sim, value):
    """Pin the program-cadence CFL on @p sim, mirroring test_install_cadence_routing.

    Uses the _install_cadence path (CompiledProgramCadence(cfl=value)) when the _pops set_program_cadence
    binding is present; otherwise sets the documented pin attribute directly. Both land on the SAME
    System._program_cadence_cfl that run() reads -- the binding only gates the substeps/stride
    orchestration, not the cfl pin.
    """
    if hasattr(sim._s, "set_program_cadence"):
        sim._install_cadence(CompiledProgramCadence(substeps=1, stride=1, cfl=value))
    else:
        sim._program_cadence_cfl = value if value == "program" else float(value)
    assert sim._program_cadence_cfl == value, \
        "cadence cfl was not pinned (got %r)" % sim._program_cadence_cfl


def test_run_cfl_none_uses_pinned_cadence():
    """run(cfl=None) forwards the pinned program-cadence CFL to step_cfl (not the 0.4 fallback)."""
    sim = pops.System(n=8, L=1.0, periodic=True)
    _pin_cadence_cfl(sim, 0.5)
    captured = _instrument(sim)
    steps = sim.run(t_end=0.01, cfl=None)
    assert steps == 1, "run should take exactly one instrumented step (got %r)" % steps
    assert captured["cfl"] == 0.5, \
        "run(cfl=None) should default to the pinned cadence cfl 0.5 (got %r)" % captured["cfl"]


def test_run_explicit_cfl_overrides_pinned_cadence():
    """An explicit run(cfl=0.9) overrides the pinned cadence CFL."""
    sim = pops.System(n=8, L=1.0, periodic=True)
    _pin_cadence_cfl(sim, 0.5)
    captured = _instrument(sim)
    steps = sim.run(t_end=0.01, cfl=0.9)
    assert steps == 1, "run should take exactly one instrumented step (got %r)" % steps
    assert captured["cfl"] == 0.9, \
        "an explicit run(cfl=0.9) should override the pinned cadence cfl (got %r)" % captured["cfl"]


def test_run_cfl_none_without_cadence_uses_default():
    """With no cadence pinned, run(cfl=None) falls back to the historical 0.4 default."""
    sim = pops.System(n=8, L=1.0, periodic=True)
    assert sim._program_cadence_cfl is None, "no cadence cfl should be pinned on a fresh System"
    captured = _instrument(sim)
    sim.run(t_end=0.01, cfl=None)
    assert captured["cfl"] == 0.4, \
        "run(cfl=None) with no cadence should use the 0.4 default (got %r)" % captured["cfl"]


def test_run_cfl_program_uses_dt_bound_factor_one():
    """cadence cfl='program' routes run(cfl=None) to step_cfl(1.0)."""
    sim = pops.System(n=8, L=1.0, periodic=True)
    _pin_cadence_cfl(sim, "program")
    captured = _instrument(sim)
    steps = sim.run(t_end=0.01, cfl=None)
    assert steps == 1, "run should take exactly one instrumented step (got %r)" % steps
    assert captured["cfl"] == 1.0, \
        "run(cfl=None) with cfl='program' should call step_cfl(1.0), got %r" % captured["cfl"]


def main():
    test_run_cfl_none_uses_pinned_cadence()
    test_run_explicit_cfl_overrides_pinned_cadence()
    test_run_cfl_none_without_cadence_uses_default()
    test_run_cfl_program_uses_dt_bound_factor_one()
    print("OK  test_run_cfl_default: run(cfl=None) defaults to the pinned cadence cfl, "
          "an explicit cfl overrides, no cadence -> 0.4")


if __name__ == "__main__":
    main()
