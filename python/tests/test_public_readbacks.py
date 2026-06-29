import pytest

from pops.runtime.profile import PerformanceSummary, Profile
from pops.runtime.amr_system import AmrSystem
from pops.runtime.system import System


class _FakeNativeSystem:
    def __init__(self):
        self.calls = []

    def get_state(self, name):
        return ("local-state", name)

    def state_global(self, name):
        return ("global-state", name)

    def n_vars(self, name):
        if name != "plasma":
            raise RuntimeError("unknown block")
        return 3

    def solve_fields(self):
        self.calls.append("solve_fields")

    def aux_component(self, comp):
        return ("aux", comp)

    def program_diagnostics(self):
        return {"mass": 3.0, "rho_min": 0.5}

    def profile_report(self):
        return "\n".join((
            "Profiler report (total 0.002000 s, 1 scopes)",
            "  step  count=2  total=0.002000s  mean=0.001000s  min=0.001000s  max=0.001000s",
            "counters:  steps=2  kernels=4",
        ))


def _system_with_fake_native():
    sim = object.__new__(System)
    sim._s = _FakeNativeSystem()
    return sim


def _amr_system_with_fake_native():
    sim = object.__new__(AmrSystem)
    sim._s = _FakeNativeSystem()
    return sim


def test_public_state_readback_routes_to_native_runtime():
    sim = _system_with_fake_native()

    assert sim.get_state("plasma") == ("local-state", "plasma")
    assert sim.get_state("plasma", global_=True) == ("global-state", "plasma")

    with pytest.raises(TypeError):
        sim.get_state(object())


def test_public_current_fields_readback_uses_canonical_aux_components():
    sim = _system_with_fake_native()

    assert sim.get_current_fields("plasma") == {
        "phi": ("aux", 0),
        "grad_x": ("aux", 1),
        "grad_y": ("aux", 2),
    }
    assert sim._s.calls == []

    sim.get_current_fields("plasma", refresh=True)
    assert sim._s.calls == ["solve_fields"]

    with pytest.raises(TypeError):
        sim.get_current_fields(object())


def test_public_recorded_scalars_and_profile_summary_are_structured():
    sim = _system_with_fake_native()

    assert sim.get_recorded_scalars() == {"mass": 3.0, "rho_min": 0.5}

    summary = sim.profile_summary(Profile.Advanced())
    assert isinstance(summary, PerformanceSummary)
    assert summary.profile == Profile.Advanced()
    assert summary.scopes()["step"]["count"] == 2
    assert summary.to_dict()["counters"]["kernels"] == 4


def test_private_runtime_readback_mutation_and_rhs_stay_hidden():
    sim = _system_with_fake_native()

    with pytest.raises(AttributeError):
        getattr(sim, "eval_rhs")
    with pytest.raises(AttributeError):
        getattr(sim, "set_state")


def test_amr_public_recorded_scalars_and_profile_summary_are_structured():
    sim = _amr_system_with_fake_native()

    assert sim.get_recorded_scalars() == {"mass": 3.0, "rho_min": 0.5}

    summary = sim.profile_summary(Profile.Basic())
    assert isinstance(summary, PerformanceSummary)
    assert summary.profile == Profile.Basic()
    assert summary.counters()["steps"] == 2
