"""Small regression gates for the Spec-corrective public contract.

These tests are intentionally host-only: they validate API shape and early
validation before any Kokkos/C++ build is needed.
"""

import pytest

import pops
from pops import model
from pops.math import unknown
from pops.runtime._system_unified_install import _SystemUnifiedInstall


def test_equation_cannot_be_used_as_bool():
    eq = unknown("x") == 1.0
    with pytest.raises(TypeError, match="PoPS Equation cannot be used as a Python bool\\."):
        bool(eq)


def test_compile_problem_program_is_public_argument_and_not_mixed_with_time():
    with pytest.raises(TypeError, match="pass program="):
        pops.compile_problem(model=model.Module("m"), program=object(), time=object())

    with pytest.raises(ValueError, match="program must be"):
        pops.compile_problem(model=model.Module("m"))


def test_compiled_install_rejects_per_instance_time_policy_before_runtime():
    class FakeCompiled:
        so_path = "/tmp/not-loaded-problem.so"
        model = model.Module("m")

    class FakeSystem(_SystemUnifiedInstall):
        def _validate_install_arguments(self, compiled, instances, params, aux, solvers):
            return None

    with pytest.raises(TypeError, match=r"instances\['plasma'\]\['time'\]"):
        FakeSystem()._install_compiled(
            FakeCompiled(),
            instances={"plasma": {"time": object()}},
        )


def test_state_space_rejects_duplicate_components_and_is_immutable():
    with pytest.raises(ValueError, match="duplicate component"):
        model.StateSpace("U", ("rho", "rho"))

    state = model.StateSpace("U", ("rho", "mx"), roles={"rho": "Density"})
    assert state.components == ("rho", "mx")
    with pytest.raises(AttributeError, match="immutable"):
        state.components = ("rho",)
    with pytest.raises(TypeError):
        state.roles["rho"] = "Other"


def test_module_hash_rejects_unstructured_body_repr_fallback():
    class OpaqueBody:
        __slots__ = ()

    mod = model.Module("m")
    U = mod.state_space("U", ("rho",))
    mod.operator(name="opaque", signature=(U,) >> model.Rate(U),
                 kind="local_rate", expr=OpaqueBody())

    with pytest.raises(TypeError, match="repr\\(\\)"):
        mod.module_hash()
