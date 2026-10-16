"""Executable evidence: genuine PoPS fields, Kokkos kernels and Program regions."""
import numpy as np
import pops
import pytest

from examples.api_v02.algebra_checks import build_algebra_check
from tests.python.support.native_execution_context import artifact_execution_context


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("kind", (
    "elementary", "jvp", "vjp", "simultaneous_product", "nested_iteration", "budget_fallback", "budget_reject"))
def test_api_v02_algebra_native(kind, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec = build_algebra_check(kind)
    artifact = pops.compile(pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact.verify()
    runtime = pops.bind(artifact, initial_values=dict(spec.initial_values),
                       resources={"execution_context": artifact_execution_context(artifact)})
    if kind == "budget_reject":
        before = np.asarray(runtime.state_global("samples")).copy()
        with pytest.raises((ValueError, RuntimeError), match="iteration|structured|rejected"):
            pops.run(runtime, t_end=spec.t_end, max_steps=1, console=False)
        np.testing.assert_array_equal(runtime.state_global("samples"), before)
        assert runtime.time() == 0 and runtime.macro_step() == 0
    else:
        pops.run(runtime, t_end=spec.t_end, max_steps=1, console=False)
        metrics = spec.invariants(runtime)
        assert metrics["passed"], metrics
        assert runtime.time() == spec.t_end and runtime.macro_step() == 1
