"""Real Program/PoPS storage and MPI uniformity for deeply nested finite values."""
import numpy as np
import pops
import pytest
from examples.api_v02.finite_structural_depth import build_finite_structural_depth
from tests.python.support.api_v02_native import _compile, _bind
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 900


def test_nested_array_variant_product_executes_with_inactive_arms_and_restarts(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("finite_structural_depth", builder=build_finite_structural_depth,
                                     depth=160)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    actual = collective_call(world, lambda: runtime.state_global("method"))
    diagnostics = collective_call(world, runtime.program_report)
    with collective_check(world):
        assert diagnostics.diagnostics["structural.decoded"] == 7.
        if actual is not None:
            np.testing.assert_array_equal(actual, np.full_like(actual, 7.))
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"deep_structure")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    values = []
    for active in (runtime, resumed):
        collective_call(world, lambda active=active: pops.run(active, t_end=.02, max_steps=1, console=False))
        values.append(collective_call(world, lambda active=active: active.state_global("method")))
    with collective_check(world):
        if values[0] is not None:
            np.testing.assert_array_equal(values[0], np.full_like(values[0], 14.))
            np.testing.assert_array_equal(values[0], values[1])
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks", 1 if world is None else int(world.size))
