"""Actual installed Dim=1 lake reception; NumPy is an independent oracle only."""
import os
import numpy as np
import pops
import pytest
from tests.python.unit.numerics.test_m07_hydrostatic_public_gap import example, oracle

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _check(world, callback):
    failure = b""
    if world.rank == 0:
        try:
            callback()
        except Exception as caught:
            failure = (type(caught).__name__+": "+str(caught)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    assert not failure, failure.decode()


@pytest.mark.parametrize("n", oracle.RESOLUTIONS)
@pytest.mark.parametrize("order", oracle.ORDERS)
def test_full_time_wet_lake_keeps_true_averages_and_signed_topography(
    isolated_native_cache, native_cxx, kokkos_root, n, order, tmp_path,
):
    del isolated_native_cache, native_cxx, kokkos_root
    if os.environ.get("POPS_NATIVE_DIM") != "1":
        pytest.fail("M07 requires explicit POPS_NATIVE_DIM=1 and installed rebuilt Dim=1")
    permutation = [oracle.ORDERS[0].index(name) for name in order]
    initial = np.ascontiguousarray(oracle.initial_averages(n)[permutation])
    expected, _ = oracle.trajectory(n, order=order)
    case, layout, _ = example.build_case(n, order=order)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    assert artifact.resolved_dimension == 1
    execution = pops.ExecutionContext.mpi_world(artifact)
    world = execution.communicator.handle
    runtime = pops.bind(artifact, initial_state={"lake": initial},
                        resources={"execution_context": execution})
    before = np.asarray(runtime.state_global("lake"))
    _check(world, lambda: np.testing.assert_array_equal(before.reshape(initial.shape), initial))
    report = pops.run(runtime, t_end=oracle.T_END, max_steps=5*n, console=False)
    after = np.asarray(runtime.state_global("lake"))

    def check():
        path = tmp_path / "lake.npz"
        np.savez_compressed(path, state=after.reshape(initial.shape), time=runtime.time())
        with np.load(path) as saved:
            state = saved["state"][np.argsort(permutation)]
            np.testing.assert_allclose(saved["state"], expected, rtol=0.,
                                       atol=oracle.EQUILIBRIUM_TOLERANCE)
            assert abs(float(saved["time"])-oracle.T_END) <= 2.e-14
        assert np.isfinite(state).all() and np.all(state[0] > 0.)
        assert max(np.max(np.abs(state[0]+state[2]-1)), np.max(np.abs(state[1])),
                   np.max(np.abs(state[2]-oracle.initial_averages(n)[2]))) <= oracle.EQUILIBRIUM_TOLERANCE
        assert report.accepted_steps == 5*n and report.rejected_steps == 0
        assert runtime.macro_step() == 5*n
    _check(world, check)
