"""Installed genuine Dim=1 axial B.1 reception; never substitutes Dim=2."""
import os

import numpy as np
import pops
import pytest

from tests.python.unit.moments.test_m15_axial_reception import _load, oracle

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, 1/800


def _compile(order):
    if os.environ.get("POPS_NATIVE_DIM") != "1":
        pytest.fail("M15 requires explicit POPS_NATIVE_DIM=1 and rebuilt installed native Dim=1")
    example = _load("api040_m15_hyqmom_axial_b1.py")
    case, layout, _ = example.build_case(N, order=order, dt=DT)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    assert artifact.resolved_dimension == 1
    execution = pops.ExecutionContext.mpi_world(artifact)
    return artifact, execution, execution.communicator.handle


def _check(world, callback):
    error = b""
    if world.rank == 0:
        try:
            callback()
        except Exception as caught:
            error = (type(caught).__name__+": "+str(caught)).encode()
    error = world.broadcast_bytes(error, root=0)
    assert not error, error.decode()


@pytest.mark.parametrize("order", oracle.ORDERS)
def test_axial_b1_native_hll_two_steps_match_independent_oracle(
    isolated_native_cache, native_cxx, kokkos_root, order,
):
    del isolated_native_cache, native_cxx, kokkos_root
    artifact, execution, world = _compile(order)
    initial = np.ascontiguousarray(oracle.initial_averages(N)[list(order)])
    runtime = pops.bind(artifact, initial_state={"moments": initial},
                        resources={"execution_context": execution})
    before = np.asarray(runtime.state_global("moments"))
    report = pops.run(runtime, t_end=2*DT, max_steps=2, console=False)
    after = np.asarray(runtime.state_global("moments"))
    expected = oracle.trajectory(N, steps=2)[0][list(order)]

    def check():
        np.testing.assert_array_equal(before.reshape(initial.shape), initial)
        np.testing.assert_allclose(after.reshape(initial.shape), expected, rtol=0, atol=3.e-11)
        oracle.require_admissible(after.reshape(initial.shape)[np.argsort(order)])
        assert report.accepted_steps == 2 and report.rejected_steps == 0
        assert runtime.macro_step() == 2 and abs(runtime.time()-2*DT) < 2.e-14
    _check(world, check)


@pytest.mark.parametrize("label", tuple(oracle.invalid_states(N)))
def test_axial_b1_inadmissible_initial_state_is_refused_without_publication(
    isolated_native_cache, native_cxx, kokkos_root, label,
):
    del isolated_native_cache, native_cxx, kokkos_root
    artifact, execution, world = _compile(oracle.ORDERS[1])
    control = pops.bind(artifact,
                        initial_state={"moments": np.ascontiguousarray(
                            oracle.initial_averages(N)[list(oracle.ORDERS[1])])},
                        resources={"execution_context": execution})
    control_report = pops.run(control, t_end=DT, max_steps=1, console=False)
    control_counts = world.allgather_bytes(str(control_report.accepted_steps).encode())
    assert all(count == b"1" for count in control_counts), control_counts
    initial = np.ascontiguousarray(oracle.invalid_states(N)[label][list(oracle.ORDERS[1])])
    runtime = None
    try:
        runtime = pops.bind(artifact, initial_state={"moments": initial},
                            resources={"execution_context": execution})
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    except Exception as caught:
        reason = str(caught)
    else:
        reason = ""
    reasons = world.allgather_bytes(reason.encode())
    bound = world.allgather_bytes(b"bound" if runtime is not None else b"unbound")
    assert len(set(bound)) == 1, "bind did not converge across ranks"
    after = np.asarray(runtime.state_global("moments")) if runtime is not None else None

    def check():
        example = _load("api040_m15_hyqmom_axial_b1.py")
        assert all(example.is_native_admission_refusal(item.decode()) for item in reasons), reasons
        if runtime is not None:
            np.testing.assert_array_equal(after.reshape(initial.shape), initial)
            assert runtime.time() == 0 and runtime.macro_step() == 0
    _check(world, check)
