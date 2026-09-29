"""Owner-only M15 face refusal must converge before the Python attempt envelope."""
import os

import numpy as np
import pops
import pytest

from tests.python.unit.moments.test_m15_axial_reception import _load, oracle

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def test_periodic_generated_face_refusal_is_collective_before_retry_envelope(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    if os.environ.get("POPS_NATIVE_DIM") != "1":
        pytest.fail("this regression requires installed native Dim=1")
    case, layout, _ = _load("api040_m15_hyqmom_axial_b1.py").build_case(
        8, dt=1 / 800,
    )
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    execution = pops.ExecutionContext.mpi_world(artifact)
    world = execution.communicator.handle
    if world.size < 2:
        pytest.skip("the regression requires at least two MPI ranks")

    initial = np.ascontiguousarray(oracle.invalid_states(8)["negative_density"])
    runtime = None
    try:
        runtime = pops.bind(
            artifact, initial_state={"moments": initial},
            resources={"execution_context": execution},
        )
        bind_reason = ""
    except Exception as error:
        bind_reason = type(error).__name__ + ": " + str(error)
    # Distinguish a rank-local bind failure from the native step defect. Never let
    # a successful peer enter System.step while another rank collects bind results.
    bind_reasons = world.allgather_bytes(bind_reason.encode())
    assert not any(bind_reasons), bind_reasons
    assert runtime is not None

    try:
        pops.run(runtime, t_end=1 / 800, max_steps=1, console=False)
        step_reason = ""
    except Exception as error:
        step_reason = type(error).__name__ + ": " + str(error)
    step_reasons = world.allgather_bytes(step_reason.encode())
    assert all(b"prepared ND hyperbolic face evaluation refused publication status=1" in row
               for row in step_reasons), step_reasons
    assert runtime.time() == 0 and runtime.macro_step() == 0
    gathered = np.asarray(runtime.state_global("moments"))
    if world.rank == 0:
        np.testing.assert_array_equal(gathered.reshape(initial.shape), initial)
