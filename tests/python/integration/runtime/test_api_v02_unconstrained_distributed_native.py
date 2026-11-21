"""N10 lot 4 (DM3, CC7): distributed cases preserve their committed state.

The ordered algebra uses plan-balanced offsets in DM3 mode. The historical
roundtrip declares the offsets required by its owner and CSR assertions.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest

from examples.api_v02.distributed_algebra_v14 import build_distributed_algebra
from examples.api_v02.distributed_program_checks import build_distributed_program
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile
from tests.python.support.unconstrained import (
    HISTORICAL_ROUNDTRIP_PLACEMENT, without_storage_constraints,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CASES = {
    "distributed_algebra_ordered": lambda ranks: build_distributed_algebra("ordered", ranks=ranks),
    "distributed_program_roundtrip": lambda ranks: build_distributed_program("roundtrip",
                                                                             ranks=ranks),
}


@pytest.mark.parametrize("name", ["distributed_algebra_ordered", "distributed_program_roundtrip"])
def test_a_distributed_case_runs_bitwise_alike_in_dm3_mode(
        name, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N10 lot 4 unconstrained " + name)
    assert world is not None and int(world.size) in (2, 3), "explicit MPI2 or MPI3 reception"
    ranks = int(world.size)
    placement = (HISTORICAL_ROUNDTRIP_PLACEMENT[ranks]
                 if name == "distributed_program_roundtrip" else None)

    def builder():
        return CASES[name](ranks)

    def dm3_builder():
        with without_storage_constraints(placement=placement):
            return builder()

    states, owners = [], []
    for label, build in (("offsets", builder), ("dm3", dm3_builder)):
        runtime, _, metrics = execute_case(
            build, world, "N10 lot 4 %s %s mpi%d" % (name, label, ranks))
        states.append(collective_call(world, lambda runtime=runtime: np.asarray(
            runtime.state_global("method"))))
        owners.append([row for row in metrics if row.get("metrics_owner")])
    with collective_check(world):
        assert owners[0] and owners[0] == owners[1]
        assert states[0].dtype == states[1].dtype and states[0].shape == states[1].shape
        assert states[0].tobytes() == states[1].tobytes()
    record_property("mpi_ranks", ranks)
    record_property("plan_placement_constraint", json.dumps(
        None if placement is None else asdict(placement), sort_keys=True))
    record_property("metrics", json.dumps(owners[0], sort_keys=True))
