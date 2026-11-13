"""N10 lot 4 (DM3, CC7; addendum §2): a distributed case runs alike without its partition.

On exactly MPI2 or MPI3. ``distributed_algebra_v14`` (``ordered``) and
``distributed_program_checks`` (``roundtrip``) are built twice, with the offsets they write and
with the plan's balanced partition (``tests/python/support/unconstrained.py``). On two ranks the
partitions differ ((0, 1, 4) against (0, 2, 4), and so on). Both builds pass their own invariants
and commit the same global state, bitwise: the reductions are ordered by global index. A
difference is reported and bisected, never absorbed (CF14).
"""
import json

import numpy as np
import pytest

from examples.api_v02.distributed_algebra_v14 import build_distributed_algebra
from examples.api_v02.distributed_program_checks import build_distributed_program
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile
from tests.python.support.unconstrained import without_storage_constraints

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CASES = {
    "distributed_algebra_ordered": lambda ranks: build_distributed_algebra("ordered", ranks=ranks),
    "distributed_program_roundtrip": lambda ranks: build_distributed_program("roundtrip",
                                                                             ranks=ranks),
}


def _unconstrained(builder):
    def build():
        with without_storage_constraints():
            return builder()
    return build


@pytest.mark.parametrize("name", ["distributed_algebra_ordered", "distributed_program_roundtrip"])
def test_a_distributed_case_runs_bitwise_alike_on_the_plans_partition(
        name, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N10 lot 4 unconstrained " + name)
    assert world is not None and int(world.size) in (2, 3), "explicit MPI2 or MPI3 reception"
    ranks = int(world.size)

    def builder():
        return CASES[name](ranks)
    states, owners = [], []
    for label, build in (("offsets", builder), ("balanced", _unconstrained(builder))):
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
    record_property("metrics", json.dumps(owners[0], sort_keys=True))
