"""Installed MPI2/MPI3 reception; no source execution is counted as native."""
import json
import pytest

from examples.api_v02.distributed_algebra_v14 import build_distributed_algebra
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("kind", ("ordered", "nested", "budget", "typed_reduce"))
def test_distributed_structural_algebra_native(kind, isolated_native_cache,
        native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="distributed structural algebra v14")
    assert world is not None and int(world.size) in (2, 3)
    runtime, artifact, metrics = execute_case(
        lambda: build_distributed_algebra(kind, ranks=int(world.size)), world,
        "distributed-algebra-v14-"+kind+"-mpi"+str(world.size))
    with collective_check(world):
        owners = [row for row in metrics if row.get("metrics_owner")]
        assert owners and all(row["max_error"] == 0. for row in owners)
        assert runtime.macro_step() == 1 and runtime.time() == .01
        record_property("mpi_ranks", int(world.size))
        record_property("compiled_data_gate", kind)
        record_property("rank_one_empty", int(world.size) == 3)
        record_property("artifact_identity", artifact.artifact_identity.hexdigest)
        record_property("metrics", json.dumps(metrics, sort_keys=True))
