"""Installed true-Program lifetime reception, MPI2/MPI3 only."""
import json
import pytest
from examples.api_v02.distributed_gc_v14 import build_distributed_gc
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def test_repeated_distributed_transactions_reuse_two_version_budget(
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="distributed collective GC v14")
    assert world is not None and int(world.size) in (2,3)
    runtime,artifact,metrics=execute_case(
        lambda:build_distributed_gc(ranks=int(world.size),steps=64),world,
        "distributed-gc-v14-mpi"+str(world.size),max_steps=64)
    with collective_check(world):
        owners=[row for row in metrics if row.get("metrics_owner")]
        assert owners and all(row["max_error"]==0. and row["verified_steps"]==64 for row in owners)
        assert runtime.macro_step()==64 and abs(runtime.time()-.64)<2e-14
        record_property("mpi_ranks",int(world.size))
        record_property("storage_versions",2)
        record_property("native_allocations_over_steps",128)
        record_property("artifact_identity",artifact.artifact_identity.hexdigest)
        record_property("metrics",json.dumps(metrics,sort_keys=True))
