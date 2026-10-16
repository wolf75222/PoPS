"""Installed native reception of distributed Program algebra, on exactly MPI2 or MPI3.

The three-rank profile has a genuinely empty middle shard.  The field is only the
transaction carrier; imported data are checked by compiled algebra before commit.
"""
import hashlib
import json
import numpy as np
import pops
import pytest

from examples.api_v02.distributed_program_checks import BASE, build_distributed_program
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _profile():
    _, world = native_profile(dimension=2, scope="distributed Program algebra")
    assert world is not None and int(world.size) in (2, 3), "explicit MPI2 or MPI3 reception required"
    return world


def _failure_shards(payload):
    assert type(payload) is bytes and payload.startswith(b"POPSFAIL1")
    cursor = len(b"POPSFAIL1")
    def word():
        nonlocal cursor
        assert len(payload)-cursor >= 8
        value = int.from_bytes(payload[cursor:cursor+8], "big")
        cursor += 8
        return value
    count = word()
    assert count <= (len(payload)-cursor)//8
    shards = []
    for _ in range(count):
        size = word()
        assert size <= len(payload)-cursor
        shards.append(payload[cursor:cursor+size])
        cursor += size
    assert cursor == len(payload)
    return tuple(shards)


@pytest.mark.parametrize("kind", ["roundtrip", "user_payload", "corrupt_data"])
def test_distributed_program_checks_native_data_before_commit(kind, isolated_native_cache,
        native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    world = _profile()
    runtime, artifact, metrics = execute_case(
        lambda: build_distributed_program(kind, ranks=int(world.size)), world,
        "distributed-program-"+kind+"-mpi"+str(world.size))
    with collective_check(world):
        owners = [row for row in metrics if row.get("metrics_owner")]
        assert owners and all(row["max_error"] == 0. for row in owners)
        assert all(row["mpi_ranks"] == int(world.size) for row in owners)
        assert runtime.macro_step() == 1
        record_property("mpi_ranks", int(world.size))
        record_property("compiled_data_gate", kind)
        record_property("rank_one_empty", int(world.size) == 3)
        record_property("artifact_identity", artifact.artifact_identity.hexdigest)
        record_property("metrics", json.dumps(metrics, sort_keys=True))


def test_corrupt_import_unhandled_failure_preserves_payload_and_rolls_back(
        isolated_native_cache, native_cxx, kokkos_root, record_property, tmp_path):
    del isolated_native_cache, native_cxx, kokkos_root
    world = _profile()
    spec = collective_call(world, lambda: build_distributed_program(
        "corrupt_data", ranks=int(world.size), recover=False))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan,
        route="distributed-unhandled-mpi"+str(world.size), compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values),
        resources={"execution_context": artifact_execution_context(artifact)}))
    before = collective_call(world, lambda: runtime.state_global("method"))
    error = None
    try:
        pops.run(runtime, t_end=spec.t_end, max_steps=1, console=False)
    except Exception as caught:
        error = caught
    after = collective_call(world, lambda: runtime.state_global("method"))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error, StepAttemptRejected), repr(error)
        schema = error.typed_failure_schema
        assert json.loads(schema)["schema"] == "pops.typed-failure@1"
        shards = _failure_shards(error.typed_failure_payload)
        assert len(shards) == int(world.size)
        for shard in shards:
            # ContinuationCodec words are canonical little endian; the outer envelope is big endian.
            assert b"POPSCONT1" in shard and schema.encode() in shard
            # The actual compiled check selected qid2: source gid3 was corrupted.
            assert (7001).to_bytes(8, "little") in shard
            assert (BASE+4).to_bytes(8, "little") in shard
            assert (BASE+3).to_bytes(8, "little") in shard
        assert runtime.time() == 0. and runtime.macro_step() == 0
        if before is not None:
            np.testing.assert_array_equal(after, before)
        (tmp_path/"distributed-failure.bin").write_bytes(error.typed_failure_payload)
        record_property("mpi_ranks", int(world.size))
        record_property("artifact_identity", artifact.artifact_identity.hexdigest)
        record_property("typed_failure_sha256", hashlib.sha256(error.typed_failure_payload).hexdigest())
        record_property("typed_failure_shards", len(shards))
