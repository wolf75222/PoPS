"""Native ABI reception: typed payload bytes, exact integers, MPI shards, rollback."""
import hashlib
import json
import struct

import numpy as np
import pops
import pytest
from tests.python.support.api_v02_receipts import _shards

from examples.api_v02.terminal_failure_checks import build_terminal_failure, EXACT_CODE
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]



@pytest.mark.parametrize("kind", ["scalar", "field"])
def test_terminal_failure_keeps_typed_payload_and_rolls_back(kind, isolated_native_cache,
        native_cxx, kokkos_root, record_property, tmp_path):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="typed terminal failure " + kind)
    spec = collective_call(world, lambda: build_terminal_failure(kind))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="terminal-failure-"+kind, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=dict(spec.initial_values),
        resources={"execution_context": artifact_execution_context(artifact)}))
    before = collective_call(world, lambda: runtime.state_global("method"))
    error = None
    try:
        pops.run(runtime, t_end=.01, max_steps=1, console=False)
    except Exception as caught:
        error = caught
    after = collective_call(world, lambda: runtime.state_global("method"))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error, StepAttemptRejected), repr(error)
        schema = error.typed_failure_schema
        assert json.loads(schema)["schema"] == "pops.typed-failure@1"
        shards = _shards(error.typed_failure_payload)
        assert len(shards) == (1 if world is None else world.size)
        for shard in shards:
            # ContinuationCodec words are canonical little endian; the outer envelope is big endian.
            assert b"POPSCONT1" in shard and schema.encode() in shard
            assert EXACT_CODE.to_bytes(8, "little") in shard
            assert struct.pack("<d", .25) in shard
        if kind == "field" and len(shards) > 1:
            assert len(set(shards)) > 1, "rank-local field ownership was replaced with one shard"
        assert runtime.time() == 0 and runtime.macro_step() == 0
        if before is not None:
            np.testing.assert_array_equal(after, before)
        (tmp_path/(kind+"-typed-failure.bin")).write_bytes(error.typed_failure_payload)
        record_property("typed_failure_bytes", len(error.typed_failure_payload))
        record_property("typed_failure_sha256", hashlib.sha256(error.typed_failure_payload).hexdigest())
        record_property("typed_failure_shards", json.dumps([len(shard) for shard in shards]))
        record_property("artifact_identity", artifact.artifact_identity.hexdigest)
