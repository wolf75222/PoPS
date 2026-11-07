"""A typed terminal failure compiles in a Program without finite operations.

Its payload packs through finite structural storage, so the generated C++ needs the finite helpers
that the finite operations would otherwise bring.
"""
import json

import numpy as np
import pops
import pytest
from pops.time import FixedDt
from pops.time.variant_types import Failure, Int64, Product, Result, type_of

from examples.api_v02.common import CaseSpec
from examples.api_v02.methods_T01_T12 import _initial, _storage
from examples.api_v02.terminal_failure_checks import EXACT_CODE
from tests.python.support.api_v02_receipts import _shards
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _integer_failure():
    case, layout, model, block, state, p, q, initial = _storage(
        "terminal_failure_integer", ("a", "b"), (1., 2.), cells=(8, 4), sampling="cell_average")
    initial += np.arange(32, dtype=float).reshape(1, 4, 8) / 128
    errors = Failure(retry=Product(code=Int64))
    rejected = p.variant(Result(type_of(q.n), errors), "err",
                         p.variant(errors, "retry", p.product(code=p.int64(EXACT_CODE))))
    p.commit(q.next, p.value("must_not_publish", p.require_success(rejected), at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    .01, .01, lambda runtime: {"passed": False}, "native terminal typed failure integer")


def test_integer_terminal_failure_compiles_and_rolls_back(isolated_native_cache, native_cxx,
                                                          kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="typed terminal failure without finite operations")
    spec = collective_call(world, _integer_failure)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="terminal-failure-integer",
                                         compile_artifact=pops.compile)
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
            assert b"POPSCONT1" in shard and schema.encode() in shard
            assert EXACT_CODE.to_bytes(8, "little") in shard
        assert runtime.time() == 0 and runtime.macro_step() == 0
        if before is not None:
            np.testing.assert_array_equal(after, before)
