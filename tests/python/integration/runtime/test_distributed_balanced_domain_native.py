"""Identical equations on any MPI lane, including globally/rank-local empty domains."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from examples.api_v02.common import CaseSpec, non_owner_metrics, state_array
from examples.api_v02.methods_T01_T12 import _initial, _storage
from pops.algebra import Distributed
from pops.time import FixedDt
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def build_balanced_domain(size):
    case, layout, _, block, state, program, q, initial = _storage(
        "balanced_native_" + str(size), ("a",), (1.0,), cells=(8, 4), sampling="cell_average"
    )
    distributed = Distributed(program)
    domain = distributed.balanced_domain(program.int64(size), capacity=16)
    values = program.require_success(
        distributed.local_map(
            domain,
            lambda builder, index: builder.capture(lambda body: distributed.gid(index) + 1),
            byte_capacity=65536,
        )
    )
    total = distributed.reduce(
        values,
        program.int64(0),
        lambda builder, total, value: total + value,
        order="global_gid_ascending",
        byte_capacity=65536,
    )
    accepted = program.value("sum_global_gids", program.float64(total) * q.n, at=q.next.point)
    program.commit(q.next, accepted)
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    expected = initial * (size * (size + 1) // 2)

    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual - expected)))
        return dict(passed=error == 0.0, metrics_owner=True, max_error=error)

    return CaseSpec(
        case,
        layout,
        _initial(case, block[state], initial),
        {"method": block[state]},
        0.01,
        0.01,
        check,
        "native balanced placement with rank-independent equations",
    )


@pytest.mark.parametrize("size", (0, 1, 7))
def test_balanced_domain_native(
    size, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="balanced distributed domain")
    runtime, artifact, metrics = execute_case(
        lambda: build_balanced_domain(size), world, "balanced distributed domain"
    )
    with collective_check(world):
        program = artifact.program
        assert program is not None
        path = Path(program.so_path).resolve()
        record_property("program_hash", runtime._executor._s.installed_program_hash())
        record_property("program_dso_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        record_property("metrics", str(metrics))
