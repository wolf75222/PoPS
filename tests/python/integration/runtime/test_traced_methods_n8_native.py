"""N8 lot 1, native: the traced profiles run bit for bit like their b6 builder.

The profiles whose stages read one rate of their model write their step as the traced method
``ssprk2`` of ``examples/api_v02/models_M01_M14.py``; ``_ssprk2``, the b6 builder they used, stays
as the reference (MG1). Both routes are compiled, bound and run; the case invariants of b6 must
pass on both, and the final states must be equal bitwise, as the accepted step counts.

CI runs three distinct spatial routes: an authored scalar flux (M01), two summed rates with
their own realizations (M04) and a coordinated face with a bed product (M07). The same test over
all seven profiles also passed for M02, M03, M24 and M25 (1311 s on four cores, M25 alone 620 s;
handoff of N8 lot 1). ``tests/python/unit/time/test_traced_methods_n8.py`` compares the IR of the
seven and the C++ of six.
"""
import numpy as np
import pytest

import examples.api_v02.models_M01_M14 as early
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

# profile -> (module that calls _traced_ssprk2, builder, b6 program name, block)
PROFILES = {
    "M01": (early, early.build_m01, "M01_SSPRK2", "fluid"),
    "M04": (early, early.build_m04, "M04_combined_SSPRK2", "fluid"),
    "M07": (early, early.build_m07, "M07_SSPRK2", "fluid"),
}


def _run(build, block, world, route):
    runtime, artifact, rows = execute_case(build, world, route)
    state = collective_call(world, lambda: runtime.state_global(block))
    steps = collective_call(world, runtime.macro_step)
    return (None if state is None else np.asarray(state).copy()), steps, artifact, rows


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_traced_profile_runs_bitwise_like_b6(profile, monkeypatch, isolated_native_cache,
                                             native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    module, build, b6_name, block = PROFILES[profile]
    _, world = native_profile(dimension=2, scope="N8 traced profile %s" % profile)
    traced = _run(build, block, world, "N8 %s traced method" % profile)

    def b6_route(case, block_handle, state, flow, dt):
        return early._ssprk2(case, block_handle, state, lambda p, u, point, label: flow(u), dt,
                             name=b6_name)
    monkeypatch.setattr(module, "_traced_ssprk2", b6_route)
    b6 = _run(build, block, world, "N8 %s b6 builder" % profile)
    with collective_check(world):
        assert traced[2].program_name == "ssprk2" and b6[2].program_name == b6_name
        assert all(row["passed"] for row in b6[3]) and all(row["passed"] for row in traced[3])
        assert traced[1] == b6[1] > 0
        if b6[0] is not None or traced[0] is not None:
            assert b6[0] is not None and traced[0] is not None
            assert b6[0].dtype == traced[0].dtype == np.float64
            # Bitwise, not within a tolerance: the same stages in the same order.
            assert b6[0].tobytes() == traced[0].tobytes()
    record_property("artifact_identities", (b6[2].artifact_identity.hexdigest,
                                            traced[2].artifact_identity.hexdigest))
