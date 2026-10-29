"""N3a: Rusanov and HLL written in Python (pops.lib.numerics) against b6's native routes.

Each pair compiles the same case twice, once with the native face and once with the Python face,
binds the same initial data and takes the same fixed steps. The two final states must agree to
rounding; their L1 and L-infinity differences are recorded. The Python face is also received by
the case's own invariants. Both HLL rows take an interval that the case states: on M01 both waves
of each axis are positive, so only the sL >= 0 branch runs; on M03 sL < 0 < sR on every face, so
the two-wave state runs. The derived hll(F, Uh) needs a face input that riemann.User lacks (CF15,
ROM-99; tests/python/unit/numerics/test_lib_numerics_faces.py).
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from examples.api_v02.models_M01_M14 import (build_m01, build_m03, m01_hll, m03_hll, native_hll,
                                             native_rusanov, rusanov)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import (
    compile_profile_plan_once, native_profile, profile_gather_value,
)


# case -> (builder, native face, Python face, declare the signed pair of each axis)
PAIRS = {
    "M01-rusanov": (build_m01, native_rusanov, rusanov, False),
    "M03-rusanov": (build_m03, native_rusanov, rusanov, False),
    "M01-hll": (build_m01, native_hll, m01_hll, True),
    "M03-hll": (build_m03, native_hll, m03_hll, True),
}


def _run(world, build, scheme, signed, route):
    spec = collective_call(world, lambda: build(scheme=scheme, signed_wave_speeds=signed))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                           layout=spec.layout))
    artifact = compile_profile_plan_once(world, resolved, route=route,
                                         compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values),
        resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=32,
                                                     console=False))
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    state = collective_call(world, lambda: runtime.state_global("fluid"))
    return report.accepted_steps, metrics, state


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("pair", tuple(PAIRS))
def test_python_face_matches_the_native_route(pair, isolated_native_cache, native_cxx,
                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope=f"N3a {pair}")
    build, native, python, signed = PAIRS[pair]
    native_steps, _, native_state = _run(world, build, native, signed, f"N3a {pair} native")
    python_steps, metrics, python_state = _run(world, build, python, signed, f"N3a {pair} Python")
    gathered = profile_gather_value(world, metrics)
    with collective_check(world):
        assert native_steps == python_steps >= 2
        owners = [row for row in gathered if row.get("metrics_owner", True)]
        assert owners and all(row["passed"] for row in owners), owners
        if native_state is not None:
            native_state, python_state = np.asarray(native_state), np.asarray(python_state)
            assert native_state.shape == python_state.shape
            difference = python_state - native_state
            l1, linf = float(np.mean(np.abs(difference))), float(np.max(np.abs(difference)))
            record_property("l1", l1)
            record_property("linf", linf)
            record_property("bitwise_equal", bool(np.array_equal(python_state, native_state)))
            # Rounding level: a few ulps of the state per step. Both routes evaluate the same
            # formula in the same order; only floating-point contraction may differ by compiler.
            scale = max(1., float(np.max(np.abs(native_state))))
            assert linf <= 16*np.finfo(float).eps*scale*native_steps, (l1, linf)
