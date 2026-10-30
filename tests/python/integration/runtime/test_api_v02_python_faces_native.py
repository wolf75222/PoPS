"""N3a, ROM-99: Rusanov and HLL written in Python (pops.lib.numerics) against b6's native routes.

Each pair compiles the same case twice, once with the native face and once with the Python face,
binds the same initial data and takes the same fixed steps. The two final states must agree to
rounding; their L1 and L-infinity differences are recorded. The Python face is also received by
the case's own invariants. The HLL rows run the derived ``hll(F, Uh)``: its face reads the signed
interval (sL, sR) that the model declares along the face normal (CF15, ROM-99). On M01 both waves
of each axis are positive, so only the sL >= 0 branch runs; on M03 (v.n - c, v.n + c) and on M16
(the eigenvalues of the B.1 flux Jacobian) sL < 0 < sR on every face, so the two-wave state runs.
On the nonreal spectrum of M16's oblique profile the derived face refuses the step, as the native
one does.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from examples.api_v02.models_M01_M14 import build_m01, build_m03, native_hll, native_rusanov
from examples.api_v02.models_M15_M18 import build_m16
from pops.lib.numerics import hll, rusanov
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import (
    compile_profile_plan_once, native_profile, profile_gather_value,
)


# case -> (builder, native face, Python face, builder options, evolved block)
SIGNED = {"signed_wave_speeds": True}
PAIRS = {
    "M01-rusanov": (build_m01, native_rusanov, rusanov, {}, "fluid"),
    "M03-rusanov": (build_m03, native_rusanov, rusanov, {}, "fluid"),
    "M01-hll": (build_m01, native_hll, hll, SIGNED, "fluid"),
    "M03-hll": (build_m03, native_hll, hll, SIGNED, "fluid"),
    "M16-hll": (build_m16, native_hll, hll, {}, "moments"),
}


def _bind(world, spec, route):
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                           layout=spec.layout))
    artifact = compile_profile_plan_once(world, resolved, route=route,
                                         compile_artifact=pops.compile)
    return collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values),
        resources={"execution_context": artifact_execution_context(artifact)}))


def _run(world, build, scheme, options, block, route):
    spec = collective_call(world, lambda: build(scheme=scheme, **options))
    runtime = _bind(world, spec, route)
    report = collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=32,
                                                     console=False))
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    state = collective_call(world, lambda: runtime.state_global(block))
    return report.accepted_steps, metrics, state


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("pair", tuple(PAIRS))
def test_python_face_matches_the_native_route(pair, isolated_native_cache, native_cxx,
                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope=f"N3a {pair}")
    build, native, python, options, block = PAIRS[pair]
    native_steps, _, native_state = _run(world, build, native, options, block,
                                         f"N3a {pair} native")
    python_steps, metrics, python_state = _run(world, build, python, options, block,
                                               f"N3a {pair} Python")
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


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_derived_hll_refuses_the_nonreal_oblique_spectrum_of_m16(isolated_native_cache,
                                                                  native_cxx, kokkos_root):
    """The signed interval of a trace with a nonreal Jacobian spectrum is not finite.

    The native HLL refuses such a face (test_api_v02_models_M15_M21.py, same profile); the
    derived face reads (sL, sR) as non-finite and refuses it too, so the step is refused and the
    state is not published (ROM-99).
    """
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="ROM-99 M16 oblique Python HLL")
    spec = collective_call(world, lambda: build_m16(cells=16, steps=4, oblique_refusal=True,
                                                    scheme=hll))
    runtime = _bind(world, spec, "ROM-99 M16 oblique Python HLL")
    initial = collective_call(world, lambda: runtime.state_global("moments"))
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=spec.dt, max_steps=1,
                                                             console=False))
    final = collective_call(world, lambda: runtime.state_global("moments"))
    clock = collective_call(world, lambda: (float(runtime.time()), int(runtime.macro_step())))
    with collective_check(world):
        assert all(failure is not None and failure[2] and
                   "hyperbolic face evaluation refused publication" in failure[1]
                   for failure in failures), failures
        assert clock == (0., 0), clock
        if initial is not None:
            np.testing.assert_array_equal(final, initial)
