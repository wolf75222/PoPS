"""True native endpoint and boundary traces for the lifted spatial Strang case."""
import math
import numpy as np
import pytest
from examples.api_v02.method_splitting_boundary import build_t03_dirichlet
from tests.python.support import splitting_boundary_oracle as reference
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def test_native_dirichlet_lifted_strang_order_and_physical_clock(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="T03 nonautonomous Dirichlet splitting")
    errors = []
    for steps in (4, 8, 16):
        h = .2/steps
        spec = collective_call(world, lambda: build_t03_dirichlet(dt=h, steps=steps))
        runtime, artifact, _ = execute_case(lambda: spec, world, "T03 Dirichlet")
        raw = collective_call(world, lambda: _image(runtime))
        with collective_check(world):
            packet = _decode(spec, raw)["root"]
            values = packet["values"]
            u = np.asarray(values["u"]["values"])
            np.testing.assert_allclose(u, reference.discrete(7, h, steps), rtol=0, atol=2.e-10)
            errors.append(float(np.max(abs(u-reference.exact(7, .2)))))
            assert abs(values["physical_time"]-.2) < 1.e-14
            assert abs(values["midpoint"]-(.2-h/2)) < 1.e-14
            # FixedDt clips the final request to the remaining physical interval.
            # Test the exact composition identity against that actual request;
            # nominal decimal h and end_time - accumulated_time need not have
            # the same Float64 representation.
            assert values["algorithmic_duration"] == 2*values["reached_duration"]
            assert abs(values["reached_duration"]-h) <= 2*math.ulp(.2)
            assert abs(values["left_trace"]-math.exp(.2)) < 1.e-14
            assert abs(values["right_trace"]-2*math.exp(.2)) < 1.e-14
            assert abs(values["mid_left"]-math.exp(.2-h/2)) < 1.e-14
            assert abs(values["mid_right"]-2*math.exp(.2-h/2)) < 1.e-14
            assert values["naive_B_left_residual"] > 1
            assert values["naive_B_right_residual"] > 2
        record_property(f"artifact_{steps}", artifact.artifact_identity.hexdigest)
    with collective_check(world):
        assert all(3.8 < a/b < 4.2 for a, b in zip(errors, errors[1:]))
    record_property("semidiscrete_errors", str(errors))
