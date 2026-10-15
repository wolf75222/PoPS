"""Actual PoPS D1Q3 nonlinear face/volume equations and rejected-interval reconstruction."""
import pytest

from examples.api_v02.method_nonlinear_interval_transport import build_nonlinear_interval_transport
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("retry", (False, True))
def test_nonlinear_interval_transport_and_retry(retry, isolated_native_cache, native_cxx,
                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="T09 nonlinear isothermal D1Q3 interval")
    runtime, artifact, metrics, report = execute_case(
        lambda: build_nonlinear_interval_transport(retry=retry), world,
        "T09 nonlinear interval retry="+str(retry), return_report=True)
    record_property("program_hash", artifact.program_hash)
    record_property("actual_metrics", str(metrics))
    record_property("accepted_steps", report.accepted_steps)
    record_property("rejected_steps", report.rejected_steps)
    with collective_check(world):
        assert abs(runtime.time()-.006) < 2.e-14
        assert runtime.macro_step() == 3 and report.accepted_steps == 3
        assert report.rejected_steps > 0 if retry else report.rejected_steps == 0
        owners = [item for item in metrics if item["metrics_owner"]]
        assert len(owners) == 1
        assert owners[0]["full_state_quadrature_error"] < 4.e-13
        assert owners[0]["mass_defect"] < 4.e-14
        assert owners[0]["momentum_defect"] < 4.e-14
        assert all(owners[0][name] > 1.e-8 for name in (
            "missing_temporal_equilibrium", "frozen_interval", "frozen_transported_equilibrium"))
