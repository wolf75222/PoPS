import pytest
from examples.api_v02.method_authored_stability import build_authored_stability
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("retry", (False, True))
def test_signed_midpoint_compiled_energy_admission(retry, isolated_native_cache, native_cxx,
                                                  kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="signed midpoint authored numerical admission")
    runtime, artifact, metrics, report = execute_case(
        lambda: build_authored_stability(retry=retry), world,
        "authored energy admission retry="+str(retry), return_report=True)
    record_property("program_hash", artifact.program_hash)
    record_property("metrics", str(metrics))
    with collective_check(world):
        assert runtime.macro_step() == 3 and report.accepted_steps == 3
        assert report.rejected_steps > 0 if retry else report.rejected_steps == 0
