"""The changed request must enter actual native spatial characteristic fluxes."""
import pytest

from examples.api_v02.method_interval_transport import build_interval_transport
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("duration,retry",((.01,False),(.006,False),(.01,True)))
def test_spatial_interval_reconstruction_uses_runtime_request(duration,retry,
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="spatial Duhamel native interval")
    runtime,artifact,metrics,report=execute_case(lambda:build_interval_transport(dt=duration,retry=retry),
        world,"T09 native duration "+str(duration)+" retry "+str(retry),return_report=True)
    record_property("program_hash",artifact.program_hash)
    record_property("actual_metrics",str(metrics))
    with collective_check(world):
        assert abs(runtime.time()-3*duration)<2.e-14
        assert runtime.macro_step()==3
        assert report.accepted_steps==3
        assert report.rejected_steps>0 if retry else report.rejected_steps==0
