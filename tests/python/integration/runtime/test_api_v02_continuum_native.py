"""Real PoPS receivers for authored radiation, Hall, BN and GPR equations."""
import pytest

from examples.api_v02.models_M22_M25_M28 import build_m22,build_m23,build_m24,build_m25
from tests.python.support.collective_checks import collective_check
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("builder",[build_m22,build_m23,build_m24,build_m25])
def test_continuum_profile(builder,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope=builder.__name__)
    _,artifact,metrics=execute_case(builder,world,builder.__name__)
    record_property("program_hash",artifact.program_hash)
    record_property("actual_metrics",str(metrics))


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_hall_zero_and_ohmic_are_different_physics_in_same_native_primitive(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="Hall versus Ohmic")
    # Compare at one physical time with identical meshes and step counts.
    dt=4.e-5
    _,_,zero=execute_case(lambda:build_m23(hall=0.,dt=dt),world,"M23 zero Hall")
    _,_,hall=execute_case(lambda:build_m23(hall=.1,dt=dt),world,"M23 Hall")
    _,_,ohmic=execute_case(lambda:build_m23(hall=.1,ohmic=.05,dt=dt),world,"M23 Hall and Ohmic")
    with collective_check(world):
        if zero[0].get("metrics_owner",True):
            assert hall[0]["frequency"]>1.2*zero[0]["frequency"]
            assert ohmic[0]["amplitude"]<hall[0]["amplitude"]
            assert abs(ohmic[0]["energy_loss"])<2e-12
