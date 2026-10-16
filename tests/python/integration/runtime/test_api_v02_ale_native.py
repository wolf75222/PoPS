"""Dim1 ALE receiver: real moving volumes, GCL and conservative updates."""
import pytest

from examples.api_v02.models_M22_M25_M28 import build_m28
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("amplitude,nonuniform",[(.04,False),(0.,True),(.04,True)])
def test_actual_ale_volumes(amplitude,nonuniform,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=1,scope="ALE actual geometry")
    _,artifact,metrics=execute_case(lambda:build_m28(amplitude=amplitude,nonuniform=nonuniform),world,"M28 ALE")
    record_property("program_hash",artifact.program_hash)
    record_property("actual_geometry_and_conservation",str(metrics))
