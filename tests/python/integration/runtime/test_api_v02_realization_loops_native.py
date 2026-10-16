"""Prepared true PoPS reception; requires integrated/installed delivery checker identity."""
import pytest
from examples.api_v02.realization_loop_checks import build_delivery_case
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize('kind,factor', [('loop1', 1.), ('overwrite2', 2.)])
def test_native_final_loop_delivery_is_executed_once_in_the_commit(kind, factor,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope='API02 loop delivery '+kind)
    runtime, artifact, metrics = execute_case(lambda: build_delivery_case(kind), world,
                                              'loop-delivery-'+kind)
    with collective_check(world):
        assert runtime.macro_step() == 1 and runtime.time() == pytest.approx(.02)
        for row in metrics:
            if row.get('metrics_owner', True):
                assert row['retained_factor'] == factor
    record_property('artifact_identity', artifact.artifact_identity.hexdigest)
    record_property('metrics', str(metrics))
