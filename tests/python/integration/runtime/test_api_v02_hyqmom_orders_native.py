"""True Dim1 S04 HyQMOM orders2/4, smooth and discontinuous FV profiles."""
import pytest

from examples.api_v02.models_M15_orders import build_m15_orders
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile


pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


@pytest.mark.parametrize("order",(2,4))
@pytest.mark.parametrize("regime",("smooth","riemann"))
def test_native_hyqmom_multiple_orders_and_regimes(order,regime,isolated_native_cache,
                                                 native_cxx,kokkos_root,record_property):
    del native_cxx,kokkos_root
    _,world=native_profile(dimension=1,scope="S04 HyQMOM multiple orders")
    _,artifact,metrics=execute_case(lambda:build_m15_orders(order=order,regime=regime),
                                  world,"S04 order%d %s"%(order,regime))
    record_property("program_hash",artifact.program_hash)
    record_property("actual_metrics",str(metrics))
    from examples.api_v02.reception import artifact_sources
    receipt=isolated_native_cache/"generated-equations"
    receipt.mkdir()
    record_property("generated_artifacts",str(artifact_sources(artifact,receipt)))
