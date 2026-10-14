"""N02 native receiver; compile and execution are required for qualification."""
import pytest
from tests.python.support.multisupport_children_v14 import run_success, run_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

@pytest.mark.parametrize('mode',('double_consume',))
def test_multisupport_parent_failure_restores_all_supports_and_histories(mode,isolated_native_cache,native_cxx,kokkos_root):
    run_failure(mode,isolated_native_cache,native_cxx,kokkos_root)
