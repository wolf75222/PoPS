"""N02 native receiver; compile and execution are required for qualification."""
import pytest
from tests.python.support.multisupport_children_v14 import run_success, run_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

def test_multisupport_two_children_transform_retry_and_checkpoint(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    run_success(tmp_path,isolated_native_cache,native_cxx,kokkos_root)
