"""Native Decision receivers; qualification requires execution of this exact build."""
import pytest
from tests.python.support.decision_assess_v14 import run_success, run_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

def test_decision_nested_children_accept_and_checkpoint(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    run_success('nested_success',tmp_path,isolated_native_cache,native_cxx,kokkos_root)
