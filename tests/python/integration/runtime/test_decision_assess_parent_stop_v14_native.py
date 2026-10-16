"""Native Decision receiver; qualification requires execution of this exact build."""
import pytest
from tests.python.support.decision_assess_v14 import run_failure

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

@pytest.mark.parametrize('mode',('parent_stop',))
def test_decision_stop_and_budget_preserve_payload_and_complete_parent_rollback(mode,isolated_native_cache,native_cxx,kokkos_root):
    run_failure(mode,isolated_native_cache,native_cxx,kokkos_root)
