"""Native spatial receivers; exact equations are shared without changing scenarios."""
import pytest
from tests.python.support.scientific_spatial_v14 import run_success, run_failure

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

@pytest.mark.parametrize('count,krylov', ((5,'gmres'),))
def test_scientific_spatial_complete_solution_and_restart(count, krylov, tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    run_success(count, krylov, tmp_path, isolated_native_cache, native_cxx, kokkos_root)
