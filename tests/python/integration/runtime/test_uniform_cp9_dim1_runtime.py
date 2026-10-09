"""Explicit CI dimension entry; actual Native execution is required."""
import os
import pytest
from tests.python.support.dimensional_uniform_cp9_runtime import run_installed_dimensional_cp9_two_states
pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]

def test_installed_dimensional_cp9_two_states(tmp_path, record_property, isolated_native_cache, native_cxx, kokkos_root):
    if os.environ.get('POPS_NATIVE_DIM') != '1':
        raise RuntimeError('this catalog entry requires exact POPS_NATIVE_DIM=1')
    return run_installed_dimensional_cp9_two_states(1, tmp_path, record_property, isolated_native_cache, native_cxx, kokkos_root)
