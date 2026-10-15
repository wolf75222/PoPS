"""Retained reduced M19-M21 regressions; full phase-space receivers are separate."""
import pytest

from tests.python.support.api_v02_moment_profiles import run_native_moment_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("model", ("M19", "M20", "M21"))
def test_native_closed_multicell_profile(isolated_native_cache, model, record_property):
    del isolated_native_cache
    run_native_moment_profile(model, record_property)
