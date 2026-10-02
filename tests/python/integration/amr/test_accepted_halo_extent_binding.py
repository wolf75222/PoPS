"""Real pybind config interoperability; no mesh, runtime, compiler or JIT needed."""
import pytest


@pytest.mark.native_loader
def test_accepted_halo_extent_uses_real_ranked_converter():
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(2)
    config = native.AmrSystemConfig()
    config.accepted_halo_contract_version = 1
    config.accepted_halo_extent = (1, 2)
    assert config.accepted_halo_extent == (1, 2)
    assert config.accepted_halo_contract_version == 1
    # The setter must reject malformed input without changing its prior typed value.
    for value in ((1,), (1, 2, 3), [1, 2], (True, 2), (1.0, 2), (0, 2), (-1, 2)):
        with pytest.raises((TypeError, ValueError)):
            config.accepted_halo_extent = value
        assert config.accepted_halo_extent == (1, 2)
