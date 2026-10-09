"""Resolved principal rows install storage while their Program owns transport."""
import pytest

from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.runtime._bricks_scheme import Spatial
from pops.runtime._state_storage import StateStorageSpatial
from tests.python.unit.codegen.test_principal_numerical_captures import captured_group


@pytest.mark.parametrize("widths,reverse", (((1, 1), False), ((2, 3), False), ((2, 3), True)))
def test_group_rows_select_the_same_exact_storage_route_as_the_generated_model(widths, reverse):
    resolved, _ = captured_group(widths=widths, reverse=reverse, face=True)
    for block in resolved.blocks:
        method = block.numerics.primary_spatial()
        adapter = method.runtime_spatial()
        assert type(adapter) is StateStorageSpatial
        assert (adapter.limiter, adapter.flux, adapter.recon) == (
            "state_storage", "unavailable", "conservative")
        assert adapter.ghost_depth == method.ghost_depth
        emitter, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
            resolved_operations=block.resolved_operations, numerics=block.numerics)
        source = _emit_bricks(emitter._m)[1]
        assert "program_only_storage = true" in source
        assert "program_state_ghost_depth = %d;" % adapter.ghost_depth in source


def test_local_finite_volume_retains_its_native_transport_route():
    from tests.python.support.symbolic_path_case import declarations
    from pops.numerics import FiniteVolume, reconstruction, riemann, variables

    _, state, flux, _, _ = declarations()
    method = FiniteVolume(flux=flux, variables=variables.Conservative(state),
                          reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    assert type(method.runtime_spatial()) is Spatial
