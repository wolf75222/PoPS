"""ADC-942: AMR diffusion lowers through stage ghosts and the sole reflux authority."""

from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.integration.runtime.test_public_diffusion_matrix import _build


def _combined_source():
    case, _, _ = _build("constant", 8, 1.0e-4, transport=(0.2, -0.1))
    model = dict(case._block_registry.items())["heat"]["model"]
    lowered, _ = lower_and_validate(model)
    return emit_cpp_program(list(case._time_registry)[0], model=lowered,
                            target="amr_system")


def test_explicit_amr_diffusion_prepares_stage_and_registers_each_face_basis_once():
    source = _combined_source()
    prepared = source.index("ctx.prepare_generated_state(0,")
    applied = source.index(".apply(", prepared)
    diffusive_basis = source.index("ctx.attach_diffusive_flux_basis(0,", applied)
    transport_basis = source.index("ctx.neg_div_flux_default_with_faces_into(0,", applied)
    assert prepared < applied < diffusive_basis < transport_basis
    assert source.count("ctx.attach_diffusive_flux_basis(") == 1
    assert source.count("ctx.neg_div_flux_default_with_faces_into(") == 1
    assert "combined_transport_diffusion_stability" in source


def test_amr_accepted_exchange_inventory_uses_composite_active_cells():
    source = _combined_source()
    assert source.count("ctx.pointwise_active_mask(0,") == 1
    assert source.count(".stage_accepted_exchanges(ctx,0,") == 1
    assert "prepared_amr_ghosts" not in source
