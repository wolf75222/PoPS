"""Tensor AMR consumes an implicit composite residual with its exact stencil contract."""
import pytest
import pops
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.integration.runtime.test_amr_implicit_diffusion import build
from tests.python.unit.codegen.test_generic_diffusion import generic_case


@pytest.mark.parametrize("kind,components", (("tensor",1), ("tensor_mms",1), ("tensor",2)))
def test_tensor_composite_implicit_resolves_faces_halos_and_consumed_residual(kind, components):
    case, layout = build(16, kind=kind, periodic_witness=True, components=components)
    plan = pops.resolve(pops.validate(case), layout=layout)
    operations = next(iter(plan.resolved_operations.values()))
    emitter = lower_and_validate(plan.blocks[0].model, resolved_operations=operations)[0]
    code = emit_cpp_program(plan.time, model=emitter, target="amr_system")
    assert "PreparedDiffusion<pops::kNativeDimension, %d, true>" % components in code
    assert code.count("ctx.solve_spatial_hierarchy(") == 1
    assert "stage_accepted_exchanges" in code and "attach_diffusive_flux_basis" not in code
    assert ".reference_residual_norm" in code and ".residual_norm" in code
    assert any(row.stencil_radius == 2 for row in operations.operations
               if (row.guarantees.get("numerical_method") or {}).get("method") == "tensor_diffusion")


def test_tensor_explicit_amr_does_not_reuse_uniform_stability_proof():
    model, plan = generic_case(2, tensor=True)
    emitter = lower_and_validate(model, resolved_operations=next(iter(plan.resolved_operations.values())))[0]
    with pytest.raises(ValueError, match="proven composite stability bound"):
        emit_cpp_program(plan.time, model=emitter, target="amr_system")
