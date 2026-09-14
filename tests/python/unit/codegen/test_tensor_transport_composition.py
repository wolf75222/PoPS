"""Resolved methods retain independent transport and tensor ownership on one block."""
import pops
import pytest
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.integration.runtime.test_tensor_transport_composition import composition_case


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_separate_transport_and_tensor_emit_both_operators_and_required_halos(dimension):
    case, layout, model = composition_case(dimension=dimension)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    authored_flux = model._dsl._m._flux
    authored_eigenvalues = model._dsl._m._eig
    operations = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=operations)[0]
    from pops.codegen._resolved_operation_inputs import _reference
    diffusion = next(value for value in resolved.time._values if value.op == "diffusive_rhs")
    handle = diffusion.attrs["operator_handle"]
    declaration = handle.declaration_ref or handle
    exact = "operation:"+_reference(declaration)
    root = "operation:"+_reference(diffusion.attrs["physical_balance"].balance.handle)
    assert exact != root
    assert next(row for row in operations.operations if row.identity == exact).guarantees[
        "numerical_method"]["method"] == "tensor_diffusion"
    code = emit_cpp_program(resolved.time, model=emitter)
    body = _emit_bricks(emitter._m)[1]
    assert "PreparedDiffusion<pops::kNativeDimension, 2, true>" in code
    assert "ctx.neg_div_flux_default_with_faces_into(" in code
    assert "program_state_ghost_depth = 2;" in body
    assert "program_only_storage = true" not in body
    assert max(operations.halo_requirements().values()) == 2
    selected = [row.guarantees.get("numerical_method") for row in operations.operations]
    assert any(method and method.get("method") == "tensor_diffusion" for method in selected)
    assert any(method and method.get("method") == "finite_volume" for method in selected)
    assert model._dsl._m._flux is authored_flux and model._dsl._m._eig is authored_eigenvalues
    assert not hasattr(model._dsl._m, "_program_state_ghost_depth")


def test_implicit_tensor_partition_is_not_charged_to_explicit_transport_bound():
    case, layout, model = composition_case(implicit=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    operations = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=operations)[0]
    code = emit_cpp_program(resolved.time, model=emitter)
    assert "PreparedDiffusion<pops::kNativeDimension, 2, true>" in code
    assert "ctx.neg_div_flux_default_with_faces_into(" in code
    assert "pops::runtime::program::PreparedSpatialResidual<" in code
    assert ".explicit_frequency()" not in code
    assert "transport_faces_" in code and "ctx.stage_exchange_batch(" in code
    assert code.count("combined_transport_diffusion_stability") == 1
    assert code.index("combined_transport_diffusion_stability") < code.index("_residual = [&]")
    assert "program_state_ghost_depth = 2;" in _emit_bricks(emitter._m)[1]


def test_competing_native_transport_configurations_are_still_refused():
    case, layout, _ = composition_case(competing_transport=True)
    with pytest.raises(ValueError, match="distinct runtime configurations"):
        pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("flux", ("rusanov", "hll"))
def test_shared_endpoint_frequency_contract_authenticates_supported_fluxes(flux):
    from types import SimpleNamespace
    from pops.numerics import reconstruction, riemann
    from pops.numerics.transport_frequency import transport_frequency_contract

    selected = SimpleNamespace(reconstruction=reconstruction.FirstOrder(),
                               riemann=riemann.Rusanov() if flux == "rusanov" else riemann.HLL())
    assert transport_frequency_contract(selected)["provider"] == "native_endpoint_model_wave_envelope"


@pytest.mark.parametrize("reconstruction_name,flux_name,waves", (
    ("WENO5", "Rusanov", None), ("FirstOrder", "Roe", None),
    ("FirstOrder", "HLLC", None), ("FirstOrder", "HLL", "einfeldt"),
    ("FirstOrder", "HLL", "davis"),
))
def test_shared_endpoint_frequency_contract_refuses_unproved_providers(
        reconstruction_name, flux_name, waves):
    from types import SimpleNamespace
    from pops.numerics import reconstruction, riemann
    from pops.numerics.transport_frequency import transport_frequency_contract

    wave_provider = None if waves is None else getattr(riemann.waves, waves.title())()
    flux = getattr(riemann, flux_name)(**({} if wave_provider is None else {"waves": wave_provider}))
    selected = SimpleNamespace(reconstruction=getattr(reconstruction, reconstruction_name)(), riemann=flux)
    with pytest.raises(ValueError, match="combined diffusion"):
        transport_frequency_contract(selected)


def test_grouped_composition_requests_retained_faces_without_evaluating_again():
    from pops.codegen.program_emit_control import _emit_contiguous_rhs_group

    case, layout, model = composition_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    operations = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=operations)[0]
    rate = next(value for value in resolved.time._values if value.op == "rhs")
    for target in ("system", "amr_system"):
        variables = {rate.inputs[0].id: "initial"}
        lines = []
        _emit_contiguous_rhs_group((rate,), {rate.block: 0}, variables, lines, 1000,
                                   target=target, model=emitter)
        requests = [line for line in lines if "ctx.rhs_group(" in line]
        assert len(requests) == 1
        assert ", &transport_faces_%d}" % rate.id in requests[0]
        assert ("program-flux-family" in requests[0]) == (target == "amr_system")
        assert not any("ctx.neg_div_flux" in line or "ctx.rhs_into(" in line for line in lines)
        assert ("accepted_transport", rate.id) in variables
