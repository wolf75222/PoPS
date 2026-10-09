"""Resolved static external providers preserve exact authored SSP compositions."""
from fractions import Fraction
from types import MappingProxyType
import pytest
import pops
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
from pops.model.provider_pack import ProviderPack, ProviderEntry
from pops.time.field_context import FieldContext
from tests.python.integration.runtime.test_public_diffusion_matrix import _build
from tests.python.integration.runtime.test_public_drift_diffusion_matrix import _solved_potential_case


def resolved(kind="variable", method="euler"):
    case, layout, _ = _build(kind, 16, .0001, method=method)
    plan = pops.resolve(pops.validate(case), layout=layout)
    from pops.time._program.detach import detach_compiled_program
    return detach_compiled_program(plan.time), ProgramModelGraph.from_resolved_blocks(plan.blocks)


def token_and_rate(program):
    token = next(v for v in program._values if v.op == "input_fields")
    rate = next(v for v in program._values if v.op == "diffusive_rhs")
    return token, rate


@pytest.mark.parametrize("kind", ("variable", "diagonal_linear", "diagonal_smooth"))
@pytest.mark.parametrize("method", ("euler", "ssprk2"))
def test_actual_matrix_fixtures_keep_exact_static_fields_and_authored_weights(kind, method):
    program, authority = resolved(kind, method)
    before = program._serialize()
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    assert not reason and certificate.coefficient == 1
    expected = (Fraction(1),) if method == "euler" else (Fraction(1,2), Fraction(1,2))
    assert certificate.b == expected
    assert [weights for _, weights in accepted_diffusive_quadrature(program, model_authority=authority)] == [
        {1: value} for value in expected]
    assert program._serialize() == before
    assert all(v.block._instance_registry is None for v in program._values if v.block is not None)


@pytest.mark.parametrize("fault", (
    "wrong_point", "wrong_context_stage", "wrong_context_components", "different_rate",
    "wrong_field_owner", "wrong_formal_space", "missing_input", "extra_attr",
    "changed_emitted_producer", "changed_emitted_slot", "missing_resolved_source",
    "wrong_resolved_identity", "missing_actual_component", "negative_weight", "unused_mutation",
))
def test_static_proof_refuses_changed_observation_source_or_accepted_equation(fault):
    program, authority = resolved(method="ssprk2")
    token, rate = token_and_rate(program)
    if fault == "wrong_point":
        object.__setattr__(token, "point", next(iter(program._commits.values())).point)
    elif fault == "wrong_context_stage":
        object.__setattr__(token, "field_context", FieldContext(token.attrs["field"],
            ((token.block, token.inputs[0].id+1),), token.space.components))
    elif fault == "wrong_context_components":
        object.__setattr__(token, "field_context", FieldContext(token.attrs["field"],
            ((token.block, token.inputs[0].id),), token.space.components[:-1]))
    elif fault == "different_rate":
        _, other = resolved("diagonal_linear")
        other_module = next(iter(other.source_modules_by_owner.values()))
        attrs = dict(token.attrs); attrs["for_rate"] = other_module.operator_handle("physical_balance")
        object.__setattr__(token, "attrs", attrs)
    elif fault == "wrong_field_owner":
        other, _ = resolved("diagonal_linear")
        attrs = dict(token.attrs); attrs["field"] = token_and_rate(other)[0].attrs["field"]
        object.__setattr__(token, "attrs", attrs)
    elif fault == "wrong_formal_space":
        from pops.model import FieldSpace
        object.__setattr__(token, "space", FieldSpace("fields", components=("diffusivity",)))
    elif fault == "missing_input":
        object.__setattr__(token, "inputs", ())
    elif fault == "extra_attr":
        object.__setattr__(token, "attrs", {**token.attrs, "opaque": True})
    elif fault in {"changed_emitted_producer", "changed_emitted_slot", "missing_actual_component"}:
        emitter = authority.model_for_block(token.block)
        pack = emitter._auxiliary_provider_pack
        rows = [(key, pack.contract(key), pack.declared_entry(key)) for key in pack]
        if fault == "missing_actual_component": rows = rows[:-1]
        else:
            key, contract, entry = rows[0]
            entry = ProviderEntry("computed" if fault == "changed_emitted_producer" else entry.producer,
                                  entry.available, entry.slot if fault == "changed_emitted_producer" else 1)
            rows[0] = key, contract, entry
        from pops.codegen.program_emit_kernels import _model_impl
        object.__setattr__(_model_impl(emitter), "_auxiliary_provider_pack", ProviderPack(rows, capacity=pack.capacity))
    elif fault == "missing_resolved_source":
        authority._resolved_provider_sources = MappingProxyType({})
    elif fault == "wrong_resolved_identity":
        name = token.block.local_id; instance, source, identity = authority._resolved_provider_sources[name]
        authority._resolved_provider_sources = MappingProxyType({name: (instance, source, identity+"changed")})
    elif fault == "negative_weight":
        endpoint = next(iter(program._commits.values()))
        object.__setattr__(endpoint, "attrs", {"coeffs": ({0:Fraction(3,2)}, {0:Fraction(-1,2)}, {1:Fraction(3,2)})})
    else:
        object.__setattr__(program._values[-1], "op", "project")
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    assert certificate is None and reason, fault


def test_live_or_detached_token_without_resolved_provider_source_is_not_certified():
    program, _ = resolved()
    certificate, reason = prove_accepted_update_ssp(program)
    assert certificate is None and reason


def test_actual_publication_reprojects_authored_runtime_input_and_refuses_static_token():
    case, layout = _solved_potential_case(16, .0001)
    p = case._time
    rate = next(v for v in p._values if v.op == "diffusive_rhs")
    token = p.input_fields(rate.inputs[0], for_rate=rate.attrs["operator_handle"])
    object.__setattr__(rate, "inputs", (rate.inputs[0], token))
    object.__setattr__(rate, "field_context", token.field_context)
    plan = pops.resolve(pops.validate(case), layout=layout)
    authority = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    actual_rate = next(v for v in plan.time._values if v.op == "diffusive_rhs")
    pack = authority.resolved_provider_pack_for_block(actual_rate.block)
    assert any(entry.producer != "runtime_input" for entry in (pack.declared_entry(key) for key in pack))
    certificate, reason = prove_accepted_update_ssp(plan.time, model_authority=authority)
    assert certificate is None and ("requires runtime_input" in reason or "publication" in reason)


@pytest.mark.parametrize("fault", ("stencil_radius", "native_route"))
def test_current_resolved_plan_payload_cannot_borrow_its_cached_issued_identity(fault):
    from dataclasses import replace
    from pops.codegen.resolved_operations import ResolvedOperationPlan
    program, authority = resolved(method="ssprk2")
    token, _ = token_and_rate(program)
    _instance, plan, issued_identity = authority._resolved_provider_sources[token.block.local_id]
    changed = tuple(replace(operation, **(
        {"stencil_radius": operation.stencil_radius + 1} if fault == "stencil_radius"
        else {"native_route": "forged-route"})) for operation in plan.operations)
    object.__setattr__(plan, "operations", changed)
    assert plan.identity.token == issued_identity
    with pytest.raises((TypeError, ValueError)):
        ResolvedOperationPlan.from_data(plan.to_data())
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    assert certificate is None and reason
