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
from tests.python.integration.runtime.test_combined_transport_derived_auxiliary import _build_derived_transport


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


@pytest.mark.parametrize("fault", ("neighbour", "untyped", "boolean_depth", "extra_effect"))
def test_affine_stencil_metadata_cannot_bypass_accepted_effect_proof(fault):
    from pops.time.stencil import StencilAccess
    program, authority = resolved(method="ssprk2")
    endpoint = next(iter(program._commits.values()))
    assert endpoint.attrs["stencil_access"] == StencilAccess.pointwise()
    attrs = dict(endpoint.attrs)
    if fault == "neighbour":
        attrs["stencil_access"] = StencilAccess.nearest_neighbour()
    elif fault == "untyped":
        attrs["stencil_access"] = StencilAccess.pointwise().to_data()
    elif fault == "boolean_depth":
        access = StencilAccess.pointwise()
        object.__setattr__(access, "required_ghost_depth", False)
        attrs["stencil_access"] = access
    else:
        attrs["evaluation_partition"] = "unproved"
    object.__setattr__(endpoint, "attrs", attrs)
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    assert certificate is None and ("stencil" in reason or "effect" in reason)


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


def _derived_case(method="euler", *, recipe="direct"):
    from pops._ir import ValueExpr
    from pops._ir.expr import Sign, Var
    from pops.fields import AnalyticAux, DerivedAux
    from pops.analytic import time
    case, layout = _build_derived_transport(16, .0001, method)
    state = next(v for v in case._time._values if v.op == "state")
    module = state.block._instance_registry.spec(state.block.local_id)["model"]
    velocity = module.aux_providers()["velocity"]
    if recipe == "chain":
        relay = module.aux_handle(module.aux_field("relay"))
        module.aux_provider(DerivedAux(relay, velocity.expression))
        expression = ValueExpr(relay)
    elif recipe == "unobserved":
        expression = ValueExpr(module.aux_handle(module.aux_field("unobserved")))
    elif recipe == "unobserved_field":
        expression = ValueExpr(module.field_handle(
            module.field_space("unobserved_field", components=("unobserved",))))
    elif recipe == "raw_capture":
        expression = Var("imposed", "field")
    elif recipe == "unknown_operation":
        expression = Sign(velocity.expression)
    elif recipe == "unknown_recipe":
        frame = layout.mesh.frame
        temporal = module.aux_handle(module.aux_field("temporal", frame=frame.canonical_id))
        module.aux_provider(AnalyticAux(temporal, time(case._time.clock), frame=frame))
        expression = ValueExpr(temporal)
    elif recipe == "state":
        expression = ValueExpr(module.state_handle(module.state_spaces()["U"]))
    elif recipe == "foreign":
        foreign = pops.Model("foreign derived owner").module
        # Same local spelling as the real aux target still has a different owner.
        expression = ValueExpr(foreign.aux_handle(foreign.aux_field("velocity")))
    elif recipe == "cycle":
        relay = module.aux_handle(module.aux_field("relay"))
        module.aux_provider(DerivedAux(relay, ValueExpr(velocity.target)))
        expression = ValueExpr(relay)
    else:
        assert recipe == "direct"
        return case, layout
    # Alter authoring data before source resolution. The proof must authenticate
    # these actual registered recipes, rather than a separately invented graph.
    object.__setattr__(velocity, "expression", expression)
    return case, layout


def _resolved_derived(method="euler", *, recipe="direct"):
    from pops.time._program.detach import detach_compiled_program
    case, layout = _derived_case(method, recipe=recipe)
    plan = pops.resolve(pops.validate(case), layout=layout)
    return detach_compiled_program(plan.time), ProgramModelGraph.from_resolved_blocks(plan.blocks)


@pytest.mark.parametrize("method", ("euler", "ssprk2"))
@pytest.mark.parametrize("recipe", ("direct", "chain"))
def test_static_input_observation_closes_actual_transitive_derived_auxiliary(method, recipe):
    import sys
    native_before = {name: value for name, value in sys.modules.items()
                     if name == "pops._pops" or name.startswith("pops._native.dim")}
    program, authority = _resolved_derived(method, recipe=recipe)
    before = program._serialize()
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    expected = (Fraction(1),) if method == "euler" else (Fraction(1,2), Fraction(1,2))
    assert not reason and certificate.coefficient == 1 and certificate.b == expected
    assert [weights for _, weights in accepted_diffusive_quadrature(program, model_authority=authority)] == [
        {1: value} for value in expected]
    assert program._serialize() == before
    assert {name: value for name, value in sys.modules.items()
            if name == "pops._pops" or name.startswith("pops._native.dim")} == native_before


@pytest.mark.parametrize("recipe", ("unobserved", "unobserved_field", "unknown_operation", "unknown_recipe"))
def test_static_derived_read_refuses_unobserved_or_unsupported_mathematical_captures(recipe):
    program, authority = _resolved_derived(recipe=recipe)
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    assert certificate is None and ("static" in reason or "unknown" in reason), reason


@pytest.mark.parametrize("recipe", ("state", "foreign", "cycle", "raw_capture"))
def test_static_derived_recipe_cannot_resolve_state_foreign_or_cyclic_source(recipe):
    case, layout = _derived_case(recipe=recipe)
    with pytest.raises((TypeError, ValueError)):
        pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("fault", ("source_recipe", "emitter_dependencies", "emitter_recipe", "resolved_recipe"))
def test_static_derived_read_reauthenticates_current_source_resolved_recipe_and_emitter(fault):
    from pops._ir.expr import Const
    from pops.codegen.program_emit_kernels import _model_impl
    program, authority = _resolved_derived(method="ssprk2", recipe="chain")
    token, _ = token_and_rate(program)
    module = authority.source_module_for_owner(authority.owner_for_block(token.block))
    emitter = _model_impl(authority.model_for_block(token.block))
    if fault == "source_recipe":
        object.__setattr__(module.aux_providers()["relay"], "expression", Const(.9))
    elif fault == "emitter_dependencies":
        routes = {key: dict(route) for key, route in emitter._auxiliary_provider_routes.items()}
        key = next(key for key in routes if key.component == "velocity")
        routes[key]["dependencies"] = ()
        routes[key]["contracts"] = ()
        object.__setattr__(emitter, "_auxiliary_provider_routes", routes)
    elif fault == "emitter_recipe":
        producer = next(route["producer"] for key, route in emitter._auxiliary_provider_routes.items()
                        if key.component == "relay")
        object.__setattr__(producer, "expression", Const(.9))
    else:
        import copy
        _, plan, identity = authority._resolved_provider_sources[token.block.local_id]
        evidence = copy.deepcopy(plan.to_data()["provider_evidence"])
        next(route for route in evidence["auxiliary_routes"]
             if route["key"]["component"] == "relay")["expression"] = {"const": "forged"}
        object.__setattr__(plan, "provider_evidence", evidence)
        assert plan.identity.token == identity
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    assert certificate is None and reason, fault


def test_observed_field_component_cannot_cover_a_same_named_auxiliary_declaration():
    from pops.codegen.component_provider_packs import (
        auxiliary_component_slot, resolve_component_provider_packs,
    )
    case, layout, _ = _build("variable", 16, .0001)
    state = next(v for v in case._time._values if v.op == "state")
    model = state.block._instance_registry.spec(state.block.local_id)["model"]
    module = getattr(model, "module", model)
    # These are separate registered Handles/ComponentKeys in one exact owner.
    # Coverage cannot follow from their shared source coordinate alone.
    module.aux_field("diffusivity")
    pack = resolve_component_provider_packs(module).auxiliary
    colliding = tuple(key for key in pack if key.component == "diffusivity")
    assert {key.space_kind for key in colliding} == {"aux", "field"}
    assert len(colliding) == 2 and len({key.owner_qid for key in colliding}) == 1
    with pytest.raises(ValueError, match="ambiguous"):
        auxiliary_component_slot(pack, owner_qid=str(module.owner_path.canonical()), name="diffusivity")
    # The current native lowering also refuses this source before it can become
    # a resolved ProgramModelGraph; never forge one to bypass that source guard.
    with pytest.raises(ValueError, match="distinct typed fields share native auxiliary name"):
        pops.resolve(pops.validate(case), layout=layout)
