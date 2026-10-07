"""Pure generated producer defaults follow actual consumer events; Source only."""
import re

import pytest
import pops
from pops._ir import ValueExpr
from pops.analytic import time, x
from pops.codegen._compile_emit import _emit_auxiliary_route_registration
from pops.codegen.module_lowering import _module_to_model, lower_and_validate
from pops.codegen.module_codegen import emit_cpp_elliptic_field
from pops.domain import Rectangle
from pops.fields import AnalyticAux, DerivedAux, FieldOutput, InputAux
from pops.frames import Cartesian2D
from pops.math import Reaction, laplacian
from pops.model import Module
from pops.time import Program


EVENTS = ("before_residual", "before_field_solve")


def _policies(source):
    return re.findall(r"AuxiliaryEvaluationPolicy\{.*?AuxiliaryFreshness::(?:once|evaluation)\}", source)


@pytest.mark.parametrize("target", ["system", "amr_system"])
@pytest.mark.parametrize("components", [("q",), ("q", "passive")])
def test_pure_input_aux_expression_default_is_recomputable_for_each_requested_consumer_event(target, components):
    module = Module("arbitrary pure expression")
    module.state_space("State", components)
    imposed = module.aux_handle(module.aux_field("supplied_data"))
    derived = module.aux_handle(module.aux_field("computed_data"))
    module.aux_provider(InputAux(imposed))
    module.aux_provider(DerivedAux(derived, ValueExpr(imposed) * ValueExpr(imposed)))
    carrier = _module_to_model(module)._m
    source = _emit_auxiliary_route_registration(carrier, target=target)
    policies = _policies(source)
    assert len(policies) == 2
    once = next(policy for policy in policies if "Freshness::once" in policy)
    evaluated = next(policy for policy in policies if "Freshness::evaluation" in policy)
    assert once == "AuxiliaryEvaluationPolicy{AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once}"
    assert "std::vector<AuxiliaryEvaluationEvent>{" in evaluated
    assert re.findall(r"AuxiliaryEvaluationEvent::(\w+)", evaluated) == list(EVENTS)
    assert "Kokkos::parallel_for" in source
    assert "candidate->find" in source  # the exact prepared dependency closure still supplies values
    assert "actual_boundary" not in evaluated  # policy never fabricates a point or clock


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_actual_field_rhs_analytic_derived_closure_emits_event_defaults_without_retiming_source(target):
    frame = Rectangle("declared frame", (0,0), (1,1)).frame(Cartesian2D())
    program = Program("declared consuming clock")
    model = pops.Model("unrelated physical owner", frame=frame)
    state = model.state("State", components=("density",))
    imposed = model.auxiliary("analytic_data", frame=frame.canonical_id)
    transformed = model.auxiliary("derived_data", frame=frame.canonical_id)
    field = model.field("potential")
    operation = model.field_operator("arbitrary_rhs", unknown=field,
        equation=-laplacian(field) + Reaction(field, 1) == transformed * state[0],
        outputs=(FieldOutput("observation", field),))
    module = model.module
    module.aux_provider(AnalyticAux(module.aux_handle(module.aux()["analytic_data"]),
                                  x(frame) + time(program.clock), frame=frame))
    module.aux_provider(DerivedAux(module.aux_handle(module.aux()["derived_data"]),
        ValueExpr(module.aux_handle(module.aux()["analytic_data"])) * 2))
    lowered, _ = lower_and_validate(model, facade=model)
    carrier = getattr(lowered, "_m", lowered)
    source = _emit_auxiliary_route_registration(carrier, target=target)
    derived_policies = [policy for policy in _policies(source) if "Freshness::evaluation" in policy]
    assert len(derived_policies) == 2
    for policy in derived_policies:
        assert re.findall(r"AuxiliaryEvaluationEvent::(\w+)", policy) == list(EVENTS)
    assert 'context.point.require_physical_time(' in source
    assert program.clock.qualified_id in source
    assert "geometry.cell_coordinate" in source
    field_source = emit_cpp_elliptic_field(carrier, operation.name, "ArbitraryFieldRhs")
    assert "elliptic_rhs(const State& U, const pops::ProviderValues<1>& a)" in field_source
    assert "pops::provider_value<0>(a)" in field_source
    # Output ownership retains its existing initialization/once policy.
    output_policies = [policy for policy in _policies(source) if "Freshness::once" in policy]
    assert output_policies
    assert all(policy == "AuxiliaryEvaluationPolicy{AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once}"
               for policy in output_policies)


def test_provider_free_model_adds_no_derived_default_policy_or_event_vector():
    module = Module("provider free")
    module.state_space("State", ("q",))
    source = _emit_auxiliary_route_registration(_module_to_model(module)._m)
    assert "AuxiliaryProviderKind::derived" not in source
    assert "std::vector<AuxiliaryEvaluationEvent>" not in source
    assert "AuxiliaryFreshness::evaluation" not in source


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_unrelated_temporal_aux_does_not_gain_legacy_topology_event_permission(target):
    frame = Rectangle("unrelated declared frame", (0, 0), (1, 1)).frame(Cartesian2D())
    program = Program("unrelated declared clock")
    model = pops.Model("unrelated temporal owner", frame=frame)
    model.state("State", components=("passive",))
    model.auxiliary("temporal_observation", frame=frame.canonical_id)
    module = model.module
    module.aux_provider(AnalyticAux(module.aux_handle(module.aux()["temporal_observation"]),
                                  time(program.clock), frame=frame))
    lowered, _ = lower_and_validate(model, facade=model)
    source = _emit_auxiliary_route_registration(getattr(lowered, "_m", lowered), target=target)
    policy, = _policies(source)
    assert re.findall(r"AuxiliaryEvaluationEvent::(\w+)", policy) == list(EVENTS)
    for diagnostic in ("initialization", "after_regrid", "nonlinear_iteration", "output"):
        assert "AuxiliaryEvaluationEvent::" + diagnostic not in policy
    # Permissions never synthesize physical authority for a diagnostic clock.
    assert "context.point.require_physical_time(" in source
    assert program.clock.qualified_id in source
    assert "qualify_physical_evaluation(" not in source
