"""Exact physical identities, quadrature and partitioned native emission contracts."""
from dataclasses import replace
from fractions import Fraction
import pytest
from pops.model import PhysicalDimension, PhysicalSupport
from pops.mesh import PhysicalSupportMap, VelocityQuadrature
from tests.python.integration.runtime.test_physical_support_mapping import (
    PHASE, PHYSICAL, UNITLESS, resolve_physical_case)


def test_exact_velocity_quadrature_and_homonymous_supports():
    q = VelocityQuadrature(-2, 2, 12, UNITLESS)
    assert q.weight == Fraction(1, 3)
    assert q.weight * q.cells == 4
    foreign = PhysicalSupport((("x", "other-population-domain"),))
    with pytest.raises(ValueError, match="support identities"):
        PhysicalSupportMap(PHASE, foreign, q)
    with pytest.raises(ValueError, match="support identities"):
        PhysicalSupportMap(PhysicalSupport(tuple((str(i), "domain") for i in range(6))), PHYSICAL, q)


@pytest.fixture(scope="module")
def physical_resolved(tmp_path_factory):
    return resolve_physical_case(tmp_path_factory.mktemp("physical-source"))[1]


def test_missing_map_and_wrong_direction_refused(physical_resolved):
    moment = next(row.requirement for row in physical_resolved.layout_plan.mappings
                  if row.requirement.physical_map.quadrature is not None)
    with pytest.raises(ValueError, match="explicit physical_map"):
        replace(moment, physical_map=None)
    with pytest.raises(ValueError, match="direction"):
        replace(moment, physical_map=PhysicalSupportMap(PHYSICAL, PHASE))
    with pytest.raises(ValueError, match="no inverse closure"):
        replace(moment, reverse_of="invented inverse")
    with pytest.raises(ValueError, match="units disagree"):
        replace(moment, physical_map=PhysicalSupportMap(PHASE, PHYSICAL,
            VelocityQuadrature(-2, 2, 12, PhysicalDimension((("velocity", Fraction(1)),)))))


def test_physical_layout_geometry_is_not_an_average_or_rank_guess(physical_resolved):
    from pops.runtime._physical_mapping import validate_physical_geometry
    plan = physical_resolved.layout_plan
    moment = next(row.requirement for row in plan.mappings if int(row.requirement.operation) == 2)
    phase = plan.normalized(moment.source_layout).native_spatial_layout
    field = plan.normalized(moment.target_layout).native_spatial_layout
    validate_physical_geometry(moment, phase, field)
    from types import SimpleNamespace
    bad = SimpleNamespace(shape=phase.shape, lower=(0, -1), upper=phase.upper,
        periodicity=phase.periodicity, coordinate_system=phase.coordinate_system)
    with pytest.raises(ValueError, match="quadrature"):
        validate_physical_geometry(moment, bad, field)


def test_actual_field_and_coupling_partitions_emit_natively(physical_resolved):
    from pops.codegen.program_slicing import slice_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph
    plan = physical_resolved
    assignments = {row.subject.local_id: row.layout for row in plan.layout_plan.assignments
                   if row.subject_kind == "block"}
    sources = []
    for layout in plan.layout_plan.layouts:
        selected = tuple(row for row in plan.blocks if assignments[row.name] == layout.handle)
        sliced = slice_program(plan.time, tuple(row.name for row in selected))
        from pops.time._program.detach import detach_compiled_program
        sliced = detach_compiled_program(sliced)
        source = emit_program_graph(sliced.to_graph(), lowering_program=sliced,
            model_graph=ProgramModelGraph.from_resolved_blocks(selected), target="system", field_plans={})
        sources.append(source)
    assert sum("publish_field_components" in source for source in sources) == 1
    assert sum(any(b"pops_explicit_physical_map" in payload.content
                       for payload in row.component_type.package.payloads)
               for row in plan.component_inputs) == 2
    assert plan.resolved_dimension == 2
    assert len(plan.program_field_plans) == 1


def test_detached_physical_child_bind_uses_consumed_field_provider(physical_resolved):
    """Exercise the child argument/bind seam with the real detached plan and sliced Program.

    Binary compilation is outside this source test: model metadata comes from the actual
    lowering adapter, while provider authority comes from CompiledPlanRecord as it does at bind.
    """
    from types import SimpleNamespace
    from pops.codegen._compiled_artifact import CompiledPlanRecord
    from pops.codegen.inspect_compiled import _build_arguments, _auxiliary_packs_by_block
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_slicing import slice_program
    from pops.runtime._bind_validation import validate_install_arguments
    from pops.time._program.detach import detach_compiled_program

    plan = physical_resolved
    compiled = SimpleNamespace(plan=CompiledPlanRecord.from_resolved(plan),
                               bind_schema=plan.bind_schema, target=plan.target)
    assignments = {row.subject.local_id: row.layout for row in plan.layout_plan.assignments
                   if row.subject_kind == "block"}
    packs = _auxiliary_packs_by_block(compiled)
    field_pack = packs["field_observation"]
    field_key = next(key for key in field_pack if key.component == "sample_grad_x")
    claim, = next(block for block in compiled.plan.blocks if block.name == "field_observation"
                  ).resolved_operations.provider_evidence["program_field_publications"]
    assert field_pack.declared_entry(field_key).producer == claim["producer"]
    assert claim["producer"] != "runtime_input"
    saw_field_child = False
    for layout in plan.layout_plan.layouts:
        selected = tuple(row for row in plan.blocks if assignments[row.name] == layout.handle)
        sliced = detach_compiled_program(slice_program(plan.time, tuple(row.name for row in selected)))
        rows = []
        for block in selected:
            adapter, module = lower_and_validate(block.model, state_space=block.state_spaces[0],
                                                resolved_operations=block.resolved_operations)
            space = module.state_spaces()[block.state_spaces[0]]
            rows.append(SimpleNamespace(block_name=block.name, state_space=space.name,
                n_vars=len(space.components), cons_names=space.components, params={},
                provider_components=tuple(adapter._m._provider_components), model=adapter))
        arguments = _build_arguments(compiled, sliced, tuple(rows))
        assert arguments.aux == {}
        child_view = SimpleNamespace(arguments=lambda: arguments)
        validate_install_arguments(SimpleNamespace(block_names=lambda: ()), child_view,
                                   {row.name: {} for row in selected}, {}, {}, field_plans={})
        publications = [value for value in sliced._values if value.op == "field_publication"]
        if "field_observation" in arguments.instances:
            saw_field_child = True
            publication, = publications
            assert publication.attrs["field_problem_identity"] == claim["producer"]
            pending, reachable = [publication], {}
            while pending:
                value = pending.pop()
                if value.id not in reachable:
                    reachable[value.id] = value
                    pending.extend(value.inputs)
            assert any(value.op == "solve_outcome" for value in reachable.values())
        else:
            assert not publications
    assert saw_field_child


def test_physical_output_does_not_exempt_foreign_input_homonym(physical_resolved):
    from types import SimpleNamespace
    from pops.codegen._compiled_artifact import CompiledPlanRecord
    from pops.codegen.inspect_compiled import _build_aux_arguments, _auxiliary_packs_by_block
    from pops.model.provider_pack import ProviderEntry, ProviderPack
    from pops.runtime._bind_validation import collect_missing_arguments

    packs = _auxiliary_packs_by_block(SimpleNamespace(
        plan=CompiledPlanRecord.from_resolved(physical_resolved)))
    output_pack = packs["field_observation"]
    key = next(iter(output_pack))
    foreign_key = replace(key, owner_qid=key.owner_qid + "/foreign-model")
    input_pack = ProviderPack([(foreign_key, output_pack.contract(key),
                               ProviderEntry("runtime_input", True, 0))], capacity=1)
    rows = tuple(SimpleNamespace(block_name=name, provider_components=(key.component,))
                 for name in ("field_observation", "foreign"))
    arguments = _build_aux_arguments(rows, {},
        {"field_observation": output_pack, "foreign": input_pack})
    assert arguments == {key.component: {"layout": "cell", "required": True}}
    missing = collect_missing_arguments(SimpleNamespace(aux=arguments), set(), set(), set())
    assert len(missing) == 1 and key.component in missing[0]
    # Without the authenticated publication, even this exact local spelling is an input.
    unbound = ProviderPack([(key, output_pack.contract(key),
                            ProviderEntry("runtime_input", True, 0))], capacity=1)
    assert _build_aux_arguments(rows[:1], {}, {"field_observation": unbound}) == arguments
    with pytest.raises(ValueError, match="differs from its resolved ProviderPack"):
        _build_aux_arguments((SimpleNamespace(block_name="field_observation",
                              provider_components=(key.component, "undeclared")),), {},
                             {"field_observation": output_pack})


def test_representation_and_unknown_units_fail_before_provider_binding():
    from types import SimpleNamespace
    from pops.model import StateSpace
    def port(support, *, sampling="cell_average", units=(UNITLESS,)):
        space = StateSpace("homonymous", ("value",), support=support,
                           representation="conservative", sampling=sampling, units=units)
        return SimpleNamespace(subject=SimpleNamespace(space=space))
    moment = PhysicalSupportMap(PHASE, PHYSICAL, VelocityQuadrature(-2, 2, 12, UNITLESS))
    with pytest.raises(ValueError, match="representation/sampling"):
        moment.validate_ports(port(PHASE, sampling="point"), port(PHYSICAL))
    with pytest.raises(ValueError, match="explicit source and target units"):
        moment.validate_ports(port(PHASE, units=(None,)), port(PHYSICAL))
