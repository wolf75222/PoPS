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
