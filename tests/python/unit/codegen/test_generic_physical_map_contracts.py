"""Discriminating public support contracts independent of a field-solve scenario."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
import pytest

from pops.model import PhysicalDimension, PhysicalSupport
from pops.mesh import AxisQuadrature, PhysicalSupportMap, VelocityQuadrature
from pops.runtime._physical_mapping import physical_mapping_schedule, validate_physical_geometry

UNIT = PhysicalDimension()
X = ("q", "position-domain")
V = ("p", "momentum-domain")
W = ("w", "second-momentum-domain")


def geometry(shape, lower, upper, periodicity=None):
    return SimpleNamespace(shape=shape, lower=lower, upper=upper,
                           periodicity=(True,) * len(shape) if periodicity is None else periodicity,
                           coordinate_system="cartesian")


def test_explicit_signed_axis_measure_and_permuted_support():
    source = PhysicalSupport((V, X))
    target = PhysicalSupport((X,))
    signed = AxisQuadrature(0, -2, 2, 4, UNIT,
                            weights=(Fraction(-3, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(3, 2)))
    mapping = PhysicalSupportMap(source, target, reductions=(signed,), target_axes=(1,))
    assert mapping.source_to_target == (-1, 1)
    assert mapping.operation_abi == 2
    assert mapping.reductions[0].weights == (Fraction(-3, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(3, 2))
    source_geometry = geometry((4, 7), (-2, 0), (2, 1))
    target_geometry = geometry((1, 7), (0, 0), (1, 1))
    validate_physical_geometry(SimpleNamespace(physical_map=mapping), source_geometry, target_geometry)
    with pytest.raises(ValueError, match="quadrature"):
        validate_physical_geometry(SimpleNamespace(physical_map=mapping),
                                   geometry((4, 7), (-1, 0), (2, 1)), target_geometry)
    altered = replace(mapping, reductions=(replace(signed, weights=(1, 1, 1, 1)),))
    assert altered.to_data() != mapping.to_data()


def test_two_reduction_axes_dim3_and_explicit_extension_are_distinct():
    mapping = PhysicalSupportMap(PhysicalSupport((V, X, W)), PhysicalSupport((X,)),
        reductions=(AxisQuadrature(2, 0, 3, 3, UNIT, weights=(1, 2, 4)),
                    AxisQuadrature(0, -1, 1, 2, UNIT)), target_axes=(2,))
    assert mapping.source_to_target == (-1, 2, -1)
    assert tuple(row.axis for row in mapping.reductions) == (0, 2)
    assert mapping.native_dimension == 3
    validate_physical_geometry(SimpleNamespace(physical_map=mapping),
        geometry((2, 5, 3), (-1, 0, 0), (1, 1, 3)), geometry((1, 1, 5), (0, 0, 0), (1, 1, 1)))
    extension = PhysicalSupportMap(mapping.target_support, mapping.source_support,
                                  source_axes=(2,), native_dimension=3)
    assert extension.source_to_target == (-1, -1, 1)
    assert extension.operation_abi == 3
    assert extension.to_data()["storage"]["inverse_closure"] is False
    with pytest.raises(ValueError, match="exactly once"):
        replace(mapping, reductions=(mapping.reductions[0],))
    with pytest.raises(ValueError, match="valid source_axes"):
        replace(mapping, source_axes=(0, 0, 2))


def test_uniform_compatibility_map_survives_dataclass_reconstruction():
    mapping = PhysicalSupportMap(PhysicalSupport((V, X)), PhysicalSupport((X,)),
                                 VelocityQuadrature(-2, 2, 4, UNIT))
    assert mapping.reductions[0].axis == 0
    assert replace(mapping) == mapping
    explicit = PhysicalSupportMap(mapping.source_support, mapping.target_support,
                                  reductions=mapping.reductions)
    assert explicit.to_data() == mapping.to_data()


def transfer(name, source, target, *, after=False, operation=2, subject=None):
    return SimpleNamespace(mapping_id=name, operation_abi=operation,
        source_layout_id=source, target_layout_id=target, source_subject_id=source + "_quantity",
        target_subject_id=subject or target + "_" + name,
        synchronization_uri="pops://synchronization/" + ("after-source-step@1" if after else "before-step@1"))


def test_multiple_maps_schedule_uses_explicit_effects_not_opcode_or_declaration_order():
    rows = (transfer("accepted_broadcast", "a", "b", operation=3),
            transfer("committed_reduction", "b", "c", after=True),
            transfer("other_moment", "a", "c"),
            transfer("observation", "c", "d", after=True, operation=3))
    plan = physical_mapping_schedule(rows, ("d", "b", "a", "c", "independent"))
    assert plan == physical_mapping_schedule(tuple(reversed(rows)), ("a", "b", "c", "d", "independent"))
    assert plan.accepted_captures == ("accepted_broadcast", "other_moment")
    positions = {event: index for index, event in enumerate(plan.events)}
    for before, after in ((('map', 'accepted_broadcast'), ('step', 'b')),
                          (('step', 'b'), ('map', 'committed_reduction')),
                          (('map', 'committed_reduction'), ('step', 'c')),
                          (('step', 'c'), ('map', 'observation')),
                          (('map', 'observation'), ('step', 'd'))):
        assert positions[before] < positions[after]
    assert sum(kind == "step" for kind, _ in plan.events) == 5


def test_cycle_and_ambiguous_atomic_overwrites_refuse_instead_of_inventing_stages():
    with pytest.raises(ValueError, match="continuation"):
        physical_mapping_schedule((transfer("ab", "a", "b", after=True),
                                   transfer("ba", "b", "a", after=True)), ("a", "b"))
    with pytest.raises(ValueError, match="overwrite"):
        physical_mapping_schedule((transfer("ab", "a", "b", subject="same"),
                                   transfer("cb", "c", "b", after=True, subject="same")), ("a", "b", "c"))


def test_provider_package_contains_data_and_common_kernel_only(tmp_path):
    from pops.mesh.native_physical_mapping import _native_source
    manifest = SimpleNamespace(component_id="test.component", semantic_digest=SimpleNamespace(token="semantic"),
                               manifest_digest=SimpleNamespace(token="manifest"))
    mapping = PhysicalSupportMap(PhysicalSupport((V, X)), PhysicalSupport((X,)),
                                 reductions=(AxisQuadrature(0, -1, 1, 2, UNIT, weights=(-2, 3)),))
    source = _native_source(mapping, manifest).decode()
    assert '#include <pops/runtime/dynamic/physical_support_transfer.hpp>' in source
    assert 'apply_physical_support_transfer(descriptor, request, status)' in source
    assert 'const double weights[] = {-2.0, 3.0};' in source
    assert 'for (' not in source and 'parallel_for' not in source
    tiny = replace(mapping, reductions=(replace(mapping.reductions[0], weights=(Fraction(1, 10**500), 1)),))
    with pytest.raises(ValueError, match="finite float64"):
        _native_source(tiny, manifest)


def test_public_field_program_retains_nonzero_internal_stage(tmp_path):
    from tests.python.integration.runtime.test_physical_support_mapping import resolve_physical_case
    from pops.codegen.program_field_plan import _nodes
    _, resolved = resolve_physical_case(tmp_path, field_stage=Fraction(1, 2))
    solves = [node for node in _nodes(resolved.time) if node.op == "solve_linear"]
    assert len(solves) == 1
    data = solves[0].point.to_data()
    coordinates = tuple(data.get("partitions", {"main": data}).values())
    assert all(row["offset"] == {"kind": "rational", "numerator": "1", "denominator": "2"}
               for row in coordinates)


def test_full_axis_reduction_has_explicit_scalar_support_in_native_dim1():
    scalar = PhysicalSupport(())
    assert scalar.to_data() == {"kind": "physical_support", "coordinates": []}
    mapping = PhysicalSupportMap(PhysicalSupport((V,)), scalar,
                                 reductions=(AxisQuadrature(0, -1, 1, 2, UNIT),))
    assert mapping.native_dimension == 1
    assert mapping.source_to_target == (-1,)
    validate_physical_geometry(SimpleNamespace(physical_map=mapping),
                               geometry((2,), (-1,), (1,)), geometry((1,), (0,), (1,)))
    assert PhysicalSupport.from_data(scalar.to_data()) == scalar
    with pytest.raises(TypeError):
        PhysicalSupport(None)
