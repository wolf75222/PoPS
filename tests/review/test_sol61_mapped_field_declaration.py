"""Source-only physical declaration authority; no Native admission claim."""
from fractions import Fraction
import pytest
import pops
from pops.model import FieldSpace, PhysicalDimension, PhysicalSupport
from pops.fields import FieldProblem
from tests.python.unit.fields.test_program_field_problem import field_case
from pops.time import FailRun

UNIT = PhysicalDimension()
LENGTH = PhysicalDimension((("length", 1),))
SUPPORT = PhysicalSupport((("x", "declared slab"),))


def declared_case():
    case, field, problem, program, values, point = field_case()
    declared = FieldSpace("temperature", components=("temperature",), support=SUPPORT,
                         units=(UNIT,), sampling="cell")
    from pops.model import Handle, OwnerPath
    from pops.math import laplacian
    from pops.fields import FieldBoundary, SharedMeanGauge, bcs
    unknown = Handle("temperature", kind="field", owner=OwnerPath.model("declared physical field"))
    physical = FieldProblem("declared", unknowns=(unknown,), equations=(-laplacian(unknown) == problem.equations[0].rhs,),
        boundaries=(FieldBoundary(unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
        gauge=SharedMeanGauge((unknown,)), unknown_spaces={unknown: declared}, coordinate_units=(LENGTH,))
    # Re-register under a distinct public field authority, with the real method.
    from pops.fields import FieldDiscretization
    from pops.fields.methods import CellCenteredSecondOrder
    from pops.solvers import CG
    observed = case.field(physical, FieldDiscretization(method=CellCenteredSecondOrder(),
        boundaries=(), observation_axes=(0,), solver=CG(max_iter=4000, rel_tol=1e-11, abs_tol=1e-12)))
    solution = observed.observe(program.solve(observed, values=values, at=point).consume(action=FailRun()))
    return physical, declared, solution, observed, program


def test_declared_consumed_scalar_and_gradient_have_equation_owned_units():
    physical, space, solution, field, program = declared_case()
    scalar = solution[field[physical.unknowns[0]]]
    assert scalar.space == space and scalar.space.sampling == "cell"
    gradient = solution.gradient(field[physical.unknowns[0]], dimension=2)
    assert gradient.space.units == (PhysicalDimension((("length", -1),)), None)
    assert gradient.space.support == SUPPORT
    from pops.fields._observation_contract import validate_field_observation, validate_field_gradient
    validate_field_observation(scalar)
    validate_field_gradient(gradient)
    from pops.time._graph.base import _canonical_int
    assert _canonical_int(solution.packed.inputs[0].inputs[0].attrs["solve_request"]["physical_problem"]["field_problem"]["schema_version"]) == 2
    assert len(solution.packed.inputs[0].inputs[0].attrs["solve_request"]["physical_problem"]["field_problem"]["observation_spaces"]) == 1


def test_legacy_field_problem_wire_has_no_observation_metadata():
    case, _, problem, _, _, _ = field_case()
    problem = problem.resolve_references(case.resolve)
    assert set(problem.to_data()) == {"schema_version", "unknowns", "equations", "boundaries", "gauge", "branch", "outputs"}
    assert problem.to_data()["schema_version"] == 1


@pytest.mark.parametrize("change", ("missing", "sampling", "units"))
def test_declaration_refuses_missing_or_untyped_physical_authority(change):
    _, _, problem, _, _, _ = field_case()
    spaces = {} if change == "missing" else {problem.unknowns[0]: FieldSpace("bad", components=("p",),
        support=SUPPORT, units=None if change == "units" else (UNIT,),
        sampling="cell_average" if change == "sampling" else "cell")}
    with pytest.raises(ValueError, match="spaces|FieldSpace"):
        FieldProblem(problem.name, unknowns=problem.unknowns, equations=problem.equations,
                     boundaries=problem.boundaries, gauge=problem.gauge, unknown_spaces=spaces)


@pytest.mark.parametrize("axis,factor", [(1, 1), (True, 1), (0, True), (0, Fraction(10**500))])
def test_invalid_mapping_port_refuses_before_graph_mutation(axis, factor):
    physical, _, solution, field, program = declared_case()
    before = program._serialize()
    with pytest.raises((TypeError, ValueError)):
        solution.mapping_port(field[physical.unknowns[0]], derivative_axis=axis, factor=factor)
    assert program._serialize() == before


def test_mapping_port_keeps_signed_exact_factor_and_field_space():
    physical, space, solution, field, program = declared_case()
    before = program._serialize()
    port = solution.mapping_port(field[physical.unknowns[0]], derivative_axis=0, factor=Fraction(-3, 2))
    assert port.to_data()["factor"] == {"numerator": "-3", "denominator": "2"}
    assert port.quantity_space().units == (PhysicalDimension((("length", -1),)),)
    assert program._serialize() == before
