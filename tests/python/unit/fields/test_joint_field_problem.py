"""Physical tuple, exact ownership and shared-gauge contracts for general fields."""
from __future__ import annotations

import pytest

from pops.fields import (
    FieldBoundary, FieldDiscretization, FieldOperator, FieldOutput, FieldProblem,
    FieldProblemError, FieldStorageBinding, SharedMeanGauge,
)
from pops.fields import bcs
from pops.fields.definition import normalize_field_definition
from pops.fields.gauges import MeanValueGauge
from pops.fields.methods import CellCenteredSecondOrder
from pops.fields.nullspace import ConstantNullspace
from pops.math import ValueExpr, laplacian
from pops._ir.elliptic import Reaction
from pops.model import Handle, OwnerPath
from pops.solvers import CartesianCG


def handles():
    owner = OwnerPath.model("joint")
    return tuple(Handle(name, kind="field", owner=owner) for name in ("phi1", "phi2"))


def joint(*, gauge=None, boundaries=()):
    phi1, phi2 = handles()
    equations = (
        -laplacian(phi1) + Reaction(phi1, 2) - Reaction(phi2, 2) == 1.5,
        -laplacian(phi2) + Reaction(phi2, 2) - Reaction(phi1, 2) == -1.5,
    )
    return FieldProblem("joint", unknowns=(phi1, phi2), equations=equations,
                        gauge=SharedMeanGauge((phi1, phi2)) if gauge is None else gauge,
                        boundaries=boundaries)


def test_joint_problem_keeps_one_equation_tuple_and_one_physical_gauge():
    problem = joint()
    assert len(problem.unknowns) == len(problem.equations) == 2
    assert problem.dependencies() == ()
    assert problem.to_data()["gauge"]["kind"] == "shared_constant_mean"
    assert not hasattr(problem, "providers")
    assert not hasattr(problem, "solver")
    assert problem.identity.domain == "field-problem"
    renamed = FieldProblem("diagnostic rename", unknowns=problem.unknowns,
                           equations=problem.equations, gauge=problem.gauge)
    assert problem.identity == renamed.identity


def test_joint_problem_refuses_independent_gauges_and_foreign_boundaries():
    phi1, phi2 = handles()
    with pytest.raises(FieldProblemError) as error:
        joint(gauge=(MeanValueGauge(0), MeanValueGauge(0)))
    assert error.value.report["code"] == "field.gauge.joint_required"
    foreign = Handle("phi1", kind="field", owner=OwnerPath.model("foreign"))
    with pytest.raises(FieldProblemError) as error:
        joint(boundaries=(FieldBoundary(foreign, bcs.BoundaryCondition(
            bcs.AllPhysicalBoundaries(), bcs.Neumann(0))),))
    assert error.value.report["code"] == "field.boundary.foreign_unknown"
    with pytest.raises(FieldProblemError, match="one symbolic equation"):
        FieldProblem("missing", unknowns=(phi1, phi2), equations=(-laplacian(phi1) == 0,))


def test_same_local_unknown_names_keep_distinct_qualified_storage():
    first = Handle("phi", kind="field", owner=OwnerPath.model("first"))
    second = Handle("phi", kind="field", owner=OwnerPath.model("second"))
    layout = Handle("mesh", kind="layout", owner=OwnerPath.case("assembled"))
    storage = FieldStorageBinding((first, second), layout)
    assert storage.to_data()["unknowns"][0] != storage.to_data()["unknowns"][1]
    assert storage.identity != FieldStorageBinding((second, first), layout).identity
    assert not hasattr(storage, "state_block")


def test_combined_constructor_normalizes_the_same_physical_boundary_and_numerics():
    phi = handles()[0]
    rho = Handle("density", kind="state", owner=OwnerPath.model("load"))
    equation = -laplacian(phi) == ValueExpr(rho)
    relation = bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Neumann(0))
    outputs = (FieldOutput("phi", phi),)
    legacy = FieldOperator("poisson", unknown=phi, equation=equation, outputs=outputs,
                           providers=Handle("load", kind="field_operator", owner=OwnerPath.model("load")))
    assert isinstance(legacy, FieldProblem)
    combined_plan = FieldDiscretization(method=CellCenteredSecondOrder(),
        boundaries=(relation,), solver=CartesianCG(), nullspace=ConstantNullspace(), gauge=MeanValueGauge(0))
    separated = FieldProblem("poisson", unknowns=(phi,), equations=(equation,),
        boundaries=(FieldBoundary(phi, relation),), gauge=SharedMeanGauge((phi,)), outputs=outputs)
    numerical_plan = FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(), solver=CartesianCG())
    assert normalize_field_definition(legacy, combined_plan).to_data() == \
        normalize_field_definition(separated, numerical_plan).to_data()


def test_shared_native_nullspace_provider_authenticates_packed_width_and_gauge():
    from pops.fields._joint_nullspace import shared_constant_nullspace
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.time import Program
    program = Program("joint-contract")
    operator = program.matrix_free_operator("joint", domain="vector", range_="vector", ncomp=2)
    program.set_apply(operator, lambda _p, _out, value: value)
    rhs = program.scalar_field("rhs", ncomp=2)
    problem = LinearProblem(operator, rhs, nullspace=shared_constant_nullspace(),
        gauge=SharedMeanGauge(handles()),
        properties=LinearOperatorProperties.symmetric_positive_definite_on_nullspace_complement())
    assert problem.canonical_nullspace_contract()["contract"] == {
        "basis": "shared-constant-vector", "basis_count": 1, "components": 2}
