from __future__ import annotations

import pytest

from pops.descriptors import Descriptor
from pops.fields import (
    CompositeHierarchySolve,
    ConstantNullspace,
    FieldDiscretization,
    InferHierarchyFromLayout,
    LevelByLevelSolve,
    MeanValueGauge,
)
from pops.fields.bcs import BoundaryCondition, Dirichlet, XMax, XMin
from pops.ir import ValueExpr
from pops.model import Handle, OwnerPath


class Method(Descriptor):
    category = "field_method"

    def __init__(self, stencil: str) -> None:
        self.stencil = stencil

    def options(self) -> dict:
        return {"stencil": self.stencil}

    def to_data(self) -> dict:
        return {"type": "Method", "stencil": self.stencil}


class Solver(Descriptor):
    category = "elliptic_solver"

    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    def options(self) -> dict:
        return {"algorithm": self.algorithm}

    def to_data(self) -> dict:
        return {"type": "Solver", "algorithm": self.algorithm}


class Preconditioner(Descriptor):
    category = "preconditioner"

    def __init__(self, sweeps: int) -> None:
        self.sweeps = sweeps

    def options(self) -> dict:
        return {"sweeps": self.sweeps}

    def to_data(self) -> dict:
        return {"type": "Preconditioner", "sweeps": self.sweeps}


def _plan() -> FieldDiscretization:
    return FieldDiscretization(
        method=Method("cell_centered"),
        boundaries=(BoundaryCondition(XMin(), Dirichlet(0)),),
        solver=Solver("multigrid"),
        preconditioner=Preconditioner(2),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0),
        hierarchy_policy=CompositeHierarchySolve(),
    )


def test_field_discretization_owns_all_and_only_numerical_choices() -> None:
    plan = _plan()

    assert plan.validate()
    assert plan.identity.domain == "field-discretization"
    assert plan.capabilities().supports("derives_order")
    assert plan.capabilities().supports("derives_ghost_depth")
    assert not hasattr(plan, "order")
    assert not hasattr(plan, "ghost_depth")
    assert not hasattr(plan, "unknown")
    assert not hasattr(plan, "equation")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("method", Method("spectral")),
        ("boundaries", (BoundaryCondition(XMax(), Dirichlet(1)),)),
        ("solver", Solver("fft")),
        ("preconditioner", Preconditioner(3)),
        ("nullspace", None),
        ("gauge", MeanValueGauge(1)),
        ("hierarchy_policy", LevelByLevelSolve()),
    ],
)
def test_each_discretization_field_changes_identity(field: str, replacement: object) -> None:
    plan = _plan()
    baseline = plan.identity

    setattr(plan, field, replacement)

    assert plan.identity != baseline


def test_nested_method_solver_boundary_nullspace_and_gauge_mutation_is_observed() -> None:
    mutations = (
        lambda plan: setattr(plan.method, "stencil", "compact"),
        lambda plan: setattr(plan.solver, "algorithm", "krylov"),
        lambda plan: setattr(plan.boundaries[0].condition, "value", 2),
        lambda plan: setattr(plan.gauge, "value", 3),
    )
    for mutate in mutations:
        plan = _plan()
        before = plan.identity
        mutate(plan)
        assert plan.identity != before

    with pytest.raises(AttributeError, match="no configurable fields"):
        _plan().nullspace.marker = "changed"


def test_nullspace_and_gauge_are_separate_and_compatibility_is_validated() -> None:
    missing_gauge = FieldDiscretization(
        method=Method("cell_centered"),
        boundaries=(),
        solver=Solver("multigrid"),
        nullspace=ConstantNullspace(),
    )
    with pytest.raises(ValueError, match="explicit gauge"):
        missing_gauge.validate()

    independent_gauge = FieldDiscretization(
        method=Method("cell_centered"),
        boundaries=(),
        solver=Solver("multigrid"),
        gauge=MeanValueGauge(0),
        hierarchy_policy=InferHierarchyFromLayout(),
    )
    assert independent_gauge.validate()


def test_duplicate_boundary_authority_and_string_selectors_fail() -> None:
    duplicate = FieldDiscretization(
        method=Method("cell_centered"),
        boundaries=(
            BoundaryCondition(XMin(), Dirichlet(0)),
            BoundaryCondition(XMin(), Dirichlet(1)),
        ),
        solver=Solver("multigrid"),
    )
    with pytest.raises(ValueError, match="more than one condition"):
        duplicate.validate()

    with pytest.raises(TypeError, match="String algorithm selector rejected"):
        FieldDiscretization(method="second_order", boundaries=(), solver=Solver("mg"))


def test_unregistered_descriptor_projection_fails_loudly() -> None:
    class OpaqueMethod(Descriptor):
        category = "field_method"

    plan = FieldDiscretization(method=OpaqueMethod(), boundaries=(), solver=Solver("multigrid"))
    with pytest.raises(TypeError, match="small exact to_data"):
        _ = plan.identity


def test_third_party_descriptor_uses_local_protocol_without_registration() -> None:
    class ThirdPartyMethod(Descriptor):
        category = "field_method"

        def to_data(self) -> dict:
            return {"type": "ThirdPartyMethod", "stencil": "compact"}

    plan = FieldDiscretization(method=ThirdPartyMethod(), boundaries=(), solver=Solver("multigrid"))

    assert plan.identity.domain == "field-discretization"


def test_boundary_expression_references_resolve_with_the_plan() -> None:
    owner = OwnerPath.fresh(OwnerPath.model("authoring").kind, "authoring")
    boundary_value = Handle("wall_value", kind="parameter", owner=owner)
    plan = FieldDiscretization(
        method=Method("cell_centered"),
        boundaries=(BoundaryCondition(XMin(), Dirichlet(ValueExpr(boundary_value))),),
        solver=Solver("multigrid"),
    )

    assert plan.validate()
    assert plan.inspect()["identity"] is None
    with pytest.raises(ValueError, match="authoring-owned"):
        _ = plan.identity

    canonical_owner = OwnerPath.model("resolved")
    resolved = plan.resolve_references(lambda handle: handle._resolved(canonical_owner))
    assert resolved.validate()
    assert resolved.declaration_references()[0].is_resolved
    assert resolved.identity.domain == "field-discretization"
