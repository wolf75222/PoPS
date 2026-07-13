from __future__ import annotations

import importlib.util

import pytest

from pops.fields import (
    DerivedField,
    FieldOperator,
    FieldOutput,
    FieldProviderContribution,
    FieldProviderPack,
    GradientOutput,
)
from pops.ir import ValueExpr
from pops.math import laplacian
from pops.model import Handle, OwnerPath


def _handle(name: str, *, kind: str = "aux", owner: OwnerPath | None = None) -> Handle:
    return Handle(name, kind=kind, owner=owner or OwnerPath.model("electrostatic"))


def _operator(*, rhs: Handle | None = None) -> FieldOperator:
    phi = _handle("phi")
    rho = rhs or _handle("rho")
    equation = -laplacian(ValueExpr(phi)) == ValueExpr(rho)
    return FieldOperator(
        "electrostatic",
        unknown=phi,
        equation=equation,
        providers=_handle("electrostatic_residual", kind="field_operator"),
        outputs=(FieldOutput("phi", phi), GradientOutput("electric_field", phi)),
    )


def test_field_operator_contains_physics_only_and_has_canonical_identity() -> None:
    operator = _operator()

    assert operator.validate()
    assert operator.identity.domain == "field-operator"
    assert operator.inspect()["identity"] == operator.identity.token
    assert {reference.local_id for reference in operator.declaration_references()} == {
        "phi",
        "rho",
        "electrostatic_residual",
    }
    for numerical_name in (
        "method",
        "boundaries",
        "solver",
        "preconditioner",
        "nullspace",
        "gauge",
        "hierarchy_policy",
        "cadence",
    ):
        assert not hasattr(operator, numerical_name)


def test_field_operator_identity_observes_unknown_equation_and_outputs() -> None:
    baseline = _operator()
    baseline_identity = baseline.identity

    changed_unknown = _operator()
    changed_unknown.unknown = _handle("potential")
    assert changed_unknown.identity != baseline_identity

    changed_equation = _operator()
    changed_equation.equation = -laplacian(ValueExpr(changed_equation.unknown)) == ValueExpr(
        _handle("charge")
    )
    assert changed_equation.identity != baseline_identity

    changed_output = _operator()
    changed_output.outputs[1].sign = 1
    assert changed_output.identity != baseline_identity


def test_field_operator_resolves_handles_in_equation_and_outputs() -> None:
    authoring_owner = OwnerPath.fresh(OwnerPath.model("authoring").kind, "authoring")
    phi = _handle("phi", owner=authoring_owner)
    rho = _handle("rho", owner=authoring_owner)
    operator = FieldOperator(
        "field",
        unknown=phi,
        equation=(-laplacian(ValueExpr(phi)) == ValueExpr(rho)),
        providers=_handle("field_residual", kind="field_operator", owner=authoring_owner),
        outputs=(GradientOutput("gradient", phi), DerivedField("rho_copy", ValueExpr(rho))),
    )

    assert operator.validate()
    with pytest.raises(ValueError, match="authoring-owned"):
        _ = operator.identity

    canonical_owner = OwnerPath.model("resolved")
    resolved = operator.resolve_references(lambda handle: handle._resolved(canonical_owner))
    assert resolved.validate()
    assert all(reference.is_resolved for reference in resolved.declaration_references())
    assert resolved.outputs[1].expression.handle.is_resolved
    assert resolved.identity.domain == "field-operator"


def test_field_provider_pack_is_ordered_owner_qualified_and_identity_relevant() -> None:
    phi = _handle("phi")
    rho = _handle("rho")
    first = _handle("ions", kind="field_operator")
    second = _handle("electrons", kind="field_operator")
    operator = FieldOperator(
        "coupled",
        unknown=phi,
        equation=(-laplacian(ValueExpr(phi)) == ValueExpr(rho)),
        providers=FieldProviderPack((
            FieldProviderContribution(first, 1.0),
            FieldProviderContribution(second, -1.0),
        )),
        outputs=(FieldOutput("phi", phi),),
    )

    data = operator.to_data()["providers"]["contributions"]
    assert [row["coefficient"] for row in data] == [1.0, -1.0]
    assert [row["provider"]["local_id"] for row in data] == ["ions", "electrons"]


def test_legacy_field_problem_public_surface_is_removed() -> None:
    import pops.fields as fields

    assert not hasattr(fields, "FieldProblem")
    assert not hasattr(fields, "PoissonProblem")
    assert not hasattr(fields, "HoldPrevious")
    assert importlib.util.find_spec("pops.fields.problem") is None


def test_field_operator_freeze_rejects_mutation() -> None:
    operator = _operator().freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        operator.equation = operator.equation
