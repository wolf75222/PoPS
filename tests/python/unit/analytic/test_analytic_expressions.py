from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pops.analytic import (
    AnalyticTruthValueError,
    PredicateExpr,
    ScalarExpr,
    abs as analytic_abs,
    angle,
    atan2,
    between,
    clamp,
    constant,
    coordinate,
    coordinates,
    cos,
    exp,
    hypot,
    log,
    maximum,
    minimum,
    norm,
    param,
    radius,
    sin,
    sqrt,
    where,
    x,
    y,
)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.model import Handle
from pops.params import ConstParam, RuntimeParam


def _frame(name: str = "box"):
    return Rectangle(name, (-1.0, -1.0), (1.0, 1.0)).frame(Cartesian2D())


def test_coordinates_are_typed_and_bound_to_one_frame() -> None:
    frame = _frame()
    x_value, y_value = coordinates(frame)

    assert x_value.same_as(x(frame))
    assert y_value.same_as(y(frame))
    assert x_value.same_as(coordinate(frame, frame.x))
    assert x_value.frame_id == frame.canonical_id
    assert x_value.to_data()["root"] == {
        "kind": "scalar",
        "op": "coordinate",
        "frame_id": frame.canonical_id,
        "axis": frame.x.to_dict(),
    }

    with pytest.raises(TypeError, match="CartesianAxis"):
        coordinate(frame, "x")
    with pytest.raises(ValueError, match="different frames"):
        x_value + x(_frame("other"))


def test_scalar_math_builds_a_data_only_canonical_tree() -> None:
    frame = _frame()
    x_value, y_value = coordinates(frame)
    radius = hypot(x_value, y_value)
    angle = atan2(y_value, x_value)
    field = (
        analytic_abs(radius - 0.4)
        + sin(4 * angle)
        + cos(angle)
        + exp(-radius)
        + log(radius + 1)
        + minimum(radius, 1)
        + maximum(radius, 0)
    ) / 7

    assert isinstance(field, ScalarExpr)
    assert field.frame_id == frame.canonical_id
    assert field.validate()
    assert field.measure().node_count > 20
    assert json.loads(json.dumps(field.to_data())) == field.to_data()
    assert ScalarExpr.from_data(field.to_data()).same_as(field)

    reflected = 2 + x_value * 3 - 4 / (y_value + 5)
    assert reflected.frame_id == frame.canonical_id
    assert ScalarExpr.from_data(reflected.to_data()).same_as(reflected)


def test_polar_helpers_and_clamp_remain_generic_expression_composition() -> None:
    frame = _frame()
    radial = radius(frame, center={frame.x: 0.25, frame.y: -0.5})
    angular = angle(frame, center=(0.25, -0.5))
    bounded = clamp(sin(4.0 * angular) + radial, -1.0, 1.0)

    assert radial.op == "hypot"
    assert angular.op == "atan2"
    assert bounded.op == "minimum"
    assert bounded.frame_id == frame.canonical_id
    with pytest.raises(ValueError, match="exactly two"):
        radius(frame, center=(0.0,))
    with pytest.raises(ValueError, match="every frame axis"):
        angle(frame, center={frame.x: 0.0})


def test_norm_is_variadic_balanced_and_uses_the_native_hypot_primitive() -> None:
    x_value, y_value = coordinates(_frame())
    vector_norm = norm(x_value, y_value, 2.0)
    sequence_norm = norm((x_value, y_value, 2.0))

    assert vector_norm.same_as(sequence_norm)
    assert vector_norm.op == "hypot"
    assert vector_norm.frame_id == x_value.frame_id
    assert norm(x_value).op == "abs"
    with pytest.raises(ValueError, match="at least one"):
        norm()


def test_predicates_are_explicit_and_where_is_generic() -> None:
    frame = _frame()
    x_value, y_value = coordinates(frame)
    radius = sqrt(x_value * x_value + y_value * y_value)
    annulus = between(radius, 0.35, 0.40)
    upper_right = (x_value >= 0) & (y_value >= 0)
    selected = where(annulus & ~upper_right, 0.9 + 0.1 * sin(4 * atan2(y_value, x_value)), 1e-4)

    assert isinstance(annulus, PredicateExpr)
    assert isinstance(selected, ScalarExpr)
    assert selected.frame_id == frame.canonical_id
    assert PredicateExpr.from_data(annulus.to_data()).same_as(annulus)
    assert ScalarExpr.from_data(selected.to_data()).same_as(selected)

    all_comparisons = (
        x_value == 0,
        x_value != 0,
        x_value < 0,
        x_value <= 0,
        x_value > 0,
        x_value >= 0,
    )
    assert tuple(item.op for item in all_comparisons) == ("eq", "ne", "lt", "le", "gt", "ge")
    assert ((x_value < 0) | (x_value > 0)).op == "or"


def test_expressions_are_immutable_non_hashable_and_never_truthy() -> None:
    x_value = x(_frame())
    predicate = x_value > 0

    with pytest.raises(FrozenInstanceError):
        x_value._op = "constant"  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(x_value)
    with pytest.raises(TypeError, match="unhashable"):
        hash(predicate)
    with pytest.raises(AnalyticTruthValueError, match="no Python truth value"):
        bool(x_value)
    with pytest.raises(AnalyticTruthValueError, match="no Python truth value"):
        bool(predicate)
    with pytest.raises(AnalyticTruthValueError, match="no Python truth value"):
        if x_value < 0 < x_value:
            pass
    with pytest.raises(TypeError, match="use same_as"):
        _ = predicate == predicate


def test_parameter_reads_keep_handles_separate_and_resolve_through_the_case() -> None:
    import pops

    case = pops.Case("analytic-parameter-case")
    runtime = case.param(RuntimeParam("amplitude", default=1.0))
    constant_value = case.param(ConstParam("offset", 0.25))
    expression = param(runtime) * 2.0 + param(constant_value)

    assert hash(runtime) == hash(runtime)
    assert runtime == runtime
    assert expression.has_parameters is True
    authoring = expression.to_data()
    assert authoring["root"]["arguments"][0]["arguments"][0]["op"] == "parameter"
    assert (
        authoring["root"]["arguments"][0]["arguments"][0]["reference"]
        ["ownership_phase"]
        == "authoring"
    )
    with pytest.raises(TypeError, match="unsupported key set"):
        ScalarExpr.from_data(authoring)

    resolved = expression.resolve_references(case.resolve)
    rebuilt = ScalarExpr.from_data(resolved.to_data())
    assert rebuilt.same_as(resolved)
    references = (
        resolved.to_data()["root"]["arguments"][0]["arguments"][0]["reference"],
        resolved.to_data()["root"]["arguments"][1]["reference"],
    )
    assert {row["param_kind"] for row in references} == {"runtime", "const"}

    for invalid in (
        RuntimeParam("raw_declaration"),
        "amplitude",
        True,
        Handle("state", kind="state", owner=case.owner_path),
    ):
        with pytest.raises(TypeError, match="exact ParamHandle"):
            param(invalid)

    foreign = pops.Case("foreign-analytic-parameter-case")
    foreign_value = foreign.param(RuntimeParam("amplitude", default=2.0))
    with pytest.raises((KeyError, ValueError)):
        param(foreign_value).resolve_references(case.resolve)


def test_literals_and_operator_kinds_fail_closed() -> None:
    x_value = x(_frame())

    for value in (True, False, lambda: 1.0, object(), 1 + 2j):
        with pytest.raises(TypeError, match="finite real scalar"):
            constant(value)
    for value in (math.inf, -math.inf, math.nan, 10**10000):
        with pytest.raises(ValueError, match="finite"):
            constant(value)
    with pytest.raises(TypeError, match="finite scalar literal"):
        _ = x_value**x_value
    with pytest.raises(TypeError, match="PredicateExpr operands"):
        _ = (x_value > 0) & True
    with pytest.raises(TypeError, match="PredicateExpr"):
        where(True, x_value, 0)
    with pytest.raises(ValueError, match="lower bound"):
        between(x_value, 2, 1)
    with pytest.raises(TypeError, match="canonical"):
        ScalarExpr()
    with pytest.raises(TypeError, match="canonical"):
        PredicateExpr()


def test_serialization_rejects_unknown_forged_or_noncanonical_data() -> None:
    expression = where(between(x(_frame()), -0.5, 0.5), 1.0, 0.0)
    payload = expression.to_data()

    extra = copy.deepcopy(payload)
    extra["legacy"] = True
    with pytest.raises(TypeError, match="unsupported shape"):
        ScalarExpr.from_data(extra)

    unknown = copy.deepcopy(payload)
    unknown["root"]["op"] = "python_callback"
    with pytest.raises(ValueError, match="unsupported analytic scalar operation"):
        ScalarExpr.from_data(unknown)

    wrong_root = copy.deepcopy(payload)
    wrong_root["expression_type"] = "predicate"
    with pytest.raises(ValueError, match="expression_type"):
        ScalarExpr.from_data(wrong_root)

    noncanonical_number = constant(1).to_data()
    noncanonical_number["root"]["value"] = 1
    with pytest.raises(TypeError, match="constant data"):
        ScalarExpr.from_data(noncanonical_number)

    nonfinite = constant(1).to_data()
    nonfinite["root"]["value"] = {"binary64": float("nan").hex()}
    with pytest.raises(ValueError, match="finite"):
        ScalarExpr.from_data(nonfinite)


def test_validation_has_explicit_depth_and_node_budgets() -> None:
    expression = x(_frame())
    for _ in range(8):
        expression = sin(expression)

    assert expression.measure(max_depth=9, max_nodes=9).depth == 9
    with pytest.raises(ValueError, match="max_depth=8"):
        expression.validate(max_depth=8)
    with pytest.raises(ValueError, match="max_nodes=8"):
        expression.validate(max_nodes=8)
    with pytest.raises(TypeError, match="positive integer"):
        expression.validate(max_depth=True)
    with pytest.raises(ValueError, match=">= 1"):
        expression.validate(max_nodes=0)

    payload = expression.to_data()
    with pytest.raises(ValueError, match="max_depth=8"):
        ScalarExpr.from_data(payload, max_depth=8)
    with pytest.raises(ValueError, match="max_nodes=8"):
        ScalarExpr.from_data(payload, max_nodes=8)


def test_validation_matches_the_native_postfix_stack_budget() -> None:
    frame = _frame()
    condition = x(frame) > 0.0
    expression = constant(0.0)
    for _ in range(31):
        expression = where(condition, 1.0, expression)

    statistics = expression.measure()
    assert statistics.depth < 64
    assert statistics.node_count < 4096
    assert statistics.required_stack == 63

    too_wide = where(condition, 1.0, expression)
    with pytest.raises(ValueError, match="max_stack=64"):
        too_wide.validate()
    with pytest.raises(ValueError, match="max_depth must be <= 64"):
        expression.validate(max_depth=65)
    with pytest.raises(ValueError, match="max_nodes must be <= 4096"):
        expression.validate(max_nodes=4097)
    with pytest.raises(ValueError, match="max_stack must be <= 64"):
        expression.validate(max_stack=65)


def test_analytic_package_imports_without_the_native_extension() -> None:
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "python")
    program = """
import json
import sys
from pops.analytic import coordinates, sqrt, between, where
from pops.domain import Rectangle
from pops.frames import Cartesian2D
frame = Rectangle('box', (-1, -1), (1, 1)).frame(Cartesian2D())
x, y = coordinates(frame)
profile = where(between(sqrt(x*x + y*y), 0.35, 0.40), 1.0, 1e-4)
assert 'pops._pops' not in sys.modules
print(json.dumps(profile.to_data(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=root, env=env,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["expression_type"] == "scalar"
