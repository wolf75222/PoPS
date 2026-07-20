from __future__ import annotations

import pytest

import pops
from pops.analytic import angle, between, param, radius, sin, where, x
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.model import BindSchema
from pops.model.resolved_bindings import ResolvedBindings
from pops.params import ConstParam, RuntimeParam
from pops.runtime._analytic_expression_lowering import lower_analytic_components
from pops.runtime._initial_source_lowering import native_binary64, validate_initial_source


def _profile():
    frame = Rectangle("domain", (-0.5, -0.5), (0.5, 0.5)).frame(Cartesian2D())
    radial_coordinate = radius(frame)
    angular_coordinate = angle(frame)
    density = where(
        between(radial_coordinate, 0.35, 0.40),
        0.9 + 0.1 * sin(4.0 * angular_coordinate),
        1.0e-4,
    )
    return frame, density.to_data()


def test_diocotron_expression_lowers_to_typed_postfix_program():
    frame, expression = _profile()
    ((opcodes, literals),) = lower_analytic_components(
        [expression], frame_id=frame.canonical_id)

    assert opcodes[-1] == "where"
    assert "between" in opcodes
    assert "hypot" in opcodes
    assert "atan2" in opcodes
    assert len(opcodes) == len(literals)
    assert all(isinstance(value, float) for value in literals)


def test_lowering_accepts_only_canonical_binary64_and_rejects_wrong_frame():
    frame, expression = _profile()
    lower_analytic_components([expression], frame_id=frame.canonical_id)

    noncanonical = _profile()[1]
    noncanonical["root"]["arguments"][2]["value"] = 1.0e-4
    with pytest.raises(TypeError, match="canonical binary64"):
        lower_analytic_components([noncanonical], frame_id=frame.canonical_id)

    with pytest.raises(ValueError, match="another frame"):
        lower_analytic_components([expression], frame_id="pops.other-frame")


@pytest.mark.parametrize("number", (0.0, -0.0, 1.0, -1.25, 5e-324))
def test_native_binary64_accepts_the_exact_hex_spelling_emitted_by_pops(number):
    result = native_binary64({"binary64": number.hex()}, where="test value")

    assert result.hex() == number.hex()


@pytest.mark.parametrize(
    "alias",
    (
        "1.0",
        "0x1p0",
        " 0x1.0000000000000p+0",
        "0X1.0000000000000P+0",
    ),
)
def test_native_binary64_rejects_noncanonical_aliases(alias):
    with pytest.raises(ValueError, match="payload is not canonical"):
        native_binary64({"binary64": alias}, where="test value")


def test_parameter_leaves_lower_from_exact_effective_bindings_to_constants() -> None:
    frame = Rectangle("parameter-domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    case = pops.Case("analytic-lowering-parameters")
    runtime = case.param(RuntimeParam("amplitude", default=1.0))
    offset = case.param(ConstParam("offset", 0.25))
    expression = (param(runtime) * x(frame) + param(offset)).resolve_references(case.resolve)
    schema = BindSchema.from_problem(case)
    compile_values = schema.resolve_compile()

    first = schema.resolve_bind({runtime: 2.5}, compile_values=compile_values)
    second = schema.resolve_bind({runtime: 4.0}, compile_values=compile_values)
    ((first_opcodes, first_literals),) = lower_analytic_components(
        [expression.to_data()], frame_id=frame.canonical_id, bindings=first)
    ((second_opcodes, second_literals),) = lower_analytic_components(
        [expression.to_data()], frame_id=frame.canonical_id, bindings=second)

    assert first_opcodes == second_opcodes == ("constant", "x", "mul", "constant", "add")
    assert first_literals[0] == 2.5
    assert second_literals[0] == 4.0
    assert first_literals[3] == second_literals[3] == 0.25
    assert "parameter" not in first_opcodes

    with pytest.raises(TypeError, match="exact ResolvedBindings"):
        lower_analytic_components(
            [expression.to_data()], frame_id=frame.canonical_id, bindings={runtime: 3.0})
    with pytest.raises(TypeError, match="exact ResolvedBindings"):
        lower_analytic_components([expression.to_data()], frame_id=frame.canonical_id)

    canonical_runtime = schema.slot(runtime).handle
    missing = ResolvedBindings(schema, {}, {})
    with pytest.raises(ValueError, match="no effective value"):
        lower_analytic_components(
            [expression.to_data()], frame_id=frame.canonical_id, bindings=missing)
    for invalid in (float("inf"), float("nan"), True):
        forged = ResolvedBindings(
            schema,
            {**dict(first.values), canonical_runtime: invalid},
            dict(first.sources),
        )
        with pytest.raises((TypeError, ValueError), match="finite real|finite"):
            lower_analytic_components(
                [expression.to_data()], frame_id=frame.canonical_id, bindings=forged)


def test_parameter_lowering_rejects_a_foreign_authenticated_schema() -> None:
    frame = Rectangle("parameter-foreign-domain", (0.0, 0.0), (1.0, 1.0)).frame(
        Cartesian2D())
    owner = pops.Case("analytic-parameter-owner")
    handle = owner.param(RuntimeParam("amplitude", default=1.0))
    expression = param(handle).resolve_references(owner.resolve)

    foreign = pops.Case("analytic-parameter-foreign")
    foreign_handle = foreign.param(RuntimeParam("amplitude", default=1.0))
    schema = BindSchema.from_problem(foreign)
    bindings = schema.resolve_bind(
        {foreign_handle: 2.0}, compile_values=schema.resolve_compile())
    with pytest.raises(ValueError, match="not authenticated"):
        lower_analytic_components(
            [expression.to_data()], frame_id=frame.canonical_id, bindings=bindings)


@pytest.mark.parametrize(
    "source",
    (
        {
            "native_route": "constant_field",
            "components": [{"binary64": (0.25).hex()}],
        },
        {
            "native_route": "gaussian_field",
            "frame_id": "pops.frame.test",
            "center": {
                "x": {"binary64": (0.25).hex()},
                "y": {"binary64": (0.75).hex()},
            },
            "background": {"binary64": (0.1).hex()},
            "amplitude": {"binary64": (0.9).hex()},
            "inverse_width": {"binary64": (80.0).hex()},
        },
        {
            "native_route": "analytic_expression",
            "frame_id": "pops.frame.test",
            "components": [{"expression_type": "scalar"}],
        },
    ),
    ids=("constant", "gaussian", "analytic"),
)
def test_native_initial_routes_uniformly_reject_additional_schema_keys(source):
    canonical = {
        **source,
        "projection": {
            "schema_version": 1,
            "projection": "conservative_cell_average",
            "formal_order": 2,
            "ghost_depth": [1],
        },
    }
    validate_initial_source(canonical, where="test initial source")

    forged = {**canonical, "unexpected": "must-not-be-ignored"}
    with pytest.raises(TypeError, match="requires exactly keys"):
        validate_initial_source(forged, where="test initial source")
