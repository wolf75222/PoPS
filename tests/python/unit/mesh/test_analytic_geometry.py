from __future__ import annotations

import copy
import json

import pytest

import pops
from pops.analytic import PredicateExpr, constant, coordinates, param
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.geometry import (
    Disc,
    HalfPlane,
    LevelSet,
    NoWall,
    complement,
    difference,
    intersection,
    union,
)
from pops.params import RuntimeParam


def _frame(name: str = "box"):
    return Rectangle(name, (-2.0, -1.0), (2.0, 3.0)).frame(Cartesian2D())


def test_level_set_accepts_only_canonical_scalar_expressions() -> None:
    x_value, _ = coordinates(_frame())
    geometry = LevelSet(x_value - 0.5)

    assert geometry.frame_id == x_value.frame_id
    assert geometry.options() == {
        "active_when": "phi<0",
        "expression": geometry.expression.to_data(),
    }
    assert json.loads(json.dumps(geometry.to_data())) == geometry.to_data()
    assert LevelSet.from_data(geometry.to_data()).expression.same_as(geometry.expression)

    for value in (0.0, "x - 0.5", lambda x: x, x_value < 0):
        with pytest.raises(TypeError, match="ScalarExpr"):
            LevelSet(value)


def test_parameterized_level_set_refuses_the_prebind_geometry_ambiguity() -> None:
    case = pops.Case("parameterized-level-set")
    radius = case.param(RuntimeParam("radius", default=0.5))
    x_value, _ = coordinates(_frame())

    with pytest.raises(NotImplementedError, match="signed during resolve"):
        LevelSet(x_value - param(radius))


def test_disc_level_set_uses_explicit_or_frame_center_without_callback() -> None:
    frame = _frame()
    centered = Disc(radius=0.5).level_set(frame)
    explicit = Disc(center=(0.25, -0.5), radius=0.75).level_set(frame)

    centered_root = centered.expression.to_data()["root"]
    explicit_root = explicit.expression.to_data()["root"]
    assert centered_root["op"] == "sub"
    assert centered_root["arguments"][0]["op"] == "hypot"
    assert explicit_root["arguments"][0]["op"] == "hypot"
    assert centered.frame_id == explicit.frame_id == frame.canonical_id

    with pytest.raises(ValueError, match="bounded frame"):
        Disc().level_set(Cartesian2D())


def test_half_plane_level_set_has_negative_active_side() -> None:
    frame = _frame()
    geometry = HalfPlane(point=(0.25, -0.5), normal=(2.0, -1.0)).level_set(frame)
    root = geometry.expression.to_data()["root"]

    assert root["op"] == "add"
    assert [argument["op"] for argument in root["arguments"]] == ["mul", "mul"]
    assert geometry.to_data()["active_when"] == "phi<0"

    with pytest.raises(ValueError, match="exactly two"):
        HalfPlane(point=(0.0,), normal=(1.0, 0.0))
    with pytest.raises(ValueError, match="non-zero"):
        HalfPlane(normal=(0.0, 0.0))


def test_no_wall_is_the_generic_all_active_geometry() -> None:
    geometry = NoWall().level_set(_frame())

    assert geometry.expression.same_as(constant(-1.0))


def test_boolean_composition_is_generic_immutable_and_canonical() -> None:
    frame = _frame()
    disc_geometry = Disc(center=(0.0, 0.0), radius=1.0)
    right_geometry = HalfPlane(point=(0.0, 0.0), normal=(-1.0, 0.0))
    disc = disc_geometry.level_set(frame)
    right = right_geometry.level_set(frame)
    disc_before = disc.to_data()
    right_before = right.to_data()

    joined = union(disc_geometry, right_geometry).level_set(frame)
    common = (disc_geometry & right_geometry).level_set(frame)
    cut = (disc_geometry - right_geometry).level_set(frame)
    outside = (~disc_geometry).level_set(frame)

    assert joined.expression.op == "minimum"
    assert common.expression.op == "maximum"
    assert cut.expression.op == "maximum"
    assert cut.expression.to_data()["root"]["arguments"][1]["op"] == "neg"
    assert outside.expression.op == "neg"
    assert (disc_geometry | right_geometry).level_set(frame).expression.same_as(joined.expression)
    assert intersection(disc, right).level_set(frame).expression.same_as(common.expression)
    assert difference(disc, right).level_set(frame).expression.same_as(cut.expression)
    assert complement(disc).level_set(frame).expression.same_as(outside.expression)
    assert disc.to_data() == disc_before
    assert right.to_data() == right_before
    with pytest.raises(TypeError, match="no Python truth value"):
        bool(disc_geometry)


def test_nary_composition_and_frame_mismatch_fail_closed() -> None:
    frame = _frame()
    first = Disc(center=(-0.5, 0.0), radius=0.75).level_set(frame)
    second = Disc(center=(0.5, 0.0), radius=0.75).level_set(frame)
    third = HalfPlane().level_set(frame)

    assert union(first, second, third).level_set(frame).expression.measure().node_count > 3
    assert intersection(first, second, third).level_set(frame).expression.measure().node_count > 3
    with pytest.raises(ValueError, match="different frames"):
        union(first, Disc(radius=0.5).level_set(_frame("other"))).level_set(frame)
    with pytest.raises(TypeError, match="Geometry"):
        difference(first, constant(0.0))


def test_nary_composition_balances_hundreds_of_level_sets() -> None:
    frame = _frame()
    geometries = tuple(
        Disc(center=(-1.0 + index / 128.0, 0.0), radius=0.125).level_set(frame)
        for index in range(256)
    )

    joined = union(*geometries).level_set(frame)
    common = intersection(*geometries).level_set(frame)

    for geometry, operation in ((joined, "minimum"), (common, "maximum")):
        statistics = geometry.expression.measure()
        assert geometry.expression.op == operation
        assert statistics.depth <= 16
        assert LevelSet.from_data(geometry.to_data()).expression.same_as(
            geometry.expression)


def test_level_set_decoder_rejects_forged_sign_or_predicate_data() -> None:
    geometry = Disc(radius=0.5).level_set(_frame())
    forged = copy.deepcopy(geometry.to_data())
    forged["active_when"] = "phi>0"
    with pytest.raises(ValueError, match="sign convention"):
        LevelSet.from_data(forged)

    predicate = PredicateExpr.from_data((geometry.expression < 0).to_data())
    forged = copy.deepcopy(geometry.to_data())
    forged["expression"] = predicate.to_data()
    with pytest.raises(ValueError, match="expression_type"):
        LevelSet.from_data(forged)
