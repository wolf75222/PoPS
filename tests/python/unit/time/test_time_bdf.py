"""Final BDF factories: typed local implicit operator, exact history and coefficients."""
from __future__ import annotations

from fractions import Fraction
import inspect

import pytest

import pops.lib.time as libtime
from pops.physics._facade import Model
from pops.time import Program
from typed_program_support import state_refs


def _authoring(name="bdf"):
    model = Model(name + "_model")
    model.conservative_vars("u", "v")
    explicit = model.rate("explicit", flux=False, sources=())
    implicit = model.local_linear_map("implicit", [[-1, 0], [0, -1]])
    block, state = state_refs(Program("refs"), "fluid", model=model)
    return block[state], explicit, implicit


def _node(program, op):
    return next(node for node in program._serialize()["nodes"] if node["op"] == op)


@pytest.mark.parametrize("order", (1, 2))
def test_local_bdf_is_an_ordinary_valid_program(order):
    state, _, implicit = _authoring("local%d" % order)
    program = libtime.BDF(state, implicit=implicit, order=order)
    assert type(program) is Program
    assert program.validate() is True
    operations = [value.op for value in program._values]
    assert operations.count("solve_local_linear") == 1
    assert "matrix_free_operator" not in operations
    assert "rhs_jacvec" not in operations
    assert "solve_linear" not in operations


def test_bdf2_declares_and_reads_one_exact_state_history():
    state, _, implicit = _authoring("history")
    program = libtime.BDF(state, implicit=implicit, order=2)
    serialized = program._serialize()
    assert serialized["histories"] == [{
        "name": "fluid.state",
        "lag": 1,
        "ncomp": None,
        "state": serialized["nodes"][0]["state"],
    }]
    assert any(node["op"] == "history" for node in serialized["nodes"])


def test_bdf_coefficients_remain_exact_rationals():
    state, _, implicit = _authoring("exact")
    bdf1 = libtime.BDF(state, implicit=implicit, order=1)
    bdf2 = libtime.BDF(state, implicit=implicit, order=2)
    assert _node(bdf1, "solve_local_linear")["attrs"]["a_coeff"] == [[
        1, {"kind": "integer", "value": "1"}
    ]]
    assert _node(bdf2, "solve_local_linear")["attrs"]["a_coeff"] == [[
        1, {"kind": "rational", "numerator": "2", "denominator": "3"}
    ]]
    combine = _node(bdf2, "linear_combine")["attrs"]["coeffs"]
    assert any(item == [[0, {
        "kind": "rational", "numerator": "-1", "denominator": "3"
    }]] for item in combine)


def test_bdf_can_add_one_exact_typed_explicit_rate():
    state, explicit, implicit = _authoring("imex")
    program = libtime.BDF(
        state, implicit=implicit, explicit=explicit, order=2)
    handles = {
        value.attrs["operator_handle"]
        for value in program._values
        if "operator_handle" in value.attrs
    }
    assert explicit in handles and implicit in handles
    assert program.validate() is True


def test_bdf_rejects_legacy_global_newton_and_shape_knobs():
    state, _, implicit = _authoring("guards")
    parameters = tuple(inspect.signature(libtime.BDF).parameters)
    assert parameters == (
        "state", "implicit", "order", "explicit", "fields", "solve_action")
    for kwargs in (
        {"newton_max": 3},
        {"krylov_max": 40},
        {"ncomp": 2},
        {"flux": True},
        {"sources": ()},
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            libtime.BDF(state, implicit=implicit, order=1, **kwargs)


@pytest.mark.parametrize("order", (0, 3, True, 1.5, "1"))
def test_bdf_rejects_unsupported_orders(order):
    state, _, implicit = _authoring("bad-order")
    with pytest.raises((TypeError, ValueError), match="order"):
        libtime.BDF(state, implicit=implicit, order=order)


def test_bdf_rejects_unqualified_state_and_free_operator_name():
    state, _, implicit = _authoring("typed")
    with pytest.raises(TypeError, match="OperatorHandle"):
        libtime.BDF(state, implicit="implicit", order=1)
    with pytest.raises(TypeError, match=r"block\[state\]|Handle"):
        libtime.BDF(object(), implicit=implicit, order=1)
