#!/usr/bin/env python3
"""Operator-first BDF macros.

``pops.lib.time.bdf`` is a ready-made local-linear time macro. It composes typed
operator handles and produces canonical Program IR. It is not the old implicit
flux/Newton builder and does not accept source/flux/string selectors.
"""
import sys

import pytest

adctime = pytest.importorskip("pops.time")
libtime = pytest.importorskip("pops.lib.time")
from pops import model
from pops.ir.expr import Const


def _module(name, ncomp=2):
    mod = model.Module(name)
    components = tuple("q%d" % i for i in range(ncomp))
    U = mod.state_space("U", components)
    fields = mod.field_space("fields", ("phi",))
    rhs = mod.operator(
        name="rhs",
        signature=(U,) >> model.Rate(U),
        kind="local_source",
        expr=[Const(0.0)] * ncomp,
    )
    rhs_with_fields = mod.operator(
        name="rhs_with_fields",
        signature=(U, fields) >> model.Rate(U),
        kind="local_source",
        expr=[Const(0.0)] * ncomp,
    )
    linear = mod.operator(
        name="linear",
        signature=() >> model.LocalLinearOperator(U, U),
        kind="local_linear_operator",
        expr=[[0.0 for _ in components] for _ in components],
    )
    linear_with_fields = mod.operator(
        name="linear_with_fields",
        signature=(fields,) >> model.LocalLinearOperator(U, U),
        kind="local_linear_operator",
        expr=[[0.0 for _ in components] for _ in components],
    )
    fields_from_state = mod.operator(
        name="fields_from_state",
        signature=(U,) >> fields,
        kind="field_operator",
        capabilities={"default": True},
        expr=Const(0.0),
    )
    return mod, {
        "rhs": rhs,
        "rhs_with_fields": rhs_with_fields,
        "linear": linear,
        "linear_with_fields": linear_with_fields,
        "fields": fields_from_state,
    }


def _emit(program, module):
    program.validate()
    return program.emit_cpp_program(model=module)


def _ops(program):
    return [v.op for v in program._values]


def test_bdf1_local_linear_operator_first():
    mod, h = _module("bdf1")
    P = adctime.Program("bdf1").bind_operators(mod)

    out = libtime.bdf(P, "blk", 1, implicit_operator=h["linear"], rhs_operator=h["rhs"])

    assert out.vtype == "state"
    ops = _ops(P)
    assert "call" in ops
    assert "solve_local_linear" in ops
    assert "rhs" not in ops
    assert "solve_fields" not in ops
    assert "linear_source" not in ops
    src = _emit(P, mod)
    assert "GeneratedModule::Operators::linear" in src
    assert "GeneratedModule::Operators::rhs" in src


def test_bdf2_history_and_fields_operator_first():
    mod, h = _module("bdf2")
    P = adctime.Program("bdf2").bind_operators(mod)

    out = libtime.bdf(
        P,
        "blk",
        2,
        implicit_operator=h["linear_with_fields"],
        rhs_operator=h["rhs_with_fields"],
        fields_operator=h["fields"],
    )

    assert out.vtype == "state"
    ops = _ops(P)
    assert "call" in ops
    assert "store_history" in ops
    assert "history" in ops
    assert "solve_local_linear" in ops
    assert "rhs" not in ops
    assert "solve_fields" not in ops
    assert "linear_source" not in ops
    src = _emit(P, mod)
    assert 'ctx.store_history("blk.U"' in src
    assert 'ctx.history("blk.U", 1)' in src
    assert "GeneratedModule::Operators::fields_from_state" in src
    assert "GeneratedModule::Operators::linear_with_fields" in src
    assert "GeneratedModule::Operators::rhs_with_fields" in src


def test_bdf_rejects_bad_order():
    mod, h = _module("bdf_bad_order")
    for bad in (0, 3, True, 1.5, "1"):
        P = adctime.Program("bad").bind_operators(mod)
        with pytest.raises(ValueError, match="order"):
            libtime.bdf(P, "blk", bad, implicit_operator=h["linear"])


def test_bdf_legacy_selector_kwargs_rejected():
    mod, h = _module("bdf_legacy_kwargs")
    cases = [
        {"linear_source": "lorentz"},
        {"sources": ["default"]},
        {"flux": True},
    ]
    for kwargs in cases:
        P = adctime.Program("legacy").bind_operators(mod)
        with pytest.raises(TypeError):
            libtime.bdf(P, "blk", 1, implicit_operator=h["linear"], **kwargs)


def test_bdf_rejects_string_operator_selectors():
    mod, _ = _module("bdf_string_selectors")
    P = adctime.Program("string").bind_operators(mod)
    with pytest.raises(TypeError, match="typed operator handles"):
        libtime.bdf(P, "blk", 1, implicit_operator="linear")


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("PASS test_time_bdf")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
