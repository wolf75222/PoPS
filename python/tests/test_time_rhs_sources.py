#!/usr/bin/env python3
"""Operator-first source/rate selection in Programs.

The final API does not let a Program choose physics with selector lists or
boolean transport switches. The Module declares typed operators and the Program
calls their handles explicitly:

    flux            : U -> Rate(U)
    source_default  : U -> Rate(U)
    decay           : U -> Rate(U)

This test checks that the Program IR records those choices as call nodes and that
codegen routes through ``GeneratedModule::Operators`` instead of Program-side RHS
selectors.
"""

from pops.codegen.program_emit_module_ops import operator_function_name
from pops.ir.expr import Const
from pops import model
from pops import time as t
import pops.lib.time as libtime


_C = 0.7
_D = 0.5


def _source_module(name="rhs_sources"):
    m = model.Module(name + "_module")
    U = m.state_space("U", ("rho",))
    flux = m.operator(
        "flux",
        signature=(U,) >> model.Rate(U),
        kind="grid_operator",
        capabilities={"produces_rate": True, "supports_device": True, "default": True},
        expr={"x": [Const(0.0)], "y": [Const(0.0)]},
    )
    default = m.operator(
        "source_default",
        signature=(U,) >> model.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True, "default": True},
        expr=[Const(_C)],
    )
    decay = m.operator(
        "decay",
        signature=(U,) >> model.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(_D)],
    )
    return m, flux, default, decay


def _program(name, module):
    return t.Program(name).bind_operators(module)


def _state(P, block="plasma"):
    return P.state("U", block=block).n


def _call_fn(module, op):
    reg = module.operator_registry()
    return "GeneratedModule::Operators::%s" % operator_function_name(reg.id_of(op.name), op.name)


def _one_step(module, op, name="step"):
    P = _program(name, module)
    U = _state(P)
    R = P.call(op, U, name="R")
    P.commit("plasma", P.linear_combine("%s_out" % name, U + P.dt * R))
    return P


def test_flux_only_is_an_explicit_operator_call_not_a_source_selector():
    m, flux, default, _ = _source_module("flux_only")
    P = _one_step(m, flux, "flux_only")
    assert P.validate() is True
    assert all(v.op != "rhs" for v in P._values), "Program must not build legacy rhs nodes"
    assert [v.attrs["operator"] for v in P._values if v.op == "call"] == ["flux"]

    src = P.emit_cpp_program(model=m)
    assert _call_fn(m, flux) + "(ctx, 0," in src
    assert _call_fn(m, default) + "(ctx, 0," not in src
    assert "ctx.neg_div_flux_default_into" in src
    assert "ctx.rhs_into(" not in src


def test_default_and_named_sources_are_distinct_handles():
    m, _, default, decay = _source_module("source_handles")
    P_default = _one_step(m, default, "default_source")
    P_decay = _one_step(m, decay, "decay_source")

    assert P_default._ir_hash() != P_decay._ir_hash()
    assert [v.attrs["operator"] for v in P_default._values if v.op == "call"] == ["source_default"]
    assert [v.attrs["operator"] for v in P_decay._values if v.op == "call"] == ["decay"]

    src_default = P_default.emit_cpp_program(model=m)
    src_decay = P_decay.emit_cpp_program(model=m)
    assert _call_fn(m, default) + "(ctx, 0," in src_default
    assert _call_fn(m, decay) + "(ctx, 0," not in src_default
    assert _call_fn(m, decay) + "(ctx, 0," in src_decay
    assert _call_fn(m, default) + "(ctx, 0," not in src_decay


def test_composed_sources_are_explicit_multiple_calls():
    m, _, default, decay = _source_module("composed_sources")
    P = _program("composed", m)
    U = _state(P)
    S0 = P.call(default, U, name="S0")
    S1 = P.call(decay, U, name="S1")
    P.commit("plasma", P.linear_combine("out", U + P.dt * S0 + P.dt * S1))

    assert P.validate() is True
    calls = [v.attrs["operator"] for v in P._values if v.op == "call"]
    assert calls == ["source_default", "decay"]
    src = P.emit_cpp_program(model=m)
    assert _call_fn(m, default) + "(ctx, 0," in src
    assert _call_fn(m, decay) + "(ctx, 0," in src
    assert ("sou" + "rces=") not in src


def test_forward_euler_uses_typed_rhs_operator():
    m, _, default, _ = _source_module("forward_euler")
    P = _program("fe", m)
    libtime.forward_euler(P, "plasma", rhs_operator=default)
    assert P.validate() is True
    assert [v.attrs["operator"] for v in P._values if v.op == "call"] == ["source_default"]
    assert _call_fn(m, default) + "(ctx, 0," in P.emit_cpp_program(model=m)


def test_forward_euler_rejects_source_selector_kwargs():
    m, _, default, _ = _source_module("fe_bad_kwargs")
    try:
        bad_kw = "sou" + "rces"
        libtime.forward_euler(_program("bad", m), "plasma", rhs_operator=default,
                              **{bad_kw: ("default",)})
    except TypeError as exc:
        assert "sou" + "rces" in str(exc)
    else:
        raise AssertionError("forward_euler must not accept selector kwargs")


def test_string_operator_selectors_are_rejected():
    m, _, _, _ = _source_module("string_reject")
    P = _program("bad", m)
    U = _state(P)
    try:
        P.call("source_default", U)
    except TypeError as exc:
        assert "typed operator handle" in str(exc)
    else:
        raise AssertionError("P.call must reject string operator selectors")

    try:
        libtime.forward_euler(P, "plasma", rhs_operator="source_default")
    except TypeError as exc:
        assert "typed operator handles" in str(exc)
    else:
        raise AssertionError("time macros must reject string operator selectors")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("PASS test_time_rhs_sources")


if __name__ == "__main__":
    main()
