#!/usr/bin/env python3
"""Operator-first separation of transport and source stages.

The old Program-level transport toggle is gone from the public model. A Program
builds a source-only stage by calling a typed source operator, and a transport
stage by calling a typed flux/rate operator. This test keeps the original
regression intent, but proves it through explicit handles and
``GeneratedModule::Operators`` calls.
"""

from pops.codegen.program_emit_module_ops import operator_function_name
from pops.ir.expr import Const
from pops import model
from pops import time as t


def _transport_source_module(name="transport_source"):
    m = model.Module(name + "_module")
    U = m.state_space("U", ("rho",))
    flux = m.operator(
        "flux",
        signature=(U,) >> model.Rate(U),
        kind="grid_operator",
        capabilities={"produces_rate": True, "supports_device": True, "default": True},
        expr={"x": [Const(1.0)], "y": [Const(0.0)]},
    )
    source = m.operator(
        "source_default",
        signature=(U,) >> model.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True, "default": True},
        expr=[Const(0.7)],
    )
    decay = m.operator(
        "decay",
        signature=(U,) >> model.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(0.5)],
    )
    return m, flux, source, decay


def _program(name, module):
    return t.Program(name).bind_operators(module)


def _state(P, block="plasma"):
    return P.state("U", block=block).n


def _call_fn(module, op):
    reg = module.operator_registry()
    return "GeneratedModule::Operators::%s" % operator_function_name(reg.id_of(op.name), op.name)


def _step(module, op, name="step", block="plasma"):
    P = _program(name, module)
    U = _state(P, block)
    R = P.call(op, U, name="R")
    P.commit(block, P.linear_combine("%s_out" % name, U + P.dt * R))
    return P


def test_source_only_stage_calls_only_source_operator():
    m, flux, source, _ = _transport_source_module("source_only")
    P = _step(m, source, "source_only")
    src = P.emit_cpp_program(model=m)

    assert [v.attrs["operator"] for v in P._values if v.op == "call"] == ["source_default"]
    assert _call_fn(m, source) + "(ctx, 0," in src
    assert _call_fn(m, flux) + "(ctx, 0," not in src
    assert "ctx.source_default_into(b, state, out);" in src
    assert "ctx.rhs_into(" not in src


def test_transport_only_stage_calls_only_flux_operator():
    m, flux, source, _ = _transport_source_module("transport_only")
    P = _step(m, flux, "transport_only")
    src = P.emit_cpp_program(model=m)

    assert [v.attrs["operator"] for v in P._values if v.op == "call"] == ["flux"]
    assert _call_fn(m, flux) + "(ctx, 0," in src
    assert _call_fn(m, source) + "(ctx, 0," not in src
    assert "ctx.neg_div_flux_default_into(b, state, out);" in src
    assert "ctx.rhs_into(" not in src


def test_source_and_transport_split_are_two_explicit_calls():
    m, flux, source, _ = _transport_source_module("split")
    P = _program("split", m)
    U = _state(P)
    H = P.call(flux, U, name="transport")
    U1 = P.linear_combine("after_transport", U + P.dt * H)
    S = P.call(source, U1, name="source")
    P.commit("plasma", P.linear_combine("out", U1 + P.dt * S))
    src = P.emit_cpp_program(model=m)

    assert [v.attrs["operator"] for v in P._values if v.op == "call"] == ["flux", "source_default"]
    assert _call_fn(m, flux) + "(ctx, 0," in src
    assert _call_fn(m, source) + "(ctx, 0," in src
    assert src.index(_call_fn(m, flux)) < src.index(_call_fn(m, source))


def test_distinct_stage_choices_have_distinct_ir_hashes():
    m, flux, source, decay = _transport_source_module("hashes")
    h_flux = _step(m, flux, "p")._ir_hash()
    h_source = _step(m, source, "p")._ir_hash()
    h_decay = _step(m, decay, "p")._ir_hash()
    assert len({h_flux, h_source, h_decay}) == 3


def test_source_operator_routes_to_each_block_index():
    m, _, source, _ = _transport_source_module("two_blocks")
    P = _program("two_blocks", m)
    Ua = _state(P, "a")
    Ub = _state(P, "b")
    Ra = P.call(source, Ua, name="Ra")
    Rb = P.call(source, Ub, name="Rb")
    P.commit("a", P.linear_combine("out_a", Ua + P.dt * Ra))
    P.commit("b", P.linear_combine("out_b", Ub + P.dt * Rb))
    src = P.emit_cpp_program(model=m)

    fn = _call_fn(m, source)
    assert fn + "(ctx, 0," in src
    assert fn + "(ctx, 1," in src
    assert 'pops_program_block_name' in src and '"a"' in src and '"b"' in src


def test_zero_stage_is_identity_without_operator_call():
    m, flux, source, decay = _transport_source_module("identity")
    P = _program("identity", m)
    U = _state(P)
    P.commit("plasma", P.linear_combine("out", 1.0 * U))
    src = P.emit_cpp_program(model=m)

    assert [v for v in P._values if v.op == "call"] == []
    assert _call_fn(m, flux) + "(ctx, 0," not in src
    assert _call_fn(m, source) + "(ctx, 0," not in src
    assert _call_fn(m, decay) + "(ctx, 0," not in src


def test_string_operator_selectors_are_rejected():
    m, _, _, _ = _transport_source_module("bad_string")
    P = _program("bad_string", m)
    U = _state(P)
    try:
        P.call("source_default", U)
    except TypeError as exc:
        assert "typed operator handle" in str(exc)
    else:
        raise AssertionError("P.call must reject string operator selectors")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("PASS test_time_rhs_flux_false")


if __name__ == "__main__":
    main()
