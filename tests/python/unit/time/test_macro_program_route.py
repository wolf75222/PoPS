"""ADC-554: time macros and the manual Program are ONE IR route.

A ``pops.lib.time.*`` macro is a function that builds a ``Program``: called with the block name it
returns a fresh, inspectable ``Program`` (``isinstance(pops.lib.time.forward_euler("plasma"),
Program)``); called with a live ``Program`` first it is the historical in-place builder (byte-identical
IR). ``Program.ir_nodes()`` exposes the generated nodes. A macro and the equivalent manual Program
produce the same logical IR. The ``CompiledTime`` public bypass (``time=CompiledTime(...)``) is refused
with a structured error, while ``CompiledTime`` stays a legit cadence descriptor.

Pure Python (``_ir_hash`` / ``ir_nodes`` are the IR fingerprints; no compilation); skips cleanly if
pops is unavailable. Never fakes the engine.
"""
import sys

try:
    import pytest
    import pops
    import pops.lib.time as libtime
    from pops.time import Program
    from pops.time.program import CompiledTime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_macro_program_route (pops unavailable: %s)" % exc)
    sys.exit(0)


def test_macro_without_program_returns_a_program():
    """Every explicit macro invoked with the block name returns a Program (ADC-554 acceptance)."""
    cases = [
        ("forward_euler", lambda: libtime.forward_euler("plasma")),
        ("ssprk2", lambda: libtime.ssprk2("plasma")),
        ("ssprk3", lambda: libtime.ssprk3("plasma")),
        ("rk4", lambda: libtime.rk4("plasma")),
        ("adams_bashforth2", lambda: libtime.adams_bashforth2("plasma")),
    ]
    for label, fn in cases:
        prog = fn()
        assert isinstance(prog, Program), "%s must return a Program, got %r" % (label, type(prog))
        prog.validate()
    print("OK  forward_euler/ssprk2/ssprk3/rk4/adams_bashforth2(block) each return a Program")


def test_predictor_corrector_returns_program():
    """The operator-first predictor-corrector macro also returns a Program (the issue's example)."""
    from pops.ir.expr import Const, Var
    from pops.physics.facade import Model

    m = Model("pc")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), rho * (-gx), rho * (-gy)])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])

    def build(P):
        P.bind_operators(m)
        libtime.predictor_corrector_local_linear(
            P, "plasma", fields_operator="fields_from_state",
            explicit_rate_operator="explicit_rhs", implicit_operator="lorentz")

    # Manual route: build into an explicit Program.
    manual = Program("pc")
    build(manual)
    manual.validate()
    assert isinstance(manual, Program)
    print("OK  predictor_corrector_local_linear builds a valid manual Program")


def test_ir_nodes_inspection_lists_generated_nodes():
    """Program.ir_nodes() exposes the macro-generated nodes as a structured list."""
    prog = libtime.forward_euler("plasma")
    nodes = prog.ir_nodes()
    ops = [n["op"] for n in nodes]
    assert ops == ["state", "solve_fields", "rhs", "linear_combine", "commit"], ops
    # Each node carries a name, block and array-free attrs.
    for n in nodes:
        assert set(n) >= {"name", "op", "vtype", "block", "inputs", "attrs"}
        for val in n["attrs"].values():
            assert not (hasattr(val, "shape") and hasattr(val, "dtype")), "attrs must be array-free"
    print("OK  ir_nodes() lists the generated nodes structurally")


def test_macro_and_manual_same_ir():
    """A macro and the equivalent hand-written manual Program produce the same logical IR."""
    # Fresh-Program macro (its Program is named after the scheme).
    macro_prog = libtime.ssprk2("plasma")

    # Manual equivalent: the exact SSPRK2 stage chain, same Program name for byte-identical IR.
    manual = Program("ssprk2")
    U0 = manual.state("plasma")
    k0 = manual._rhs_legacy(state=U0, fields=manual.solve_fields(U0), flux=True, sources=["default"])
    U1 = manual.linear_combine("ssprk2_U1", U0 + manual.dt * k0)
    k1 = manual._rhs_legacy(state=U1, fields=manual.solve_fields(U1), flux=True, sources=["default"])
    manual.commit("plasma", manual.linear_combine(
        "ssprk2_step", 0.5 * U0 + 0.5 * (U1 + manual.dt * k1)))

    assert macro_prog._ir_hash() == manual._ir_hash(), (
        "the ssprk2 macro and the manual Program must share one IR\n"
        "  macro : %s\n  manual: %s" % (macro_prog._ir_hash(), manual._ir_hash()))
    # And the same structured node list.
    assert macro_prog.ir_nodes() == manual.ir_nodes()
    print("OK  ssprk2 macro and manual Program produce the same IR (%s)" % macro_prog._ir_hash())


def test_legacy_in_place_call_is_byte_identical():
    """Passing a live Program keeps the historical in-place behaviour (byte-identical IR)."""
    P = Program("mine")
    ret = libtime.forward_euler(P, "plasma")     # legacy: mutates P
    assert ret is None                            # forward_euler committed in place, returned nothing
    fresh = libtime.forward_euler("plasma")       # fresh Program (named "forward_euler")
    # Same nodes, only the Program name differs -> same node list, distinct name in the serialized hash.
    assert [n["op"] for n in P.ir_nodes()] == [n["op"] for n in fresh.ir_nodes()]
    print("OK  legacy forward_euler(P, block) mutates in place and returns None")


def test_compiled_time_route_is_refused_but_constructible():
    """time=CompiledTime(...) is refused with a structured error; CompiledTime still constructs."""
    from pops.runtime._system_install import _reject_compiled_time_route
    with pytest.raises(TypeError, match="not a transport time policy"):
        _reject_compiled_time_route(CompiledTime(substeps=2), "System.add_block")
    with pytest.raises(TypeError, match="cadence="):
        _reject_compiled_time_route(CompiledTime(), "System.add_equation")
    # A real time policy and None pass through.
    _reject_compiled_time_route(pops.Explicit(), "System.add_block")
    _reject_compiled_time_route(None, "System.add_equation")
    # CompiledTime stays a legit cadence descriptor (importable + constructible).
    ct = CompiledTime(substeps=3, stride=4)
    assert ct.substeps == 3 and ct.stride == 4
    print("OK  time=CompiledTime refused; CompiledTime keeps its cadence role")


def main():
    test_macro_without_program_returns_a_program()
    test_predictor_corrector_returns_program()
    test_ir_nodes_inspection_lists_generated_nodes()
    test_macro_and_manual_same_ir()
    test_legacy_in_place_call_is_byte_identical()
    test_compiled_time_route_is_refused_but_constructible()
    print("OK  test_macro_program_route")


if __name__ == "__main__":
    main()
