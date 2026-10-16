"""Partial scalar equations must fail inside their native capture boundary."""
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.time import Program
from typed_program_support import typed_state


def test_nonfinite_scalar_arithmetic_is_captured_after_a_collective_rendezvous():
    p = Program("checked_scalar")
    state = typed_state(p, "material", state_name="U")
    captured = p.capture(lambda q: q.requested_dt() / (q.requested_dt() - q.requested_dt()))
    recovered = p.match(captured, ok=lambda q, x: x,
                        err=lambda q, e: q.requested_dt() * 0. + 7.)
    p.record_scalar("recovered", recovered)
    p.commit(state.next, p.value("unchanged", state.n, at=state.next.point))
    source = emit_cpp_program(p)
    guard = source.index('"nonfinite scalar arithmetic or invalid division"')
    assert source.rfind("pops::all_reduce_max", 0, guard) >= 0
    assert source.index("try {") < guard < source.index("catch (const pops::runtime::program::StepAttemptRejected&)")
    assert p._serialize()["version"] >= 28


def test_finiteness_is_a_native_predicate_and_rejects_implicit_array_casts():
    p = Program("finite_predicate")
    state = typed_state(p, "material", state_name="U")
    dt = p.requested_dt()
    finite = p.isfinite(dt)
    selected = p.branch(finite, lambda q: dt, lambda q: dt * 0.)
    p.record_scalar("selected", selected)
    p.commit(state.next, p.value("unchanged", state.n, at=state.next.point))
    source = emit_cpp_program(p)
    assert "std::isfinite" in source
    with pytest.raises(ValueError, match="expected a scalar"):
        p.isfinite(p.int64(4))


def test_common_elementary_algebra_is_closed_over_native_scalar_regions():
    from pops import math as pm
    p = Program("scalar_elementary")
    state = typed_state(p, "material", state_name="U")
    expression = p.capture(lambda t: pm.sin(t.requested_dt()) + pm.log(t.float64(2.))
                           + pm.sqrt(t.float64(3.)) + abs(-t.requested_dt()) + t.requested_dt()**2.3)
    scalar = p.require_success(expression)
    p.record_scalar("elementary", scalar)
    p.commit(state.next, p.value("unchanged", state.n, at=state.next.point))
    source = emit_cpp_program(p)
    assert "Kokkos::sin" in source and "Kokkos::log" in source and "std::pow" in source
    assert "scalar expression domain or nonfinite result" in source


def test_native_scalar_and_field_component_share_equations_in_both_operand_orders():
    from pops import math as pm
    from pops.time.expressions import encode_expressions, ProgramScalar
    from pops._ir.expr import Const
    p = Program("mixed_scalar_field")
    state = typed_state(p, "material", state_name="U")
    h, u = p.requested_dt(), state.n[0]
    for expression in (u + h, h + u, u - h, h - u, u * h, h * u,
                       u / h, h / u, pm.minimum(h, u), pm.maximum(u, h)):
        _, _, inputs = encode_expressions((expression,), p)
        assert {value.id for value in inputs} == {h.id, state.n.id}
    # Equal declaration names never identify independent SSA inputs for AD.
    other = Program("mixed_scalar_field").requested_dt()
    x = ProgramScalar(h)
    assert x.__pops_ir_diff__(None, ProgramScalar(h), {}).value == Const(1).value
    assert x.__pops_ir_diff__(None, ProgramScalar(other), {}).value == Const(0).value


def test_public_derivatives_dispatch_only_on_mathematical_graph_type():
    from pops.algebra import graph, Product, Float64, GraphPairings, jvp, vjp
    from pops.graph_functions import GraphFunction
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        return a["x"] * a["x"]
    @graph(inputs=Product(value=Float64), output=Float64)
    def output_pairing(p, a):
        return a["value"]
    @graph(inputs=Product(value=Product(x=Float64)), output=Product(x=Float64))
    def input_pairing(p, a):
        return a["value"]
    assert type(jvp(f)) is GraphFunction
    assert type(vjp(f, pairings=GraphPairings(output_pairing, input_pairing))) is GraphFunction
    with pytest.raises(TypeError, match="returned graph"):
        jvp(f, direction=(1,))
