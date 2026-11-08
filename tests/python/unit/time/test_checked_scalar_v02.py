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


def test_collective_identity_follows_exact_operations_of_block_reductions():
    """Spec 0.3 SV20, duty 2: only a value that one collective gave every rank skips its vote."""
    from types import SimpleNamespace
    from pops.codegen.program_emit_structured_control import collective_identical

    def value(identity, op, *inputs, block=None, **attrs):
        return SimpleNamespace(id=identity, op=op, inputs=inputs, block=block, attrs=attrs)

    reduction = value(1, "reduce", block=0)
    observation = value(2, "reduce")
    step = value(3, "requested_dt")
    quotient = value(4, "scalar_op", reduction, reduction, fn="div")
    assert collective_identical(reduction) and collective_identical(quotient)
    assert collective_identical(value(5, "compare", quotient))
    assert collective_identical(value(6, "scalar_isfinite", quotient))
    assert collective_identical(value(7, "scalar_expression", reduction,
                                      expression_nodes=(("input", 0, 0), ("neg", 0))))
    assert not collective_identical(observation)
    assert not collective_identical(value(8, "scalar_op", quotient, step, fn="mul"))
    assert not collective_identical(value(9, "scalar_expression", reduction,
                                          expression_nodes=(("input", 0, 0), ("exp", 0))))


def test_reference_cg_rejects_its_collective_scalars_locally():
    """The CG of method_krylov stops on a dot_all and steps by quotients of dot_alls: no vote on
    them (SV20, duty 2). Its iterate budget is not a collective result and is still agreed on."""
    import pops
    from examples.api_v02.methods_T01_T12 import build_t04_global
    from pops.codegen.program_models import ProgramModelGraph

    spec = build_t04_global()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time,
                              model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.dot_all(") == 10
    assert "iterate stop differs across execution lane" not in source
    assert "replicated array Float64 bits differ" not in source
    assert source.count("iterate budget differs across execution lane") == 2
    guards = ('"nonfinite scalar arithmetic or invalid division"',
              '"nonfinite affine coefficient expression"', '"replicated array has a nonfinite element"')
    for guard in guards:
        position = source.find(guard)
        assert position >= 0
        while position >= 0:
            statement = source.rfind("if (", 0, position)
            assert "all_reduce" not in source[statement:position]
            position = source.find(guard, position + 1)
