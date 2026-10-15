"""GraphFunction lowers into existing native C++ operations after closed-data specialization."""
import json

from pops.codegen.program_codegen import emit_cpp_program
from pops.graph_functions import GraphFunction, graph
from pops.time import Program
from pops.time.variant_types import Failure, Float64, Int64, Product, Result, type_of
from typed_program_support import typed_state


def _commit(p, q):
    p.commit(q.next, p.value('accepted', q.n, at=q.next.point))


def test_serialized_nested_function_lowers_native_regions_without_callback_opcode():
    @graph(inputs=Product(x=Float64, n=Int64), output=Float64)
    def algorithm(p, a):
        return p.fold(a['x'], a['n'], lambda q, x, i:
                      q.branch(x > 0, lambda r: 2*x, lambda r: 3*x), max_steps=4)
    loaded = GraphFunction.from_data(json.loads(json.dumps(algorithm.to_data())))
    p = Program('caller')
    q = typed_state(p, 'q', state_name='U')
    p.record_scalar('one', loaded(p, x=p.float64(1), n=p.int64(2)))
    p.record_scalar('two', loaded(p, x=p.float64(-1), n=p.int64(3)))
    _commit(p, q)
    source = emit_cpp_program(p)
    assert source.count('for (; iteration_') == 2
    assert 'graph_argument' not in source and 'PyObject' not in source
    assert 'ctx.record_scalar("one"' in source and 'ctx.record_scalar("two"' in source


def test_result_boundary_lowers_capture_and_checked_scalar_failure():
    T = Result(Float64, Failure())
    @graph(inputs=Product(x=Float64), output=T)
    def f(p, a):
        return p.variant(T, 'ok', 1/a['x'])
    p = Program('caller')
    q = typed_state(p, 'q', state_name='U')
    r = f(p, x=p.float64(0))
    p.record_scalar('status', p.match(r, ok=lambda b, value: value,
                                     err=lambda b, failure: b.float64(-7)))
    _commit(p, q)
    source = emit_cpp_program(p)
    assert 'catch (const pops::runtime::program::StepAttemptRejected&' in source
    assert 'isfinite' in source
    # The closed sum now uses flat native storage; both active discriminants
    # must still be published by the actual success/failure paths.
    assert f'typed_capture_{r.id}_s0 = 0;' in source
    assert f'typed_capture_{r.id}_s0 = 1;' in source


def test_field_algebra_uses_original_authority_and_native_storage():
    p = Program('fields')
    q = typed_state(p, 'q', state_name='U')
    T = type_of(q.n)
    @graph(inputs=Product(value=T), output=T)
    def f(g, a):
        return g.value('double', 2*a['value'])
    value = f(p, value=q.n)
    p.commit(q.next, p.value('accepted', value, at=q.next.point))
    source = emit_cpp_program(p)
    assert 'ctx.scratch_state' in source and 'ctx.axpy' in source
    assert value.space is q.n.space and value.state_ref is q.n.state_ref
