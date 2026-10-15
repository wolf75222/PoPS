"""Bool equations keep SSA ownership, closed identity, and native placements."""
import copy
import json

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_finite_kokkos import admission
from pops.graph_functions import GraphFunction, graph
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Bool, Int64, Product, Result
from typed_program_support import typed_state


def _carrier():
    program = Program("boolean_algebra")
    state = typed_state(program, "material", state_name="U")
    _ = state.n
    return program, state


def _finish(program, state):
    program.commit(state.next, program.value("accepted", state.n, at=state.next.point))
    assert program.validate()
    cpp = emit_cpp_program(program)
    assert cpp == emit_cpp_program(detach_compiled_program(program))
    return cpp


def test_bool_equations_are_pure_ssa_and_keep_all_operand_evaluations():
    p, state = _carrier()
    a, b = p.float64(2) > 1, p.float64(4) < 7
    values = (a & b, a | b, ~a, True & a, False | b, a & False, b | True)
    assert all(value.vtype == "bool" and value.op == "bool_op" for value in values)
    assert values[0].inputs == (a, b)
    assert values[2].inputs == (a,)
    assert not any(value.op == "branch" for value in p._values)
    cpp = _finish(p, state)
    assert all(cpp.count("const bool boolean_%d =" % value.id) == 1 for value in values)
    assert "&&" in cpp and "||" in cpp and "(!(" in cpp
    # Python short circuiting would attempt a host truth conversion and is refused.
    with pytest.raises(TypeError):
        bool(values[0])


@pytest.mark.parametrize("kind", ("integer", "scalar", "foreign", "python_int", "field"))
def test_boolean_operands_are_typed_owned_and_failure_is_atomic(kind):
    p, state = _carrier()
    operand = {"integer": lambda: p.int64(1), "scalar": lambda: p.float64(1),
               "foreign": lambda: Program("foreign").boolean(True),
               "python_int": lambda: 1, "field": lambda: state.n}[kind]()
    a = p.boolean(True)
    before = p._ir_hash(), p._next_id
    with pytest.raises((TypeError, ValueError)):
        a & operand
    assert (p._ir_hash(), p._next_id) == before
    with pytest.raises((TypeError, ValueError)):
        operand | a
    assert (p._ir_hash(), p._next_id) == before


def test_branch_local_bool_cannot_escape_its_lexical_scope():
    p, _ = _carrier()
    leaked = []
    def arm(b):
        result = b.boolean(True)
        leaked.append(result)
        return result
    p.branch(p.boolean(True), arm, lambda b: b.boolean(False))
    a = p.boolean(True)
    before = p._ir_hash()
    with pytest.raises(ValueError):
        a & leaked[0]
    assert p._ir_hash() == before


def test_closed_boolean_graph_roundtrip_rejects_wrong_arity_or_operation():
    @graph(inputs=Product(a=Bool, b=Bool), output=Bool)
    def formula(p, args):
        return (args["a"] & ~args["b"]) | False
    wire = json.loads(json.dumps(formula.to_data()))
    restored = GraphFunction.from_data(wire)
    assert restored.identity == formula.identity
    for operation in ("xor-forged", "not"):
        bad = copy.deepcopy(wire)
        node = next(node for node in bad["nodes"] if node["op"] == "bool_op"
                    and dict(node["attrs"]["$map"])["operation"] == "and")
        node["attrs"]["$map"] = [["operation", operation]]
        # A forged graph must fail validation or identity authentication, never lower.
        with pytest.raises((TypeError, ValueError)):
            GraphFunction.from_data(bad)


def test_boolean_map_keeps_real_kokkos_placement():
    p, state = _carrier()
    domain = p.range_set(7, capacity=7, ownership="replicated")
    output = p.map(domain, lambda b, i: ((b.index_value(i) > 1) &
                    ~(b.index_value(i) > 5)) | (b.index_value(i) == 0))
    assert admission(output)[0] is None
    cpp = _finish(p, state)
    kernel = cpp.split("KOKKOS_FUNCTION void operator()(", 1)[1].split("\n};", 1)[0]
    assert "&&" in kernel and "||" in kernel and "(!(" in kernel
    assert "MPI_" not in kernel and "ctx." not in kernel


def test_boolean_distributed_local_map_uses_native_bool_values_without_collectives():
    from pops.codegen.program_emit_distributed_arrays import OPS, emit_distributed_op
    p = Program("distributed_boolean")
    d = Distributed(p)
    domain = d.domain((0, 2, 4), capacity=4)
    kind = Result(Bool, Int64)
    d.local_map(domain, lambda b, i: b.variant(kind, "ok",
                ((d.gid(i) > 0) & ~(d.gid(i) > 2)) | (d.gid(i) == 0)), byte_capacity=4096)
    var, lines = {}, []
    for value in p._values:
        if value.op == "int64_constant":
            var[value.id] = "std::int64_t(%s)" % value.attrs["value"]
        elif value.op in OPS:
            emit_distributed_op(p, value, var, None, lines, [], {}, None, "system")
    local = "\n".join(lines).split("[&](auto distributed_index){", 1)[1]
    assert "->boolean()" in local and "&&" in local and "||" in local and "(!(" in local
    assert "MPI_" not in local and "all_reduce" not in local and "ctx." not in local
