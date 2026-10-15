"""Invocation metadata stays typed, detached and authenticated at execution."""
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from typed_program_support import typed_state


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_level_is_runtime_int64_in_nested_composition(target):
    p = Program("explicit_invocation_metadata")
    q = typed_state(p, "q", state_name="U")
    level = p.level()
    assert level.vtype == "int64" and level.inputs == () and level.block is None
    result = p.branch(level == 0, lambda b: b.level() + 1, lambda b: b.level() + 2)
    p.record_scalar("invocation", p.float64(result))
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    assert p.validate()
    source = emit_cpp_program(p, target=target)
    assert source.count(" = ctx.invocation_level();") == 3
    assert source == emit_cpp_program(detach_compiled_program(p), target=target)


def test_level_cannot_leak_from_a_lexical_region():
    p = Program("level_lexical_owner")
    leaked = []
    def child(b):
        value = b.level()
        leaked.append(value)
        return value
    p.branch(p.boolean(True), child, lambda b: b.int64(0))
    one = p.int64(1)
    before = p._ir_hash(), p._next_id
    with pytest.raises(ValueError):
        one + leaked[0]
    assert (p._ir_hash(), p._next_id) == before
