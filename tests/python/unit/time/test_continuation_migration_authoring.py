"""The migration receiver is one equation graph independent of MPI cardinality."""
import re

from examples.api_v02.continuation_migration_amr import build_continuation_migration_amr
from pops.codegen.program_codegen import emit_cpp_program
import pops


def test_migration_graph_uses_native_balanced_domains_and_keeps_aliases():
    spec = build_continuation_migration_amr(cells=(8, 8), steps=4)
    pops.resolve(pops.validate(spec.case), layout=spec.layout)
    program = spec.case._time
    source = emit_cpp_program(program, target="amr_system")
    assert ".issue_balanced(" in source
    assert "begin_continuation_forest" in source
    assert "begin_continuation_parent" in source
    assert "history_alias" in source
    # Post-sync is a separately generated C++ callback.  A binding cached in
    # the time-step SSA environment cannot establish its lexical lane here.
    bindings = re.findall(r"auto& (distributed_\d+)_registry = ctx\.distributed_registry\(\);", source)
    assert len(bindings) >= 3
    for token in bindings:
        declaration = f"const auto& {token}_lane = {token}_registry.lane();"
        assert source.count(declaration) == 1
        registry_at = source.index(f"auto& {token}_registry =")
        declaration_at = source.index(declaration)
        use_at = source.index(f"::phase({token}_lane,", declaration_at)
        assert registry_at < declaration_at < use_at
    assert "phase(distributed_lane," not in source
    again = build_continuation_migration_amr(cells=(8, 8), steps=4)
    assert program._ir_hash() == again.case._time._ir_hash()
