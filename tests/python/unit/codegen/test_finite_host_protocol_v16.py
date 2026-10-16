"""Source-level structural contracts; execution is received separately in real PoPS."""
from pops.codegen.program_emit_finite_arrays import _check_domain, _store_cell
from pops.codegen.program_codegen import emit_cpp_program
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from typed_program_support import typed_state


def test_domain_validation_keeps_exact_declared_family_capacity_and_live_pointer():
    lines = []
    _check_domain({"ownership": "replicated", "capacity": 9, "family": 3}, "domain", lines)
    assert lines == ["pops::runtime::program::finite_collective::require_domain(domain, 9, 3, ctx.prepared_execution_lane());"]


def test_cell_assignment_has_one_allocation_vote_before_publication():
    lines = []
    _store_cell("value", "destination", lines, "pending")
    assert lines == ["std::any pending;",
        "pops::runtime::program::finite_collective::assign_any(pending, value, ctx.prepared_execution_lane());",
        "destination = std::move(pending);"]


def test_closed_host_map_boundary_preserves_device_kernel_and_detachment():
    p = Program("finite_host_protocol")
    q = typed_state(p, "q", state_name="U")
    _ = q.n
    domain = p.range_set(4, capacity=5, ownership="replicated")
    values = p.map(domain, lambda b, i: b.float64(b.index_value(i))*2)
    p.reduce(values, p.float64(0), lambda b, s, x: s+x, order="ascending")
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    assert p.validate()
    source = emit_cpp_program(p)
    assert source == emit_cpp_program(detach_compiled_program(p))
    assert "[&]() POPS_FINITE_HOST_NOINLINE -> pops_algebra_finite::Array" in source
    assert "KOKKOS_FUNCTION void operator()(const std::int64_t index) const" in source
    kernel = source.split("KOKKOS_FUNCTION void operator()(", 1)[1].split("\n};", 1)[0]
    assert "NOINLINE" not in kernel and "finite_collective" not in kernel
    assert "finite_collective::allocate_view(" in source
    assert "finite_collective::mirror_copy(" in source
    assert "finite_collective::assign_any(" in source
    assert "finite_collective::allocate_values(" in source
    assert "_output_0_exhausted" not in source
