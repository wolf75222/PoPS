"""Typed IR interning preserves equations, calls, captures and correspondence."""

import re

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_kernel_registry import DeviceKernel, kernel_scope, kernel_definitions
from pops.codegen.program_source_map import source_map
from pops.time import Program
from typed_program_support import typed_state


def _program(bodies):
    p = Program("kernel_interning")
    q = typed_state(p, "material", state_name="U")
    _ = q.n
    domain = p.range_set(7, capacity=7, ownership="replicated")
    outputs = [p.map(domain, body) for body in bodies]
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    return p, outputs


def _kernels(source):
    return re.findall(r"^struct (pops_device_kernel_[a-f0-9]+) \{", source, re.MULTILINE)


def test_identical_equations_share_one_named_functor_and_keep_every_invocation():
    p, outputs = _program([lambda b, i: b.index_value(i) * 7 + 3] * 5)
    source = emit_cpp_program(p)
    assert len(_kernels(source)) == 1
    assert source.count('Kokkos::parallel_for("pops_finite_map_') == 5
    assert source.count("pops_finite_kernel_launches.fetch_add") == 5
    assert "KOKKOS_LAMBDA" not in source
    evidence = source_map(source)
    assert evidence["complete"]
    groups = [row for row in evidence["operations"] if row["channel"] == "interned_kokkos_body"]
    assert groups and len({row["shared_emission_group"] for row in groups}) == 1
    assert all(row["effective_ranges"] for row in groups)
    assert {output.id for output in outputs} <= {row["ssa"] for row in evidence["operations"]}
    assert source == emit_cpp_program(p)


@pytest.mark.parametrize(
    "bodies",
    (
        (lambda b, i: b.index_value(i) * 7, lambda b, i: b.index_value(i) * 8),
        (lambda b, i: b.index_value(i), lambda b, i: b.float64(b.index_value(i))),
        (lambda b, i: b.float64(1.25), lambda b, i: b.float64(1.5)),
    ),
)
def test_changed_literal_or_scalar_type_cannot_reuse_the_other_contract(bodies):
    p, _ = _program(bodies)
    assert len(_kernels(emit_cpp_program(p))) == 2


def test_noncommutative_capture_order_remains_part_of_the_equation():
    p, _ = _program([])
    domain = p.range_set(3, capacity=3, ownership="replicated")
    left, right = p.float64(11.0), p.float64(2.0)
    p.map(domain, lambda b, i: left - right)
    p.map(domain, lambda b, i: right - left)
    source = emit_cpp_program(p)
    assert len(_kernels(source)) == 2
    assert "capture_2 - capture_3" in source or "capture_3 - capture_2" in source


def test_registry_reset_after_failed_emission_does_not_leak_helpers():
    @kernel_scope
    def failed():
        kernel = DeviceKernel()
        kernel.capture("std::int64_t", "first")
        kernel.invocation(["(void)index;"])
        raise ValueError("failure after interning")

    with pytest.raises(ValueError, match="failure after interning"):
        failed()

    @kernel_scope
    def clean():
        return kernel_definitions()

    assert clean().strip() == ""
    with pytest.raises(RuntimeError, match="translation-unit"):
        kernel_definitions()
