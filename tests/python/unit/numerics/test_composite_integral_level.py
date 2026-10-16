"""Explicit integration domains preserve tower authority and exact control types."""
from types import SimpleNamespace

import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_composite import _contract, _helper
from pops.codegen.program_lowerability import all_ops
from pops.codegen.program_models import ProgramModelGraph
from examples.api_v02.method_relaxed_composite import build_t10_composite
from tests.python.unit.numerics.test_api_v02_composite_integral import _authoring


def integral_value(level=None):
    program, state = _authoring()
    result = []
    def body(builder):
        selected = None if level is None else builder.int64(level)
        result.append(builder.composite_integral("volume", 1., over=state.n, level=selected))
    program.after_synchronization(body)
    return result[0]


def test_unselected_integral_retains_its_original_schema_and_support():
    value = integral_value()
    assert value.attrs["schema_version"] == 1
    assert "level_input" not in value.attrs and len(value.inputs) == 1
    _contract(value)


@pytest.mark.parametrize("level", (0, 1, -1, 2**63-1))
def test_selector_is_owned_exact_int64_and_range_is_checked_by_live_hierarchy(level):
    value = integral_value(level)
    index = value.attrs["level_input"]
    assert value.attrs["schema_version"] == 2 and index == 1
    assert value.inputs[index].vtype == "int64"
    assert value.inputs[index].attrs["value"] == level
    assert all(node[0] != "input" for node in value.attrs["expression_nodes"])
    _contract(value)
    assert "capture_1" not in _helper(value)


@pytest.mark.parametrize("bad", (0, 1., True))
def test_plain_python_selector_is_not_an_invocation_owned_value(bad):
    program, state = _authoring()
    with pytest.raises((TypeError, ValueError), match="owned|ProgramValue|Int64|int64|native"):
        program.after_synchronization(lambda p:
            p.composite_integral("bad", 1., over=state.n, level=bad))


def test_selector_from_another_program_is_rejected():
    program, state = _authoring()
    foreign = pops.Program("foreign").int64(0)
    with pytest.raises(ValueError, match="Program|program"):
        program.after_synchronization(lambda p:
            p.composite_integral("bad", 1., over=state.n, level=foreign))


@pytest.mark.parametrize("change", (
    {"schema_version": 1}, {"level_input": 0}, {"level_input": 20},
    {"expression_nodes": (("input", 1, 0),)},
))
def test_detached_lowering_cannot_forge_or_consume_the_domain_as_an_integrand(change):
    value = integral_value(0)
    copied = SimpleNamespace(attrs={**value.attrs, **change}, inputs=value.inputs, block=value.block)
    with pytest.raises(ValueError, match="level|Int64"):
        _contract(copied)


@pytest.mark.parametrize("ratio", (1, 2))
def test_rotation_declares_stable_polynomials_in_separate_physical_level_domains(ratio):
    spec = build_t10_composite(temporal_ratio=ratio, requested_dt=.4, t_end=.7,
                              clip_to_frontier=True)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    values = list(all_ops(plan.time))
    integrals = [v for v in values if v.op == "composite_integral"]
    assert len(integrals) == 4
    assert [v.inputs[v.attrs["level_input"]].attrs["value"] for v in integrals] == [0, 0, 1, 1]
    for value in integrals:
        # Current State is support only. The polynomial reads the immutable
        # macro entry and h, never a rounded provisional-minus-entry field.
        support = value.attrs["support_input"]
        assert all(node[0] != "input" or node[1] != support
                   for node in value.attrs["expression_nodes"])
    cpp = emit_cpp_program(plan.time, target="amr_system",
                           model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert cpp.count("struct ProgramCompositeIntegrand_") == 4
    for value in integrals:
        assert "ctx.composite_integral(0, composite_integrand_%d, " % value.id in cpp
