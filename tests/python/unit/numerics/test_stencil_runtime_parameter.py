"""A ``StencilFlux`` body reads a runtime parameter of the model of its block (N3 lot 1).

The body of a ``riemann.User`` face reads a ``RuntimeParam`` of its model through
``ctx.program_params(block)``; the body of a ``StencilFlux`` reads it the same way, so that a
coefficient written as ``model.parameter`` in the surface (nu of L1, §17.1) is bind data and not
generated code. The value is not in the identity: two defaults give the same semantic identity,
the same artifact hash and the same C++. Only a RuntimeParam declared by the model of the block is
read, and only one that a formula of that model reads (it has a slot in the block's table); any
other is refused by its name. The native run with two values of nu is
``tests/python/integration/runtime/test_stencil_runtime_parameter_native.py``.
"""
import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_params import program_param_entries
from pops.codegen.program_models import ProgramModelGraph

from tests.python.support.stencil_runtime_parameter import heat_case


def _resolved(nu_default=0.01, **options):
    case, layout, *_ = heat_case(nu_default=nu_default, **options)
    return pops.resolve(pops.validate(case), layout=layout)


def _cpp(resolved):
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def test_the_body_reads_the_parameter_from_the_block_table():
    source = _cpp(_resolved())
    assert "const pops::RuntimeParams stencil_params_" in source
    assert "const auto params = stencil_params_" in source
    assert "params.get(0)" in source


def test_two_stencils_bind_the_table_each_in_its_own_name():
    source = _cpp(_resolved(stencils=2))
    bindings = [line.strip() for line in source.splitlines()
                if line.strip().startswith("const pops::RuntimeParams stencil_params_")]
    assert len(bindings) == 2 and len(set(bindings)) == 2


def test_the_value_of_the_parameter_is_not_in_the_identity():
    one, other = _resolved(0.01), _resolved(0.05)
    assert one.snapshot.semantic_identity == other.snapshot.semantic_identity
    assert one.snapshot.artifact_hash == other.snapshot.artifact_hash
    assert _cpp(one) == _cpp(other)


def test_the_parameter_is_routed_to_its_block_for_bind():
    resolved = _resolved()
    entries = program_param_entries(resolved.time, ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))
    assert [(block, name, index) for block, name, index, _ in entries] == [(0, "nu", 0)]


def test_a_parameter_of_another_model_is_refused_by_its_name():
    with pytest.raises(ValueError, match="'nu', which the model of the block of its state does "
                                         "not declare"):
        heat_case(other_model_parameter=True)


def test_a_parameter_that_no_formula_of_the_model_reads_is_refused_by_its_name():
    with pytest.raises(ValueError, match="'nu', which no formula of the model of its block reads"):
        _cpp(_resolved(declare_flux=False))
