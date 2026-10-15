"""Closed retained equations and independent counterexamples; native reception is separate."""
from copy import deepcopy

import numpy as np
import pops
import pytest

from examples.api_v02.method_retained_stencil import build_retained_stencil, initial_state, reference
from examples.api_v02.method_scoped_stencil import face_equation
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.model.stencil_codec import decode_stencil, encode_stencil


@pytest.mark.parametrize("retention", ("history", "continuation"))
def test_retained_face_equations_emit_complete_native_contract(retention):
    spec = build_retained_stencil(retention=retention)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.configure_stencil_expressions(") == 1
    assert source.index("ctx.configure_stencil_expressions(") < source.index("ctx.install(")
    assert "ctx.capture_stencil_exchanges(" in source
    assert "acceptance guard differs across execution lane" in source
    if retention == "history":
        assert "ctx.history(" in source and "ctx.store_history(" in source
    else:
        assert "ctx.continuation_publish(" in source
        assert "ContinuationKind::Field" in source
        assert "ctx.continuation_bind_field_provenance(source, destination, lifetime, metadata...)" in source


def test_ab2_oracle_distinguishes_noop_euler_and_missing_old_faces():
    initial = initial_state()
    final, changes, terms = reference(initial, steps=3)
    assert np.max(abs(final-initial)) > .02
    assert len(terms[-1]) == 2 and terms[-1][1][0] < 0
    missing_old = terms[-1][0][0]*terms[-1][0][1]
    exact = sum(weight*faces for weight, faces in terms[-1])
    assert np.max(abs(exact-missing_old)) > .005
    assert np.max(abs(changes[1]-changes[0])) > .001


@pytest.mark.parametrize("speed", (1., -.6))
def test_stencil_graph_codec_preserves_closed_equations_without_callback(speed):
    original = face_equation(speed)
    wire = encode_stencil(original)
    restored = decode_stencil(wire)
    assert encode_stencil(restored) == wire
    assert restored.to_data() == original.to_data()
    assert not hasattr(restored, "body")


def test_stencil_graph_codec_rejects_forward_reference_and_orphan_symbol():
    wire = deepcopy(encode_stencil(face_equation(1.)))
    changed = deepcopy(wire)
    name, _, metadata = changed["nodes"][-1]
    changed["nodes"] = (*changed["nodes"][:-1], (name, (len(changed["nodes"]),), metadata))
    with pytest.raises(ValueError, match="forward reference"):
        decode_stencil(changed)
    changed = deepcopy(wire)
    changed["nodes"] = (*changed["nodes"], ("Var", (), ("unbound", "stencil_sample")))
    with pytest.raises(ValueError, match="orphan"):
        decode_stencil(changed)


def test_stencil_graph_codec_preserves_lazy_boolean_and_rounded_nodes():
    from pops._ir import expr as ir
    from pops._ir.control_expr import Rounded, Where
    from pops.numerics.stencil_flux import StencilFlux
    def body(sample, spacing, axis):
        value = sample((0,), 0)
        predicate = ir.Compare("gt", value, ir.Const(0))
        return (Where(predicate, Rounded(value*value), ir.Abs(value)),), ir.Const(1)/spacing[axis]
    original = StencilFlux(dimension=1, components=1, radius=1, body=body)
    assert decode_stencil(encode_stencil(original)).to_data() == original.to_data()


def test_retained_checkpoint_capacity_is_derived_from_installed_program_not_archive():
    from pops.runtime._checkpoint_exchanges import exchange_checkpoint_byte_capacity
    spec = build_retained_stencil()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    capacity = exchange_checkpoint_byte_capacity(resolved.time, cells=(64,), dimension=2,
        rank_capacity=3, resolved_plan=resolved)
    assert type(capacity) is int and capacity > 64*4*2048


@pytest.mark.parametrize("retention", ("history", "continuation"))
def test_default_retained_budget_covers_sources_hidden_in_closed_graph(retention):
    from pops.codegen.program_stencil_budget import uniform_stencil_expression_budget
    program = build_retained_stencil(retention=retention).case._time
    bases, terms = uniform_stencil_expression_budget(program)
    assert bases >= 3 and terms >= 1


def test_default_fold_budget_counts_runtime_invocations_and_preserves_explicit_ceiling():
    from examples.api_v02.method_scoped_stencil import build_scoped_stencil
    from pops.codegen.program_stencil_budget import uniform_stencil_expression_budget
    program = build_scoped_stencil(mode="fold").case._time
    assert uniform_stencil_expression_budget(program)[0] >= 2
    program.flux_expression_budget = (1, 1)
    assert uniform_stencil_expression_budget(program) == (1, 1)
