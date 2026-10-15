"""Closed scoped equations and guard ordering; native outcomes have separate receivers."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_scoped_stencil import build_scoped_stencil, reference_step
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_lowerability import all_ops


@pytest.mark.parametrize("mode", ("branch", "fold"))
def test_scoped_faces_capture_after_guard_and_publish_selected_candidate(mode):
    spec = build_scoped_stencil(mode=mode)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    nodes = [node for node in all_ops(resolved.time) if node.op == "stencil_flux_divergence"]
    assert nodes and all(node.region != 0 for node in nodes)
    for node in nodes:
        completion = "ctx.complete_program_value_write(std::move(value_write_%d))" % node.id
        capture = "ctx.capture_stencil_exchanges(stencil_rhs_%d" % node.id
        # Value-authority instrumentation is optional when no field solve consumes
        # this graph; the prepared face kernel itself still converges its guard.
        if completion in source:
            assert source.index(completion) < source.index(capture)
        assert source.index("const pops::Real stencil_frequency_%d" % node.id) < source.index(capture)
        assert source.index("ctx.prepare_stencil_expression_write(stencil_rhs_%d, {});" % node.id) < source.index(capture)
        assert source.index("ctx.finish_stencil_expression_write(stencil_rhs_%d);" % node.id) < source.index(capture)
    assert source.count("ctx.publish_stencil_exchanges(") == 1
    assert "ctx.stage_exchange_batch(" not in source
    assert source.index("ctx.publish_stencil_exchanges(") > source.rindex("ctx.capture_stencil_exchanges(")


def test_two_evaluations_cannot_be_replaced_by_one_frozen_face_snapshot():
    initial = np.array([[[1., 1.2, .8, 1.1, .7, 1.3, .9, 1.]]])
    final, evaluated = reference_step(initial, dt=.04, dx=1/8, velocity=1., iterations=2)
    frozen = initial - .04 * 8 * (np.roll(evaluated[0], -1, axis=-1) - evaluated[0])
    assert np.max(abs(evaluated[0] - evaluated[1])) > .05
    assert np.max(abs(final - frozen)) > .01
    assert np.max(abs(final - initial)) > .05
