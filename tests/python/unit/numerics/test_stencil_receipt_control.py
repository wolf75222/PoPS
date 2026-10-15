"""Reject control-flow receipt loss instead of accepting an unrecorded face transfer."""
from fractions import Fraction
import re

import pops
import pytest

from pops.codegen.program_emit_stencil_flux import accepted_stencil_quadrature
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.numerics.stencil_flux import StencilFlux
from pops.time import FixedDt
from examples.api_v02.methods_T01_T12 import _storage


def _case(kind):
    case, layout, _, _, _, p, q, _ = _storage(
        "receipt_" + kind, ("q",), (1.,), cells=(8, 4), sampling="cell_average")
    old = q.n
    flux = StencilFlux(dimension=2, components=1, radius=1,
        body=lambda sample, spacing, axis:
            ((sample(tuple(-int(d == axis) for d in range(2)), 0),), 1./spacing[axis]))
    rate = p.stencil_flux_divergence("root_faces", old, flux=flux)
    candidate = p.value("root_candidate", old + p.dt*rate, at=q.next.point)
    untouched = p.value("root_untouched", old, at=q.next.point)
    if kind == "state_branch":
        selected = p.branch(p.boolean(True), lambda b: candidate, lambda b: untouched)
    elif kind == "rate_branch":
        selected_rate = p.branch(p.boolean(True), lambda b: rate, lambda b: rate)
        selected = p.value("from_branch", old + p.dt*selected_rate, at=q.next.point)
    elif kind == "state_capture":
        selected = p.require_success(p.capture(lambda b: candidate))
    elif kind == "rate_capture":
        selected_rate = p.require_success(p.capture(lambda b: rate))
        selected = p.value("from_capture", old + p.dt*selected_rate, at=q.next.point)
    elif kind == "state_match":
        selected = p.match(p.failure(untouched, 1), ok=lambda b, value: value,
                           err=lambda b, error: candidate)
    elif kind == "rate_match":
        selected_rate = p.match(p.failure(rate, 1), ok=lambda b, value: value,
                                err=lambda b, error: rate)
        selected = p.value("from_match", old + p.dt*selected_rate, at=q.next.point)
    elif kind == "state_fold":
        selected = p.fold(untouched, 1,
            lambda b, value, index: b.value("root_capture_in_fold", candidate))
    elif kind == "product_projection":
        selected = p.product(candidate=candidate)["candidate"]
    elif kind == "fresh_local_rate":
        local = p.rate("source_on_predictor", (candidate[0]**2,), of=candidate)
        selected = p.value("only_source_is_accepted", old + p.dt*local, at=q.next.point)
    elif kind == "direct":
        selected = candidate
    else:
        raise AssertionError(kind)
    p.commit(q.next, selected)
    p.step_strategy(FixedDt(.001))
    case.program(p)
    return case, layout, p, rate


@pytest.mark.parametrize("kind", (
    "state_branch", "rate_branch", "state_capture", "rate_capture",
    "state_match", "rate_match", "state_fold", "product_projection",
))
def test_control_result_cannot_silently_drop_root_stencil_receipt(kind):
    _, _, program, _ = _case(kind)
    with pytest.raises(ValueError, match="hides stencil faces outside affine quadrature"):
        accepted_stencil_quadrature(program)


def test_whole_codegen_publishes_only_the_owned_selected_stencil_candidate():
    case, layout, _, _ = _case("state_branch")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    # The static affine helper still rejects ambiguous control above. The complete
    # compiler instead carries native face provenance on the selected owned Field.
    selected = re.search(r"ctx\.publish_stencil_exchanges\(0, (\(\*branch_\d+_s0_owner\))\);", cpp)
    assert selected is not None
    assert cpp.count("ctx.publish_stencil_exchanges(") == 1
    assert "ctx.commit_many({{&u0, &%s}});" % selected.group(1) in cpp
    assert cpp.count("ctx.capture_stencil_exchanges(") == 1


def test_direct_flux_is_retained_and_a_fresh_rate_does_not_inherit_predictor_faces():
    _, _, program, rate = _case("direct")
    rows = accepted_stencil_quadrature(program)
    assert len(rows) == 1 and rows[0][0] is rate and rows[0][1] == {1: Fraction(1)}
    _, _, fresh, _ = _case("fresh_local_rate")
    assert accepted_stencil_quadrature(fresh) == ()
