"""Interval control captures retain authority; timestamps alone never replace it."""
import pytest
import pops

from examples.api_v02.method_interval_regions import build_interval_regions
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.unit.time.test_realization_contributions import fixture


@pytest.mark.parametrize("amr", (False, True))
def test_interval_regions_emit_checks_inside_all_native_control_scopes(amr):
    spec = build_interval_regions(amr=amr)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, model=ProgramModelGraph.from_resolved_blocks(plan.blocks),
                           target="amr_system" if amr else "system")
    assert cpp.count("interval_attempt_") >= 6
    assert cpp.count(".same_attempt(ctx.resource_attempt())") >= 3
    assert cpp.count("ctx.boundary_evaluation_point(0)") >= 5
    assert cpp.count("pops::all_reduce_max(interval_valid_") >= 3
    assert "iteration_stop_" in cpp and "capture_status_" in cpp


def test_interval_cannot_be_relabelled_to_a_second_program_in_a_loop():
    _, p, q, _ = fixture()
    _, other, foreign, _ = fixture()
    window = p.interval()
    with pytest.raises(ValueError, match="another Program"):
        other.fold(foreign.n, 2, lambda b, value, index: window.bind(b, value, name="foreign"))
    assert q.n.prog is p


def test_interval_sidecar_is_not_fabricated_from_carried_endpoint_values():
    _, p, q, _ = fixture()
    window = p.interval()
    def body(b, value, index):
        from pops.intervals import Interval
        begin = window.begin+b.float64(0)
        with pytest.raises((ValueError, NotImplementedError), match="issued|root"):
            Interval(begin, window.duration, window.end, "reached")
        return value
    p.fold(q.n, 1, body)
