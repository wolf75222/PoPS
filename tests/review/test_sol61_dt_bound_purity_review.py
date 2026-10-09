"""Public refusal after readonly allocations, orphan rejection and exact retry."""
import copy

import pops
import pytest

from pops.codegen.program_codegen import _emit_dt_bound, emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt


@pytest.mark.parametrize("mutation", ("cadence", "strategy", "history"))
def test_partial_query_allocation_is_rolled_back_and_orphans_cannot_be_reused(mutation):
    frame = Rectangle("retry_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("retry_model", frame=frame)
    state = model.state("U", components=("left", "right"))
    case = pops.Case("retry_case")
    fluid, query = case.block("fluid", model), case.block("query", model)
    program = pops.Program("retry_program")
    q = program.state(fluid[state])
    program.commit(q.next, program.value("identity", (q.n[0], q.n[1]), at=q.next.point))
    program.step_strategy(FixedDt(.1))
    case.program(program)
    before = (copy.deepcopy(program._serialize()), program._next_id, program._next_region,
              tuple(program._time_states))
    escaped = []

    def failed_query(P, cfl):
        readonly = P.state(query[state]).n
        escaped.append(readonly)
        result = cfl / (1 + P.dot_all(readonly, readonly))
        if mutation == "cadence":
            P.cadence(substeps=2)
        elif mutation == "strategy":
            P.step_strategy(FixedDt(.2))
        else:
            P.keep_history(q, depth=2)
        return result

    with pytest.raises(ValueError, match="read-only callback changed authoring metadata"):
        program.set_dt_bound(failed_query)
    assert (program._serialize(), program._next_id, program._next_region,
            tuple(program._time_states)) == before
    assert not program.has_dt_bound() and len(program.commits()) == 1
    with pytest.raises(ValueError, match="was not authored by this Program"):
        program.dot_all(escaped[0], escaped[0])
    assert program._serialize() == before[0]

    def pure_query(P, cfl):
        readonly = P.state(query[state]).n
        return cfl / (1 + P.dot_all(readonly, readonly))

    program.set_dt_bound(pure_query)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    _, body, _ = _emit_dt_bound(resolved.time, graph)
    assert body.count("ctx.dot_all(1,") == 1 and "ctx.state(1)" in body
    assert "commit" not in body and "pops_install_program" in source
    assert len(program.commits()) == 1
