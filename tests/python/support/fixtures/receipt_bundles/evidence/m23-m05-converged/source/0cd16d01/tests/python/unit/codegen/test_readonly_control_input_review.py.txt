"""Read-only State input authority through lazy regions and a separate dt bound."""
from types import SimpleNamespace

import pops
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.inspect_compiled import _build_arguments
from pops.codegen.program_emit_field_routes import _walk_program_nodes
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt
from pops.time.references import handle_data


def test_nested_branch_and_dt_bound_readonly_inputs_do_not_become_publications():
    frame = Rectangle("input_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    case = pops.Case("qualified_nested_inputs")
    subjects = {}
    for name in ("dual", "nested_data", "bound_data", "unused"):
        model = pops.Model(name, frame=frame)
        state = model.state("U", components=("u",))
        subjects[name] = case.block(name, model)[state]
    program = pops.Program("nested_inputs")
    q, data, bound = (program.state(subjects[name])
                      for name in ("dual", "nested_data", "bound_data"))
    selected = program.branch(program.norm2(q.n) > 0,
        lambda P: P.branch(P.norm2(data.n) > 0, lambda inner: q.n, lambda inner: q.n),
        lambda P: q.n)
    program.commit(q.next, program.value("accepted", selected, at=q.next.point))
    program.set_dt_bound(lambda P, cfl: cfl/(1 + P.norm2(bound.n)))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    rows = []
    for block in resolved.blocks:
        adapter = graph.model_for_block(block.name)
        rows.append(SimpleNamespace(block_name=block.name, state_space=block.state_spaces[0],
            n_vars=adapter._m.n_vars, cons_names=tuple(adapter._m.cons_names), params={},
            provider_components=tuple(adapter._m._provider_components), model=adapter))
    compiled = SimpleNamespace(plan=CompiledPlanRecord.from_resolved(resolved),
                               bind_schema=resolved.bind_schema, target=resolved.target)
    program = resolved.time
    before = program._ir_hash()
    commits = tuple(program.commits())
    assert len(commits) == 1 and commits[0].block_ref.local_id == "dual"
    top = {value.state_ref.block_ref.local_id for value in program._values if value.op == "state"}
    assert "nested_data" not in top and "bound_data" not in top
    nested = {value.state_ref.block_ref.local_id: value.state_ref
              for value in _walk_program_nodes(program._values) if value.op == "state"}
    bounded = {value.state_ref.block_ref.local_id: value.state_ref
               for value in _walk_program_nodes(program._dt_bound[0]) if value.op == "state"}
    assert "nested_data" in nested and "bound_data" in bounded
    arguments = _build_arguments(compiled, program, tuple(rows))
    assert set(arguments.instances) == {"dual", "nested_data", "bound_data"}
    for name, reference in (("nested_data", nested["nested_data"]),
                            ("bound_data", bounded["bound_data"])):
        assert arguments.instances[name]["state_identity"] == handle_data(reference)
        assert arguments.instances[name]["required"] is True
    assert tuple(program.commits()) == commits and program._ir_hash() == before
