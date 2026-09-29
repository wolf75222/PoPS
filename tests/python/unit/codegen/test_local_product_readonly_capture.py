"""A captured State is a bind input, without becoming an algebraic unknown."""
from types import SimpleNamespace

import pops

from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.inspect_compiled import _build_arguments
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual


def readonly_capture_case():
    frame = Rectangle("capture_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    case = pops.Case("one_unknown_frozen_target")
    subjects = []
    for name in ("dual", "target"):
        model = pops.Model(name, frame=frame)
        state = model.state("U", components=("a", "b", "c"))
        subjects.append(case.block(name, model)[state])
    program = pops.Program("one_unknown")
    dual, target = (program.state(subject) for subject in subjects)
    seed = program.value("seed", dual.n, at=dual.next.point)

    def residual(P, unknowns, *, frozen):
        return {"dual": tuple(unknowns["dual"][i]**2-frozen[i] for i in range(3))}

    solved = program.solve(LocalResidual(residual, {"dual": seed},
        captures={"frozen": target.n}), solver=LocalNewton(tolerance=1e-12)
        ).consume(action=FailRun())
    program.commit(dual.next, solved[subjects[0].block_ref])
    program.step_strategy(FixedDt(.01))
    case.program(program)
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4),
                                  periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout)


def _inspection_fixture(resolved):
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    rows = []
    for block in resolved.blocks:
        adapter = graph.model_for_block(block.name)
        rows.append(SimpleNamespace(block_name=block.name,
            state_space=block.state_spaces[0], n_vars=adapter._m.n_vars,
            cons_names=tuple(adapter._m.cons_names), params={},
            provider_components=tuple(adapter._m._provider_components), model=adapter))
    compiled = SimpleNamespace(plan=CompiledPlanRecord.from_resolved(resolved),
                               bind_schema=resolved.bind_schema, target=resolved.target)
    return graph, compiled, tuple(rows)


def test_one_unknown_product_already_accepts_frozen_foreign_block_capture():
    resolved = readonly_capture_case()
    graph, _, _ = _inspection_fixture(resolved)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert "prepare_local_nonlinear_problem<3>" in source
    assert "prepare_local_nonlinear_problem<6>" not in source
    assert len(resolved.time.commits()) == 1
    token, = [value for value in resolved.time._values
              if value.attrs.get("problem_kind") == "local_residual_product"]
    assert token.attrs["output_count"] == 1
    assert token.attrs["product_widths"] == (3, 3)
    assert token.inputs[0].block != token.inputs[1].block
    assert "local product requires co-located" in source


def test_readonly_captured_state_remains_an_exact_required_bind_input():
    resolved = readonly_capture_case()
    _, compiled, rows = _inspection_fixture(resolved)
    arguments = _build_arguments(compiled, resolved.time, rows)
    assert set(arguments.instances) == {"dual", "target"}
    assert arguments.instances["target"]["required"] is True
    assert arguments.instances["target"]["components"] == 3
    assert set(arguments.layout_runtime["ghost_depth_by_block"]) == {"dual", "target"}
