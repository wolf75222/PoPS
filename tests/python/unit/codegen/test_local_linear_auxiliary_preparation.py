"""Local linear prerequisites at actual authored stage/SSA state; Source only."""
from fractions import Fraction
import re

import pytest
import pops
from pops.analytic import time, x
from pops.codegen.cpp_strings import cpp_string_expression
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_kernels import program_provider_consumer_qid
from pops.domain import Rectangle
from pops.fields import AnalyticAux
from pops.frames import Cartesian2D
from pops.solvers import DenseLU
from pops.time import FailRun, LocalLinear, Program, StagePoint, TimePoint


def _source(*, target, auxiliary="time"):
    program = Program("local coefficient stage")
    frame = Rectangle("coefficient frame", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("local coefficient owner", frame=frame)
    state = model.state("material", components=("amount",))
    name = "renamed_coefficient" if auxiliary == "renamed-time" else "coefficient"
    coefficient = (2 if auxiliary is None else
                   model.auxiliary(name, frame=frame.canonical_id))
    operator = model.operator("linear coefficient", returns=model.local_linear_operator(
        "linear coefficient", on=state, matrix=((-coefficient,),)))
    if auxiliary is not None:
        expression = x(frame) if auxiliary == "spatial" else time(program.clock)
        module = model.module
        module.aux_provider(AnalyticAux(module.aux_handle(module.aux()[name]),
                                      expression, frame=frame))
    case = pops.Case("local prerequisite case")
    block = case.block("exact owner", model)
    current = program.state(block[state])
    point = StagePoint("distinct coordinates", {
        "explicit": TimePoint(program.clock, offset=Fraction(1, 7)),
        "implicit": TimePoint(program.clock, offset=Fraction(4, 7))})
    predictor = program.value("exact predictor", 1 * current.n, at=point)
    linear = program.value("local operator", operator(program=program), at=point)
    solved = program.solve(LocalLinear(operator=program.I - program.dt * linear, rhs=predictor),
                           solver=DenseLU(), name="local evaluation").consume(action=FailRun())
    program.commit(current.next, program.value("endpoint", solved, at=current.next.point))
    lowered, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=lowered, target=target)
    node = next(value for value in program._values if value.op == "solve_local_linear")
    begin = source.index("const auto _pt%d =" % node.id)
    end = source.index("ctx.profile_record(", begin)
    qid = program_provider_consumer_qid(lowered, node.id, node.block)
    return source, source[begin:end], predictor, qid


@pytest.mark.parametrize("target", ("system", "amr_system"))
@pytest.mark.parametrize("auxiliary", ("time", "renamed-time", "spatial"))
def test_local_linear_prepares_exact_consumer_at_implicit_point_before_any_view(target, auxiliary):
    source, body, predictor, qid = _source(target=target, auxiliary=auxiliary)
    preparation = ("ctx.prepare_provider_values_for_solve(" if target == "system"
                   else "ctx.prepare_provider_values(")
    position = body.index(preparation)
    stage = body.index("ctx.set_stage_time(4, 7);")
    loop = body.index("for (int li = 0;")
    view = body.index("ctx.template provider_values_view<1>(")
    kernel = body.index("pops::for_each_cell(")
    assert stage < position < loop < view < kernel
    assert body.count(preparation) == 1
    expected = re.escape(preparation + cpp_string_expression(qid) + ", 0, u%d, " % predictor.id)
    assert re.search(expected + r"\d+\)", body)
    assert "provider_values_view<1>(%s, 0, li)" % cpp_string_expression(qid) in body
    assert "ctx.aux(" not in source
    assert "if (solve_failure_ == 0 && !pops::detail::mat_inverse<1>" in body
    assert ".solved_value_available()" in body


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_provider_free_local_linear_keeps_no_publication_or_nonempty_provider_reads(target):
    source, body, _, _ = _source(target=target, auxiliary=None)
    assert "ctx.prepare_provider_values" not in source
    assert "ctx.template provider_values_view<0>(" in body
    assert not re.search(r"provider_values_view<[1-9]\d*>", body)
    assert "auxiliary_nonfinite_candidate" not in body
    assert "if (solve_failure_ == 0 && !pops::detail::mat_inverse<1>" in body
