"""Split state endpoints and the active operator clock are distinct exact authorities."""
from fractions import Fraction
from types import SimpleNamespace
import re

import pops
import pytest

from pops.numerics.terms import Flux
from pops.time import Program, StagePoint, TimePoint
from pops.time._evaluation_point import evaluation_partition, evaluation_stage_fraction
from pops.time._program.detach import detach_compiled_program
from pops.codegen.program_emit_solve import _rhs_stage_fraction, _solve_stage_fraction
from tests.python.unit.time.test_time_std_imex_lie_ab import _authoring


def test_actual_native_split_failure_program_emits_all_qualified_subflows():
    from tests.python.integration.runtime.test_temporal_method_order_matrix import author_case
    from pops.codegen.program_graph_lowering import emit_program_graph
    from pops.codegen.program_models import ProgramModelGraph

    case, layout, program = author_case("split_failure")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    detached = detach_compiled_program(resolved.time)
    evaluations = [value for value in detached._values
                   if value.op in {"rhs", "local_transform"}]
    assert [value.attrs["evaluation_partition"] for value in evaluations] == [
        "first", "second", "first"]
    assert [evaluation_stage_fraction(value) for value in evaluations] == [0, 1, Fraction(1, 2)]
    assert not any("evaluation_partition" in value.attrs for value in detached._values
                   if value.op in {"state", "linear_combine", "store_history"})
    source = emit_program_graph(detached.to_graph(), lowering_program=detached,
                                model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert re.findall(r"ctx.set_stage_time\((\d+), (\d+)\);", source) == [
        ("0", "1"), ("1", "1"), ("1", "2")]
    assert "ctx.pointwise_status_max(" in source and "StepAttemptRejected(" in source
    rebuilt = program._rebuild(lambda _value: True)
    assert rebuilt.to_graph().graph_hash == program.to_graph().graph_hash
    assert "evaluation_partition" in str(detached._serialize())


@pytest.mark.parametrize("method", ("imex_euler", "imex_ars222"))
def test_existing_ark_factories_keep_explicit_rhs_qualification(method):
    from tests.python.integration.runtime.test_temporal_method_order_matrix import author_case
    from pops.codegen.program_graph_lowering import emit_program_graph
    from pops.codegen.program_models import ProgramModelGraph

    case, layout, _program = author_case(method)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    detached = detach_compiled_program(resolved.time)
    for value in detached._values:
        if value.op == "rhs":
            assert "evaluation_partition" not in value.attrs
            assert _rhs_stage_fraction(value) == Fraction(value.point.time_for("explicit").offset.to_python())
    source = emit_program_graph(detached.to_graph(), lowering_program=detached,
                                model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "ctx.set_stage_time(" in source


def test_distinct_split_coordinates_require_exact_operator_partition():
    program = Program("partition selection")
    point = StagePoint("two independent clocks", {
        "first": TimePoint(program.clock, Fraction(1, 4)),
        "second": TimePoint(program.clock, Fraction(3, 4)),
    })
    for partition, expected in (("first", Fraction(1, 4)), ("second", Fraction(3, 4))):
        value = SimpleNamespace(point=point, attrs={"evaluation_partition": partition})
        assert _rhs_stage_fraction(value) == _solve_stage_fraction(value) == expected
    with pytest.raises(ValueError, match="requires an explicit evaluation partition"):
        _rhs_stage_fraction(SimpleNamespace(point=point, attrs={}))
    with pytest.raises(ValueError, match="not declared"):
        _rhs_stage_fraction(SimpleNamespace(point=point, attrs={"evaluation_partition": "explicit"}))
    with pytest.raises(ValueError, match="non-empty string"):
        _rhs_stage_fraction(SimpleNamespace(point=point, attrs={"evaluation_partition": 1}))


def test_both_split_rhs_subflows_use_their_own_coordinate():
    from pops.lib.time import Strang
    state, _, _, _ = _authoring("both_split_rates")

    def subflow(program, current, fraction, *, at):
        rate = program.rhs(state=current, terms=[Flux()])
        return program.value("subflow", current + fraction * program.dt * rate, at=at)

    program = Strang(state, first=subflow, second=subflow)
    rates = [value for value in program._values if value.op == "rhs"]
    assert [value.attrs["evaluation_partition"] for value in rates] == ["first", "second", "first"]
    assert rates[1].point.time_for("first").offset.to_python() == Fraction(1, 2)
    assert rates[2].point.time_for("second").offset.to_python() == 1
    assert [_rhs_stage_fraction(value) for value in rates] == [0, 0, Fraction(1, 2)]


def test_ark_operator_semantics_remain_exact_and_do_not_generalize_by_name():
    program = Program("ARK coordinates")
    coordinates = {"explicit": TimePoint(program.clock, Fraction(1, 3)),
                   "implicit": TimePoint(program.clock, Fraction(2, 3))}
    value = SimpleNamespace(point=StagePoint("ark", coordinates), attrs={})
    assert _rhs_stage_fraction(value) == Fraction(1, 3)
    assert _solve_stage_fraction(value) == Fraction(2, 3)
    with pytest.raises(ValueError, match="requires an explicit evaluation partition"):
        evaluation_stage_fraction(value)
    value.point = StagePoint("not an ARK stage", {**coordinates, "other": TimePoint(program.clock)})
    with pytest.raises(ValueError, match="requires an explicit evaluation partition"):
        _rhs_stage_fraction(value)


def test_scoped_partition_restores_nested_owner_and_exception_boundaries():
    state, _, _, _ = _authoring("split_scope")
    program, foreign = Program("outer"), Program("foreign")
    current, other = program.state(state).n, foreign.state(state).n
    rhs = lambda owner, value: owner.rhs(state=value, terms=[Flux()])
    with evaluation_partition(program, "first"):
        first = rhs(program, current)
        assert "evaluation_partition" not in rhs(foreign, other).attrs
        with evaluation_partition(foreign, "second"):
            assert rhs(foreign, other).attrs["evaluation_partition"] == "second"
            assert "evaluation_partition" not in rhs(program, current).attrs
        with pytest.raises(RuntimeError, match="subflow failed"):
            with evaluation_partition(program, "second"):
                assert rhs(program, current).attrs["evaluation_partition"] == "second"
                raise RuntimeError("subflow failed")
        restored = rhs(program, current)
    assert first.attrs["evaluation_partition"] == restored.attrs["evaluation_partition"] == "first"
    assert "evaluation_partition" not in rhs(program, current).attrs


def test_qualifier_survives_rebuild_and_changes_semantic_graph_identity():
    state, _, _, _ = _authoring("split_identity")
    program = Program("identity")
    current = program.state(state)
    with evaluation_partition(program, "second"):
        rate = program.rhs(state=current.n, terms=[Flux()])
    program.commit(current.next, program.value("next", current.n + program.dt * rate,
                                               at=current.next.point))
    original = program.to_graph().graph_hash
    rebuilt = program._rebuild(lambda _value: True)
    assert rebuilt.to_graph().graph_hash == original
    rebuilt_rate = next(value for value in rebuilt._values if value.op == "rhs")
    assert rebuilt_rate.attrs["evaluation_partition"] == "second"
    program._replace_value(rate, attrs={key: value for key, value in rate.attrs.items()
                                       if key != "evaluation_partition"})
    assert program.to_graph().graph_hash != original
