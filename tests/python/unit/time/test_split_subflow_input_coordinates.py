"""Public split callbacks distinguish input evaluation from returned state endpoints."""
from fractions import Fraction
import re

import pytest
import pops
from pops.analytic import time
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.fields import AnalyticAux
from pops.frames import Cartesian2D
from pops.lib.time import Lie, Strang
from pops.time import StagePoint
from pops.time._evaluation_point import evaluation_stage_fraction


def _coordinate(point, partition):
    selected = point.time_for(partition) if isinstance(point, StagePoint) else point
    return Fraction(selected.step) + Fraction(selected.offset.to_python())


def _authored(factory, *, at_endpoint=False, prefix="arbitrary_map"):
    frame = Rectangle(prefix + " frame", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model(prefix + " model", frame=frame)
    state = model.state("material", components=("amount",))
    coefficient = model.auxiliary("authored_clock_coefficient", frame=frame.canonical_id)
    operation = model.local_transform(prefix, (state[0] + coefficient,), valid_if=state[0] >= 0)
    case = pops.Case(prefix + " case")
    block = case.block("selected material", model)
    windows = []

    def flow(partition):
        def builder(program, current, fraction, *, at):
            begin, end = _coordinate(current.point, partition), _coordinate(at, partition)
            windows.append((partition, begin, end, Fraction(fraction)))
            evaluated = (program.value(partition + " explicit evaluation state", 1 * current, at=at)
                         if at_endpoint else current)
            transformed = program.transform(evaluated, transform=operation)
            return program.value(partition + " returned endpoint", transformed, at=at)
        return builder

    program = factory(block[state], first=flow("first"), second=flow("second"))
    module = model.module
    module.aux_provider(AnalyticAux(module.aux_handle(module.aux()["authored_clock_coefficient"]),
                                  time(program.clock), frame=frame))
    lowered, _ = lower_and_validate(model, facade=model)
    return program, lowered, tuple(windows)


@pytest.mark.parametrize("factory,expected", [
    (Strang, (("first", 0, Fraction(1, 2), Fraction(1, 2)),
              ("second", 0, 1, 1),
              ("first", Fraction(1, 2), 1, Fraction(1, 2)))),
    (Lie, (("first", 0, 1, 1), ("second", 0, 1, 1))),
])
@pytest.mark.parametrize("prefix", ["arbitrary_map", "renamed_independent_map"])
def test_public_callback_input_points_follow_each_subflow_window(factory, expected, prefix):
    program, model, windows = _authored(factory, prefix=prefix)
    assert windows == expected
    transforms = [node for node in program._values if node.op == "local_transform"]
    assert [evaluation_stage_fraction(node) for node in transforms] == [row[1] for row in expected]
    assert all(node.point == node.inputs[0].point for node in transforms)
    assert all(row[2] - row[1] == row[3] for row in windows)
    source = emit_cpp_program(program, model=model)
    stages = re.findall(r"ctx.set_stage_time\((\d+), (\d+)\);", source)
    assert stages == [(str(Fraction(row[1]).numerator), str(Fraction(row[1]).denominator))
                      for row in expected]
    assert source.count("ctx.prepare_provider_values(") == len(expected)


@pytest.mark.parametrize("factory", [Strang, Lie])
@pytest.mark.parametrize("prefix", ["arbitrary_map", "renamed_independent_map"])
def test_explicitly_authored_endpoint_evaluation_is_distinct_from_default_input(factory, prefix):
    entry, entry_model, windows = _authored(factory, prefix=prefix)
    endpoint, endpoint_model, endpoint_windows = _authored(factory, at_endpoint=True, prefix=prefix)
    assert windows == endpoint_windows
    entry_nodes = [node for node in entry._values if node.op == "local_transform"]
    endpoint_nodes = [node for node in endpoint._values if node.op == "local_transform"]
    assert [evaluation_stage_fraction(node) for node in entry_nodes] == [row[1] for row in windows]
    assert [evaluation_stage_fraction(node) for node in endpoint_nodes] == [row[2] for row in windows]
    assert evaluation_stage_fraction(entry_nodes[1]) == 0
    assert evaluation_stage_fraction(endpoint_nodes[1]) == 1
    assert entry_nodes[1].point.time_for("first") != entry_nodes[1].point.time_for("second")
    with pytest.raises(ValueError, match="requires an explicit evaluation partition"):
        from types import SimpleNamespace
        evaluation_stage_fraction(SimpleNamespace(point=entry_nodes[1].point, attrs={}))
    assert emit_cpp_program(entry, model=entry_model) != emit_cpp_program(endpoint, model=endpoint_model)
