"""Independent source reception; no author helper, native/JIT run or numerical substitute."""
import pops
import pytest

from pops._ir.quantity import PhysicalDimension
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import FixedDt
from pops.time.points import TimePoint

DIMENSIONLESS = PhysicalDimension()


def authored(*, units=DIMENSIONLESS, initial=.73, gamma=.41, typed=True, rate_first=False):
    frame = Rectangle("independent-capture-domain", (0., 0.), (2., 1.)).frame(Cartesian2D())
    model = pops.Model("independent-source", frame=frame)
    state = model.state("vector", components=("first", "second"))
    source = model.source("retained_original", on=state, value=(-gamma*state[0], -gamma*state[1]))
    balance = model.rate("original_balance", equation=ddt(state) == source)
    selected = balance.select(source)
    case = pops.Case("independent-typed-capture")
    block = case.block("cells", model)
    numerical = DiscretizationPlan()
    numerical.rates.add(selected, StateStorage())
    case.numerics(numerical, block=block)
    program = pops.Program("independent-capture-program")
    current = program.state(block[state])
    integral = (program.integral_state("budget", initial=initial, units=units) if typed else
                program.integral_state("budget", initial=initial))
    capture = program.integral_value(integral, at=current.n.point, scope="candidate") if typed else 1
    rhs = selected(current.n)
    expressions = tuple((program.dt*capture*rhs[i] + current.n[i] if rate_first else
                         current.n[i] + program.dt*capture*rhs[i]) for i in range(2))
    candidate = program.value("native-feedback", expressions, at=current.next.point)
    program.commit(current.next, candidate)
    program.step_strategy(FixedDt(.02))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(7, 3), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, integral, current, capture


def emitted(args):
    case, layout, *_ = args
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_true_pod_consumption_precedes_real_cell_kernel_and_keeps_original_source():
    args = authored()
    program = args[2]
    code = emitted(args)
    capture = next(value for value in program._values if value.op == "integral_candidate")
    assert capture.vtype == "scalar" and not capture.is_field() and not capture.inputs
    assert len([value for value in program._values if value.op == "source"]) == 1
    assert code.count("ctx.capture_integral_candidate(") == 1
    assert code.count("ctx.integral_candidate_value(") == 1
    consumed = code.index("ctx.integral_candidate_value(")
    assert code.index("pops::for_each_cell", consumed) > consumed
    assert "global_value_%d" % capture.id in code
    assert "integral_capture_%dA" % capture.id not in code
    assert args[3].identity.startswith("pops.integral.v2/")


@pytest.mark.parametrize("mutation", ({"units": "not-json"}, {"scope": "accepted"},
                                       {"capture_version": 2}, {"integral": "foreign-owner-budget"}))
def test_metadata_mutations_fail_closed_before_emission(mutation):
    args = authored()
    program = args[2]
    capture = next(value for value in program._values if value.op == "integral_candidate")
    program._replace_value(capture, attrs={**capture.attrs, **mutation})
    with pytest.raises(ValueError, match="metadata changed"):
        emitted(args)


def test_owner_point_and_scope_refusals_are_atomic():
    _, _, program, integral, current, capture = authored()
    before = (program._next_id, len(program._values))
    with pytest.raises(ValueError, match="exact IntegralState"):
        pops.Program("foreign-program").integral_value(integral, at=current.n.point, scope="candidate")
    with pytest.raises(ValueError, match="candidate scope"):
        program.integral_value(integral, at=current.n.point, scope="accepted")
    assert (program._next_id, len(program._values)) == before
    later = program.integral_value(integral, at=current.next.point, scope="candidate")
    before = (program._next_id, len(program._values))
    with pytest.raises(ValueError, match="same exact point"):
        program.value("wrong-time", (later*current.n[0], later*current.n[1]), at=current.n.point)
    assert (program._next_id, len(program._values)) == before
    with pytest.raises(TypeError, match="direct Equation/FieldProblem"):
        capture.to_cpp()


def test_units_gamma_and_initial_condition_change_genuine_identities():
    first = authored()[2]._ir_hash()
    assert first != authored(gamma=.42)[2]._ir_hash()
    assert first != authored(initial=.74)[2]._ir_hash()
    assert first != authored(units=PhysicalDimension((("charge", 1),)))[2]._ir_hash()


def test_program_initial_overflow_is_not_masked_or_published():
    program = pops.Program("overflow-counterexample")
    with pytest.raises(OverflowError):
        program.integral_state("huge", initial=10**1000, units=PhysicalDimension())
    assert not program._integral_states and not program._integral_units


def test_unknown_units_cannot_be_upgraded_by_capture():
    program = pops.Program("untyped-legacy")
    quantity = program.integral_state("budget", initial=.7)
    before = (program._next_id, len(program._values))
    with pytest.raises(ValueError, match="physical units"):
        program.integral_value(quantity, at=TimePoint(program.clock), scope="candidate")
    assert (program._next_id, len(program._values)) == before
    assert "integral_units_v2" not in program._serialize()


def test_legacy_rate_first_component_expression_keeps_typed_state_template():
    # Intentionally red on c064 and its parent; root's integrated State priority supports it.
    _, _, program, _, current, _ = authored()
    rhs = next(value for value in program._values if value.op == "source")
    value = program.value("legacy_rate_first", tuple(program.dt*rhs[i] + current.n[i]
        for i in range(2)), at=current.next.point)
    assert value.vtype == "state" and value.space == current.n.space


def test_global_rate_first_component_expression_keeps_typed_state_template():
    # New composition must also ignore the global scalar as a source of cell support.
    _, _, program, _, current, capture = authored()
    rhs = next(value for value in program._values if value.op == "source")
    value = program.value("global_rate_first", tuple(program.dt*capture*rhs[i] + current.n[i]
        for i in range(2)), at=current.next.point)
    assert value.vtype == "state" and value.space == current.n.space
