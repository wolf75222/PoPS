"""N8 lot « Règle de pas compilée », native Dim2 (design note revision 1.1, section 6).

The slice's physics written by hand for b6 and N8: master's tutorial 02, ``ddt(u) == -div((1,
0.25) u)`` on 64 x 64 cells, MUSCL(VanLeer) with Rusanov (the upwind of this linear flux), inflow
at the far field on x_min and y_min, outflow on x_max and y_max, SSPRK2. The Advance is elaborated
from ``StepRule(0.45, max_dt, strategy=AdaptiveCFL(0.45, max_dt))``, as the slice's
``cfl_limit`` builds it: ``Simulation.run`` takes each step's proposal from b6's CFL selection in
``System::run_advance``, and ``pops.run`` on the same artifact takes master's ``AdaptiveCFL``
route through ``System::step_cfl``. Both reach the same steps and the same state
bit for bit. A mutant that drops ``max_dt`` is caught; a rejected attempt is retried with the dt
that master selects again on the restored state.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

import pops
from pops import math as pm
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import BudgetExhausted, Simulation
from pops.time import AdaptiveCFL
from pops.time._program.elaboration import Advance, Proposal, Retry, StepRule, elaborate
from pops.time._program.outputs import Output, Product
from pops.time._program.traced_method import Candidate, method
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.one_shot_rejection import one_shot_rejection

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CELLS = (64, 64)
AX, AY = 1.0, 0.25
FAR_FIELD = 0.05
CFL, MAX_DT, T_END = 0.45, 1.0e-2, 0.20
BINDING = 0.005                      # a max_dt below the CFL step, about 0.0070 here
FRONTIERS = (0.05, 0.1, T_END)


@method
def _ssprk2(B, p, rate):
    u, t, h = B.state, B.time, p.dt
    k0 = rate(u, t)
    k1 = rate(u + h * k0, t + h)
    return Candidate(u + h * Fraction(1, 2) * k0 + h * Fraction(1, 2) * k1, t + h)


def _snapshot(B, interval):
    return Product(u=B.state)


def _case(rule, *, attempts=1, outputs=()):
    frame = Rectangle("unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("scalar_advection", frame=frame)
    u = model.state("u", components=("u",))
    flux = model.flux("advection", state=u, frame=frame,
                      components={x: (AX * u[0],), y: (AY * u[0],)}, waves={x: (AX,), y: (AY,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux))
    plan = DiscretizationPlan()
    selection = rate.select(flux)
    plan.rates.add(selection, FiniteVolume(
        flux=flux, variables=variables.Conservative(u),
        reconstruction=reconstruction.MUSCL(reconstruction.limiters.VanLeer()),
        riemann=riemann.Rusanov()))
    case = pops.Case("scalar_advection")
    block = case.block("tracer", model)
    boundaries = frame.boundaries
    plan.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=block[u], value=FAR_FIELD),
        boundaries.x_max: Outflow(state=block[u]),
        boundaries.y_min: Inflow(state=block[u], value=FAR_FIELD),
        boundaries.y_max: Outflow(state=block[u]),
    }))
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_ssprk2(rate=lambda state: selection(state)),
                      retry=Retry(max_attempts=attempts)),
              case=case, state=block[u], proposal=rule, outputs=outputs)
    return case, Uniform(CartesianGrid(frame=frame, cells=CELLS))


def _compile(rule, monkeypatch=None, **options):
    case, layout = _case(rule, **options)
    plan = pops.resolve(pops.validate(case), layout=layout)
    if monkeypatch is None:
        return pops.compile(plan)
    with one_shot_rejection(monkeypatch, "N8 compiled step rule one-shot rejection"):
        return pops.compile(plan)


def _initial():
    x = (np.arange(CELLS[0], dtype=np.float64) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1], dtype=np.float64) + 0.5) / CELLS[1]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    bump = FAR_FIELD + 0.95 * np.exp(-120.0 * ((xx - 0.30)**2 + (yy - 0.35)**2))
    return np.ascontiguousarray(bump[np.newaxis, :, :])


def _bind(artifact):
    (binding,) = tuple(artifact.plan.initial_condition_plan.bindings)
    return pops.bind(artifact, initial_values={binding.subject: _initial()},
                     resources={"execution_context": artifact_execution_context(artifact)})


def _state(runtime):
    return np.asarray(runtime.state_global("tracer")).copy()


def _bitwise(left, right):
    assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
    # Bitwise, not within a tolerance: the same proposals, attempts and stages.
    assert left.tobytes() == right.tobytes()


def _master(artifact, frontiers):
    """master's AdaptiveCFL through pops.run, to each frontier in turn; per frontier the macro
    step and the state, and whether the run published the frontier itself."""
    runtime = _bind(artifact)
    rows = []
    for frontier in frontiers:
        pops.run(runtime, t_end=frontier, max_steps=10_000, console=False)
        # No sliver at a frontier (review P4): master's capped step lands on it exactly.
        assert runtime.time() == frontier
        rows.append((runtime.macro_step(), _state(runtime)))
    return rows


def _rule(artifact, frontiers):
    runtime = _bind(artifact)
    simulation = Simulation(runtime)
    rows = []
    for frontier in frontiers:
        result = simulation.run(frontier)
        assert result.failure is None and result.time == frontier
        rows.append((runtime.macro_step(), _state(runtime)))
    return rows, runtime


def _parity(rule_rows, master_rows):
    assert [steps for steps, _ in rule_rows] == [steps for steps, _ in master_rows]
    for (_, left), (_, right) in zip(rule_rows, master_rows, strict=True):
        _bitwise(left, right)


def _slice_rule(max_dt):
    """The rule as the slice's library word gives it: the fraction and the builder's strategy."""
    return StepRule(CFL, max_dt, strategy=AdaptiveCFL(CFL, max_dt))


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        yield {
            "slice": _compile(_slice_rule(MAX_DT)),
            "binding": _compile(_slice_rule(BINDING)),
            "retry": _compile(_slice_rule(MAX_DT), attempts=2),
            "rejected": _compile(_slice_rule(MAX_DT), patch, attempts=2),
            "outputs": _compile(_slice_rule(MAX_DT),
                                outputs=(Output(_snapshot, at=FRONTIERS[:2]),)),
        }


def test_the_rule_runs_the_steps_of_master_adaptive_cfl(artifacts):
    """The slice's rule, CFL = 0.45 and max_dt = 1e-2 (which does not bind): to T_END in one run,
    and to each frontier in turn, the steps and the states are master's, bit for bit."""
    artifact = artifacts["slice"]
    control = artifact.program.program._advance_control
    assert control.step_rule == (CFL, MAX_DT) and control.first_dt is None
    assert control.to_data()["schema"] == "pops.advance-control@4"
    rule_rows, runtime = _rule(artifact, (T_END,))
    master_rows = _master(artifact, (T_END,))
    _parity(rule_rows, master_rows)
    assert rule_rows[0][0] > 20
    # Review P1: the step that lands on the frontier names it; master writes strategy:max_dt.
    assert runtime._executor._s.last_dt_bound() == "frontier"
    _parity(_rule(artifact, FRONTIERS)[0], _master(artifact, FRONTIERS))


def test_an_output_time_is_a_frontier_of_the_rule(artifacts):
    """An Output at 0.05 and 0.1: the rule's run lands on both inside one run and publishes the
    state that master reaches with a run to each."""
    runtime = _bind(artifacts["outputs"])
    simulation = Simulation(runtime)
    (output,) = simulation._outputs
    result = simulation.run(T_END)
    rows = result.outputs[output.output]
    assert [row.time for row in rows] == [0.0, 0.05, 0.1]
    master_rows = _master(artifacts["slice"], FRONTIERS)
    assert [row.macro_step for row in rows[1:]] == [steps for steps, _ in master_rows[:2]]
    for row, (_, state) in zip(rows[1:], master_rows[:2], strict=True):
        _bitwise(np.asarray(row["u"]).reshape(state.shape), state)
    _bitwise(_state(runtime), master_rows[2][1])


def test_max_dt_binds_and_a_mutant_that_drops_it_is_caught(artifacts):
    """max_dt = 0.005, below the CFL step: every step but none is longer, and parity holds. The
    mutant passes max_dt = +inf to the native run: fewer steps (P6) and another state."""
    artifact = artifacts["binding"]
    rule_rows, _ = _rule(artifact, (T_END,))
    master_rows = _master(artifact, (T_END,))
    _parity(rule_rows, master_rows)
    assert rule_rows[0][0] == 40  # T_END / max_dt
    runtime = _bind(artifact)
    simulation = Simulation(runtime)
    native = simulation._run_native

    def mutant(until, proposal, max_steps, schedule, rule):
        return native(until, proposal, max_steps, schedule, (rule[0], math.inf))
    simulation._run_native = mutant
    simulation.run(T_END)
    assert runtime.macro_step() < rule_rows[0][0]
    with pytest.raises(AssertionError):
        _parity([(runtime.macro_step(), _state(runtime))], master_rows)


def test_a_rejected_attempt_is_retried_with_master_dt(artifacts, monkeypatch):
    """Retry(2), the first candidate rejected once (the one-shot patch of « Réutilisation entre
    stades »): the retry runs at the dt that master selects again on the restored state (P2), and
    the run ends bit for bit where the unpatched one does."""
    del monkeypatch
    patched = _bind(artifacts["rejected"])
    simulation = Simulation(patched)
    with pytest.raises(BudgetExhausted) as first:
        simulation.run(T_END, max_steps=1)
    assert first.value.result.steps == 1 and first.value.result.rejections == 1
    reselected = _bind(artifacts["retry"])
    dt = reselected._executor._s.step_cfl(CFL, max_dt=min(MAX_DT, T_END))
    assert patched.time() == dt == reselected.time()
    _bitwise(_state(patched), _state(reselected))
    result = simulation.run(T_END)
    assert result.rejections == 0
    plain_rows, _ = _rule(artifacts["retry"], (T_END,))
    assert patched.macro_step() == plain_rows[0][0]
    _bitwise(_state(patched), plain_rows[0][1])


def test_the_rule_refuses_a_proposal_and_a_global_bound(artifacts):
    runtime = _bind(artifacts["slice"])
    simulation = Simulation(runtime)
    with pytest.raises(TypeError, match="the step rule of this Advance chooses every proposal"):
        simulation.run(T_END, proposal=Proposal(0.001))
    # Review P2: a global bound of add_dt_bound is an external closure. A bound runtime refuses
    # one, so none reaches a run of the rule; run_advance also refuses it (defence in depth).
    with pytest.raises(RuntimeError, match="composition is frozen once pops.bind completes"):
        runtime._executor._s.add_dt_bound("probe", lambda: 1.0)
    assert runtime._executor._s.last_dt_bound() == ""
    simulation.run(0.05)
    assert runtime.time() == 0.05
