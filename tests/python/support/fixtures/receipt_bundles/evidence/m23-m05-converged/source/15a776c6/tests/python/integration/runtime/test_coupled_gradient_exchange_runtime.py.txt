"""Independent native matrix/ledger witness and finite-input failure injections."""
import json
import re

import numpy as np
import pops
import pytest

from pops import math
from pops._native_collectives import allgather_value
from pops._native_selector import select_native_dimension
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check, state_snapshots,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
CELLS, DT = 16, 1.e-5
WEIGHTS = (.5, 1.5)
VECTOR = np.array((1., 2., -1.))
D = np.outer(VECTOR, VECTOR)
R = np.array(((0., -.3, .2), (.3, 0., -.1), (-.2, .1, 0.)))


def _case(order):
    frame = CartesianDomain("periodic", lower=(0.,), upper=(1.,)).frame(Cartesian1D())
    model = pops.Model("independent_matrix", frame=frame)
    labels = ("north", "east", "third")
    state = model.state("u", components=tuple(labels[i] for i in order))
    def matrix(values):
        return tuple(tuple(float(values[i,j]) for j in order) for i in order)
    flux = model.coupled_gradient_flux("law", state=state,
        dissipative=matrix(D), reversible=matrix(R))
    rate = model.rate("evolution", equation=math.ddt(state) ==
                      WEIGHTS[0]*math.div(flux)+WEIGHTS[1]*math.div(flux))
    case = pops.Case("independent_coupled_matrix")
    block = case.block("mixture", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, CoupledGradient(flux=flux))
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS,),
                                  periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout), block[state]


def _initial():
    x = (np.arange(CELLS)+.5)/CELLS
    return np.stack((1+.1*np.cos(2*np.pi*x), .4*np.sin(2*np.pi*x),
                     -.2+.15*np.cos(4*np.pi*x)))


def _rate(values):
    laplacian = (np.roll(values,1,axis=1)-2*values+np.roll(values,-1,axis=1))*CELLS**2
    return sum(WEIGHTS)*(D+R)@laplacian


def _stage_states(contexts, initial, predictor):
    """Use the retained IR stage identity, independently of ledger ordering."""
    result = {}
    seen = set()
    assert len(contexts) == 2
    for context in contexts:
        stages = re.findall(r"StagePoint\(name='ssprk2_stage_([01])',", context)
        assert len(stages) == 1, context
        stage = int(stages[0])
        assert stage not in seen, contexts
        seen.add(stage)
        result[context] = (initial, predictor)[stage]
    assert seen == {0, 1}
    return result


@pytest.mark.parametrize("order", ((0,1,2), (2,0,1)))
def test_matrix_flux_occurrences_and_rollback(isolated_native_cache, native_cxx,
        kokkos_root, tmp_path, record_property, order):
    del isolated_native_cache, native_cxx, kokkos_root
    native = select_native_dimension(1)
    world = native.mpi_world()
    resolved, subject = collective_call(world, lambda: _case(order))
    artifact = compile_resolved_plan_once(world, resolved,
        route="independent coupled matrix", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))

    def bind(values):
        # Finish local allocations/conversions before a peer may enter bind.
        prepared = collective_call(world, lambda: np.ascontiguousarray(values[list(order)]))
        return collective_call(world, lambda: pops.bind(artifact,
            initial_values={subject: prepared},
            resources={"execution_context": context}))

    initial = collective_call(world, _initial)
    runtime = bind(initial)
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    actual = state_snapshots(runtime, world, ("mixture",))[0]
    local = collective_call(world, runtime._executor._program_exchange_records)
    batches = allgather_value(world,local)
    with collective_check(world):
        records = [row for batch in batches for row in batch]
        assert runtime.time() == DT and runtime.macro_step() == 1
        assert len(records) == 3*CELLS*2*2*2
        if world.size == 2:
            assert sorted(map(len, batches)) == [0, len(records)]
        if world.rank == 0:
            predictor = initial+DT*_rate(initial)
            expected = .5*(initial+predictor+DT*_rate(predictor))
            actual = actual.reshape(3,CELLS)[np.argsort(order)]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=3.e-12)
            contexts = tuple(dict.fromkeys(row['evaluation_context'] for row in records))
            stages = _stage_states(contexts, initial, predictor)
            increments = np.zeros_like(initial)
            identities = set()
            assert len({row['operation_identity'] for row in records}) == 1
            for row in records:
                tokens = row['quadrature_identity'].split('/')
                cell, axis, side, component = (int(token.split(':')[1]) for token in tokens)
                occurrence = int(row['occurrence_identity'].rsplit(':',1)[1])
                key = (row['occurrence_identity'],row['evaluation_context'],row['quadrature_identity'])
                assert key not in identities
                identities.add(key)
                assert axis == 0 and occurrence in (0,1)
                assert 0 <= cell < CELLS and side in (0,1) and 0 <= component < 3
                assert row['occurrence_identity'] == row['operation_identity'] + '/occurrence:' + str(occurrence)
                assert row['multiplicity'] == 1 and row['face_measure'] == 1
                assert row['orientation'] == (-1 if side == 0 else 1)
                assert row['temporal_weight'] == DT/2*WEIGHTS[occurrence]
                physical_component = order[component]
                w = (D+R)@stages[row['evaluation_context']]
                left, right = ((cell,(cell+1)%CELLS) if side else ((cell-1)%CELLS,cell))
                face = CELLS*(w[physical_component,right]-w[physical_component,left])
                assert abs(row['numerical_flux']-face) < 3.e-12
                increments[physical_component,cell] += row['integrated_amount']
            np.testing.assert_allclose(increments,(actual-initial)/CELLS,rtol=0,atol=3.e-13)
            np.testing.assert_allclose(increments.sum(axis=1),0,rtol=0,atol=3.e-13)
            np.savez_compressed(tmp_path/'accepted_matrix.npz',initial=initial,final=actual,
                                increment=increments,order=order,dt=DT)
            (tmp_path/'accepted_ledger.json').write_text(json.dumps(records,indent=2)+'\n')

    # Only finite inputs are bound. A single large cell makes the component law
    # overflow during evaluation; no nonfinite initial value bypasses that route.
    with collective_check(world):
        bad = initial.copy()
        bad[:,CELLS//3] = 1.e308
    failed = bind(bad)
    before = state_snapshots(failed,world,("mixture",))[0]
    ledger_before = collective_call(world,failed._executor._program_exchange_records)
    for attempt in range(2):
        _, failures = collective_attempt(world,lambda:pops.run(
            failed,t_end=DT,max_steps=1,console=False))
        with collective_check(world):
            assert all(row is not None and row[2] for row in failures), failures
            assert all('invalid_evaluation' in row[1] for row in failures), failures
            assert failed.time() == 0 and failed.macro_step() == 0
            record_property('finite_input_refusal_%d'%attempt,repr(failures))
        ledger_after = collective_call(world,failed._executor._program_exchange_records)
        after = state_snapshots(failed,world,("mixture",))[0]
        with collective_check(world):
            assert ledger_after == ledger_before
            if world.rank == 0:
                np.testing.assert_array_equal(after,before)
