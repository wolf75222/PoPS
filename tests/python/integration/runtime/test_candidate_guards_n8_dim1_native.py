"""N8 lot 6, native Dim1: admissibility guards of M15 (spec 0.3 IR4, T38, T39).

M15 writes its explicit Euler step between two admissibility guards, ``admissible_euler`` of
``examples/api_v02/models_M15_orders.py``. Its b6 Program, kept verbatim, is selected by patching
``traced_admissible_euler`` (MG1). Both routes run through ``pops.run``; the b6 invariants pass on
both, the step counts are equal and the final states are equal bitwise.

Through ``Simulation.run``, M15 shows both failure sites. An accepted state outside the domain (one
cell of negative density) fails the guard of ``B.state``, a fixed site: the run raises
``NotAdmissible`` tagged ``trial`` after one attempt, with no rejection counted. A proposal that
takes the candidate out of the domain fails the guard of the candidate, a retryable site: its
rejections spend ``Retry.max_attempts`` and end in ``BudgetExhausted`` tagged ``trial``, caused by
the ``NotAdmissible`` of the last attempt. Both leave the accepted state and the macro step.
M17, Dim2, and the reuse of T7 across attempts are in ``test_candidate_guards_n8_native.py``.
"""
import numpy as np
import pytest

import pops
import examples.api_v02.models_M15_orders as orders
from pops.runtime._simulation import BudgetExhausted, NotAdmissible, Simulation
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

SITE = "admissible_euler/body#%d/domain"


def _b6_route(case, block, state, dt, *, name, b6, **components):
    return b6()


def _array(world, read):
    value = collective_call(world, read)
    return None if value is None else np.asarray(value).copy()


def _run(build, world, route):
    runtime, artifact, rows = execute_case(build, world, route)
    state = _array(world, lambda: runtime.state_global("moments"))
    steps = collective_call(world, runtime.macro_step)
    return state, steps, artifact, rows


def _bitwise(left, right):
    if left is not None or right is not None:
        assert left is not None and right is not None
        assert left.dtype == right.dtype == np.float64
        # Bitwise, not within a tolerance: the same stages and guards in the same order.
        assert left.tobytes() == right.tobytes()


def _parity(build, name, dimension, profile, monkeypatch, record_property):
    _, world = native_profile(dimension=dimension, scope="N8 lot 6 guarded profile %s" % profile)
    traced = _run(build, world, "N8 lot 6 %s traced method" % profile)
    monkeypatch.setattr(orders, "traced_admissible_euler", _b6_route)
    b6 = _run(build, world, "N8 lot 6 %s b6 builder" % profile)
    with collective_check(world):
        assert traced[2].program_name == b6[2].program_name == name
        assert all(row["passed"] for row in b6[3]) and all(row["passed"] for row in traced[3])
        assert traced[1] == b6[1] > 0
        _bitwise(b6[0], traced[0])
    record_property("artifact_identities", (b6[2].artifact_identity.hexdigest,
                                            traced[2].artifact_identity.hexdigest))


def test_m15_runs_bitwise_like_b6(monkeypatch, isolated_native_cache, native_cxx, kokkos_root,
                                  record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _parity(orders.build_m15_orders, "M15_S04_Euler", 1, "M15", monkeypatch, record_property)


# --- M15 through Simulation.run -----------------------------------------------------------------

def _route(attempts, dt=None):
    def route(case, block, state, proposal, *, name, b6, **components):
        del b6
        return elaborate(Advance(orders.admissible_euler(**components),
                                 retry=Retry(max_attempts=attempts)),
                         case=case, state=block[state],
                         proposal=Proposal(proposal if dt is None else dt), name=name)
    return route


def _simulation(world, label, route, *, initial=None, build=orders.build_m15_orders,
                patch_target=(orders, "traced_admissible_euler"), monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(*patch_target, route)
        spec = collective_call(world, build)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="N8 lot 6 " + label,
                                         compile_artifact=pops.compile)
    values = {subject: (value if initial is None else initial(value)).copy()
              for subject, value in spec.initial_values.items()}
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=values,
        resources={"execution_context": artifact_execution_context(artifact)}))
    return spec, runtime, Simulation(runtime)


def _failed(world, runtime, simulation, until, errors):
    start = _array(world, lambda: runtime.state_global("moments"))

    def failed():
        try:
            simulation.run(until)
        except errors as error:
            return error
        raise AssertionError("the run did not fail")
    error = collective_call(world, failed)
    after = _array(world, lambda: runtime.state_global("moments"))
    with collective_check(world):
        # The failure leaves the accepted state and the macro step of the step.
        _bitwise(start, after)
        assert runtime.macro_step() == 0
    return error


def _outside(value):
    """One cell of negative density: the accepted state is outside the strict Hankel domain."""
    value = value.copy()
    value[0, 3] = -value[0, 3]
    return value


def test_an_inadmissible_state_fails_the_fixed_guard_of_b_state(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=1, scope="N8 lot 6 fixed guard")
    for attempts in (1, 3):
        spec, runtime, simulation = _simulation(
            world, "M15 inadmissible state Retry(%d)" % attempts, _route(attempts),
            initial=_outside, monkeypatch=monkeypatch)
        error = _failed(world, runtime, simulation, spec.t_end, (NotAdmissible,))
        failure = error.result.failure
        assert type(error) is NotAdmissible and failure.variant is NotAdmissible
        assert failure.origin == "trial" and failure.payload["site"] == SITE % 0
        assert failure.payload["attempts"] == 1 and error.result.rejections == 0
        assert error.result.steps == 0


def test_a_candidate_outside_the_domain_spends_the_attempts(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=1, scope="N8 lot 6 retryable guard")
    # dt = 2: forward Euler takes the smooth order-4 data out of the strict Hankel domain
    # (minimum Hankel determinant -0.14 by the numpy reference of the profile).
    for attempts in (1, 3):
        _, runtime, simulation = _simulation(
            world, "M15 large dt Retry(%d)" % attempts, _route(attempts, dt=2.0),
            monkeypatch=monkeypatch)
        error = _failed(world, runtime, simulation, 2.0, (BudgetExhausted,))
        failure = error.result.failure
        assert type(error) is BudgetExhausted and failure.origin == "trial"
        assert failure.payload["attempts"] == attempts == error.result.rejections
        cause = failure.payload["cause"]
        assert cause.variant is NotAdmissible and cause.origin == "trial"
        assert cause.payload["site"] == SITE % 1
