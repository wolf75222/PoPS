"""Full accepted-step witness for a shared imported W/A law in native diffusive faces."""
from dataclasses import replace
from pathlib import Path
import ctypes
import time

import numpy as np
import pops
import pytest
from pops import math
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.model import Signature
from pops.numerics import Diffusion, DiscretizationPlan
from test_native_constitutive import constitutive_function
from interaction_test_layout import interaction_grid
from tests.python.support.native_execution_context import artifact_execution_context


def constitutive_case(directory, *, n):
    grid = interaction_grid(n=n)
    model = pops.Model("joint_constitutive", frame=grid.frame)
    state = model.state("U", components=("q",))
    function = constitutive_function(directory, counters=True)
    function = replace(function, signature=Signature((state.space,), function.signature.output))
    call = function((state[0],), occurrence="constitutive-endpoint")
    flux = model.diffusive_flux("conduction", state=state,
        value=math.CoeffGradient(call.transform[0], tuple(call.diffusivity)))
    rate = model.rate("rate", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("native_constitutive_faces")
    block = case.block("heat", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(pops.time.FixedDt(1e-6))
    case.program(program)
    return pops.resolve(pops.validate(case), layout=Uniform(grid))


def _initial(n):
    coordinate = (np.arange(n) + .5) / n
    return 2 + .1 * np.sin(2*np.pi*coordinate[:, None]) * np.sin(2*np.pi*coordinate[None, :])


def _rhs(q):
    """Independent arithmetic face interpolation and conservative periodic differences."""
    w = q + q*q
    result = np.zeros_like(q)
    # Native x is NumPy's final axis; the public ranked state is (component,y,x).
    for axis, offset in ((1, 1.), (0, 2.)):
        a = offset + q*q
        flux_right = .5 * (a + np.roll(a, -1, axis)) * (np.roll(w, -1, axis) - w) * q.shape[axis]
        result += (flux_right - np.roll(flux_right, 1, axis)) * q.shape[axis]
    return result


def _expected(n):
    q = _initial(n)
    for _ in range(100):
        first = _rhs(q)
        q = q + .5e-6 * (first + _rhs(q + 1e-6*first))
    return q


@pytest.fixture(scope="module", params=(16, 32, 64))
def compiled(request, tmp_path_factory):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    n = request.param
    resolved = constitutive_case(tmp_path_factory.mktemp("constitutive") / "external", n=n)
    source = emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved))
    assert source.count("constitutive::evaluate(") == 2
    assert source.count("constitutive::jacobian(") == 2
    started = time.perf_counter()
    artifact = pops.compile(resolved)
    native = ctypes.CDLL(str(artifact.program.so_path))
    native.constitutive_calls.restype = native.constitutive_jacobians.restype = ctypes.c_long
    return n, artifact, native, time.perf_counter()-started, len(source.encode())


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_full_native_joint_constitutive_face_matrix(compiled, record_property):
    n, artifact, native, elapsed, source_bytes = compiled
    simulation = pops.bind(artifact, initial_state={"heat": _initial(n)[None]},
        resources={"execution_context": artifact_execution_context(artifact)})
    cells = sum(int(np.prod(np.asarray(hi)-lo)) for lo, hi in simulation.local_boxes("heat"))
    native.constitutive_reset()
    # Match the native clock's 100 additions, not Python's compensated sum.
    end = 0.
    for _ in range(100):
        end += 1e-6
    report = pops.run(simulation, t_end=end, max_steps=100)
    assert report.accepted_steps == 100 and report.rejected_steps == 0
    np.testing.assert_allclose(simulation.state_global("heat")[0], _expected(n), rtol=0, atol=1e-11)
    assert abs(simulation.integral("heat", 0)-2) < 1e-11
    assert native.constitutive_calls() == native.constitutive_jacobians() == 200*cells
    rows = simulation._executor_for_block("heat")._program_exchange_records()
    assert rows and all(np.isfinite(row["integrated_amount"]) for row in rows)
    record_property("n", n)
    record_property("accepted_steps", report.accepted_steps)
    record_property("actual_primal_calls_local", native.constitutive_calls())
    record_property("actual_exact_jacobian_calls_local", native.constitutive_jacobians())
    record_property("constitutive_sampling", "one call per active cell per stage; materialized W/A shared by adjacent faces")
    record_property("compile_seconds", elapsed)
    record_property("program_source_bytes", source_bytes)
    record_property("program_binary_bytes", Path(artifact.program.so_path).stat().st_size)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_failed_constitutive_endpoint_cannot_publish(compiled):
    n, artifact, native, _elapsed, _source_bytes = compiled
    state = _initial(n)[None]
    state[0, -1, -1] = -1
    simulation = pops.bind(artifact, initial_state={"heat": state},
        resources={"execution_context": artifact_execution_context(artifact)})
    native.constitutive_reset()
    from pops._bootstrap import StepAttemptRejected
    with pytest.raises(StepAttemptRejected):
        pops.run(simulation, t_end=1e-6, max_steps=1)
    np.testing.assert_array_equal(simulation.state_global("heat"), state)
    assert simulation._executor_for_block("heat")._program_exchange_records() == []
