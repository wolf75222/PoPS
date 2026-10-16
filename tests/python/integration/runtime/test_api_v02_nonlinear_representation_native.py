"""Real compiled Q obligations: physical values and rejection-before-publication."""
import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.nonlinear_representation import build, physical_step


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("capacity", [64 << 20, 1], ids=["executed-equation", "workspace-atomic-rejection"])
def test_exact_nonlinear_representation(isolated_native_cache, record_property, capacity):
    del isolated_native_cache
    native, world = native_profile(scope="executed nonlinear Field representation")
    case, layout, initial, dt = collective_call(world, lambda: build(workspace_byte_capacity=capacity))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="nonlinear representation",
                                         compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values={subject: initial}, resources={"execution_context": context}))
    before = collective_call(world, lambda: np.asarray(runtime.state_global("material")).copy())
    rank = 0 if world is None else int(world.rank)
    with collective_check(world):
        if rank == 0:
            generated = "\n".join(row.program._generated_cpp or "" for row in artifact.layout_programs)
            assert "ctx.own_field_representation(" in generated
            assert "ctx.own_discharged_field_representation(" in generated
            assert "CapturedTransformReads<" in generated
    if capacity == 1:
        _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=dt, console=False))
        after = collective_call(world, lambda: np.asarray(runtime.state_global("material")).copy())
        with collective_check(world):
            assert errors and all(errors), errors
            assert runtime.time() == 0 and runtime.macro_step() == 0
            np.testing.assert_array_equal(after, before)
        record_property("rejections", str(errors))
    else:
        collective_call(world, lambda: pops.run(runtime, t_end=2*dt, max_steps=2, console=False))
        actual = collective_call(world, lambda: np.asarray(runtime.state_global("material")).copy())
        expected = physical_step(physical_step(initial, dt), dt)
        with collective_check(world):
            assert runtime.time() == 2*dt and runtime.macro_step() == 2
            if rank == 0:
                np.testing.assert_allclose(actual.reshape(initial.shape), expected, rtol=0, atol=2.e-10)
        record_property("equation", "Q(q)-U_n-dt*A*Q(q)=0")
    record_property("native_path", str(native.__file__))
