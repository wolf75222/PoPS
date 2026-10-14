"""Native face failure must not publish a partial conservative update."""
import numpy as np
import pops
import pytest

from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.time import FixedDt
from examples.api_v02.models_M01_M14 import _domain, _initial
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("failure", ("negative_frequency", "nonfinite_flux"))
def test_stencil_bad_face_rejected_before_commit(failure, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="stencil face refusal")
    frame, layout = _domain("stencil_refusal",(8,8))
    model = pops.Model("closed_scalar_storage",frame=frame)
    state = model.state("state",components=("u",))
    case = pops.Case("stencil_refusal_"+failure); block = case.block("scalar",model)
    plan = DiscretizationPlan(); plan.states.add(state,StateStorage(ghost_depth=1)); case.numerics(plan,block=block)
    def face(sample,spacing,axis):
        q = sample((0,0),0)
        if failure == "nonfinite_flux":
            return (1/q,), 1.
        return (q,), -1.
    flux = StencilFlux(dimension=2,components=1,radius=1,body=face)
    program = pops.Program("reject_before_publish"); u = program.state(block[state])
    rate = program.stencil_flux_divergence("closed_face",u.n,flux=flux)
    program.commit(u.next,program.value("candidate",u.n+program.dt*rate,at=u.next.point))
    program.step_strategy(FixedDt(.01)); case.program(program)
    initial = np.ones((1,8,8)); initial[0,3,4] = 0.
    bindings = _initial(case,block[state],initial)
    resolved = collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact = compile_profile_plan_once(world,resolved,route=failure,compile_artifact=pops.compile)
    runtime = collective_call(world,lambda:pops.bind(artifact,initial_values=bindings,
        resources={"execution_context":artifact_execution_context(artifact)}))
    _, failures = collective_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    with collective_check(world):
        assert all(error and error[0] in {"RuntimeError", "ValueError"}
                   and any(word in error[1] for word in ("stencil", "frequency", "nonfinite"))
                   for error in failures), failures
    after = collective_call(world,lambda:runtime.state_global("scalar"))
    with collective_check(world):
        if after is not None:
            np.testing.assert_array_equal(np.asarray(after).reshape(initial.shape),initial)
        assert runtime.time() == 0 and runtime.macro_step() == 0


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_native_stencil_reads_beyond_an_arbitrary_method_order(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="wide authored stencil")
    frame,layout=_domain("wide_periodic",(32,32))
    model=pops.Model("wide_scalar_storage",frame=frame)
    state=model.state("state",components=("u",))
    case=pops.Case("wide_scalar_faces");block=case.block("scalar",model)
    flux=StencilFlux(dimension=2,components=1,radius=9,
        body=lambda sample,h,axis: ((.2*sample((9,0),0),),1/h[0]) if axis==0 else ((0.,),0.))
    plan=DiscretizationPlan();plan.states.add(state,StateStorage(ghost_depth=max(flux.ghost_extent)))
    case.numerics(plan,block=block)
    p=pops.Program("wide_face_step");q=p.state(block[state])
    rhs=p.stencil_flux_divergence("wide_face",q.n,flux=flux)
    p.commit(q.next,p.value("candidate",q.n+p.dt*rhs,at=q.next.point))
    p.step_strategy(FixedDt(.001));case.program(p)
    x,y=np.meshgrid((np.arange(32)+.5)/32,(np.arange(32)+.5)/32)
    initial=(1+.2*np.cos(2*np.pi*x)+.1*np.sin(2*np.pi*y))[None]
    expected=initial+.001*.2*32*(np.roll(initial,-9,axis=2)-np.roll(initial,-10,axis=2))
    bindings=_initial(case,block[state],initial)
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=compile_profile_plan_once(world,resolved,route="wide stencil",compile_artifact=pops.compile)
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=bindings,
        resources={"execution_context":artifact_execution_context(artifact)}))
    collective_call(world,lambda:pops.run(runtime,t_end=.001,max_steps=1,console=False))
    actual=collective_call(world,lambda:runtime.state_global("scalar"))
    with collective_check(world):
        if actual is not None:
            np.testing.assert_allclose(np.asarray(actual).reshape(initial.shape),expected,rtol=0,atol=2e-14)
        assert runtime.time()==.001 and runtime.macro_step()==1
