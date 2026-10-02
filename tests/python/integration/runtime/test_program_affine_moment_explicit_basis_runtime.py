"""Installed Native explicit basis/binding, degree5; single-level synchronous AMR.

No full HyQMOM physics/hyperbolicity, multilevel, GPU or performance qualification.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import CartesianMonomialBasis
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.params import RuntimeParam
from pops.solvers import DenseLU
from pops.time import FailRun, FixedDt, LocalLinear, every
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.atomic_native_capture import save_phase,execute_captured_step
from tests.python.support.m16_explicit_native_capture import retain_provenance,persisted_capture

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
SCHEMA='pops.program-affine-explicit-native-fixture@2'
DEGREE,N,DT,OMEGA=5,4,.125,16./3.
INDICES=tuple((p,q) for q in range(DEGREE+1) for p in range(DEGREE+1-q))
# Independent permutations of basis indices and actual State slot names.
BASIS=CartesianMonomialBasis(INDICES[::2]+INDICES[1::2])
NAMES=tuple('entry_'+str(i) for i in range(len(INDICES)))
BINDING=dict(zip(INDICES,NAMES[::-1],strict=True))
ATOMS=((Fraction(-1,4),Fraction(1,2),Fraction(1,2)),
       (Fraction(1,2),Fraction(-1,4),Fraction(1,2)),
       (Fraction(3,4),Fraction(1,4),Fraction(1)))


def atom_moments(step=0):
    # a=Omega*DT/2 is the declared rational 1/3, hence c=4/5,s=3/5.
    c,s=Fraction(4,5),Fraction(3,5)
    atoms=ATOMS
    for _ in range(step):
        atoms=tuple((c*x+s*y,-s*x+c*y,w) for x,y,w in atoms)
    by_index={index:float(sum(w*x**index[0]*y**index[1] for x,y,w in atoms)) for index in INDICES}
    return np.array([by_index[index] for name in NAMES for index in INDICES if BINDING[index]==name])


def build(mode):
    if mode not in ('success','nonfinite','overflow'): raise ValueError('unknown fixture mode')
    frame=Rectangle('monomial_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('explicit_polynomial',frame=frame)
    state=model.state('unlabelled_population',components=NAMES)
    flux=model.flux('zero_transport',state=state,frame=frame,
                    components={axis:tuple(0*v for v in state) for axis in frame.axes},
                    waves={axis:(0.,)*len(NAMES) for axis in frame.axes})
    rate=model.rate('stationary',equation=ddt(state)==-div(flux))
    matrix=[[0.]*len(NAMES) for _ in NAMES]
    x,y=NAMES.index(BINDING[(1,0)]),NAMES.index(BINDING[(0,1)])
    matrix[x][y],matrix[y][x]=OMEGA,-OMEGA
    operator=model.operator('skew',returns=model.local_linear_operator('skew',on=state,matrix=tuple(map(tuple,matrix))))
    plan=DiscretizationPlan(); plan.rates.add(rate,FiniteVolume(flux=flux,
        variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    case=pops.Case('explicit_affine_case'); block=case.block('population',model=model); case.numerics(plan,block=block)
    program=pops.Program('explicit_affine'); q=program.state(block[state])
    midpoint=program.solve(LocalLinear(operator=program.I-(program.dt/2)*program.linear_source(operator),rhs=q.n),
                           solver=DenseLU()).consume(action=FailRun())
    mean=program.value('endpoint',2*midpoint-q.n,at=q.n.point)
    if mode=='nonfinite':
        # Initial values remain finite; the real compiled expression overflows density.
        mean=program.value('nonfinite_endpoint',1.e308*mean,at=q.n.point)
    pushed=program.affine_moment_update(q.n,mean,linear_operator=operator,theta_dt=program.dt/2,
                                      basis=BASIS,components=BINDING)
    program.commit(q.next,program.value('accepted',pushed+program.dt*rate(pushed),at=q.next.point))
    program.step_strategy(FixedDt(DT)); case.program(program)
    case.initials.add(InitialCondition(state=block[state],value=BindArray(),projection=ConservativeCellAverage()))
    transfer=AMRTransfer(); transfer.state(block[state],StateTransfer())
    threshold=case.param(RuntimeParam('unused_refinement_threshold',default=1000.))
    layout=AMR(grid=CartesianGrid(frame=frame,cells=(N,N),periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=1,ratios=()),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])[BINDING[(0,0)]]>case.value(threshold)),Buffer(cells=0)),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(100,clock=program.clock)),transfer=transfer,execution=AMRExecution.synchronous())
    return case,layout


def initial(mode):
    moments=atom_moments()
    if mode=='overflow':
        # Every input is finite. Highest-degree raw moments make the polynomial
        # expansion overflow: (c+s)^5 * max/4 > max, unlike nonfinite endpoint.
        for index in INDICES:
            if sum(index)==DEGREE: moments[NAMES.index(BINDING[index])]=np.finfo(np.float64).max/4
    return np.ascontiguousarray(np.broadcast_to(moments[:,None,None],(len(NAMES),N,N)))


def capture(world,runtime):
    values=collective_call(world,lambda:np.array(runtime.block_level_state_global('population',0),copy=True))
    blob=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
    clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),runtime.n_levels()))
    return values,blob,clock


@pytest.mark.parametrize('mode',('success','nonfinite','overflow'))
def test_installed_program_explicit_monomial_binding(mode,tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'installed package required'
    case,layout=collective_call(world,lambda:build(mode))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else
              compile_resolved_plan_once(world,resolved,route='explicit-affine-'+mode,compile_artifact=pops.compile))
    directory=collective_directory(world,tmp_path/'explicit-affine')
    def root():return world is None or world.rank==0
    record_property('explicit_affine_provenance',str(directory/'provenance.json'))
    # Authenticate every component and the exact population partition before bind.
    collective_call(world,lambda:retain_provenance(artifact,resolved,native,directory) if root() else None)
    subject=resolved.initial_condition_plan.bindings[0].subject
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={subject:initial(mode)},
                resources={'execution_context':artifact_execution_context(artifact)}))
    attempts=[];phases=[];capture_failures=[];images={}
    def persist_receipt(status):
        if not root():return
        receipt={'schema':SCHEMA,'mode':mode,'degree':DEGREE,'dimension':2,
            'ranks':1 if world is None else int(world.size),'status':status,
            'basis':BASIS.to_data(),'binding':[[list(i),BINDING[i]] for i in INDICES],
            'phases':list(phases),'clocks':{phase:images[phase][2] for phase in phases},
            'attempts':attempts,'capture_failures':capture_failures,
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
                     if p.is_file() and p.name!='receipt.json'},'root_scientific_approval':False}
        (directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,allow_nan=False)+'\n')
    def saved(phase,image):
        images[phase]=image;phases.append(phase)
        if root():save_phase(directory,phase,image)
        persist_receipt('partial-captures')
    def failed(phase,failures):
        capture_failures.append({'phase':phase,'failures':failures})
        persist_receipt('capture-failed')
    record_property('explicit_affine_receipt',str(directory/'receipt.json'))
    before=persisted_capture(world,lambda:capture(world,runtime),
        lambda image:saved('before',image),lambda failures:failed('before',failures))
    def on_attempt(failures):
        attempts.append({'failures':failures})
        persist_receipt('native-run-failed' if any(failures) else 'native-run-returned')
    def after_capture():
        return persisted_capture(world,lambda:capture(world,runtime),lambda image:saved('after',image),
                                 lambda failures:failed('after',failures))
    def on_after(image,failed_run):
        persist_receipt('native-run-failed-captured' if failed_run else 'captures-complete')
    outcome,failures=collective_attempt(world,lambda:execute_captured_step(world,
        lambda:pops.run(runtime,t_end=DT,max_steps=1,console=False),after_capture,on_attempt,on_after))
    report=None if outcome is None else outcome[0]
    after=images.get('after')
    with collective_check(world):
        assert after is not None and not capture_failures,capture_failures
        np.testing.assert_allclose(before[0].reshape((len(NAMES),N,N)),initial(mode),rtol=0,atol=0)
        if mode=='success':
            assert not any(failures) and report.accepted_steps==1
            expected=np.broadcast_to(atom_moments(1)[:,None,None],(len(NAMES),N,N))
            np.testing.assert_allclose(after[0].reshape(expected.shape),expected,rtol=4e-12,atol=2e-13)
            assert after[2]==(DT,1,1)
        else:
            assert all(failure is not None and failure[2] is True and
                       ('non-finite' in failure[1] or 'nonfinite' in failure[1])
                       for failure in failures),failures
            assert before[0].dtype==after[0].dtype and before[0].shape==after[0].shape
            assert before[0].tobytes()==after[0].tobytes()
            assert before[1]==after[1] and before[2]==after[2]==(0.,0,1)
