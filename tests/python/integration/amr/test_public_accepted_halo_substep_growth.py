"""acceptedHaloTemporalDiscriminator@1, Source-authored Native test.

dc/dt=0, dm/dt=m, zero flux, explicit Forward Euler, periodic8x8,
two levels and temporal5/2 remainder. Active saved initial cells are the
oracle inputs; fine expected factor is (1+2DT/5)^2*(1+DT/5).
Two macrosteps and exact CP12/accepted9 restart/replay are captured before
checks, including grown carrier bytes and metadata/epoch payload members.
This closes a temporal realization witness only when ROOT runs it; it does
not prove halo-stage failure rollback, a general halo oracle or SCI reception.
"""
import math
import pops
from pops.amr import AcceptedHaloPreparation, AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy
from pops.analytic import cos, x
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import BergerRigoutsos, ConservativeInjection, CoarseFineInjection, StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every

FRAME = Rectangle("periodic evolving marker", (0., 0.), (1., 1.)).frame(Cartesian2D())
THRESHOLD = .7
DT = 1/64


def build_growth():
    subcycled = True
    from fractions import Fraction
    from pops.amr import AMRClockRelation, AMRRemainderPolicy
    shape, tag_buffer, transfer_kind = (8, 8), 0, "linear"
    order = (0,1)
    MODEL = pops.Model("evolving generic transport", frame=FRAME)
    STATE = MODEL.state("U", components=tuple(("constant","marker")[i] for i in order), sampling="cell_average")
    FLUX = MODEL.flux("zero", frame=FRAME, state=STATE,
        components={axis:tuple(0*v for v in STATE) for axis in FRAME.axes},
        waves={axis:tuple(0*v for v in STATE) for axis in FRAME.axes})
    SOURCE = MODEL.source("physical uniform drive", on=STATE,
        value=tuple(0*STATE[j] if i==0 else STATE[j] for j,i in enumerate(order)))
    RATE = MODEL.rate("linear physical evolution", equation=ddt(STATE) == -div(FLUX)+SOURCE)
    case = pops.Case("evolving accepted halo and public restart")
    block = case.block("marker", MODEL)
    numerics = DiscretizationPlan()
    numerics.rates.add(RATE, FiniteVolume(flux=FLUX, variables=variables.Conservative(STATE),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = pops.Program("evolving forward Euler")
    value = program.state(block[STATE])
    accepted = program.value("evolving accepted", value.n + program.dt*RATE(value.n), at=value.next.point)
    program.commit(value.next, accepted)
    program.store_history("accepted evolving image", accepted, depth=1)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    marker = cos(2*math.pi*x(FRAME))
    case.initials.add(InitialCondition(state=block[STATE], value=Analytic(frame=FRAME,
        components=tuple((1+0*marker,marker)[i] for i in order)), projection=ConservativeCellAverage()))
    policy = StateTransfer() if transfer_kind == "linear" else StateTransfer(
        prolongation=ConservativeInjection(), coarse_fine=CoarseFineInjection())
    transfer = AMRTransfer()
    transfer.state(block[STATE], policy)
    threshold = case.param(RuntimeParam("tag-threshold", default=THRESHOLD))
    layout = AMR(grid=CartesianGrid(frame=FRAME, cells=shape, periodic=PeriodicAxes(FRAME.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[STATE])["marker"] > case.value(threshold)), Buffer(cells=tag_buffer)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=(AMRExecution.subcycled((AMRClockRelation(0,1,Fraction(5,2),
            AMRRemainderPolicy.EXPLICIT_FINAL_SUBSTEP),), accepted_halo=AcceptedHaloPreparation(cells=(1,1)))
            if subcycled else AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation(cells=(1,1)))),
        clustering=BergerRigoutsos(minimum_efficiency=1., minimum_box_size=1, maximum_box_size=4))
    return case, layout

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from tests.python.support.accepted_halo_substep_oracle import CONTRACT,check,discriminate,factor
from tests.python.integration.amr.test_public_evolving_accepted_halo import observe,publish,same
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolving_accepted_halo_oracle import halo_rows

@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_accepted_halo_five_halves_substep_growth(isolated_native_cache,tmp_path,record_property):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native=select_native_dimension(2); world=native.mpi_world()
    with collective_check(world):
        assert native.__abi_version__==8
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout=collective_call(world,build_growth)
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=compile_resolved_plan_once(world,resolved,route=CONTRACT,compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    def bind(): return pops.bind(artifact,resources={'execution_context':context})
    runtime=collective_call(world,bind)
    directory=collective_directory(world,tmp_path/'substep-growth')
    record_property('accepted_halo_temporal_discriminator',str(directory))
    images={'initial':observe(world,runtime)}
    publish(world,directory,'initial',images['initial'])
    for step,phase in ((1,'accepted'),(2,'continuous')):
        report=collective_call(world,lambda step=step:pops.run(runtime,t_end=step*DT,max_steps=1,console=False))
        images[phase]=observe(world,runtime);publish(world,directory,phase,images[phase])
        with collective_check(world):assert report.accepted_steps==1 and report.rejected_steps==0
        if step==1:path=collective_call(world,lambda:runtime.checkpoint(directory/'accepted-checkpoint'))
    restarted=collective_call(world,bind)
    collective_call(world,lambda:restarted.restart(path))
    images['reloaded']=observe(world,restarted);publish(world,directory,'reloaded',images['reloaded'])
    collective_call(world,lambda:pops.run(restarted,t_end=2*DT,max_steps=1,console=False))
    images['replay']=observe(world,restarted);publish(world,directory,'replay',images['replay'])
    with collective_check(world):
        if world.rank==0:
            with np.load(path,allow_pickle=False) as checkpoint:
                payload={key:checkpoint[key].copy() for key in checkpoint.files}
            accepted=json.loads(str(payload['amr_accepted_contract']))
            (directory/'checkpoint-proof.json').write_text(json.dumps({
                'contract':CONTRACT,'payload_version':int(payload['pops_amr_checkpoint_version']),
                'accepted_contract':accepted,'members':{key:{'dtype':str(value.dtype),
                    'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                    for key,value in payload.items()}},indent=2)+'\n')
            assert int(payload['pops_amr_checkpoint_version'])==12
            assert type(accepted['schema_version']) is int and accepted['schema_version']==9
            assert accepted['accepted_halo']==images['accepted'][2][2]
        same(images['accepted'],images['reloaded']);same(images['continuous'],images['replay'])
        proof={'contract':CONTRACT,'dt':DT,'factors':[factor(0),factor(1)],'errors':{}}
        assert len(images['initial'][1])==2
        assert images['initial'][1][0][1].sum()==48
        assert images['initial'][1][1][1].sum()==64
        proof.update(discriminate(*images['initial'][1][1]))
        for phase,steps in (('initial',0),('accepted',1),('continuous',2),('reloaded',1),('replay',2)):
            image=images[phase]
            assert image[2][-1][:2]==(steps*DT,steps)
            halo_rows(image[2][2])
            assert image[2][3]==images['initial'][2][3]
            for level,((current,mask),(initial,oldmask)) in enumerate(zip(image[1],images['initial'][1],strict=True)):
                np.testing.assert_array_equal(mask,oldmask)
                proof['errors'][phase+str(level)]=check(initial,current,mask,steps,level)
        if world.rank==0:(directory/'temporal-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    record_property('artifact_identity',artifact.artifact_identity.token)
    record_property('native_sha256',hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
