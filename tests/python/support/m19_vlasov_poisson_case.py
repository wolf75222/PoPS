"""Declared nondimensional Vlasov–Poisson on finite v, periodic x; no collision law."""
from fractions import Fraction
import pops
from pops.model import Module,FieldSpace,Handle,OwnerPath,PhysicalSupport,PhysicalDimension
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import ddt,div,laplacian
from pops.fields import FieldProblem,FieldBoundary,FieldDiscretization,SharedMeanGauge,bcs,AnalyticAux,AuxiliaryBoundary
from pops.fields.methods import CellCenteredSecondOrder
from pops.analytic import coordinate
from pops.solvers import CG
from pops.time import FixedDt,FailRun
from pops.mesh import (CartesianGrid,PeriodicAxes,PhysicalSupportMap,AxisQuadrature,LayoutPlanBuilder,
    LayoutMappingOperation,LayoutRepresentation,LayoutSynchronization,native_physical_mapping)
from pops.layouts import Uniform
from pops.numerics import DiscretizationPlan,FiniteVolume,reconstruction,riemann,variables
from pops.numerics.terms import Flux
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import NoFlux
UNIT=PhysicalDimension()
PHASE=PhysicalSupport((('velocity','finite [-1,1]'),('position','periodic [0,1]')))
POSITION=PhysicalSupport((('position','periodic [0,1]'),))
DT=Fraction(1,128)

def build(directory,*,nx=8,nv=4,charge=1,provider_factory=native_physical_mapping):
    phase_frame=Rectangle('v then x',(-1,0),(1,1)).frame(Cartesian2D())
    physical_frame=Rectangle('x then hidden',(0,0),(1,1)).frame(Cartesian2D())
    from pops.model import Rate
    from pops.model.flux_waves import FluxWaveLaw
    phase=Module('kinetic charge transport',frame=phase_frame)
    sentinel_model=Module('independent inert material',frame=phase_frame)
    spectator=sentinel_model.state_space('unrelated',('sentinel_a','sentinel_b'),support=PHASE,units=(UNIT,UNIT),sampling='cell_average')
    f=phase.state_space('distribution',('number_density_in_phase',),support=PHASE,units=(UNIT,),sampling='cell_average')
    incoming=phase.field_space('lifted electric force',('electric',),support=PHASE,units=(UNIT,),sampling='cell')
    electric=phase.field_symbols(incoming)[0]
    from pops._ir.expr import Var
    velocity=Var('velocity_coordinate','aux')
    phase.aux_field('velocity_coordinate','cell_scalar',frame=phase_frame.canonical_id,unit='1')
    va,xa=phase_frame.axes
    coordinate_f=phase.state_symbols(f)[0]
    flux=phase.operator(name='phase_transport',signature=(f,incoming)>>Rate(f),kind='grid_operator',
        requirements={'aux':('velocity_coordinate',)},
        expr={'x':(charge*electric*coordinate_f,), 'y':(velocity*coordinate_f,)},
        lowering={'flux_wave_law':FluxWaveLaw(f,{'x':(charge*electric,), 'y':(velocity,)},
            signed_bounds={'x':(charge*electric,charge*electric), 'y':(velocity,velocity)})})
    rate=phase.rate_operator('collisionless_vlasov',state_space=phase.state_handle(f),flux=True,fluxes=(flux,),default_flux=flux,sources=[])
    module=phase
    aux=module.aux_handle(module.aux()['velocity_coordinate'])
    module.aux_provider(AnalyticAux(aux,coordinate(phase_frame,va),frame=phase_frame,boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    density=Module('number moment',frame=physical_frame)
    rho_space=density.state_space('reduced number',('density',),support=POSITION,units=(UNIT,),sampling='cell_average')
    case=pops.Case('finite velocity neutral periodic Vlasov Poisson')
    sb=case.block('a_spectator',sentinel_model);pb=case.block('z_phase',phase,states=(phase.state_handle(f),));db=case.block('m_number',density)
    fs=pb[phase.state_handle(f)];rs=db[density.state_handle(rho_space)];ss=sb[sentinel_model.state_handle(spectator)]
    target=pb[module.field_handle(incoming)]
    potential=Handle('potential',kind='field',owner=OwnerPath.model('electrostatics'))
    physical=FieldProblem('neutral periodic Poisson',unknowns=(potential,),
        equations=(-laplacian(potential)==charge*(density.state_symbols(rho_space)[0]-1),),
        boundaries=(FieldBoundary(potential,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),),
        gauge=SharedMeanGauge((potential,),0),
        unknown_spaces={potential:FieldSpace('electrostatic potential',('phi',),support=POSITION,units=(UNIT,),sampling='cell')},coordinate_units=(UNIT,))
    field=case.field(physical,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),observation_axes=(0,),solver=CG(max_iter=200,rel_tol=1e-12,abs_tol=1e-12)))
    p=pops.Program('SSPRK2 recompute number Poisson electric each stage')
    kinetic,number,sentinel=p.state(fs),p.state(rs),p.state(ss)
    reduction=PhysicalSupportMap(PHASE,POSITION,source_axes=(0,1),target_axes=(0,),native_dimension=2,reductions=(AxisQuadrature(0,-1,1,nv,UNIT),))
    lift=PhysicalSupportMap(POSITION,PHASE,source_axes=(0,),target_axes=(0,1),native_dimension=2)
    mapped=[]
    def evaluate(value,label):
        stage=number.stage(value.point.name,point=value.point)
        moment=p.map(reduction,source=value,target=stage)
        solution=field.observe(p.solve(field,values={rs:moment},at=moment.point).consume(action=FailRun()))
        port=solution.mapping_port(field[potential],derivative_axis=0,factor=-1)
        context=solution.publish_mapped(lift,{(target,'electric'):port},states={fs:value})
        mapped.append((port,solution))
        return p.rhs(state=value,fields=context,terms=[Flux()]),moment
    initial_point=p.stage('initial accepted coordinate',c=0)
    initial=p.value('read accepted distribution',kinetic.n,at=initial_point)
    r0,n0=evaluate(initial,'initial')
    point=p.stage('predictor endpoint',c=1)
    predictor=p.value('Euler predictor',kinetic.n+p.dt*r0,at=point)
    r1,n1=evaluate(predictor,'predictor')
    accepted=p.value('SSPRK2 accepted',Fraction(1,2)*kinetic.n+Fraction(1,2)*(predictor+p.dt*r1),at=kinetic.next.point)
    p.commit(kinetic.next,accepted)
    p.commit(number.next,p.value('last stage density',n1,at=number.next.point))
    p.commit(sentinel.next,p.value('unrelated unchanged',sentinel.n,at=sentinel.next.point))
    p.step_strategy(FixedDt(float(DT)));case.program(p)
    numerical=DiscretizationPlan();numerical.rates.add(rate,FiniteVolume(flux=(flux,),variables=variables.Conservative(phase.state_handle(f)),reconstruction=reconstruction.FirstOrder(),riemann=riemann.HLL(waves=riemann.waves.ExplicitPair())))
    numerical.boundaries.add(TransportBoundarySet({phase_frame.boundaries.x_min:NoFlux(state=fs),phase_frame.boundaries.x_max:NoFlux(state=fs)},periodic=PeriodicAxes((xa,))))
    case.numerics(numerical,block=pb)
    validated=pops.validate(case);subjects=validated.layout_subjects();builder=LayoutPlanBuilder(validated.owner_path.canonical())
    grids=(Uniform(CartesianGrid(frame=phase_frame,cells=(nv,nx),periodic=PeriodicAxes((xa,)))),Uniform(CartesianGrid(frame=physical_frame,cells=(nx,1),periodic=PeriodicAxes(physical_frame.axes))))
    layouts=(builder.layout('kinetic storage',grids[0]),builder.layout('physical storage',grids[1]))
    for row in subjects.blocks:builder.assign_block(row,layouts[row.local_id=='m_number'])
    for row in subjects.states:builder.assign_state(row,layouts[row.block_ref.local_id=='m_number'])
    for row in subjects.fields:builder.assign_field(row,layouts[0] if row.block_ref else layouts[1])
    requirements=[]
    states_by_block={row.block_ref.local_id:row for row in subjects.states}
    req,=builder.require_mapping(layouts[0],layouts[1],source=states_by_block['z_phase'],target=states_by_block['m_number'],operation=LayoutMappingOperation(reduction.operation_abi),synchronization=LayoutSynchronization.PROGRAM_POINT_V1,source_representation=LayoutRepresentation.CELL_AVERAGE_V1,target_representation=LayoutRepresentation.CELL_AVERAGE_V1,physical_map=reduction);requirements.append(req)
    for port,solution in mapped[:1]:
        req,=builder.require_mapping(layouts[1],layouts[0],source=field,target=target,operation=LayoutMappingOperation(lift.operation_abi),synchronization=LayoutSynchronization.PROGRAM_POINT_V1,source_representation=LayoutRepresentation.CELL_FIELD_V1,target_representation=LayoutRepresentation.CELL_FIELD_V1,source_observation=port,target_component='electric',physical_map=lift)
        if req not in requirements:requirements.append(req)
    providers=tuple(provider_factory(row,directory) for row in requirements)
    layout=builder.resolve(**subjects.to_dict(),providers=providers)
    return pops.resolve(validated,layout=layout,layout_providers=dict(zip(layouts,grids)),components=tuple(row.component for row in providers),compile_options={'model_source_policy':'require'})
