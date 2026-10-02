"""Independent output-State wave laws through the public compiler view."""
import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops._ir.expr import Const
from pops.codegen.module_lowering import lower_and_validate
from pops.model.flux_waves import flux_waves, common_flux_waves
from pops.model import ModuleManifest


def make(include_rates=False):
    frame=Rectangle('waves_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('wave_supports',frame=frame)
    a=model.species('earlier',state=('r','s'))
    b=model.species('later',state=('x','y','z'))
    rates=[]
    for state,name,speed in ((a,'transport_a',2.),(b,'transport_b',3.)):
        flux=model.flux(name,state=state,frame=frame,
          components={axis:tuple(speed*q for q in state) for axis in frame.axes},
          waves={axis:tuple(Const(speed) for q in state) for axis in frame.axes})
        from pops.math import ddt, div
        rates.append(model.rate("rate_"+name,equation=ddt(state)==-div(flux)))
    return (model,a,b,tuple(rates)) if include_rates else (model,a,b)


def test_public_distinct_state_fluxes_lower_exact_waves_and_roundtrip():
    model,a,b=make()
    module=model.module
    for state,speed in ((a,2.),(b,3.)):
        lowered,source=lower_and_validate(model,state_space=state.space.name)
        assert all(len(values)==len(state.components) for values in lowered._m._eig.values())
        assert all(value.value==speed for values in lowered._m._eig.values() for value in values)
    manifest=module.manifest()
    assert ModuleManifest.from_dict(manifest.to_dict()).to_dict()==manifest.to_dict()
    laws=[operator.lowering['flux_wave_law'] for operator in module.operator_registry() if operator.kind=='grid_operator']
    assert {law.to_data()['contract'] for law in laws}=={'pops.flux-wave-law@1'}
    assert len({law.output_state.name for law in laws})==2


def test_joint_ambiguity_refused_but_explicit_global_authority_kept():
    model,a,b=make();module=model.module
    names={operator.name for operator in module.operator_registry() if operator.kind=='grid_operator'}
    with pytest.raises(ValueError,match='ambiguous'):common_flux_waves(module,names)
    module.eigenvalues(x=(Const(7.),),y=(Const(7.),))
    assert common_flux_waves(module,names,global_authority=True) is module._eigenvalues
    assert all(tuple(values)[0].value in (2.,3.) for operator in module.operator_registry() if operator.kind=='grid_operator' for values in flux_waves(module,operator).values())


def test_two_public_transports_resolve_and_emit_separately():
    from pops.math import ddt,div
    from pops.numerics import DiscretizationPlan,FiniteVolume,reconstruction,riemann,variables
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.time import FixedDt
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    model,a,b,rates=make(True);case=pops.Case('independent_transport_case');program=pops.Program('two_transport_step')
    for state,name,rate in ((a,'transport_a',rates[0]),(b,'transport_b',rates[1])):
        flux=model.fluxes[name]
        block=case.block('block_'+state.space.name,model,states=(state,))
        plan=DiscretizationPlan();plan.rates.add(rate,FiniteVolume(flux=flux,variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
        case.numerics(plan,block=block)
        q=program.state(block[state]);rhs=rate(q.n)
        program.commit(q.next,program.value('endpoint_'+state.space.name,q.n+program.dt*rhs,at=q.next.point))
    program.step_strategy(FixedDt(.001));case.program(program)
    layout=Uniform(CartesianGrid(frame=model.frame,cells=(4,3),periodic=PeriodicAxes(model.frame.axes)))
    resolved=pops.resolve(pops.validate(case),layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source=emit_cpp_program(resolved.time,model=graph)
    assert 'pops_program_block_count() { return 2; }' in source
    for block in resolved.blocks:
        lowered,_=lower_and_validate(block.model,state_space=block.state_spaces[0],resolved_operations=block.resolved_operations,numerics=block.numerics)
        expected=2 if block.name=='block_earlier' else 3
        assert all(len(values)==expected for values in lowered._m._eig.values())


@pytest.mark.parametrize('mutation',('contract','state','shape','boolroot'))
def test_manifest_wave_codec_refuses_foreign_or_malformed(mutation):
    model,a,b=make();data=model.module.manifest().to_dict()
    law=data['operators'][0]['lowering_route']['flux_wave_law']
    if mutation=='contract':law['contract']='pops.flux-wave-law@99'
    elif mutation=='state':law['output_state']['name']='foreign'
    elif mutation=='shape':law['values']['x']['roots'].pop()
    else:law['values']['x']['roots'][0]=True
    with pytest.raises((ValueError,TypeError)):ModuleManifest.from_dict(data)


def test_state_dependent_wave_identity_is_stable_and_frozen():
    frame=Rectangle('nonlinear_wave_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('nonlinear_wave_support',frame=frame)
    a=model.species('unused',state=('s',));b=model.species('density_last',state=('moment','density'))
    model.flux('nonlinear',state=b,frame=frame,
       components={axis:(b[0]*b[1],b[1]*b[1]/2) for axis in frame.axes},
       waves={axis:(b[1],b[1]) for axis in frame.axes})
    first=model.module.module_hash()
    manifest=model.module.manifest().to_dict()
    assert first==model.module.module_hash()
    assert ModuleManifest.from_dict(manifest).to_dict()==manifest
    model.module.freeze()
    with pytest.raises(TypeError):next(op for op in model.module.operator_registry() if op.kind=='grid_operator').lowering['foreign']=1


@pytest.mark.parametrize('counts',((2,21),(1,4)))
def test_arbitrary_independent_support_arities(counts):
    frame=Rectangle('ranked_waves',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('ranked_wave_model',frame=frame)
    states=[model.species('support_'+str(k),state=tuple('component_'+str(i) for i in range(count))) for k,count in enumerate(counts)]
    from pops.math import ddt,div
    for k,state in enumerate(states):
        coefficients={axis:(k+1)*(j+1) for j,axis in enumerate(frame.axes)}
        flux=model.flux('flux_'+str(k),state=state,frame=frame,
            components={axis:tuple(value*q for q in state) for axis,value in coefficients.items()},
            waves={axis:tuple(value for q in state) for axis,value in coefficients.items()})
        model.rate('balance_'+str(k),equation=ddt(state)==-div(flux))
    for k,state in enumerate(states):
        lowered,_=lower_and_validate(model,state_space=state.space.name)
        assert {axis:tuple(v.value for v in values) for axis,values in lowered._m._eig.items()}=={
            axis:((k+1)*(j+1),)*counts[k] for j,axis in enumerate(('x','y'))}


def test_forged_live_output_authority_is_refused():
    from pops.model import FluxWaveLaw
    model,a,b=make();operators=[op for op in model.module.operator_registry() if op.kind=='grid_operator']
    operators[0].lowering['flux_wave_law']=operators[1].lowering['flux_wave_law']
    with pytest.raises(ValueError,match='output State'):flux_waves(model.module,operators[0])


def test_actual_emitted_wave_methods_keep_second_state_slots_host(tmp_path):
    import shutil,subprocess
    from pathlib import Path
    from pops.codegen.module_codegen import _emit_bricks
    from pops.math import ddt,div
    frame=Rectangle('host_wave_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('host_wave_model',frame=frame)
    earlier=model.species('earlier',state=('unrelated',))
    later=model.species('later',state=('moment','density'))
    flux=model.flux('state_wave',state=later,frame=frame,
      components={axis:(later[0]*later[1],later[1]*later[1]/2) for axis in frame.axes},
      waves={axis:(later[1],later[1]) for axis in frame.axes})
    model.rate('balance',equation=ddt(later)==-div(flux))
    selected,_=lower_and_validate(model,state_space=later.space.name)
    text=_emit_bricks(selected._m)[1]
    pos=text.index('POPS_HD pops::Real max_wave_speed')
    start=text.rfind('template <int Axis>',0,pos);brace=text.index('{',pos);depth=1;end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    method=text[start:end]
    assert 'density = U[1]' in method
    cpp=tmp_path/'waves.cpp';exe=tmp_path/'waves'
    cpp.write_text('#include <pops/core/state/state.hpp>\nstruct Witness {using State=pops::StateVec<2>; static constexpr int dimension=2;\n'+method+'\n};\nint main(){Witness::State s{};s[0]=12345;s[1]=7;return Witness{}.max_wave_speed<0>(s,0)==7 && Witness{}.max_wave_speed<1>(s,0)==7 ? 0:1;}\n')
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    run=subprocess.run([compiler,'-std=c++20','-DPOPS_NATIVE_DIM=2','-O2','-fno-fast-math','-ffp-contract=off','-Werror=return-type','-I',str(Path(__file__).resolve().parents[2]/'include'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    subprocess.run([str(exe)],check=True)
