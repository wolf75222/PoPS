"""Signed operator-first flux authority; Source/real-header Host, no Native engine."""
from pathlib import Path
import os,sys,subprocess
import numpy as np
import pytest
from pops.model import Module,Rate,FluxWaveLaw,ModuleManifest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops._ir.expr import Const,Minimum,Maximum
from pops.codegen.module_lowering import lower_and_validate
from pops.numerics.riemann.waves import provider_of,check_hll_waves
from pops.codegen.loader import CompiledModel

def independent(signed=True):
    frame=Rectangle('nonkinetic material',(0,0),(1,1)).frame(Cartesian2D())
    module=Module('two component signed material',frame=frame)
    state=module.state_space('transport',('heat','dye'),sampling='cell_average')
    u=module.state_symbols(state)
    # Deliberate reversed authoring axis insertion; spectrum contains negative, positive and zero.
    values={'y':(Const(0),Const(3)),'x':(Const(-2),Const(0))}
    pairs={axis:(Minimum(*speeds),Maximum(*speeds)) for axis,speeds in values.items()} if signed else None
    flux=module.operator('material_flux',signature=state>>Rate(state),kind='grid_operator',
        expr={'y':(0*u[0],3*u[1]),'x':(-2*u[0],0*u[1])},
        lowering={'flux_wave_law':FluxWaveLaw(state,values,signed_bounds=pairs)})
    module.rate_operator('transport_rate',state_space=module.state_handle(state),flux=True,fluxes=(flux,),default_flux=flux,sources=[])
    return module,state,flux

def compiled_metadata(carrier):
    # Real inert CompiledModel API. No fabricated Native binary or engine is loaded.
    source=provider_of(carrier)
    return CompiledModel('source-metadata-only-unloaded.so','production',tuple(carrier._m.cons_names),
        ('custom',)*carrier._m.n_vars,(),carrier._m.n_vars,None,0,{}, {'cpu':True},
        'source-only','0'*64,'clang++','c++20',2,wave_speeds=source is not None,
        wave_speed_provider=None if source is None else source.kind)

def test_nonvp_actual_legacy_refusal_then_signed_lowering_and_bind_guard():
    old,_,_=independent(False);legacy,_=lower_and_validate(old)
    with pytest.raises(ValueError,match='emits none'):check_hll_waves('explicit_pair',compiled_metadata(legacy),'public bind metadata')
    module,state,flux=independent();carrier,_=lower_and_validate(module)
    check_hll_waves('explicit_pair',compiled_metadata(carrier),'public bind metadata')
    assert provider_of(carrier).kind=='explicit_pair'
    assert carrier.eval_wave_speeds(np.array([7.,11.]),{},0)==(-2.,0.)
    assert carrier.eval_wave_speeds(np.array([7.,11.]),{},1)==(0.,3.)
    data=module.manifest().to_dict()
    assert ModuleManifest.from_dict(data).to_dict()==data
    assert next(op for op in module.operator_registry() if op.kind=='grid_operator').lowering['flux_wave_law'].to_data()['contract']=='pops.flux-wave-law@2'

@pytest.mark.parametrize('mutation',('missingaxis','pairarity','boolroot','unknownkey'))
def test_signed_codec_rejects_malformed_authority(mutation):
    module,_,_=independent();data=module.manifest().to_dict()
    law=next(row for row in data['operators'] if row['kind']=='grid_operator')['lowering_route']['flux_wave_law']
    if mutation=='missingaxis':law['signed_bounds'].pop('y')
    elif mutation=='pairarity':law['signed_bounds']['x']['roots'].pop()
    elif mutation=='boolroot':law['signed_bounds']['x']['roots'][0]=True
    else:law['foreign']=1
    with pytest.raises((ValueError,TypeError)):ModuleManifest.from_dict(data)

def test_vp_all_three_models_exact_spatial_and_wave_metadata(tmp_path):
    from tests.python.support.m19_vlasov_poisson_case import build
    from pops.codegen._compiler_lowering import require_compiler_lowering
    from pops.numerics import StateStorage
    resolved=build(tmp_path)
    rows={b.name:b for b in resolved.blocks}
    assert set(rows)=={'a_spectator','m_number','z_phase'}
    for name,block in rows.items():
        carrier=require_compiler_lowering(block.model).emit_model
        if name=='z_phase':
            assert provider_of(carrier).kind=='explicit_pair'
            check_hll_waves('explicit_pair',compiled_metadata(carrier),'VP active bind metadata')
        else:
            assert type(block.spatial)is StateStorage
            assert provider_of(carrier)is None and carrier._m._program_only_storage_axes==('x','y')

def test_actual_signed_model_translation_unit_against_public_headers(tmp_path):
    from pops.codegen._compile_emit import emit_cpp_native_loader
    module,_,_=independent();carrier,_=lower_and_validate(module)
    source=tmp_path/'signed-material.cpp';source.write_text(emit_cpp_native_loader(carrier._m))
    root=Path(__file__).resolve().parents[2]
    headers=Path(os.environ.get('POPS_TEST_HOST_SDK_INCLUDE',str(Path(sys.prefix)/'include')))
    assert (headers/'Kokkos_Core.hpp').is_file()
    command=['clang++','-std=c++20','-fsyntax-only','-Xpreprocessor','-fopenmp','-DPOPS_HAS_KOKKOS','-DPOPS_NATIVE_DIM=2','-I',str(root/'include'),'-I',str(headers),str(source)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

@pytest.mark.parametrize('bounds', ({'x':(0,1)}, {'x':(0,), 'y':(0,1)}, {'x':(0,1,2),'y':(0,1)}))
def test_authoring_bounds_refuse_axis_or_pair_contradictions(bounds):
    module,state,_=independent(False)
    with pytest.raises((ValueError,TypeError)):
        FluxWaveLaw(state,{'x':(Const(-2),Const(0)),'y':(Const(0),Const(3))},signed_bounds=bounds)


def test_same_output_group_does_not_borrow_missing_signed_pair():
    from pops.model.flux_waves import common_flux_signed_bounds
    module,state,_=independent()
    u=module.state_symbols(state)
    module.operator('second_material_flux',signature=state>>Rate(state),kind='grid_operator',
        expr={'x':(-2*u[0],0*u[1]),'y':(0*u[0],3*u[1])},
        lowering={'flux_wave_law':FluxWaveLaw(state,{'x':(Const(-2),Const(0)),'y':(Const(0),Const(3))})})
    names={op.name for op in module.operator_registry() if op.kind=='grid_operator'}
    with pytest.raises(ValueError,match='ambiguous signed bounds'):common_flux_signed_bounds(module,names)
