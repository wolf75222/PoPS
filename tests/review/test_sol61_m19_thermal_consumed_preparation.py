import math
from types import SimpleNamespace
import pytest
import numpy as np
from tests.python.support.m19_thermal_consumed_case import build, reference, WEIGHTS

@pytest.mark.parametrize('reverse',(False,True))
@pytest.mark.parametrize('gradient',(False,True))
def test_public_temperature_solve_two_mapped_consumers_resolve_emit(tmp_path,reverse,gradient):
    resolved=build(tmp_path,reverse=reverse,gradient=gradient)
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph
    assignments={r.subject.local_id:r.layout for r in resolved.layout_plan.assignments if r.subject_kind=='block'}
    emitted=[]
    for layout in resolved.layout_plan.layouts:
        blocks=tuple(b for b in resolved.blocks if assignments[b.name]==layout.handle)
        program=detach_compiled_program(slice_program(resolved.time,tuple(b.name for b in blocks)))
        emitted.append(emit_program_graph(program.to_graph(),lowering_program=program,
            model_graph=ProgramModelGraph.from_resolved_blocks(blocks),target='system',field_plans={}))
    assert len(resolved.layout_plan.layouts)==2 and len(resolved.layout_plan.mappings)==2
    assert resolved.blocks[0].name!='zeta_laminated_material'
    assert sum(text.count('publish_field_components(') for text in emitted)==2
    assert any('mapped_field_' in text for text in emitted)

@pytest.mark.parametrize('gradient',(False,True))
def test_independent_discrete_helmholtz_and_consumption_discriminators(gradient):
    initial,expected,phi,load=reference(gradient=gradient)
    assert np.max(abs(load-phi))>.2
    assert np.max(abs((phi-(np.roll(phi,-1)-2*phi+np.roll(phi,1))*2)-load))<1e-15
    for i,name in enumerate(('reservoir_z','reservoir_a')):
        delta=expected[name][1]-initial[name][1]
        assert np.max(abs(delta))>1e-4
        assert np.array_equal(initial[name][(0,2),:,:],expected[name][(0,2),:,:])
        stale=2*(i+1)*sum(float(w)*float(v) for w,v in zip(WEIGHTS[i],load))/64
        if not gradient: assert abs(float(delta[0,0])-stale)>1e-4

def test_abi9_hidden_partition_and_foreign_native_facts_refuse():
    from pops.runtime._mapped_field_capability import require_mapped_consumed_field_output
    def artifact(*irs):
        return SimpleNamespace(layout_programs=tuple(SimpleNamespace(program=SimpleNamespace(program=SimpleNamespace(_serialize=lambda ir=ir:ir))) for ir in irs))
    hidden=artifact({'nodes':[]},{'nodes':[{'op':'field_map_pack'}]})
    for facts in ({'abi_version':8,'mapped_consumed_field_output':True},
                  {'abi_version':9,'mapped_consumed_field_output':False},
                  {'abi_version':9,'mapped_consumed_field_output':1},
                  {'abi_version':9.0,'mapped_consumed_field_output':True}):
        calls=[]
        with pytest.raises(RuntimeError,match='ABI9'):
            require_mapped_consumed_field_output(hidden,capability_reader=lambda target:calls.append(target) or facts)
        assert calls==['production']
    require_mapped_consumed_field_output(artifact({'nodes':[{'op':'layout_map_import','attrs':{'contract':'state-map@1'}}]}),
        capability_reader=lambda target:pytest.fail('legacy State map must not request Field capability'))

def test_actual_capture_preserves_first_getter_when_second_refuses(tmp_path):
    """Execute the real capture body with Source-only observations, never a Native substitute."""
    import ast
    import json
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'python/integration/runtime/test_m19_thermal_consumed_runtime.py'
    tree=ast.parse(path.read_text())
    outer=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='test_installed_thermal_consumed_field_two_destinations')
    capture=next(node for node in outer.body if isinstance(node,ast.FunctionDef) and node.name=='capture')
    class SourceOnly:
        consumer_cursors=SimpleNamespace(to_data=lambda:{'explicit':'SourceOnly'})
        def time(self):return .125
        def macro_step(self):return 8
        def local_boxes(self,name):return ((0,0),(0,0))
        def state_global(self,name):
            if name=='zeta_laminated_material':raise ValueError('second getter refused')
            return np.full((3,3,1),7.)
        def checkpoint(self,path):pytest.fail('checkpoint after getter refusal')
    def save_json(path,value):path.write_text(json.dumps(value,allow_nan=False))
    env={'np':np,'runtime':SourceOnly(),'world':None,'rank':0,'directory':tmp_path,
         'NAMES':('reservoir_z','zeta_laminated_material','reservoir_a'),
         'collective_call':lambda world,f:f(),'save_json':save_json,
         'pin':lambda path:{'file':str(path)},'Path':Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[capture],type_ignores=[])),str(path),'exec'),env)
    with pytest.raises(ValueError,match='second getter refused'):env['capture']('partial')
    assert np.array_equal(np.load(tmp_path/'partial-rank0-reservoir_z.npy'),np.full((3,3,1),7.))
    receipt=json.loads((tmp_path/'partial-rank0.json').read_text())
    assert receipt['capture_complete'] is False and tuple(receipt['files'])==('reservoir_z',)
    assert receipt['consumer_cursors']=={'explicit':'SourceOnly'} and receipt['clock']==[.125,8]
