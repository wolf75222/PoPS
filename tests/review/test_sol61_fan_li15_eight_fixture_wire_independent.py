"""Non-author frozen wire checks; synthetic serialization, never Native proof."""
import ast
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from tests.review import sol61_fan_li15_eight_saved_reader as reader

FREEZE='02cdcb8e9169b7ecf4f1271ea2c7856d92b49513'
FILE='tests/python/integration/runtime/test_fan_li15_full_eight_step_public_runtime.py'


def test_actual_frozen_phase_serializer_preserves_reverse_component_order(tmp_path):
    root=Path(__file__).resolve().parents[2]
    source=subprocess.check_output(['git','show',FREEZE+':'+FILE],cwd=root,text=True)
    tree=ast.parse(source)
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='save_phase']
    assert len(selected)==1
    scope={'np':np,'json':json,'SCHEMA':'pops.fan-li15-public-composition-native-fixture@2',
           'CARRIER_CAPTURE':{'status':'unavailable'}}
    exec(compile(ast.Module(body=selected,type_ignores=[]),FILE,'exec'),scope)
    q=np.broadcast_to(reader.initial()[::-1,None,:],(15,16,16)).copy()
    q[7,3,4]=-0.0  # wire-only signed-zero witness, not a scientific state
    for step in range(9):
        phase='initial' if step==0 else 'accepted'+str(step)
        scope['save_phase'](tmp_path,phase,(q,(float(step*reader.DT),step)))
        loaded=np.load(tmp_path/(phase+'.npy'),allow_pickle=False)
        assert loaded.dtype==q.dtype and loaded.shape==q.shape and loaded.tobytes()==q.tobytes()
        assert json.loads((tmp_path/(phase+'.clock.json')).read_text())==[float(step*reader.DT),step]
    assert len(list(tmp_path.glob('*.npy')))==9 and not list(tmp_path.glob('*.carriers'))


@pytest.mark.parametrize('mutation',('missing','bool_tick','swapped_axes'))
def test_nine_wire_files_do_not_imply_scientific_acceptance(mutation):
    q=np.broadcast_to(reader.initial()[:,None,:],(15,16,16)).copy()
    states=[q.copy() for _ in range(9)]
    clocks=[{'time':float(i*reader.DT),'tick':i} for i in range(9)]
    if mutation=='missing':states.pop()
    elif mutation=='bool_tick':clocks[0]['tick']=False
    else:states[0]=states[0].transpose(0,2,1)
    with pytest.raises(ValueError):reader.audit_states(states,clocks,reader.INDICES)
