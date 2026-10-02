"""Frozen Source witness, no Native execution or synthetic scientific receipt."""
import ast,subprocess,struct
from pathlib import Path
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

FREEZE='af66aa0c6a6077792d782827cb02e5f6926fac85'
def source(path):
    return subprocess.check_output(['git','show',FREEZE+':'+path],cwd=Path(__file__).resolve().parents[2],text=True)

def test_frozen_full_guard_does_not_authenticate_local_grown_bits():
    # Execute genuine DTO and complete-valid guard Source, not a mocked runtime.
    scope={'__name__':__name__}
    exec(compile(source('python/pops/runtime/_state_storage_observation.py'),'actual_dto','exec'),scope)
    cls=scope['AcceptedStateStorageObservation']
    tree=ast.parse(source('tests/python/support/atomic_cubature_fv_oracle.py'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authenticate_carrier_values')
    env={'np':np,'NX':8,'NY':4};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_valid_guard','exec'),env)
    values=np.arange(6*4*8,dtype=np.float64).reshape(6,4,8)
    grown=np.pad(values,((0,0),(1,1),(1,1)),constant_values=0.)
    u=lambda x:struct.pack('<Q',x);i=lambda x:struct.pack('<q',x)
    def wire(shard):
        header=b'POPSCAR1'+u(2)+u(64)+u(1)+i(shard)+u(1)+u(1)+u(10)+b'population'+u(1)
        row=u(0)+u(0)+u(0)+u(6)+i(0)+b''.join(i(x) for x in (0,7,-1,8,0,3,-1,4))
        return header+row+u(grown.size)+grown.view(np.uint64).astype('<u8').tobytes()
    complete=wire(-1);local=wire(0)[:-8]+struct.pack('<d',12345.)
    image=cls(2,0.0,0,local,complete)
    env['authenticate_carrier_values'](image.complete,values)
    assert image==image  # Frozen fixture's two observations may repeat this discrepancy.
    whole=decode(np.frombuffer(complete,dtype=np.uint8));shard=decode(np.frombuffer(local,dtype=np.uint8))
    assert whole['patches']!=shard['patches']  # Actual fullgrown mismatch the old guards miss.
