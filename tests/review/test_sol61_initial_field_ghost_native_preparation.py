"""Source-only genuine public planning and synthetic wire adversaries, no Native claim."""
import struct
import numpy as np
import pytest
from tests.python.support.initial_field_ghost_native_oracle import check,DT

def wire(*,steps=0,ghost=None):
    b=bytearray(b'POPSCAR1')
    def word(x,signed=False):b.extend(int(x).to_bytes(8,'little',signed=signed))
    for x in (2,64,1):word(x)
    word(-1,True)
    for x in (2,1,6):word(x)
    b.extend(b'marker');word(2)
    for level in range(2):
        for x in (0,level,0,2):word(x)
        word(0,True)
        for axis in range(2):
            for x in (0,1,-1,2):word(x,True)
        data=np.zeros((2,4,4));data[1]=2*(1+DT)**steps
        data[1,1:3,0]=data[1,1,1]+1+steps*DT if ghost is None else ghost
        word(data.size)
        b.extend(data.astype('<f8').tobytes())
    return bytes(b)

def fields(steps):return [(np.full((2,2),2*(1+DT)**steps),np.ones((2,2),bool)) for _ in range(2)]

def test_actual_public_field_ghost_plan():
    import pops
    from tests.python.support.initial_field_ghost_native_case import build
    case,layout=build();plan=pops.resolve(pops.validate(case),layout=layout)
    component,=plan.component_inputs
    evidence=component.component_manifest.signature['inferred_boundary_expression']
    assert {i['name'] for i in evidence['interfaces']}=={'ghost_boundary','accepted_initial_ghost'}
    assert len(evidence['dependencies']['fields'])==1 and not evidence['dependencies']['states']
    assert len(plan.field_plans)==1
    assert layout.execution.runtime_execution_data()['mode']=='synchronous'

@pytest.mark.parametrize('steps',(0,1))
def test_independent_uniform_reference_wire(steps):check(wire(steps=steps),fields(steps),steps)

@pytest.mark.parametrize('ghost',(2*(1+DT),3+DT,2*(1+DT)+1,0))
def test_wrong_boundary_stale_field_or_missing_time_refused(ghost):
    with pytest.raises(AssertionError):check(wire(steps=1,ghost=ghost),fields(1),1)

def test_stale_field_provider_refused():
    with pytest.raises(AssertionError):check(wire(steps=1),fields(0),1)
