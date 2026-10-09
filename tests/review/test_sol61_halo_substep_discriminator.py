"""SourceOnly oracle tests; Native node execution belongs to ROOT."""
import numpy as np
import pytest
from tests.python.support.accepted_halo_substep_oracle import DT,check,discriminate,factor

def arrays():
    mask=np.ones((2,2),dtype=bool)
    initial=np.stack((np.ones((2,2)),np.array([[.8,.9],[-.8,-.9]])))
    return initial,mask

@pytest.mark.parametrize('level',(0,1))
@pytest.mark.parametrize('steps',(1,2))
def test_closed_euler_factor_accepts(level,steps):
    initial,mask=arrays();current=initial.copy()
    for _ in range(steps):
        for dt in ((DT,) if level==0 else (2*DT/5,2*DT/5,DT/5)):
            current[1]=current[1]+dt*current[1]
    check(initial,current,mask,steps,level)
    assert discriminate(initial,mask)['roundoff_envelope']>0

@pytest.mark.parametrize('schedule',((DT,),(DT/2,DT/2),(2*DT/5,2*DT/5)))
def test_actual_wrong_euler_schedule_refused(schedule):
    initial,mask=arrays();current=initial.copy()
    for dt in schedule:current[1]=current[1]+dt*current[1]
    with pytest.raises(AssertionError):check(initial,current,mask,1,1)

def test_actual_public_physical_plan_source():
    import pops
    from tests.python.integration.amr.test_public_accepted_halo_substep_growth import build_growth
    case,layout=build_growth()
    plan=pops.resolve(pops.validate(case),layout=layout)
    data=layout.execution.runtime_execution_data()
    assert data['schema_version']==3
    assert data['mode']=='subcycled'
    assert data['relations'][0]['temporal_ratio']=={'numerator':5,'denominator':2}
    assert plan is not None
