"""Pure public Source authoring; no Native readiness inference."""
import sys
import pytest
import pops
from tests.python.support.evolving_accepted_halo_case import build

@pytest.mark.parametrize("subcycled",(False,True))
def test_public_evolving_halo_authoring(subcycled):
    case,layout=build(subcycled)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert resolved is not None
    data=layout.runtime_layout_data()["execution"]
    assert data["accepted_halo"]["cells"] == (1,1)
    if subcycled:
        assert data["relations"][0]["temporal_ratio"] == {"numerator":5,"denominator":2}
        assert data["relations"][0]["remainder_policy"] == "explicit_final_substep"
    assert not any(n=="_pops" or n.endswith("._pops") for n in sys.modules)

import copy
import numpy as np
from tests.python.support.evolving_accepted_halo_oracle import full_carrier,halo_rows,HALO_ROWS

def synthetic_archive(constant,elapsed=0.):
    values=np.zeros((2,16),dtype=np.float64);values[constant]=1.;values[1-constant]=.2+elapsed
    return dict(dim=2,real=64,ranks=1,shard=-1,levels=2,blocks=["marker"],patches=[
        dict(key=(0,l,0),owner=0,components=2,axes=[(0,1,-1,2)]*2,bits=tuple(values.view(np.uint64).ravel())) for l in (0,1)])

@pytest.mark.parametrize("constant",(0,1))
@pytest.mark.parametrize("attack",("stale-ghost","wild-ghost","axes","inventory","owner","halo-type"))
def test_full_grown_oracle_refuses_adversary(constant,attack):
    boxes=(((0,0,2,2),),((0,0,2,2),));old=synthetic_archive(constant);new=synthetic_archive(constant,.01)
    full_carrier(old,new,boxes,.01,1,constant)
    if attack=="halo-type":
        rows=copy.deepcopy(HALO_ROWS);rows[0][-1]=1
        with pytest.raises(AssertionError):halo_rows(rows)
        return
    if attack in ("stale-ghost","wild-ghost"):
        bits=np.asarray(new["patches"][1]["bits"],dtype=np.uint64).copy().view(np.float64).reshape(2,16)
        # offset0 corresponds (-1,-1): a ghost, never a valid cell.
        bits[1-constant,0]=.2 if attack=="stale-ghost" else 12345.
        new["patches"][1]["bits"]=tuple(bits.view(np.uint64).ravel())
    elif attack=="axes":new["patches"][1]["axes"]=[(0,1,0,1)]*2
    elif attack=="inventory":new["patches"].pop()
    elif attack=="owner":new["patches"][1]["owner"]=-1
    with pytest.raises(AssertionError):full_carrier(old,new,boxes,.01,1,constant)

@pytest.mark.parametrize("geometry",(((),((0,0,2,2),)),(((1,(0,0),(1,1)),),)))
def test_fine_only_or_structured_patch_boxes_cannot_replace_complete_writer_geometry(geometry):
    old=synthetic_archive(0);new=synthetic_archive(0,.01)
    with pytest.raises(AssertionError):full_carrier(old,new,geometry,.01,1,0)
