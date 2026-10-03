"""Independent explicit allocation admission; Source only, no Native receipt."""
import copy
import pytest
from tests.python.support.evolved_stage_v_capture import validate_initial_envelope

def authority():
 return {'blocks':['Q0','Q1','forcing'],'components':{'Q0':1,'Q1':1,'forcing':2},'levels':2,'boxes':{'0':[((0,1),(0,1))],'1':[((0,1),(0,1))]},'shape':[2,2]}

def image():
 return dict(dim=2,real=64,shard=-1,ranks=2,levels=2,blocks=['Q0','Q1','forcing'],patches=[dict(key=(b,l,0),components=2 if b==2 else 1,owner=0,axes=[(0,1,-1,2)]*2,bits=(0,)*(32 if b==2 else 16)) for b in range(3) for l in range(2)])

def test_valid_independent_complete_authority():validate_initial_envelope(image(),2,authority=authority())

def test_complete_global_empty_rows_must_refuse():
 blob=image();blob['patches']=[]
 with pytest.raises(ValueError):validate_initial_envelope(blob,2,authority=authority())

def test_width_two_registry_omission_must_refuse():
 blob=image();blob['blocks']=['Q0','forcing'];blob['patches']=[p for p in blob['patches'] if p['key'][0]<2]
 with pytest.raises(ValueError):validate_initial_envelope(blob,2,authority=authority())

@pytest.mark.parametrize('fault',('missing-level','component','patch-index','geometry','shard','world'))
def test_independent_full_authority_mutants(fault):
 blob=image()
 if fault=='missing-level':blob['patches'].pop()
 if fault=='component':blob['patches'][0]['components']=3
 if fault=='patch-index':blob['patches'][0]['key']=(0,0,9)
 if fault=='geometry':blob['patches'][0]['axes']=[(0,0,-1,2)]*2
 if fault=='shard':blob['shard']=0
 if fault=='world':blob['ranks']=1
 with pytest.raises(ValueError):validate_initial_envelope(blob,2,authority=authority())
