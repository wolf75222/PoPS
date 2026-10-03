"""Independent Source metadata adversaries; not Native evidence."""
from copy import deepcopy
import pytest
from pops.identity import make_identity
from pops.runtime._run_manifest import RunManifest
from tests.python.integration.runtime.test_public_evolved_original_stage import authenticate_replay_continuation

def setup():
 bind=make_identity('bind',{'independent':19})
 accepted=RunManifest(bind_identity=bind,start_time=0.,start_macro_step=0,controls={'t_end':.1,'max_steps':1,'step_transaction':{},'output_mode':'current-directory'})
 previous=make_identity('run',{'prior':3});lineage=make_identity('run',{'prior-branch':4})
 rows=[{'owner_path':[],'previous_owner_run':previous.token,'previous_owner_lineage':lineage.token},{'owner_path':['alpha'],'previous_owner_run':None,'previous_owner_lineage':None},{'owner_path':['omega'],'previous_owner_run':previous.token,'previous_owner_lineage':None}]
 canonical=[{'owner_path':tuple(r['owner_path']),'previous_owner_run':None if r['previous_owner_run'] is None else previous.to_data(),'previous_owner_lineage':None if r['previous_owner_lineage'] is None else lineage.to_data()} for r in rows]
 epoch=make_identity('run',{'continuation':'checkpoint_restart_epoch@1','source_run_identity':accepted.run_identity.to_data(),'owner_authorities':canonical})
 replay=RunManifest(bind_identity=bind,start_time=.1,start_macro_step=1,controls={'t_end':.1,'max_steps':1,'step_transaction':{},'output_mode':'current-directory'},continuation_identity=epoch)
 return accepted,replay,{'contract':'pops.evolved-stage-restart-authority@1','owners':rows,'restored_source_run':accepted.run_identity.token}

def test_complete_multilayout_epoch_with_prior_lineage():
 a,r,e=setup();assert authenticate_replay_continuation(a,r,e)==r.continuation_identity

@pytest.mark.parametrize('attack',['dropped','ordering','lineage','impostor'])
def test_live_multilayout_authorities_cannot_be_replaced(attack):
 a,r,e=setup();e=deepcopy(e)
 if attack=='dropped':e['owners'].pop()
 if attack=='ordering':e['owners'][1:]=reversed(e['owners'][1:])
 if attack=='lineage':e['owners'][0]['previous_owner_lineage']=None
 if attack=='impostor':e['restored_source_run']=make_identity('run',{'impostor':2}).token
 with pytest.raises((AssertionError,ValueError)):authenticate_replay_continuation(a,r,e)
