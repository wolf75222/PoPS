"""Pure State-only CP9 archive audit. Pins are external; never ROOT approval."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode,require


def strict_json(path):
    def pairs(rows):
        out={}
        for key,value in rows:
            require(key not in out,'duplicate JSON key');out[key]=value
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def receive(directory,pins):
    root=Path(directory)
    require(type(pins) is dict and pins,'external pins required')
    for name,digest in pins.items():
        require(type(name) is str and Path(name).name==name,'pin path differs')
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'external pin differs')
    def load(name):
        require(name in pins,'unpinned evidence');return strict_json(root/name)
    receipt=load('receipt.json')
    require(receipt['schema']=='pops.uniform-state-carrier-checkpoint-native-fixture@1','fixture schema differs')
    profile=receipt['profile'];require(profile in ('fanli15','two-transports'),'profile differs')
    cp,final=(4,8) if profile=='fanli15' else (1,2)
    require(type(receipt['ranks']) is int and receipt['ranks']>0,'ranks differ')
    require(type(receipt['checkpoint_step']) is int and receipt['checkpoint_step']==cp and type(receipt['final_step']) is int and receipt['final_step']==final,'step profile differs')
    for name,digest in receipt['files'].items():
        require(name in pins and pins[name]==digest,'receipt closure differs')
    require(receipt['root_approval'] is False and receipt['field_history_fullgrown_qualified'] is False and receipt['ghost_formula_qualified'] is False,'scope differs')
    phases=receipt['phases'];require(type(phases) is list and len(phases)==len(set(phases)),'phase duplicates')
    images={};negative_zero=False
    for phase in phases:
        require(type(phase) is str and Path(phase).name==phase,'phase path differs')
        filename=phase+'.carriers';require(filename in pins,'carrier not pinned')
        raw=(root/filename).read_bytes();image=decode(np.frombuffer(raw,dtype=np.uint8).copy())
        require(image['dim']==2 and image['real']==64 and image['shard']==-1 and image['levels']==1 and image['ranks']==receipt['ranks'] and image['patches'],'complete authority differs')
        require(set(p['key'][0] for p in image['patches'])==set(range(len(image['blocks']))),'complete block coverage differs')
        clock=load(phase+'.clock.json')
        require(type(clock) is list and len(clock)==2 and type(clock[0]) is float and math.isfinite(clock[0]) and clock[0]>=0 and type(clock[1]) is int and clock[1]>=0,'clock types differ')
        negative_zero |= any(0x8000000000000000 in p['bits'] for p in image['patches'])
        images[phase]=(raw,clock)
    def equal(a,b):require(a in images and b in images and images[a]==images[b],'full storage/clock differs')
    equal('accepted'+str(cp),'reloaded');equal('accepted'+str(cp),'full-restored-after-legacy')
    for step in range(cp+1,final+1):equal('continuous'+str(step),'replay'+str(step))
    require('accepted.npz' in pins,'checkpoint not pinned')
    with np.load(root/'accepted.npz',allow_pickle=False) as archive:
        version=archive['pops_checkpoint_version'];carrier=archive['state_carriers_checkpoint']
        require(version.shape==() and version.dtype.kind in 'iu' and int(version)==9,'checkpoint version differs')
        require(carrier.dtype==np.uint8 and carrier.ndim==1 and carrier.tobytes()==images['accepted'+str(cp)][0],'checkpoint full state differs')
    for attack in ('truncated-carrier','projection-contradiction','legacy8'):
        equal('before-'+attack,'after-'+attack)
        failures=load(attack+'.failure.json')
        require(type(failures) is list and len(failures)==receipt['ranks'] and all(type(f) is list and len(f)==3 and type(f[0]) is str and type(f[1]) is str and type(f[2]) is bool for f in failures),'all-rank refusal differs')
    return {'schema':'sol61.uniform-checkpoint9-offline@1','profile':profile,'phases':len(images),'global_state_storage_exact':True,'signed_zero_observed':negative_zero,'rank_local_join_qualified':False,'field_history_cache_qualified':False,'ghost_formula_qualified':False,'root_approval':False}
