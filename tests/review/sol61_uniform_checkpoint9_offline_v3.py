"""Pure State-only CP9 archive audit @3, exact named storage join. Pins are external; never ROOT approval."""
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


def valid_block_join(meta,image,expected):
    """NPY indices follow declared capture order; carrier indices follow native registry."""
    names=image['blocks']
    require(len(set(names))==len(names) and set(names)==set(expected),'authored partition differs')
    require(meta['blocks']==list(expected) and meta['expected_blocks']==list(expected),'valid block authority differs')
    return tuple((saved_index,names.index(name),name) for saved_index,name in enumerate(expected))


def receive(directory,pins):
    root=Path(directory)
    require(type(pins) is dict and pins,'external pins required')
    for name,digest in pins.items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,'pin path differs')
        require(type(digest) is str and len(digest)==64 and not (root/name).is_symlink() and (root/name).is_file(),'pin shape/file differs')
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'external pin differs')
    def load(name):
        require(name in pins,'unpinned evidence');return strict_json(root/name)
    receipt=load('receipt.json')
    require(receipt['schema']=='pops.uniform-state-carrier-checkpoint-native-fixture@2','fixture schema differs')
    profile=receipt['profile'];require(profile in ('fanli15','two-transports'),'profile differs')
    cp,final=(4,8) if profile=='fanli15' else (1,2)
    require(type(receipt['ranks']) is int and receipt['ranks']>0,'ranks differ')
    require(type(receipt['checkpoint_step']) is int and receipt['checkpoint_step']==cp and type(receipt['final_step']) is int and receipt['final_step']==final,'step profile differs')
    for name,digest in receipt['files'].items():
        require(name in pins and pins[name]==digest,'receipt closure differs')
    preparation=load('preparation.json')
    require(preparation['schema']=='pops.uniform-state-carrier-checkpoint-preparation@2' and preparation['profile']==profile and preparation['before_bind'] is True and preparation['root_approval'] is False,'preparation differs')
    provenance=load('provenance/receipt.json')
    require(provenance['schema']=='pops.uniform-state-checkpoint-provenance@2' and provenance['before_bind'] is True and provenance['root_approval'] is False,'provenance scope differs')
    require(receipt['root_approval'] is False and receipt['field_history_fullgrown_qualified'] is False and receipt['ghost_formula_qualified'] is False,'scope differs')
    phases=receipt['phases'];require(type(phases) is list and len(phases)==len(set(phases)),'phase duplicates')
    required={'initial','accepted'+str(cp),'reloaded','full-restored-after-legacy','legacy-valid-only-restored'} | {'accepted'+str(k) for k in range(1,cp+1)} | {'continuous'+str(k) for k in range(cp+1,final+1)} | {'replay'+str(k) for k in range(cp+1,final+1)} | {side+'-'+attack for side in ('before','after') for attack in ('truncated-carrier','projection-contradiction','legacy8')}
    require(set(phases)==required,'phase coverage differs')
    expected={'gas':15} if profile=='fanli15' else {'early_transport':2,'late_transport':3}
    extent=16 if profile=='fanli15' else 8
    require(provenance['blocks']==list(expected) and len(provenance['models'])==len(expected),'all-model provenance partition differs')
    def retained(row):
        require(type(row) is dict and set(row)=={'path','sha256'},'retained pin differs')
        matches=[name for name in pins if row['path'].endswith('/'+name)]
        require(len(matches)==1 and pins[matches[0]]==row['sha256'],'retained/external join differs')
        return row['sha256']
    retained(provenance['manifest'])
    for name,model in zip(expected,provenance['models']):
        proof=model['actual_source'];require(model['component']=='block-'+name and proof['complete'] is True,'C25 complete Model differs')
        require(retained(model['DSO'])==proof['binary_sha256'] and retained(model['cpp'])==proof['source_sha256'],'C25 retained hash differs')
        retained(model['sidecar']);retained(model['module_ir'])
        for companion in model['companions'].values():retained(companion)
    program=provenance['program'];require(program['block_names']==list(expected),'Program partition differs')
    for key in ('DSO','sidecar','cpp','ir'):retained(program[key])
    for companion in program['companions'].values():retained(companion)
    images={};negative_zero=False
    for phase in phases:
        require(type(phase) is str and Path(phase).name==phase,'phase path differs')
        filename=phase+'.carriers';require(filename in pins,'carrier not pinned')
        raw=(root/filename).read_bytes();image=decode(np.frombuffer(raw,dtype=np.uint8).copy())
        require(image['dim']==2 and image['real']==64 and image['shard']==-1 and image['levels']==1 and image['ranks']==receipt['ranks'] and image['patches'],'complete authority differs')
        require(len(set(image['blocks']))==len(image['blocks']) and set(image['blocks'])==set(expected),'authored partition differs')
        require(set(p['key'][0] for p in image['patches'])==set(range(len(image['blocks']))),'complete block coverage differs')
        clock=load(phase+'.clock.json')
        require(type(clock) is list and len(clock)==2 and type(clock[0]) is float and math.isfinite(clock[0]) and clock[0]>=0 and type(clock[1]) is int and clock[1]>=0,'clock types differ')
        negative_zero |= any(0x8000000000000000 in p['bits'] for p in image['patches'])
        for rank in range(receipt['ranks']):
            meta=load(phase+'.rank%d.phase.json'%rank)
            require(meta['schema']=='pops.uniform-state-checkpoint-phase@2' and meta['contract']=='accepted-state-storage-observation@1' and meta['capture_complete'] is True,'phase metadata differs')
            require(type(meta['rank']) is int and meta['rank']==rank and type(meta['ranks']) is int and meta['ranks']==receipt['ranks'] and type(meta['dimension']) is int and meta['dimension']==2,'phase rank authority differs')
            require(type(meta['time']) is float and type(meta['macro_step']) is int and [meta['time'],meta['macro_step']]==clock and meta['runtime_clock']==clock,'phase point differs')
            localname=phase+'.rank%d.carriers'%rank;wholename=phase+'.rank%d.complete.carriers'%rank
            require(localname in pins and wholename in pins,'missing shard pin')
            require((root/wholename).read_bytes()==raw,'all-rank complete differs')
            local=decode(np.frombuffer((root/localname).read_bytes(),dtype=np.uint8).copy())
            require(tuple(local[k] for k in ('dim','real','ranks','levels','blocks'))==tuple(image[k] for k in ('dim','real','ranks','levels','blocks')) and local['shard']==rank,'local authority differs')
            require(local['patches']==[p for p in image['patches'] if p['owner'] in (-1,rank)],'local grown bits differ')
            for saved_index,index,name in valid_block_join(meta,image,expected):
                filename=phase+'.rank%d.block%d.npy'%(rank,saved_index);require(filename in pins,'valid not pinned')
                values=np.load(root/filename,allow_pickle=False)
                require(values.dtype==np.float64 and values.shape==(expected[name],extent,extent),'valid dtype/rank differs')
                coverage=np.zeros(values.shape[1:],dtype=np.int64)
                for patch in image['patches']:
                    if patch['key'][0]!=index:continue
                    axes=patch['axes'];shape=tuple(a[3]-a[2]+1 for a in reversed(axes))
                    bits=np.asarray(patch['bits'],dtype=np.uint64).reshape((patch['components'],)+shape)
                    require(patch['components']==values.shape[0] and all(0<=a[0]<=a[1]<values.shape[-1-axis] for axis,a in enumerate(axes)),'valid geometry differs')
                    source=(slice(None),)+tuple(slice(a[0]-a[2],a[1]-a[2]+1) for a in reversed(axes))
                    target=(slice(None),)+tuple(slice(a[0],a[1]+1) for a in reversed(axes))
                    require(np.array_equal(bits[source],values.view(np.uint64)[target]),'valid/carrier contradiction')
                    coverage[target[1:]]+=1
                require(np.all(coverage==1),'valid coverage differs')
        images[phase]=(raw,clock)
    def equal(a,b):require(a in images and b in images and images[a]==images[b],'full storage/clock differs')
    equal('accepted'+str(cp),'reloaded');equal('accepted'+str(cp),'full-restored-after-legacy')
    for step in range(cp+1,final+1):equal('continuous'+str(step),'replay'+str(step))
    require('accepted.npz' in pins,'checkpoint not pinned')
    with np.load(root/'accepted.npz',allow_pickle=False) as archive:
        version=archive['pops_checkpoint_version'];carrier=archive['state_carriers_checkpoint']
        require(version.shape==() and version.dtype.kind in 'iu' and int(version)==9,'checkpoint version differs')
        for index,name in enumerate(expected):
            values=archive['state_'+name];saved=np.load(root/('accepted%d.rank0.block%d.npy'%(cp,index)),allow_pickle=False)
            require(values.dtype==saved.dtype and values.shape==saved.shape and values.tobytes()==saved.tobytes(),'checkpoint valid projection differs')
        require(carrier.dtype==np.uint8 and carrier.ndim==1 and carrier.tobytes()==images['accepted'+str(cp)][0],'checkpoint full state differs')
    for attack in ('truncated-carrier','projection-contradiction','legacy8'):
        equal('before-'+attack,'after-'+attack)
        failures=load(attack+'.failure.json')
        require(type(failures) is list and len(failures)==receipt['ranks'] and all(type(f) is list and len(f)==3 and type(f[0]) is str and type(f[1]) is str and type(f[2]) is bool for f in failures),'all-rank refusal differs')
    return {'schema':'sol61.uniform-checkpoint9-offline@3','profile':profile,'phases':len(images),'global_state_storage_exact':True,'signed_zero_observed':negative_zero,'rank_local_join_qualified':True,'field_history_cache_qualified':False,'ghost_formula_qualified':False,'root_approval':False}
