"""Pure SDK20 AMR gather engineering body receiver; never imports PoPS/Native."""
import hashlib,json
from pathlib import Path
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode,require
CONTRACT='sol61.sdk20-amr-gather-object-bytes-engineering-reception@1'
BLOCKS=('Q0','Q1','forcing');WIDTHS=(1,1,3)

def json_load(path):
    def pairs(items):
        result={}
        for k,v in items:require(k not in result,'duplicate key');result[k]=v
        return result
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def reconstruct(archive,block,level):
    n=8*2**level;width=WIDTHS[block];coverage=np.zeros((n,n),dtype=np.int64)
    result=np.zeros((width,n,n),dtype=np.float64)
    rows=[p for p in archive['patches'] if p['key'][:2]==(block,level)]
    require(bool(rows),'missing block-level patches')
    for p in rows:
        require(p['components']==width,'component width differs')
        (xl,xh,xg,xG),(yl,yh,yg,yG)=p['axes']
        require(0<=xl<=xh<n and 0<=yl<=yh<n,'valid box outside domain')
        grown=np.asarray(p['bits'],dtype=np.uint64).reshape(width,yG-yg+1,xG-xg+1)
        result.view(np.uint64)[:,yl:yh+1,xl:xh+1]=grown[:,yl-yg:yh-yg+1,xl-xg:xh-xg+1]
        coverage[yl:yh+1,xl:xh+1]+=1
    require(np.all(coverage<=1),'overlapping valid boxes')
    require(np.all(coverage==1) if level==0 else np.any(coverage==0) and np.any(coverage==1),'coarse/fine coverage differs')
    return result,coverage,rows

def receive(directory,pins,policy,ranks):
    directory=Path(directory)
    require(policy in ('replicated','partitioned','empty-owner') and type(ranks) is int and ranks in (1,2),'profile policy/world differs')
    require(type(pins) is dict and bool(pins),'external pins required')
    for name,pin in pins.items():
        require(type(name) is str and Path(name).name==name and type(pin) is str,'invalid pin path')
        path=directory/name;require(path.is_file() and not path.is_symlink(),'missing/symlink export')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==pin,'external pin differs')
    canonical=None;metrics=[]
    for rank in range(ranks):
        receipt_name='receipt-rank%d.json'%rank;require(receipt_name in pins,'missing rank receipt')
        receipt=json_load(directory/receipt_name)
        require(receipt['schema']=='pops.amr-gather-object-bytes-engineering@1' and type(receipt['rank']) is int and receipt['rank']==rank and type(receipt['size']) is int and receipt['size']==ranks and receipt['policy']==policy,'rank receipt authority differs')
        before=receipt['clock_before'];after=receipt['clock_after']
        require(type(before) is list and len(before)==3 and type(before[0]) is float and np.isfinite(before[0]) and type(before[1]) is int and before[1]>=0 and json.dumps(before,sort_keys=True)==json.dumps(after,sort_keys=True),'clock/temporal relation changed')
        require(receipt['root_seal'] is False,'local receipt claims ROOT seal')
        expected_names={'initial-rank%d.carriers'%rank,'written-rank%d.carriers'%rank}|{'rank%d-block%d-level%d.npy'%(rank,b,l) for b in range(3) for l in range(2)}
        require(type(receipt['files']) is dict and set(receipt['files'])==expected_names and all(name in pins and pins[name]==digest for name,digest in receipt['files'].items()),'missing/foreign body file authority')
        initial=decode(np.frombuffer((directory/('initial-rank%d.carriers'%rank)).read_bytes(),dtype=np.uint8))
        written_bytes=(directory/('written-rank%d.carriers'%rank)).read_bytes()
        written=decode(np.frombuffer(written_bytes,dtype=np.uint8))
        for archive in (initial,written):
            require(tuple(archive[k] for k in ('dim','real','ranks','shard','levels','blocks'))==(2,64,ranks,-1,2,list(BLOCKS)),'carrier authority differs')
        if canonical is None:canonical=written_bytes
        require(written_bytes==canonical,'complete carrier differs between ranks')
        require([(p['key'],p['components'],p['owner'],p['axes']) for p in initial['patches']]==[(p['key'],p['components'],p['owner'],p['axes']) for p in written['patches']],'setter altered geometry/ownership')
        topology=[];coarse_geometry=None
        for b,name in enumerate(BLOCKS):
            for level in (0,1):
                expected,covered,rows=reconstruct(written,b,level)
                actual=np.load(directory/('rank%d-block%d-level%d.npy'%(rank,b,level)),allow_pickle=False)
                require(actual.dtype==np.float64 and actual.shape==expected.shape and actual.tobytes()==expected.tobytes(),'getter/carrier object bytes differ')
                declared=(np.arange(expected.size,dtype=np.float64).reshape(expected.shape)%31-15)/8;declared[:,::2,::2]=-0.
                require(expected[:,covered==1].tobytes()==declared[:,covered==1].tobytes(),'engineering setter valid pattern differs')
                require(np.any(expected.view(np.uint64)==np.uint64(1<<63)),'signed-zero witness absent')
                topology.append({'block':name,'level':level,'holes':int(np.sum(covered==0))})
                if level==0:
                    geometry=[(p['key'][2],p['owner'],p['axes']) for p in rows]
                    if coarse_geometry is None:coarse_geometry=geometry
                    require(geometry==coarse_geometry,'coarse block layouts differ')
        require(receipt['topology']==topology,'declared topology differs')
        owners=[owner for _,owner,_ in coarse_geometry]
        require(all(o==-1 for o in owners) if policy=='replicated' else all(o>=0 for o in owners),'coarse ownership policy differs')
        counts=[sum(owner in (-1,r) for owner in owners) for r in range(ranks)]
        require(receipt['coarse_local_boxes_by_rank']==counts,'coarse local count differs')
        if ranks==2 and policy=='empty-owner':require(0 in counts and max(counts)>0,'no empty COARSE owner')
        if ranks==2 and policy=='partitioned':require(all(c>0 for c in counts),'partitioned coarse owner absent')
        metrics.append({'rank':rank,'coarse_local_boxes':counts,'topology':topology,'clock_unchanged':True})
    return {'contract':CONTRACT,'policy':policy,'ranks':ranks,'metrics':metrics,'native_authority':False,'root_scientific_approval':False,'scope':'engineering private valid setter/public getter only; no step/rollback/Ghost formula','provenance_join_required':True}


def join_provenance(directory,pins,binary_exports,*,native_sha256,header_signature):
    """Explicit externally pinned local binary exports; never infer remote-path aliases.

    Caller joins the separate actual SDK20 build audit and whole-run before/after
    identities. This checks retained evidence correspondence, not a compiler graph.
    """
    directory=Path(directory)
    require('provenance.json' in pins,'missing provenance external pin')
    require(hashlib.sha256((directory/'provenance.json').read_bytes()).hexdigest()==pins['provenance.json'],'provenance external pin differs')
    proof=json_load(directory/'provenance.json')
    require(proof['schema']=='pops.m16-explicit-retained-provenance@2' and proof['native']['sha256']==native_sha256 and proof['root_scientific_approval'] is False,'provenance/native authority differs')
    layout=proof['layout_program']
    require(layout['target']=='amr_system' and set(layout['blocks'])==set(BLOCKS) and len(layout['blocks'])==3,'full AMR partition differs')
    require(set(binary_exports)==set(BLOCKS)|{'program'},'explicit full binary correspondence required')
    rows=proof['model_binaries'];require(len(rows)==3 and {row['block'] for row in rows}==set(BLOCKS),'model inventory differs')
    for name,pin in proof['files'].items():
        require(Path(name).name==name and name in pins and pins[name]==pin,'retained file reference differs')
        require(hashlib.sha256((directory/name).read_bytes()).hexdigest()==pin,'retained file bytes differ')
    def binary(key,expected):
        path,pin=binary_exports[key];path=Path(path)
        require(not path.is_symlink() and path.is_file() and pin==expected and hashlib.sha256(path.read_bytes()).hexdigest()==pin,'actual binary export differs')
    for row in rows:
        evidence=row['actual_source'];retained=row['retained_source']
        require(evidence['complete'] is True and evidence['status']=='available' and evidence['source_sha256']==retained['sha256']==pins[retained['file']] and evidence['binary_sha256']==row['sha256'] and evidence['header_signature']==header_signature,'complete actual model TU evidence differs')
        binary(row['block'],row['sha256'])
    require('headers='+header_signature+';' in proof['program']['abi_key'],'Program header signature differs')
    binary('program',proof['program']['sha256'])
    return {'models':list(BLOCKS),'compiler_hash_recorded_not_live_rehashed':True,'compiler_to_dso_graph_proof':False,'native_authority':False}
