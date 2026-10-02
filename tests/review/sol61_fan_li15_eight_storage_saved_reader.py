"""Pure storage layer for Uniform fixture@3; no PoPS/Native imports or approvals."""
import hashlib,json
from pathlib import Path
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode,require
from tests.review import sol61_fan_li15_eight_saved_reader as math_reader
CONTRACT='sol61.fan-li15-eight-uniform-storage-reception@1'
PHASES=('initial',)+tuple('accepted'+str(i) for i in range(1,9))

def load_json(path):
    def pairs(rows):
        d={}
        for k,v in rows:
            require(k not in d,'duplicate metadata key');d[k]=v
        return d
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite metadata')))

def audit_storage(directory,pins):
    directory=Path(directory)
    require(type(pins) is dict and bool(pins),'external pins required')
    for name,pin in pins.items():
        require(type(name) is str and Path(name).name==name and type(pin) is str and len(pin)==64,'invalid external pin')
        path=directory/name
        require(not path.is_symlink() and path.is_file(),'missing or symlink capture')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==pin,'external pin differs')
    counts=[];world=None
    for step,phase in enumerate(PHASES):
        names=(phase+'.storage.json',phase+'.npy',phase+'.clock.json')
        require(all(n in pins for n in names),'missing phase storage/valid/clock pin')
        meta=load_json(directory/names[0]);clock=load_json(directory/names[2])
        require(type(meta) is dict and set(meta)=={'contract','dimension','time','macro_step','ranks','complete','rank_local'},'storage metadata members differ')
        require(meta['contract']=='accepted-state-storage-observation@1' and type(meta['dimension']) is int and meta['dimension']==2,'storage contract/dimension differs')
        require(type(clock) is list and len(clock)==2 and type(clock[0]) is float and type(clock[1]) is int,'clock representation differs')
        require(type(meta['time']) is float and np.isfinite(meta['time']) and type(meta['macro_step']) is int and (meta['time'],meta['macro_step'])==(clock[0],clock[1]) and clock[1]==step and abs(clock[0]-step*1e-4)<1e-14,'storage point differs')
        ranks=meta['ranks'];require(type(ranks) is int and ranks>=1,'invalid rank count')
        if world is None:world=ranks
        require(ranks==world and type(meta['rank_local']) is list and len(meta['rank_local'])==ranks,'rank inventory differs')
        def blob(row,expected):
            require(type(row) is dict and set(row)=={'file','sha256'} and row['file']==expected and row['file'] in pins and row['sha256']==pins[row['file']],'storage reference/pin differs')
            return decode(np.frombuffer((directory/row['file']).read_bytes(),dtype=np.uint8))
        whole=blob(meta['complete'],phase+'.complete.carriers')
        envelope=(2,64,ranks,-1,1,['gas'])
        require(tuple(whole[k] for k in ('dim','real','ranks','shard','levels','blocks'))==envelope,'complete wire authority differs')
        for rank,row in enumerate(meta['rank_local']):
            local=blob(row,phase+'.rank'+str(rank)+'.carriers')
            require(tuple(local[k] for k in ('dim','real','ranks','shard','levels','blocks'))==(2,64,ranks,rank,1,['gas']),'local wire authority differs')
            require(local['patches']==[p for p in whole['patches'] if p['owner'] in (-1,rank)],'local/complete patch bits differ')
        values=np.load(directory/names[1],allow_pickle=False)
        require(values.dtype==np.float64 and values.shape==(15,16,16),'valid representation differs')
        coverage=np.zeros((16,16),dtype=np.int64)
        for patch in whole['patches']:
            require(patch['components']==15 and patch['key'][:2]==(0,0),'component/level differs')
            (xl,xh,xg,xG),(yl,yh,yg,yG)=patch['axes']
            require(0<=xl<=xh<16 and 0<=yl<=yh<16,'valid geometry differs')
            coverage[yl:yh+1,xl:xh+1]+=1
            bits=np.asarray(patch['bits'],dtype=np.uint64).reshape(15,yG-yg+1,xG-xg+1)
            require(np.all(np.isfinite(bits.view(np.float64))),'nonfinite grown storage')
            require(np.array_equal(bits[:,yl-yg:yh-yg+1,xl-xg:xh-xg+1],values.view(np.uint64)[:,yl:yh+1,xl:xh+1]),'valid/carrier bits differ')
        require(np.all(coverage==1),'complete coverage differs');counts.append(len(whole['patches']))
    return {'contract':CONTRACT,'phases':list(PHASES),'ranks':world,'patch_counts':counts,'native_authority':False,'root_scientific_approval':False,'ghost_formula_qualified':False,'checkpoint_restart_qualified':False}

def receive(directory,pins,order):
    storage=audit_storage(directory,pins)
    science=math_reader.receive(directory,pins,order,tuple((p+'.npy',p+'.clock.json') for p in PHASES))
    return {'storage':storage,'science':science,'native_authority':False,'root_scientific_approval':False}
