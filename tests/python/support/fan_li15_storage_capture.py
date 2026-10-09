"""Pure fixture integrity checks; independent wire decoder, never a runtime backend."""
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode,require


def observe_storage(runtime):
    observer=getattr(runtime,'observe_accepted_state_storage',None)
    if not callable(observer):
        raise TypeError('SDK lacks public accepted-state-storage-observation@1; no fallback')
    return observer()


def validate_storage(image,values,clock,*,rank,ranks):
    require(image.contract=='accepted-state-storage-observation@1','storage contract differs')
    require(type(image.dimension) is int and image.dimension==2,'storage dimension differs')
    require(type(image.time) is float and type(image.macro_step) is int
            and (image.time,image.macro_step)==clock,'storage clock differs')
    require(type(values) is np.ndarray and values.dtype==np.float64 and values.shape==(15,16,16),'valid image differs')
    require(type(image.rank_local) is bytes and type(image.complete) is bytes,'storage bytes differ')
    local=decode(np.frombuffer(image.rank_local,dtype=np.uint8));whole=decode(np.frombuffer(image.complete,dtype=np.uint8))
    for archive,shard in ((local,rank),(whole,-1)):
        require((archive['dim'],archive['real'],archive['ranks'],archive['shard'],archive['levels'],archive['blocks'])
                ==(2,64,ranks,shard,1,['gas']),'Uniform storage envelope differs')
    expected=[p for p in whole['patches'] if p['owner'] in (-1,rank)]
    require(local['patches']==expected,'rank-local/complete storage differs')
    cover=np.zeros((16,16),dtype=np.int64)
    for patch in whole['patches']:
        require(patch['components']==15 and patch['key'][:2]==(0,0),'storage component/level differs')
        x,y=patch['axes'];xl,xh,xg,xG=x;yl,yh,yg,yG=y
        require(0<=xl<=xh<16 and 0<=yl<=yh<16,'valid geometry outside original grid')
        cover[yl:yh+1,xl:xh+1]+=1
        bits=np.asarray(patch['bits'],dtype=np.uint64).reshape(15,yG-yg+1,xG-xg+1)
        require(np.array_equal(bits[:,yl-yg:yh-yg+1,xl-xg:xh-xg+1],values.view(np.uint64)[:,yl:yh+1,xl:xh+1]),'stored valid bits differ')
    require(np.all(cover==1),'storage complete valid coverage differs')
    return whole


def persist_storage_phase(world,directory,phase,image,save_valid):
    """Persist raw bulk observations before any integrity admission."""
    import hashlib,json
    from tests.python.support.collective_checks import collective_call
    values,clock,storage=image
    rank=0 if world is None else world.rank
    ranks=1 if world is None else world.size
    collective_call(world,lambda:(directory/(phase+'.rank'+str(rank)+'.carriers')).write_bytes(storage.rank_local))
    def global_files():
        if rank==0:
            (directory/(phase+'.complete.carriers')).write_bytes(storage.complete)
            save_valid(phase,(values,clock))
            def pin(name):return {'file':name,'sha256':hashlib.sha256((directory/name).read_bytes()).hexdigest()}
            metadata={'contract':storage.contract,'dimension':storage.dimension,'time':storage.time,
                'macro_step':storage.macro_step,'ranks':ranks,
                'complete':pin(phase+'.complete.carriers'),
                'rank_local':[pin(phase+'.rank'+str(r)+'.carriers') for r in range(ranks)]}
            (directory/(phase+'.storage.json')).write_text(json.dumps(metadata,sort_keys=True,allow_nan=False)+'\n')
    collective_call(world,global_files)


def guard_persisted_storage(world,image,on_failure):
    """Vote the unchanged integrity guard after persistence, before next run."""
    from tests.python.support.collective_checks import collective_attempt,collective_call
    values,clock,storage=image;original=None
    def check():
        nonlocal original
        try:return validate_storage(storage,values,clock,rank=0 if world is None else world.rank,ranks=1 if world is None else world.size)
        except Exception as error:
            original=error
            raise
    _,failures=collective_attempt(world,check)
    if any(failures):
        collective_call(world,lambda:on_failure(failures))
        if original is not None:
            original.add_note('storage integrity collective failures: '+repr(failures))
            raise original
        raise RuntimeError('peer storage integrity guard failed: '+repr(failures))
