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
