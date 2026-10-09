"""Independent POPSCAR1 and ExactContractBuilder decoder; no PoPS imports."""
import hashlib
import math
import struct
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode(raw):
    require(type(raw) is np.ndarray and raw.dtype == np.uint8 and raw.ndim == 1, "carrier typed archive differs")
    data = raw.tobytes()
    require(data[:8] == b"POPSCAR1", "carrier magic differs")
    at = 8
    def word(signed=False):
        nonlocal at
        require(at + 8 <= len(data), "carrier truncated")
        value = int.from_bytes(data[at:at+8], "little", signed=signed)
        at += 8
        return value
    dim, real, ranks, shard, levels, count = word(), word(), word(), word(True), word(), word()
    require(dim in (1, 2, 3) and real in (32, 64) and 0 < ranks <= 2**63-1 and levels > 0
            and -1 <= shard < ranks and 0 < count <= (len(data)-at)//8, "carrier authority differs")
    blocks = []
    for _ in range(count):
        length = word()
        require(0 < length <= len(data)-at, "carrier name size differs")
        name = data[at:at+length].decode("utf-8", errors="strict")
        at += length
        require(name not in blocks, "carrier duplicate block")
        blocks.append(name)
    count = word()
    require(count <= (len(data)-at)//((6+4*dim)*8), "carrier row count exceeds bytes")
    patches = []
    for _ in range(count):
        block, level, patch, components, owner = word(), word(), word(), word(), word(True)
        axes = [tuple(word(True) for _ in range(4)) for _ in range(dim)]
        size = word()
        key = (block, level, patch)
        require(block < len(blocks) and level < levels and components > 0 and -1 <= owner < ranks
                and (shard == -1 or owner in (-1, shard))
                and (not patches or patches[-1]["key"] < key), "carrier patch authority/order differs")
        require(all(glo <= lo <= hi <= ghi for lo, hi, glo, ghi in axes), "carrier geometry differs")
        require(size == components*math.prod(ghi-glo+1 for lo, hi, glo, ghi in axes)
                and size <= (len(data)-at)//8, "carrier payload shape differs")
        bits = tuple(word() for _ in range(size))
        require(real == 64 or all(v <= 2**32-1 for v in bits), "carrier scalar width differs")
        patches.append(dict(key=key, components=components, owner=owner, axes=axes, bits=bits))
    require(at == len(data), "carrier trailing bytes")
    return dict(dim=dim, real=real, ranks=ranks, shard=shard, levels=levels, blocks=blocks, patches=patches)



def reconstruct_level(archive,block,level,cells):
    rows=[p for p in archive['patches'] if p['key'][:2]==(block,level)]
    require(rows,'missing block-level storage');width=rows[0]['components']
    result=np.zeros((width,cells,cells),dtype=np.float64);coverage=np.zeros((cells,cells),dtype=np.int64)
    for p in rows:
        require(p['components']==width,'component width differs')
        (xl,xh,xgl,xgh),(yl,yh,ygl,ygh)=p['axes']
        require(0<=xl<=xh<cells and 0<=yl<=yh<cells,'out of domain valid box')
        grown=np.asarray(p['bits'],dtype=np.uint64).view(np.float64).reshape(width,ygh-ygl+1,xgh-xgl+1)
        result[:,yl:yh+1,xl:xh+1]=grown[:,yl-ygl:yh-ygl+1,xl-xgl:xh-xgl+1]
        coverage[yl:yh+1,xl:xh+1]+=1
    require(np.max(coverage)==1,'overlapping storage');return result,coverage
