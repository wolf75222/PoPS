"""Independent bounded readers of native PoPS receipts; no test-module imports."""
import json
import struct

import numpy as np


def _decode(spec, image, *, rank=None):
    """Independent, bounded reader of the native receipt, never an equation evaluator."""
    at = 0
    def word():
        nonlocal at
        if at+8 > len(image): raise AssertionError('truncated native image')
        value = struct.unpack_from('<Q', image, at)[0]; at += 8
        return value
    def blob():
        nonlocal at
        count = word()
        if count > len(image)-at: raise AssertionError('truncated native string')
        value = image[at:at+count]; at += count
        return value
    def text(): return blob().decode()
    def bounded_count(minimum=8):
        count = word(); assert count <= (len(image)-at)//minimum
        return count
    def signed(): return struct.unpack('<q',struct.pack('<Q',word()))[0]
    magic = text(); assert magic in {'POPSCONT1','POPSCONT2'}
    identity = text(); dimension = word(); assert dimension in (1,2,3)
    publication = word()
    realms, domains, distributed_domains, distributed_arrays = [], [], [], []
    storage = None
    if magic == 'POPSCONT1':
        domains = [dict(size=word(), capacity=word(), family=word(), version=word(), alias=i)
                   for i in range(bounded_count(32))]
    else:
        for _ in range(bounded_count(24)):
            issuer = (word(),word()); replicated = word(); assert replicated in (0,1)
            realms.append(dict(issuer=issuer,replicated=bool(replicated)))
        for i in range(bounded_count(40)):
            realm = word(); assert realm < len(realms)
            domains.append(dict(realm=realms[realm],size=signed(),capacity=signed(),
                                family=signed(),version=signed(),alias=i))
        count = bounded_count(40)
        if count:
            issuer=(word(),word()); storage=dict(bytes=word(),versions=word())
            for i in range(count):
                version,family,capacity=word(),signed(),signed()
                offsets=[signed() for _ in range(bounded_count())]
                assert len(offsets)>=2 and offsets[0]==0 and all(a<=b for a,b in zip(offsets,offsets[1:]))
                assert 0<=offsets[-1]<=capacity and version>0 and family>=0
                distributed_domains.append(dict(issuer=issuer,version=version,family=family,
                    capacity=capacity,size=offsets[-1],offsets=offsets,alias=i))
    for domain in domains:
        assert 0<=domain['size']<=domain['capacity'] and domain['family']>=0 and domain['version']>0
    root = next(v for v in spec.case._time._values if v.op == 'continuation_open')
    schema = root.attrs['schema']['read']
    kinds={'bool':0,'int64':1,'scalar':2,'product':3,'variant':4,'index_set':5,'index':6,'array':7}
    def native_kind(sig):
        kind=kinds[sig['vtype']]
        return kind+4 if kind in (5,6,7) and sig.get('ownership')=='contiguous_distributed' else kind
    def shape(sig):
        name=sig['vtype']; children=tuple(shape(child) for _,child in sig.get('members',()))
        if name=='array': children=(shape(sig['element']),)
        return (native_kind(sig),tuple(key for key,_ in sig.get('members',())),
                sig.get('family',-1),sig.get('capacity',0),children)
    known_arrays={}
    def collect_types(sig):
        if sig['vtype']=='array':
            if native_kind(sig)==11: known_arrays[shape(sig)]=sig
            collect_types(sig['element'])
        for _,child in sig.get('members',()): collect_types(child)
    collect_types(schema)
    def table_signature(data):
        # The native type identity is a bounded binary structural descriptor, not UTF-8.
        cursor=0
        def u():
            nonlocal cursor
            assert cursor+8<=len(data)
            result=struct.unpack_from('<Q',data,cursor)[0];cursor+=8;return result
        def s():
            nonlocal cursor
            n=u();assert n<=len(data)-cursor
            result=data[cursor:cursor+n].decode();cursor+=n;return result
        assert s()=='pops.continuation-transport-type@2'
        def node(depth=0):
            assert depth<=128
            kind=u();n=u();assert n<=len(data)//8
            names=tuple(s() for _ in range(n))
            family=struct.unpack('<q',struct.pack('<Q',u()))[0];capacity=u()
            bundle=u();assert bundle in (0,1)
            bindings=u();assert bindings<=len(data)//16
            for _ in range(bindings):
                for _ in range(2):
                    count=u();assert count<=len(data)//8
                    for _ in range(count):u()
            count=u();assert count<=len(data)//8
            return (kind,names,family,capacity,tuple(node(depth+1) for _ in range(count)))
        key=node();assert cursor==len(data)
        assert key in known_arrays, 'Array table type absent from declared continuation'
        return known_arrays[key]
    def value(sig):
        kind = word(); name = sig['vtype']
        assert kind==native_kind(sig), (kind,name)
        if name in {'bool', 'int64', 'scalar'}:
            assert kind == {'bool':0, 'int64':1, 'scalar':2}[name]
            bits = word()
            return bool(bits) if name == 'bool' else (struct.unpack('<q',struct.pack('<Q',bits))[0]
                if name == 'int64' else struct.unpack('<d',struct.pack('<Q',bits))[0])
        if name == 'product':
            assert kind == 3 and word() == len(sig['members'])
            return {key:value(child) for key,child in sig['members']}
        if name == 'variant':
            arm=word();assert arm<len(sig['members']) and word()==1
            key,child=sig['members'][arm]
            return {'tag':key,'value':value(child)}
        if name == 'index_set':
            return (distributed_domains if kind==9 else domains)[word()]
        if name == 'index':
            domain=(distributed_domains if kind==10 else domains)[word()];offset=signed()
            assert 0<=offset<domain['size']
            return {'domain':domain,'offset':offset}
        if name == 'array':
            if kind==11:
                slot=word();assert slot<len(distributed_arrays), 'forward Array alias'
                result=distributed_arrays[slot];assert result['type_shape']==shape(sig)
                return result
            domain = domains[word()]; n = word()
            assert n == domain['size'] and n <= len(image)//8
            return {'domain':domain, 'values':[value(sig['element']) for _ in range(n)]}
        raise AssertionError('unexpected receiver type '+name)
    if magic=='POPSCONT2':
        for i in range(bounded_count(40)):
            version=word();sig=table_signature(blob());slot=word();target=word();charge=word();count=word()
            assert slot<len(distributed_domains) and target<=len(distributed_domains)
            domain=distributed_domains[slot];assert count<=len(image)//8 and charge>=32 and version>0
            assert domain['family']==sig['family'] and domain['capacity']==sig['capacity']
            sizes=np.diff(domain['offsets'])
            assert count in sizes
            if rank is not None: assert 0<=rank<len(sizes) and count==sizes[rank]
            array=dict(domain=domain,version=version,alias=i,type_shape=shape(sig),charge=charge,
                index_domain=distributed_domains[target-1] if target else None,
                values=[value(sig['element']) for _ in range(count)])
            distributed_arrays.append(array)
    out = value(schema)
    assert at == len(image)
    return {'root':out,'identity':identity,'publication':publication,'domains':domains,
            'wire':magic,'realms':realms,'distributed_domains':distributed_domains,
            'distributed_arrays':distributed_arrays,'storage':storage,'dimension':dimension}



def _image(runtime):
    return runtime._executor._s._checkpoint_continuation_image()



def _shards(payload):
    assert type(payload) is bytes
    assert payload.startswith(b"POPSFAIL1")
    cursor = len(b"POPSFAIL1")
    def word():
        nonlocal cursor
        assert len(payload)-cursor >= 8
        result = int.from_bytes(payload[cursor:cursor+8], "big")
        cursor += 8
        return result
    count = word()
    assert count <= (len(payload)-cursor)//8
    result = []
    for _ in range(count):
        size = word()
        assert size <= len(payload)-cursor
        result.append(payload[cursor:cursor+size])
        cursor += size
    assert cursor == len(payload)
    return tuple(result)



def _error_value(wire,signature):
    """Bounded independent reader of scalar/Product/Variant failure payloads."""
    cursor=0
    def word():
        nonlocal cursor
        assert cursor+8<=len(wire)
        result=struct.unpack_from('<Q',wire,cursor)[0];cursor+=8;return result
    def text():
        nonlocal cursor
        n=word();assert n<=len(wire)-cursor
        result=wire[cursor:cursor+n].decode();cursor+=n;return result
    assert text()=='POPSCONT1'
    declared=text();assert json.loads(declared)['type']==signature
    assert word()==2
    word()  # Failure images carry no accepted publication claim.
    assert word()==0  # These declared errors contain no domain values.
    def value(sig):
        kind=word();name=sig['vtype']
        if name=='product':
            assert kind==3 and word()==len(sig['members'])
            return {key:value(child) for key,child in sig['members']}
        if name=='variant':
            arm=word();assert kind==4 and arm<len(sig['members']) and word()==1
            key,child=sig['members'][arm];return {'tag':key,'value':value(child)}
        assert kind=={'bool':0,'int64':1,'scalar':2}[name]
        bits=word()
        return bool(bits) if name=='bool' else struct.unpack('<q' if name=='int64' else '<d',struct.pack('<Q',bits))[0]
    result=value(signature);assert cursor==len(wire);return result
