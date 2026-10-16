"""SSA differentiation of distributed finite language operations, never model dispatch."""
from pops.time._program.structured_control import value_signature
from pops.time.variant_types import Product, ValueType, type_of


def transform(compiler, record, xs, ds):
    from pops.graph_derivatives import tangent_type, _ref
    from pops._finite_differential import array_map, zero_like, transport_capacity
    from pops._differential_results import paired_result, _unpack_error
    from pops.time._program.distributed_arrays import Distributed
    p, op, attrs = compiler.p, record['op'], record['attrs']
    capacity = dict(attrs['$map']).get('byte_capacity')
    dist = Distributed(p)
    if op == 'distributed_import':
        return compiler.codec.inline(record), dist.import_values(ds[0],xs[1],byte_capacity=capacity)
    if op == 'distributed_local_at':
        return compiler.codec.inline(record), p.at(ds[0],xs[1])
    if op == 'distributed_requests':
        # Topology is held fixed only on an open branch. Transform its actual body:
        # comparisons on continuous inputs must retain the same AD surface guard
        # as map/fold bodies. The intermediate Int64 map has collective typed-failure
        # handling; raw Requests callbacks do not carry arbitrary user failures.
        # Reconstructing Requests afterwards retains its exact target witness even
        # for an empty domain, without fabricating a global Index on the host.
        def checked_gid(q,index):
            pair = compiler.region(record['blocks']['body_block'],
                {_ref(attrs,'body_index'):(index,q.product())},_ref(attrs,'body'))
            return q.index_value(pair['value'])
        gids = array_map(p,xs[0],checked_gid)
        value = dist.requests(xs[0],xs[1],lambda q,index:q.index(xs[1],q.at(gids,index)))
        return value, zero_like(p,value)
    if op == 'distributed_map':
        domain = xs[0]
        def map_body(q,index):
            def evaluate(r):
                pair = compiler.region(record['blocks']['body_block'],
                    {_ref(attrs,'body_index'):(index,r.product())},_ref(attrs,'body'))
                value = pair['value']
                if value.vtype == 'variant' and set(dict(value_signature(value)['members'])) == {'ok','err'}:
                    return paired_result(r,value,pair['tangent'])
                return pair
            return q.capture(evaluate)
        paired = dist.local_map(domain,map_body,byte_capacity=transport_capacity())
        original = ValueType(compiler.codec.decode(record['signature']))
        derivative = tangent_type(original)
        original_error = ValueType(dict(original.signature['members'])['err'])
        reason = ValueType(dict(original_error.signature['members'])['reason'])
        user_error = ValueType(dict(reason.signature['members'])['user'])
        def ok(q,values):
            primary = array_map(q,domain,lambda r,i:r.at(values,i)['value'])
            tangent = array_map(q,domain,lambda r,i:r.at(values,i)['tangent'])
            return q.product(value=q.variant(original,'ok',primary),
                             tangent=q.variant(derivative,'ok',tangent))
        def err(q,error):
            def runtime(r,payload):
                return r.product(value=r.variant(reason,'runtime',payload),
                    tangent=r.variant(tangent_type(reason),'runtime',zero_like(r,payload)))
            def user(r,payload):
                pair = _unpack_error(r,payload,user_error)
                return r.product(value=r.variant(reason,'user',pair['value']),
                    tangent=r.variant(tangent_type(reason),'user',pair['tangent']))
            cause = q.match(error['reason'],runtime=runtime,user=user)
            failure = q.product(rank=error['rank'],gid=error['gid'],reason=cause['value'])
            direction = q.product(rank=q.product(),gid=q.product(),reason=cause['tangent'])
            return q.product(value=q.variant(original,'err',failure),
                             tangent=q.variant(derivative,'err',direction))
        result = p.match(paired,ok=ok,err=err)
        return result['value'],result['tangent']
    if op in ('distributed_reduce','distributed_scatter_reduce','distributed_scan'):
        values = xs[0]
        paired_values = array_map(p,p.domain(values),lambda q,i:
            q.product(value=q.at(values,i),tangent=q.at(ds[0],i)))
        initial_index = 3 if op == 'distributed_scatter_reduce' else 1
        initial = p.product(value=xs[initial_index],tangent=ds[initial_index])
        def reduce_body(q,state,item):
            pair = compiler.region(record['blocks']['body_block'], {
                _ref(attrs,'body_argument'):(state['value'],state['tangent']),
                _ref(attrs,'body_value'):(item['value'],item['tangent'])},_ref(attrs,'body'))
            if op == 'distributed_scan':
                return q.product(state=q.product(value=pair['value']['state'],tangent=pair['tangent']['state']),
                    value=q.product(value=pair['value']['value'],tangent=pair['tangent']['value']))
            return pair
        kw = dict(order='global_gid_ascending',byte_capacity=transport_capacity())
        if op == 'distributed_reduce':
            result=dist.reduce(paired_values,initial,reduce_body,**kw)
            return result['value'],result['tangent']
        if op == 'distributed_scatter_reduce':
            result=dist.scatter_reduce(paired_values,xs[1],xs[2],initial,reduce_body,**kw)
            domain=xs[2]
            return (array_map(p,domain,lambda q,i:q.at(result,i)['value']),
                    array_map(p,domain,lambda q,i:q.at(result,i)['tangent']))
        result=dist.scan(paired_values,initial,reduce_body,**kw)
        domain=p.domain(values)
        return (p.product(state=result['state']['value'],values=array_map(p,domain,
                    lambda q,i:q.at(result['values'],i)['value'])),
                p.product(state=result['state']['tangent'],values=array_map(p,domain,
                    lambda q,i:q.at(result['values'],i)['tangent'])))
    raise NotImplementedError('no distributed graph derivative rule for '+op)
