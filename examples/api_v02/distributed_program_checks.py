"""Compiled distributed data checks carried to a real PoPS commit.

The Uniform field is a reception carrier, not a claim of a spatial physical model.
All source/imported element checks are authored Program algebra and execute natively.
"""
import numpy as np
import pops
from pops.time import FixedDt
from pops.time._program.distributed_arrays import Distributed
from pops.time._program.structured_control import value_signature
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, ValueType, Variant
from .common import CaseSpec, state_array, non_owner_metrics
from .methods_T01_T12 import _storage, _initial

BASE = (1 << 60)+17
Detail = Variant(even=Int64, odd=Product(identifier=Int64, score=Float64))
Element = Product(identifier=Int64, score=Float64, owner=Int64, detail=Detail)
UserError = Product(code=Int64, identifier=Int64, data=Product(gid=Int64, stamp=Float64))
CheckError = Product(code=Int64, qid=Int64, observed=Int64, expected=Int64)


def _all(p, *conditions):
    result = conditions[0]
    for condition in conditions[1:]:
        result = p.branch(result, lambda q, condition=condition: condition,
                          lambda q: q.boolean(False))
    return result


def _require_condition(p, condition, value, code=7999):
    typ = Result(Float64, Product(code=Int64))
    checked = p.branch(condition,
        lambda q: q.variant(typ, 'ok', q.float64(value)),
        lambda q: q.variant(typ, 'err', q.product(code=q.int64(code))))
    return p.require_success(checked)


def _checked_status(p, result, value):
    """Retain the map's exact error payload while tying the scalar to its success tag."""
    error_type = ValueType(dict(value_signature(result)['members'])['err'])
    typ = Result(Float64, error_type)
    projected = p.match(result,
        ok=lambda q, values: q.variant(typ, 'ok', q.float64(value)),
        err=lambda q, error: q.variant(typ, 'err', error))
    return p.require_success(projected)


def build_distributed_program(kind='roundtrip', *, ranks=2, recover=True):
    if type(ranks) is not int or ranks not in (2, 3):
        raise ValueError('this reception explicitly supports exactly two or three MPI ranks')
    if kind not in {'roundtrip', 'user_payload', 'corrupt_data'}:
        raise ValueError('unknown distributed reception '+kind)
    case, layout, model, block, state, p, q, initial = _storage(
        'distributed_'+kind+'_'+str(ranks), ('value',), (1.,), cells=(8, 4), sampling='cell_average')
    old = q.n
    d = Distributed(p)
    # Rank 1 is empty only in the 3-rank case. The remaining local sizes are asymmetric.
    source_offsets = (0, 1, 4) if ranks == 2 else (0, 1, 1, 4)
    request_offsets = (0, 2, 6) if ranks == 2 else (0, 2, 2, 6)
    row_offsets = (0, 1, 3) if ranks == 2 else (0, 1, 1, 3)
    source_domain = d.domain(source_offsets, capacity=4)
    request_domain = d.domain(request_offsets, capacity=6)
    row_domain = d.domain(row_offsets, capacity=3)
    source_result_type = Result(Element, UserError)
    def source_body(b, index):
        gid = d.gid(index)
        def normal(z):
            identifier = gid+BASE
            if kind == 'corrupt_data':
                identifier = z.branch(gid == 3, lambda t: identifier+1, lambda t: identifier)
            score = z.float64(gid)+.25
            detail = z.branch(gid % 2 == 0,
                lambda t: t.variant(Detail, 'even', gid+BASE),
                lambda t: t.variant(Detail, 'odd', t.product(identifier=gid+BASE, score=score)))
            return z.variant(source_result_type, 'ok', z.product(
                identifier=identifier, score=score, owner=d.rank(), detail=detail))
        if kind == 'user_payload':
            return b.branch(gid == 3,
                lambda z: z.variant(source_result_type, 'err', z.product(
                    code=z.int64(7002), identifier=gid+BASE,
                    data=z.product(gid=gid, stamp=z.float64(7.25)))), normal)
        return normal(b)
    source_result = d.local_map(source_domain, source_body, byte_capacity=65536)
    if kind == 'user_payload':
        def recover_source(b, failure):
            def user(z, payload):
                ok = _all(z, failure['rank'] == ranks-1, failure['gid'] == 3,
                    payload['code'] == 7002, payload['identifier'] == BASE+3,
                    payload['data']['gid'] == 3, payload['data']['stamp'] == 7.25)
                return _require_condition(z, ok, 5.)
            return b.match(failure['reason'], user=user,
                runtime=lambda z, payload: _require_condition(z, z.boolean(False), 5.))
        status = p.match(source_result,
            ok=lambda b, values: _require_condition(b, b.boolean(False), 5.), err=recover_source)
        expected_factor = 5.
    else:
        source = p.require_success(source_result)
        requests = d.requests(request_domain, source_domain,
            lambda b, index: d.index(source_domain, (3*d.gid(index)+1) % 4))
        imported = d.import_values(source, requests, byte_capacity=65536)
        incidence = d.csr(row_domain, requests, lambda b, offset: offset*2, nnz_capacity=6)
        csr_imported = d.import_csr(source, incidence, byte_capacity=65536)
        checked_type = Result(Int64, CheckError)
        def verify(b, index):
            qid = d.gid(index)
            target = (3*qid+1) % 4
            expected_id = target+BASE
            expected_score = b.float64(target)+.25
            expected_owner = b.branch(target == 0, lambda z: z.int64(0), lambda z: z.int64(ranks-1))
            left, right = d.at_local(imported, index), d.at_local(csr_imported, index)
            def detail_ok(value):
                return b.match(value['detail'],
                    even=lambda z, payload: _all(z, target % 2 == 0, payload == expected_id),
                    odd=lambda z, payload: _all(z, target % 2 == 1,
                        payload['identifier'] == expected_id, payload['score'] == expected_score))
            valid = _all(b, left['identifier'] == expected_id, right['identifier'] == expected_id,
                left['score'] == expected_score, right['score'] == expected_score,
                left['owner'] == expected_owner, right['owner'] == expected_owner,
                detail_ok(left), detail_ok(right))
            return b.branch(valid,
                lambda z: z.variant(checked_type, 'ok', expected_id),
                lambda z: z.variant(checked_type, 'err', z.product(code=z.int64(7001),
                    qid=qid, observed=left['identifier'], expected=expected_id)))
        checked = d.local_map(request_domain, verify, byte_capacity=65536)
        if kind == 'corrupt_data' and recover:
            def recover_check(b, failure):
                def user(z, payload):
                    ok = _all(z, failure['rank'] == ranks-1, failure['gid'] == 2,
                        payload['code'] == 7001, payload['qid'] == 2,
                        payload['observed'] == BASE+4, payload['expected'] == BASE+3)
                    return _require_condition(z, ok, 7.)
                return b.match(failure['reason'], user=user,
                    runtime=lambda z, payload: _require_condition(z, z.boolean(False), 7.))
            status = p.match(checked, ok=lambda b, values: _require_condition(b, b.boolean(False), 7.),
                             err=recover_check)
            expected_factor = 7.
        else:
            status = _checked_status(p, checked, 3.)
            expected_factor = 3.
    # A numerical Program dependency on the checked Result controls the actual state commit.
    p.commit(q.next, p.value('accepted_after_distributed_checks', status*old, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    subject = block[state]
    def invariants(runtime):
        actual = state_array(runtime, 'method', initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected_factor*initial)))
        return {'passed': bool(np.isfinite(actual).all() and error < 1e-14), 'max_error': error,
                'expected_factor': expected_factor, 'mpi_ranks': ranks, 'metrics_owner': True}
    return CaseSpec(case, layout, _initial(case, subject, initial), {'method': subject}, .01, .01,
        invariants, 'compiled distributed map/request/import/CSR checks '+kind+'; no distributed reduction/AMR claim')
