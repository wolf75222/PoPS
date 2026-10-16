"""Finite distributed equations checked by compiled algebra before publication.

The Uniform field only carries the accepted transaction. This is an algebra and
transport reception, not a discretized physical PDE or an AMR qualification.
"""
import numpy as np

from pops.time import FixedDt
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Float64, Int64, Product, Result, Variant, type_of
from .common import CaseSpec, non_owner_metrics, state_array
from .distributed_program_checks import _all, _require_condition
from .methods_T01_T12 import _initial, _storage


def build_distributed_algebra(kind="ordered", *, ranks=2):
    if ranks not in (2, 3):
        raise ValueError("this reception requires exactly MPI2 or MPI3")
    if kind not in ("ordered", "nested", "budget", "typed_reduce", "terminal", "terminal_sparse"):
        raise ValueError("unknown distributed algebra reception")
    case, layout, model, block, state, p, q, initial = _storage(
        "distributed_algebra_v14_"+kind+"_"+str(ranks), ("value",), (1.,),
        cells=(8, 4), sampling="cell_average")
    old = q.n
    d = Distributed(p, storage_bytes=8*1024*1024,
                    storage_versions=1 if kind == "budget" else 128)
    offsets = (0, 1, 4) if ranks == 2 else (0, 1, 1, 4)
    domain = d.domain(offsets, capacity=4)
    cap = 262144
    source = p.require_success(d.local_map(domain,
        lambda b, i: b.variant(Result(Int64, Int64), "ok", d.gid(i)+1),
        byte_capacity=cap))
    order = "global_gid_ascending"
    def total(a, zero=0):
        return d.reduce(a, p.int64(zero), lambda b, s, x: s+x,
                        order=order, byte_capacity=cap)
    if kind in ("terminal", "terminal_sparse"):
        indices=d.requests(domain,domain,lambda b,i:d.index(domain,(3*d.gid(i)+1)%4))
        if kind=="terminal":
            payload=p.product(aliases=p.product(left=source,right=source),
                              domain=domain,indices=indices,source=source)
        else:
            refs=Variant(empty=Int64,refs=Product(indices=type_of(indices),
                                                left=type_of(source),right=type_of(source)))
            def sparse(b,i):
                gid=d.gid(i)
                return b.branch(gid==3,
                    lambda z:z.variant(refs,"refs",z.product(indices=indices,left=source,right=source)),
                    lambda z:z.variant(refs,"empty",gid))
            entries=p.require_success(d.local_map(domain,
                lambda b,i:b.capture(lambda z:sparse(z,i)),byte_capacity=cap))
            payload=p.product(entries=entries)
        status=p.require_success(p.variant(Result(Float64,type_of(payload)),"err",payload))
        factor=0.  # unreachable: native reception asserts rollback and exact payload instead
    elif kind == "ordered":
        target = d.domain((0, 1, 2) if ranks == 2 else (0, 1, 1, 2), capacity=2)
        # A generic map of Index payloads must be interchangeable with requests().
        indices = p.require_success(d.local_map(domain,
            lambda b, i: b.capture(lambda z: d.index(target, d.gid(i) % 2)),
            byte_capacity=cap))
        scatter = d.scatter_reduce(source, indices, target, p.int64(0),
            lambda b, s, x: s*10+x, order=order, byte_capacity=cap)
        scan = d.scan(source, p.int64(0),
            lambda b, s, x: b.product(state=s+x, value=s+x),
            order=order, byte_capacity=cap)
        decimal = d.reduce(source, p.int64(0), lambda b, s, x: s*10+x,
                           order=order, byte_capacity=cap)
        empty = d.domain(tuple(0 for _ in range(ranks+1)), capacity=1)
        empty_values = p.require_success(d.local_map(empty,
            lambda b, i: b.variant(Result(Int64, Int64), "ok", b.int64(99)), byte_capacity=cap))
        empty_indices = p.require_success(d.local_map(empty,
            lambda b, i: b.capture(lambda z: d.index(domain, z.int64(0))), byte_capacity=cap))
        empty_scatter = d.scatter_reduce(empty_values, empty_indices, domain, p.int64(2),
            lambda b, s, x: s+x, order=order, byte_capacity=cap)
        empty_csr = d.csr(empty, empty_indices, lambda b, i: b.int64(0),
                          nnz_capacity=1, target=domain)
        empty_import = d.import_csr(source, empty_csr, byte_capacity=cap)
        valid = _all(p, decimal == 1234, scan["state"] == 10,
                     total(scan["values"]) == 20, total(scatter) == 37,
                     total(empty_scatter) == 8, total(empty_import, 9) == 9)
        status = _require_condition(p, valid, 3., code=14001)
        factor = 3.
    elif kind == "nested":
        def payload(b, i):
            gid = d.gid(i)
            local = b.range_set(3, capacity=3, ownership="replicated")
            data = b.map(local, lambda z, j: gid*10+z.index_value(j))
            first = b.index(local, 0)
            # source is a complete immutable distributed value, encoded by a
            # registered version; this never pretends its local shard is global.
            return b.product(domain=local, data=data, first=first, source=source)
        nested = p.require_success(d.local_map(domain,
            lambda b, i: b.capture(lambda z: payload(z, i)), byte_capacity=cap))
        requests = d.requests(domain, domain,
            lambda b, i: d.index(domain, (3*d.gid(i)+1) % 4))
        imported = d.import_values(nested, requests, byte_capacity=cap)
        def check(b, i):
            gid = d.gid(i)
            remote = (3*gid+1) % 4
            value = d.at_local(imported, i)
            first = value["first"]
            valid = _all(b,
                b.same_domain(value["domain"], b.domain(value["data"])),
                b.same_domain(value["domain"], b.domain(first)),
                b.at(value["data"], first) == remote*10,
                b.same_domain(d.domain_of(value["source"]), domain),
                d.at_local(value["source"], i) == gid+1)
            return b.branch(valid,
                lambda z: z.variant(Result(Int64, Int64), "ok", z.int64(1)),
                lambda z: z.variant(Result(Int64, Int64), "err", gid))
        checked = p.require_success(d.local_map(domain, check, byte_capacity=cap))
        status = _require_condition(p, total(checked) == 4, 5., code=14002)
        factor = 5.
    elif kind == "budget":
        refused = d.local_map(domain,
            lambda b, i: b.variant(Result(Int64, Int64), "ok", d.gid(i)+99),
            byte_capacity=cap)
        def rejected(b, error):
            return b.match(error["reason"],
                runtime=lambda z, detail: _require_condition(z, detail["code"] == 2, 7., code=14003),
                user=lambda z, detail: _require_condition(z, z.boolean(False), 7., code=14004))
        recovered = p.match(refused,
            ok=lambda b, array: _require_condition(b, b.boolean(False), 7., code=14005),
            err=rejected)
        status = recovered*_require_condition(p, total(source) == 10, 1., code=14006)
        factor = 7.
    else:
        error = Variant(rejected_value=Product(value=Int64))
        def combine(b, s, x):
            out = b.branch(x == 3,
                lambda z: z.variant(Result(Int64, error), "err",
                    z.variant(error, "rejected_value", z.product(value=x))),
                lambda z: z.variant(Result(Int64, error), "ok", s+x))
            return b.require_success(out)
        result = p.capture(lambda b: d.reduce(source, b.int64(0), combine,
            order=order, byte_capacity=cap))
        def rejected(b, error):
            return b.match(error,
                rejected_value=lambda z, payload: _require_condition(z, payload["value"] == 3, 11., code=14007),
                runtime=lambda z, payload: _require_condition(z, z.boolean(False), 11., code=14008))
        status = p.match(result,
            ok=lambda b, value: _require_condition(b, b.boolean(False), 11., code=14009),
            err=rejected)
        factor = 11.
    p.commit(q.next, p.value("accepted_after_distributed_algebra", status*old, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    subject = block[state]
    def invariants(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-factor*initial)))
        return {"passed": bool(np.isfinite(actual).all() and error == 0.),
                "max_error": error, "expected_factor": factor,
                "mpi_ranks": ranks, "metrics_owner": True}
    return CaseSpec(case, layout, _initial(case, subject, initial), {"method": subject},
        .01, .01, invariants, "compiled finite distributed algebra "+kind+
        "; exact authority/ordered multiplicity/empty shards; Uniform transaction carrier")
