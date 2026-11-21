"""An irregular incidence evaluated with typed imports and an ordered scatter."""

import pops
from pops.algebra import Distributed
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Product, type_of

from .common import CaseSpec
from .models_M01_M14 import _domain

ROW_COUNT = 4
ROWS = (0, 0, 0, 2, 2, 2, 2, 2, 3)
COLUMNS = (0, 1, 2, 3, 4, 3, 1, 5, 2)
SIGNS = (1., 1., 1., 1., 1., -1., -1., 1., -1.)
SAMPLES = (float(1 << 53), 1., -float(1 << 53), 4., .5, 8.)


def _select(b, gid, values, literal):
    selected = literal(b, values[-1])
    for index in reversed(range(len(values) - 1)):
        selected = b.branch(gid == index,
                            lambda q, value=values[index]: literal(q, value),
                            lambda q, value=selected: value)
    return selected


def build_distributed_ragged_dm3(*, place_domains, transport):
    """Publish the local column inventory, imported occurrences and reduced rows.

    The caller supplies domain placement and transport resources at graph construction.
    """
    _, layout = _domain("ragged_dm3", (4, 4))
    case = pops.Case("distributed_ragged_dm3")
    p = pops.Program("distributed_ragged_dm3")
    d = Distributed(p)

    rows, entries, columns = place_domains(d, (ROW_COUNT, len(ROWS), len(SAMPLES)))

    def local_map(domain, body):
        return p.require_success(transport(d.local_map, domain, body))

    owned_gids = local_map(columns, lambda b, i: b.capture(lambda q: d.gid(i)))

    def source_value(b, i):
        gid = d.gid(i)
        return b.capture(lambda q: q.product(
            gid=gid, owner=d.rank(), value=_select(q, gid, SAMPLES, lambda z, x: z.float64(x))))

    source = local_map(columns, source_value)
    requests = d.requests(entries, columns, lambda b, i: d.index(
        columns, _select(b, d.gid(i), COLUMNS, lambda q, x: q.int64(x))))
    imported = transport(d.import_values, source, requests)

    def observe(b, i):
        payload = d.at_local(imported, i)
        return b.capture(lambda q: q.product(
            gid=d.gid(i), column=payload["gid"], owner=payload["owner"], value=payload["value"]))

    occurrences = local_map(entries, observe)
    weighted = local_map(entries, lambda b, i: b.capture(
        lambda q: _select(q, d.gid(i), SIGNS, lambda z, x: z.float64(x))
        * d.at_local(occurrences, i)["value"]))
    destinations = d.requests(entries, rows, lambda b, i: d.index(
        rows, _select(b, d.gid(i), ROWS, lambda q, x: q.int64(x))))
    reduced = transport(d.scatter_reduce, weighted, destinations, rows, p.float64(0.),
                        lambda b, acc, value: acc + value, order="global_gid_ascending")
    row_values = local_map(rows, lambda b, i: b.capture(
        lambda q: q.product(gid=d.gid(i), value=d.at_local(reduced, i))))

    initial = p.product(owned_gids=owned_gids, occurrences=occurrences, rows=row_values,
                        time=p.float64(0.), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    dt = p.requested_dt()
    proposal = p.product(dt=dt)
    continued = p.product(
        owned_gids=current["owned_gids"], occurrences=current["occurrences"], rows=current["rows"],
        time=current["time"] + dt, step=current["step"] + p.int64(1))
    envelope = p.product(continuation=continued, increment=p.product(),
                         observations=p.product(), checks=p.product())
    candidate = p.continuation_freeze(base, proposal, envelope)

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal),
                          candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product()))
    def assess(q, view):
        return q.product(accepted=view["candidate"]["continuation"]["step"] == 1,
                         checks=q.product())

    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    return CaseSpec(case, layout, {}, {}, .01, .01,
                    lambda runtime: {"passed": runtime.macro_step() == 1}, __doc__)
