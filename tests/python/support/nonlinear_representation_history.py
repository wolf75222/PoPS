"""A solved coordinate is retained and its exact Q is executed after History reads."""
import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import DerivativeStrategy, FailRun, FixedDt, ImplicitStage, SolveUnknown


def build(*, reject_after_first=False, cells=4):
    frame = Rectangle("representation-domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("representation-equations", frame=frame)
    state = model.state("U", components=("u", "v"))
    u, v = state
    source = model.source("rate", on=state, value=(-u+v, u-2*v))
    rate = model.rate("evolution", equation=ddt(state) == source)
    mapping = model.local_transform("physical", (u+u*u, v+v*v))
    case = pops.Case("closed-source-representation")
    block = case.block("material", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, StateStorage())
    case.numerics(numerics, block=block)
    p = pops.Program("closed-source-representation",
                     representation_workspace_byte_capacity=64 << 20,
                     history_field_metadata_budget=(8 << 20, 8 << 20))
    temporal = p.state(block[state])
    seed = p.value("seed", temporal.n, at=temporal.next.point)
    request = ImplicitStage(rate, temporal.n, p.dt, accumulation=mapping).request(
        unknown=SolveUnknown("coordinates", seed), seed=seed,
        derivative=DerivativeStrategy("finite_difference"))
    raw = p.solve(request, solver=Newton(tolerance=1.e-12, max_iterations=20,
        linear_tolerance=1.e-11, linear_max_iterations=80, restart=20)).consume(action=FailRun())[0]
    physical = p.transform(raw, transform=mapping)
    p.store_history("coordinates", raw, depth=1)
    retained = p.history("coordinates", lag=1, space=raw.space,
                         block=block, state_ref=block[state])
    historical_physical = p.transform(retained, transform=mapping)
    # This reduction consumes Q(old coordinate); the native receiver checks its
    # independent physical value, including after restart. It is not a raw read.
    p.record_scalar("retained.physical.u", p.sum_component(historical_physical, 0))
    p.record_scalar("retained.physical.v", p.sum_component(historical_physical, 1))
    if reject_after_first:
        window = p.interval()
        p.guard("reject_after_history_write", physical,
                window.begin < p.float64(.005), action=FailRun())
    p.commit(temporal.next, p.value("accepted", physical, at=temporal.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
        projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)))
    x, y = np.meshgrid((np.arange(cells)+.5)/cells, (np.arange(cells)+.5)/cells)
    initial = np.stack((.8+.1*x, .4+.05*y))
    return case, layout, initial, .01


def physical_step(values, dt):
    # Independent inversion of I-dt*[[-1,1],[1,-2]]. No coordinate or Newton code.
    u, v = values
    determinant = 1+3*dt+dt*dt
    return np.stack((((1+2*dt)*u+dt*v)/determinant,
                     (dt*u+(1+dt)*v)/determinant))


def history_metadata(image):
    """Independent bounded framing reader for the accepted POPSEX06 receipt."""
    class Reader:
        def __init__(self, value):
            self.data, self.position = bytes(value), 0
        def word(self):
            end = self.position+8
            if end > len(self.data):
                raise ValueError("truncated history metadata word")
            value = int.from_bytes(self.data[self.position:end], "little")
            self.position = end
            return value
        def blob(self):
            size = self.word()
            end = self.position+size
            if end > len(self.data):
                raise ValueError("truncated history metadata blob")
            value = self.data[self.position:end]
            self.position = end
            return value
        def count(self, width):
            size = self.word()
            if size > (len(self.data)-self.position)//width:
                raise ValueError("history metadata count exceeds framing")
            return size
        def finish(self):
            if self.position != len(self.data):
                raise ValueError("trailing history metadata bytes")

    outer = Reader(image)
    if outer.word().to_bytes(8, "little") != b"POPSEX06":
        raise ValueError("expected History metadata receipt POPSEX06")
    exchanges = outer.blob()
    samples = {}
    for _ in range(outer.count(16)):
        name, data = outer.blob(), outer.blob()
        if name in samples:
            raise ValueError("duplicate history sample")
        samples[name] = data
    metadata = Reader(outer.blob())
    outer.finish()
    supports = []
    for _ in range(metadata.count(56)):
        supports.append((metadata.blob(), metadata.blob(), *(metadata.word() for _ in range(4)), metadata.blob()))
    obligations = [(metadata.blob(), metadata.blob()) for _ in range(metadata.count(16))]
    fields = {}
    for _ in range(metadata.count(16)):
        name = metadata.blob()
        if name in fields:
            raise ValueError("duplicate history Field")
        slots = []
        for _ in range(metadata.count(16)):
            support, obligation = metadata.word(), metadata.word()
            if not 1 <= support <= len(supports) or not 1 <= obligation <= len(obligations):
                raise ValueError("coordinate history lost its support or representation")
            slots.append((supports[support-1], obligations[obligation-1]))
        fields[name] = slots
    metadata.finish()
    if set(fields) != set(samples) or set(fields) != {b"coordinates"}:
        raise ValueError("History sample/metadata tables differ")
    if len(fields[b"coordinates"]) != 2 or not exchanges:
        raise ValueError("History ring depth or accepted exchange receipt differs")
    for support, obligation in fields[b"coordinates"]:
        if not support[0] or not support[1] or any(not value for value in obligation):
            raise ValueError("empty History source description")
    return fields[b"coordinates"]
