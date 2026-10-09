"""Closed homogeneous Q evolution on a genuine composite AMR tower."""
import struct

import numpy as np

from tests.python.support.captured_diffusion_mms import CONTROLS, FD_STEP
from tests.python.support.evolved_stage_mms import ACCEPTANCE, DIFFUSION, accumulation

DT = .01
DENSE_BYTES = 256 * 1024**2


def published_history_image(name, level, raw_slots, durations, sample, step):
    """Label the two raw slots after publication, authenticated by real POPSHID1 bytes.

    This is the bounded two-step fixture protocol, not a numerical runtime or a
    general history decoder. Store writes slot zero; publication rotates it to one.
    """
    if type(step) is not int or step not in (1, 2) or len(raw_slots) != 2:
        raise ValueError("published history fixture requires its two-step, two-slot protocol")
    header = b"POPSHID1"+struct.pack("<Q", len(name.encode()))+name.encode()+struct.pack("<qQ", level, 2)
    expected = header+b"".join(struct.pack("<QQQQ", 2,
        int.from_bytes(struct.pack("<d", start), "little"),
        int.from_bytes(struct.pack("<d", DT), "little"), 1)
        for start in (0., (step-1)*DT))
    if bytes(sample) != expected or tuple(float(value).hex() for value in durations) != (DT.hex(), DT.hex()):
        raise ValueError("published history sample differs from its exact name/level/window/slot authority")
    return {name: np.asarray(raw_slots[1]).copy(),
            name+"-previous": np.asarray(raw_slots[0]).copy(),
            "history_sample_identity_"+name: np.frombuffer(bytes(sample), dtype=np.uint8).copy()}


def closed_data(width):
    initial = np.array((.15, .25))[:width]
    target = initial.copy()
    target[0] += .01
    if width == 2:
        total = float(np.sum(accumulation(initial[:, None])))
        b = 1+.2*target[0]
        rhs = total-target[0]-target[0]**2
        target[1] = 2*rhs/(b+np.sqrt(b*b+4*1.1*rhs))
    q0, q1 = (accumulation(value[:, None])[:, 0] for value in (initial, target))
    load = (q1-q0)/DT
    if width == 2:
        load[1] = -load[0]
    return initial, target, q0, load


def build(cells, width):
    import pops
    from pops._ir.elliptic import DivCoeffGrad, Reaction
    from pops._ir.quantity import PhysicalSupport
    from pops.domain import Rectangle
    from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, bcs
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.amr import (
        AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
        Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag,
    )
    from pops.analytic import cos, x
    from pops.layouts import AMR
    from pops.lib.amr import BergerRigoutsos, StateTransfer
    from pops.lib.initial import Analytic
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.solvers import Newton
    from pops.time import EvolvedOriginalFieldRate, EvolvedOriginalFieldStage, FailRun, FixedDt, every

    frame = Rectangle("composite-Q-box", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    support = PhysicalSupport((("x", "composite-Q-box"), ("y", "composite-Q-box")))
    models = tuple(pops.Model("material-%d" % i, frame=frame) for i in range(width))
    states = tuple(model.state("Q%d" % i, components=("amount",), sampling="cell_average", support=support)
        for i, model in enumerate(models))
    temperature = tuple(models[0].field("T%d" % i) for i in range(width))
    auxiliary = models[0].field("z") if width == 2 else None
    load_model = pops.Model("prescribed-homogeneous-transfer", frame=frame)
    forcing = load_model.state("forcing", components=tuple("f%d" % i for i in range(width))+("mesh_marker",))
    # Physical Q, rates and constraints precede every temporal decision.
    q = [Reaction(temperature[0], 1+ValueExpr(temperature[0]))]
    if width == 2:
        a, b = temperature
        q = [Reaction(a, 1+ValueExpr(a))+Reaction(b, .1*ValueExpr(b)),
             Reaction(b, 1+ValueExpr(b))+Reaction(a, .2*ValueExpr(b))]
    rates = tuple(EvolvedOriginalFieldRate(spatial=sum(
        (DivCoeffGrad(temperature[j], float(DIFFUSION[i, j])) for j in range(1, width)),
        DivCoeffGrad(temperature[0], float(DIFFUSION[i, 0]))), additive=forcing[i]) for i in range(width))
    constraints = ({auxiliary: Reaction(auxiliary, 1)+Reaction(temperature[0], -.25)
        +Reaction(temperature[1], -.5) == 0} if auxiliary is not None else {})
    zero = load_model.flux("stationary", frame=frame, state=forcing,
        components={axis:tuple(0*v for v in forcing) for axis in frame.axes},
        waves={axis:tuple(0*v for v in forcing) for axis in frame.axes})
    rate = load_model.rate("load-preserved", equation=ddt(forcing) == -div(zero))
    case = pops.Case("composite-original-Q-evolution")
    blocks = tuple(case.block("Q%d" % i, model) for i, model in enumerate(models))
    load_block = case.block("forcing", load_model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=zero, variables=variables.Conservative(forcing),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=load_block)
    program = pops.Program("composite-evolved-original-stage")
    previous = tuple(program.state(block[state]) for block, state in zip(blocks, states, strict=True))
    load = program.state(load_block[forcing])
    point = previous[0].next.point
    unknowns = (*temperature, *((auxiliary,) if auxiliary is not None else ()))
    order = tuple(reversed(range(width)))
    stage = EvolvedOriginalFieldStage("original-composite-evolution", unknowns=unknowns,
        evolved_unknowns=tuple(temperature[i] for i in order), accumulation=tuple(q[i] for i in order),
        spatial_rhs=tuple(rates[i] for i in order), previous=tuple(states[i][0] for i in order),
        tau=program.temporal_tau(program.dt, at=point), constraints=constraints,
        boundaries=tuple(FieldBoundary(u, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
            for u in unknowns))
    solver = Newton(**CONTROLS, right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=DENSE_BYTES)
    field = case.field(stage.problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=FD_STEP, face_policy="Arithmetic@1"), boundaries=(), solver=solver))
    captures = {block[state]: time.n for block, state, time in zip(blocks, states, previous, strict=True)}
    captures[load_block[forcing]] = load.n
    request = field.bind_program_inputs(program=program, values=captures, at=point, solver=solver)
    observed = field.observe(program.solve(request, solver=solver).consume(action=FailRun()))
    for time in previous:
        program.commit(time.next, observed.evolved_state(target=time.next))
    for u in unknowns:
        program.store_history(u.name, observed[field[u]], depth=1, owner_block=blocks[0])
    program.commit(load.next, program.value("preserved-load", 1*load.n, at=load.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    _, _, q0, load_values = closed_data(width)
    marker = 1+.04*cos(2*np.pi*(x(frame)-.5))
    transfer = AMRTransfer()
    for block, state, value in zip(blocks, states, q0, strict=True):
        case.initials.add(InitialCondition(state=block[state], value=Analytic(frame=frame, components=(0*marker+float(value),)),
            projection=ConservativeCellAverage()))
        transfer.state(block[state], StateTransfer())
    case.initials.add(InitialCondition(state=load_block[forcing],
        value=Analytic(frame=frame, components=(*tuple(0*marker+float(value) for value in load_values), marker)), projection=ConservativeCellAverage()))
    transfer.state(load_block[forcing], StateTransfer())
    threshold = case.param(RuntimeParam("mesh-refinement-threshold", default=1.03))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(load_block[forcing])["mesh_marker"] > case.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous(), clustering=BergerRigoutsos(maximum_box_size=4))
    return case, layout


def check_saved(rows, cells, width, previous, step, initial):
    _, target, q0, load = closed_data(width)
    volumes, residuals, projection, constraints = [], [], [], []
    amounts = np.zeros(width)
    assert len(rows) == 2
    assert np.any(rows[0]["active"]) and not np.all(rows[0]["active"])
    for level, row in enumerate(rows):
        active = row["active"]
        shape = (cells*2**level,)*2
        q = np.stack([row["Q%d" % i].reshape(shape) for i in range(width)])
        t = np.stack([row["T%d" % i].reshape(shape) for i in range(width)])
        forcing = row["forcing"].reshape(width+1, *shape)
        reference = initial if previous is None else previous
        expected_previous = np.stack([reference[level]["Q%d" % i].reshape(shape) for i in range(width)])
        residuals.append(float(np.max(np.abs((q-expected_previous-DT*forcing[:width])[:, active]))))
        projection.append(float(np.max(np.abs((q-accumulation(t))[:, active]))))
        assert np.max(np.abs(t[:, active]-t[:, active][:, :1])) <= ACCEPTANCE
        assert np.max(np.abs(forcing[:width, active]-load[:, None])) <= ACCEPTANCE
        if width == 2:
            constraints.append(float(np.max(np.abs((row["z"].reshape(shape)-.25*t[0]-.5*t[1])[active]))))
        if step == 1:
            assert np.max(np.abs(t[:, active]-target[:, None])) <= ACCEPTANCE
        volume = 1./(cells*2**level)**2
        volumes.append(int(np.count_nonzero(active))*volume)
        amounts += np.sum(q[:, active], axis=1)*volume
    assert sum(volumes) == 1.
    assert max(residuals+projection+constraints) <= ACCEPTANCE
    balance = amounts-q0-step*DT*load
    assert np.max(np.abs(balance)) <= ACCEPTANCE
    if width == 2:
        assert abs(np.sum(amounts)-np.sum(q0)) <= ACCEPTANCE
    return {"homogeneous_Q_equation_linf":max(residuals), "Q_projection_linf":max(projection),
        "constraint_linf":max(constraints, default=0.), "Q_balance":balance.tolist(),
        "Q_amounts":amounts.tolist(), "composite_volume":sum(volumes)}
