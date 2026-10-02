"""Public evolving periodic two-component accepted-halo reception; tests only."""
import math
import pops
from pops.amr import AcceptedHaloPreparation, AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy
from pops.analytic import cos, x
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import BergerRigoutsos, ConservativeInjection, CoarseFineInjection, StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every

FRAME = Rectangle("periodic evolving marker", (0., 0.), (1., 1.)).frame(Cartesian2D())
THRESHOLD = .7
DT = 1/64


def build(subcycled=False):
    from fractions import Fraction
    from pops.amr import AMRClockRelation, AMRRemainderPolicy
    shape, tag_buffer, transfer_kind = (8, 8), 0, "linear"
    order = (1,0) if subcycled else (0,1)
    MODEL = pops.Model("evolving generic transport", frame=FRAME)
    STATE = MODEL.state("U", components=tuple(("constant","marker")[i] for i in order), sampling="cell_average")
    FLUX = MODEL.flux("zero", frame=FRAME, state=STATE,
        components={axis:tuple(0*v for v in STATE) for axis in FRAME.axes},
        waves={axis:tuple(0*v for v in STATE) for axis in FRAME.axes})
    SOURCE = MODEL.source("physical uniform drive", on=STATE,
        value=tuple(0*STATE[j] + i for j,i in enumerate(order)))
    RATE = MODEL.rate("linear physical evolution", equation=ddt(STATE) == -div(FLUX)+SOURCE)
    case = pops.Case("evolving accepted halo and public restart")
    block = case.block("marker", MODEL)
    numerics = DiscretizationPlan()
    numerics.rates.add(RATE, FiniteVolume(flux=FLUX, variables=variables.Conservative(STATE),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = pops.Program("evolving forward Euler")
    value = program.state(block[STATE])
    accepted = program.value("evolving accepted", value.n + program.dt*RATE(value.n), at=value.next.point)
    program.commit(value.next, accepted)
    program.store_history("accepted evolving image", accepted, depth=1)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    marker = cos(2*math.pi*x(FRAME))
    case.initials.add(InitialCondition(state=block[STATE], value=Analytic(frame=FRAME,
        components=tuple((1+0*marker,marker)[i] for i in order)), projection=ConservativeCellAverage()))
    policy = StateTransfer() if transfer_kind == "linear" else StateTransfer(
        prolongation=ConservativeInjection(), coarse_fine=CoarseFineInjection())
    transfer = AMRTransfer()
    transfer.state(block[STATE], policy)
    threshold = case.param(RuntimeParam("tag-threshold", default=THRESHOLD))
    layout = AMR(grid=CartesianGrid(frame=FRAME, cells=shape, periodic=PeriodicAxes(FRAME.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[STATE])["marker"] > case.value(threshold)), Buffer(cells=tag_buffer)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=(AMRExecution.subcycled((AMRClockRelation(0,1,Fraction(5,2),
            AMRRemainderPolicy.EXPLICIT_FINAL_SUBSTEP),), accepted_halo=AcceptedHaloPreparation(cells=(1,1)))
            if subcycled else AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation(cells=(1,1)))),
        clustering=BergerRigoutsos(minimum_efficiency=1., minimum_box_size=1, maximum_box_size=4))
    return case, layout
