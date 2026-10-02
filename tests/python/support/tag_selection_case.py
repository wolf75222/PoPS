"""Public stationary transport and periodic rectangular tag selection."""
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

FRAME = Rectangle("periodic stationary marker", (0., 0.), (1., 1.)).frame(Cartesian2D())
MODEL = pops.Model("stationary generic transport", frame=FRAME)
STATE = MODEL.state("U", components=("constant", "marker"), sampling="cell_average")
FLUX = MODEL.flux("zero", frame=FRAME, state=STATE,
    components={axis: tuple(0*v for v in STATE) for axis in FRAME.axes},
    waves={axis: tuple(0*v for v in STATE) for axis in FRAME.axes})
RATE = MODEL.rate("stationary", equation=ddt(STATE) == -div(FLUX))
THRESHOLD = .7
DT = 1/64


def build(shape, tag_buffer, transfer_kind):
    case = pops.Case("authored tag reach and numerical parent coverage")
    block = case.block("marker", MODEL)
    numerics = DiscretizationPlan()
    numerics.rates.add(RATE, FiniteVolume(flux=FLUX, variables=variables.Conservative(STATE),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = pops.Program("stationary forward Euler")
    value = program.state(block[STATE])
    accepted = program.value("stationary accepted", value.n + program.dt*RATE(value.n), at=value.next.point)
    program.commit(value.next, accepted)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    marker = cos(2*math.pi*x(FRAME))
    case.initials.add(InitialCondition(state=block[STATE], value=Analytic(frame=FRAME,
        components=(1+0*marker, marker)), projection=ConservativeCellAverage()))
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
        execution=AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation()),
        clustering=BergerRigoutsos(minimum_efficiency=1., minimum_box_size=1, maximum_box_size=4))
    return case, layout
