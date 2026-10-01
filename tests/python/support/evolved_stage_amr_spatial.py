"""Nonconstant composite Q witness and independent strip/extrusion FV oracle.

Only public authoring constructs build the Case. Reference fluxes below use saved
arrays and actual masks, and call no PoPS spatial operator or generated helper.
"""
import numpy as np

from tests.python.support.evolved_stage_amr import (
    ACCEPTANCE, CONTROLS, DENSE_BYTES, DT, FD_STEP, closed_data,
)
from tests.python.support.evolved_stage_mms import DIFFUSION, accumulation


def build(cells=8):
    width = 2
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
    from pops.analytic import CellBounds, cos, sin, x
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
    _, _, _, load_values = closed_data(width)
    position = x(frame)
    a, b = .15+.02*cos(2*np.pi*position), .25+.015*sin(2*np.pi*position)
    profiles = (a+a*a+.1*b*b, b+b*b+.2*a*b)
    bounds = CellBounds(frame)
    lo, hi = bounds.lower(frame.x), bounds.upper(frame.x)
    transverse = bounds.upper(frame.y)-bounds.lower(frame.y)
    def primitive(position):
        k = 2*np.pi
        c, s = cos(k*position), sin(k*position)
        c2, s2 = cos(2*k*position), sin(2*k*position)
        return (
            (.15+.15**2+.02**2/2+.1*(.25**2+.015**2/2))*position
            +.02*(1+2*.15)*s/k +.02**2*s2/(4*k)
            -.1*2*.25*.015*c/k -.1*.015**2*s2/(4*k),
            (.25+.25**2+.015**2/2+.2*.15*.25)*position
            -.015*(1+2*.25)*c/k -.015**2*s2/(4*k)
            +.2*.25*.02*s/k -.2*.15*.015*c/k
            -.2*.02*.015*c2/(4*k))
    integral = tuple((right-left)*transverse for left, right in zip(primitive(lo), primitive(hi), strict=True))
    marker = 1+.04*cos(2*np.pi*(position-.5))
    marker_integral = ((hi-lo)+.04*(sin(2*np.pi*(hi-.5))-sin(2*np.pi*(lo-.5)))/(2*np.pi))*transverse
    transfer = AMRTransfer()
    for block, state, profile, exact in zip(blocks, states, profiles, integral, strict=True):
        case.initials.add(InitialCondition(state=block[state], value=Analytic(frame=frame,
            components=(profile,), cell_integrals=(exact,)), projection=ConservativeCellAverage()))
        transfer.state(block[state], StateTransfer())
    case.initials.add(InitialCondition(state=load_block[forcing], value=Analytic(frame=frame,
        components=(*tuple(0*marker+float(value) for value in load_values), marker),
        cell_integrals=(*tuple(float(value)*bounds.measure for value in load_values), marker_integral)),
        projection=ConservativeCellAverage()))
    transfer.state(load_block[forcing], StateTransfer())
    threshold = case.param(RuntimeParam("mesh-refinement-threshold", default=1.03))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(load_block[forcing])["mesh_marker"] > case.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous(), clustering=BergerRigoutsos(maximum_box_size=4))
    return case, layout



def initial_cell_average(cells):
    """Independent high-order integration of the declared trig H, not an MMS load.

    Exact CellBounds primitives are supplied to native initialization. This oracle
    evaluates H at Gauss nodes instead; 32 nodes resolve these smooth functions to
    roundoff on the witness cells. No PoPS expression evaluator is used here.
    """
    nodes, weights = np.polynomial.legendre.leggauss(32)
    x = (np.arange(cells)[:, None]+.5+.5*nodes)/cells
    temperature = np.stack((.15+.02*np.cos(2*np.pi*x), .25+.015*np.sin(2*np.pi*x)))
    return np.einsum("icg,g->ic", accumulation(temperature), weights)/2


def metric_arrays(cells, level, valid):
    """Derived measures, explicitly not values from a native volume getter.

    The fixture declares a Cartesian unit square without EB. Actual native shape
    and patch boxes must match this descriptor before these arrays are archived.
    """
    n = cells*2**level
    if valid.shape != (n, n) or valid.dtype != np.bool_:
        raise ValueError("actual native valid mask disagrees with the declared Cartesian shape")
    return {"valid":valid.copy(), "cartesian_cell_volume":np.full((n, n), 1./n**2),
        "declared_no_EB_kappa":np.ones((n, n)), "x_edges":np.arange(n+1, dtype=float)/n,
        "y_edges":np.arange(n+1, dtype=float)/n}


def strip_geometry(rows, cells):
    if len(rows) != 2:
        raise ValueError("this independent reference requires two ratio-two levels")
    native_boxes = rows[0]["native_patch_boxes"]
    if native_boxes.dtype != np.int64 or native_boxes.ndim != 2 or native_boxes.shape[1] != 5:
        raise ValueError("archived native patch boxes must retain exact integer ranked bounds")
    declared_valid = [np.zeros((cells*2**level,)*2, dtype=bool) for level in range(2)]
    for level, lx, ly, ux, uy in native_boxes:
        if level not in (0, 1) or not (0 <= lx <= ux < cells*2**level and 0 <= ly <= uy < cells*2**level):
            raise ValueError("native patch boxes leave the declared two-level Cartesian domain")
        declared_valid[level][ly:uy+1, lx:ux+1] = True
    masks = []
    for level, row in enumerate(rows):
        valid, active = row["valid"], row["active"]
        n = cells*2**level
        if valid.dtype != np.bool_ or active.dtype != np.bool_ or valid.shape != (n, n) or active.shape != (n, n):
            raise ValueError("saved actual masks must have the exact ranked Cartesian shape")
        if np.any(active & ~valid) or not np.array_equal(valid, np.broadcast_to(valid[0], valid.shape)):
            raise ValueError("reference requires actual strips spanning the full periodic y extent")
        if not np.array_equal(active, np.broadcast_to(active[0], active.shape)):
            raise ValueError("reference requires full-y composite coverage")
        np.testing.assert_array_equal(row["native_base_shape"], np.array([cells, cells], dtype=np.int64))
        np.testing.assert_array_equal(row["native_patch_boxes"], native_boxes)
        np.testing.assert_array_equal(valid, declared_valid[level])
        expected = metric_arrays(cells, level, valid)
        for key in ("cartesian_cell_volume", "declared_no_EB_kappa", "x_edges", "y_edges"):
            np.testing.assert_array_equal(row[key], expected[key])
        masks.append((valid[0], active[0]))
    covered = ~masks[0][1]
    if not np.all(masks[0][0]) or not np.any(covered) or np.all(covered):
        raise ValueError("reference requires a genuine partial coarse level")
    if not np.array_equal(masks[1][0], np.repeat(covered, 2)) or not np.array_equal(masks[1][1], masks[1][0]):
        raise ValueError("actual fine coverage must match exact ratio-two coarse coverage")
    return covered, masks[1][0]


def composite_flux(temperature, covered, fine_valid):
    """Independent periodic 1D flux reference extruded in y.

    Covered coarse T is averaged from fine T first. Quadratic interpolation at
    missing fine-cell centers uses parent offsets +/-1/4. Interface coarse face
    flux is replaced by the mean of its two fine transverse faces, identical for
    this extrusion. Both signed cross entries of D remain in every face flux.
    """
    coarse, fine = (np.asarray(value, dtype=float).copy() for value in temperature)
    width, cells = coarse.shape
    if width != 2 or fine.shape != (2, 2*cells) or covered.shape != (cells,) or fine_valid.shape != (2*cells,):
        raise ValueError("reference vector/mask shapes do not match the declared coupled quotient")
    coarse[:, covered] = fine.reshape(2, cells, 2).mean(axis=2)[:, covered]
    def fine_value(index):
        index %= 2*cells
        if fine_valid[index]:
            return fine[:, index]
        parent = index//2
        offset = -.25 if index % 2 == 0 else .25
        return (offset*(offset-1)/2*coarse[:, (parent-1) % cells]
            +(1-offset**2)*coarse[:, parent]
            +offset*(offset+1)/2*coarse[:, (parent+1) % cells])
    fine_faces = np.stack([DIFFUSION @ ((fine_value(face)-fine_value(face-1))*2*cells)
        for face in range(2*cells)], axis=1)
    coarse_faces = DIFFUSION @ ((coarse-np.roll(coarse, 1, axis=1))*cells)
    interface = covered != np.roll(covered, 1)
    coarse_faces[:, interface] = fine_faces[:, 2*np.flatnonzero(interface)]
    divergence = ((np.roll(coarse_faces, -1, axis=1)-coarse_faces)*cells,
        (np.roll(fine_faces, -1, axis=1)-fine_faces)*2*cells)
    return divergence, (coarse_faces, fine_faces), interface, coarse


def check_initial(rows, cells):
    strip_geometry(rows, cells)
    amount = np.zeros(2)
    for level, row in enumerate(rows):
        n = cells*2**level
        q = np.stack([row["Q%d" % i].reshape(n, n) for i in range(2)])
        expected = np.broadcast_to(initial_cell_average(n)[:, None, :], q.shape)
        assert np.max(np.abs((q-expected)[:, row["valid"]])) <= ACCEPTANCE
        measure = row["cartesian_cell_volume"]*row["declared_no_EB_kappa"]
        amount += np.sum(q[:, row["active"]]*measure[row["active"]], axis=1)
    expected_amount = initial_cell_average(1)[:, 0]
    assert np.max(np.abs(amount-expected_amount)) <= ACCEPTANCE
    return amount


def check_saved(rows, cells, width, previous, step, initial):
    if width != 2 or step not in (1, 2):
        raise ValueError("nonconstant fixture has two carriers and two accepted steps")
    covered, fine_valid = strip_geometry(rows, cells)
    initial_amount = check_initial(initial, cells)
    temperature, q, qprev, forcing = [], [], [], []
    constraint = 0.
    reference = initial if previous is None else previous
    for level, row in enumerate(rows):
        n = cells*2**level
        t2 = np.stack([row["T%d" % i].reshape(n, n) for i in range(2)])
        active, valid = row["active"], row["valid"]
        assert np.all(np.isfinite(t2[:, valid]))
        assert np.max(np.abs((t2-t2[:, :1, :])[:, valid])) <= ACCEPTANCE
        temperature.append(t2[:, 0])
        q.append(np.stack([row["Q%d" % i].reshape(n, n) for i in range(2)]))
        qprev.append(np.stack([reference[level]["Q%d" % i].reshape(n, n) for i in range(2)]))
        forcing.append(row["forcing"].reshape(3, n, n)[:2])
        constraint = max(constraint, float(np.max(np.abs((row["z"].reshape(n, n)-.25*t2[0]-.5*t2[1])[active]))))
    divergence, faces, interface, restricted_t = composite_flux(temperature, covered, fine_valid)
    assert np.max(np.abs(temperature[0][:, covered]-restricted_t[:, covered])) <= ACCEPTANCE
    fine_h = accumulation(temperature[1]).reshape(2, cells, 2).mean(axis=2)
    coarse_h = accumulation(restricted_t)
    coarse_q = q[0][:, 0]
    assert np.max(np.abs(coarse_q[:, covered]-fine_h[:, covered])) <= ACCEPTANCE
    jensen_gap = fine_h[:, covered]-coarse_h[:, covered]
    # Fixed pre-native discriminator, distinct from the scientific error guard.
    assert np.max(np.abs(jensen_gap)) > 1e-10
    amounts, diffusion_amounts = np.zeros(2), np.zeros(2)
    residual, projection, flux_activity = 0., 0., 0.
    references = []
    for level, row in enumerate(rows):
        n, active = cells*2**level, row["active"]
        diff = np.broadcast_to(divergence[level][:, None], q[level].shape)
        t = np.broadcast_to(temperature[level][:, None], q[level].shape)
        residual = max(residual, float(np.max(np.abs((q[level]-qprev[level]-DT*(diff+forcing[level]))[:, active]))))
        projection = max(projection, float(np.max(np.abs((q[level]-accumulation(t))[:, active]))))
        flux_activity = max(flux_activity, float(np.max(np.abs(diff[:, active]))))
        measure = row["cartesian_cell_volume"]*row["declared_no_EB_kappa"]
        amounts += np.sum(q[level][:, active]*measure[active], axis=1)
        diffusion_amounts += np.sum(diff[:, active]*measure[active], axis=1)
        references.append({"independent_divergence":diff.copy(), "independent_x_face_flux_density":faces[level].copy(),
            "independent_coarse_interface_faces":interface.copy(),
            "independent_restricted_temperature":restricted_t.copy()})
    _, _, _, load = closed_data(2)
    balance = amounts-initial_amount-step*DT*load
    assert max(residual, projection, constraint) <= ACCEPTANCE
    assert flux_activity > 1e-6  # No constant-state diffusion bypass.
    assert np.max(np.abs(diffusion_amounts)) <= ACCEPTANCE
    assert np.max(np.abs(balance)) <= ACCEPTANCE
    assert abs(np.sum(amounts-initial_amount)) <= ACCEPTANCE
    return {"original_F_composite_linf":residual, "Q_projection_active_linf":projection,
        "constraint_linf":constraint, "Q_balance":balance.tolist(), "Q_amounts":amounts.tolist(),
        "closed_diffusion_amounts":diffusion_amounts.tolist(), "flux_activity_linf":flux_activity,
        "restriction_Jensen_gap_linf":float(np.max(np.abs(jensen_gap)))}, references
