"""Small public AMR authoring fixture for tests of later phase records.

The returned value comes from ``validate -> resolve``.  Optional parameter declarations are added
to the same physical model before validation, so metadata tests never graft an unrelated schema
onto an already-resolved plan.
"""
from __future__ import annotations

import pops
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, StagePoint, TimePoint, every


def resolved_amr_plan(
    *,
    block_names=("fluid",),
    parameters=(),
    tag_parameter=None,
    cells=8,
    name="phase-record-amr",
):
    """Resolve one complete public AMR case with one conservative state per block."""
    names = tuple(block_names)
    if not names or any(not isinstance(block, str) or not block for block in names):
        raise TypeError("block_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("block_names contains a duplicate")
    if type(cells) is not int or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    declarations = tuple(parameters)
    if tag_parameter is not None and (
            not isinstance(tag_parameter, str) or not tag_parameter):
        raise TypeError("tag_parameter must be a non-empty parameter name or None")

    frame = Rectangle(name + "-domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    case = pops.Case(name + "-case-" + "-".join(names))
    model = Model(name + "-model", frame=frame)
    parameter_handles = {}
    for declaration in declarations:
        handle = model.param(declaration)
        if handle.local_id in parameter_handles:
            raise ValueError("parameters contains duplicate name %r" % handle.local_id)
        parameter_handles[handle.local_id] = handle
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.2 * rho,), y_axis: (-0.1 * rho,)},
        waves={x_axis: (0.2,), y_axis: (-0.1,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    rows = []
    for block_name in names:
        block = case.block(block_name, model, states=(state,))
        instance = block[state]

        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(numerics, block=block)
        rows.append((block_name, block, instance, rate))

    program = pops.Program(name + "-program")
    endpoints = []
    for block_name, _block, instance, rate in rows:
        temporal = program.state(instance)
        stage = StagePoint(block_name + "_stage", {"main": TimePoint(program.clock, 0)})
        rhs = program.value(block_name + "_rhs", rate(temporal.n), at=stage)
        next_value = program.value(
            block_name + "_next",
            temporal.n + program.dt * rhs,
            at=temporal.next.point,
        )
        endpoints.append((temporal.next, next_value))
    for endpoint, value in endpoints:
        program.commit(endpoint, value)
    program.step_strategy(FixedDt(1.0e-3))
    case.program(program)

    transfer = AMRTransfer()
    for index, (_block_name, _block, instance, _rate) in enumerate(rows):
        case.initials.add(InitialCondition(
            state=instance,
            value=Constant((1.0 + 0.1 * index,)),
            projection=ConservativeCellAverage(),
        ))
        transfer.state(instance, StateTransfer())

    if tag_parameter is None:
        from pops.params import RuntimeParam

        threshold = case.param(RuntimeParam(name + "-refine", default=0.5))
    else:
        try:
            declaration_handle = parameter_handles[tag_parameter]
        except KeyError:
            raise ValueError(
                "tag_parameter %r is not present in parameters" % tag_parameter
            ) from None
        threshold = rows[0][1][declaration_handle]

    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(cells, cells),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(rows[0][2]) > ValueExpr(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
    )


__all__ = ["resolved_amr_plan"]
