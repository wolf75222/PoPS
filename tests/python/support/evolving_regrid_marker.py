"""A real independent diagnostic state that grows to request explicit new AMR tags."""
import pops
from pops.initial import InitialCondition
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage


def add_growing_marker(case, program, frame, *, initial, growth, name="regrid_marker"):
    """Preserve the tested PDE while adding the declared ODE m'=growth, zero flux."""
    model = pops.Model(name, frame=frame)
    state = model.state("U", components=("marker",))
    marker, = state
    flux = model.flux("zero", state=state, frame=frame,
                     components={axis: (0. * marker,) for axis in frame.axes},
                     waves={axis: (0.,) for axis in frame.axes})
    source = model.source("diagnostic_growth", on=state, value=(growth + 0. * marker,))
    rate = model.rate("diagnostic_rate", equation=ddt(state) == -div(flux) + source)
    growth_operator = model.module.operator_handle("diagnostic_growth")
    block = case.block(name, model, states=(state,))
    subject = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                       reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    temporal = program.state(subject)
    accepted = program.value("growing_regrid_marker",
                             temporal.n + program.dt * program.source(growth_operator, temporal.n),
                             at=temporal.next.point)
    program.commit(temporal.next, accepted)
    case.initials.add(InitialCondition(state=subject, value=initial,
                                     projection=ConservativeCellAverage()))
    return subject
