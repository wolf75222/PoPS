"""A case module written to disk, for identity probes across paths and comments (N11 lot 2a).

``resolve_written_case`` writes one small transport case to ``directory`` and resolves it, so a
test can load the same source from two checkouts, with comment lines inserted, or with one constant
changed, and compare identities.
"""

import importlib.util

import pops


_CASE_SOURCE = '''
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative
from pops.projection import ConservativeCellAverage
{comment}

def build():
    frame = Rectangle("box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    flux = model.flux("F", frame=frame, state=state,
                      components={{frame.x: (q,), frame.y: (0 * q,)}},
                      waves={{frame.x: (1,), frame.y: (0,)}})
    velocity = model.vector("a", frame=frame, components={{frame.x: 1, frame.y: 0}})
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - {loss} * q ** 2)
    case = pops.Case("advection")
    block = case.block("tracer", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=Conservative(state),
                                      reconstruction=MUSCL(VanLeer()),
                                      riemann=ScalarUpwind(velocity=velocity)))
    case.numerics(plan, block=block)
    case.program(SSPRK2(block[state], rate=rate))
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout
'''

COMMENT = "# A comment moves every line of the case below it.\n# It is not part of any formula.\n"


def resolve_written_case(directory, *, comment="", loss="0.5"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "provenance_case.py"
    path.write_text(_CASE_SOURCE.format(comment=comment, loss=loss))
    spec = importlib.util.spec_from_file_location("provenance_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, layout = module.build()
    return pops.resolve(pops.validate(case), layout=layout), str(path)


def compile_identities(resolved):
    snapshot = resolved.snapshot
    return {
        "artifact_hash": snapshot.artifact_hash,
        "plan_identity": resolved.plan_identity.token,
        "semantic_identity": snapshot.semantic_identity.token,
    }
