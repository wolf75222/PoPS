"""T05 (spec section 17.9, merged into M18) in the proposed surface: a nested implicit closure, maximum entropy.

The moments of weights 1, v and v^2 of a 1x1v kinetic density are evolved. Their flux needs the next moment,
which the closure takes from the entropy ansatz g = exp(l0 + l1 v + l2 v^2) whose moments equal the evolved
ones: an implicit relation of the model with a selected branch (R16), solved per cell inside every flux
evaluation (IR18). The domain of the realized closure is the set of moments that the 24-point Gauss-Hermite
rule can represent, smaller than Realizable(mom) (R17); a state outside it fails the attempt (NotAdmissible).
"""

import pops
from pops.math import *
from pops.lib.numerics import pointwise, hll
from pops.lib.time import ssprk2, cfl_limit
from pops.lib.output import hdf5
from pops.lib.solve import newton, dense_lu

CELLS, CFL, T_END = 200, 0.4, 0.5     # each a value chosen for the gallery

space = pops.Box(x=pops.Periodic(0, 1))          # value chosen for the gallery: 1x1v on a periodic interval
velocity = pops.Box(v=(-infinity, infinity))     # only integrated over: its layout is a quadrature rule
x = space.x
v = velocity.v

model = pops.Model("moment_closure")
mom = model.quantity("mom", pops.Components(space, m0=pops.Scalar(admissible=pops.Positive()),
                                            m1=pops.Scalar(), m2=pops.Scalar()))
# lam is a constitutive unknown (Q1): the closure determines it from mom, and no balance evolves it.
lam = model.quantity("lam", pops.Components(space, l0=pops.Scalar(), l1=pops.Scalar(), l2=pops.Scalar()))


# The basis 1, v, v^2: value chosen for the gallery.
def basis(v):
    """The weights of the evolved moments, one field of mom each (value chosen for the gallery)."""
    return pops.record(m0=1, m1=v, m2=v**2)


g = exp(lam.l0 + lam.l1 * v + lam.l2 * v**2)     # the entropy ansatz; it lives on space x velocity, undeclared
closure = model.relation(integral(basis(v) * g, over=velocity) == mom, unknowns=lam,
                         select=pops.Realizable(mom))
flux = model.flux(of=mom, value=integral(basis(v) * v * g, over=velocity))
model.balance(ddt(mom) == -div(flux))
# With three moments the ansatz is a Gaussian, so the closed system is 1-D Euler with gamma = 3 and its wave
# speeds are u - c, u and u + c with c = sqrt(3 theta). They bound the speeds that HLL and cfl_limit read.
u = mom.m1 / mom.m0
theta = mom.m2 / mom.m0 - u**2
model.wave_speeds(flux, bounds=(u - sqrt(3 * theta), u + sqrt(3 * theta)))

numerics = pops.Numerics(
    represent={mom: pops.CellAverage(), lam: pops.PointValue()},
    schemes={closure: pointwise(), flux: hll()},     # each integral over velocity uses the rule of its layout
)

# Temperature 1/2 makes the initial ansatz proportional to exp(-v^2), the weight of the Gauss-Hermite rule.
# lam starts at the Gaussian of these moments, the first iterate of the closure; a point value is sampled exactly,
# so it is written alone.
density = 1 + sin(2 * pi * x) / 2                # value chosen for the gallery
program = pops.Program(
    advance=pops.Advance(ssprk2(rate=model), dt=cfl_limit(flux, cfl=CFL),
                         tolerance=pops.Tolerance(rtol=1e-10, atol=1e-12)),     # value chosen for the gallery
    initial={mom: pops.project(pops.record(m0=density, m1=0, m2=density / 2), quadrature=pops.GaussLegendre(2)),
             lam: pops.record(l0=log(density / sqrt(pi)), l1=0, l2=-1)},
    solvers={closure: newton(linear=dense_lu())},     # the Jacobian is exact unless stated (SU9)
)
# The output times are a value chosen for the gallery.
outputs = [pops.Output(pops.record(mom=mom), when=pops.every(time=0.1), write=hdf5("moment_closure.h5"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={space: pops.Uniform(cells=CELLS), velocity: pops.GaussHermite(24)}, outputs=outputs)
artifact = pops.compile(case, target=pops.OpenMP())     # value chosen for the gallery
simulation = pops.bind(artifact)
result = simulation.run(until=T_END)
