"""Two public scalar flux graphs that need the point of face evaluation."""

import pytest

import pops.surface as surface
from pops.lib.numerics import rusanov, upwind
from pops.lib.time.methods import ssprk2
from pops.surface.math import cos, ddt, div, pi, vector


def _case(domain, law, scheme, cells):
    model = surface.Model("scalar_conservation")
    u = model.quantity("u", surface.Scalar(domain))
    initial = model.data("initial", surface.Scalar(domain))
    flux = model.flux(of=u, value=law(u))
    model.balance(ddt(u) == -div(flux))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: scheme()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=0.002),
                              initial={u: initial})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=cells)})


@pytest.mark.parametrize("spatial", (False, True))
def test_rotation_lowers_the_point_of_the_signed_flux(spatial):
    domain = surface.Box(x=surface.Periodic(1, 2), y=surface.Periodic(-2, -1))
    x, y = domain.axes
    if spatial:
        law = lambda u: vector(-1.5 - y, x - 1.5) * u
    else:
        law = lambda u: vector(-0.25, 0.25) * u
    _case(domain, law, upwind, (8, 8))


@pytest.mark.parametrize("spatial", (False, True))
def test_kinematic_film_lowers_the_point_of_the_state_dependent_flux(spatial):
    domain = surface.Box(x=surface.Periodic(1, 2))
    x, = domain.axes
    if spatial:
        law = lambda h: (1 + cos(2 * pi * (x - 1)) / 4) * h**3
    else:
        law = lambda h: 1.25 * h**3
    _case(domain, law, rusanov, (16,))
