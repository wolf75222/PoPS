"""Multi-StateSpace wave speeds belong to the selected transport State."""

import pops
import pytest

from pops import math
from pops.codegen.module_lowering import lower_and_validate
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.model import Module
from pops._ir.expr import Const


@pytest.mark.parametrize("species_order", (("A", "B"), ("B", "A")))
@pytest.mark.parametrize("route_order", (("A", "B", "A"), ("B", "A", "A")))
@pytest.mark.parametrize("with_waves", (False, True))
def test_only_selected_transport_state_receives_its_waves(
        species_order, route_order, with_waves):
    frame = Rectangle("selected_waves_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("selected_waves", frame=frame)
    states = {name: model.species(name, state=(name.lower(),)) for name in species_order}
    b = states["B"]
    flux = model.flux(
        "transport", frame=frame, state=b,
        components={axis: (b[0],) for axis in frame.axes},
        waves={axis: (b[0],) for axis in frame.axes} if with_waves else None,
    )
    model.rate("transport", equation=math.ddt(b) == -math.div(flux))
    source_hash = model.module.module_hash()

    for selected in route_order:
        carrier, module = lower_and_validate(model, state_space=selected)
        assert module is model.module
        assert tuple(carrier.cons_names) == (selected.lower(),)
        if selected == "A":
            assert not carrier._m._flux
            assert not carrier._m._eig
        else:
            assert carrier._m._flux
            assert bool(carrier._m._eig) == with_waves
    assert model.module.module_hash() == source_hash
    assert not model._dsl._m._eig


def test_single_state_orphan_wave_declaration_remains_invalid():
    module = Module("orphan_wave")
    module.state_space("fluid", ("rho",))
    module.eigenvalues(x=(Const(1.),), y=(Const(1.),))
    with pytest.raises(ValueError, match="set_eigenvalues must cover the exact set_flux axis set"):
        lower_and_validate(module)


def test_selected_transport_flux_does_not_silently_inherit_foreign_waves():
    frame = Rectangle("two_flux_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("two_fluxes", frame=frame)
    a = model.species("A", state=("a",))
    b = model.species("B", state=("b",))
    flux_a = model.flux("flux_a", frame=frame, state=a,
                        components={axis: (a[0],) for axis in frame.axes})
    flux_b = model.flux("flux_b", frame=frame, state=b,
                        components={axis: (b[0],) for axis in frame.axes},
                        waves={axis: (b[0],) for axis in frame.axes})
    model.rate("advance_a", equation=math.ddt(a) == -math.div(flux_a))
    model.rate("advance_b", equation=math.ddt(b) == -math.div(flux_b))

    with pytest.raises(ValueError, match="foreign qualified quantity"):
        lower_and_validate(model, state_space="A")
