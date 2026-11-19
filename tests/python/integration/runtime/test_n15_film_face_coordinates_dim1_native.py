"""A periodic kinematic film law samples its thickness-dependent wave speed at a face."""
import numpy as np
import pytest

import pops.surface as surface
from pops._native_selector import select_native_dimension
from pops.lib.numerics import rusanov
from pops.lib.time.methods import ssprk2
from pops.surface.math import cos, ddt, div, pi

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CELLS = 16
DT = 0.002


def build_film_case():
    domain = surface.Box(x=surface.Periodic(1, 2))
    model = surface.Model("periodic_kinematic_film")
    thickness = model.quantity("thickness", surface.Scalar(domain))
    initial = model.data("initial", surface.Scalar(domain))
    flux = model.flux(of=thickness, value=(1 + cos(2*pi*(domain.x-1))/4)*thickness**3)
    model.balance(ddt(thickness) == -div(flux))
    numerics = surface.Numerics(represent={thickness: surface.CellAverage()},
                                schemes={flux: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=DT),
                              initial={thickness: initial})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)}), thickness, initial


def _rate(thickness):
    upstream = np.roll(thickness, 1)
    mobility = 1 + np.cos(2 * np.pi * np.arange(CELLS) / CELLS) / 4
    radius = 3 * mobility * np.maximum(upstream**2, thickness**2)
    flux = (mobility * (upstream**3 + thickness**3) - radius * (thickness - upstream)) / 2
    return CELLS * (flux - np.roll(flux, -1))


def test_native_periodic_film_samples_the_same_point_in_both_traces(tmp_path):
    select_native_dimension(1)
    case, thickness, initial = build_film_case()
    artifact = surface.compile(case, target=surface.OpenMP())
    centers = (np.arange(CELLS) + 0.5) / CELLS
    data = 0.55 + 0.18 * np.sin(2 * np.pi * centers) + 0.05 * np.cos(4 * np.pi * centers)
    first = data + DT * _rate(data)
    expected = (data + first + DT * _rate(first)) / 2
    result = surface.bind(artifact, data={initial: data}, directory=tmp_path).run(until=DT)
    np.testing.assert_allclose(np.asarray(result.state[thickness]), expected,
                               rtol=0, atol=8 * np.finfo(float).eps)
    assert result.steps == 1 and result.rejections == 0 and result.time == DT
