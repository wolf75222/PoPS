"""A linear rotational transport law changes the upwind side across an offset box."""
import numpy as np
import pytest

import pops.surface as surface
from pops._native_selector import select_native_dimension
from pops.lib.numerics import upwind
from pops.lib.time.methods import ssprk2
from pops.surface.math import ddt, div, vector

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

CELLS = 8
DT = 0.002


def build_rotation_case():
    domain = surface.Box(x=surface.Periodic(1, 2), y=surface.Periodic(-2, -1))
    model = surface.Model("rotation_on_an_offset_box")
    tracer = model.quantity("tracer", surface.Scalar(domain))
    initial = model.data("initial", surface.Scalar(domain))
    flux = model.flux(of=tracer, value=vector(-1.5-domain.y, domain.x-1.5)*tracer)
    model.balance(ddt(tracer) == -div(flux))
    numerics = surface.Numerics(represent={tracer: surface.CellAverage()},
                                schemes={flux: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=DT),
                              initial={tracer: initial})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=(CELLS, CELLS))}), tracer, initial


def _rate(tracer, centers):
    vx = 0.5 - centers[None, :]
    vy = centers[:, None] - 0.5
    xflux = vx * np.where(vx > 0, np.roll(tracer, 1, axis=0), tracer)
    yflux = vy * np.where(vy > 0, np.roll(tracer, 1, axis=1), tracer)
    return CELLS * (xflux - np.roll(xflux, -1, axis=0) + yflux - np.roll(yflux, -1, axis=1))


def test_native_rotation_samples_tangential_points_and_switches_upwind_side(tmp_path):
    select_native_dimension(2)
    case, tracer, initial = build_rotation_case()
    artifact = surface.compile(case, target=surface.OpenMP())
    centers = (np.arange(CELLS) + 0.5) / CELLS
    profile = 0.55 + 0.18 * np.sin(2 * np.pi * centers) + 0.05 * np.cos(4 * np.pi * centers)
    data = profile[:, None] + np.sin(2 * np.pi * centers[None, :]) / 8
    first = data + DT * _rate(data, centers)
    expected = (data + first + DT * _rate(first, centers)) / 2
    result = surface.bind(artifact, data={initial: data}, directory=tmp_path).run(until=DT)
    np.testing.assert_allclose(np.asarray(result.state[tracer]), expected,
                               rtol=0, atol=8 * np.finfo(float).eps)
    assert result.steps == 1 and result.rejections == 0 and result.time == DT
