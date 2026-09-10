"""Public full-tensor diffusion: exact FV Fourier averages, energy and conservation.

Run separately in each native dimension with POPS_NATIVE_DIM=1,2,3. Every run
uses all three declared resolutions; no source-only result qualifies this matrix.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pops
import pytest
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.provider_pack import ProviderPack
from pops.numerics import TensorDiffusion, DiscretizationPlan
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
REFINEMENTS = (16, 32, 64)


def _case(dimension, n, dt):
    frame = CartesianDomain("periodic_tensor", (0.,)*dimension, (1.,)*dimension).frame(
        (Cartesian1D, Cartesian2D, Cartesian3D)[dimension-1]())
    model = pops.Model("entropy_mixture", frame=frame)
    state = model.state("inventory", components=("first", "second"))
    u, v = state
    metric = model.aux("metric")
    constant = np.eye(dimension)
    constant[0,0] = 2.
    if dimension > 1:
        constant[0,-1] = constant[-1,0] = .3
    tensor = tuple(tuple(.1*float(entry)*metric for entry in row) for row in constant)
    flux = model.diffusive_flux("rotated_conduction", state=state, value=(
        math.CoeffGradient(2*u+.25*v, tensor), math.CoeffGradient(.25*u+1.5*v, tensor)))
    rate = model.rate("balance", equation=math.ddt(state)==math.div(flux))
    case = pops.Case("variable_tensor")
    block = case.block("mixture", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, TensorDiffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = pops.Program("forward_euler")
    q = program.state(block[state])
    field_values = program.input_fields(q.n, for_rate=rate)
    derivative = rate(q.n, field_values)
    candidate = program.value("candidate", q.n+program.dt*derivative, at=q.next.point)
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, Uniform(CartesianGrid(frame=frame, cells=(n,)*dimension,
        periodic=PeriodicAxes(frame.axes))), constant


def _mode_average(wave, coordinates, n):
    phase = sum(float(k)*x for k,x in zip(wave, coordinates, strict=True))
    return np.exp(2j*np.pi*phase)*np.prod(np.sinc(np.asarray(wave)/n))


def _continuous_rate_average(wave, coordinates, n, tensor):
    """Analytical cell averages of div(.1*(1+.2 sin(2pi x))*A*grad(e^ikx))."""
    wave = np.asarray(wave, dtype=float)
    shift = np.zeros_like(wave)
    shift[0] = 1
    center = _mode_average(wave, coordinates, n)
    plus = _mode_average(wave+shift, coordinates, n)
    minus = _mode_average(wave-shift, coordinates, n)
    quadratic = float(wave@tensor@wave)
    directional = float(tensor[0]@wave)
    return .1*(2*np.pi)**2*(-quadratic*center +
        .1j*(quadratic*(plus-minus)+directional*(plus+minus)))


def test_public_variable_rotated_tensor_cross_gradient_full_matrix(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    dimension = int(os.environ.get("POPS_NATIVE_DIM", "2"))
    errors, rows = [], []
    for n in REFINEMENTS:
        coordinates = tuple(reversed(np.meshgrid(*([(np.arange(n)+.5)/n]*dimension), indexing="ij")))
        first_wave = np.ones(dimension)
        second_wave = np.ones(dimension)
        second_wave[0] = 2
        if dimension > 1:
            second_wave[-1] = -1
        initial = np.stack((2+_mode_average(first_wave, coordinates, n).real,
                            3+_mode_average(second_wave, coordinates, n).imag))
        dt = .01/(dimension*n*n)
        case, layout, tensor = _case(dimension, n, dt)
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        pack = ProviderPack.from_data(artifact.plan.blocks[0].resolved_operations.to_data()
                                     ["provider_evidence"]["auxiliary"])
        keys = [key for key in pack if pack.declared_entry(key).producer=="runtime_input"]
        assert len(keys)==1 and keys[0].component=="metric"
        runtime = pops.bind(artifact, initial_state={"mixture": np.ascontiguousarray(initial)},
            aux={keys[0]: np.ascontiguousarray(1+.2*np.sin(2*np.pi*coordinates[0]))},
            resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
        assert report.accepted_steps == 1
        result = np.asarray(runtime.state_global("mixture")).reshape(initial.shape)
        first = _continuous_rate_average(first_wave, coordinates, n, tensor).real
        second = _continuous_rate_average(second_wave, coordinates, n, tensor).imag
        expected = np.stack((2*first+.25*second, .25*first+1.5*second))
        error = float(np.sqrt(np.mean(((result-initial)/dt-expected)**2)))
        errors.append(error)
        inventory_error = np.max(np.abs(np.mean(result-initial, axis=tuple(range(1,dimension+1)))))
        def entropy(values):
            u,v=values
            return float(np.mean(u*u+.25*u*v+.75*v*v))
        assert inventory_error < 2e-13
        assert entropy(result) < entropy(initial)
        rows.append({"dimension":dimension,"cells_per_axis":n,"l2_rate_error":error,
                     "inventory_error":float(inventory_error),"entropy_before":entropy(initial),
                     "entropy_after":entropy(result),"accepted_steps":report.accepted_steps})
    orders = np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    assert np.all(orders > 1.8), (errors, orders)
    destination = os.environ.get("POPS_DIFFUSION_QUALIFICATION_EVIDENCE_DIR")
    if destination:
        path=Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path/f"tensor-dim{dimension}.json").write_text(json.dumps(
            {"schema":"pops.tensor-diffusion.v1","rows":rows,"orders":orders.tolist()},indent=2)+"\n")
