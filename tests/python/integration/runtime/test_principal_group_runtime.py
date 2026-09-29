"""C11 on the installed native Dim2 carrier, with a separate Rusanov oracle."""
import numpy as np
import pytest
import pops

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.numerics.variables import Conservative
from pops.projection import ConservativeCellAverage
from pops.time import ExternalTimeGrid
from tests.python.support.native_execution_context import artifact_execution_context


CELLS, DT = 8, .002


def _problem(matrix, *, vector, permutation):
    size = len(matrix)
    frame = Rectangle("principal-domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("principal-vector" if vector else "principal-scalars", frame=frame)
    if vector:
        states = (model.state("vector", components=tuple("c%d" % i for i in range(size))),)
        components = tuple(states[0])
    else:
        states = tuple(model.species("state_%d" % permutation[i], state=("q",))
                       for i in range(size))
        components = tuple(state[0] for state in states)
    flux_values = tuple(sum(float(matrix[i, j]) * components[j] for j in range(size))
                        for i in range(size))
    bound = float(np.max(np.sum(np.abs(matrix), axis=1)))
    fluxes, rates = [], []
    for i, state in enumerate(states):
        row = flux_values if vector else (flux_values[i],)
        flux = model.flux("F%d" % i, frame=frame, state=state,
            components={frame.x: row, frame.y: (0.,) * len(row)},
            waves={frame.x: (bound,) * len(row), frame.y: (0.,) * len(row)})
        fluxes.append(flux)
        rates.append(model.rate("R%d" % i, equation=ddt(state) == -div(flux)))
    case = pops.Case("principal-native")
    blocks = tuple(case.block("block_%d" % permutation[i], model, states=(state,))
                   for i, state in enumerate(states))
    for block, state, flux, rate in zip(blocks, states, fluxes, rates, strict=True):
        plan = DiscretizationPlan()
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=Conservative(state),
            reconstruction=FirstOrder(), riemann=Rusanov(),
            sampling=tuple(other for other in states if other != state)))
        case.numerics(plan, block=block)
        case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                         projection=ConservativeCellAverage()))
    program = pops.Program("principal-euler")
    quantities = tuple(program.state(block[state]) for block, state in zip(blocks, states, strict=True))
    bindings = dict(zip(states, (quantity.n for quantity in quantities), strict=True))
    evaluations = tuple(rate(quantity.n) if vector else rate(quantity.n, bindings=bindings)
                        for rate, quantity in zip(rates, quantities, strict=True))
    for quantity, rhs in zip(quantities, evaluations, strict=True):
        result = program.value("next", quantity.n + program.dt * rhs, at=quantity.next.point)
        program.commit(quantity.next, result)
    program.step_strategy(ExternalTimeGrid("time_grid"))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, blocks, states, bound


def _run(matrix, initial, *, vector, permutation):
    case, layout, blocks, states, _ = _problem(matrix, vector=vector, permutation=permutation)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    context = artifact_execution_context(artifact)
    values = {case.resolve(block[state]): initial.copy() if vector else initial[i:i+1].copy()
              for i, (block, state) in enumerate(zip(blocks, states, strict=True))}
    runtime = pops.bind(artifact, initial_values=values, resources={"execution_context": context})
    report = pops.run(runtime, t_end=DT, max_steps=1, time_grid=(0., DT))
    actual = [runtime.state_global(block.local_id) for block in blocks]
    world = context.communicator.handle
    rank = 0 if world is None else world.rank
    result = None
    failure = b""
    if rank == 0:
        try:
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            result = (np.asarray(actual[0]).reshape(initial.shape) if vector else
                      np.concatenate([np.asarray(value).reshape(1, CELLS, CELLS) for value in actual]))
        except Exception as error:
            failure = repr(error).encode()
    if world is not None:
        failure = world.broadcast_bytes(failure, root=0)
    assert not failure, failure.decode()
    return result, world


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("size", (2, 3, 5))
@pytest.mark.parametrize("permuted", (False, True))
def test_native_principal_scalars_match_vector_and_independent_face_oracle(
        isolated_native_cache, native_cxx, kokkos_root, size, permuted):
    del isolated_native_cache, native_cxx, kokkos_root
    matrix = np.array([[1. + .1 * i if i == j else .03 * (i + j + 1)
                        for j in range(size)] for i in range(size)])
    permutation = tuple(reversed(range(size))) if permuted else tuple(range(size))
    matrix = matrix[np.ix_(permutation, permutation)]
    x = np.arange(CELLS, dtype=float) / CELLS
    line = np.stack([1. + .1 * (i + 1) * np.cos(2 * np.pi * x + .2 * i)
                     for i in permutation])
    initial = np.ascontiguousarray(np.broadcast_to(line[:, None, :], (size, CELLS, CELLS)))
    right = np.roll(line, -1, axis=1)
    bound = float(np.max(np.sum(np.abs(matrix), axis=1)))
    face = .5 * (matrix @ line + matrix @ right) - .5 * bound * (right - line)
    expected_line = line + DT * CELLS * (np.roll(face, 1, axis=1) - face)
    expected = np.broadcast_to(expected_line[:, None, :], initial.shape)
    vector, _ = _run(matrix, initial, vector=True, permutation=permutation)
    scalars, world = _run(matrix, initial, vector=False, permutation=permutation)
    failure = b""
    if world is None or world.rank == 0:
        try:
            np.testing.assert_allclose(vector, expected, rtol=0., atol=5.e-13)
            np.testing.assert_allclose(scalars, expected, rtol=0., atol=5.e-13)
            np.testing.assert_allclose(scalars, vector, rtol=0., atol=5.e-13)
            np.testing.assert_allclose(scalars.sum(axis=(1, 2)), initial.sum(axis=(1, 2)),
                                       rtol=0., atol=2.e-12)
        except Exception as error:
            failure = repr(error).encode()
    if world is not None:
        failure = world.broadcast_bytes(failure, root=0)
    assert not failure, failure.decode()
