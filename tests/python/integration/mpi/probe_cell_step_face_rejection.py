"""Measure a genuine face rejection under the compiled cell step rule on two ranks."""
from __future__ import annotations

import ctypes
import os

import numpy as np
import pops
import pops.surface as surface
from pops._native_selector import select_native_dimension
from pops.layouts import Uniform
from pops.lib.numerics import rusanov
from pops.lib.time.methods import ssprk2
from pops.runtime._simulation import FaceFrequencyStop, Simulation
from pops.surface.math import ddt, div, vector
from tests.python.integration.mpi.probe_cell_step_bound_collectives import _BandedGrid, _runtime


def _artifact():
    shape = (16, 4)
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("face_rejection")
    u = model.quantity("u", surface.Scalar(domain))
    initial = model.data("initial", surface.Scalar(domain))
    flux = model.flux(of=u, value=vector(u * u / 2, 0 * u))
    model.balance(ddt(u) == -div(flux))

    def bound(cell):
        return cell.width(domain.x)

    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=bound),
                              initial={u: initial})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=shape)})
    elaborated = case.elaborated
    grid = elaborated.layout.mesh
    layout = Uniform(_BandedGrid(frame=grid.frame, cells=shape, periodic=grid.periodic))
    return pops.compile(pops.resolve(pops.validate(elaborated.case), layout=layout)), shape


def _reject(artifact, shape, rank, library, left, right):
    block, = artifact.plan.blocks
    binding, = artifact.plan.initial_condition_plan.bindings
    values = np.full(shape, 0.25)
    values[3, 2] = left
    values[12, 2] = right
    local_values = np.ascontiguousarray(values.T[np.newaxis])
    instance = _runtime(artifact, initial_values={binding.subject: local_values})
    native = instance._executor._s
    boxes = native.local_boxes(block.name)
    assert len(boxes) == 1 and tuple(boxes[0][0]) == (rank * shape[0] // 2, 0)
    assert float(np.max(native.local_state(block.name, 0))) == (left, right)[rank]
    before = np.asarray(instance.state_global(block.name)).tobytes()
    simulation = Simulation(instance)
    library.pops_measure_selection_begin()
    try:
        simulation.run(0.10)
    except FaceFrequencyStop as error:
        frequency = error.frequency
        assert error.result.rejections == 1 and error.result.steps == 0
    else:
        raise AssertionError("the compiled face guard accepted an unstable proposal")
    finally:
        collectives, traversals = ctypes.c_int(), ctypes.c_int()
        library.pops_measure_selection_end(ctypes.byref(collectives), ctypes.byref(traversals))
    assert frequency == shape[0] * max(left, right)
    assert instance.time() == 0.0 and instance.macro_step() == 0
    assert np.asarray(instance.state_global(block.name)).tobytes() == before
    print(f"face refusal rank={rank} maxima={left}/{right} frequency={frequency.hex()} "
          f"allreduce={collectives.value} reductions={traversals.value}", flush=True)


def main():
    world = select_native_dimension(2).mpi_world()
    assert world.size == 2
    library = ctypes.CDLL(os.environ["POPS_STEP_BOUND_PROBE"])
    artifact, shape = _artifact()
    _reject(artifact, shape, world.rank, library, 0.25, 2.0)
    _reject(artifact, shape, world.rank, library, 1.5, 3.0)


if __name__ == "__main__":
    main()
