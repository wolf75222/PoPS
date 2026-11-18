"""Measure selection alone on authored T02 MUSCL and the L1 Dim2 counterpart.

Launch on one or two MPI ranks in the installed Dim2 SDK. Preload the PMPI wrapper in
`cell_step_bound_probe.c` and use that same library as the Kokkos tools library.
`POPS_STEP_BOUND_PROBE` names the absolute path of the library.
"""
from __future__ import annotations

import ast
import ctypes
import math
import os
from pathlib import Path

import numpy as np
import pops
from pops._native_selector import select_native_dimension
from pops.layouts import Uniform
from pops.mesh import CartesianGrid
from tests.python.integration.runtime.test_cell_step_rule_n8_native import NU, _compile_l1
from tests.python.support.native_execution_context import artifact_execution_context


def _tutorial():
    path = Path("examples/surface/tutorials/scalar_advection/02_openmp_explicit_ssprk2.py")
    module = ast.parse(path.read_text(), filename=str(path))
    for index, statement in enumerate(module.body):
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "artifact"
            for target in statement.targets
        ):
            scope = {"__file__": str(path), "__name__": "__main__"}
            module.body = module.body[:index + 1]
            exec(compile(module, str(path), "exec"), scope)
            return scope["artifact"].compiled, {}, (0.45 * (1.0 / 64)) / 1.0
    raise AssertionError("T02 has no artifact declaration")


def _l1():
    artifact, _, nu = _compile_l1()
    compiled = artifact.compiled
    slots = [slot.handle for slot in compiled.bind_schema.runtime_slots
             if slot.handle.local_id == nu.name]
    (slot,) = slots
    h = 1.0 / 64
    return compiled, {slot: NU}, 1 / (1 / h + (4 * NU) / h**2)


def _runtime(artifact, params=None, initial_values=None):
    return pops.bind(artifact, params=params, initial_values=initial_values,
                        resources={"execution_context": artifact_execution_context(artifact)})


def _measured_call(library, native, identity=None):
    library.pops_measure_selection_begin()
    try:
        result = native._step_bound() if identity is None else native._step_bound(identity)
    except (ValueError, RuntimeError) as error:
        result = str(error)
    finally:
        collectives, traversals = ctypes.c_int(), ctypes.c_int()
        library.pops_measure_selection_end(ctypes.byref(collectives), ctypes.byref(traversals))
    return result, collectives.value, traversals.value


def _measure(library, world, build, label):
    artifact, params, expected = build()
    runtime = _runtime(artifact, params=params)
    native = runtime._executor._s
    block, = artifact.plan.blocks
    local_boxes = len(native.local_boxes(block.name))
    bound, collectives, traversals = _measured_call(library, native)
    assert isinstance(bound, float), bound
    print("artifact", label, world.rank, artifact.so_path, flush=True)
    row = label, world.rank, local_boxes, collectives, traversals, bound.hex(), expected.hex()
    print("selection", *row, flush=True)
    return row


class _BandedGrid(CartesianGrid):
    """Same constant cell widths with an authenticated two-box decomposition."""

    def native_spatial_data(self):
        data = super().native_spatial_data()
        nx, ny = self.cells
        data["decomposition"] = {
            "schema_version": 1,
            "kind": "axis_bands",
            "boxes": [
                {"lower": [0, 0], "upper_exclusive": [nx // 2, ny]},
                {"lower": [nx // 2, 0], "upper_exclusive": [nx, ny]},
            ],
        }
        return data


def _edge_artifact():
    import pops.surface as surface
    from pops.lib.numerics import upwind
    from pops.lib.time.methods import ssprk2
    from pops.surface.math import ddt, div, vector

    shape = (16, 4)
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("cell_bound_diagnostic")
    u = model.quantity("u", surface.Scalar(domain))
    initial = model.data("initial", surface.Scalar(domain))
    flux = model.flux(of=u, value=vector(1, 0) * u)
    model.balance(ddt(u) == -div(flux))

    def bound(cell):
        return cell.width(domain.x) / cell.value(u)

    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=bound),
                              initial={u: initial})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=shape)})
    elaborated = case.elaborated
    grid = elaborated.layout.mesh
    layout = Uniform(_BandedGrid(frame=grid.frame, cells=shape, periodic=grid.periodic))
    return pops.compile(pops.resolve(pops.validate(elaborated.case), layout=layout)), shape


def _edge_cases(library, world):
    artifact, shape = _edge_artifact()
    block, = artifact.plan.blocks
    binding, = artifact.plan.initial_condition_plan.bindings
    ones = np.ones(shape)
    finite = ones.copy()
    finite[9, 2] = 2.0
    remote = ones.copy()
    remote[9, 2] = np.nan
    remote[10, 2] = np.nan
    nonpositive = ones.copy()
    nonpositive[1, 0] = -1.0
    nonpositive[0, 1] = -1.0
    mixed = remote.copy()
    mixed[1, 0] = -1.0
    cases = (
        ("finite", finite, (1.0 / shape[0]) / 2),
        ("rest", np.zeros(shape), math.inf),
        ("remote", remote, "step_bound_nonfinite", "(9, 2)"),
        ("nonpositive", nonpositive, "step_bound_nonpositive", "(1, 0)"),
        ("mixed", mixed, "step_bound_nonfinite", "(9, 2)"),
    )
    for label, values, expected, *cell in cases:
        ranked = np.ascontiguousarray(values.T[np.newaxis])
        runtime = _runtime(artifact, initial_values={binding.subject: ranked})
        native = runtime._executor._s
        boxes = native.local_boxes(block.name)
        assert len(boxes) == 2 // world.size
        if world.size == 2:
            assert tuple(boxes[0][0]) == (world.rank * shape[0] // 2, 0)
        result, collectives, traversals = _measured_call(library, native)
        print("edge", label, world.rank, collectives, traversals, result, flush=True)
        assert (collectives, traversals) == (1, len(boxes))
        if cell:
            assert isinstance(result, str) and result.startswith(expected), result
            assert "at the cell " + cell[0] in result, result
        else:
            assert isinstance(result, float) and result.hex() == expected.hex(), result

    # One rank refuses dispatch before running its kernel. Its peer never waits for a second
    # collective, and both ranks report the same plan failure.
    if world.size == 2:
        identity = native._step_bound_identity() if world.rank == 0 else "different-bound"
        result, collectives, traversals = _measured_call(library, native, identity)
        print("edge", "dispatch", world.rank, collectives, traversals, result, flush=True)
        assert (collectives, traversals) == (1, 1 - world.rank)
        assert "names a step bound that the installed Program does not export" in result


def main():
    world = select_native_dimension(2).mpi_world()
    assert world.size in (1, 2)
    library = ctypes.CDLL(os.environ["POPS_STEP_BOUND_PROBE"])
    rows = (_measure(library, world, _tutorial, "T02-MUSCL"),
            _measure(library, world, _l1, "L1-advection-diffusion"))
    for label, rank, boxes, collectives, traversals, bound, expected in rows:
        assert bound == expected, (label, rank, bound, expected)
        assert (collectives, traversals) == (1, boxes), (label, rank, boxes, collectives, traversals)
    _edge_cases(library, world)


if __name__ == "__main__":
    main()
