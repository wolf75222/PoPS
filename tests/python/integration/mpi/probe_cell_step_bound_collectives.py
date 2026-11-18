"""Measure selection alone on authored T02 MUSCL and the L1 Dim2 counterpart.

Launch on two MPI ranks in the installed Dim2 SDK. Preload the PMPI wrapper in
`cell_step_bound_probe.c` and use that same library as the Kokkos tools library.
`POPS_STEP_BOUND_PROBE` names the absolute path of the library.
"""
from __future__ import annotations

import ast
import ctypes
import os
from pathlib import Path

import pops
from pops._native_selector import select_native_dimension
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


def _measure(library, world, build, label):
    artifact, params, expected = build()
    runtime = pops.bind(artifact, params=params,
                        resources={"execution_context": artifact_execution_context(artifact)})
    native = runtime._executor._s
    library.pops_measure_selection_begin()
    try:
        bound = native._step_bound()
    finally:
        collectives, traversals = ctypes.c_int(), ctypes.c_int()
        library.pops_measure_selection_end(ctypes.byref(collectives), ctypes.byref(traversals))
    print("artifact", label, world.rank, artifact.so_path, flush=True)
    row = label, world.rank, collectives.value, traversals.value, bound.hex(), expected.hex()
    print("selection", *row, flush=True)
    return row


def main():
    world = select_native_dimension(2).mpi_world()
    assert world.size == 2
    library = ctypes.CDLL(os.environ["POPS_STEP_BOUND_PROBE"])
    rows = (_measure(library, world, _tutorial, "T02-MUSCL"),
            _measure(library, world, _l1, "L1-advection-diffusion"))
    for label, rank, collectives, traversals, bound, expected in rows:
        assert bound == expected, (label, rank, bound, expected)
        assert (collectives, traversals) == (1, 1), (label, rank, collectives, traversals)


if __name__ == "__main__":
    main()
