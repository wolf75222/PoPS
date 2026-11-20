"""Measure a genuine face rejection under the compiled cell step rule on two ranks."""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
import locale
import os
import sys

import numpy as np
import pops
import pops.surface as surface
from pops._native_selector import select_native_dimension
from pops.layouts import Uniform
from pops.lib.numerics import rusanov
from pops.lib.time.methods import ssprk2
from pops.runtime._simulation import BudgetExhausted, Simulation
from pops.surface.math import ddt, div, vector
from pops.time._step.strategy import FACE_STEP_FREQUENCY_SCHEMA
from tests.python.integration.mpi.probe_cell_step_bound_collectives import _BandedGrid, _runtime


def _artifact(attempts):
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
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=bound, attempts=attempts),
                              initial={u: initial})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=shape)})
    elaborated = case.elaborated
    grid = elaborated.layout.mesh
    layout = Uniform(_BandedGrid(frame=grid.frame, cells=shape, periodic=grid.periodic))
    return pops.compile(pops.resolve(pops.validate(elaborated.case), layout=layout)), shape


@contextmanager
def _locale(numeric_locale, rank):
    previous_locale = locale.setlocale(locale.LC_NUMERIC)
    try:
        if numeric_locale is not None:
            locale.setlocale(locale.LC_NUMERIC, numeric_locale)
            assert locale.localeconv()["decimal_point"] == ","
            formatted = ctypes.create_string_buffer(64)
            snprintf = ctypes.CDLL(None).snprintf
            snprintf.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_char_p)
            snprintf(formatted, len(formatted), b"%a", ctypes.c_double(48.0))
            assert formatted.value == b"0x1,8p+5", formatted.value
            print(f"locale rank={rank} LC_NUMERIC={locale.setlocale(locale.LC_NUMERIC)} "
                  f"printf={formatted.value!r}", flush=True)
        yield
    finally:
        if numeric_locale is not None:
            locale.setlocale(locale.LC_NUMERIC, previous_locale)


def _bind(artifact, shape, rank, left, right):
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
    return instance, block.name


def _reject(artifact, shape, rank, library, left, right, numeric_locale):
    from pops.runtime._step_strategy import _face_step_frequency

    instance, block = _bind(artifact, shape, rank, left, right)
    native = instance._executor._s
    before = np.asarray(instance.state_global(block)).tobytes()
    simulation = Simulation(instance)
    with _locale(numeric_locale, rank):
        library.pops_measure_selection_begin()
        try:
            simulation.run(0.10)
        except BudgetExhausted as error:
            failure = error.result.failure
            assert failure.origin == "trial" and failure.payload["budget"] == "max_attempts"
            assert failure.payload["face_schema"] == FACE_STEP_FREQUENCY_SCHEMA
            assert error.result.diagnostics["native"]["attempts"] == 1
            frequency = failure.payload["face_frequency"]
            assert error.result.rejections == 1 and error.result.steps == 0
        else:
            raise AssertionError("the compiled face guard accepted an unstable proposal")
        finally:
            collectives, traversals = ctypes.c_int(), ctypes.c_int()
            library.pops_measure_selection_end(ctypes.byref(collectives), ctypes.byref(traversals))
        # Inspect the agreed bytes on a second attempt outside the measured run.
        try:
            native._step_at_fraction(1.0 / shape[0], 1.0)
        except RuntimeError as error:
            assert error.typed_failure_schema == FACE_STEP_FREQUENCY_SCHEMA
            payload = error.typed_failure_payload
            assert payload == (b"0x1p+5" if right == 2.0 else b"0x1.8p+5")
            assert _face_step_frequency(error) == frequency
        else:
            raise AssertionError("the native face guard did not return its typed payload")
    assert frequency == shape[0] * max(left, right)
    assert instance.time() == 0.0 and instance.macro_step() == 0
    assert np.asarray(instance.state_global(block)).tobytes() == before
    print(f"face refusal rank={rank} maxima={left}/{right} frequency={frequency.hex()} "
          f"payload={payload!r} "
          f"allreduce={collectives.value} reductions={traversals.value}", flush=True)


def _retry(artifact, shape, rank):
    left, right = 1.5, 3.0
    target, block = _bind(artifact, shape, rank, left, right)
    reference, _ = _bind(artifact, shape, rank, left, right)
    expected_dt = 1.0 / (shape[0] * right)
    try:
        Simulation(target).run(0.10, max_steps=1)
    except BudgetExhausted as error:
        result = error.result
        assert result.failure.payload["budget"] == "max_steps"
    else:
        raise AssertionError("the corrected first step did not stop at the accepted-step budget")
    assert (result.steps, result.rejections, result.time) == (1, 1, expected_dt)
    report = pops.step(reference, t_bound=0.10, console=False)
    assert (report.accepted_steps, report.rejected_steps, reference.time()) == (1, 1, expected_dt)
    assert target.macro_step() == reference.macro_step() == 1
    assert np.asarray(target.state_global(block)).tobytes() == (
        np.asarray(reference.state_global(block)).tobytes())
    print(f"face retry rank={rank} maxima={left}/{right} dt={expected_dt.hex()} "
          "accepted=1 rejected=1 state=bitwise", flush=True)


def main():
    world = select_native_dimension(2).mpi_world()
    assert world.size == 2
    numeric_locale = sys.argv[1] if len(sys.argv) == 2 else None
    library = ctypes.CDLL(os.environ["POPS_STEP_BOUND_PROBE"])
    artifact, shape = _artifact(attempts=1)
    _reject(artifact, shape, world.rank, library, 0.25, 2.0, numeric_locale)
    _reject(artifact, shape, world.rank, library, 1.5, 3.0, numeric_locale)
    artifact, shape = _artifact(attempts=2)
    _retry(artifact, shape, world.rank)


if __name__ == "__main__":
    main()
