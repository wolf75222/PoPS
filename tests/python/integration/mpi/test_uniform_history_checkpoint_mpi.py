#!/usr/bin/env python3
"""Public two-rank Uniform history checkpoint/restart contract.

Every runtime in this manifest-owned ``mpiexec`` entrypoint is produced by the final lifecycle

``Case -> validate -> resolve -> compile -> ExecutionContext.mpi_world -> bind -> run``.

The test never imports a direct engine, a Python collective adapter, or a checkpoint implementation
module.  It proves that the public collective checkpoint/restart surface preserves the dense AB2
rate ring and accepted clock bit-for-bit, rejects rank-divergent checkpoint and restart paths before
publication/mutation, and continues identically to an uninterrupted public ``RuntimeInstance``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
# The MPI lane normally imports the assembled/installed package.  Append (never prepend) the
# checkout only so shared test support remains importable without masking that package.
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(_REPOSITORY_ROOT))

from tests.python.support.requirements import (  # noqa: E402
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_mpi_or_skip,
)

try:
    import numpy as np

    import pops
    import pops.lib.time as libtime
    from pops.codegen import Production
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.time import FixedDt
except (Exception, SystemExit) as exc:
    require_mpi_or_skip("Uniform history MPI runtime imports are unavailable: %s" % exc)


N = 16
DT = 0.01
NSTEPS = 6
SOURCE_COEFFICIENT = 0.75


def _resolved_public_ab2() -> Any:
    """Author and resolve one complete public Uniform AB2 case without native side seams."""
    frame = Rectangle(
        "uniform-history-mpi-domain",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("uniform-history-mpi-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    zero = 0.0 * rho
    flux = model.flux(
        "stationary-transport",
        frame=frame,
        state=state,
        components={x_axis: (zero,), y_axis: (zero,)},
        waves={x_axis: (zero,), y_axis: (zero,)},
    )
    source = model.source(
        "linear-growth",
        on=state,
        value=(SOURCE_COEFFICIENT * rho,),
    )
    rate = model.rate(
        "growth-rate",
        equation=ddt(state) == -div(flux) + source,
    )

    case = pops.Case("uniform-history-mpi-case")
    block = case.block("tracer", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = libtime.AdamsBashforth(block_state, rate=rate, order=2)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(_REPOSITORY_ROOT / "include")},
    )


def _initial_state() -> Any:
    coordinate = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    return np.ascontiguousarray(rho.reshape(1, N, N), dtype=np.float64)


def _bind(artifact: Any, context: Any, initial: Any) -> Any:
    return pops.bind(
        artifact,
        initial_state={"tracer": np.array(initial, copy=True, order="C")},
        resources={"execution_context": context},
    )


def _advance(runtime: Any, steps: int) -> Any:
    return pops.run(
        runtime,
        t_end=float(runtime.time()) + steps * DT,
        max_steps=steps,
    )


def _rings(runtime: Any) -> dict[str, tuple[Any, ...]]:
    return {
        name: tuple(
            np.asarray(runtime.history_global(name, slot), dtype=np.float64).copy()
            for slot in range(runtime.history_depth(name))
        )
        for name in runtime.history_names()
    }


def _rings_equal(
    left: dict[str, tuple[Any, ...]],
    right: dict[str, tuple[Any, ...]],
) -> bool:
    return left.keys() == right.keys() and all(
        len(left[name]) == len(right[name])
        and all(
            np.array_equal(first, second)
            for first, second in zip(left[name], right[name], strict=True)
        )
        for name in left
    )


def _expected_ab2(initial: Any, steps: int) -> Any:
    """Independent scalar AB2 recurrence; startup duplicates the current rate into its ring."""
    accepted = np.array(initial, copy=True, order="C")
    previous_rate = SOURCE_COEFFICIENT * accepted
    for step in range(steps):
        current_rate = SOURCE_COEFFICIENT * accepted
        if step == 0:
            following = accepted + DT * current_rate
        else:
            following = accepted + DT * (1.5 * current_rate - 0.5 * previous_rate)
        previous_rate = current_rate
        accepted = following
    return accepted


def _assert_failure(
    operation: Any,
    error_type: type[BaseException],
    fragments: tuple[str, ...],
) -> str:
    try:
        operation()
    except error_type as error:
        message = str(error)
        assert all(fragment in message for fragment in fragments), message
        return message
    raise AssertionError("collective public operation unexpectedly succeeded")


def _checkpoint_target(artifact: Any) -> Path:
    identity = artifact.artifact_identity.hexdigest
    digest = hashlib.sha256((identity + "|uniform-history-mpi").encode("utf-8")).hexdigest()[:20]
    return Path(os.environ.get("TMPDIR", "/tmp")) / ("pops-uniform-history-mpi-%s.npz" % digest)


def test_uniform_history_checkpoint_restart_mpi() -> None:
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_mpi_or_skip("Uniform history MPI toolchain unavailable: %s" % missing)

    # The production cache owns a process-shared identity lock, so every rank may call the same
    # public compile transition. Exactly one publisher builds; peers authenticate the cached bytes.
    artifact = pops.compile(_resolved_public_ab2())
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    size = int(world.size)
    rank = int(world.rank)
    if size < 2:
        require_mpi_or_skip("Uniform history MPI contract requires mpiexec -n 2 (size=%d)" % size)

    initial = _initial_state()
    half = NSTEPS // 2
    target = _checkpoint_target(artifact)
    # Every rank removes the same stale name before entering the first collective operation.  The
    # following divergent-path rejection is the synchronization point before any publication.
    target.unlink(missing_ok=True)

    continuous = _bind(artifact, context, initial)
    _advance(continuous, half)
    rings_at_half = _rings(continuous)
    assert rings_at_half, "public AB2 runtime exposes no dense rate history"
    _advance(continuous, NSTEPS - half)
    reference = np.asarray(continuous.state_global("tracer"), dtype=np.float64).copy()

    interrupted = _bind(artifact, context, initial)
    _advance(interrupted, half)
    assert _rings_equal(rings_at_half, _rings(interrupted))

    divergent_checkpoint = (
        target.parent / ("rank-%d-uniform-history" % rank) / "must-not-publish.npz"
    )
    _assert_failure(
        lambda: interrupted.checkpoint(divergent_checkpoint),
        ValueError,
        ("rank 1", "staging directory differs across ranks"),
    )
    assert not divergent_checkpoint.exists()

    checkpoint = Path(interrupted.checkpoint(target))
    assert checkpoint == target and checkpoint.is_file()

    restored = _bind(artifact, context, initial)
    restored.restart(checkpoint)
    assert _rings_equal(rings_at_half, _rings(restored))
    assert restored.macro_step() == half
    assert restored.time() == half * DT

    # A rank-divergent public restart path must fail coherently before the accepted state, clock,
    # history, or consumer cursors are touched. The lower transaction-provider protocol itself is
    # covered by its dedicated unit test, not by reaching through this public MPI acceptance.
    preflight_target = _bind(artifact, context, initial)
    state_before_failure = np.asarray(
        preflight_target.state_global("tracer"), dtype=np.float64
    ).copy()
    history_shape_before_failure = tuple(
        (name, preflight_target.history_depth(name)) for name in preflight_target.history_names()
    )
    wrong_restart = (
        checkpoint if rank == 0 else checkpoint.with_name("rank-one-wrong-uniform-history.npz")
    )
    _assert_failure(
        lambda: preflight_target.restart(wrong_restart),
        ValueError,
        ("rank 1", "restart target differs across ranks"),
    )
    assert np.array_equal(
        state_before_failure,
        np.asarray(preflight_target.state_global("tracer"), dtype=np.float64),
    )
    assert preflight_target.macro_step() == 0
    assert preflight_target.time() == 0.0
    assert history_shape_before_failure == tuple(
        (name, preflight_target.history_depth(name)) for name in preflight_target.history_names()
    )

    _advance(restored, NSTEPS - half)
    result = np.asarray(restored.state_global("tracer"), dtype=np.float64).copy()

    assert np.array_equal(reference, result)
    assert restored.macro_step() == continuous.macro_step() == NSTEPS
    assert restored.time() == continuous.time() == NSTEPS * DT
    np.testing.assert_allclose(
        reference,
        _expected_ab2(initial, NSTEPS),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    target.unlink(missing_ok=True)
    shutil.rmtree(divergent_checkpoint.parent, ignore_errors=True)


def _run_all() -> None:
    test_uniform_history_checkpoint_restart_mpi()
    print("PASS test_uniform_history_checkpoint_mpi", flush=True)


if __name__ == "__main__":
    _run_all()
