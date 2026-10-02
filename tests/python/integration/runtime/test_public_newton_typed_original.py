"""Genuine installed Native Original Newton policies; ROOT runs Serial/MPI SDK8."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pops
import pytest
from pops.solvers import Newton
from pops.solvers.tolerances import Relative, AbsoluteFloor, Absolute
from tests.python.integration.runtime import test_nonlinear_mixed_field_runtime as uniform
from tests.python.integration.runtime import test_public_amr_original_field as amr
from tests.python.support.collective_checks import collective_call, collective_attempt, collective_check, state_snapshots


def solver(backend, policy):
    tolerance = {"relative": Relative(1e-11), "floor": Relative(1e-11, floor=AbsoluteFloor(1e-11)),
                 "absolute": Absolute(1e-11), "overflow": Relative(sys.float_info.max)}[policy]
    # Preserve the existing backend's iteration, Krylov and FD realization.
    return Newton(tolerance=tolerance, max_iterations=20,
        linear_tolerance=1e-9 if backend == "uniform" else 1e-8,
        linear_max_iterations=150 if backend == "uniform" else 240, restart=60)


def snapshot(world, runtime, backend):
    if backend == "amr":
        image = collective_call(world, lambda: bytes(runtime._executor.checkpoint_state_carriers()))
        rows, carriers, history, lifecycle = amr.capture(world, runtime)
        return [a for row in rows for a in row], (image, carriers, history, lifecycle)
    arrays = state_snapshots(runtime, world, ("forcing", "parameter"))
    report = collective_call(world, runtime.program_report)
    clock = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
    return arrays, (report.histories, report.diagnostics, clock)


@pytest.mark.parametrize("backend", ("uniform", "amr"))
@pytest.mark.parametrize("policy", ("relative", "floor", "absolute", "overflow"))
def test_public_typed_original_policy(isolated_native_cache, tmp_path, record_property, backend, policy):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    package = Path(pops.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    native = select_native_dimension(2)
    assert native.__abi_version__ == 8
    world = native.mpi_world()
    selected = solver(backend, policy)
    if backend == "uniform":
        runtime, artifact, target, loads, coefficient = uniform.bind_native(world, native, (0, 1), solver=selected)
        # Original zero-seed residual, full DOF L2 (no Schur/reduced reference).
        with collective_check(world):
            assert float(np.linalg.norm(loads)) > 1
    else:
        # A declared constant physical seed changes initialization only. At seed2,
        # the first Original reaction >=10 whereas its load <.2, so volume-L2>1.
        runtime, artifact = amr.bind_case(world, 16, (2, 0, 1), solver=selected,
            physical_seed=2. if policy == "overflow" else None)
    before, prior = snapshot(world, runtime, backend)
    report, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    after, final = snapshot(world, runtime, backend)
    # Persist rank-local exact images before any admission assertion.
    with collective_check(world):
        directory = tmp_path / (backend + "-" + policy + "-rank" + str(world.rank))
        directory.mkdir()
        path = directory / "states.npz"
        np.savez(path, **{f"before{i}": a for i, a in enumerate(before)},
                      **{f"after{i}": a for i, a in enumerate(after)})
        if backend == "amr":
            (directory / "before-carriers.bin").write_bytes(prior[0])
            (directory / "after-carriers.bin").write_bytes(final[0])
        receipt = {"schema": "sol61.newton-typed-original-native@1", "backend": backend, "policy": policy,
            "package_path": str(package), "native_path": str(Path(native.__file__).resolve()),
            "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "native_abi_version": native.__abi_version__, "rank": world.rank, "size": world.size,
            "artifact_identity": artifact.artifact_identity.token,
            "states_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "carrier_sha256": ({phase: hashlib.sha256((directory / (phase+"-carriers.bin")).read_bytes()).hexdigest()
                for phase in ("before", "after")} if backend == "amr" else None), "failures": failures,
            "solver": selected.to_data()}
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    record_property("typed_original_receipt", str(directory / "receipt.json"))
    with collective_check(world):
        if policy == "overflow":
            assert all(failures), failures
            assert all("cutoff" in row[1] or "threshold" in row[1] for row in failures), failures
            assert final == prior
            for lhs, rhs in zip(before, after, strict=True):
                np.testing.assert_array_equal(lhs, rhs)
        else:
            assert not any(failures), failures
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            if backend == "uniform":
                for lhs, rhs in zip(before, after, strict=True):
                    np.testing.assert_array_equal(lhs, rhs)
            if backend == "amr" and world.rank == 0:
                amr.check_original_saved([tuple(after[i:i+4]) for i in range(0, len(after), 4)], 16, 3, .01)
    if policy != "overflow" and backend == "uniform":
        for run in range(2):
            values = [collective_call(world, lambda c=c: runtime.history_global(f"solve{run}_component{c}", 1)) for c in range(2)]
            with collective_check(world):
                if world.rank == 0:
                    solved = np.asarray(values).reshape(2, *reversed(uniform.CELLS))
                    assert np.max(np.abs(solved-target)) < uniform.STATE_ATOL
                    assert np.max(np.abs(uniform.original_lhs(solved, coefficient)-loads)) < uniform.RESIDUAL_ATOL
