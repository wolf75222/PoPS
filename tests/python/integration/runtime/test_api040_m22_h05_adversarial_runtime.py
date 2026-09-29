"""Independent installed H05 locality, capture and collective-domain witness."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from tests.python.support.native_execution_context import artifact_execution_context
from test_user_numerical_bodies_runtime import _compile, _root_check


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _case_module():
    folder = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    old_path = sys.path[:]
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            "api040_m22_h05_adversarial", folder / "api040_m22_h05.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def _agree(world, payload):
    if world is None:
        return (payload,)
    from pops._native_collectives import allgather_value
    return tuple(allgather_value(world, payload))


def _states(runtime):
    return tuple(np.asarray(runtime.state_global(name)).copy()
                 for name in ("radiation", "matter"))


@pytest.mark.parametrize("reverse", (False, True))
def test_h05_nonuniform_cells_rebind_and_collective_negative_rate(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, reverse):
    del isolated_native_cache, native_cxx, kokkos_root
    example = _case_module()
    case, layout, subjects, authored_rate = example.build_case(reverse=reverse)
    artifact, world = _compile(case, layout, "H05-adversarial-%s" % reverse)
    i, j = np.indices((4, 4), dtype=float)
    initial = np.stack((2. + .1*i + .03*j, .5 + .02*i + .07*j))
    assert initial.shape == (2, 4, 4) and np.min(initial) > 0

    # Rebind one compiled artifact. The authored, block-qualified handle must
    # authenticate through the artifact's exact BindSchema; no name lookup.
    for rate in (.8, 0., -.8):
        runtime = None
        bind_error = ("", "")
        try:
            runtime = pops.bind(
                artifact,
                initial_values={subject: initial[index:index+1].copy()
                                for index, subject in enumerate(subjects)},
                params={authored_rate: rate},
                resources={"execution_context": artifact_execution_context(artifact)},
            )
        except Exception as error:
            bind_error = (type(error).__name__, str(error))
        bind_errors = _agree(world, bind_error)
        assert all(kind == "" for kind, _ in bind_errors), bind_errors
        assert runtime is not None
        before = _states(runtime)

        report = None
        run_error = ("", "")
        try:
            report = pops.run(runtime, t_end=.4, max_steps=1, console=False)
        except Exception as error:
            run_error = (type(error).__name__, str(error))
        run_errors = _agree(world, run_error)
        if rate < 0:
            assert all(kind == "RuntimeError" and "nonnegative_rate" in message
                       for kind, message in run_errors), run_errors
            assert runtime.time() == 0. and runtime.macro_step() == 0
        else:
            assert all(kind == "" for kind, _ in run_errors), run_errors
            assert report is not None and report.accepted_steps == 1
            assert report.rejected_steps == 0
            assert runtime.time() == pytest.approx(.4) and runtime.macro_step() == 1
        after = _states(runtime)

        def check():
            path = tmp_path / ("h05_adversarial_%s_%s.npz" % (reverse, rate))
            np.savez_compressed(path, initial=np.stack(before).reshape(initial.shape),
                                final=np.stack(after).reshape(initial.shape))
            with np.load(path) as saved:
                old, new = saved["initial"], saved["final"]
                np.testing.assert_array_equal(old, initial)
                if rate < 0:
                    np.testing.assert_array_equal(new, old)
                    return
                # Closed-form inverse of the H05 two-by-two BE system, evaluated
                # independently at each cell instead of importing the case oracle.
                a = .4*rate
                expected = np.stack((((1+a)*old[0]+a*old[1])/(1+2*a),
                                     (a*old[0]+(1+a)*old[1])/(1+2*a)))
                np.testing.assert_allclose(new, expected, rtol=0, atol=2e-11)
                np.testing.assert_allclose(new.sum(axis=0), old.sum(axis=0),
                                           rtol=0, atol=2e-11)
                residual = np.stack((new[0]-old[0]+a*(new[0]-new[1]),
                                     new[1]-old[1]-a*(new[0]-new[1])))
                assert np.max(np.abs(residual)) <= 2e-11
                assert np.min(new) >= 0
                if rate == 0:
                    np.testing.assert_array_equal(new, old)
                else:
                    assert np.max(np.abs(new[0]-new[0, 0, 0])) > .1

        _root_check(world, check)
