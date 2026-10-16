"""Typed solved-field histories cross actual 2:1 AMR regrids by authored equations."""

import numpy as np
import pops
import pytest
from examples.api_v02.method_temporal_field import build_temporal_field
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import (
    collective_call,
    collective_check,
    collective_attempt,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.runtime.test_api_v02_history_transfer_native import (
    _image,
    _same,
    _sample_rows,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200


def _check_projected_observation(image, world):
    _, samples = _sample_rows(image["metadata"]["observed", 1][0])
    # All slots were reconstructed from the parent ring, including the recycled one.
    with collective_check(world):
        assert len(samples) == 3
        assert all(row["kind"] == 3 for row in samples), samples
        assert len({row["digest"] for row in samples}) == 3
        fine = image["states"][1]
        if fine is not None:
            live = np.asarray(fine).reshape(3, 32, 32)[1] > 0
            assert live.any()
            for slot, row in enumerate(samples):
                observed = np.asarray(image["rings"]["observed", 1, slot]).reshape(32, 32)
                # Independent analytic PDE solution: -Delta(phi)+phi=1+t^2.
                # The history authority labels invocation starts; the solve is at its end.
                expected = 1 + (row["time"] + row["interval"]) ** 2
                np.testing.assert_allclose(observed[live], expected, rtol=0, atol=3.0e-9)
            assert abs(samples[1]["time"] + samples[1]["interval"] - image["time"]) < 1.0e-15
            # CopyCurrent would put the newest endpoint in every retained slot.
            assert (
                abs(
                    (samples[1]["time"] + samples[1]["interval"]) ** 2
                    - (samples[2]["time"] + samples[2]["interval"]) ** 2
                )
                > 1.0e-4
            )


def test_earned_output_history_uses_authored_transfer_at_real_refinement_ratio(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(
        "output_history_2to1", builder=build_temporal_field, regrid_observation=True
    )
    runtime = _bind(spec, artifact, world)
    before_patches = tuple(runtime.patch_boxes())
    collective_call(world, lambda: pops.run(runtime, t_end=0.16, max_steps=4, console=False))
    image = _image(runtime, world)
    with collective_check(world):
        assert tuple(runtime.patch_boxes()) != before_patches
    _check_projected_observation(image, world)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(
        world, lambda: runtime.checkpoint(str(directory / "output-history"))
    )
    restarted = _bind(spec, artifact, world)
    collective_call(world, lambda: restarted.restart(checkpoint))
    _same(image, _image(restarted, world), world)
    for active in (runtime, restarted):
        collective_call(
            world, lambda active=active: pops.run(active, t_end=0.18, max_steps=1, console=False)
        )
    _same(_image(runtime, world), _image(restarted, world), world)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks", 1 if world is None else int(world.size))


def test_missing_output_temporal_law_rejects_new_earned_coverage_atomically(
    isolated_native_cache, native_cxx, kokkos_root
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(
        "output_history_no_law",
        builder=build_temporal_field,
        regrid_observation=True,
        observation_transfer=False,
    )
    runtime = _bind(spec, artifact, world)  # cold 2:1 bootstrap remains legitimate
    collective_call(world, lambda: pops.run(runtime, t_end=0.12, max_steps=3, console=False))
    before = _image(runtime, world)
    _, failures = collective_attempt(
        world, lambda: pops.run(runtime, t_end=0.16, max_steps=1, console=False)
    )
    with collective_check(world):
        assert failures and all(failures), failures
        assert all("history" in row[1].lower() and "clock" in row[1].lower() for row in failures), (
            failures
        )
    _same(before, _image(runtime, world), world)
