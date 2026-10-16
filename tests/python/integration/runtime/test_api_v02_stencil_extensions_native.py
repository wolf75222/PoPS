"""Prepared physical and coarse/fine extensions are consumed by real C++ faces."""
from dataclasses import replace

import numpy as np
import pops
import pytest

from examples.api_v02.method_stencil_extensions import build_stencil_extensions
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.amr_snapshots import level_valid_mask
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200


def _metrics(spec, runtime, world):
    rows = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        owners = [row for row in rows if row.get("metrics_owner", True)]
        assert len(owners) == 1 and owners[0]["passed"], rows
    return rows


def _levels(runtime, world):
    return [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(runtime.n_levels())]


@pytest.mark.parametrize("sample", ("state", "cell"))
def test_native_stencil_uses_outflow_source_extension(sample, isolated_native_cache,
        native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("stencil_outflow_"+sample, builder=build_stencil_extensions,
        sample=sample, amr=False, periodic=False)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=2, console=False))
    rows = _metrics(spec, runtime, world)
    with collective_check(world):
        for row in rows:
            if row.get("metrics_owner", True):
                assert row["reference_error"] < 3.e-13
                assert row["change"] > 1.e-4
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("metrics", str(rows))


def test_native_amr_cell_sample_keeps_exact_stage_and_cf_extension(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    images = []; layouts = []; identities = []
    for sample in ("state", "cell"):
        spec, artifact, world = _compile("stencil_amr_"+sample, builder=build_stencil_extensions,
                                         sample=sample)
        runtime = _bind(spec, artifact, world)
        collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=2, console=False))
        rows = _metrics(spec, runtime, world)
        images.append(_levels(runtime, world)); layouts.append(tuple(runtime.patch_boxes()))
        identities.append(artifact.artifact_identity.hexdigest)
        with collective_check(world):
            assert runtime.n_levels() == 2
            assert all(row["active_counts"][0] and row["active_counts"][1]
                       for row in rows if row.get("metrics_owner", True))
    with collective_check(world):
        assert layouts[0] == layouts[1]
        for direct, sampled in zip(*images, strict=True):
            if direct is not None:
                np.testing.assert_array_equal(sampled, direct)
    record_property("artifact_identities", str(identities))


def test_native_solved_field_has_nonzero_cf_halo_freshness_and_restart(tmp_path,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    # A constant nonzero solution is annihilated by a divergence, including every
    # coarse/fine face. A zero/uninitialized ghost creates an O(dt/dx) defect.
    constant, artifact, world = _compile("stencil_field_constant", builder=build_stencil_extensions,
                                          sample="field", offset=1., amplitude=0.)
    runtime = _bind(constant, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=constant.t_end, max_steps=2, console=False))
    constant_rows = _metrics(constant, runtime, world)
    images = []; layouts = []; field_changes = []
    for offset in (1., 2.):
        spec = collective_call(world, lambda offset=offset:
            build_stencil_extensions(sample="field", offset=offset))
        # Only data vary. Bind them through the subjects that authored this
        # Artifact; a fresh Case's handles are not aliases of those subjects.
        with collective_check(world):
            assert spec.subjects.keys() == constant.subjects.keys()
            assert spec.dt == constant.dt and spec.t_end == constant.t_end
        spec = replace(constant, invariants=spec.invariants, initial_values={
            constant.subjects[name]: spec.initial_values[subject].copy()
            for name, subject in spec.subjects.items()
        })
        runtime = _bind(spec, artifact, world)
        collective_call(world, lambda: pops.run(runtime, t_end=spec.dt, max_steps=1, console=False))
        resumed = None
        if offset == 1.:
            directory = collective_directory(world, tmp_path)
            checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"stencil-field")))
            resumed = _bind(spec, artifact, world)
            collective_call(world, lambda: resumed.restart(checkpoint))
        collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=1, console=False))
        _metrics(spec, runtime, world)
        image = _levels(runtime, world)
        images.append(image); layouts.append(tuple(runtime.patch_boxes()))
        # Both SSPRK stages run distinct field solves; the second consumed field
        # must change on genuinely evolving, spatially varying data.
        changes = []
        for level in range(runtime.n_levels()):
            start = collective_call(world, lambda level=level:
                runtime.history_global("start_potential", level, 1))
            end = collective_call(world, lambda level=level:
                runtime.history_global("end_potential", level, 1))
            if start is not None:
                difference = abs(np.asarray(end)-np.asarray(start))
                changes.append(float(np.nanmax(difference)))
        field_changes.append(changes)
        if resumed is not None:
            collective_call(world, lambda: pops.run(resumed, t_end=spec.t_end, max_steps=1, console=False))
            resumed_image = _levels(resumed, world)
            left = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
            right = collective_call(world, lambda: bytes(resumed._executor.program_accepted_state()))
            with collective_check(world):
                assert left and left == right
                for a, b in zip(image, resumed_image, strict=True):
                    if a is not None:
                        np.testing.assert_array_equal(a, b)
    with collective_check(world):
        assert layouts[0] == layouts[1]
        for level, (base, shifted) in enumerate(zip(*images, strict=True)):
            if base is not None:
                # Dense gathers represent missing fine-level patches by finite
                # zeros. The base domain and exact native patch boxes define support.
                support = level_valid_mask(runtime, level, refinement_ratio=2)
                a, b = np.asarray(base).reshape(support.shape), np.asarray(shifted).reshape(support.shape)
                assert support.any()
                assert np.isfinite(a[support]).all() and np.isfinite(b[support]).all()
                np.testing.assert_allclose(b[support]-a[support], 1., rtol=0, atol=3.e-10)
        if any(field_changes):
            assert all(max(changes) > 1.e-9 for changes in field_changes), field_changes
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("constant_metrics", str(constant_rows))
    record_property("executed_field_changes", str(field_changes))
