"""Actual installed public pipeline; ROOT owns Native execution after rebuild."""
from pathlib import Path
import json
import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile, rank_and_size
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.support.tag_selection_case import build, DT
from tests.review.sol61_tag_selection_oracle import physical_tags, receive
from tests.review.sol61_amr_full_carrier_offline import decode
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.tag_phase_capture import capture


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("shape", ((8, 8), (8, 12)))
@pytest.mark.parametrize("tag_buffer", (0, 1))
@pytest.mark.parametrize("transfer_kind", ("linear", "injection"))
def test_public_tag_buffer_preserves_periodic_selection_and_parent_ghosts(
        shape, tag_buffer, transfer_kind, tmp_path, record_property, isolated_native_cache):
    del isolated_native_cache
    native, world = native_profile(scope="public tag selection")
    rank, size = rank_and_size(world)
    record_property("native_abi_version", native.__abi_version__)
    record_property("native_profile", "MPI" if world is not None else "Serial")
    case, layout = collective_call(world, lambda: build(shape, tag_buffer, transfer_kind))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = (pops.compile(resolved) if world is None else compile_resolved_plan_once(
        world, resolved, route="public tag selection and parent coverage", compile_artifact=pops.compile))
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context":context}))
    tag_contract = collective_call(world, lambda: runtime._executor.checkpoint_tag_selection_contract())
    capture_directory = None
    if shape == (8,8) and tag_buffer == 0 and transfer_kind == "linear":
        capture_directory = collective_directory(world,tmp_path/"tag-phase-capture")
        path=capture(world,runtime,artifact,native,capture_directory,"bound",tag_contract,pops.__file__)
        record_property("tag_bound_capture_index",str(path))
    before = collective_call(world, lambda: tuple(runtime.patch_boxes()))
    coarse = collective_call(world, lambda: runtime.block_level_state_global("marker", 0))
    with collective_check(world):
        coarse = np.asarray(coarse).reshape(2, shape[1], shape[0])
        np.testing.assert_array_equal(coarse[1] > .7, physical_tags(shape))
        coarse_active, fine = receive(before, shape, tag_buffer)
        assert runtime.n_levels() == 2
    actual_coarse = collective_call(world, lambda: composite_active_mask(runtime, 0, refinement_ratio=2))
    actual_fine = collective_call(world, lambda: composite_active_mask(runtime, 1, refinement_ratio=2))
    with collective_check(world):
        np.testing.assert_array_equal(actual_coarse, coarse_active)
        np.testing.assert_array_equal(actual_fine, fine)
    # Exercise actual spatial halo/coarse-fine machinery, even though this PDE is stationary.
    report = collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    if capture_directory is not None:
        path=capture(world,runtime,artifact,native,capture_directory,"accepted",tag_contract,pops.__file__)
        record_property("tag_accepted_capture_index",str(path))
    after = collective_call(world, lambda: tuple(runtime.patch_boxes()))
    image = collective_call(world, lambda: bytes(runtime._executor.checkpoint_state_carriers()))
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        assert after == before
        archive = decode(np.frombuffer(image, dtype=np.uint8))
        assert archive["dim"] == 2 and archive["real"] == 64 and archive["shard"] == -1
        assert archive["blocks"] == ["marker"] and archive["levels"] == 2
        for patch in archive["patches"]:
            assert patch["components"] == 2
            assert all(glo <= lo-1 and ghi >= hi+1 for lo,hi,glo,ghi in patch["axes"])
            values = np.array(patch["bits"], dtype=np.uint64).view(np.float64).reshape(2, -1)
            assert np.isfinite(values).all()
            # Constant component proves the full grown carrier was filled consistently;
            # it includes periodic and coarse/fine ghosts, not just valid NPZ cells.
            np.testing.assert_array_equal(values[0], np.ones(values.shape[1]))
        assert (runtime.time(), runtime.macro_step()) == (DT, 1)
    for key,value in (("dimension",2),("rank",rank),("size",size),
                      ("artifact_identity",artifact.artifact_identity.token),("tag_buffer",tag_buffer),
                      ("transfer_kind",transfer_kind),("tag_selection_contract",json.dumps(tag_contract)),
                      ("package",str(Path(pops.__file__).resolve())),("native",str(Path(native.__file__).resolve()))):
        record_property(key,value)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("tag_buffer,coarsen,cadence,max_levels", (
    (0, False, 1, 2),
    (1, False, 1, 2),
    (1, True, 1, 2),
    (1, False, 2, 2),
    (1, False, 1, 3),
))
def test_stationary_regrids_do_not_redilate_retained_tag_buffer(
        tag_buffer, coarsen, cadence, max_levels, record_property, isolated_native_cache):
    """A fixed periodic predicate must retain its selected hierarchy and conservative state."""
    del isolated_native_cache
    native, world = native_profile(scope="public tag selection")
    rank, size = rank_and_size(world)
    record_property("native_abi_version", native.__abi_version__)
    record_property("native_profile", "MPI" if world is not None else "Serial")
    shape = (16, 12)
    case, layout = collective_call(world, lambda: build(
        shape, tag_buffer, "linear", regrid_every=cadence, coarsen=coarsen,
        max_levels=max_levels))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = (pops.compile(resolved) if world is None else compile_resolved_plan_once(
        world, resolved, route="stationary AMR buffer regrid regression", compile_artifact=pops.compile))
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context": context}))
    initial_boxes = collective_call(world, lambda: tuple(runtime.patch_boxes()))
    with collective_check(world):
        assert runtime.n_levels() == max_levels
        if max_levels == 2:
            receive(initial_boxes, shape, tag_buffer)

    def snapshot():
        states = []
        integral = np.zeros(2)
        for level in range(max_levels):
            scale = 2**level
            state = np.asarray(runtime.block_level_state_global("marker", level)).reshape(
                2, shape[1]*scale, shape[0]*scale).copy()
            active = composite_active_mask(runtime, level, refinement_ratio=2)
            integral += state[:, active].sum(axis=1)/(shape[0]*shape[1]*scale**2)
            states.append(state)
        return states, integral

    initial_states, initial_integral = collective_call(world, snapshot)
    for step in range(1, 5):
        report = collective_call(world, lambda: pops.run(
            runtime, t_end=step*DT, max_steps=1, console=False))
        boxes = collective_call(world, lambda: tuple(runtime.patch_boxes()))
        states, integral = collective_call(world, snapshot)
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert runtime.macro_step() == step
            assert boxes == initial_boxes, f"fixed tags changed coverage at accepted step {step}"
            for actual, expected in zip(states, initial_states, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0., atol=2.e-14)
            np.testing.assert_allclose(integral, initial_integral, rtol=0., atol=2.e-14)
            np.testing.assert_allclose(integral[0], 1., rtol=0., atol=2.e-14)
    for key, value in (("dimension", 2), ("rank", rank), ("size", size),
                       ("tag_buffer", tag_buffer), ("coarsen", coarsen), ("cadence", cadence),
                       ("levels", max_levels), ("artifact_identity", artifact.artifact_identity.token),
                       ("package", str(Path(pops.__file__).resolve())),
                       ("native", str(Path(native.__file__).resolve()))):
        record_property(key, value)
