"""Actual installed public pipeline; ROOT owns Native execution after rebuild."""
from pathlib import Path
import sys
import json
import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
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
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout = collective_call(world, lambda: build(shape, tag_buffer, transfer_kind))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="public tag selection and parent coverage",
        compile_artifact=pops.compile)
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
    for key,value in (("dimension",2),("rank",int(world.rank)),("size",int(world.size)),
                      ("artifact_identity",artifact.artifact_identity.token),("tag_buffer",tag_buffer),
                      ("transfer_kind",transfer_kind),("tag_selection_contract",json.dumps(tag_contract)),
                      ("package",str(Path(pops.__file__).resolve())),("native",str(Path(native.__file__).resolve()))):
        record_property(key,value)
