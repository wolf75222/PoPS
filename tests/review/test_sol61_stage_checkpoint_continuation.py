"""Read-only real native checkpoints; archive comparison is not a native solve."""
import copy
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pops.runtime._checkpoint_manifest import (
    IDENTITY_KEY, MANIFEST_KEY, _identity_from_json,
    _seal_checkpoint_payload_with_identities, inspect_checkpoint_payload_integrity,
)
from pops.runtime._run_manifest import RunManifest
from pops.time import FixedDt
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.integration.runtime.test_public_evolved_original_stage import compare_checkpoint_replay


@pytest.fixture
def checkpoints():
    directory = os.environ.get("POPS_STAGE_CHECKPOINT_EVIDENCE")
    if not directory:
        pytest.skip("ROOT immutable native evidence directory must be explicitly supplied")
    result = []
    for case in range(8):
        root = Path(directory)/"pytest-tmp"/("test_public_evolved_original_s%d" % case)/"evolved-original-stage"
        paths, images, authority = {}, {}, {}
        accepted_run, accepted_restart = None, None
        for phase in ("accepted", "continuous", "replay"):
            path = root/(phase+"-checkpoint.npz")
            with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as z:
                image = {key: z[key].copy() for key in z.files}
            manifest, restart = inspect_checkpoint_payload_integrity(image, runtime_kind="uniform")
            dt = float(image["t"])/(1 if phase == "accepted" else 2)
            strategy = FixedDt(dt)
            run = RunManifest(bind_identity=_identity_from_json(manifest["bind_identity"]),
                continuation_identity=accepted_run if phase == "replay" else None,
                start_time=0. if phase == "accepted" else dt,
                start_macro_step=0 if phase == "accepted" else 1,
                controls={"t_end":float(image["t"]), "max_steps":1, "output_mode":"current-directory",
                    "step_transaction":{"strategy":strategy.to_data(), "controls":strategy.runtime_controls_data({})}})
            assert run.run_identity == _identity_from_json(manifest["run_identity"])
            authority[phase] = {name:_identity_from_json(manifest[name+"_identity"]).token
                for name in ("semantic", "artifact", "bind")}
            authority[phase].update(run=run.run_identity.token, run_manifest=run.to_dict(),
                restart=restart.token, clock=manifest["clock"],
                last_restart=accepted_restart if phase == "replay" else None)
            paths[phase], images[phase] = path, image
            if phase == "accepted":
                accepted_run, accepted_restart = run.run_identity, restart.token
        result.append((paths, images, authority))
    return result


def test_real_eight_native_pairs_have_only_expected_continuation_differences(checkpoints):
    for paths, images, authority in checkpoints:
        report = compare_checkpoint_replay(paths, authority)
        assert report["exact_payload_and_manifest"]
        left, right = images["continuous"], images["replay"]
        assert {key for key in left if left[key].tobytes() != right[key].tobytes()} == {MANIFEST_KEY, IDENTITY_KEY}


@pytest.mark.parametrize("mutation", ("value", "dtype", "shape", "clock", "semantic", "artifact",
    "bind", "lineage", "last_restart", "controls", "run_digest", "restart_token"))
def test_continuation_comparator_refuses_resealed_or_provenance_corruption(checkpoints, tmp_path, mutation):
    paths, images, authority = copy.deepcopy(checkpoints[0])
    assert compare_checkpoint_replay(paths, authority)["exact_payload_and_manifest"]
    image = images["replay"]
    manifest = json.loads(str(image[MANIFEST_KEY]))
    if mutation in {"value", "dtype", "shape"}:
        key = next(key for key in image if key not in {MANIFEST_KEY, IDENTITY_KEY}
            and image[key].dtype == np.float64 and image[key].size > 1)
        if mutation == "value":
            image[key].flat[0] += .125
        elif mutation == "dtype":
            image[key] = image[key].astype(np.float32)
        else:
            image[key] = image[key].reshape(-1)
        if mutation == "shape" and image[key].shape == images["continuous"][key].shape:
            image[key] = image[key].reshape(1, -1)
        image.pop(MANIFEST_KEY)
        image.pop(IDENTITY_KEY)
        restart = _seal_checkpoint_payload_with_identities(image, runtime_kind="uniform",
            semantic=_identity_from_json(manifest["semantic_identity"]),
            artifact=_identity_from_json(manifest["artifact_identity"]),
            bind=_identity_from_json(manifest["bind_identity"]), run=_identity_from_json(manifest["run_identity"]))
        authority["replay"]["restart"] = restart.token
        path = tmp_path/"mutated-checkpoint.npz"
        np.savez(path, **image)
        paths["replay"] = path
    elif mutation == "clock":
        authority["replay"]["clock"]["macro_step"] += 1
    elif mutation in {"semantic", "artifact", "bind"}:
        authority["replay"][mutation] = authority["replay"]["run"]
    elif mutation == "last_restart":
        authority["replay"]["last_restart"] = authority["continuous"]["restart"]
    elif mutation == "restart_token":
        authority["replay"]["restart"] = authority["continuous"]["restart"]
    else:
        run = RunManifest.from_dict(authority["replay"]["run_manifest"])
        if mutation == "run_digest":
            authority["replay"]["run"] = authority["continuous"]["run"]
        else:
            controls = dict(run.controls)
            if mutation == "controls":
                controls["max_steps"] += 1
            altered = RunManifest(bind_identity=run.bind_identity, start_time=run.start_time,
                start_macro_step=run.start_macro_step, controls=controls,
                continuation_identity=None if mutation == "lineage" else run.continuation_identity)
            authority["replay"]["run_manifest"] = altered.to_dict()
            authority["replay"]["run"] = altered.run_identity.token
    with pytest.raises((AssertionError, ValueError, TypeError)):
        compare_checkpoint_replay(paths, authority)
