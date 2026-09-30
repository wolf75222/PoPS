"""Save genuine public ALE phases; collective operations precede root-only IO."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots


def save_moving_snapshot(world, runtime, artifact, identity, frame, directory, phase):
    root = world is None or int(world.rank) == 0
    state, = state_snapshots(runtime, world, ("fluid",))
    geometry = collective_call(world, lambda: runtime._executor._s._output_moving_geometry_snapshot(identity, frame))
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / phase))
    result = None
    with collective_check(world):
        if root:
            checkpoint = Path(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as image:
                wire = image["program_exchange_state"].tobytes()
                offsets = tuple(int(value) for value in image["program_exchange_offsets"])
                assert len(offsets) == (1 if world is None else int(world.size)) + 1
                assert offsets[0] == 0 and offsets[-1] == len(wire)
                images = tuple(wire[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True))
            assert all(image.startswith(b"POPSEX04") for image in images)
            result = dict(state=state, node_coordinates=np.asarray(geometry["node_coordinates"]).copy(),
                          cell_volumes=np.asarray(geometry["cell_volumes"]).copy(),
                          generation=int(geometry["generation"]), time=float(runtime.time()),
                          step=int(runtime.macro_step()))
            np.savez(directory / (phase + "-state.npz"), **result)
            receipt = dict(phase=phase, artifact=artifact.artifact_identity.token,
                           platform=artifact.platform_manifest.to_data(), dimension=artifact.resolved_dimension,
                           moving_identity=identity, physical_frame=frame,
                           rank=0, size=1 if world is None else int(world.size),
                           time=result["time"], macro_step=result["step"], generation=result["generation"],
                           checkpoint=str(checkpoint),
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           image_sha256=[hashlib.sha256(image).hexdigest() for image in images])
            (directory / (phase + "-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
            result["images"] = images
    return checkpoint, result


def assert_exact_moving_snapshot(before, after):
    for name in ("state", "node_coordinates", "cell_volumes"):
        assert before[name].dtype == after[name].dtype and before[name].shape == after[name].shape
        assert before[name].tobytes() == after[name].tobytes(), name
    for name in ("generation", "time", "step", "images"):
        assert before[name] == after[name], name
