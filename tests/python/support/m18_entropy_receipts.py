"""Save actual native M18 buffers; this helper never creates external owner pins."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_entropy_snapshot(world, runtime, artifact, directory, phase, *, failures=()):
    dual, target = state_snapshots(runtime, world, ("dual", "target"))
    with collective_check(world):
        assert len(runtime._layout_plan.layouts) == 1
        layout = runtime._layout_plan.layouts[0]
    geometry = collective_call(world, lambda: runtime._snapshot_builder._geometry(layout, 0))
    local = {}
    for block in ("dual", "target"):
        local[block] = collective_call(world, lambda block=block: runtime.local_boxes(block))
    if world is None:
        ownership = (local,)
    else:
        from pops._native_collectives import allgather_value
        ownership = collective_call(world, lambda: allgather_value(world, local))
    with collective_check(world):
        time, step = runtime.time(), runtime.macro_step()
        temporal = runtime._executor._temporal_restart_state.to_data()
        bind_identity = runtime.bind_identity.token
        run_identity = None if runtime.last_run_identity is None else runtime.last_run_identity.token
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / phase))
    result = None
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            assert tuple(geometry.cell_shape) == (4, 5)
            arrays = dict(dual=np.asarray(dual).reshape(3, 4, 5).copy(),
                          target=np.asarray(target).reshape(3, 4, 5).copy(),
                          cell_volumes=np.asarray(geometry.cell_volumes).copy(),
                          coverage=np.asarray(geometry.coverage).copy(),
                          valid_cells=np.asarray(geometry.valid_cells).copy())
            np.savez_compressed(directory / (phase + "-state.npz"), **arrays)
            checkpoint = Path(checkpoint)
            result = dict(phase=phase, time=time, time_hex=float(time).hex(), macro_step=step,
                          artifact_identity=artifact.artifact_identity.token,
                          bind_identity=bind_identity, run_identity=run_identity,
                          dimension=artifact.resolved_dimension,
                          ranks=1 if world is None else int(world.size),
                          raw_state_shapes={"dual":list(np.asarray(dual).shape),
                                            "target":list(np.asarray(target).shape)},
                          storage="component,y,x", geometry=geometry.to_data(),
                          local_boxes_by_rank=ownership, temporal=temporal,
                          failures=failures, checkpoint=checkpoint.name,
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            _json(directory / (phase + "-receipt.json"), result)
    return checkpoint, result


def save_entropy_provenance(world, artifact, directory, *, fixture, example):
    from pops._native_selector import selected_native_module
    native = selected_native_module(required=True)
    with collective_check(world):
        native_path = Path(native.__file__).resolve()
        packages = []
        for block in artifact.blocks:
            library = ctypes.CDLL(str(block.model.so_path))
            version = library.pops_native_system_package_abi_version
            version.argtypes = []
            version.restype = ctypes.c_int
            packages.append(dict(block=block.name, abi_version=version(),
                                 binary_sha256=hashlib.sha256(Path(block.model.so_path).read_bytes()).hexdigest()))
        identity = dict(native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                        native_path=str(native_path), abi_key=str(native.abi_key()),
                        native_capabilities=dict(native.module_capabilities("module")),
                        native_dimension=int(native.__native_dimension__),
                        system_packages=packages)
    if world is None:
        identities = (identity,)
    else:
        from pops._native_collectives import allgather_value
        identities = collective_call(world, lambda: allgather_value(world, identity))
    with collective_check(world):
        assert all(row == identities[0] for row in identities)
        if world is None or int(world.rank) == 0:
            _json(directory / "provenance.json", dict(
                schema="sol61.m18-native-provenance@1", artifact_identity=artifact.artifact_identity.token,
                dimension=artifact.resolved_dimension, ranks=len(identities),
                platform=artifact.platform_manifest.to_data(), native_by_rank=identities,
                compiled_plan=artifact.plan._payload(),
                compiled_components=artifact._current_component_evidence(),
                program_ir=artifact.program.program._serialize(include_provenance=False),
                program_ir_hash=artifact.program.program._ir_hash(),
                source_sha256={"fixture":hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
                               "example":hashlib.sha256(Path(example).read_bytes()).hexdigest(),
                               "snapshot":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                constants=dict(nodes=[-1.,-.5,0.,.5,1.], weights=[.1,.2,.4,.2,.1],
                               basis=[[1.,1.,1.,1.,1.],[-1.,-.5,0.,.5,1.],[1.,.25,0.,.25,1.]],
                               cells=[5,4], dt=.01, unknowns=3, max_iterations=12,
                               original_residual_tolerance=2.e-11, max_backtracks=16,
                               minimum_step=2.**-16, safeguard="backtracking"),
                safe_rebind="new bind, same artifact/context/zero seed, restored interior input; no target commit"))
    return identity
