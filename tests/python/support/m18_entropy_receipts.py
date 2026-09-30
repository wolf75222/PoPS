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


def _owner_path(path):
    path = Path(path).absolute()
    # Inspect the supplied name before resolve erases symbolic aliases.
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("M18 execution origin path aliases are forbidden")
    return path.resolve(strict=False)


def _owner_file(path):
    path = _owner_path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("M18 execution origin must be an actual regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _fixture_commit(fixture):
    import re
    import subprocess
    parent = Path(fixture).resolve(strict=True).parent
    result = subprocess.run(["git", "-C", str(parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    commit = result.stdout.strip()
    if re.fullmatch("[0-9a-f]{40}", commit) is None:
        raise ValueError("M18 execution fixture checkout has no exact source commit")
    return commit


def _execution_owner_record(artifact, fixture):
    import sys
    import pops
    from pops._native_selector import selected_native_module
    from pops.codegen.toolchain import pops_include
    prefix = Path(sys.prefix).resolve()
    python = _owner_file(pops.__file__)
    sdk = _owner_file(Path(pops_include()) / "pops_headers.manifest")
    native = _owner_file(selected_native_module(required=True).__file__)
    for leaf in (python, sdk, native):
        if not Path(leaf["path"]).is_relative_to(prefix):
            raise ValueError("M18 execution requires actual installed Python/SDK/native origins")
    packages = {}
    for block in artifact.blocks:
        if block.name not in ("dual", "target") or block.name in packages:
            raise ValueError("M18 execution System package inventory differs")
        packages[block.name] = _owner_file(block.model.so_path)
    if set(packages) != {"dual", "target"}:
        raise ValueError("M18 execution requires dual and target System package paths")
    programs = tuple(row.program for row in artifact.layout_programs) or (artifact.program,)
    generated = {}
    incomplete = False
    for program in programs:
        paths = getattr(program, "generated_sources", ()) if program is not None else ()
        seen = set()
        for path in paths:
            path = _owner_path(path)
            if path in seen:
                raise ValueError("M18 Program generated source inventory contains duplicates")
            seen.add(path)
            if not path.is_file():
                incomplete = True
                continue
            leaf = _owner_file(path)
            if Path(leaf["path"]).suffix != ".cpp":
                raise ValueError("M18 generated source metadata must name actual cpp files")
            if leaf["path"] in generated and generated[leaf["path"]] != leaf:
                raise ValueError("M18 shared generated source changed during capture")
            generated[leaf["path"]] = leaf
    cpp = None if incomplete or not generated else [generated[path] for path in sorted(generated)]
    return dict(schema="sol61.m18-execution-owner-metadata@1", source_commit=_fixture_commit(fixture),
                python_package=python, sdk=sdk, native=native, system_packages=packages, generated_cpp=cpp)


def _owner_consensus(records):
    if not records:
        raise ValueError("M18 execution owner has no rank records")
    base = {key: value for key, value in records[0].items() if key != "generated_cpp"}
    if any({key: value for key, value in row.items() if key != "generated_cpp"} != base for row in records):
        raise ValueError("M18 execution origins differ across ranks")
    # A verified peer cache load does not retain generated_sources metadata. Only
    # direct paths actually captured by an executing compiler owner are published.
    stored = [row["generated_cpp"] for row in records if row["generated_cpp"] is not None]
    if stored and any(row != stored[0] for row in stored):
        raise ValueError("M18 recorded generated source inventories differ across ranks")
    return dict(base, generated_cpp=stored[0] if stored else None)


def save_entropy_execution_owner(world, artifact, directory, *, fixture):
    """Capture actual execution origins outside the closed scientific phase inventory.

    This record is execution-owner-attested; independent ROOT approval remains
    necessary. It is not an external seal, scientific result or source/CPP mapping.
    """
    record = collective_call(world, lambda: _execution_owner_record(artifact, fixture))
    if world is None:
        records = (record,)
    else:
        from pops._native_collectives import allgather_value
        records = collective_call(world, lambda: allgather_value(world, record))
    agreed = collective_call(world, lambda: _owner_consensus(records))
    with collective_check(world):
        destination = Path(directory).resolve().parent / "m18-execution-owner-metadata.json"
    if world is not None:
        paths = collective_call(world, lambda: allgather_value(world, str(destination)))
    with collective_check(world):
        # The compiler owner's explicit CPP inventory has now been shared. Each
        # peer reopens it, including cache peers without local source metadata.
        for row in agreed["generated_cpp"] or ():
            if _owner_file(row["path"]) != row:
                raise ValueError("M18 elected generated source changed before recording")
        # Reread actual providers and their bytes, never replace initial authority
        # with a new hash. This also detects changed paths/selected providers.
        if _execution_owner_record(artifact, fixture) != record:
            raise ValueError("M18 execution origins changed before recording")
    if world is not None:
        confirmed = collective_call(world, lambda: allgather_value(world, agreed))
        with collective_check(world):
            if any(path != str(destination) for path in paths) or any(row != agreed for row in confirmed):
                raise ValueError("M18 execution owner publication differs across ranks")
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            with destination.open("x") as output:
                output.write(json.dumps(agreed, indent=2, allow_nan=False) + "\n")
    return destination
