"""Receive the public product-support map, not a Vlasov/BGK simulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_generic_physical_maps import resolve_generic_maps
from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

NAMES = ("population", "integral", "weighted", "extended")
CASES = ((4, 3, 3), (2, 5, 1), (7, 3, 5))


def resolve_product_chain(directory, nx, nv, width, *, reverse=False, provider_factory=None):
    # These are complete authored quadrature weights, not hidden normalization.
    weights = tuple((-1) ** j * (j + 1) for j in range(nv))
    return resolve_generic_maps(directory, nx=nx, nv=nv,
        components=tuple("quantity_%d" % c for c in range(width)),
        weights=weights, reverse=reverse,
        **({} if provider_factory is None else {"provider_factory": provider_factory})), weights


def input_and_expected(nx, nv, width, weights):
    x = np.arange(nx, dtype=np.float64)[None, :, None]
    j = np.arange(nv, dtype=np.float64)[None, None, :]
    c = np.arange(width, dtype=np.float64)[:, None, None]
    population = (c + 1) * (x + 1) + (c + 2) * j + (-1.) ** c * x * j
    # Independent affine moment sums: no call to a provider/helper, no reshape summation.
    zeroth = sum(weights)
    first = sum(j * w for j, w in enumerate(weights))
    weighted = ((c + 1) * (x + 1) * zeroth + ((c + 2) + (-1.) ** c * x) * first)
    integral = 4 * (c + 1) * (x + 1) + 2 * (nv - 1) * ((c + 2) + (-1.) ** c * x)
    expected = (population, integral.transpose(0, 2, 1), weighted.transpose(0, 2, 1),
                np.broadcast_to(weighted, (width, nx, nv)).copy())
    initial = dict(zip(NAMES, (population, np.full((width, 1, nx), -101.),
                              np.full((width, 1, nx), -103.), np.full((width, nx, nv), -107.)), strict=True))
    return initial, expected


def _load_published_provider(requirement, directory):
    """Peers authenticate the elected source publication without rewriting it."""
    from pops import interfaces
    from pops.external import SourceComponentPackage, load
    from pops.mesh.layout_mapping import NativeLayoutMapping
    from pops.mesh.native_physical_mapping import _native_source
    root = Path(directory) / requirement.qualified_id.rsplit("::", 1)[-1]
    package = load(root / "physical-map.pops.json")
    if type(package) is not SourceComponentPackage:
        raise TypeError("product map requires the elected source component package")
    component = package.require("map", interface=interfaces.Transfer)()
    manifest = component.component_manifest
    if manifest.to_data()["signature"]["physical_map"] != requirement.physical_map.to_data():
        raise ValueError("published product support differs from the local exact map")
    if (root / "physical_map.cpp").read_bytes() != _native_source(requirement.physical_map, manifest):
        raise ValueError("published product source differs from the authenticated common lowering")
    return NativeLayoutMapping(component, (requirement,))


def _prepare_artifact(world, base, nx, nv, width, reverse):
    base = collective_directory(world, base)
    providers = base / "providers"
    root = world is None or int(world.rank) == 0
    resolved = collective_call(world, lambda: resolve_product_chain(
        providers, nx, nv, width, reverse=reverse)[0] if root else None)
    if world is None:
        artifact = collective_call(world, lambda: pops.compile(resolved))
    else:
        # The root has finished publishing all source files before peers read them.
        resolved = collective_call(world, lambda: resolved if root else resolve_product_chain(
            providers, nx, nv, width, reverse=reverse, provider_factory=_load_published_provider)[0])
        artifact = compile_resolved_plan_once(world, resolved,
            route="m19-product-%d-%d-%d-%s" % (nx, nv, width, reverse), compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    with collective_check(world):
        assert context.communicator.identity == ("serial" if world is None else "MPI_COMM_WORLD")
        if world is not None:
            assert (context.communicator.handle.rank, context.communicator.handle.size) == (world.rank, world.size)
    directory = collective_directory(world, base / "m19-product-receipts")
    return artifact, context, directory


def _checkpoint_target(world, directory, phase):
    with collective_check(world):
        if not isinstance(phase, str) or not phase or Path(phase).name != phase:
            raise ValueError("product checkpoint phase must be an exact basename")
        target = (Path(directory) / phase).resolve()
        if not target.is_relative_to(Path(directory).resolve()):
            raise ValueError("product checkpoint escapes the shared receipt directory")
    if world is not None:
        from pops._native_collectives import allgather_value
        paths = allgather_value(world, str(target))
        with collective_check(world):
            if any(path != str(target) for path in paths):
                raise ValueError("product checkpoint path differs across MPI ranks")
    return target


def _save(world, runtime, artifact, directory, phase):
    target = _checkpoint_target(world, directory, phase)
    states = state_snapshots(runtime, world, NAMES)
    ownership = {name: collective_call(world, lambda name=name: runtime.local_boxes(name)) for name in NAMES}
    if world is None:
        owners = (ownership,)
    else:
        from pops._native_collectives import allgather_value
        owners = collective_call(world, lambda: allgather_value(world, ownership))
    checkpoint = collective_call(world, lambda: runtime.checkpoint(target))
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            np.savez_compressed(directory / (phase + "-state.npz"), **dict(zip(NAMES, states, strict=True)))
            checkpoint = Path(checkpoint)
            receipt = dict(schema="sol61.m19-product-state@1", phase=phase,
                time=runtime.time(), time_hex=float(runtime.time()).hex(), macro_step=runtime.macro_step(),
                artifact_identity=artifact.artifact_identity.token, bind_identity=runtime.bind_identity.token,
                mapping_counts=runtime._executor.mapping_report(), local_boxes_by_rank=owners,
                checkpoint=checkpoint.name, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                saved_state_sha256=hashlib.sha256((directory / (phase + "-state.npz")).read_bytes()).hexdigest())
            (directory / (phase + "-receipt.json")).write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return checkpoint, states


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("nx,nv,width", CASES)
@pytest.mark.parametrize("reverse", (False, True))
def test_native_product_reduce_lift_restart(nx, nv, width, reverse, tmp_path, record_property,
                                           isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension, selected_native_module
    from pops.codegen._native_mpi import native_mpi_communicator
    native = select_native_dimension(2)
    world = native.mpi_world() if native_mpi_communicator(native) == "MPI_COMM_WORLD" else None
    artifact, context, directory = _prepare_artifact(world, tmp_path, nx, nv, width, reverse)
    with collective_check(world):
        assert artifact.resolved_dimension == 2
        directory.mkdir(exist_ok=True)
        native = selected_native_module(required=True)
        assert int(native.__native_dimension__) == 2
        native_path = Path(native.__file__).resolve()
        identity = dict(schema="sol61.m19-product-native@1", native_path=str(native_path),
            native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(), abi_key=str(native.abi_key()),
            python_package=str(Path(pops.__file__).resolve()), native_capabilities=dict(native.module_capabilities("module")),
            artifact_identity=artifact.artifact_identity.token, platform=artifact.platform_manifest.to_data(),
            compiled_components=artifact._current_component_evidence(), compiled_plan=artifact.plan._payload(),
            dimensions=dict(nx=nx, nv=nv, components=width), physical_equations="finite reduction and constant extension only",
            source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                str(Path(resolve_generic_maps.__code__.co_filename).resolve()):
                    hashlib.sha256(Path(resolve_generic_maps.__code__.co_filename).read_bytes()).hexdigest()})
        if world is None or int(world.rank) == 0:
            (directory / "provenance.json").write_text(json.dumps(identity, indent=2, allow_nan=False) + "\n")
    weights = tuple((-1) ** j * (j + 1) for j in range(nv))
    initial, expected = input_and_expected(nx, nv, width, weights)
    instance = collective_call(world, lambda: pops.bind(artifact, initial_state=initial, resources={"execution_context": context}))
    _save(world, instance, artifact, directory, "initial")
    collective_call(world, lambda: pops.run(instance, t_end=.01, max_steps=1, console=False))
    checkpoint, states = _save(world, instance, artifact, directory, "accepted")
    with collective_check(world):
        for actual, exact in zip(states, expected, strict=True):
            np.testing.assert_allclose(actual, exact, rtol=0, atol=2e-12)
        assert instance.time() == .01 and instance.macro_step() == 1
        assert set(instance._executor.mapping_report().values()) == {1}
    restarted = collective_call(world, lambda: pops.bind(artifact, initial_state=initial, resources={"execution_context": context}))
    collective_call(world, lambda: restarted.restart(checkpoint))
    _, restored = _save(world, restarted, artifact, directory, "restored")
    with collective_check(world):
        assert restarted.time() == .01 and restarted.macro_step() == 1
        assert restarted._executor.mapping_report() == instance._executor.mapping_report()
        for actual, exact in zip(restored, states, strict=True):
            np.testing.assert_array_equal(actual, exact)
    collective_call(world, lambda: pops.run(restarted, t_end=.02, max_steps=1, console=False))
    _, replayed = _save(world, restarted, artifact, directory, "replayed")
    with collective_check(world):
        for actual, exact in zip(replayed, expected, strict=True):
            np.testing.assert_allclose(actual, exact, rtol=0, atol=2e-12)
        assert restarted.time() == .02 and restarted.macro_step() == 2
        assert set(restarted._executor.mapping_report().values()) == {2}
        record_property("native_dimension", 2)
        record_property("mpi_rank", 0 if world is None else int(world.rank))
        record_property("mpi_size", 1 if world is None else int(world.size))
        record_property("native_sha256", identity["native_sha256"])
        record_property("saved_receipts", str(directory))
