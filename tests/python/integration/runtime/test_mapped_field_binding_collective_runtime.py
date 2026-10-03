"""Prospective SDK27 MPI2 DTO-refusal witness; no Native result is implied."""
import io
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.support.m19_thermal_consumed_case import build, reference, NAMES
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_m19_thermal_consumed_runtime import save_json, pin


def mutate_dto(spec, execution, kind, rank):
    """Copy only the rank-local argument; preserve the real other-rank DTO."""
    spec, execution = dict(spec), dict(execution)
    if rank == 1:
        if kind == "bool": spec["mapped_field_components"] = True
        elif kind == "overflow": spec["mapped_field_components"] = 2**63
        elif kind == "execution": execution.pop("context_version")
        else: raise ValueError("unknown injection")
    return spec, execution


def exact_payload(before, after):
    assert before.keys() == after.keys()
    from pops.runtime._checkpoint_manifest import MANIFEST_KEY, IDENTITY_KEY
    for name in before:
        a, b = before[name], after[name]
        if name in (MANIFEST_KEY, IDENTITY_KEY):
            continue  # Fresh checkpoint lifecycle seals; no physical member excluded.
        if name.startswith("layout_checkpoint_"):
            with np.load(io.BytesIO(a.tobytes()), allow_pickle=False) as left, np.load(
                    io.BytesIO(b.tobytes()), allow_pickle=False) as right:
                exact_payload({k: left[k].copy() for k in left.files},
                              {k: right[k].copy() for k in right.files})
        else:
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name


@pytest.mark.compiler
@pytest.mark.native_loader
def test_rank_local_mapped_dto_refusals_preserve_accepted_storage(
        tmp_path, record_property, monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    import pops.runtime._multi_layout_executor as routes
    native = select_native_dimension(2)
    assert native_mpi_communicator(native) == "MPI_COMM_WORLD", "requires actual MPI2"
    world = native.mpi_world()
    rank = int(world.rank)
    with collective_check(world):
        assert int(world.size) == 2
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        caps = native.module_capabilities("production")
        assert caps["abi_version"] == 9 and caps["mapped_consumed_field_output"] is True
    directory = collective_directory(world, tmp_path / "mapped-binding-refusal")
    collective_call(world, lambda: directory.mkdir(exist_ok=True))
    providers = directory / "providers"
    collective_call(world, lambda: providers.mkdir(exist_ok=True))
    resolved = collective_call(world, lambda: build(providers) if rank == 0 else None)
    if rank != 0:
        from tests.python.integration.runtime.test_m19_product_support_runtime import _load_published_provider
    resolved = collective_call(world, lambda: resolved if rank == 0 else
        build(providers, provider_factory=_load_published_provider))
    artifact = compile_resolved_plan_once(world, resolved, route="mapped binding refusal", compile_artifact=pops.compile)
    collective_call(world, lambda: retain_v_provenance(artifact, native, directory, rank))
    actual_preparations = []
    original = routes._prepare_layout_transfer_route
    def retain_actual_arguments(prepared, execution):
        result = original(prepared, execution)
        if prepared.spec.get("mapped_field_components") == 1:
            actual_preparations.append((prepared, execution))
        return result
    # Observe a genuine installation, without replacing native calls or results.
    monkeypatch.setattr(routes, "_prepare_layout_transfer_route", retain_actual_arguments)
    initial, _, _, _ = reference()
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_state=initial,
        resources={"execution_context": artifact_execution_context(artifact)}))
    with collective_check(world):
        assert len(actual_preparations) == 2
    prepared, execution = actual_preparations[0]
    source = prepared.source_engine._native_step_target()
    target = prepared.target_engine._native_step_target()
    def capture(label):
        clock = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
        values = {}
        for name in NAMES:
            value = collective_call(world, lambda name=name: runtime.state_global(name))
            path = directory / (label + "-rank%d-" % rank + name + ".npy")
            collective_call(world, lambda: np.save(path, value, allow_pickle=False))
            values[name] = np.asarray(value).copy()
        cp = collective_call(world, lambda: runtime.checkpoint(directory / (label + "-checkpoint")))
        collective_call(world, lambda: save_json(directory / (label + "-rank%d.json" % rank),
            {"clock": list(clock), "checkpoint": pin(Path(cp)), "rank": rank, "ranks": 2}))
        with np.load(cp, allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
        return clock, values, payload
    before = capture("before")
    for kind in ("bool", "overflow", "execution"):
        spec, context = mutate_dto(prepared.spec, execution, kind, rank)
        # The real binding must finish its own world vote before this helper gathers.
        result, failures = collective_attempt(world, lambda:
            source._prepare_layout_transfer(target, prepared.component.native_handle, spec, context))
        collective_call(world, lambda: save_json(directory / (kind + "-rank%d-failure.json" % rank),
            {"schema": "sol61.mapped-binding-collective-refusal@1", "failures": failures,
             "rank": rank, "ranks": 2, "root_received": False}))
        after = capture("after-" + kind)
        with collective_check(world):
            assert result is None
            assert len(failures) == 2 and all(f is not None and
                "layout-transfer DTO preparation failed collectively" in f[1] for f in failures)
            assert before[0] == after[0]
            exact_payload(before[1], after[1])
            exact_payload(before[2], after[2])
    record_property("mapped_binding_refusal_directory", str(directory))
