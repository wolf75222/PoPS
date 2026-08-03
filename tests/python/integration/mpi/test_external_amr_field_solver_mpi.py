#!/usr/bin/env python3
"""Two-rank runtime proof for the external FieldTopology@2 + FieldSolver@2 AMR bridge.

The oracle launches one public ``resolve -> compile -> bind -> run`` route with a genuinely
distributed coarse level and fine level.  The same component pair survives a layout-changing
regrid, is rematerialized under exact communicator consensus, rolls back one typed collective
failure, and refuses a rank-local candidate divergence without publishing it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pops
from pops import _pops, interfaces
from pops._native_collectives import allgather_value, barrier, broadcast_value
from pops.amr import PatchLayout
from pops.external import build_source_package_manifest, load
from pops.fields import ExternalFieldSolver
from pops.lib.amr import BergerRigoutsos
from pops.lib.initial import Gaussian

from _compile_once import compile_resolved_plan_once
from tests.python.integration._final_field_program import (
    resolve_periodic_field_program,
    scalar_advection_field_model,
)
from tests.python.integration.native_loader.test_external_field_solver_runtime import (
    _manifest,
    _moving_amr_program,
    _mpi_faulted_solver_source,
    _topology_source,
)


_COMM = _pops.mpi_world()
_fails = 0


def chk(condition: Any, label: str) -> None:
    """Record one all-rank assertion and keep the script's exit status collective."""
    global _fails
    flags = tuple(bool(value) for value in allgather_value(_COMM, bool(condition)))
    passed = all(flags)
    if int(_COMM.rank) == 0:
        print("  [%s] %s" % ("OK " if passed else "XX ", label), flush=True)
    if not passed:
        _fails += 1


def _require_two_rank_world() -> None:
    if int(_COMM.size) != 2:
        raise RuntimeError(
            "external AMR field bridge proof requires exactly mpiexec -n 2; size=%d"
            % int(_COMM.size)
        )


@contextmanager
def _shared_temporary_directory() -> Iterator[Path]:
    root = (
        tempfile.mkdtemp(prefix="pops-external-amr-field-mpi-")
        if int(_COMM.rank) == 0
        else None
    )
    shared = Path(broadcast_value(_COMM, root, root=0))
    try:
        yield shared
    finally:
        barrier(_COMM)
        if int(_COMM.rank) == 0:
            shutil.rmtree(shared, ignore_errors=True)
        barrier(_COMM)


def _publish_component(
    shared: Path,
    *,
    name: str,
    interface: Any,
    source_factory: Callable[[Any], str],
    manifest_parameters: tuple[dict[str, str], ...] = (),
    instance_parameters: dict[str, Any] | None = None,
) -> Any:
    """Publish source once, then load the exact package on every rank."""
    root = shared / name
    alias = name.replace("-", "_")
    manifest = _manifest(name, interface, manifest_parameters)
    source_name = name + ".cpp"
    manifest_path = root / (name + ".pops.json")
    publication: tuple[bool, str] | None = None
    if int(_COMM.rank) == 0:
        try:
            root.mkdir()
            source = source_factory(manifest).encode()
            (root / source_name).write_bytes(source)
            package = build_source_package_manifest(
                components={alias: manifest},
                payloads={source_name: ("source", source)},
            )
            manifest_path.write_text(json.dumps(package), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 -- broadcast before peers enter the loader
            publication = (False, "%s: %s" % (type(exc).__name__, exc))
        else:
            publication = (True, "")
    publication = broadcast_value(_COMM, publication, root=0)
    if not publication[0]:
        raise RuntimeError("rank 0 component publication failed: " + publication[1])

    component = None
    load_error = ""
    try:
        factory = load(manifest_path).require(alias, interface=interface)
        component = factory(
            **({} if instance_parameters is None else instance_parameters)
        )
    except Exception as exc:  # noqa: BLE001 -- aggregate before any later collective
        load_error = "%s: %s" % (type(exc).__name__, exc)
    errors = tuple(allgather_value(_COMM, load_error))
    if any(errors):
        raise RuntimeError(
            "component package load differs across ranks: "
            + "; ".join(
                "rank %d: %s" % (rank, error)
                for rank, error in enumerate(errors)
                if error
            )
        )
    if component is None:
        raise RuntimeError("component package loader returned no instance")
    return component


def _world_digest(value: Any) -> tuple[str, ...]:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return tuple(allgather_value(_COMM, hashlib.sha256(payload).hexdigest()))


def _level_state(runtime: Any, level: int) -> np.ndarray:
    return np.asarray(
        runtime.block_level_state_global("material", level), dtype=np.float64
    ).copy()


def _accepted_snapshot(runtime: Any, slot: str) -> dict[str, Any]:
    return {
        "time": runtime.time(),
        "step": runtime.macro_step(),
        "levels": tuple(_level_state(runtime, level) for level in range(runtime.n_levels())),
        "potential": np.asarray(
            runtime.field_potential_global(slot), dtype=np.float64
        ).copy(),
        "boxes": tuple(runtime.patch_boxes()),
        "owners": tuple(
            tuple(runtime._executor.level_owner_ranks(level))
            for level in range(runtime.n_levels())
        ),
        "providers": runtime.inspect().to_dict()["instance"]["field_providers"],
    }


def _snapshot_is_exact(runtime: Any, slot: str, expected: dict[str, Any]) -> bool:
    actual = _accepted_snapshot(runtime, slot)
    return (
        actual["time"] == expected["time"]
        and actual["step"] == expected["step"]
        and actual["boxes"] == expected["boxes"]
        and actual["owners"] == expected["owners"]
        and actual["providers"] == expected["providers"]
        and np.array_equal(actual["potential"], expected["potential"])
        and len(actual["levels"]) == len(expected["levels"])
        and all(
            np.array_equal(value, reference)
            for value, reference in zip(
                actual["levels"], expected["levels"], strict=True
            )
        )
    )


def _set_marker(path: Path, present: bool) -> None:
    if int(_COMM.rank) == 0:
        if present:
            path.write_text("fault", encoding="utf-8")
        elif path.exists():
            path.unlink()
    barrier(_COMM)


def test_external_amr_field_bridge_executes_and_refuses_collectively() -> None:
    _require_two_rank_world()
    if int(_COMM.rank) == 0:
        print("== external AMR FieldTopology@2 + FieldSolver@2 under two-rank MPI ==")
    with _shared_temporary_directory() as shared:
        collective_fault = shared / "collective-fault"
        divergent_fault = shared / "rank-local-fault"
        topology = _publish_component(
            shared,
            name="mpi-amr-topology",
            interface=interfaces.FieldTopology,
            source_factory=lambda manifest: _topology_source(
                manifest,
                require_multilevel=True,
                require_distributed=True,
                periodic_axes=0,
            ),
        )
        solver = _publish_component(
            shared,
            name="mpi-amr-solver",
            interface=interfaces.FieldSolver,
            source_factory=lambda manifest: _mpi_faulted_solver_source(
                manifest,
                collective_fault_marker=collective_fault,
                divergent_fault_marker=divergent_fault,
            ),
            manifest_parameters=({"name": "answer", "kind": "runtime"},),
            instance_parameters={"answer": 7},
        )
        provider = ExternalFieldSolver(
            topology=topology,
            solver=solver,
            relative_tolerance=1.0e-11,
            absolute_tolerance=0.0,
            max_iterations=23,
        )
        model = scalar_advection_field_model("external-amr-field-mpi")
        x_axis, y_axis = model.frame.axes
        resolved = resolve_periodic_field_program(
            model,
            _moving_amr_program,
            name="external-amr-field-mpi",
            block_name="material",
            target="amr_system",
            n=8,
            regrid_every=2,
            field_solver=provider,
            initial_profile=Gaussian(
                frame=model.frame,
                center={x_axis: 0.25, y_axis: 0.5},
                background=0.8,
                amplitude=4.0,
                inverse_width=80.0,
            ),
            components=(topology, solver),
            anchored_field=True,
            patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=4),
            clustering=BergerRigoutsos(maximum_box_size=4),
        )
        threshold, = (
            runtime_slot.handle
            for runtime_slot in resolved.bind_schema.runtime_slots
            if runtime_slot.handle.local_id
            == "external-amr-field-mpi_refine_threshold"
        )
        artifact = compile_resolved_plan_once(
            _COMM,
            resolved,
            route="external-amr-field-mpi",
            compile_artifact=pops.compile,
        )
        runtime = pops.bind(
            artifact,
            params={threshold: 1.2},
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
        )
        slot, = runtime.field_provider_slots()
        chk(runtime.n_levels() == 2, "bind materializes a two-level AMR hierarchy")
        owners = tuple(
            tuple(runtime._executor.level_owner_ranks(level)) for level in (0, 1)
        )
        local_patch_counts = tuple(
            allgather_value(
                _COMM,
                len(runtime._executor.output_state_local_pieces("material", level)),
            )
            for level in (0, 1)
        )
        chk(
            all(set(level_owners) == {0, 1} for level_owners in owners)
            and all(all(count > 0 for count in counts) for counts in local_patch_counts),
            "both L0 and L1 own real local patches on both MPI ranks",
        )

        boxes_initial = tuple(runtime.patch_boxes())
        first = pops.run(runtime, t_end=8.0e-2, max_steps=1, console=False)
        first_provider = runtime.inspect().to_dict()["instance"]["field_providers"][0]
        first_layout = first_provider["materialized_layout_identity"]
        chk(
            first.accepted_steps == 1
            and first_provider["materialized"]
            and len(set(_world_digest(first_provider))) == 1,
            "the first composite solve publishes one exact provider report on every rank",
        )

        regrids_before = runtime.amr.explain_regrid().regrid_count
        second = pops.run(runtime, t_end=2.4e-1, max_steps=2, console=False)
        second_provider = runtime.inspect().to_dict()["instance"]["field_providers"][0]
        chk(
            second.accepted_steps == 2
            and runtime.amr.explain_regrid().regrid_count > regrids_before
            and tuple(runtime.patch_boxes()) != boxes_initial
            and second_provider["materialized_layout_identity"] != first_layout
            and len(set(_world_digest(second_provider))) == 1,
            "a layout-changing regrid rematerializes the exact component pair collectively",
        )

        _set_marker(collective_fault, True)
        before_collective_failure = _accepted_snapshot(runtime, slot)
        collective_error = None
        try:
            pops.run(runtime, t_end=3.2e-1, max_steps=1, console=False)
        except RuntimeError as exc:
            collective_error = str(exc)
        collective_errors = tuple(allgather_value(_COMM, collective_error))
        chk(
            len(set(collective_errors)) == 1
            and collective_errors[0] is not None
            and "invalid_evaluation action=fail_run" in collective_errors[0],
            "one typed FieldSolver failure reaches every rank with the same FailRun outcome",
        )
        chk(
            _snapshot_is_exact(runtime, slot, before_collective_failure),
            "collective FailRun restores levels, potential, clock, topology and provider evidence",
        )
        _set_marker(collective_fault, False)
        retry = pops.run(runtime, t_end=3.2e-1, max_steps=1, console=False)
        chk(
            retry.accepted_steps == 1 and runtime.macro_step() == 4,
            "the exact accepted state remains retryable after collective rollback",
        )

        _set_marker(divergent_fault, True)
        before_divergence = _accepted_snapshot(runtime, slot)
        divergent_error = None
        try:
            pops.run(runtime, t_end=4.0e-1, max_steps=1, console=False)
        except RuntimeError as exc:
            divergent_error = str(exc)
        divergent_errors = tuple(allgather_value(_COMM, divergent_error))
        chk(
            len(set(divergent_errors)) == 1
            and divergent_errors[0] is not None
            and "provider report differs between MPI ranks" in divergent_errors[0],
            "a rank-local non-finite candidate is refused by exact report consensus",
        )
        chk(
            _snapshot_is_exact(runtime, slot, before_divergence),
            "rank-divergent refusal publishes no field, state, clock or topology mutation",
        )


def _run_all() -> int:
    functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for function in functions:
        function()
    if int(_COMM.rank) == 0:
        print(
            "\n%s test_external_amr_field_solver_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails),
            flush=True,
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
