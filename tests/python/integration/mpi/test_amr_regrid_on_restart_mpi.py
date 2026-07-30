#!/usr/bin/env python3
"""Two-rank fail-closed proof for the operational RegridOnRestart transaction.

Rank one first injects a local accepted-state validation failure immediately before the restart
route's first composite reduction. Every rank must receive one coherent failure and roll back
without entering a mismatched collective. The same fresh runtime then retries successfully,
rebinds the AB2 history/lagged-flux topology, publishes one common transformed-hierarchy receipt,
and carries that continuation identity into the next run manifest.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from _compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import (
        allgather_value,
        barrier,
        broadcast_bytes,
        broadcast_value,
    )
    from tests.python.integration.amr.test_amr_regrid_on_restart import (
        DT,
        NSTEPS,
        _resolved,
    )
    from tests.python.integration.runtime.test_shared_interface_runtime import (
        _flux_component,
        _load_example,
    )
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("RegridOnRestart MPI runtime import failed: %s" % exc)


_COMM = _pops.mpi_world()
_fails = 0
_SHARED_DT = 1.0e-3
_SHARED_SOURCE_STEPS = 2
_SHARED_VELOCITY_X = 20.0


def chk(condition: Any, label: str) -> None:
    global _fails
    if int(_COMM.rank) == 0:
        print("  [%s] %s" % ("OK " if condition else "XX ", label), flush=True)
    if not condition:
        _fails += 1


def _require_world() -> None:
    if int(_COMM.size) < 2:
        require_mpi_or_skip(
            "RegridOnRestart collective proof needs mpiexec -n 2; size=%d" % int(_COMM.size)
        )


@contextmanager
def _shared_temporary_directory() -> Iterator[Path]:
    root = tempfile.mkdtemp(prefix="pops-regrid-restart-mpi-") if int(_COMM.rank) == 0 else None
    shared = Path(broadcast_value(_COMM, root, root=0))
    barrier(_COMM)
    try:
        yield shared
    finally:
        barrier(_COMM)
        if int(_COMM.rank) == 0:
            shutil.rmtree(shared, ignore_errors=True)
        barrier(_COMM)


def _bind(artifact):
    context = pops.ExecutionContext.mpi_world(artifact)
    return pops.bind(artifact, resources={"execution_context": context})


def _shared_flux_component(root: Path):
    """Compile one exact external component on rank zero and reconstruct it on every peer."""
    component = None
    failure = ""
    if int(_COMM.rank) == 0:
        try:
            component_root = root / "component-authority"
            component_root.mkdir(parents=True, exist_ok=True)
            component = _flux_component(component_root)
        except Exception as exc:  # noqa: BLE001 -- publish one coherent compile failure
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = broadcast_value(_COMM, failure, root=0)
    if failure:
        raise RuntimeError("shared NumericalFlux component compilation failed: " + failure)
    component_metadata = (
        {
            "component_id": component.component_id,
            "component_manifest": component.component_manifest.token,
            "runtime_manifest": dict(component.runtime_contract.manifest_data),
            "platform_manifest": component.platform_manifest.to_data(),
            "entry_symbols": dict(component.entry_symbols),
            "binary_identity": component.binary_identity.token,
            "artifact_identity": component.artifact_identity.token,
            "source_package": (
                component.source_package.token if component.source_package is not None else None
            ),
            "fixed_signature": component.fixed_signature,
            "suffix": component.suffix,
        }
        if component is not None
        else None
    )
    metadata = broadcast_value(
        _COMM,
        component_metadata,
        root=0,
    )
    binary = broadcast_bytes(
        _COMM,
        component.binary if component is not None else b"",
        root=0,
    )
    local_failure = ""
    try:
        if not isinstance(metadata, dict) or not binary:
            raise RuntimeError("rank zero published an incomplete NumericalFlux component artifact")
        from pops.identity import Identity

        if component is None:
            from pops.external.artifacts import CompiledComponentArtifact, ComponentRuntimeContract
            from pops.interfaces import NumericalFlux
            from pops.model import ComponentManifest
            from pops.runtime._platform_manifest import PlatformManifest

            manifest = ComponentManifest.from_data(metadata["runtime_manifest"])
            component = CompiledComponentArtifact(
                component_id=metadata["component_id"],
                component_manifest=Identity.from_token(metadata["component_manifest"]),
                runtime_contract=ComponentRuntimeContract.from_manifest(manifest),
                interface=NumericalFlux,
                platform_manifest=PlatformManifest.from_data(metadata["platform_manifest"]),
                entry_symbols=dict(metadata["entry_symbols"]),
                binary_identity=Identity.from_token(metadata["binary_identity"]),
                binary=binary,
                source_package=(
                    Identity.from_token(metadata["source_package"])
                    if metadata["source_package"] is not None
                    else None
                ),
                fixed_signature=metadata["fixed_signature"],
                suffix=metadata["suffix"],
            )
        component.verify()
        if component.artifact_identity != Identity.from_token(metadata["artifact_identity"]):
            raise RuntimeError(
                "broadcast NumericalFlux artifact identity changed during reconstruction"
            )
    except Exception as exc:  # noqa: BLE001 -- close local authentication before the next collective
        local_failure = "%s: %s" % (type(exc).__name__, exc)

    failures = tuple(str(value) for value in allgather_value(_COMM, local_failure))
    if any(failures):
        details = "; ".join(
            "rank %d: %s" % (rank, message)
            for rank, message in enumerate(failures)
            if message
        )
        raise RuntimeError("shared NumericalFlux component authentication failed: " + details)
    if component is None:
        raise RuntimeError("collective component authentication returned without an artifact")
    return component


def _accepted_image(runtime, *, blocks=("tracer",)):
    native = runtime._executor._s
    levels = int(runtime.n_levels())
    histories = tuple(
        (
            str(name),
            tuple(
                np.asarray(runtime.history_global(name, slot), dtype=np.float64).copy()
                for slot in range(int(runtime.history_depth(name)))
            ),
        )
        for name in runtime.history_names()
    )
    return {
        "time": float(runtime.time()),
        "step": int(runtime.macro_step()),
        "boxes": tuple(tuple(int(value) for value in box) for box in runtime.patch_boxes()),
        "owners": tuple(
            tuple(int(rank) for rank in native.level_owner_ranks(level)) for level in range(levels)
        ),
        "states": tuple(
            np.asarray(
                runtime.block_level_state_global(block, level),
                dtype=np.float64,
            ).copy()
            for block in blocks
            for level in range(levels)
        ),
        "histories": histories,
        "program_state": bytes(native.program_accepted_state()),
        "flux_ledger": tuple(tuple(map(str, row)) for row in native.program_flux_ledger_manifest()),
        "interface_flux_ledger": tuple(
            tuple(map(str, row)) for row in native.program_interface_flux_ledger_manifest()
        ),
        "synchronization": tuple(tuple(map(str, row)) for row in native.program_sync_manifest()),
        "transfer_routes": tuple(
            tuple(map(str, row)) for row in native.checkpoint_transfer_routes()
        ),
        "regrid_count": int(native.checkpoint_regrid_count()),
        "topology_epoch": int(native.checkpoint_topology_epoch()),
        "run_identity": runtime.last_run_identity,
        "consumer_cursors": runtime.consumer_cursors.to_data(),
    }


def _same_image(left, right) -> bool:
    arrays = {"states", "histories"}
    histories_equal = [name for name, _slots in left["histories"]] == [
        name for name, _slots in right["histories"]
    ] and all(
        len(current_slots) == len(recorded_slots)
        and all(
            np.array_equal(current, recorded)
            for current, recorded in zip(current_slots, recorded_slots, strict=True)
        )
        for (_name, current_slots), (_expected_name, recorded_slots) in zip(
            left["histories"], right["histories"], strict=True
        )
    )
    return (
        {key: value for key, value in left.items() if key not in arrays}
        == {key: value for key, value in right.items() if key not in arrays}
        and len(left["states"]) == len(right["states"])
        and all(
            np.array_equal(current, recorded)
            for current, recorded in zip(left["states"], right["states"], strict=True)
        )
        and histories_equal
    )


def _shared_ab2_program(left_state, right_state, rate):
    from pops.time import Dense, FixedDt, StagePoint, TimePoint

    program = pops.Program("shared_interface_ab2")
    left = program.state(left_state)
    right = program.state(right_state)
    stage = StagePoint("shared_ab2_stage", {"main": TimePoint(program.clock, 0)})
    left_rate = program.value("left_rate", rate(left.n), at=stage)
    right_rate = program.value("right_rate", rate(right.n), at=stage)

    histories = []
    for temporal, current, name in (
        (left, left_rate, "tracer.rate"),
        (right, right_rate, "right.rate"),
    ):
        program.store_history(name, current, depth=1, checkpoint_policy=Dense())
        histories.append(
            program.history(
                name,
                lag=1,
                space=current.space,
                block=temporal.block,
                state_ref=temporal.state,
            )
        )

    left_next = program.value(
        "left_next",
        left.n + program.dt * (Fraction(3, 2) * left_rate - Fraction(1, 2) * histories[0]),
        at=left.next.point,
    )
    right_next = program.value(
        "right_next",
        right.n + program.dt * (Fraction(3, 2) * right_rate - Fraction(1, 2) * histories[1]),
        at=right.next.point,
    )
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)
    program.step_strategy(FixedDt(_SHARED_DT))
    return program


def _shared_interface_resolved(component):
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        PatchLayout,
        Tag,
    )
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import BergerRigoutsos, StateTransfer
    from pops.lib.initial import BindArray
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid
    from pops.mesh.boundaries import BlockInterfaceSide, ConservativeInterface
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.output import Checkpoint, ConsumerGraph, RegridOnRestart
    from pops.projection import ConservativeCellAverage
    from pops.time import every

    example = _load_example()
    core = example.build_authoring(output_root=Path("unused/shared-interface-mpi"))
    right = core.case.block("right", model=core.model)
    right_state = right[core.state]
    finite_volume = FiniteVolume(
        flux=core.flux,
        variables=variables.Conservative(core.state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.ScalarUpwind(velocity=core.velocity),
    )
    boundaries = core.frame.boundaries

    def numerics(state):
        plan = DiscretizationPlan()
        plan.rates.add(core.rate, finite_volume)
        plan.boundaries.add(
            TransportBoundarySet(
                {
                    boundaries.x_min: Inflow(state=state, value=core.inlet_x_value),
                    boundaries.x_max: Outflow(state=state),
                    boundaries.y_min: Inflow(state=state, value=core.inlet_y_value),
                    boundaries.y_max: Outflow(state=state),
                }
            )
        )
        return plan

    left_numerics = numerics(core.tracer_state)
    right_numerics = numerics(right_state)
    ConservativeInterface(
        "tracer_to_right",
        left=BlockInterfaceSide(core.tracer_state, boundaries.x_max),
        right=BlockInterfaceSide(right_state, boundaries.x_min),
        numerical_flux=component,
        permutation=(0,),
        right_normal_translation=1.0,
    ).attach(left_numerics, right_numerics)
    core.case.numerics(left_numerics, block=core.tracer)
    core.case.numerics(right_numerics, block=right)
    core.case.initials.add(
        InitialCondition(
            state=core.tracer_state,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    core.case.initials.add(
        InitialCondition(
            state=right_state,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    program = _shared_ab2_program(core.tracer_state, right_state, core.rate)
    core.case.program(program)
    core.case.consumers(
        ConsumerGraph.from_consumers(
            (
                Checkpoint(
                    schedule=every(10_000, clock=program.clock),
                    target="unused/shared-interface-restart-mpi",
                    hierarchy=RegridOnRestart(),
                ),
            )
        )
    )

    transfer = AMRTransfer()
    transfer.state(core.tracer_state, StateTransfer())
    transfer.state(right_state, StateTransfer())
    tagging = AMRTagging(
        rules=(
            Tag(ValueExpr(core.tracer_state) > core.case.value(core.refine_threshold)),
            Tag(ValueExpr(right_state) > core.case.value(core.refine_threshold)),
            Buffer(cells=1),
        ),
        hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )
    resolved = pops.resolve(
        pops.validate(core.case),
        layout=AMR(
            grid=CartesianGrid(frame=core.frame, cells=(8, 8)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=tagging,
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
            patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=4),
            clustering=BergerRigoutsos(maximum_box_size=4),
        ),
        components=(component,),
        compile_options={"include": str(Path(__file__).resolve().parents[4] / "include")},
    )

    left_initial = np.zeros((1, 8, 8), dtype=np.float64)
    right_initial = np.zeros((1, 8, 8), dtype=np.float64)
    left_initial[0, :, -1:] = 1.0
    right_initial[0, :, :1] = 3.0
    params = {
        core.case.resolve(handle, block=block): value
        for block in (core.tracer, right)
        for handle, value in (
            # dx_fine=1/16 and the 2:1 subcycle gives CFL_fine=0.16. Across the two
            # macro-steps this moves the profile by 0.64 fine cell: enough to change the
            # thresholded hierarchy without approaching the first-order stability limit.
            (core.velocity_x_param, _SHARED_VELOCITY_X),
            (core.velocity_y_param, 1.0e-12),
            (core.inlet_x_param, 0.0),
            (core.inlet_y_param, 0.0),
        )
    }
    params.update(
        {
            core.case.resolve(core.refine_threshold): 0.10,
            core.case.resolve(core.coarsen_threshold): 0.04,
        }
    )
    interface = resolved.blocks[0].numerics.boundaries[0].interfaces[0]
    return (
        resolved,
        {
            "initial_values": {
                core.tracer_state: left_initial,
                right_state: right_initial,
            },
            "params": params,
        },
        interface.qualified_id,
    )


def test_regrid_on_restart_mpi_collective_rollback_and_lineage() -> None:
    _require_world()
    if int(_COMM.rank) == 0:
        print("== RegridOnRestart two-rank collective rollback + lineage ==", flush=True)

    resolved = _resolved()
    artifact = compile_resolved_plan_once(
        _COMM,
        resolved,
        route="regrid-on-restart-mpi",
        compile_artifact=pops.compile,
    )
    source = _bind(artifact)
    report = pops.run(
        source,
        t_end=NSTEPS * DT,
        max_steps=NSTEPS,
        console=False,
    )
    chk(report.accepted_steps == NSTEPS, "source reaches the exact accepted checkpoint boundary")
    chk(bool(tuple(source.history_names())), "source checkpoint carries a real multistep history")

    with _shared_temporary_directory() as root:
        checkpoint = source.checkpoint(root / "moving-profile")
        restarted = _bind(artifact)
        rollback_image = _accepted_image(restarted)

        from pops.runtime import _amr_checkpoint_contract as contract

        original_validation = contract.validate_restored_contract

        def fail_on_rank_one(sim, payload):
            if int(_COMM.rank) == 1:
                raise RuntimeError("injected rank-local pre-collective validation failure")
            return original_validation(sim, payload)

        contract.validate_restored_contract = fail_on_rank_one
        caught = False
        try:
            restarted.restart(checkpoint)
        except RuntimeError:
            caught = True
        finally:
            contract.validate_restored_contract = original_validation

        chk(
            all(allgather_value(_COMM, caught)),
            "one rank-local validation fault fails every rank coherently",
        )
        chk(
            _same_image(_accepted_image(restarted), rollback_image),
            "failed collective restart restores each rank's complete accepted image",
        )

        # Let the complete native transform finish, then make one rank report a different accepted
        # contract identity.  The post-transform consensus must reject it and the outer restart
        # transaction must restore owners, histories, ledgers, Program bytes and topology exactly.
        from pops.runtime import _amr_checkpoint_v3 as checkpoint_codec

        original_contract_identity = checkpoint_codec._restart_accepted_contract_identity
        identity_calls = 0

        def diverge_after_native_transform(sim):
            nonlocal identity_calls
            identity_calls += 1
            identity = original_contract_identity(sim)
            if identity_calls == 2 and int(_COMM.rank) == 1:
                from pops.identity import make_identity

                return make_identity(
                    "restart-accepted-contract",
                    {"injected_rank": 1, "native_identity": identity},
                ).token
            return identity

        checkpoint_codec._restart_accepted_contract_identity = diverge_after_native_transform
        divergent_identity_caught = False
        try:
            restarted.restart(checkpoint)
        except RuntimeError:
            divergent_identity_caught = True
        finally:
            checkpoint_codec._restart_accepted_contract_identity = original_contract_identity

        chk(
            all(allgather_value(_COMM, divergent_identity_caught)),
            "rank-divergent post-transform contract identity fails every rank coherently",
        )
        chk(
            all(value == 2 for value in allgather_value(_COMM, identity_calls)),
            "identity divergence is injected only after the native transform",
        )
        chk(
            _same_image(_accepted_image(restarted), rollback_image),
            "post-transform identity divergence rolls back owners, histories and Program audit",
        )

        restart_identity = restarted.restart(checkpoint)
        continuation_identity = restarted.last_run_identity
        receipt = restarted._executor.last_restart_regrid_receipt()
        chk(
            restart_identity == restarted.last_restart_identity,
            "successful retry publishes the authenticated restart identity",
        )
        chk(
            receipt["changed"] is True
            and receipt["before"]["topology_identity"] != receipt["after"]["topology_identity"],
            "successful retry performs one real structural hierarchy transform",
        )
        from pops.identity import Identity

        chk(
            receipt["schema_version"] == 2
            and Identity.from_token(receipt["accepted_contract_identity_before"]).domain
            == "restart-accepted-contract"
            and Identity.from_token(receipt["accepted_contract_identity_after"]).domain
            == "restart-accepted-contract"
            and Identity.from_token(receipt["history_consensus_identity_before"]).domain
            == "restart-history-image"
            and Identity.from_token(receipt["history_consensus_identity_after"]).domain
            == "restart-history-image",
            "receipt authenticates accepted contracts and each phase-local history consensus",
        )
        chk(
            np.allclose(
                [row["value"] for row in receipt["composite_integrals_after"]],
                [row["value"] for row in receipt["composite_integrals_before"]],
                rtol=2.0e-12,
                atol=2.0e-13,
            ),
            "the collective hierarchy transform conserves every block component",
        )
        chk(
            len(
                set(
                    allgather_value(
                        _COMM,
                        (
                            restart_identity.token,
                            continuation_identity.token,
                            json.dumps(
                                receipt,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                        ),
                    )
                )
            )
            == 1,
            "restart identity, continuation identity and receipt agree on every rank",
        )

        continued = pops.run(
            restarted,
            t_end=float(restarted.time()) + DT,
            max_steps=1,
            console=False,
        )
        manifest = restarted._executor.last_run_manifest
        chk(
            manifest.continuation_identity == continuation_identity,
            "the next run manifest authenticates the transformed continuation lineage",
        )
        chk(
            continued.accepted_steps == 1
            and continued.run_identity != source.last_run_identity
            and all(
                np.all(np.isfinite(np.asarray(restarted.history_global(name, slot))))
                for name in restarted.history_names()
                for slot in range(int(restarted.history_depth(name)))
            ),
            "the transformed multistep continuation advances without colliding with its source",
        )


def test_regrid_on_restart_mpi_shared_interface_transaction() -> None:
    _require_world()
    if int(_COMM.rank) == 0:
        print("== RegridOnRestart two-rank shared-interface transaction ==", flush=True)

    with _shared_temporary_directory() as root:
        component = _shared_flux_component(root)
        component_identities = allgather_value(_COMM, component.artifact_identity.token)
        chk(
            len(set(component_identities)) == 1,
            "every rank receives the same native NumericalFlux artifact",
        )
        resolved, bind_inputs, interface_identity = _shared_interface_resolved(component)
        artifact = compile_resolved_plan_once(
            _COMM,
            resolved,
            route="regrid-on-restart-mpi-shared-interface",
            compile_artifact=pops.compile,
        )
        source = pops.bind(
            artifact,
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
            **bind_inputs,
        )
        report = pops.run(
            source,
            t_end=_SHARED_SOURCE_STEPS * _SHARED_DT,
            max_steps=_SHARED_SOURCE_STEPS,
            console=False,
        )
        source_image = _accepted_image(source, blocks=("tracer", "right"))
        chk(
            report.accepted_steps == _SHARED_SOURCE_STEPS,
            "shared-interface source fills its AB2 history",
        )
        chk(
            len(source.history_names()) == 2
            and all(source.history_depth(name) >= 1 for name in source.history_names()),
            "both shared-interface endpoints carry persistent AB2 histories",
        )
        chk(
            bool(source_image["interface_flux_ledger"]),
            "the accepted checkpoint image carries a non-empty shared-interface flux audit",
        )
        chk(
            len(
                set(
                    allgather_value(
                        _COMM,
                        source_image["interface_flux_ledger"],
                    )
                )
            )
            == 1,
            "the accepted shared-interface audit agrees exactly on every rank",
        )
        checkpoint_integral = source.integral("tracer") + source.integral("right")
        checkpoint = source.checkpoint(root / "shared-interface-moving-profile")

        restarted = pops.bind(
            artifact,
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
            **bind_inputs,
        )
        rollback_image = _accepted_image(restarted, blocks=("tracer", "right"))
        from pops.runtime import _amr_checkpoint_v3 as checkpoint_codec

        original_contract_identity = checkpoint_codec._restart_accepted_contract_identity
        identity_calls = 0

        def diverge_after_native_transform(sim):
            nonlocal identity_calls
            identity_calls += 1
            identity = original_contract_identity(sim)
            if identity_calls == 2 and int(_COMM.rank) == 1:
                from pops.identity import make_identity

                return make_identity(
                    "restart-accepted-contract",
                    {
                        "injected_rank": 1,
                        "native_identity": identity,
                        "shared_interface": interface_identity,
                    },
                ).token
            return identity

        checkpoint_codec._restart_accepted_contract_identity = diverge_after_native_transform
        caught = False
        try:
            restarted.restart(checkpoint)
        except RuntimeError:
            caught = True
        finally:
            checkpoint_codec._restart_accepted_contract_identity = original_contract_identity

        chk(
            all(allgather_value(_COMM, caught)),
            "post-transform shared-interface identity fault fails every rank coherently",
        )
        chk(
            all(value == 2 for value in allgather_value(_COMM, identity_calls)),
            "shared-interface fault is injected only after the native hierarchy transform",
        )
        chk(
            _same_image(
                _accepted_image(restarted, blocks=("tracer", "right")),
                rollback_image,
            ),
            "shared-interface fault rolls back owners, histories, ledgers and Program bytes",
        )

        restarted.restart(checkpoint)
        transformed_image = _accepted_image(restarted, blocks=("tracer", "right"))
        receipt = restarted._executor.last_restart_regrid_receipt()
        chk(
            receipt["changed"] is True and transformed_image["boxes"] != source_image["boxes"],
            "retry commits one real distributed hierarchy transform",
        )
        chk(
            len(
                set(
                    allgather_value(
                        _COMM,
                        json.dumps(
                            receipt,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    )
                )
            )
            == 1,
            "the shared-interface restart receipt agrees exactly on every rank",
        )
        chk(
            not transformed_image["interface_flux_ledger"],
            "retry clears the stale accepted interface audit on the transformed topology",
        )
        chk(
            [name for name, _slots in transformed_image["histories"]]
            == [name for name, _slots in source_image["histories"]]
            and all(
                np.all(np.isfinite(slot))
                for _name, slots in transformed_image["histories"]
                for slot in slots
            ),
            "both AB2 histories are conservatively rematerialized on the new hierarchy",
        )
        counts_before = tuple(
            restarted._executor._s._interface_evaluation_count(interface_identity, level)
            for level in range(int(restarted.n_levels()))
        )
        continued = pops.run(
            restarted,
            t_end=float(restarted.time()) + _SHARED_DT,
            max_steps=1,
            console=False,
        )
        counts_after = tuple(
            restarted._executor._s._interface_evaluation_count(interface_identity, level)
            for level in range(int(restarted.n_levels()))
        )
        continued_image = _accepted_image(restarted, blocks=("tracer", "right"))
        chk(
            continued.accepted_steps == 1
            and all(
                after > before for before, after in zip(counts_before, counts_after, strict=True)
            ),
            "retry executes the rematerialized NumericalFlux on every active level",
        )
        chk(
            bool(continued_image["interface_flux_ledger"])
            and continued_image["interface_flux_ledger"] != source_image["interface_flux_ledger"],
            "continuation republishes an interface audit qualified by the new topology",
        )
        chk(
            len(
                set(
                    allgather_value(
                        _COMM,
                        continued_image["interface_flux_ledger"],
                    )
                )
            )
            == 1,
            "the rematerialized shared-interface audit agrees exactly on every rank",
        )
        chk(
            np.isclose(
                restarted.integral("tracer") + restarted.integral("right"),
                checkpoint_integral,
                rtol=0.0,
                atol=2.0e-13,
            ),
            "post-retry shared-interface continuation remains globally conservative",
        )


def _run_all() -> int:
    test_regrid_on_restart_mpi_collective_rollback_and_lineage()
    test_regrid_on_restart_mpi_shared_interface_transaction()
    if int(_COMM.rank) == 0:
        print(
            "\n%s test_amr_regrid_on_restart_mpi (%d check failures)"
            % ("FAIL" if _fails else "PASS", _fails),
            flush=True,
        )
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
