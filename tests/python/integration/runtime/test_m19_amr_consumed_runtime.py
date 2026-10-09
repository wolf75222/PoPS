"""Installed SDK witness requiring the current published ABI; Source collection is not Native reception."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pops
from pops._generated_release_contract import NATIVE_ABI_VERSION
import pytest
from tests.python.support.m19_amr_consumed_case import build, NAMES, DT
from tests.python.support.collective_checks import (
    collective_attempt,
    collective_call,
    collective_check,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.review.sol61_m19_amr_consumed_offline import (
    verify_step,
    checkpoint_children,
    checkpoint_states,
    receive_phase,
    assert_rollback,
)


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n")


def pin(path):
    return {"file": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("reverse", (False, True))
def test_installed_amr_consumed_field_regrid_and_nonfinite_rollback(
    tmp_path, record_property, isolated_native_cache, native_cxx, kokkos_root, reverse
):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops._native_collectives import allgather_value

    native = select_native_dimension(2)
    world = native.mpi_world() if native_mpi_communicator(native) == "MPI_COMM_WORLD" else None
    rank = 0 if world is None else int(world.rank)
    ranks = 1 if world is None else int(world.size)
    directory = collective_directory(world, tmp_path / "amr-consumed-field")
    collective_call(world, lambda: directory.mkdir(exist_ok=True))
    with collective_check(world):
        assert (
            Path(pops.__file__).resolve().is_relative_to(Path(__import__("sys").prefix).resolve())
        )
        caps = native.module_capabilities("production")
        assert type(caps["abi_version"]) is int
        assert caps["abi_version"] == NATIVE_ABI_VERSION
        assert caps["mapped_consumed_field_output_amr"] is True
    # One elected writer publishes the authenticated test component package; peers only load it.
    resolved = collective_call(
        world, lambda: build(directory / "providers", reverse=reverse) if rank == 0 else None
    )
    if world is not None:
        resolved = collective_call(
            world,
            lambda: resolved if rank == 0 else build(directory / "providers", reverse=reverse),
        )
    artifact = (
        pops.compile(resolved)
        if world is None
        else compile_resolved_plan_once(
            world, resolved, route="AMR consumed Field", compile_artifact=pops.compile
        )
    )
    collective_call(world, lambda: retain_v_provenance(artifact, native, directory, rank))
    collective_call(
        world,
        lambda: save(
            directory / ("prepared-rank%d.json" % rank),
            {
                "schema": "sol61.amr-consumed-field-witness@1",
                "rank": rank,
                "ranks": ranks,
                "reverse": reverse,
                "dt": DT,
                "blocks": list(NAMES),
                "artifact": artifact.artifact_identity.token,
                "native": pin(native.__file__),
                "C25_before_bind": True,
                "Native_received": False,
            },
        ),
    )
    runtime = collective_call(
        world, lambda: pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    )

    def capture(label):
        files = {}
        clock = [runtime.time(), runtime.macro_step()]
        cursors = runtime.consumer_cursors.to_data()
        checkpoint = collective_call(
            world, lambda: runtime.checkpoint(directory / (label + "-checkpoint"))
        )
        carried, authority = collective_call(world, lambda: checkpoint_states(checkpoint))

        def metadata(complete, checkpoint=None):
            save(
                directory / ("%s-rank%d.json" % (label, rank)),
                {
                    "schema": "sol61.amr-consumed-phase@1",
                    "rank": rank,
                    "ranks": ranks,
                    "phase": label,
                    "clock": clock,
                    "consumer_cursors": cursors,
                    "files": files,
                    "checkpoint": checkpoint,
                    "authority": authority,
                    "capture_complete": complete,
                },
            )

        collective_call(world, lambda: metadata(False, pin(checkpoint)))
        for name in NAMES:
            value = collective_call(
                world,
                lambda name=name: np.asarray(
                    runtime.block_level_state_global(name, authority[name]["level"]),
                    dtype=np.float64,
                ).reshape(carried[name].shape),
            )
            path = directory / ("%s-rank%d-%s.npy" % (label, rank, name))
            collective_call(
                world, lambda path=path, value=value: np.save(path, value, allow_pickle=False)
            )
            files[name] = pin(path)
            collective_call(world, lambda: metadata(False, pin(checkpoint)))
        values = {name: np.load(row["file"], allow_pickle=False) for name, row in files.items()}
        collective_call(world, lambda: receive_phase(checkpoint, values, clock))
        collective_call(world, lambda: metadata(True, pin(checkpoint)))
        return values, Path(checkpoint)

    trace = directory / ("integral-rank%d.log" % rank)
    old = {
        key: os.environ.get(key) for key in ("POPS_AMR_FIELD_MAP_TRACE", "POPS_AMR_FIELD_MAP_FAULT")
    }
    try:
        os.environ["POPS_AMR_FIELD_MAP_TRACE"] = str(trace)
        os.environ.pop("POPS_AMR_FIELD_MAP_FAULT", None)
        previous, initial_cp = capture("initial")
        for step in (1, 2):
            collective_call(
                world,
                lambda step=step: pops.run(runtime, t_end=step * DT, max_steps=1, console=False),
            )
            current, checkpoint = capture("accepted%d" % step)
            proof = collective_call(
                world, lambda previous=previous, current=current: verify_step(previous, current)
            )
            collective_call(
                world,
                lambda step=step, proof=proof: save(
                    directory / ("step%d-oracle-rank%d.json" % (step, rank)), proof
                ),
            )
            with collective_check(world):
                assert proof["phi_span"] > 1e-3
                assert proof["mapped_span"] > 1e-3 and proof["signed_map_max"] > 1e-4
                assert runtime.time() == step * DT and runtime.macro_step() == step
            previous = current
        with collective_check(world):
            start, end = checkpoint_children(initial_cp), checkpoint_children(checkpoint)
            assert set(start) == set(end) and all(int(row["n_levels"]) == 2 for row in end.values())
            assert any(
                int(end[key]["topology_epoch"]) > int(start[key]["topology_epoch"]) for key in start
            )
        observed = collective_call(
            world,
            lambda: (
                tuple(int(line) for line in trace.read_text().splitlines() if line.isdigit())
                if trace.exists()
                else ()
            ),
        )
        gathered = (observed,) if world is None else tuple(allgather_value(world, observed))
        with collective_check(world):
            actual = {item for rows in gathered for item in rows}
            assert actual and all(type(item) is int and 0 <= item < ranks for item in actual)
            target = max(actual)
        agreed = (target,) if world is None else tuple(allgather_value(world, target))
        with collective_check(world):
            assert all(item == target for item in agreed)
        collective_call(
            world,
            lambda: save(
                directory / ("target-rank%d.json" % rank),
                {"target": target, "observed": gathered, "agreement": agreed},
            ),
        )
        before, before_cp = capture("before-failure")
        os.environ["POPS_AMR_FIELD_MAP_FAULT"] = str(target)
        _, failures = collective_attempt(
            world, lambda: pops.run(runtime, t_end=3 * DT, max_steps=1, console=False)
        )
        collective_call(
            world,
            lambda: save(
                directory / ("failure-rank%d.json" % rank), {"failures": failures, "target": target}
            ),
        )
        os.environ.pop("POPS_AMR_FIELD_MAP_FAULT", None)
        after, after_cp = capture("after-failure")
        with collective_check(world):
            assert len(failures) == ranks and all(
                row is not None
                and row[2]
                and "nonfinite" in row[1].lower().replace("-", "").replace(" ", "")
                for row in failures
            )
            assert any(
                ("injected:" + str(target)) in path.read_text()
                for path in directory.glob("integral-rank*.log")
            )
            for name in NAMES:
                assert (
                    before[name].dtype == after[name].dtype
                    and before[name].shape == after[name].shape
                    and before[name].tobytes() == after[name].tobytes()
                )
            assert runtime.time() == 2 * DT and runtime.macro_step() == 2
            proof = assert_rollback(before_cp, after_cp)
        collective_call(world, lambda: save(directory / ("rollback-rank%d.json" % rank), proof))
        record_property(
            "AMR_consumed_field_receipt", str(directory / ("prepared-rank%d.json" % rank))
        )
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
