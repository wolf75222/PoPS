"""Public mathematical late refusal with instrumented Native-journal snapshots.

Authoring, bind and run use public APIs. Exact Aux/history rollback reads use
existing private Native instrumentation; they do not qualify a public snapshot API.
"""

import base64
import hashlib
import json
import os
import subprocess
import sysconfig
from pathlib import Path
import numpy as np
import pops
import pytest
from pops.fields import CompositeHierarchySolve
from pops.solvers.elliptic import GeometricMG
from pops.solvers.tolerances import Relative, AbsoluteFloor
from tests.python.support.public_diffusion_field_late_refusal_case import (
    author_case,
    one_level_amr_layout,
    DT,
)
from tests.python.support.diffusion_field_oracle import check_saved
from tests.python.support.diffusion_field_late_refusal_oracle import (
    check_adaptive_saved,
)
from tests.python.support.collective_checks import (
    collective_call,
    collective_check,
    collective_attempt,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context():
    from pops._native_selector import select_native_dimension
    from pops.codegen.abi import module_header_signature
    from scripts.check_packaging_manifest import read_manifest, PYTHON_SOURCE_SUFFIXES
    from scripts.verify_installed_native import verify_installed_native

    native = select_native_dimension(2)
    root = Path(__file__).resolve().parents[4]
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    package = Path(pops.__file__).resolve().parent
    assert any(
        package.is_relative_to(Path(sysconfig.get_path(k)).resolve())
        for k in ("purelib", "platlib")
    )
    assert not package.is_relative_to(root)
    names = subprocess.check_output(
        ["git", "ls-files", "--", "python/pops"], cwd=root, text=True
    ).splitlines()
    names = [n for n in names if Path(n).suffix in PYTHON_SOURCE_SUFFIXES]
    names += ["include/" + str(p) for p in read_manifest(root).installed_headers] + [
        "include/pops_headers.manifest"
    ]
    for name in names:
        installed = package / (
            name.removeprefix("python/pops/")
            if name.startswith("python/pops/")
            else name
        )
        assert sha(installed) == sha(root / name), name
    native_sha = sha(native.__file__)
    if os.environ.get("POPS_DIFFUSION_FIELD_SOURCE_SHA"):
        assert source == os.environ["POPS_DIFFUSION_FIELD_SOURCE_SHA"]
    if os.environ.get("POPS_DIFFUSION_FIELD_NATIVE_SHA"):
        assert native_sha == os.environ["POPS_DIFFUSION_FIELD_NATIVE_SHA"]
    origin = verify_installed_native(
        expect_dimension=2,
        expect_mpi=bool(native.__has_mpi__),
        expect_parallel_hdf5=bool(native.__has_parallel_hdf5__),
    )
    world = native.mpi_world() if native.__has_mpi__ else None
    return world, dict(
        Source=source,
        Native=native_sha,
        SDK=module_header_signature(),
        package=str(package),
        native_origin=origin,
        rank=int(world.rank) if world else 0,
        ranks=int(world.size) if world else 1,
    )


def build(world, method, retry=False):
    authored = collective_call(
        world,
        lambda: author_case(
            method,
            automatic_retry=retry,
            layout_factory=one_level_amr_layout,
            field_solver=GeometricMG(
                tolerance=Relative(1e-13, floor=AbsoluteFloor(1e-14)), max_cycles=100
            ),
            hierarchy_policy=CompositeHierarchySolve(),
        ),
    )
    resolved = collective_call(
        world,
        lambda: pops.resolve(
            pops.validate(authored.case),
            layout=authored.layout,
            compile_options={"model_source_policy": "require"},
        ),
    )
    artifact = (
        compile_resolved_plan_once(
            world,
            resolved,
            route="public late Field refusal",
            compile_artifact=pops.compile,
        )
        if world
        else pops.compile(resolved)
    )
    return authored, artifact


def bind(world, authored, artifact, limit):
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    return collective_call(
        world,
        lambda: pops.bind(
            artifact,
            initial_values={subject: authored.initial},
            params=({} if authored.retry else {authored.refusal_limit: limit}),
            resources={"execution_context": artifact_execution_context(artifact)},
        ),
    )


def snapshot(world, runtime):
    # These are existing Native evidence accessors, not a parameter-update seam.
    executor = runtime._executor
    native = executor._s
    state = collective_call(
        world,
        lambda: np.asarray(runtime.block_level_state_global("material", 0)).copy(),
    )
    auxiliary = tuple(
        bytes(x)
        for x in collective_call(
            world, native.capture_auxiliary_checkpoint_accepted_state
        )
    )
    program = collective_call(world, runtime.program_accepted_state)
    potentials = []
    for slot in runtime.field_provider_slots():
        for level in range(runtime.field_provider_levels(slot)):
            value = collective_call(
                world,
                lambda slot=slot, level=level: np.asarray(
                    runtime.field_potential_level_global(slot, level)
                ).copy(),
            )
            potentials.append(
                (slot, level, value.dtype.str, value.shape, value.tobytes())
            )
    histories = []
    for name in runtime.history_names():
        for level in runtime.history_levels(name):
            initialized = bool(native.history_initialized(name, level))
            fill = int(native.history_fill_count(name, level))
            rows = []
            if initialized:
                for slot in range(runtime.history_depth(name)):
                    value = collective_call(
                        world,
                        lambda name=name, level=level, slot=slot: np.asarray(
                            runtime.history_global(name, level, slot)
                        ).copy(),
                    )
                    rows.append((value.dtype.str, value.shape, value.tobytes()))
            histories.append((name, level, initialized, fill, tuple(rows)))
    return dict(
        time=runtime.time(),
        macro_step=runtime.macro_step(),
        state=(state.dtype.str, state.shape, state.tobytes()),
        auxiliary=auxiliary,
        program=program,
        potentials=tuple(potentials),
        histories=tuple(histories),
    )


def save_snapshot(world, directory, name, payload):
    def lossless(value):
        if type(value) is bytes:
            return {
                "encoding": "base64",
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, dict):
            return {k: lossless(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [lossless(v) for v in value]
        return value

    collective_call(
        world,
        lambda: (directory / (name + ".json")).write_text(
            json.dumps(lossless(payload), indent=2, sort_keys=True) + "\n"
        ),
    )


def save_artifact(world, artifact, directory):
    for i, block in enumerate(artifact.blocks):
        collective_call(
            world,
            lambda i=i, block=block: block.model.dump_cpp(
                directory / ("model%d.cpp" % i)
            ),
        )
    for i, row in enumerate(artifact.layout_programs):
        collective_call(
            world,
            lambda i=i, row=row: row.program.dump_cpp(
                directory / ("program%d.cpp" % i)
            ),
        )


def save_positive(world, runtime, authored, directory, identity, adaptive=False):
    inputs = dict(
        identity,
        method=authored.method,
        cells=[16, 12],
        kappa=0.1,
        dt=float(DT),
        mean=2.0,
        amplitude=0.5,
        rhs_offset=3.0,
        field_stage_fraction=1,
        solver_rtol=1e-13,
        solver_atol=1e-14,
        actual_context_scope=dict(
            layout="AMR", levels=1, rank=identity["rank"], ranks=identity["ranks"]
        ),
    )
    collective_call(
        world,
        lambda: (directory / "inputs.json").write_text(
            json.dumps(inputs, indent=2, sort_keys=True) + "\n"
        ),
    )
    rows = {"initial": authored.initial}
    rows["accepted"] = collective_call(
        world,
        lambda: (
            np.asarray(runtime.block_level_state_global("material", 0))
            .reshape(authored.initial.shape)
            .copy()
        ),
    )
    for name, key in [("predictor_Y", "predictor"), ("phi_stage", "phi_stage")]:
        rows[key] = collective_call(
            world,
            lambda name=name: (
                np.asarray(runtime.history_global(name, 0, 0))
                .reshape(authored.initial.shape)
                .copy()
            ),
        )
    for name, array in rows.items():
        collective_call(
            world,
            lambda name=name, array=array: np.save(
                directory / (name + ".npy"), array, allow_pickle=False
            ),
        )
    with collective_check(world):
        oracle = check_adaptive_saved(directory) if adaptive else check_saved(directory)
        assert oracle["status"] == "PASS"


@pytest.mark.parametrize("method", ["euler", "ssprk2"])
def test_public_late_refusal_same_artifact_control(
    method, tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    world, identity = context()
    authored, artifact = build(world, method)
    shared = collective_directory(world, tmp_path / ("late-refusal-" + method))
    directory = shared / ("rank%d" % identity["rank"])
    collective_call(world, lambda: directory.mkdir(exist_ok=False))
    declared = dict(
        identity,
        artifact=artifact.artifact_identity.token,
        guard="norm2(Y)<=max(limit + 0*Y)",
        failed_limit=0.0,
        control_limit=40.0,
        same_artifact=True,
        same_instance_retry=False,
        rank0_only=False,
        snapshot_scope="instrumented Native journal; no storage mutation",
    )
    collective_call(
        world,
        lambda: (directory / "declared-inputs.json").write_text(
            json.dumps(declared, indent=2, sort_keys=True) + "\n"
        ),
    )
    save_artifact(world, artifact, directory)
    failed = bind(world, authored, artifact, 0.0)
    before = snapshot(world, failed)
    save_snapshot(world, directory, "before-refusal", before)
    _, failures = collective_attempt(
        world, lambda: pops.run(failed, t_end=float(DT), max_steps=1, console=False)
    )
    collective_call(
        world,
        lambda: (directory / "refusal-diagnostics.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        ),
    )
    after = snapshot(world, failed)
    save_snapshot(world, directory, "after-refusal", after)
    with collective_check(world):
        assert len(failures) == identity["ranks"] and all(
            row is not None and row[2] and "late_field_norm_refusal" in row[1]
            for row in failures
        ), failures
        assert before == after
    control = bind(world, authored, artifact, 40.0)
    report = collective_call(
        world, lambda: pops.run(control, t_end=float(DT), max_steps=1, console=False)
    )
    with collective_check(world):
        assert report.accepted_steps == 1 and control.time() == float(DT)
    save_positive(world, control, authored, directory, identity)
    result = dict(
        declared,
        status="PASS",
        failure_records=failures,
        accepted_envelope_exact=True,
        control_report=report.to_data(),
        ticket_validator_proof=False,
        same_instance_retry=False,
        snapshot_scope="instrumented Native journal; public snapshot API not qualified",
    )
    collective_call(
        world,
        lambda: (directory / "actual-receipt.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        ),
    )
    record_property("late_refusal_evidence", str(directory))


def test_public_late_refusal_automatic_retry(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    world, identity = context()
    authored, artifact = build(world, "euler", retry=True)
    shared = collective_directory(world, tmp_path / "late-refusal-adaptive")
    directory = shared / ("rank%d" % identity["rank"])
    collective_call(world, lambda: directory.mkdir(exist_ok=False))
    declared = dict(
        identity,
        artifact=artifact.artifact_identity.token,
        guard="norm2(Y-U_n)<=0.1",
        dt_init=2 * float(DT),
        shrink=0.5,
        dt_accepted_expected=float(DT),
        t_end=2 * float(DT),
        max_steps=2,
        max_rejections=1,
        norm_bounds_before_run={"initial": [0.14, 0.16], "retry": [0.07, 0.08]},
    )
    collective_call(
        world,
        lambda: (directory / "declared-inputs.json").write_text(
            json.dumps(declared, indent=2, sort_keys=True) + "\n"
        ),
    )
    save_artifact(world, artifact, directory)
    runtime = bind(world, authored, artifact, 0.0)
    report = collective_call(
        world,
        lambda: pops.run(runtime, t_end=2 * float(DT), max_steps=2, console=False),
    )
    with collective_check(world):
        assert report.accepted_steps == 2 and report.rejected_steps == 1
        assert runtime.time() == 2 * float(DT) and runtime.macro_step() == 2
        for name in runtime.history_names():
            assert runtime._executor._s.history_fill_count(name, 0) == 1
    save_positive(world, runtime, authored, directory, identity, adaptive=True)
    result = dict(
        declared,
        status="PASS",
        effective_accepted_dt=float(DT),
        accepted_intervals=2,
        report=report.to_data(),
        scope="public automatic dt retry; not parameter mutation or rank0-only refusal",
        ticket_validator_proof=False,
    )
    collective_call(
        world,
        lambda: (directory / "actual-receipt.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        ),
    )
    record_property("late_refusal_adaptive_evidence", str(directory))
