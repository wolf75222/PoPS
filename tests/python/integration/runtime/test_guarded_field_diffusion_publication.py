"""Installed two-owner elliptic publication and nonlinear-coefficient diffusion."""

import hashlib
import json
import os
import subprocess
import sysconfig
from fractions import Fraction
from pathlib import Path

import numpy as np
import pops
import pytest
from pops.solvers import CompositeFieldGMRES
from tests.python.support.guarded_field_diffusion_case import author_case
from tests.python.support.evidence_json import evidence_dumps, ENCODING
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "variant,suffix,reaction,offset",
    [
        ("baseline", "", 3, Fraction(2, 7)),
        ("renamed", "_renamed", 3, Fraction(2, 7)),
        ("changed_expression", "", 5, Fraction(3, 11)),
    ],
    ids=["baseline", "renamed", "changed_expression"],
)
def test_public_guarded_field_coefficient_diffusion(
    variant,
    suffix,
    reaction,
    offset,
    tmp_path,
    isolated_native_cache,
    native_cxx,
    kokkos_root,
    record_property,
):
    del isolated_native_cache, native_cxx, kokkos_root
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
    native_hash = sha(native.__file__)
    if os.environ.get("POPS_GUARDED_FIELD_SOURCE_SHA"):
        assert source == os.environ["POPS_GUARDED_FIELD_SOURCE_SHA"]
    if os.environ.get("POPS_GUARDED_FIELD_NATIVE_SHA"):
        assert native_hash == os.environ["POPS_GUARDED_FIELD_NATIVE_SHA"]
    origin = verify_installed_native(
        expect_dimension=2,
        expect_mpi=bool(native.__has_mpi__),
        expect_parallel_hdf5=bool(native.__has_parallel_hdf5__),
    )
    paths = subprocess.check_output(
        ["git", "ls-files", "--", "python/pops"], cwd=root, text=True
    ).splitlines()
    paths = [p for p in paths if Path(p).suffix in PYTHON_SOURCE_SUFFIXES]
    paths += ["include/" + str(p) for p in read_manifest(root).installed_headers] + [
        "include/pops_headers.manifest"
    ]
    for name in paths:
        installed = package / (
            name.removeprefix("python/pops/")
            if name.startswith("python/pops/")
            else name
        )
        assert sha(installed) == sha(root / name), name
    world = native.mpi_world() if native.__has_mpi__ else None
    rank, ranks = (int(world.rank), int(world.size)) if world is not None else (0, 1)
    shared = collective_directory(world, tmp_path / ("guarded-field-" + variant))
    directory = shared / ("rank%d" % rank)
    collective_call(world, lambda: directory.mkdir(exist_ok=False))
    case, layout, arrays, expected = collective_call(
        world,
        lambda: author_case(
            suffix=suffix,
            reaction=reaction,
            diffusion_offset=offset,
            solver=CompositeFieldGMRES(max_iter=200, rel_tol=1e-12, abs_tol=1e-14),
        ),
    )
    state_bound = 4096 * np.finfo(np.float64).eps * 1.2
    field_bound = 16 * 12 * 0.6 / float(reaction) * 1e-12 + 1e-14 + state_bound
    inputs = dict(
        Source=source,
        Native=native_hash,
        SDK=module_header_signature(),
        variant=variant,
        suffix=suffix,
        package=str(package),
        native_origin=str(origin),
        context=native.runtime_environment_report(),
        equations={
            "field": "(-L_h+reaction I)phi=d",
            "receiver": "dq/dt=div((offset+published phi^2)grad(q))",
            "donor": "d_next=d_n",
        },
        cells=[16, 12],
        domain_lengths=[2, 3],
        dt=expected["dt"],
        forcing=0.6,
        reaction=float(reaction),
        diffusion_offset=float(offset),
        expected_coefficient=expected["coefficient"],
        sufficient_fe_frequency=expected["forward_euler_frequency"],
        state_bound=float(state_bound),
        field_bound=float(field_bound),
        solver={
            "type": "CompositeFieldGMRES",
            "max_iter": 200,
            "rel_tol": 1e-12,
            "abs_tol": 1e-14,
        },
        accepted_history_observed_physical_slot=1,
        scope={"levels": 1, "rank": rank, "ranks": ranks},
        source_helpers={
            str(Path(author_case.__code__.co_filename).resolve()): sha(
                author_case.__code__.co_filename
            )
        },
    )
    # Immutable literal inputs and independent references precede compilation/execution.
    collective_call(
        world,
        lambda: (directory / "declared-inputs.json").write_text(
            json.dumps(inputs, indent=2, sort_keys=True) + "\n"
        ),
    )
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            collective_call(
                world,
                lambda name=name, value=value: np.save(
                    directory / ("expected-" + name + ".npy"), value, allow_pickle=False
                ),
            )
    resolved = collective_call(
        world,
        lambda: pops.resolve(
            pops.validate(case),
            layout=layout,
            compile_options={"model_source_policy": "require"},
        ),
    )
    artifact = (
        compile_resolved_plan_once(
            world,
            resolved,
            route="guarded two-owner public Field diffusion",
            compile_artifact=pops.compile,
        )
        if world is not None
        else pops.compile(resolved)
    )
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
    initial = {
        row.subject: arrays[row.subject.block_ref.local_id]
        for row in artifact.plan.initial_condition_plan.bindings
    }
    runtime = collective_call(
        world,
        lambda: pops.bind(
            artifact,
            initial_values=initial,
            resources={"execution_context": artifact_execution_context(artifact)},
        ),
    )
    report = collective_call(
        world,
        lambda: pops.run(runtime, t_end=expected["dt"], max_steps=1, console=False),
    )
    actual = {}
    for role in ["donor", "receiver"]:
        actual[role] = collective_call(
            world,
            lambda role=role: (
                np.asarray(runtime.block_level_state_global(role + suffix, 0))
                .reshape(expected[role].shape)
                .copy()
            ),
        )
    actual["phi"] = collective_call(
        world,
        lambda: (
            np.asarray(runtime.history_global("actual-solved-phi" + suffix, 0, 1))
            .reshape(expected["phi"].shape)
            .copy()
        ),
    )
    for name, array in actual.items():
        collective_call(
            world,
            lambda name=name, array=array: np.save(
                directory / ("actual-" + name + ".npy"), array, allow_pickle=False
            ),
        )
    errors = {}
    with collective_check(world):
        assert (
            report.accepted_steps == 1
            and runtime.macro_step() == 1
            and runtime.time() == expected["dt"]
        )
        for name, array in actual.items():
            assert np.isfinite(array).all()
            error = float(np.max(np.abs(array - expected[name])))
            bound = field_bound if name == "phi" else state_bound
            assert error <= bound, (name, error, bound)
            errors[name] = {"max_abs_error": error, "bound": float(bound)}
    receipt = dict(
        inputs,
        status="PASS",
        artifact=artifact.artifact_identity.token,
        comparisons=errors,
        report=report.to_data(),
        evidence_encoding=ENCODING,
        limits=[
            "one-level qualification only; no refinement/coarse-fine extrapolation",
            "actual loaded installed core authenticated separately for each run",
        ],
    )
    collective_call(
        world,
        lambda: (directory / "actual-receipt.json").write_text(evidence_dumps(receipt)),
    )
    record_property("guarded_field_diffusion_evidence", str(directory))
