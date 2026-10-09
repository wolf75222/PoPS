"""ROOT-installed Dim1 reception; no source test runs native compilation here."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.integration.runtime.test_public_evolved_original_stage import (
    checkpoint_provenance, compare_checkpoint_replay,
)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.ns_conduction import (
    ACCEPTANCE, CONTROLS, CV, DT, FD_STEP, GAMMA, GAS_R, KAPPA, MU,
    build, check_step, convective_action, initial_data, primitives,
)


def capture(world, runtime, cells, step):
    q = collective_call(world, lambda: runtime.state_global("fluid"))
    with collective_check(world):
        image = {"Q": np.asarray(q).reshape(3, cells).copy()}
    metadata = {}
    for name, width in (("density", 1), ("velocity", 1), ("temperature", 1),
                        ("issued-convective-Q", 3), ("primitive-seed", 3), ("Euler-RHS", 3)):
        for slot in (0, 1):
            value = collective_call(world, lambda name=name, slot=slot: runtime.history_global(name, slot))
            with collective_check(world):
                image[name+"-slot%d" % slot] = np.asarray(value).reshape(width, cells).copy()
        fill = collective_call(world, lambda name=name: runtime._executor.history_fill_count(name))
        depth = collective_call(world, lambda name=name: runtime.history_depth(name))
        sample = collective_call(world, lambda name=name: bytes(runtime._executor.history_sample_identity(name)))
        durations = tuple(collective_call(world, lambda name=name, slot=slot:
            runtime.history_slot_dt(name, slot)) for slot in (0, 1))
        with collective_check(world):
            assert fill == min(step, 2) and depth == 2
            image[name+"-sample"] = np.frombuffer(sample, dtype=np.uint8).copy()
            metadata[name] = {"fill": fill, "depth": depth, "slot_durations": [value.hex() for value in durations]}
    diagnostics = collective_call(world, lambda: dict(runtime._executor.program_diagnostics()))
    lifecycle = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
    carrier = collective_call(world, lambda: bytes(runtime._executor.capture_auxiliary_checkpoint_accepted_state()))
    with collective_check(world):
        assert lifecycle == (step*DT, step)
        assert any(name.endswith(".rel_residual") and value <= CONTROLS["tolerance"] for name, value in diagnostics.items())
        image["auxiliary_checkpoint_accepted_state"] = np.frombuffer(carrier, dtype=np.uint8).copy()
    return image, {"histories": metadata, "diagnostics": diagnostics, "time": lifecycle[0].hex(), "step": lifecycle[1]}


def exact_images(left, right):
    assert left[1] == right[1] and left[0].keys() == right[0].keys()
    for name in left[0]:
        assert left[0][name].dtype == right[0][name].dtype and left[0][name].shape == right[0][name].shape
        assert left[0][name].tobytes() == right[0][name].tobytes(), name


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("cells", (8, 16))
def test_public_compressible_ns_conduction_original_stage_exact_replay(
        isolated_native_cache, tmp_path, record_property, cells):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(1)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout, _ = collective_call(world, lambda: build(cells))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="compressible-NS-conduction-Uniform1D",
                                         compile_artifact=pops.compile)
    initial = collective_call(world, lambda: initial_data(cells))
    def bind():
        subject = artifact.plan.initial_condition_plan.bindings[0].subject
        return pops.bind(artifact, initial_values={subject: initial.copy()},
                         resources={"execution_context": artifact_execution_context(artifact)})
    runtime = collective_call(world, bind)
    directory = collective_directory(world, tmp_path/"compressible-NS-conduction")
    q0 = collective_call(world, lambda: runtime.state_global("fluid"))
    with collective_check(world):
        np.testing.assert_array_equal(np.asarray(q0).reshape(initial.shape), initial)
    checkpoints, cp_hashes, authorities, phases = {}, {}, {}, {}
    for phase, step in (("accepted", 1), ("continuous", 2)):
        collective_call(world, lambda step=step: pops.run(runtime, t_end=step*DT, max_steps=1, console=False))
        checkpoints[phase] = collective_call(world, lambda phase=phase: runtime.checkpoint(directory/(phase+"-checkpoint")))
        cp_hashes[phase] = collective_call(world, lambda phase=phase: hashlib.sha256(bounded_bytes(checkpoints[phase])).hexdigest())
        authorities[phase] = collective_call(world, lambda phase=phase: checkpoint_provenance(runtime, checkpoints[phase]))
        phases[phase] = capture(world, runtime, cells, step)
    restored = collective_call(world, bind)
    collective_call(world, lambda: restored.restart(checkpoints["accepted"]))
    phases["reloaded"] = capture(world, restored, cells, 1)
    with collective_check(world):
        exact_images(phases["reloaded"], phases["accepted"])
    collective_call(world, lambda: pops.run(restored, t_end=2*DT, max_steps=1, console=False))
    checkpoints["replay"] = collective_call(world, lambda: restored.checkpoint(directory/"replay-checkpoint"))
    cp_hashes["replay"] = collective_call(world, lambda: hashlib.sha256(bounded_bytes(checkpoints["replay"])).hexdigest())
    authorities["replay"] = collective_call(world, lambda: checkpoint_provenance(restored, checkpoints["replay"]))
    phases["replay"] = capture(world, restored, cells, 2)
    with collective_check(world):
        exact_images(phases["replay"], phases["continuous"])
        np.testing.assert_array_equal(phases["continuous"][0]["temperature-slot0"], phases["accepted"][0]["temperature-slot1"])
        if world.rank == 0:
            equivalence = compare_checkpoint_replay(checkpoints, authorities)
            initial_path = directory/"initial.npz"
            np.savez(initial_path, Q=initial, inverse_EOS=primitives(initial),
                     coordinates=(np.arange(cells)+.5)/cells, cell_volumes=np.full(cells, 1/cells))
            saved_phases = {}
            for phase, (image, metadata) in phases.items():
                previous = initial if phase in ("accepted", "reloaded") else phases["accepted"][0]["Q"]
                psi = np.concatenate(tuple(image[name+"-slot1"] for name in ("density", "velocity", "temperature")))
                checks, references = check_step(image["Q"], psi, previous)
                euler_rhs, _ = convective_action(previous)
                np.testing.assert_allclose(image["Euler-RHS-slot1"], euler_rhs, rtol=0, atol=ACCEPTANCE)
                np.testing.assert_allclose(image["issued-convective-Q-slot1"], references["issued_Q"], rtol=0, atol=ACCEPTANCE)
                np.testing.assert_allclose(image["primitive-seed-slot1"], primitives(image["issued-convective-Q-slot1"]), rtol=0, atol=ACCEPTANCE)
                path = directory/(phase+".npz")
                np.savez(path, **image, **{"COMPUTED_REFERENCE-"+name: value for name, value in references.items()},
                         actual_time=float.fromhex(metadata["time"]), actual_step=metadata["step"])
                with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as stored:
                    for name in image:
                        np.testing.assert_array_equal(stored[name], image[name])
                saved_phases[phase] = {"path": str(path), "sha256": hashlib.sha256(bounded_bytes(path)).hexdigest(),
                                       "checks": checks, "native_metadata": metadata}
            components = [("block-"+row.name, row.model) for row in artifact.blocks]
            components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
            binaries = []
            from pops.codegen.compile_provenance import artifact_sidecar_path
            for index, (name, component) in enumerate(components):
                binary = Path(component.so_path)
                sidecar = Path(artifact_sidecar_path(str(binary)))
                assert sidecar.is_file()
                row = {"component": name, "path": str(binary), "sha256": hashlib.sha256(bounded_bytes(binary)).hexdigest(),
                       "sidecar": {"path": str(sidecar), "sha256": hashlib.sha256(bounded_bytes(sidecar)).hexdigest()},
                       "compile_command": getattr(component, "compile_command", None)}
                if name.startswith("program-"):
                    assert isinstance(component.compile_command, str) and component.compile_command
                    source = Path(component.dump_cpp(directory/("program-%d.cpp" % index)))
                    ir = Path(component.dump_ir(directory/("program-%d.ir.json" % index)))
                    row.update(source={"path": str(source), "sha256": hashlib.sha256(bounded_bytes(source)).hexdigest()},
                               ir={"path": str(ir), "sha256": hashlib.sha256(bounded_bytes(ir)).hexdigest(), "program_hash": component.program_hash})
                binaries.append(row)
            assert {Path(value).resolve() for value in checkpoints.values()}.isdisjoint(
                {Path(row["path"]).resolve() for row in saved_phases.values()} | {initial_path.resolve()})
            for phase, path in checkpoints.items():
                assert hashlib.sha256(bounded_bytes(path)).hexdigest() == cp_hashes[phase]
            receipt = {"fixture_schema": "pops.compressible-NS-conduction-native@1", "dimension": 1,
                "cells": cells, "dt": DT, "rank": world.rank, "size": world.size,
                "EOS": {"gamma": GAMMA, "R": GAS_R, "Cv": CV}, "mu": MU, "kappa": KAPPA,
                "mesh": {"cell_shape": [cells], "lower": [0.], "upper": [1.], "periodic_axes": ["x"],
                         "geometry_reference": "declared Uniform grid; coordinates and volumes are derived references"},
                "newton": CONTROLS, "fd_step": FD_STEP, "acceptance": ACCEPTANCE,
                "realization": {"Euler": "FirstOrder/Rusanov", "viscous_thermal": "Arithmetic@1/PerCandidate@1",
                                "previous": "issued@1", "projection": "piecewise_constant_cell"},
                "artifact": artifact.artifact_identity.token, "platform": artifact.platform_manifest.to_data(),
                "native": {"path": str(native.__file__), "sha256": hashlib.sha256(bounded_bytes(native.__file__)).hexdigest()},
                "binaries": binaries, "initial": {"path": str(initial_path), "sha256": hashlib.sha256(bounded_bytes(initial_path)).hexdigest()},
                "phases": saved_phases, "checkpoints": {phase: {"path": str(path), "sha256": cp_hashes[phase]}
                                                        for phase, path in checkpoints.items()},
                "checkpoint_equivalence": equivalence, "exact_restart_and_replay": True,
                "flux_reference_provenance": "COMPUTED_REFERENCE from saved actual fields, not a native face getter"}
            (directory/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        for name, value in (("artifact_identity", artifact.artifact_identity.token), ("dimension", 1),
                            ("rank", world.rank), ("size", world.size), ("evidence_path", str(directory)),
                            ("ns_conduction_receipt", str(directory/"receipt.json"))):
            record_property(name, value)
