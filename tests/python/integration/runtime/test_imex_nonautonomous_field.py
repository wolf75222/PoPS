"""Nonautonomous IMEX field-time witness on one-level AMR/MG; Native gated by Root.

This observes stage time, not multilevel refinement or Uniform screened support.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pops
import pytest
from pops._generated_release_contract import NATIVE_ABI_VERSION
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_v_capture import retain_v_provenance, pin
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.imex_nonautonomous_field_case import (
    build_case, H, U0, PHYSICAL_T0, Y_EXACT, FIELD_EXACT, ACCEPTED_EXACT,
    WRONG_FIELD_TIME_ACCEPTED, error_bounds,
)

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def test_public_nonautonomous_imex_field_reads_explicit_time(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, monkeypatch):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
    native = select_native_dimension(2)
    world = native.mpi_world()
    directory = collective_directory(world, tmp_path / "nonautonomous-imex")
    authored = collective_call(world, build_case)
    validated = collective_call(world, lambda: pops.validate(authored.case))
    resolved = collective_call(world, lambda: pops.resolve(validated, layout=authored.layout,
        compile_options={"model_source_policy": "require"}))
    # Capture the existing compiler input before its TemporaryDirectory/staging cleanup.
    # This observer never changes the compiler's arguments, result or exception.
    from pops.codegen import model_compile_evidence
    original_retain = model_compile_evidence.retain
    input_records = []

    def retain_actual_input(binary, source, command, header_signature, original):
        ordinal = len(input_records)
        record = dict(rank=int(world.rank), entry_point="model_compile_evidence.retain",
            ordinal=ordinal, source_path=str(source), binary_path=str(binary),
            command=list(command), header_signature=header_signature,
            compiler_callback=dict(module=getattr(original, "__module__", None),
                                   name=getattr(original, "__qualname__", None)),
            outcome="entered")
        input_records.append(record)
        evidence = directory / "actual-compiler-inputs"
        receipt_path = evidence / ("rank%d-input%d.json" % (world.rank, ordinal))

        def record_outcome():
            try:
                receipt_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
            except Exception:
                pass  # An observer failure must never replace the compilation outcome.

        try:
            evidence.mkdir(exist_ok=True)
            raw = Path(source).read_bytes()
            retained = evidence / ("rank%d-input%d.cpp" % (world.rank, ordinal))
            retained.write_bytes(raw)
            record.update(source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
                          retained_source=str(retained))
            record_outcome()
        except Exception as error:
            record["capture_error"] = dict(type=type(error).__name__, message=str(error))
        try:
            result = original_retain(binary, source, command, header_signature, original)
        except BaseException as error:
            record.update(outcome="raised", exception=dict(type=type(error).__name__, message=str(error)))
            record_outcome()
            raise
        else:
            record.update(outcome="returned", result_type=type(result).__name__)
            record_outcome()
            return result

    with monkeypatch.context() as patch:
        patch.setattr(model_compile_evidence, "retain", retain_actual_input)
        artifact = compile_resolved_plan_once(world, resolved, route="nonautonomous IMEX field time",
                                              compile_artifact=pops.compile)
    collective_call(world, lambda: retain_v_provenance(artifact, native, directory, world.rank))
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values={subject: authored.initial},
        resources={"execution_context": artifact_execution_context(artifact)}))
    field_bound, state_bound = error_bounds(authored.cells)
    with collective_check(world):
        assert native.module_capabilities("production")["abi_version"] == NATIVE_ABI_VERSION
        assert runtime.time() == 0 and runtime.macro_step() == 0
        assert state_bound < abs(float(WRONG_FIELD_TIME_ACCEPTED - ACCEPTED_EXACT)) / 1000
    before = collective_call(world, lambda: np.asarray(runtime.state_global("material")).copy())
    collective_call(world, lambda: np.save(directory / ("initial-rank%d.npy" % world.rank),
        before, allow_pickle=False))
    with collective_check(world):
        np.testing.assert_array_equal(before, authored.initial)
    result = collective_call(world, lambda: pops.run(runtime, t_end=float(H), max_steps=1, console=False))
    actual = collective_call(world, lambda: np.asarray(runtime.state_global("material")).copy())
    slots = collective_call(world, runtime.field_provider_slots)
    with collective_check(world):
        assert len(slots) == 1
    field = collective_call(world, lambda: np.asarray(runtime.field_potential_global(slots[0])).copy())
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / "accepted"))
    collective_call(world, lambda: np.save(directory / ("accepted-rank%d.npy" % world.rank),
        actual, allow_pickle=False))
    collective_call(world, lambda: np.save(directory / ("field-rank%d.npy" % world.rank),
        field, allow_pickle=False))
    with collective_check(world):
        assert result.accepted_steps == 1 and runtime.macro_step() == 1
        assert abs(runtime.time() - float(H)) <= 16 * np.finfo(float).eps
        np.testing.assert_allclose(actual, float(ACCEPTED_EXACT), rtol=0, atol=state_bound)
        np.testing.assert_allclose(field, float(FIELD_EXACT), rtol=0, atol=field_bound)
        assert np.min(np.abs(actual - float(WRONG_FIELD_TIME_ACCEPTED))) > .1
    def receipt():
        value = dict(scope="Nonautonomous constant-state screened-field IMEX on one-level AMR/MG; no multilevel or full model qualification",
            runtime_tau0=0, physical_time_origin=str(PHYSICAL_T0), h=str(H), U0=str(U0),
            Y_reference=str(Y_EXACT), field_reference=str(FIELD_EXACT), accepted_reference=str(ACCEPTED_EXACT),
            wrong_field_time_reference=str(WRONG_FIELD_TIME_ACCEPTED),
            bounds={"field": field_bound, "state": state_bound},
            source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                str(Path(build_case.__code__.co_filename).resolve()): hashlib.sha256(Path(build_case.__code__.co_filename).read_bytes()).hexdigest()},
            native=pin(native.__file__), abi=native.abi_key(), artifact=artifact.artifact_identity.token,
            rank=world.rank, ranks=world.size, actual_clock=[runtime.time(), runtime.macro_step()],
            checkpoint=pin(checkpoint), root_received=False)
        (directory / ("receipt-rank%d.json" % world.rank)).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    collective_call(world, receipt)
    record_property("nonautonomous_imex_receipt", str(directory / ("receipt-rank%d.json" % world.rank)))
