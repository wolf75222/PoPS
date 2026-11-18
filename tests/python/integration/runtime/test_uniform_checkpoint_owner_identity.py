"""Native Uniform checkpoints keep logical owners through presentation changes."""

from pathlib import Path
from struct import unpack_from
from types import SimpleNamespace

import numpy as np
import pytest

import pops
from examples.api_v02 import models_M14
from pops.codegen.provider_instances import require_instance_contract, runtime_key
from pops.model.provider_pack import ProviderPack
from pops.runtime._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY, seal_checkpoint_payload
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.presentation_case import FIRST, resolve_named_case


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[member].tobytes() for member in (
            "state_carriers_checkpoint", "auxiliary_checkpoint"))


def _auxiliary_contract(image):
    (size,) = unpack_from("<Q", image, 16)
    return image[24:24 + size]


def _image(runtime, names):
    arrays = [runtime.state_global(block) for block in names]
    arrays.extend(runtime.field_potential_global(slot) for slot in runtime.field_provider_slots())
    return (runtime.time(), runtime.macro_step(),
            tuple((np.asarray(value).dtype.str, np.asarray(value).shape,
                   np.asarray(value).tobytes()) for value in arrays))


def _advance(runtime, time):
    report = pops.run(runtime, t_end=time, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == 0


def _replace_member(runtime, path, destination, **members):
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files
                   if key not in (MANIFEST_KEY, IDENTITY_KEY)}
    for key, image in members.items():
        payload[key] = np.frombuffer(image, dtype=np.uint8).copy()
    seal_checkpoint_payload(runtime, payload, runtime_kind="uniform")
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    return destination


def _replay(artifact, names, initial, dt, directory, *, continuation=lambda _: (), oracle=None,
            preflight_check=None):
    def bind():
        return pops.bind(artifact, initial_values=initial,
                         resources={"execution_context": artifact_execution_context(artifact)})

    continuous = bind()
    _advance(continuous, dt)
    checkpoint = Path(continuous.checkpoint(directory))
    accepted = (_image(continuous, names), continuation(continuous))
    resumed = bind()
    resumed.restart(checkpoint)
    assert (_image(resumed, names), continuation(resumed)) == accepted
    if preflight_check is not None:
        preflight_check(continuous, resumed, checkpoint)
        assert (_image(resumed, names), continuation(resumed)) == accepted
    _advance(continuous, 2 * dt)
    _advance(resumed, 2 * dt)
    next_accepted = (_image(resumed, names), continuation(resumed))
    assert (_image(continuous, names), continuation(continuous)) == next_accepted
    if oracle is not None:
        for time in (3 * dt, 4 * dt):
            _advance(continuous, time)
            _advance(resumed, time)
        expected = oracle(continuous)
        assert expected["passed"], expected
        assert oracle(resumed) == expected
    return _snapshot(checkpoint), accepted, next_accepted


def test_transport_checkpoint_follows_logical_block_order(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path):
    del isolated_native_cache, native_cxx, kokkos_root
    # Change the relative lengths of the owner keys as well as their alphabetical order.
    # The native exact registry orders packed keys by their length-prefixed frames.
    renamed = {"case": "translated", "blocks": ("z", "earlier_name"),
               "models": ("m", "another_model")}
    images, programs, columns = [], [], []
    mesh = np.arange(64, dtype=np.float64).reshape(1, 8, 8) / 512
    for index, names in enumerate((FIRST, renamed)):
        artifact = pops.compile(resolve_named_case(names, auxiliary_offset=(0.25, 0.75)))
        blocks = names["blocks"]
        initial = {}
        for binding in artifact.plan.initial_condition_plan.bindings:
            block = binding.subject.block_ref.local_id
            initial[binding.subject] = mesh + (blocks.index(block) + 1) * 0.625
        keys = []
        for block in artifact.plan.blocks:
            operations = block.resolved_operations
            pack = ProviderPack.from_data(operations.to_data()["provider_evidence"]["auxiliary"])
            instance = require_instance_contract(operations, owner_qid=block.instance_owner_qid)
            (key,) = pack
            address = runtime_key(key, instance)
            keys.append((address.owner_qid, address.space_kind, address.space_name,
                         address.component))

        def auxiliary_storage(runtime):
            native = runtime._executor._s
            addresses = [native.auxiliary_address(*key) for key in keys]
            assert addresses[0][0] == addresses[1][0], "both columns occupy one actual group"
            fields = [native.auxiliary_component(*key) for key in keys]
            np.testing.assert_array_equal(fields[1] - fields[0], np.full(fields[0].shape, 0.5))
            return tuple(field.tobytes() for field in fields)

        def witness_columns(continuous, resumed, _):
            native = continuous._executor._s
            locations = tuple(native.auxiliary_address(*key)[1] for key in keys)
            assert tuple(resumed._executor._s.auxiliary_address(*key)[1]
                         for key in keys) == locations
            columns.append(locations)

        programs.append(artifact.program_id)
        images.append(_replay(artifact, blocks, initial, 0.01, tmp_path / ("transport-%d" % index),
                              continuation=auxiliary_storage, preflight_check=witness_columns))
    assert sorted(columns[0]) == [0, 1]
    assert columns[1] == columns[0][::-1], "presentation changes reverse live storage columns"
    assert programs[0] == programs[1]
    assert images[0][1:] == images[1][1:]
    assert images[0][0][0] == images[1][0][0], "carrier depends on the presentation owners"
    assert images[0][0][1] == images[1][0][1], "auxiliary depends on the presentation owners"


def test_m14_checkpoint_preserves_field_and_integral_authority(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, monkeypatch):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.runtime._system import System

    sealed_before_program = []

    def install_and_witness(engine, path):
        capture = engine._s.capture_auxiliary_checkpoint_accepted_state
        before = _auxiliary_contract(capture())
        engine._s.install_program(path)
        after = _auxiliary_contract(capture())
        assert before != after
        sealed_before_program.append(before)

    monkeypatch.setattr(System, "install_program", install_and_witness, raising=False)

    def verify_inner_and_legacy(continuous, resumed, checkpoint):
        # The historical readers use the same live registry and preserve its accepted payload.
        native = continuous._executor._s
        legacy_carrier = native.checkpoint_state_carriers()
        legacy_auxiliary = native.capture_auxiliary_checkpoint_accepted_state()
        assert legacy_carrier[:8] == b"POPSCAR1"
        assert legacy_auxiliary[:8] in (b"POPSAUX2", b"POPSAUX3")
        historical = _replace_member(
            continuous, checkpoint, checkpoint.parent / "historical.npz",
            state_carriers_checkpoint=legacy_carrier, auxiliary_checkpoint=legacy_auxiliary)
        resumed.restart(historical)
        assert resumed._executor._s.capture_auxiliary_checkpoint_accepted_state() == legacy_auxiliary

        carrier, auxiliary = _snapshot(checkpoint)
        assert carrier[:8] == b"POPSCAR2"
        assert auxiliary[:8] in (b"POPSAUX4", b"POPSAUX5")
        contract = _auxiliary_contract(auxiliary)
        contract_end = 24 + len(contract)
        assert b"/program/" in contract
        assert sealed_before_program
        assert all(b"/program/" not in early for early in sealed_before_program)
        corrupted = bytearray(auxiliary)
        corrupted[contract_end - 1] ^= 1
        rehashed = _replace_member(
            continuous, checkpoint, checkpoint.parent / "tampered.npz",
            auxiliary_checkpoint=bytes(corrupted))
        mixed = _replace_member(
            continuous, checkpoint, checkpoint.parent / "mixed.npz",
            state_carriers_checkpoint=legacy_carrier)

        def unexpected_begin():
            pytest.fail("the invalid inner image reached the restart transaction")

        with monkeypatch.context() as patch:
            patch.setattr(resumed._executor, "_begin_checkpoint_restart", unexpected_begin)
            with pytest.raises(ValueError, match="differs from the sealed provider registry"):
                resumed.restart(rehashed)
            with pytest.raises(ValueError, match="carrier and auxiliary checkpoint authority differ"):
                resumed.restart(mixed)

    images, programs = [], []
    for index, case_name in enumerate(("sheath", "renamed sheath")):
        def named_case(_):
            return pops.Case(case_name)

        with monkeypatch.context() as patch:
            patch.setattr(models_M14, "pops", SimpleNamespace(
                Case=named_case, Model=pops.Model, Program=pops.Program))
            spec, wall, reservoir = models_M14.build_m14(receipt=True)
        artifact = pops.compile(pops.resolve(pops.validate(spec.case), layout=spec.layout))

        def continuation(runtime):
            return (np.asarray(runtime.history_global("source_state", 1)).tobytes(),
                    runtime.integral_state(wall), runtime.integral_state(reservoir))

        programs.append(artifact.program_id)
        images.append(_replay(artifact, ("fluid",), spec.initial_values, spec.dt,
                              tmp_path / ("sheath-%d" % index), continuation=continuation,
                              oracle=spec.invariants, preflight_check=verify_inner_and_legacy))
    assert programs[0] == programs[1]
    assert images[0][1:] == images[1][1:]
    assert images[0][0][1] == images[1][0][1], "nested field authority depends on the Case name"
