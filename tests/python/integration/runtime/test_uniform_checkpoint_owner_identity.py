"""Native Uniform checkpoints keep logical owners through presentation changes."""

from pathlib import Path
from struct import unpack_from
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import pops
from examples.api_v02 import models_M14
from pops.codegen.provider_instances import require_instance_contract, runtime_key
from pops.model.provider_pack import ProviderPack
from pops.runtime._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY, seal_checkpoint_payload
from pops.runtime._uniform_checkpoint_projection import inspect_uniform_checkpoint
from pops.time._program.integrals import integral_identity
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
    images, programs, columns, projections, receivers, checkpoints = [], [], [], [], [], []
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

        def witness_columns(continuous, resumed, path):
            native = continuous._executor._s
            locations = tuple(native.auxiliary_address(*key)[1] for key in keys)
            assert tuple(resumed._executor._s.auxiliary_address(*key)[1]
                         for key in keys) == locations
            columns.append(locations)
            archive = path.read_bytes()
            before = resumed.observe_accepted_state_storage().complete
            with pytest.raises(RuntimeError, match="no complete exchange incidence authority"):
                resumed._executor._s._mounted_exchange_incidence()
            with patch.object(resumed._executor, "_begin_checkpoint_restart",
                              side_effect=AssertionError("O3 opened a restart transaction")):
                inventory = inspect_uniform_checkpoint(resumed, archive)
                if not projections:
                    with np.load(path, allow_pickle=False) as stored:
                        permuted = {name: stored[name] for name in stored.files
                                    if name not in (MANIFEST_KEY, IDENTITY_KEY)}
                    permuted["blocks"] = permuted["blocks"][::-1]
                    seal_checkpoint_payload(continuous, permuted, runtime_kind="uniform")
                    invalid = path.parent / "permuted-blocks.npz"
                    np.savez_compressed(invalid, **permuted)
                    with pytest.raises(ValueError, match="current composition"):
                        inspect_uniform_checkpoint(resumed, invalid.read_bytes())
            assert resumed.observe_accepted_state_storage().complete == before
            assert path.read_bytes() == archive
            with np.load(path, allow_pickle=False) as payload:
                assert {member.physical_name for member in inventory.members} == set(payload.files)
                for member in inventory.members:
                    array = payload[member.physical_name]
                    assert (member.dtype, member.shape) == (array.dtype.str, array.shape)
            projections.append(inventory)
            receivers.append(resumed)
            checkpoints.append(path)

        programs.append(artifact.program_id)
        images.append(_replay(artifact, blocks, initial, 0.01, tmp_path / ("transport-%d" % index),
                              continuation=auxiliary_storage, preflight_check=witness_columns))
    assert sorted(columns[0]) == [0, 1]
    assert columns[1] == columns[0][::-1], "presentation changes reverse live storage columns"
    assert programs[0] == programs[1]
    assert images[0][1:] == images[1][1:]
    assert images[0][0][0] == images[1][0][0], "carrier depends on the presentation owners"
    assert images[0][0][1] == images[1][0][1], "auxiliary depends on the presentation owners"
    a, b = projections
    assert a.binding.owner != b.binding.owner
    assert a.binding.identity == b.binding.identity
    assert [row.name for row in a.binding.blocks] == list(FIRST["blocks"])
    assert [row.name for row in b.binding.blocks] == list(renamed["blocks"])
    assert [(row.path, row.kind) for row in a.members] == [(row.path, row.kind) for row in b.members]
    for inventory, names in ((a, FIRST), (b, renamed)):
        assert [(row.path, row.physical_name) for row in inventory.members if row.path[0] == "state"
                and row.path[-1] == "values"] == [
                    (("state", index, "U", "values"), "state_" + name)
                    for index, name in enumerate(names["blocks"])]
    before = receivers[1].observe_accepted_state_storage().complete
    with patch.object(receivers[1]._executor, "_begin_checkpoint_restart",
                      side_effect=AssertionError("different named bind reached restart")):
        with pytest.raises(ValueError, match="artifact_identity"):
            receivers[1].restart(checkpoints[0])
        with pytest.raises(ValueError, match="artifact_identity"):
            inspect_uniform_checkpoint(receivers[1], checkpoints[0].read_bytes())
    assert receivers[1].observe_accepted_state_storage().complete == before


def test_projected_checkpoint_binding_covers_defaults_inputs_and_resources(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root

    def compile_with(default):
        return pops.compile(resolve_named_case(
            FIRST, supplied_auxiliary=True, runtime_loss_default=default))

    artifact = compile_with(0.5)
    changed_default = compile_with(0.75)
    assert artifact.program_id == changed_default.program_id

    def bind(compiled, *, initial_delta=0., auxiliary_delta=0., override=None, capacity=None):
        initial = {row.subject: np.full((1, 8, 8), 1. + initial_delta)
                   for row in compiled.plan.initial_condition_plan.bindings}
        auxiliary = {}
        for block in compiled.plan.blocks:
            operations = block.resolved_operations
            pack = ProviderPack.from_data(operations.to_data()["provider_evidence"]["auxiliary"])
            instance = require_instance_contract(operations, owner_qid=block.instance_owner_qid)
            (key,) = pack
            auxiliary[runtime_key(key, instance)] = np.full((8, 8), 2. + auxiliary_delta)
        resources = {"execution_context": artifact_execution_context(compiled)}
        if capacity is not None:
            resources["runtime_parameter_byte_capacity"] = capacity
        return pops.bind(compiled, initial_values=initial, aux=auxiliary, resources=resources,
                         params=override)

    owner = bind(artifact)
    capacity = owner._install_plan.runtime_parameter_storage["byte_capacity"]
    slots = artifact.bind_schema.runtime_slots
    assert len(slots) == 2
    variants = (
        bind(changed_default),
        bind(artifact, initial_delta=.25),
        bind(artifact, auxiliary_delta=.25),
        bind(artifact, override={slots[0].handle: .5}),
        bind(artifact, capacity=capacity + 64),
    )
    identities = [value.bound_snapshot._uniform_checkpoint_projection.identity
                  for value in (owner, *variants)]
    assert len(set(identities)) == 6


def test_m14_checkpoint_preserves_field_and_integral_authority(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, monkeypatch):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.runtime._system import System

    sealed_before_program = []

    install_program = System._install_uniform_program

    def install_and_witness(engine, path, component, authored, install_plan):
        capture = engine._s.capture_auxiliary_checkpoint_accepted_state
        before = _auxiliary_contract(capture())
        install_program(engine, path, component, authored, install_plan)
        after = _auxiliary_contract(capture())
        assert before != after
        sealed_before_program.append(before)

    monkeypatch.setattr(System, "_install_uniform_program", install_and_witness)

    def verify_inner_and_legacy(continuous, resumed, checkpoint, expected_incidence):
        native = resumed._executor._s
        accepted = resumed.observe_accepted_state_storage().complete
        clock = resumed.time(), resumed.macro_step()
        incidence = native._mounted_exchange_incidence()
        assert continuous._executor._s._mounted_exchange_incidence() == incidence
        assert len(incidence) == len(expected_incidence)
        actual = []
        for index, row in enumerate(incidence):
            (ordinal, block, native_block, destination, operation, occurrence, point, evaluation,
             axis, side, component, scale, units, initial) = row
            assert ordinal == index and block == native_block == 0
            assert operation and occurrence and point and evaluation
            assert units == '{"kind":"physical_dimension","powers":[]}'
            actual.append((destination, axis, side, component, scale, initial))
        assert actual == expected_incidence
        assert resumed.observe_accepted_state_storage().complete == accepted
        assert (resumed.time(), resumed.macro_step()) == clock
        # The owner can read this CP9, but the still-named exchange declaration has no O3 role.
        with patch.object(resumed._executor, "_begin_checkpoint_restart",
                          side_effect=AssertionError("O3 opened a restart transaction")):
            with pytest.raises(ValueError, match="program_exchange_state"):
                inspect_uniform_checkpoint(resumed, checkpoint.read_bytes())
        assert resumed.observe_accepted_state_storage().complete == accepted
        assert (resumed.time(), resumed.macro_step()) == clock
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

        wall_identity = integral_identity(artifact.program.program, wall.name)
        reservoir_identity = integral_identity(artifact.program.program, reservoir.name)
        expected = [
            (wall_identity, 0, 1, 0, -1.0, models_M14.INITIAL_WALL_CHARGE),
            (reservoir_identity, 0, 0, 0, 1.0, 0.0),
            (wall_identity, 0, 1, 2, 1.0, models_M14.INITIAL_WALL_CHARGE),
            (reservoir_identity, 0, 0, 2, -1.0, 0.0),
        ]
        programs.append(artifact.program_id)
        images.append(_replay(artifact, ("fluid",), spec.initial_values, spec.dt,
                              tmp_path / ("sheath-%d" % index), continuation=continuation,
                              oracle=spec.invariants,
                              preflight_check=lambda *args: verify_inner_and_legacy(*args, expected)))
    assert programs[0] == programs[1]
    assert images[0][1:] == images[1][1:]
    assert images[0][0][1] == images[1][0][1], "nested field authority depends on the Case name"
