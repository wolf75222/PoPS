"""Native Uniform checkpoints keep logical owners through presentation changes."""

from types import SimpleNamespace

import numpy as np
import pytest

import pops
from examples.api_v02 import models_M14
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.presentation_case import FIRST, resolve_named_case


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[member].tobytes() for member in (
            "state_carriers_checkpoint", "auxiliary_checkpoint"))


def _image(runtime, names):
    arrays = [runtime.state_global(block) for block in names]
    arrays.extend(runtime.field_potential_global(slot) for slot in runtime.field_provider_slots())
    return (runtime.time(), runtime.macro_step(),
            tuple((np.asarray(value).dtype.str, np.asarray(value).shape,
                   np.asarray(value).tobytes()) for value in arrays))


def _advance(runtime, time):
    report = pops.run(runtime, t_end=time, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == 0


def _replay(artifact, names, initial, dt, directory, *, continuation=lambda _: (), oracle=None):
    def bind():
        return pops.bind(artifact, initial_values=initial,
                         resources={"execution_context": artifact_execution_context(artifact)})

    continuous = bind()
    _advance(continuous, dt)
    checkpoint = continuous.checkpoint(directory)
    accepted = (_image(continuous, names), continuation(continuous))
    resumed = bind()
    resumed.restart(checkpoint)
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
    images, programs = [], []
    mesh = np.arange(64, dtype=np.float64).reshape(1, 8, 8) / 512
    for index, names in enumerate((FIRST, renamed)):
        artifact = pops.compile(resolve_named_case(names, auxiliary_offset=(0.25, 0.75)))
        blocks = names["blocks"]
        initial = {}
        for binding in artifact.plan.initial_condition_plan.bindings:
            block = binding.subject.block_ref.local_id
            initial[binding.subject] = mesh + (blocks.index(block) + 1) * 0.625
        programs.append(artifact.program_id)
        images.append(_replay(artifact, blocks, initial, 0.01, tmp_path / ("transport-%d" % index)))
    assert programs[0] == programs[1]
    assert images[0][1:] == images[1][1:]
    assert images[0][0][0] == images[1][0][0], "carrier depends on the presentation owners"
    assert images[0][0][1] == images[1][0][1], "auxiliary depends on the presentation owners"


def test_m14_checkpoint_preserves_field_and_integral_authority(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, monkeypatch):
    del isolated_native_cache, native_cxx, kokkos_root
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
                              oracle=spec.invariants))
    assert programs[0] == programs[1]
    assert images[0][1:] == images[1][1:]
    assert images[0][0][1] == images[1][0][1], "nested field authority depends on the Case name"
