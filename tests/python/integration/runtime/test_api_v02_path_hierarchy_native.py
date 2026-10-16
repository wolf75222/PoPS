"""Real C++ path barriers compose with an actual composite implicit solve."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_path_hierarchy_composition import build_path_hierarchy_composition
from tests.python.support.api_v02_native import _compile, _bind
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.relaxed_flux_receipts import fragments

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _integrated_faces(image):
    result = {}
    for face in fragments(image):
        key = (face["owner"], face["levels"], face["axis"], face["face"],
               face["coarse_face"], face["level"], face["role"])
        integral = float(face["weight"])*face["duration"]*face["area"]*face["payload"]
        result[key] = result.get(key, np.zeros_like(integral)) + integral
    assert any(np.any(value != 0) for value in result.values())
    return result


@pytest.mark.parametrize("mode,reference", (("branch", "straight"), ("fold", "unrolled")))
def test_nested_path_matches_explicit_composition_and_accepted_face_integrals(
    mode, reference, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    received = []
    for selected in (mode, reference):
        spec, artifact, world = _compile("path_hierarchy_"+selected,
            builder=build_path_hierarchy_composition, mode=selected)
        runtime = _bind(spec, artifact, world)
        report = collective_call(world, lambda runtime=runtime, spec=spec: pops.run(runtime, t_end=spec.t_end,
                                                         max_steps=1, console=False))
        metrics = collective_call(world, lambda runtime=runtime, spec=spec: spec.invariants(runtime))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert runtime.n_levels() == 2 and runtime.time() == spec.t_end
            assert metrics["passed"], metrics
        arrays = {}
        for block in ("transport", "diffusion"):
            for level in range(runtime.n_levels()):
                arrays[block, level] = collective_call(world, lambda block=block, level=level, runtime=runtime:
                    runtime.block_level_state_global(block, level))
        image = collective_call(world, lambda runtime=runtime: bytes(runtime._executor.program_accepted_state()))
        ledger = collective_call(world, lambda image=image: _integrated_faces(image))
        received.append((spec, arrays, ledger))
        record_property(selected+"_artifact", artifact.artifact_identity.hexdigest)
    with collective_check(world):
        first, second = received
        assert first[2].keys() == second[2].keys()
        for key in first[2]:
            np.testing.assert_allclose(first[2][key], second[2][key], rtol=0, atol=5.e-12)
        for key, left in first[1].items():
            right = second[1][key]
            if left is None:
                assert right is None
                continue
            np.testing.assert_allclose(left, right, rtol=0, atol=5.e-12)
        # Both authored operators must actually change their nonconstant inputs.
        for block in ("transport", "diffusion"):
            left = first[1][block, 0]
            if left is not None:
                initial = np.asarray(spec.initial_values[spec.subjects[block]])
                assert np.max(abs(np.asarray(left).reshape(initial.shape)-initial)) > 1.e-8
