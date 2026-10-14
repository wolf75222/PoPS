"""Four-component, genuinely 2D Euler face correction; no runtime substitute."""
import numpy as np
import pytest

from examples.api_v02.method_euler_correction import authored_flux, build_euler_correction, reference_step
from pops.identity import make_identity
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile, profile_gather_value


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("cfl", (.25, .4))
def test_native_euler_two_recipient_face_correction(cfl, isolated_native_cache,
                                                   native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="T12 Euler shared face correction")
    runtime, artifact, metrics = execute_case(lambda: build_euler_correction(cfl=cfl), world,
                                              "T12 Euler corrected mesh faces CFL="+str(cfl))
    local = collective_call(world, runtime._executor._program_exchange_records)
    shards = profile_gather_value(world, local)
    with collective_check(world):
        # Global shards or exact replicas are both supported; a conflicting
        # duplicate is an error, not another contribution to the same face.
        unique = {}
        for shard in shards:
            for row in shard:
                key = tuple(row[name] for name in ("operation_identity", "occurrence_identity",
                                                   "evaluation_context", "quadrature_identity"))
                assert key not in unique or unique[key] == row
                unique[key] = row
        assert len(unique) == 32*8*4*4
        selected_identity = make_identity("stencil-flux-v1", authored_flux(corrected=True).to_data()).token
        assert {row["operation_identity"] for row in unique.values()} == {selected_identity}
        assert len({row["evaluation_context"] for row in unique.values()}) == 1
        spec = build_euler_correction(cfl=cfl)
        initial = next(iter(spec.initial_values.values()))
        reference = reference_step(initial, dt=spec.dt, spacing=(1/32, 1/8))
        increments = np.zeros_like(initial)
        face_components = {}
        for row in unique.values():
            cell, axis_token, side_token, component_token = row["quadrature_identity"].split("/")
            i, j = map(int, cell.split(":")[1:])
            axis, side, component = (int(token.split(":")[1])
                                      for token in (axis_token, side_token, component_token))
            assert row["orientation"] == (1 if side == 0 else -1)
            assert row["multiplicity"] == 1 and row["temporal_weight"] == spec.dt
            assert row["face_measure"] == (1/8 if axis == 0 else 1/32)
            fi, fj = ((i+side) % 32, j) if axis == 0 else (i, (j+side) % 8)
            expected = reference["faces"][axis][component, fj, fi]
            assert abs(row["numerical_flux"]-expected) < 2.e-12
            assert abs(row["integrated_amount"]-row["orientation"]*row["face_measure"]
                       *row["numerical_flux"]*spec.dt) < 1.e-17
            increments[component, j, i] += row["integrated_amount"]
            face_components.setdefault((axis, fi, fj, component), []).append(row["integrated_amount"])
        assert len(face_components) == 32*8*2*4
        assert all(len(amounts) == 2 and amounts[0] == -amounts[1]
                   for amounts in face_components.values())
        np.testing.assert_allclose(increments, (reference["corrected"]-initial)/(32*8),
                                   rtol=0, atol=2.e-14)
        record_property("accepted_oriented_component_incidences", len(unique))
        record_property("accepted_face_component_pairs", len(face_components))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("metrics", str(metrics))
