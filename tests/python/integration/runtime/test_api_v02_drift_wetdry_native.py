"""Independent full-state reception of moving wet/dry and authored SG faces."""
import pytest

from examples.api_v02.models_M01_M14 import build_m07, build_m10
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("moving", (False, True))
def test_hydrostatic_face_both_source_deliveries(moving, isolated_native_cache,
                                               native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="M07 moving wet/dry hydrostatic FV")
    _, artifact, rows = execute_case(lambda: build_m07(nonstationary=moving),
                                     world, "M07 independent wet/dry oracle")
    with collective_check(world):
        owners = [row for row in rows if row.get("metrics_owner", True)]
        assert owners
        for row in owners:
            assert row["full_state_reference_error"] < 3.e-11
            assert row["dry_cells"] > 0 and row["minimum_depth"] >= -1.e-12
            assert row["mass_defect"] < 3.e-12 and row["frozen_bed_error"] < 3.e-13
            if moving:
                assert row["no_op_error"] > 1.e-5
                assert min(row["omitted_or_swapped_source_errors"].values()) > 1.e-5
            else:
                assert row["equilibrium_defect"] < 3.e-11
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_metrics", str(rows))


def test_authored_bernoulli_faces_live_poisson(isolated_native_cache, native_cxx,
                                             kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="M10 Python Bernoulli/live Poisson")
    _, artifact, rows = execute_case(build_m10, world, "M10 independent exponential integral")
    with collective_check(world):
        owners = [row for row in rows if row.get("metrics_owner", True)]
        assert owners
        for row in owners:
            assert row["full_state_reference_error"] < 2.e-10
            assert row["stage1_Poisson_residual"] < 2.e-8
            assert min(row["counterexample_errors"].values()) > 1.e-9
        # The artifact used by bind/run must carry the authored expression.
        source = artifact.program._generated_cpp
        assert "apply_fitted" not in source
        assert "scharfetter_gummel_bernoulli" not in source
        assert "expm1(" in source
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_metrics", str(rows))
