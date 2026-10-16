"""Native consumer of closed Python representation equations; no reference runtime."""
import pytest

from examples.api_v02.representation_checks import build_representation_check, build_scientific_alias_check
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("scales", ((2., 3., 5.), (3.5, 4., .75)))
def test_native_representation_maps(scales, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="native finite representation maps")
    _runtime, artifact, gathered = execute_case(
        lambda: build_representation_check(encode_scale=scales[0], decode_scale=scales[1],
                                           pairing_scale=scales[2]), world,
        "api_v02_representation_%s" % ("_".join(map(str, scales)),))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("metrics", gathered)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("aliased", (True, False), ids=("shared_ssa", "distinct_ssa"))
def test_native_scientific_map_argument_aliases(aliased, isolated_native_cache, native_cxx,
                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="native strict scientific map argument aliases")
    _runtime, artifact, gathered = execute_case(
        lambda: build_scientific_alias_check(aliased=aliased), world,
        "api_v02_scientific_alias_%s" % ("shared" if aliased else "distinct"))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("metrics", gathered)
