"""Actual compiled PoPS sums, full failure payloads and uniform MPI branch decisions."""
import pytest

from examples.api_v02.variant_checks import DIVERGENCE_MARKER, build_variant_case
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("kind", ("inactive_singular", "full_payload", "sum_field_loop", "iteration_payload", "array_variants"))
def test_native_variant_payload(kind, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="v09 Variant " + kind)
    _, artifact, metrics = execute_case(lambda: build_variant_case(kind), world, "v09 Variant " + kind)
    record_property("program_hash", artifact.program_hash)
    record_property("actual_metrics", str(metrics))


def test_native_rank_divergent_variant_is_captured(monkeypatch, isolated_native_cache,
                                                  native_cxx, kokkos_root, record_property):
    """One explicit fault producer: MPI2 must reject before either divergent arm executes."""
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="v09 Variant decision rendezvous")
    ranks = int(world.size) if world is not None else 1
    import pops.codegen.program_emit_structured_control as emitter
    original = emitter.emit_structured_op
    injections = []
    def inject(program, value, var, model, lines, prelude, block_idx, field_plans, target):
        original(program, value, var, model, lines, prelude, block_idx, field_plans, target)
        if (value.op == "variant_construct" and value.inputs[0].op == "int64_constant"
                and value.inputs[0].attrs["value"] == DIVERGENCE_MARKER):
            injections.append(value.id)
            lines.append("%s = ctx.prepared_execution_lane().rank() == 0 ? 0 : 1;" % var[value.id]["tag"])
    monkeypatch.setattr(emitter, "emit_structured_op", inject)
    _, artifact, metrics = execute_case(lambda: build_variant_case("divergent_choice", ranks=ranks),
                                        world, "v09 rank-divergent Variant")
    assert len(set(injections)) == 1, "fault must address one exact Variant producer"
    record_property("program_hash", artifact.program_hash)
    record_property("actual_metrics", str(metrics))
    record_property("mpi_ranks", str(ranks))
    record_property("fault_injection", "one native Variant tag replaced by rank()==0 ? 0 : 1; MPI1 controls baseline, MPI2 requires collective failure capture")
