"""Source preparation only: generated true Kokkos provider, no mocked GPU execution."""
import pytest
from pops import interfaces
from tests.python.integration.native_loader.test_external_field_solver_runtime import _manifest
from tests.python.support.gpu_field_solver_test_component import solver_source

@pytest.mark.parametrize("expression", ["7.0", "std::numeric_limits<double>::quiet_NaN()"])
def test_generated_provider_owns_mask_and_joins_actual_queue(expression):
    manifest = _manifest("owned-device-field", interfaces.FieldSolver)
    source = solver_source(manifest, solution_expression=expression,
                           write_observer_statement="\n    observed_writes += patch.material_mask.size;")
    assert manifest.component_id in source
    assert "validate_storage_execution<Memory, Exec>" in source
    assert "patch.solution.memory_space != detail::memory_kind<Memory>()" in source
    assert "Kokkos::deep_copy(execution, owned_mask, host_mask)" in source
    assert "KOKKOS_LAMBDA" in source and expression in source
    assert source.index("execution.fence();") < source.index("observed_writes +=")
    assert "solution[index]" not in source

def test_cpu_callback_observer_and_extra_headers_are_preserved():
    source = solver_source(_manifest("observer-test", interfaces.FieldSolver),
       extra_includes="#include <stdexcept>", solve_observer_statement="/* callback-entry */")
    assert "#include <stdexcept>" in source and "/* callback-entry */" in source

def test_declared_backend_manifest_is_exact_and_not_a_capability_override():
    from tests.python.support.gpu_field_solver_test_component import component_manifest
    cpu = component_manifest("backend-target", interfaces.FieldSolver, device="cpu")
    cuda = component_manifest("backend-target", interfaces.FieldSolver, device="cuda")
    assert cpu.manifest_digest != cuda.manifest_digest
    assert cuda.target["variants"][0]["device"] == "cuda"
    for wrong in (True, "gpu", "sycl", None):
        with pytest.raises(ValueError):
            component_manifest("backend-target", interfaces.FieldSolver, device=wrong)

def test_completed_kernel_faults_preserve_queue_and_owning_mask_order():
    from tests.python.support.gpu_field_solver_test_component import fault_source
    source = fault_source(_manifest("fault-order", interfaces.FieldSolver))
    assert source.index("Mask owned_mask;") < source.index("join{execution}") < source.index("owned_mask = Mask")
    assert source.index("Kokkos::parallel_for") < source.index("    execution.fence();") < source.index("++observed_writes;")
    assert source.index("++observed_writes;") < source.index("if (fault_mode == 1) throw")
    assert "Kokkos::create_mirror(Kokkos::HostSpace{}, owned_mask)" in source
    assert "fault_mode == 3" in source and "if (mode < 0 || mode > 3)" in source
