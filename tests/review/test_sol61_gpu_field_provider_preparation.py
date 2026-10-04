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
    assert source.index("Kokkos::parallel_for") < source.index("    execution.fence();") < source.index("++control->writes;")
    assert source.index("++control->writes;") < source.index("if (control->mode == 1) throw")
    assert "Kokkos::create_mirror(Kokkos::HostSpace{}, owned_mask)" in source
    assert "control->mode == 3" in source and "MAP_SHARED" in source
    kernel=source[source.index("KOKKOS_LAMBDA"):source.index("    execution.fence();")]
    assert "control" not in kernel
    assert "entry.table = &control->table" in source
    assert "pops_test_field_fault_arm" not in source


def test_fixture_control_is_actual_mapping_and_fault_targets_active_owner():
    from pathlib import Path
    source=Path('tests/python/integration/native_loader/test_gpu_external_field_dispatch_runtime.py').read_text()
    assert 'ctypes' not in source and 'GPUSharedTableControl(control_path)' in source
    assert source.index('monkeypatch.setenv("POPS_TEST_FIELD_CONTROL"') < source.index('pops.bind(')
    assert 'target=max(index for index,value in enumerate(owners) if value)' in source
    assert 'np.all(accepted[2]==0.)' in source and 'np.all(retry_image[2]==0.)' in source
