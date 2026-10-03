"""Public Source package declarations; no Native/CUDA execution claimed."""
import pytest
from tests.python.integration.runtime.test_physical_support_mapping import resolve_physical_case


def test_public_physical_provider_declares_exact_backend_variants_and_context(tmp_path):
    plan = resolve_physical_case(tmp_path)[1]
    providers = [row for row in plan.component_inputs
                 if any(b'<pops/runtime/dynamic/physical_support_transfer.hpp>' in p.content
                        for p in row.component_type.package.payloads)]
    assert len(providers) == 2
    for provider in providers:
        manifest = provider.component_manifest
        contract = manifest.signature['physical_backend_dispatch']
        assert contract['contract'] == 'physical-support-backend-dispatch@1'
        assert contract['weights'] == 'invocation-owned-backend-copy'
        assert contract['quadrature'] == 'sequential-per-output-cell'
        for device in ('cpu', 'cuda', 'hip', 'sycl', 'openmptarget'):
            manifest.require_target({'dimension': 2, 'scalar': 'float64', 'device': device, 'features': []})
        with pytest.raises(ValueError):
            manifest.require_target({'dimension': 2, 'scalar': 'float32', 'device': 'cuda', 'features': []})
        with pytest.raises(ValueError):
            manifest.require_target({'dimension': 3, 'scalar': 'float64', 'device': 'cuda', 'features': []})
        source = b'\n'.join(p.content for p in provider.component_type.package.payloads).decode()
        assert 'status, &request->execution)' in source
        assert 'const double weights[]' in source


def test_stream_capability_uses_actual_execution_type_with_other_backend(tmp_path):
    import pathlib, shutil, subprocess
    header = pathlib.Path(__file__).resolve().parents[2] / "include/pops/runtime/dynamic/physical_support_transfer.hpp"
    source = header.read_text()
    trait = source[source.index("template <class ExecutionSpace>\ninline constexpr bool borrows_native_stream"):source.index("/// The native caller authenticates")]
    probe = tmp_path / "stream_traits.cpp"
    probe.write_text("#include <type_traits>\n#define KOKKOS_ENABLE_CUDA\n#define KOKKOS_ENABLE_HIP\nnamespace Kokkos { struct Cuda {}; struct HIP {}; struct SYCL {}; struct OpenMPTarget {}; }\n" + trait + "static_assert(borrows_native_stream<Kokkos::Cuda>);\nstatic_assert(borrows_native_stream<Kokkos::HIP>);\nstatic_assert(!borrows_native_stream<Kokkos::SYCL>);\nstatic_assert(!borrows_native_stream<Kokkos::OpenMPTarget>);\nint main() {}\n")
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None
    subprocess.run([compiler, "-std=c++20", "-fsyntax-only", str(probe)], check=True)
    assert "if constexpr (!borrows_native_stream<Kokkos::DefaultExecutionSpace>)" in source
    # Type-only stand-ins exercise the exact trait with both feature macros, not a GPU runtime.
