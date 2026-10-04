"""Test-only FieldSolverV2 provider: genuine DefaultExecutionSpace writes, no host relabel."""
from tests.python.integration.native_loader.test_external_field_solver_runtime import _solver_source

_KERNEL = r"""
    using Exec = Kokkos::DefaultExecutionSpace;
    using Memory = Exec::memory_space;
    namespace detail = pops::runtime::field::field_solver_component_detail;
    detail::validate_storage_execution<Memory, Exec>(request->execution);
    if (patch.solution.memory_space != detail::memory_kind<Memory>() ||
        patch.solution.scalar_type != POPS_SCALAR_FLOAT64_V1 || patch.solution.component_count != 1) return 7;
    auto execution = detail::execution_instance<Exec>(request->execution);
    // The request mask is immutable Host topology; own a device copy until its kernel joins.
    using Mask = Kokkos::View<unsigned char*, Memory>;
    Mask owned_mask;
    using HostMask = decltype(Kokkos::create_mirror(Kokkos::HostSpace{}, Mask{}));
    HostMask host_mask;
    // Guard declared after empty owning views, so unwind joins before either allocation dies.
    struct Join { Exec& execution; ~Join() noexcept { try { execution.fence(); } catch (...) {} } } join{execution};
    owned_mask = Mask(Kokkos::view_alloc(execution, "test-field-mask"), patch.material_mask.size);
    host_mask = Kokkos::create_mirror(Kokkos::HostSpace{}, owned_mask);
    for (std::size_t point = 0; point < patch.material_mask.size; ++point) {
      if (mask[point] > 1 || labels[point] != (mask[point] == 1 ? 1 : 0)) return 5;
      host_mask(point) = mask[point];
    }
    Kokkos::deep_copy(execution, owned_mask, host_mask);
    const auto nx = patch.solution.extents[0], ny = patch.solution.extents[1];
    if (nx == 0 || ny == 0 || nx > std::numeric_limits<std::size_t>::max()/ny ||
        patch.material_mask.size != nx*ny) return 8;
    const auto sx = patch.solution.axis_strides[0], sy = patch.solution.axis_strides[1];
    const double answer = TEST_SOLUTION_EXPRESSION;
    Kokkos::parallel_for("test-field-solution", Kokkos::RangePolicy<Exec>(execution, 0, nx*ny),
      KOKKOS_LAMBDA(const std::size_t point) {
        if (owned_mask(point) == 1) {
          const auto i = point % nx, j = point / nx;
          solution[static_cast<std::ptrdiff_t>(i)*sx + static_cast<std::ptrdiff_t>(j)*sy] = answer;
        }
      });
    execution.fence();
"""

def solver_source(manifest, *, solution_expression="7.0", extra_includes="", write_observer_statement="", **kwargs):
    """Preserve real table/manifest validation; replace the CPU direct-write loop only."""
    source = _solver_source(manifest, solution_expression=solution_expression,
        extra_includes="#include <Kokkos_Core.hpp>\n#include <pops/runtime/system/prepared_field_solver_component.hpp>\n" + extra_includes, **kwargs)
    start = source.index("    for (std::size_t j = 0; j < patch.solution.extents[1]; ++j)")
    end = source.index("\n  }\n  report->status", start)
    return source[:start] + _KERNEL.replace("TEST_SOLUTION_EXPRESSION", solution_expression) + write_observer_statement + source[end:]


def fault_source(manifest):
    """Use the actual mapped table; only Host reads control before device launch."""
    from tests.python.support.external_field_fault_component import fault_source as shared_source
    source = shared_source(manifest)
    source = '#include <Kokkos_Core.hpp>\n#include <pops/runtime/system/prepared_field_solver_component.hpp>\n' + source
    start = source.index("    for (std::size_t j = 0; j < patch.solution.extents[1]; ++j)")
    end = source.index("\n  }\n  if (control->mode", start)
    kernel = _KERNEL.replace("TEST_SOLUTION_EXPRESSION", "control->mode == 3 ? std::numeric_limits<double>::quiet_NaN() : 7.0")
    observer = "\n    for (std::size_t point = 0; point < patch.material_mask.size; ++point) if (mask[point] == 1) ++control->writes;\n"
    return source[:start] + kernel + observer + source[end:]


from tests.python.support.external_field_fault_component import SharedTableControl

class GPUSharedTableControl(SharedTableControl):
    """Same true shared table, plus test-only nonfinite mode; no callback-pointer edit."""
    def arm(self, mode):
        import struct
        if type(mode) is not int or mode not in (0, 1, 2, 3):
            raise ValueError("invalid GPU test fault mode")
        super().arm(0 if mode == 3 else mode)
        if mode == 3:
            struct.pack_into("=i", self.memory, self.layout["mode"], mode)
            self.memory.flush()


def component_manifest(name, interface, *, device, parameters=()):
    """Caller must derive device from actual Native report; no runtime capability is invented."""
    from pops.model import ComponentManifest
    if device not in ("cpu", "cuda", "hip"):
        raise ValueError("test FieldSolver supports only actual CPU/CUDA/HIP")
    return ComponentManifest(uri="pops://external.test/fields/%s" % name,
        component_type=interface.name, version="1.0.0", facets=interface.facets,
        signature={"generic": True, "native_interface": interface.signature_declaration()},
        interfaces=interface.manifest_declarations(), parameters=parameters,
        target={"variants": [{"dimension": 2, "scalar": "float64", "device": device, "features": ["mpi"]}]},
        entry_points={"interface_table": "pops_component_interface_v1"})


def component(tmp_path, *, name, interface, source_factory, device,
              manifest_parameters=(), instance_parameters=None):
    """Same genuine source-package route, with one actual-backend manifest."""
    import json
    from pops.external import build_source_package_manifest, load
    root = tmp_path / name
    root.mkdir()
    alias = name.replace("-", "_")
    manifest = component_manifest(name, interface, device=device, parameters=manifest_parameters)
    source = source_factory(manifest).encode()
    source_name = name + ".cpp"
    (root/source_name).write_bytes(source)
    package = build_source_package_manifest(components={alias:manifest},payloads={source_name:("source",source)})
    manifest_path = root/(name+".pops.json")
    manifest_path.write_text(json.dumps(package))
    factory = load(manifest_path).require(alias,interface=interface)
    return factory(**({} if instance_parameters is None else instance_parameters))
