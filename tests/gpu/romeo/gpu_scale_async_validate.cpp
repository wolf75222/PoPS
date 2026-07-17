// Targeted device-ordering gate for the MultiFab arithmetic seam.
//
// The two for_each_cell launches intentionally have no fence between them and
// pops::scale().  The final blocking reduction is the first operation allowed
// to wait for the stream.  A stale host-side scale implementation (or an
// accidental host loop after an asynchronous launch) therefore fails this
// check on a non-synchronising CUDA execution space.

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>

namespace {

struct FillKernel {
  pops::Array4 values;
  pops::Real value;

  POPS_HD void operator()(int i, int j) const { values(i, j, 0) = value; }
};

struct AddKernel {
  pops::Array4 values;
  pops::Real value;

  POPS_HD void operator()(int i, int j) const { values(i, j, 0) += value; }
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    // 2048^2 valid cells, tiled so the check also exercises the MultiFab loop
    // over multiple asynchronous launches rather than a single small box.
    constexpr int n = 2048;
    constexpr int tile = 128;
    constexpr pops::Real fill = 1.0;
    constexpr pops::Real add = 3.0;
    constexpr pops::Real factor = 2.0;
    constexpr pops::Real expected_value = (fill + add) * factor;

    const pops::Box2D domain = pops::Box2D::from_extents(n, n);
    const pops::BoxArray boxes = pops::BoxArray::from_domain(domain, tile);
    const pops::DistributionMapping mapping(boxes.size(), pops::n_ranks());
    pops::MultiFab field(boxes, mapping, 1, 0);

    for (int li = 0; li < field.local_size(); ++li) {
      const pops::Box2D box = field.box(li);
      const pops::Array4 values = field.fab(li).array();
      pops::for_each_cell(box, FillKernel{values, fill});
      // Deliberately submit immediately after FillKernel: no fence or host
      // access is permitted between the two kernels.
      pops::for_each_cell(box, AddKernel{values, add});
    }

    // This must remain a device kernel.  The following blocking reduction is
    // what makes the result safe to inspect on the host.
    pops::scale(field, factor);
    const pops::Real observed_sum = pops::reduce_sum(field);
    const pops::Real expected_sum = expected_value * static_cast<pops::Real>(n) * n;

    const bool exact_sum = observed_sum == expected_sum;
    const bool finite_sum = std::isfinite(static_cast<double>(observed_sum));
    std::printf("[scale-order] exec=%s boxes=%d cells=%d observed=%.17g expected=%.17g\n",
                Kokkos::DefaultExecutionSpace::name(), boxes.size(), n * n,
                static_cast<double>(observed_sum), static_cast<double>(expected_sum));
    if (!finite_sum || !exact_sum) {
      std::printf("FAIL gpu_scale_async_validate exact reduction\n");
      rc = 1;
    } else {
      std::printf("OK gpu_scale_async_validate\n");
    }
  }
  Kokkos::finalize();
  return rc;
}
