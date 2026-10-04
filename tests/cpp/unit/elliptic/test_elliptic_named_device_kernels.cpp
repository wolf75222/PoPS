#include <gtest/gtest.h>

#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
using pops::Real;

// Actual DefaultExecutionSpace allocation; mirrors alone are accessed on the host.
// Deliberately distinct strides and component gaps catch dense/valid-box assumptions.
template <int Dim>
struct PaddedField {
  Kokkos::View<Real*> storage;
  pops::FieldView<Real, Dim> view;
  pops::Box<Dim> box;
  static constexpr std::size_t size = 4000;

  PaddedField(bool destination, int components = 3) {
    pops::detail::ensure_kokkos_initialized();
    storage = Kokkos::View<Real*>("elliptic_named_kernel_test", size);
    view.data = storage.data();
    const int origins[]{-4, 3, 11};
    const int lengths[]{7, 5, 4};
    const std::int64_t source_strides[]{2, 19, 113};
    const std::int64_t destination_strides[]{3, 29, 157};
    for (int axis = 0; axis < Dim; ++axis) {
      box.lo[axis] = view.origin[axis] = origins[axis];
      box.hi[axis] = origins[axis] + lengths[axis] - 1;
      view.extents[axis] = lengths[axis];
      view.strides[axis] = destination ? destination_strides[axis] : source_strides[axis];
    }
    view.ncomp = components;
    view.component_stride = destination ? 1300 : 1000;
  }

  std::size_t offset(const pops::Index<Dim>& cell, int component = 0) const {
    std::size_t result = component * view.component_stride;
    for (int axis = 0; axis < Dim; ++axis)
      result += (cell[axis] - view.origin[axis]) * view.strides[axis];
    return result;
  }
};

template <int Dim>
pops::FieldView<const Real, Dim> constant_view(const PaddedField<Dim>& field) {
  pops::FieldView<const Real, Dim> result{};
  result.data = field.view.data;
  result.origin = field.view.origin;
  result.extents = field.view.extents;
  for (int axis = 0; axis < Dim; ++axis)
    result.strides[axis] = field.view.strides[axis];
  result.ncomp = field.view.ncomp;
  result.component_stride = field.view.component_stride;
  return result;
}

template <int Dim, class Function>
void host_cells(const pops::Box<Dim>& box, Function function) {
  for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
    auto remaining = ordinal;
    pops::Index<Dim> cell{};
    for (int axis = 0; axis < Dim; ++axis) {
      cell[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    function(cell, ordinal);
  }
}

template <int Dim>
pops::Box<Dim> interior(const pops::Box<Dim>& box) {
  auto result = box;
  for (int axis = 0; axis < Dim; ++axis) {
    ++result.lo[axis];
    --result.hi[axis];
  }
  return result;
}

template <int Dim>
void copies_preserve_outside_ranges_and_components() {
  PaddedField<Dim> source(false), destination(true);
  auto input = Kokkos::create_mirror_view(source.storage);
  auto output = Kokkos::create_mirror_view(destination.storage);
  for (std::size_t index = 0; index < source.size; ++index) {
    input(index) = Real(index) / Real(16) - Real(7);
    output(index) = Real(-91);
  }
  Kokkos::deep_copy(source.storage, input);
  Kokkos::deep_copy(destination.storage, output);
  const auto region = interior(source.box);
  auto expected = std::vector<Real>(destination.size, Real(-91));
  host_cells(region, [&](const auto& cell, auto) {
    for (int component = 0; component < 3; ++component)
      expected[destination.offset(cell, component)] = input(source.offset(cell, component));
  });
  pops::for_each_cell(region, pops::elliptic::mg::detail::CopyVectorKernel<Dim>{
                                  destination.view, constant_view(source), 3});
  Kokkos::fence();
  Kokkos::deep_copy(output, destination.storage);
  for (std::size_t index = 0; index < expected.size(); ++index)
    ASSERT_EQ(output(index), expected[index]) << "storage offset " << index;

  // Scalar copies are also used on grown boxes and gather intersections. They
  // must copy the requested range, and leave every other component/padding byte.
  host_cells(source.box, [&](const auto& cell, auto) {
    expected[destination.offset(cell)] = input(source.offset(cell));
  });
  pops::for_each_cell(source.box, pops::elliptic::mg::detail::CopyScalarKernel<Dim>{
                                      destination.view, constant_view(source)});
  Kokkos::fence();
  Kokkos::deep_copy(output, destination.storage);
  for (std::size_t index = 0; index < expected.size(); ++index)
    ASSERT_EQ(output(index), expected[index]) << "grown storage offset " << index;
}

template <int Dim>
void projection_preserves_mask_and_active_storage() {
  PaddedField<Dim> values(true), active(false, 1);
  auto data = Kokkos::create_mirror_view(values.storage);
  auto mask = Kokkos::create_mirror_view(active.storage);
  const Real masks[]{Real(-1), Real(0), Real(0.49), Real(0.5), Real(1),
                     std::numeric_limits<Real>::quiet_NaN()};
  for (std::size_t index = 0; index < values.size; ++index) {
    data(index) = Real(index) / Real(8) + Real(2);
    mask(index) = Real(-73);
  }
  host_cells(active.box, [&](const auto& cell, auto ordinal) {
    mask(active.offset(cell)) = masks[ordinal % 6];
  });
  auto expected = std::vector<Real>(data.data(), data.data() + values.size);
  const auto original_mask = std::vector<Real>(mask.data(), mask.data() + active.size);
  const auto region = interior(values.box);
  host_cells(region, [&](const auto& cell, auto) {
    if (!(mask(active.offset(cell)) >= Real(0.5)))
      for (int component = 0; component < 3; ++component)
        expected[values.offset(cell, component)] = Real(0);
  });
  Kokkos::deep_copy(values.storage, data);
  Kokkos::deep_copy(active.storage, mask);
  pops::for_each_cell(region, pops::amr_newton_detail::ProjectUnknownsKernel<Dim>{
                                  values.view, constant_view(active), 3});
  Kokkos::fence();
  Kokkos::deep_copy(data, values.storage);
  for (std::size_t index = 0; index < expected.size(); ++index)
    ASSERT_EQ(data(index), expected[index]) << "projection storage offset " << index;
  auto after_mask = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, active.storage);
  for (std::size_t index = 0; index < active.size; ++index)
    ASSERT_EQ(std::bit_cast<std::uint64_t>(after_mask(index)),
              std::bit_cast<std::uint64_t>(original_mask[index]));
}

template <int Dim>
void slab_pack_unpack_keeps_slab_order_and_views() {
  PaddedField<Dim> source(false), destination(true);
  auto input = Kokkos::create_mirror_view(source.storage);
  auto output = Kokkos::create_mirror_view(destination.storage);
  for (std::size_t index = 0; index < source.size; ++index) {
    input(index) = Real(index) / Real(16) - Real(5);
    output(index) = Real(-47);
  }
  Kokkos::deep_copy(source.storage, input);
  Kokkos::deep_copy(destination.storage, output);
  using Complex = typename pops::PoissonFFT<Dim>::complex_type;
  typename pops::PoissonFFT<Dim>::device_view slab("named_slab", source.box.numPts());
  auto packed = Kokkos::create_mirror_view(slab);
  for (std::size_t index = 0; index < packed.extent(0); ++index)
    packed(index) = Complex(13, 19);
  Kokkos::deep_copy(slab, packed);
  const auto region = interior(source.box);
  const Real sign = Real(-0.125);
  pops::for_each_cell(region, pops::elliptic::fft_multifab_detail::PackSlabKernel<Dim>{
                                  slab, constant_view(source), source.box, sign});
  Kokkos::fence();
  Kokkos::deep_copy(packed, slab);
  host_cells(source.box, [&](const auto& cell, auto ordinal) {
    const auto expected = region.contains(cell)
                              ? Complex(static_cast<double>(sign * input(source.offset(cell))), 0)
                              : Complex(13, 19);
    ASSERT_EQ(packed(ordinal).real(), expected.real());
    ASSERT_EQ(packed(ordinal).imag(), expected.imag());
    packed(ordinal) = Complex(static_cast<double>(ordinal) / 4.0 - 2.0, 31.0);
  });
  Kokkos::deep_copy(slab, packed);
  auto expected = std::vector<Real>(destination.size, Real(-47));
  host_cells(region, [&](const auto& cell, auto ordinal_in_region) {
    (void)ordinal_in_region;
    std::int64_t ordinal = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      ordinal += (cell[axis] - source.box.lo[axis]) * stride;
      stride *= source.box.length(axis);
    }
    expected[destination.offset(cell)] = static_cast<Real>(packed(ordinal).real());
  });
  pops::for_each_cell(region, pops::elliptic::fft_multifab_detail::UnpackSlabKernel<Dim>{
                                  slab, destination.view, source.box});
  Kokkos::fence();
  Kokkos::deep_copy(output, destination.storage);
  for (std::size_t index = 0; index < expected.size(); ++index)
    ASSERT_EQ(output(index), expected[index]) << "unpack storage offset " << index;
}

// The public FFT solver's sign conversion differs from the configurable
// MultiFab adapter: preserve unary negation (including IEEE signs) exactly.
template <int Dim>
void solver_slab_transfers_preserve_exact_bits_and_padding() {
  for (const bool special_values : {false, true}) {
    constexpr std::size_t storage_size = 5200;
    PaddedField<Dim> source(false, 4), destination(true, 4);
    source.storage = Kokkos::View<Real*>("solver_source_four_components", storage_size);
    destination.storage = Kokkos::View<Real*>("solver_destination_four_components", storage_size);
    source.view.data = source.storage.data();
    destination.view.data = destination.storage.data();
    auto input = Kokkos::create_mirror_view(source.storage);
    auto output = Kokkos::create_mirror_view(destination.storage);
    for (std::size_t offset = 0; offset < storage_size; ++offset) {
      input(offset) = Real(offset) / Real(16) - Real(9);
      output(offset) = Real(-83);
    }
    const Real exceptional[]{Real(0), -Real(0), std::numeric_limits<Real>::infinity(),
                             -std::numeric_limits<Real>::infinity(),
                             std::bit_cast<Real>(std::uint64_t{0x7ff8000000000042})};
    const auto region = interior(source.box);
    host_cells(region, [&](const auto& cell, auto ordinal) {
      if (special_values && ordinal < 5)
        input(source.offset(cell)) = exceptional[ordinal];
    });
    const auto original = std::vector<Real>(input.data(), input.data() + storage_size);
    Kokkos::deep_copy(source.storage, input);
    Kokkos::deep_copy(destination.storage, output);
    using Complex = typename pops::PoissonFFT<Dim>::complex_type;
    typename pops::PoissonFFT<Dim>::device_view slab("solver_slab", source.box.numPts());
    auto packed = Kokkos::create_mirror_view(slab);
    for (std::size_t ordinal = 0; ordinal < packed.extent(0); ++ordinal)
      packed(ordinal) = Complex(23, -29);
    Kokkos::deep_copy(slab, packed);
    pops::for_each_cell(
        region, pops::fft_solver_detail::PackRhsSlabKernel<Dim>{slab, source.view, source.box});
    Kokkos::fence();
    Kokkos::deep_copy(packed, slab);
    host_cells(source.box, [&](const auto& cell, auto ordinal) {
      const auto expected = region.contains(cell) ? Complex(-original[source.offset(cell)], Real(0))
                                                  : Complex(23, -29);
      ASSERT_EQ(std::bit_cast<std::uint64_t>(packed(ordinal).real()),
                std::bit_cast<std::uint64_t>(expected.real()));
      ASSERT_EQ(std::bit_cast<std::uint64_t>(packed(ordinal).imag()),
                std::bit_cast<std::uint64_t>(expected.imag()));
    });
    auto expected = std::vector<Real>(storage_size, Real(-83));
    host_cells(region, [&](const auto& cell, auto) {
      std::size_t ordinal = 0, stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        ordinal += (cell[axis] - source.box.lo[axis]) * stride;
        stride *= source.box.length(axis);
      }
      expected[destination.offset(cell)] = packed(ordinal).real();
    });
    pops::for_each_cell(region, pops::fft_solver_detail::UnpackSolutionSlabKernel<Dim>{
                                    slab, destination.view, source.box});
    Kokkos::fence();
    Kokkos::deep_copy(output, destination.storage);
    for (std::size_t offset = 0; offset < storage_size; ++offset)
      ASSERT_EQ(std::bit_cast<std::uint64_t>(output(offset)),
                std::bit_cast<std::uint64_t>(expected[offset]))
          << "solver offset " << offset;
    auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source.storage);
    for (std::size_t offset = 0; offset < storage_size; ++offset)
      ASSERT_EQ(std::bit_cast<std::uint64_t>(after(offset)),
                std::bit_cast<std::uint64_t>(original[offset]));
  }
}

TEST(EllipticNamedDeviceKernels, SolverSlabDim1) {
  solver_slab_transfers_preserve_exact_bits_and_padding<1>();
}
TEST(EllipticNamedDeviceKernels, SolverSlabDim2) {
  solver_slab_transfers_preserve_exact_bits_and_padding<2>();
}
TEST(EllipticNamedDeviceKernels, SolverSlabDim3) {
  solver_slab_transfers_preserve_exact_bits_and_padding<3>();
}

TEST(EllipticNamedDeviceKernels, CopiesDim1) { copies_preserve_outside_ranges_and_components<1>(); }
TEST(EllipticNamedDeviceKernels, CopiesDim2) { copies_preserve_outside_ranges_and_components<2>(); }
TEST(EllipticNamedDeviceKernels, CopiesDim3) { copies_preserve_outside_ranges_and_components<3>(); }
TEST(EllipticNamedDeviceKernels, ProjectionDim1) { projection_preserves_mask_and_active_storage<1>(); }
TEST(EllipticNamedDeviceKernels, ProjectionDim2) { projection_preserves_mask_and_active_storage<2>(); }
TEST(EllipticNamedDeviceKernels, ProjectionDim3) { projection_preserves_mask_and_active_storage<3>(); }
TEST(EllipticNamedDeviceKernels, SlabDim1) { slab_pack_unpack_keeps_slab_order_and_views<1>(); }
TEST(EllipticNamedDeviceKernels, SlabDim2) { slab_pack_unpack_keeps_slab_order_and_views<2>(); }
TEST(EllipticNamedDeviceKernels, SlabDim3) { slab_pack_unpack_keeps_slab_order_and_views<3>(); }
}  // namespace
