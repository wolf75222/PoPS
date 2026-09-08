/// Independent active-cover invariants for the explicitly selected restriction/injection pair.
/// This does not claim positivity or adjointness for limited/fifth-order prolongation.
#include <gtest/gtest.h>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace {
namespace transfer = pops::amr::transfer;

template <int Dim, class F>
void visit(const pops::Box<Dim>& box, F&& f) {
  for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
    auto remaining = ordinal;
    pops::Index<Dim> cell{};
    for (int axis = 0; axis < Dim; ++axis) {
      cell[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    f(cell);
  }
}

template <int Dim>
std::size_t offset(const pops::Fab<Dim>& fab, const pops::Index<Dim>& cell, int component) {
  std::size_t value = 0, stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    value += static_cast<std::size_t>(cell[axis] - fab.grown_box().lo[axis]) * stride;
    stride *= static_cast<std::size_t>(fab.grown_box().length(axis));
  }
  return value + static_cast<std::size_t>(component) * stride;
}

template <int Dim>
void prove_selected_pair() {
  using Provider = transfer::TransferProvider<Dim, transfer::Centering::Cell>;
  pops::Index<Dim> lower{}, upper{}, fine_origin{};
  std::array<int, Dim> factors{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -4 + axis;
    upper[axis] = lower[axis] + 3;
    fine_origin[axis] = 7 - 3 * axis;
    factors[axis] = axis == 1 ? 3 : 2;
  }
  const pops::amr::RefinementRatio<Dim> ratio(factors);
  const pops::Box<Dim> coarse_box{lower, upper};
  auto covered = coarse_box;
  covered.lo[0] += 1;
  covered.hi[0] -= 1;
  pops::Box<Dim> fine_box{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_box.lo[axis] = fine_origin[axis] + (covered.lo[axis] - lower[axis]) * ratio[axis];
    fine_box.hi[axis] = fine_origin[axis] + (covered.hi[axis] - lower[axis] + 1) * ratio[axis] - 1;
  }
  const transfer::IndexMapping<Dim> mapping{lower, fine_origin};
  pops::Fab<Dim> coarse(coarse_box, 2), fine(fine_box, 2), restricted(coarse_box, 2);
  pops::Fab<Dim> test_function(coarse_box, 2), injected(fine_box, 2);
  auto ch = coarse.create_host_mirror();
  auto fh = fine.create_host_mirror();
  auto gh = test_function.create_host_mirror();
  auto rh = restricted.create_host_mirror();
  visit(coarse_box, [&](const auto& cell) {
    int code = 0;
    for (int axis = 0; axis < Dim; ++axis)
      code += (axis + 2) * (cell[axis] - lower[axis]);
    for (int c = 0; c < 2; ++c) {
      ch(offset(coarse, cell, c)) = pops::Real(1 + (code + 3 * c) % 7) / 8;
      gh(offset(test_function, cell, c)) = pops::Real((code * code + c) % 11 - 5) / 4;
    }
  });
  visit(fine_box, [&](const auto& cell) {
    int code = 0;
    for (int axis = 0; axis < Dim; ++axis)
      code += (axis + 3) * (cell[axis] - fine_box.lo[axis]);
    for (int c = 0; c < 2; ++c)
      fh(offset(fine, cell, c)) = pops::Real(1 + (code * code + 2 * c) % 7) / 8;
  });
  coarse.copy_from_host(ch);
  fine.copy_from_host(fh);
  test_function.copy_from_host(gh);
  // Host mirrors carry their owning Fab identity even when two Fab layouts match.
  // Seed the restricted field through its own mirror so uncovered values stay unchanged.
  for (std::size_t i = 0; i < ch.size(); ++i)
    rh(i) = ch(i);
  restricted.copy_from_host(rh);
  const transfer::ComponentRange components{0, 0, 2};
  const auto restrict = Provider::conservative_restriction().prepare(
      std::as_const(fine).view(), restricted.view(), covered, ratio, mapping, components);
  const auto inject = Provider::constant_injection().prepare(
      std::as_const(test_function).view(), injected.view(), fine_box, ratio, mapping, components);
  pops::for_each_cell(covered, restrict);
  pops::for_each_cell(fine_box, inject);
  Kokkos::fence();
  auto ih = injected.create_host_mirror();
  restricted.copy_to_host(rh);
  injected.copy_to_host(ih);
  const long double coarse_measure = 0.375L;
  const long double fine_measure = coarse_measure / ratio.child_count();
  for (int c = 0; c < 2; ++c) {
    long double active_mass = 0, collapsed_mass = 0, coarse_work = 0, fine_work = 0;
    visit(coarse_box, [&](const auto& cell) {
      const auto i = offset(coarse, cell, c);
      collapsed_mass += coarse_measure * rh(i);
      if (!covered.contains(cell)) {
        EXPECT_EQ(rh(i), ch(i));
        active_mass += coarse_measure * ch(i);
      } else {
        EXPECT_GE(rh(i), pops::Real(1) / 8);
        EXPECT_LE(rh(i), pops::Real(7) / 8);
        coarse_work += coarse_measure * rh(i) * gh(i);
      }
    });
    visit(fine_box, [&](const auto& cell) {
      const auto i = offset(fine, cell, c);
      active_mass += fine_measure * fh(i);
      fine_work += fine_measure * fh(i) * ih(i);
      pops::Index<Dim> parent{};
      for (int axis = 0; axis < Dim; ++axis)
        parent[axis] = lower[axis] + (cell[axis] - fine_origin[axis]) / ratio[axis];
      EXPECT_EQ(ih(i), gh(offset(test_function, parent, c)));
    });
    const long double tolerance =
        64 * std::numeric_limits<pops::Real>::epsilon() * std::max(1.0L, std::abs(active_mass));
    EXPECT_NEAR(static_cast<double>(collapsed_mass), static_cast<double>(active_mass),
                static_cast<double>(tolerance));
    EXPECT_NEAR(static_cast<double>(coarse_work), static_cast<double>(fine_work),
                static_cast<double>(tolerance));
  }
  // Check interval preservation of nonconstant positive injection separately from the signed
  // test function used in the dual-work identity.
  const auto bounded_inject = Provider::constant_injection().prepare(
      std::as_const(coarse).view(), injected.view(), fine_box, ratio, mapping, components);
  pops::for_each_cell(fine_box, bounded_inject);
  Kokkos::fence();
  injected.copy_to_host(ih);
  visit(fine_box, [&](const auto& cell) {
    for (int c = 0; c < 2; ++c) {
      EXPECT_GE(ih(offset(injected, cell, c)), pops::Real(1) / 8);
      EXPECT_LE(ih(offset(injected, cell, c)), pops::Real(7) / 8);
    }
  });
  // Constants and scalar admissibility are separate oracles, not consequences inferred from mass.
  coarse.set_val(pops::Real(0.375));
  fine.set_val(pops::Real(0.375));
  const auto constant_inject = Provider::constant_injection().prepare(
      std::as_const(coarse).view(), injected.view(), fine_box, ratio, mapping, components);
  pops::for_each_cell(fine_box, constant_inject);
  pops::for_each_cell(covered, restrict);
  Kokkos::fence();
  injected.copy_to_host(ih);
  restricted.copy_to_host(rh);
  visit(fine_box, [&](const auto& cell) {
    for (int c = 0; c < 2; ++c)
      EXPECT_EQ(ih(offset(injected, cell, c)), pops::Real(0.375));
  });
  visit(covered, [&](const auto& cell) {
    for (int c = 0; c < 2; ++c)
      EXPECT_EQ(rh(offset(restricted, cell, c)), pops::Real(0.375));
  });
}
}  // namespace

TEST(test_amr_transition_invariants,
     PartialActiveCoverConservationBoundsConstantsAndSelectedDualWork) {
  prove_selected_pair<pops::kNativeDimension>();
}
