/// @file
/// @brief Refined exact-ranked composite Poisson fidelity in 1D, 2D, and 3D.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

template <int Dim>
pops::Index<Dim> filled_index(int value) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Extent<Dim> filled_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
std::array<int, Dim> filled_ratio(int value) {
  std::array<int, Dim> result{};
  result.fill(value);
  return result;
}

template <int Dim>
std::size_t ordinal(const pops::Box<Dim>& box, const pops::Index<Dim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim, class Function>
void visit(const pops::Box<Dim>& box, Function&& function) {
  for (std::int64_t linear = 0; linear < box.numPts(); ++linear) {
    std::int64_t remainder = linear;
    pops::Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = box.lo[axis] + static_cast<int>(remainder % box.length(axis));
      remainder /= box.length(axis);
    }
    function(index);
  }
}

template <int Dim>
pops::Real exact_mode(const pops::Geometry<Dim>& geometry, const pops::Index<Dim>& index) {
  pops::Real result = pops::Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    result *= Kokkos::sin(kPi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

template <int Dim>
pops::Real exact_gradient(const pops::Geometry<Dim>& geometry, const pops::Index<Dim>& index,
                          int derivative_axis) {
  pops::Real result = kPi;
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real coordinate = geometry.cell_coordinate(axis, index[axis]);
    result *=
        axis == derivative_axis ? Kokkos::cos(kPi * coordinate) : Kokkos::sin(kPi * coordinate);
  }
  return result;
}

template <int Dim>
pops::EllipticBuildRequest<Dim> level_request(const pops::Geometry<Dim>& geometry,
                                              pops::mesh::BoxArray<Dim> boxes) {
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(boxes, ranks);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          std::move(boxes),
          distribution,
          pops::Index<Dim>{},
          {pops::BoundaryTopology<Dim>::physical(), faces, spacing},
          pops::Extent<Dim>{},
          filled_extent<Dim>(1),
          {1, 0}};
}

template <int Dim>
void fill_manufactured_rhs(pops::MultiFab<Dim>& rhs, const pops::Geometry<Dim>& geometry) {
  const pops::Real eigenvalue = static_cast<pops::Real>(Dim) * kPi * kPi;
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    visit(fab.box(), [&](const pops::Index<Dim>& index) {
      host(ordinal(fab.grown_box(), index)) = eigenvalue * exact_mode(geometry, index);
    });
    fab.copy_from_host(host);
  }
}

template <int Dim>
struct HostPatch {
  const pops::Fab<Dim>& fab;
  typename pops::Fab<Dim>::host_mirror_type values;

  explicit HostPatch(const pops::Fab<Dim>& source)
      : fab(source), values(source.create_host_mirror()) {
    source.copy_to_host(values);
  }

  pops::Real operator()(const pops::Index<Dim>& index) const {
    return values(ordinal(fab.grown_box(), index));
  }
};

template <int Dim>
void prove_refined_gradient_is_more_accurate_than_coarse_injection() {
  constexpr int cells = 12;
  const pops::Box<Dim> coarse_domain{filled_index<Dim>(0), filled_index<Dim>(cells - 1)};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = pops::Real(1);
  const pops::Geometry<Dim> coarse_geometry =
      pops::Geometry<Dim>::from_bounds(coarse_domain, pops::RealVector<Dim>{}, upper);
  const pops::Geometry<Dim> fine_geometry = coarse_geometry.refine(filled_extent<Dim>(2));

  const pops::Box<Dim> parent_patch{filled_index<Dim>(3), filled_index<Dim>(8)};
  const pops::Box<Dim> fine_patch = pops::refine(parent_patch, filled_extent<Dim>(2));
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> build{
      {level_request<Dim>(coarse_geometry,
                          pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{coarse_domain}}),
       level_request<Dim>(fine_geometry,
                          pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{fine_patch}})},
      {pops::amr::RefinementRatio<Dim>{filled_ratio<Dim>(2)}}};

  pops::CompositeFacOptions options;
  options.max_iters = 80;
  options.fine_sweeps = 6;
  options.rel_tol = pops::Real(1e-10);
  options.abs_tol = pops::Real(1e-12);
  options.coarse_rel_tol = pops::Real(1e-11);
  options.coarse_abs_tol = pops::Real(1e-13);
  options.coarse_cycles = 192;
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(build), options);
  solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                           {pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated()});
  fill_manufactured_rhs(solver.rhs_level(0), coarse_geometry);
  fill_manufactured_rhs(solver.rhs_level(1), fine_geometry);
  solver.phi_level(0).set_val(pops::Real(0));
  solver.phi_level(1).set_val(pops::Real(0));

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  ASSERT_EQ(solver.n_levels(), 2);
  ASSERT_EQ(solver.phi_level(0).local_size(), 1U);
  ASSERT_EQ(solver.phi_level(1).local_size(), 1U);

  const HostPatch<Dim> coarse(solver.phi_level(0).fab(0));
  const HostPatch<Dim> fine(solver.phi_level(1).fab(0));
  const pops::Box<Dim> sample = fine_patch.grow(-2);
  ASSERT_FALSE(sample.empty());
  pops::Real coarse_error = pops::Real(0);
  pops::Real fine_error = pops::Real(0);
  visit(sample, [&](const pops::Index<Dim>& fine_cell) {
    pops::Index<Dim> coarse_cell{};
    for (int axis = 0; axis < Dim; ++axis)
      coarse_cell[axis] = fine_cell[axis] / 2;
    for (int axis = 0; axis < Dim; ++axis) {
      pops::Index<Dim> fine_lower = fine_cell;
      pops::Index<Dim> fine_upper = fine_cell;
      pops::Index<Dim> coarse_lower = coarse_cell;
      pops::Index<Dim> coarse_upper = coarse_cell;
      --fine_lower[axis];
      ++fine_upper[axis];
      --coarse_lower[axis];
      ++coarse_upper[axis];
      const pops::Real expected = exact_gradient(fine_geometry, fine_cell, axis);
      const pops::Real fine_gradient =
          (fine(fine_upper) - fine(fine_lower)) / (pops::Real(2) * fine_geometry.spacing(axis));
      const pops::Real injected_coarse_gradient = (coarse(coarse_upper) - coarse(coarse_lower)) /
                                                  (pops::Real(2) * coarse_geometry.spacing(axis));
      fine_error = std::max(fine_error, Kokkos::abs(fine_gradient - expected));
      coarse_error = std::max(coarse_error, Kokkos::abs(injected_coarse_gradient - expected));
    }
  });
  EXPECT_TRUE(std::isfinite(static_cast<double>(fine_error)) &&
              std::isfinite(static_cast<double>(coarse_error)));
  EXPECT_LT(fine_error, pops::Real(0.75) * coarse_error)
      << "the refined composite potential must improve the gradient over coarse injection";
}

}  // namespace

TEST(test_amr_composite_poisson, RefinedGradientImprovesInOneTwoAndThreeDimensions) {
  Kokkos::ScopeGuard kokkos;
  prove_refined_gradient_is_more_accurate_than_coarse_injection<1>();
  prove_refined_gradient_is_more_accurate_than_coarse_injection<2>();
  prove_refined_gradient_is_more_accurate_than_coarse_injection<3>();
}
