#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/elliptic/amr/composite_fac_poisson.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr pops::Real kPi = pops::Real{3.141592653589793238462643383279502884L};

template <int Dim>
pops::Extent<Dim> uniform_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::RealVector<Dim> uniform_coordinates(pops::Real value) {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> rank_coordinate(int rank) {
  pops::Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
  pops::Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return index;
}

template <int Dim>
std::size_t field_offset(const pops::Box<Dim>& grown, const pops::Index<Dim>& index) {
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return offset;
}

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> constant_dirichlet(const pops::Geometry<Dim>& geometry,
                                                         pops::Real value) {
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill(pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, value});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {pops::BoundaryTopology<Dim>::physical(), faces, spacing};
}

pops::elliptic::amr::CompositeFacPreparationBudget fac_budget() {
  pops::elliptic::amr::CompositeFacPreparationBudget result;
  result.levels = 3;
  result.connections = 2;
  result.parent_child_patch_pairs = 1'000'000;
  result.interpolation_regions = 1'000'000;
  result.local_scratch_cells = 10'000'000;
  result.same_level_halo = {pops::mesh::BoxArrayValidationBudget{1024, 1'000'000},
                            1'000'000,
                            2'000'000,
                            1024,
                            1024,
                            10'000'000,
                            10'000'000,
                            10'000'000};
  result.parent_gather = {1'000'000, 1024, 10'000'000, 10'000'000, 10'000'000};
  result.fine_restriction = {1'000'000, 1024, 10'000'000, 10'000'000, 10'000'000};
  return result;
}

template <int Dim>
pops::mesh::Distribution<Dim> shifted_distribution(const pops::mesh::BoxArray<Dim>& layout,
                                                   const pops::mesh::RankSpace<Dim>& rank_space,
                                                   int level, int rank_count) {
  std::vector<pops::Index<Dim>> owners;
  owners.reserve(layout.size());
  for (std::size_t patch = 0; patch < layout.size(); ++patch)
    owners.push_back(rank_coordinate<Dim>(
        static_cast<int>((patch + static_cast<std::size_t>(level)) % rank_count)));
  return pops::mesh::Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
}

template <int Dim>
pops::elliptic::amr::CompositeFacBuildRequest<Dim> make_fac_request(int level_count, int rank_count,
                                                                    int rank,
                                                                    pops::Real boundary_value) {
  pops::Index<Dim> coarse_upper{};
  for (int axis = 0; axis < Dim; ++axis)
    coarse_upper[axis] = (axis == 0 ? 32 : 8) - 1;
  const pops::Box<Dim> coarse_domain{pops::Index<Dim>{}, coarse_upper};
  const pops::Geometry<Dim> coarse_geometry =
      pops::Geometry<Dim>::from_bounds(coarse_domain, uniform_coordinates<Dim>(pops::Real{0}),
                                       uniform_coordinates<Dim>(pops::Real{1}));

  auto rank_extent = uniform_extent<Dim>(1);
  rank_extent[0] = rank_count;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  const auto local_rank = rank_coordinate<Dim>(rank);
  const pops::mesh::BoxArrayValidationBudget layout_budget{1024, 1'000'000};
  std::array<int, Dim> ratio_values{};
  ratio_values.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio{ratio_values};

  std::vector<pops::EllipticBuildRequest<Dim>> levels;
  std::vector<pops::amr::RefinementRatio<Dim>> ratios;
  levels.reserve(static_cast<std::size_t>(level_count));
  ratios.reserve(static_cast<std::size_t>(level_count - 1));

  auto coarse_grid = uniform_extent<Dim>(4);
  const auto coarse_layout = pops::mesh::BoxArray<Dim>::from_domain(coarse_domain, coarse_grid);
  levels.push_back({coarse_geometry, coarse_layout,
                    shifted_distribution(coarse_layout, rank_space, 0, rank_count), local_rank,
                    constant_dirichlet(coarse_geometry, boundary_value), pops::Extent<Dim>{},
                    uniform_extent<Dim>(1), layout_budget});

  pops::Geometry<Dim> geometry = coarse_geometry.refine(uniform_extent<Dim>(2));
  pops::Index<Dim> fine_lower{};
  pops::Index<Dim> fine_upper = geometry.domain().hi;
  fine_lower[0] = 16;
  fine_upper[0] = 47;
  for (int axis = 1; axis < Dim; ++axis) {
    fine_lower[axis] = 4;
    fine_upper[axis] = 11;
  }
  auto fine_grid = uniform_extent<Dim>(8);
  fine_grid[0] = 4;
  const auto fine_layout =
      pops::mesh::BoxArray<Dim>::from_domain(pops::Box<Dim>{fine_lower, fine_upper}, fine_grid);
  levels.push_back({geometry, fine_layout,
                    shifted_distribution(fine_layout, rank_space, 1, rank_count), local_rank,
                    constant_dirichlet(geometry, boundary_value), pops::Extent<Dim>{},
                    uniform_extent<Dim>(1), layout_budget});
  ratios.push_back(ratio);

  if (level_count == 3) {
    geometry = geometry.refine(uniform_extent<Dim>(2));
    pops::Index<Dim> finer_lower{};
    pops::Index<Dim> finer_upper = geometry.domain().hi;
    finer_lower[0] = 48;
    finer_upper[0] = 79;
    for (int axis = 1; axis < Dim; ++axis) {
      finer_lower[axis] = 12;
      finer_upper[axis] = 19;
    }
    auto finer_grid = uniform_extent<Dim>(8);
    finer_grid[0] = 4;
    const auto finer_layout = pops::mesh::BoxArray<Dim>::from_domain(
        pops::Box<Dim>{finer_lower, finer_upper}, finer_grid);
    levels.push_back({geometry, finer_layout,
                      shifted_distribution(finer_layout, rank_space, 2, rank_count), local_rank,
                      constant_dirichlet(geometry, boundary_value), pops::Extent<Dim>{},
                      uniform_extent<Dim>(1), layout_budget});
    ratios.push_back(ratio);
  }

  return {std::move(levels), std::move(ratios), fac_budget()};
}

template <int Dim>
pops::Real maximum_constant_error(const pops::MultiFab<Dim>& field) {
  pops::Real local_error = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const std::size_t cells = static_cast<std::size_t>(fab.box().numPts());
    for (std::size_t cell = 0; cell < cells; ++cell) {
      const auto index = index_from_ordinal(fab.box(), cell);
      local_error = std::max(local_error,
                             std::abs(host(field_offset(fab.grown_box(), index)) - pops::Real{1}));
    }
  }
  return static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_error)));
}

template <int Dim>
pops::Real manufactured_mode(const pops::Geometry<Dim>& geometry, const pops::Index<Dim>& index) {
  pops::Real result = pops::Real{1};
  for (int axis = 0; axis < Dim; ++axis)
    result *= std::sin(kPi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

template <int Dim>
pops::Real screened_discrete_eigenvalue(const pops::Geometry<Dim>& geometry) {
  pops::Real result = pops::Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real angle =
        kPi / (pops::Real{2} * static_cast<pops::Real>(geometry.domain().length(axis)));
    const pops::Real inverse_spacing = pops::Real{1} / geometry.spacing(axis);
    result += pops::Real{4} * std::sin(angle) * std::sin(angle) * inverse_spacing * inverse_spacing;
  }
  return result;
}

template <int Dim>
void fill_manufactured_rhs(pops::MultiFab<Dim>& rhs, const pops::Geometry<Dim>& geometry) {
  const pops::Real eigenvalue = screened_discrete_eigenvalue(geometry);
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      host(field_offset(fab.grown_box(), index)) = eigenvalue * manufactured_mode(geometry, index);
    }
    fab.copy_from_host(host);
  }
}

struct ManufacturedDiagnostics {
  pops::Real maximum_error = 0;
  double checksum = 0;
  double exact_checksum = 0;
  double checksum_spread = 0;
};

template <int Dim>
ManufacturedDiagnostics manufactured_diagnostics(const pops::MultiFab<Dim>& field,
                                                 const pops::Geometry<Dim>& geometry) {
  std::vector<double> patch_checksum(field.layout().size(), 0.0);
  std::vector<double> patch_exact_checksum(field.layout().size(), 0.0);
  pops::Real local_maximum_error = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const std::size_t global = field.global_index(local);
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    double checksum = 0;
    double exact_checksum = 0;
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      const pops::Real value = host(field_offset(fab.grown_box(), index));
      const pops::Real exact = manufactured_mode(geometry, index);
      local_maximum_error = std::max(local_maximum_error, std::abs(value - exact));
      double weight = 1.0;
      double scale = 37.0;
      for (int axis = 0; axis < Dim; ++axis) {
        weight += scale * static_cast<double>(index[axis] - geometry.domain().lo[axis] + 1);
        scale *= 131.0;
      }
      checksum += weight * static_cast<double>(value);
      exact_checksum += weight * static_cast<double>(exact);
    }
    patch_checksum[global] = checksum;
    patch_exact_checksum[global] = exact_checksum;
  }
  pops::all_reduce_sum_inplace(patch_checksum.data(), static_cast<int>(patch_checksum.size()));
  pops::all_reduce_sum_inplace(patch_exact_checksum.data(),
                               static_cast<int>(patch_exact_checksum.size()));
  ManufacturedDiagnostics result;
  result.maximum_error =
      static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_maximum_error)));
  for (std::size_t patch = 0; patch < patch_checksum.size(); ++patch) {
    result.checksum += patch_checksum[patch];
    result.exact_checksum += patch_exact_checksum[patch];
  }
  result.checksum_spread =
      pops::all_reduce_max(result.checksum) - (-pops::all_reduce_max(-result.checksum));
  return result;
}

template <int Dim>
void expect_remote_fac_transport(const pops::elliptic::amr::CompositeFacPoisson<Dim>& solver,
                                 int rank_count) {
  if (rank_count > 1) {
    EXPECT_TRUE(solver.has_remote_same_level_halo());
    EXPECT_TRUE(solver.has_remote_parent_gather());
    EXPECT_TRUE(solver.has_remote_fine_restriction());
  }
}

template <int Dim>
void prove_screened_constant_fac(int level_count, int rank_count, int rank) {
  pops::CompositeFacOptions options;
  options.max_iters = 50;
  options.fine_sweeps = 4;
  options.rel_tol = pops::Real{5e-3};
  options.abs_tol = pops::Real{1e-10};
  options.coarse_rel_tol = pops::Real{1e-4};
  options.coarse_abs_tol = pops::Real{1e-10};
  options.coarse_cycles = 192;

  pops::elliptic::amr::CompositeFacPoisson<Dim> solver(
      make_fac_request<Dim>(level_count, rank_count, rank, pops::Real{1}), options, pops::Real{1});
  EXPECT_EQ(solver.n_levels(), level_count);
  expect_remote_fac_transport(solver, rank_count);
  for (int level = 0; level < solver.n_levels(); ++level) {
    solver.rhs_level(level).set_val(pops::Real{1});
    solver.phi_level(level).set_val(pops::Real{0});
  }

  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                               << " reference=" << report.reference_residual_norm;
  EXPECT_LT(report.residual_norm, report.reference_residual_norm);
  EXPECT_EQ(pops::all_reduce_min(static_cast<long>(report.iters)),
            pops::all_reduce_max(static_cast<long>(report.iters)));
  for (int level = 0; level < solver.n_levels(); ++level)
    EXPECT_LT(maximum_constant_error(solver.phi_level(level)), pops::Real{0.1});
}

template <int Dim>
void prove_manufactured_fac(int level_count, int rank_count, int rank) {
  pops::CompositeFacOptions options;
  options.max_iters = 80;
  options.fine_sweeps = 4;
  options.rel_tol = pops::Real{2e-4};
  options.abs_tol = pops::Real{1e-11};
  options.coarse_rel_tol = pops::Real{1e-5};
  options.coarse_abs_tol = pops::Real{1e-11};
  options.coarse_cycles = 256;

  auto request = make_fac_request<Dim>(level_count, rank_count, rank, pops::Real{0});
  std::vector<pops::Geometry<Dim>> geometries;
  geometries.reserve(request.levels.size());
  for (const auto& level : request.levels)
    geometries.push_back(level.geometry);
  pops::elliptic::amr::CompositeFacPoisson<Dim> solver(std::move(request), options, pops::Real{1});
  EXPECT_EQ(solver.n_levels(), level_count);
  expect_remote_fac_transport(solver, rank_count);
  for (int level = 0; level < solver.n_levels(); ++level) {
    fill_manufactured_rhs(solver.rhs_level(level), geometries[static_cast<std::size_t>(level)]);
    solver.phi_level(level).set_val(pops::Real{0});
  }

  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                               << " reference=" << report.reference_residual_norm;
  EXPECT_LT(report.residual_norm, report.reference_residual_norm);
  EXPECT_EQ(pops::all_reduce_min(static_cast<long>(report.iters)),
            pops::all_reduce_max(static_cast<long>(report.iters)));
  for (int level = 0; level < solver.n_levels(); ++level) {
    const ManufacturedDiagnostics diagnostics = manufactured_diagnostics(
        solver.phi_level(level), geometries[static_cast<std::size_t>(level)]);
    EXPECT_EQ(diagnostics.checksum_spread, 0.0);
    EXPECT_TRUE(std::isfinite(diagnostics.checksum));
    EXPECT_TRUE(std::isfinite(diagnostics.exact_checksum));
    EXPECT_GT(std::abs(diagnostics.exact_checksum), 1e-12);
    EXPECT_LT(diagnostics.maximum_error, pops::Real{0.15});
    EXPECT_LT(std::abs(diagnostics.checksum - diagnostics.exact_checksum),
              0.15 * std::abs(diagnostics.exact_checksum));
    if (pops::my_rank() == 0)
      std::printf("FACMMS levels=%d level=%d np=%d dim=%d error=%.17e checksum=%.17e exact=%.17e\n",
                  level_count, level, rank_count, Dim,
                  static_cast<double>(diagnostics.maximum_error), diagnostics.checksum,
                  diagnostics.exact_checksum);
  }
}

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> tensor_periodic_boundary(
    const pops::Geometry<Dim>& geometry) {
  std::array<bool, Dim> periodic{};
  periodic[0] = true;
  const auto topology = pops::BoundaryTopology<Dim>::axis_periodic(periodic);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    if (axis == 0)
      continue;
    for (const pops::BoundarySide side : {pops::BoundarySide::lower, pops::BoundarySide::upper})
      faces[static_cast<std::size_t>(pops::Face<Dim>{axis, side}.ordinal())] =
          pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real{0}};
  }
  return {topology, faces, spacing};
}

template <int Dim>
pops::runtime::program::HierarchyTensorSolverBuildRequest<Dim> periodic_tensor_request(
    int rank_count, int rank) {
  using pops::runtime::program::HierarchyTensorLevelBuildRequest;
  using pops::runtime::program::HierarchyTensorSolverBuildRequest;
  using namespace pops::runtime::program::tensor_elliptic_detail;

  auto rank_extent = uniform_extent<Dim>(1);
  rank_extent[0] = rank_count;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  const auto local_rank = rank_coordinate<Dim>(rank);
  std::array<int, Dim> ratio_values{};
  ratio_values.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio{ratio_values};

  pops::Index<Dim> coarse_upper{};
  for (int axis = 0; axis < Dim; ++axis)
    coarse_upper[axis] = 7;
  const pops::Box<Dim> coarse_domain{pops::Index<Dim>{}, coarse_upper};
  const pops::Geometry<Dim> coarse_geometry =
      pops::Geometry<Dim>::from_bounds(coarse_domain, uniform_coordinates<Dim>(pops::Real{0}),
                                       uniform_coordinates<Dim>(pops::Real{1}));
  const pops::mesh::BoxArray<Dim> coarse_layout(std::vector<pops::Box<Dim>>{coarse_domain});
  const auto coarse_distribution =
      pops::mesh::Distribution<Dim>::replicated(coarse_layout, rank_space);

  const pops::Geometry<Dim> middle_geometry = coarse_geometry.refine(uniform_extent<Dim>(2));
  pops::Index<Dim> middle_lower{};
  pops::Index<Dim> middle_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    middle_lower[axis] = 4;
    middle_upper[axis] = 11;
  }
  auto middle_grid = uniform_extent<Dim>(8);
  middle_grid[0] = 2;
  const auto middle_layout = pops::mesh::BoxArray<Dim>::from_domain(
      pops::Box<Dim>{middle_lower, middle_upper}, middle_grid);
  const auto middle_distribution = shifted_distribution(middle_layout, rank_space, 0, rank_count);

  const pops::Geometry<Dim> fine_geometry = middle_geometry.refine(uniform_extent<Dim>(2));
  pops::Index<Dim> fine_lower{};
  pops::Index<Dim> fine_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lower[axis] = 12;
    fine_upper[axis] = 19;
  }
  auto fine_grid = uniform_extent<Dim>(8);
  fine_grid[0] = 2;
  const auto fine_layout =
      pops::mesh::BoxArray<Dim>::from_domain(pops::Box<Dim>{fine_lower, fine_upper}, fine_grid);
  const auto fine_distribution = shifted_distribution(fine_layout, rank_space, 1, rank_count);

  HierarchyTensorSolverBuildRequest<Dim> request;
  request.block = 31;
  request.components = 1;
  request.levels.push_back(HierarchyTensorLevelBuildRequest<Dim>{
      coarse_geometry, tensor_periodic_boundary(coarse_geometry), coarse_layout,
      coarse_distribution, local_rank});
  request.levels.push_back(HierarchyTensorLevelBuildRequest<Dim>{
      middle_geometry, tensor_periodic_boundary(middle_geometry), middle_layout,
      middle_distribution, local_rank});
  request.levels.push_back(
      HierarchyTensorLevelBuildRequest<Dim>{fine_geometry, tensor_periodic_boundary(fine_geometry),
                                            fine_layout, fine_distribution, local_rank});
  request.ratios = {ratio, ratio};
  request.plan_identity = "pops.test.mpi-composite-fac.periodic-full-tensor";
  request.operator_contract_identity = std::string(kScalarTensorEllipticContract);
  request.assembly_field_slots = assembly_slots<Dim>();
  request.solution_field_slot = "pops.tensor-elliptic.solution";
  request.options = default_options();
  return request;
}

template <int Dim>
pops::Real tensor_coefficient(int row, int column) {
  if (row == column)
    return pops::Real{1} + pops::Real{0.25} * static_cast<pops::Real>(row);
  // In 3D the smallest diagonal is 1 and its two off-diagonal entries sum to 0.9, so this is a
  // deliberately strong cross-term signal while retaining the provider's strict diagonal-
  // dominance ellipticity contract in every dimension.
  return pops::Real{0.45};
}

template <int Dim>
pops::Real tensor_mode(const pops::Geometry<Dim>& geometry, const pops::Index<Dim>& index) {
  pops::Real result = pops::Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real frequency = axis == 0 ? pops::Real{2} * kPi : kPi;
    result *= std::sin(frequency * geometry.cell_coordinate(axis, index[axis]));
  }
  return result;
}

template <int Dim>
pops::Real tensor_rhs_value(const pops::Geometry<Dim>& geometry, const pops::Index<Dim>& index) {
  pops::Real diagonal = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real frequency = axis == 0 ? pops::Real{2} * kPi : kPi;
    const pops::Real angle = frequency * geometry.spacing(axis) / pops::Real{2};
    const pops::Real inverse_spacing = pops::Real{1} / geometry.spacing(axis);
    diagonal += tensor_coefficient<Dim>(axis, axis) * pops::Real{4} * std::sin(angle) *
                std::sin(angle) * inverse_spacing * inverse_spacing;
  }
  pops::Real result = diagonal * tensor_mode(geometry, index);
  for (int row = 0; row < Dim; ++row)
    for (int column = 0; column < Dim; ++column) {
      if (row == column)
        continue;
      pops::Real derivative = pops::Real{1};
      for (int axis = 0; axis < Dim; ++axis) {
        const pops::Real frequency = axis == 0 ? pops::Real{2} * kPi : kPi;
        const pops::Real coordinate = geometry.cell_coordinate(axis, index[axis]);
        if (axis == row || axis == column)
          derivative *= std::sin(frequency * geometry.spacing(axis)) / geometry.spacing(axis) *
                        std::cos(frequency * coordinate);
        else
          derivative *= std::sin(frequency * coordinate);
      }
      result -= tensor_coefficient<Dim>(row, column) * derivative;
    }
  return result;
}

template <int Dim>
pops::Real tensor_diagonal_rhs_value(const pops::Geometry<Dim>& geometry,
                                     const pops::Index<Dim>& index) {
  pops::Real diagonal = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real frequency = axis == 0 ? pops::Real{2} * kPi : kPi;
    const pops::Real angle = frequency * geometry.spacing(axis) / pops::Real{2};
    const pops::Real inverse_spacing = pops::Real{1} / geometry.spacing(axis);
    diagonal += tensor_coefficient<Dim>(axis, axis) * pops::Real{4} * std::sin(angle) *
                std::sin(angle) * inverse_spacing * inverse_spacing;
  }
  return diagonal * tensor_mode(geometry, index);
}

template <int Dim>
std::pair<pops::Real, pops::Real> tensor_cross_term_signal(const pops::MultiFab<Dim>& field,
                                                           const pops::Geometry<Dim>& geometry) {
  pops::Real local_cross = 0;
  pops::Real local_diagonal = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(field.box(local).numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(field.box(local), ordinal);
      const pops::Real diagonal = tensor_diagonal_rhs_value(geometry, index);
      local_diagonal = std::max(local_diagonal, std::abs(diagonal));
      local_cross = std::max(local_cross, std::abs(tensor_rhs_value(geometry, index) - diagonal));
    }
  return {static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_cross))),
          static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_diagonal)))};
}

template <int Dim>
void fill_tensor_rhs(pops::MultiFab<Dim>& rhs, const pops::Geometry<Dim>& geometry) {
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      host(field_offset(fab.grown_box(), index)) = tensor_rhs_value(geometry, index);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void fill_tensor_exact(pops::MultiFab<Dim>& field, const pops::Geometry<Dim>& geometry) {
  field.set_val(pops::Real{0});
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      host(field_offset(fab.grown_box(), index)) = tensor_mode(geometry, index);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
ManufacturedDiagnostics tensor_diagnostics(const pops::MultiFab<Dim>& field,
                                           const pops::Geometry<Dim>& geometry) {
  std::vector<double> patch_checksum(field.layout().size(), 0.0);
  std::vector<double> patch_exact_checksum(field.layout().size(), 0.0);
  pops::Real local_maximum_error = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const std::size_t global = field.global_index(local);
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      const pops::Real value = host(field_offset(fab.grown_box(), index));
      const pops::Real exact = tensor_mode(geometry, index);
      local_maximum_error = std::max(local_maximum_error, std::abs(value - exact));
      const double weight =
          static_cast<double>(global + 1) * 257.0 + static_cast<double>(ordinal + 1);
      patch_checksum[global] += weight * static_cast<double>(value);
      patch_exact_checksum[global] += weight * static_cast<double>(exact);
    }
  }
  ManufacturedDiagnostics result;
  result.maximum_error =
      static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_maximum_error)));
  if (field.distribution().replicated()) {
    // Every replica traverses the same global patch/cell order before any collective.  The spread
    // therefore detects a divergent replica directly; only rank zero contributes afterward so the
    // reported checksum is invariant under np=1/2/4 rather than multiplied by the replica count.
    double local_checksum = 0;
    double local_exact_checksum = 0;
    for (std::size_t patch = 0; patch < patch_checksum.size(); ++patch) {
      local_checksum += patch_checksum[patch];
      local_exact_checksum += patch_exact_checksum[patch];
    }
    const double value_spread =
        pops::all_reduce_max(local_checksum) - pops::all_reduce_min(local_checksum);
    const double exact_spread =
        pops::all_reduce_max(local_exact_checksum) - pops::all_reduce_min(local_exact_checksum);
    result.checksum_spread = std::max(value_spread, exact_spread);
    result.checksum = pops::all_reduce_sum(pops::my_rank() == 0 ? local_checksum : 0.0);
    result.exact_checksum = pops::all_reduce_sum(pops::my_rank() == 0 ? local_exact_checksum : 0.0);
  } else {
    // Partitioned levels have exactly one owner contribution in every global-patch slot.  Reduce
    // slots first, then fold once in immutable global patch order for decomposition-independent
    // accumulation.
    pops::all_reduce_sum_inplace(patch_checksum.data(), static_cast<int>(patch_checksum.size()));
    pops::all_reduce_sum_inplace(patch_exact_checksum.data(),
                                 static_cast<int>(patch_exact_checksum.size()));
    for (std::size_t patch = 0; patch < patch_checksum.size(); ++patch) {
      result.checksum += patch_checksum[patch];
      result.exact_checksum += patch_exact_checksum[patch];
    }
    result.checksum_spread =
        pops::all_reduce_max(result.checksum) - pops::all_reduce_min(result.checksum);
  }
  return result;
}

template <int Dim>
void prove_periodic_tensor_fac(int rank_count, int rank) {
  using namespace pops::runtime::program;
  auto request = periodic_tensor_request<Dim>(rank_count, rank);
  std::vector<pops::Geometry<Dim>> geometries;
  geometries.reserve(request.levels.size());
  for (const auto& level : request.levels)
    geometries.push_back(level.geometry);

  const pops::ExecutionLane lane = pops::ExecutionLane::world(
      "pops.test.mpi-composite-fac.periodic-tensor.dim-" + std::to_string(Dim));
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<Dim>(lane);
  const auto provider = registry->resolve(tensor_elliptic_detail::kCompositeTensorProvider);
  const auto capabilities = provider->capability_contracts();
  EXPECT_NE(std::find(capabilities.begin(), capabilities.end(),
                      "pops.hierarchy.composite-tensor-fac.partitioned-mpi"),
            capabilities.end());
  EXPECT_NE(std::find(capabilities.begin(), capabilities.end(),
                      "pops.hierarchy.composite-tensor-fac.full-tensor-nd@3"),
            capabilities.end());
  for (const auto& level : request.levels)
    EXPECT_TRUE(
        level.boundary.topology().is_periodic(pops::Face<Dim>{0, pops::BoundarySide::lower}));
  EXPECT_TRUE(provider->supports(request).accepted());

  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(request), lane);
  auto* exact = dynamic_cast<AmrTensorElliptic<Dim>*>(prepared.get());
  ASSERT_NE(exact, nullptr);
  EXPECT_TRUE(exact->borrows_execution_lane());
  if (rank_count > 1) {
    EXPECT_TRUE(exact->has_remote_same_level_halo());
    EXPECT_TRUE(exact->has_remote_parent_gather());
    EXPECT_TRUE(exact->has_remote_fine_restriction());
  }
  EXPECT_TRUE(exact->uses_replicated_parent_restriction());

  for (int level = 0; level < prepared->level_count(); ++level) {
    for (int row = 0; row < Dim; ++row)
      for (int column = 0; column < Dim; ++column)
        prepared->assembly_target(tensor_elliptic_detail::coefficient_slot(row, column), level)
            .set_val(tensor_coefficient<Dim>(row, column));
    auto& rhs = prepared->assembly_target("pops.tensor-elliptic.rhs", level);
    const auto& geometry = geometries[static_cast<std::size_t>(level)];
    fill_tensor_rhs(rhs, geometry);
    const auto [cross_signal, diagonal_signal] = tensor_cross_term_signal(rhs, geometry);
    if constexpr (Dim > 1) {
      EXPECT_GT(cross_signal, pops::Real{1});
      EXPECT_GT(cross_signal, pops::Real{0.2} * diagonal_signal);
    } else {
      EXPECT_EQ(cross_signal, pops::Real{0});
    }
    pops::MultiFab<Dim> exact_guess(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1,
                                    uniform_extent<Dim>(1));
    fill_tensor_exact(exact_guess, geometry);
    prepared->stage_initial_guess(level, &exact_guess);
  }

  const pops::SolveReport report =
      solve_prepared_hierarchy_tensor_collectively(
          *prepared, HierarchyTensorSolveControls{pops::Real{1e-5}, pops::Real{1e-12}, 100}, lane)
          .consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  EXPECT_EQ(pops::all_reduce_min(static_cast<long>(report.iters), lane),
            pops::all_reduce_max(static_cast<long>(report.iters), lane));
  for (int level = 0; level < prepared->level_count(); ++level) {
    const ManufacturedDiagnostics diagnostics =
        tensor_diagnostics(prepared->solution(level), geometries[static_cast<std::size_t>(level)]);
    EXPECT_EQ(diagnostics.checksum_spread, 0.0);
    EXPECT_TRUE(std::isfinite(diagnostics.checksum));
    EXPECT_TRUE(std::isfinite(diagnostics.exact_checksum));
    EXPECT_LT(diagnostics.maximum_error, pops::Real{0.04});
    if (std::abs(diagnostics.exact_checksum) > 1e-12)
      EXPECT_LT(std::abs(diagnostics.checksum - diagnostics.exact_checksum),
                0.04 * std::abs(diagnostics.exact_checksum));
    if (pops::my_rank() == 0)
      std::printf(
          "FACTENSORPERIODIC level=%d np=%d dim=%d error=%.17e checksum=%.17e exact=%.17e\n", level,
          rank_count, Dim, static_cast<double>(diagnostics.maximum_error), diagnostics.checksum,
          diagnostics.exact_checksum);
  }
}

int run_mpi_composite_fac(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_screened_constant_fac<pops::kNativeDimension>(2, pops::n_ranks(), pops::my_rank());
      prove_manufactured_fac<pops::kNativeDimension>(2, pops::n_ranks(), pops::my_rank());
      prove_screened_constant_fac<pops::kNativeDimension>(3, pops::n_ranks(), pops::my_rank());
      prove_manufactured_fac<pops::kNativeDimension>(3, pops::n_ranks(), pops::my_rank());
      prove_periodic_tensor_fac<pops::kNativeDimension>(pops::n_ranks(), pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact-rank composite FAC failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf(
          "OK test_mpi_composite_fac (np=%d dim=%d constant+MMS 2/3-level, periodic tensor)\n",
          pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_composite_fac, NativeDimensionUsesPartitionedCompositeHierarchy) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_composite_fac, "test_mpi_composite_fac"), 0);
}
