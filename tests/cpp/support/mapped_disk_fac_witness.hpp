#pragma once

#include <pops/runtime/amr/tensor_composite_fac.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pops::test {

struct MappedDiskFacMeasurement {
  std::vector<Real> coarse;
  std::vector<Real> fine;
  SolveReport report{};
  Real maximum_error = 0;
  bool remote_parent_gather = false;
};

// Solve -div(A grad psi) on the mapped full disk, including finite magnetic response.
// A=diag(r,1/r)+s^2 alpha r rho C^T (I-sJ)^-1 C, with the released benchmark alpha/Omega.
// The manufactured exact psi=r^3(1-r^2)cos(3 theta) vanishes on the conducting wall.
// Compare a half-domain refined patch translated
// in the periodic direction. offset=0 has a seam with no opposite fine peer;
// offset=n/2 is interior; offset=3*n/2 splits the patch across the seam.
inline MappedDiskFacMeasurement mapped_disk_fac_witness(
    int n, int fine_offset, bool replicated_parent, const ExecutionLane& lane,
    elliptic::nd::CartesianTensorStencilOptions stencil_options =
        {.zero_flux_faces=1u, .dirichlet_faces=2u, .arithmetic_diagonal=true},
    int angular_ratio = 1, int fine_sweeps = 64, int coarse_cycles = 512,
    int maximum_iterations = 200, Real damping = Real(1),
    Real relative_tolerance = Real(2e-9),
    runtime::program::tensor_fac::CoarseCorrectionMethod coarse_method =
        runtime::program::tensor_fac::CoarseCorrectionMethod::gauss_seidel,
    int coarse_restart = 64) {
  using namespace runtime::program::tensor_fac;
  using Field = MultiFab<2>;
  constexpr Real pi = Real(3.141592653589793238462643383279502884L);
  const int nt = angular_ratio * n;
  if (n < 8 || n % 4 != 0 || angular_ratio < 1 || fine_offset < 0 || fine_offset >= 2 * nt ||
      fine_offset % 2 != 0 || (lane.size() != 1 && lane.size() != 2))
    throw std::invalid_argument("invalid tensor periodic seam witness layout");
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{lane.size(), 1}};
  const Index<2> rank{lane.rank(), 0};
  const Box<2> domain{Index<2>{0, 0}, Index<2>{n - 1, nt - 1}};
  const Geometry<2> coarse_geometry = Geometry<2>::from_bounds(
      domain, RealVector<2>{0, 0}, RealVector<2>{1, 2 * pi});
  const std::array<Geometry<2>, 2> geometry{
      coarse_geometry, coarse_geometry.refine(Extent<2>{2, 2})};
  std::vector<PhysicalBoundaryConditions<2>> boundary;
  boundary.reserve(2);
  for (int level = 0; level < 2; ++level) {
    std::array<PhysicalBoundaryFace, 4> faces{};
    for (BoundarySide side : {BoundarySide::lower, BoundarySide::upper})
      faces[static_cast<std::size_t>(Face<2>{0, side}.ordinal())] =
          {side == BoundarySide::lower ? PhysicalBoundaryKind::neumann : PhysicalBoundaryKind::dirichlet, Real(0)};
    boundary.push_back(PhysicalBoundaryConditions<2>{
        BoundaryTopology<2>::axis_periodic(std::array<bool, 2>{false, true}), faces,
        RealVector<2>{geometry[level].spacing(0), geometry[level].spacing(1)}});
  }
  std::vector<Box<2>> parent_boxes;
  std::vector<Index<2>> parent_owners;
  const int parent_parts = replicated_parent ? 1 : lane.size();
  for (int part = 0; part < parent_parts; ++part) {
    parent_boxes.push_back({Index<2>{0, part * nt / parent_parts},
                            Index<2>{n - 1, (part + 1) * nt / parent_parts - 1}});
    parent_owners.push_back(Index<2>{part, 0});
  }
  const mesh::BoxArray<2> parent_layout(std::move(parent_boxes));
  const auto parent_distribution = replicated_parent
      ? mesh::Distribution<2>::replicated(parent_layout, ranks)
      : mesh::Distribution<2>::partitioned(parent_layout, ranks, std::move(parent_owners));
  std::vector<Box<2>> fine_boxes;
  std::vector<Index<2>> fine_owners;
  for (int part = 0; part < lane.size(); ++part) {
    const int lower_x = n / 2 + part * n / lane.size();
    const int upper_x = n / 2 + (part + 1) * n / lane.size() - 1;
    const int upper_y = fine_offset + nt - 1;
    fine_boxes.push_back({Index<2>{lower_x, fine_offset},
                          Index<2>{upper_x, std::min(upper_y, 2 * nt - 1)}});
    fine_owners.push_back(Index<2>{part, 0});
    if (upper_y >= 2 * nt) {
      fine_boxes.push_back({Index<2>{lower_x, 0}, Index<2>{upper_x, upper_y - 2 * nt}});
      fine_owners.push_back(Index<2>{part, 0});
    }
  }
  const mesh::BoxArray<2> fine_layout(std::move(fine_boxes));
  const auto fine_distribution =
      mesh::Distribution<2>::partitioned(fine_layout, ranks, std::move(fine_owners));
  const std::array<mesh::BoxArray<2>, 2> layouts{parent_layout, fine_layout};
  const std::array<mesh::Distribution<2>, 2> distributions{
      parent_distribution, fine_distribution};
  std::array<std::array<std::unique_ptr<Field>, 4>, 2> coefficients;
  std::array<std::unique_ptr<Field>, 2> rhs, initial, solution;
  std::array<LevelBinding<2, Field::memory_space>, 2> bindings;
  const Real phase = Real(2) * pi * Real(fine_offset) / Real(2 * nt);
  auto offset = [](const Box<2>& box, int i, int j) {
    return static_cast<std::size_t>(i - box.lo[0]) +
        static_cast<std::size_t>(j - box.lo[1]) * static_cast<std::size_t>(box.length(0));
  };
  for (int level = 0; level < 2; ++level) {
    rhs[level] = std::make_unique<Field>(layouts[level], distributions[level], rank, 1, Extent<2>{});
    initial[level] = std::make_unique<Field>(layouts[level], distributions[level], rank, 1, Extent<2>{1, 1});
    solution[level] = std::make_unique<Field>(layouts[level], distributions[level], rank, 1, Extent<2>{1, 1});
    initial[level]->set_val(Real(0));
    solution[level]->set_val(Real(0));
    for (int slot = 0; slot < 4; ++slot) {
      coefficients[level][slot] = std::make_unique<Field>(
          layouts[level], distributions[level], rank, 1, Extent<2>{1, 1});
      // Poison every ghost: successful preparation must fill it from its owner.
      coefficients[level][slot]->set_val(std::numeric_limits<Real>::quiet_NaN());
    }
    for (std::size_t local = 0; local < rhs[level]->local_size(); ++local) {
      auto& right = rhs[level]->fab(local);
      auto right_host = right.create_host_mirror();
      std::vector<typename Field::fab_type::host_mirror_type> tensor_host;
      tensor_host.reserve(4);
      for (int slot = 0; slot < 4; ++slot) {
        tensor_host.push_back(coefficients[level][slot]->fab(local).create_host_mirror());
        coefficients[level][slot]->fab(local).copy_to_host(tensor_host[slot]);
      }
      const auto valid = right.box();
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          const Real x = geometry[level].cell_coordinate(0, i);
          const Real y = geometry[level].cell_coordinate(1, j) - phase;
          constexpr Real mode = 3;
          constexpr Real epsilon = Real(0.1);
          constexpr Real alpha = Real(39.4784176e12);
          constexpr Real omega = Real(-6.28318531e12);
          constexpr Real s = Real(1e-4);
          const Real w = s * omega;
          const Real response = s * s * alpha / (Real(1) + w * w);
          const Real density = Real(1) + epsilon * x * x * std::sin(y);
          const Real density_r = Real(2) * epsilon * x * std::sin(y);
          const Real density_theta = epsilon * x * x * std::cos(y);
          const Real radial = std::pow(x, mode) * (Real(1) - x * x);
          const Real derivative = mode * std::pow(x, mode - Real(1)) -
                                  (mode + Real(2)) * std::pow(x, mode + Real(1));
          const Real psi_r = derivative * std::cos(mode * y);
          const Real psi_theta = -mode * radial * std::sin(mode * y);
          const Real symmetric = Real(1) + response * density;
          const Real skew = response * w * density;
          const std::array<Real, 4> tensor{x * symmetric, skew, -skew, symmetric / x};
          for (int slot = 0; slot < 4; ++slot)
            tensor_host[slot](offset(coefficients[level][slot]->fab(local).grown_box(), i, j)) = tensor[slot];
          right_host(offset(right.grown_box(), i, j)) =
              symmetric * Real(4) * (mode + Real(1)) * std::pow(x, mode + Real(1)) * std::cos(mode * y)
              - response * (x * density_r * psi_r + density_theta * psi_theta / x)
              - response * w * (density_r * psi_theta - density_theta * psi_r);
        }
      right.copy_from_host(right_host);
      for (int slot = 0; slot < 4; ++slot)
        coefficients[level][slot]->fab(local).copy_from_host(tensor_host[slot]);
    }
    bindings[level] = LevelBinding<2, Field::memory_space>{
        &geometry[level], &boundary[level],
        {coefficients[level][0].get(), coefficients[level][1].get(),
         coefficients[level][2].get(), coefficients[level][3].get()},
        rhs[level].get(), initial[level].get(), solution[level].get()};
  }
  const std::array<amr::RefinementRatio<2>, 1> ratios{
      amr::RefinementRatio<2>{std::array<int, 2>{2, 2}}};
  FullTensorCompositeFac<2> solver(bindings, ratios, lane, stencil_options,
                                  coarse_method, coarse_restart);
  Controls controls;
  controls.relative_tolerance = relative_tolerance;
  controls.absolute_tolerance = Real(1e-12);
  controls.maximum_iterations = maximum_iterations;
  controls.fine_sweeps = fine_sweeps;
  controls.coarse_cycles = coarse_cycles;
  controls.coarse_method = coarse_method;
  controls.coarse_restart = coarse_restart;
  controls.coarse_relative_tolerance = Real(1e-11);
  controls.correction_damping = damping;
  MappedDiskFacMeasurement result;
  result.remote_parent_gather = solver.has_remote_parent_gather();
  result.report = solver.solve(controls, lane);
  result.coarse.assign(static_cast<std::size_t>(n * nt), Real(0));
  result.fine.assign(static_cast<std::size_t>(n * nt), Real(0));
  for (int level = 0; level < 2; ++level) {
    if (level == 0 && replicated_parent && lane.rank() != 0)
      continue;
    for (std::size_t local = 0; local < solution[level]->local_size(); ++local) {
      const auto& fab = solution[level]->fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const auto valid = fab.box();
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          const Real value = host(offset(fab.grown_box(), i, j));
          if (!std::isfinite(value))
            throw std::runtime_error("mapped disk finite-magnetization FAC MMS has a non-finite solution");
          const int width = level == 0 ? nt : 2 * nt;
          const int phase_cells = level == 0 ? fine_offset / 2 : fine_offset;
          const int jj = (j - phase_cells + width) % width;
          auto& values = level == 0 ? result.coarse : result.fine;
          values[static_cast<std::size_t>((level == 0 ? i : i - n / 2) + n * jj)] = value;
          if (level == 1) {
            const Real x = geometry[level].cell_coordinate(0, i);
            const Real y = geometry[level].cell_coordinate(1, j) - phase;
            result.maximum_error = std::max(result.maximum_error,
                std::abs(value - x * x * x * (Real(1) - x * x) * std::cos(Real(3) * y)));
          }
        }
    }
  }
#ifdef POPS_HAS_MPI
  if (lane.size() > 1)
    for (auto* values : {&result.coarse, &result.fine}) {
      std::vector<Real> gathered(values->size());
      if (MPI_Allreduce(values->data(), gathered.data(), static_cast<int>(values->size()),
                        mpi_real_datatype(), MPI_SUM, lane.native_handle()) != MPI_SUCCESS)
        throw std::runtime_error("tensor seam witness gather failed");
      values->swap(gathered);
    }
#endif
  result.maximum_error = all_reduce_max(result.maximum_error, lane);
  return result;
}

inline Real mapped_disk_fac_difference(const MappedDiskFacMeasurement& a, const MappedDiskFacMeasurement& b) {
  Real difference = 0;
  for (std::size_t i = 0; i < a.coarse.size(); ++i)
    difference = std::max(difference, std::abs(a.coarse[i] - b.coarse.at(i)));
  for (std::size_t i = 0; i < a.fine.size(); ++i)
    difference = std::max(difference, std::abs(a.fine[i] - b.fine.at(i)));
  return difference;
}

}  // namespace pops::test
