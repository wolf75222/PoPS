#pragma once

#include <pops/runtime/amr/tensor_composite_fac.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace pops::test {

struct PartialMappedDiskFacMeasurement {
  SolveReport report;
  Real reference = 0;
  Real independent_residual = 0;
  Real maximum_error = 0;
  Real coefficient_ghost_difference = 0;
  Real coefficient_scale = 0;
  std::vector<long> active_cells;
  bool remote_parent_gather = false;
  bool all_values_finite = true;
};

// Independent dense reference for the documented cell-centered interpolation and boundary
// laws. It neither calls FAC's private transfers nor reproduces its iteration cycle.
struct PartialDiskReference {
  std::vector<std::vector<Box<2>>> boxes;
  std::vector<std::vector<Real>> values;
  bool coefficient = false;

  bool valid(int level, int i, int j) const {
    for (const auto& box : boxes[level])
      if (i >= box.lo[0] && i <= box.hi[0] && j >= box.lo[1] && j <= box.hi[1])
        return true;
    return false;
  }
  bool covered(int level, int i, int j) const {
    return level + 1 < static_cast<int>(boxes.size()) && valid(level + 1, 2 * i, 2 * j);
  }
  Real sample(int level, int i, int j) const {
    const int nr = 16 << level, nt = 64 << level;
    j = (j % nt + nt) % nt;
    if (i < 0)
      return sample(level, 0, j);  // zero Neumann or constant coefficient extrapolation
    if (i >= nr)
      return (coefficient ? Real(1) : Real(-1)) * sample(level, nr - 1, j);
    if (valid(level, i, j))
      return values[level][static_cast<std::size_t>(j * nr + i)];
    if (level == 0)
      throw std::logic_error("partial disk reference lacks root coverage");
    const int p = i / 2, q = j / 2, parent_nr = nr / 2;
    const Real center = sample(level - 1, p, q);
    const Real radial_slope = p == 0 ? sample(level - 1, 1, q) - center
        : p == parent_nr - 1 ? center - sample(level - 1, p - 1, q)
        : Real(0.5) * (sample(level - 1, p + 1, q) - sample(level - 1, p - 1, q));
    const Real angular_slope = Real(0.5) *
        (sample(level - 1, p, q + 1) - sample(level - 1, p, q - 1));
    return center + (i % 2 ? Real(0.25) : Real(-0.25)) * radial_slope +
                    (j % 2 ? Real(0.25) : Real(-0.25)) * angular_slope;
  }
  void average_covered_down() {
    for (int child = static_cast<int>(boxes.size()) - 1; child > 0; --child) {
      const int parent_nr = 16 << (child - 1), parent_nt = 64 << (child - 1);
      for (int j = 0; j < parent_nt; ++j)
        for (int i = 0; i < parent_nr; ++i)
          if (covered(child - 1, i, j)) {
            Real sum = 0;
            for (int b = 0; b < 2; ++b)
              for (int a = 0; a < 2; ++a)
                sum += sample(child, 2 * i + a, 2 * j + b);
            values[child - 1][static_cast<std::size_t>(j * parent_nr + i)] = sum / Real(4);
          }
    }
  }
};

inline PartialMappedDiskFacMeasurement partial_mapped_disk_fac_witness(
    int level_count, const ExecutionLane& lane) {
  using namespace runtime::program::tensor_fac;
  using Field = MultiFab<2>;
  if (level_count < 2 || level_count > 3 || (lane.size() != 1 && lane.size() != 2))
    throw std::invalid_argument("partial mapped disk witness supports two/three levels and MPI1/2");
  constexpr Real pi = Real(3.141592653589793238462643383279502884L);
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{lane.size(), 1}};
  const Index<2> rank{lane.rank(), 0};
  PartialDiskReference exact;
  exact.boxes.resize(level_count);
  exact.values.resize(level_count);
  std::vector<Geometry<2>> geometry;
  std::vector<PhysicalBoundaryConditions<2>> boundary;
  std::vector<mesh::BoxArray<2>> layouts;
  std::vector<mesh::Distribution<2>> distributions;
  geometry.reserve(level_count); boundary.reserve(level_count);
  layouts.reserve(level_count); distributions.reserve(level_count);
  for (int level = 0; level < level_count; ++level) {
    const int nr = 16 << level, nt = 64 << level;
    geometry.push_back(Geometry<2>::from_bounds(
        Box<2>{Index<2>{0, 0}, Index<2>{nr - 1, nt - 1}},
        RealVector<2>{0, 0}, RealVector<2>{16, Real(2) * pi}));
    std::array<PhysicalBoundaryFace, 4> faces{};
    faces[0] = {PhysicalBoundaryKind::neumann, Real(0)};
    faces[1] = {PhysicalBoundaryKind::dirichlet, Real(0)};
    boundary.emplace_back(BoundaryTopology<2>::axis_periodic(std::array<bool, 2>{false, true}),
                          faces, RealVector<2>{geometry[level].spacing(0), geometry[level].spacing(1)});
    // Fine extents reproduce the failing partial hierarchy; root tiling and owners are
    // deterministic test choices. The middle level retains active cells on both radial sides.
    if (level == 0) {
      for (int j = 0; j < nt; j += 8)
        for (int i = 0; i < nr; i += 8)
          exact.boxes[level].push_back({Index<2>{i, j}, Index<2>{i + 7, j + 7}});
    } else {
      const int lower = level == 1 ? 6 : 18, upper = level == 1 ? 21 : 37;
      for (int j = 0; j < nt; j += 32)
        exact.boxes[level].push_back({Index<2>{lower, j}, Index<2>{upper, j + 31}});
    }
    std::vector<Index<2>> owners;
    for (std::size_t patch = 0; patch < exact.boxes[level].size(); ++patch)
      owners.push_back(Index<2>{static_cast<int>(patch) % lane.size(), 0});
    layouts.emplace_back(exact.boxes[level]);
    distributions.push_back(mesh::Distribution<2>::partitioned(layouts[level], ranks, std::move(owners)));
    exact.values[level].resize(static_cast<std::size_t>(nr * nt), Real(0));
    for (const auto& box : exact.boxes[level])
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const Real x = geometry[level].cell_coordinate(0, i) / Real(16);
          const Real theta = geometry[level].cell_coordinate(1, j);
          exact.values[level][static_cast<std::size_t>(j * nr + i)] =
              (Real(1) - x * x) * (Real(1) + Real(0.01) * std::pow(x, 5) * std::sin(Real(5) * theta));
        }
  }
  exact.average_covered_down();
  std::array<PartialDiskReference, 4> tensors{exact, exact, exact, exact};
  for (auto& tensor : tensors) tensor.coefficient = true;
  for (int level = 0; level < level_count; ++level)
    for (const auto& box : exact.boxes[level])
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const Real r = geometry[level].cell_coordinate(0, i);
          const Real theta = geometry[level].cell_coordinate(1, j), x = r / Real(16);
          // A self-contained analytic density, distinct from the physical benchmark initial
          // state. Keep the authors' alpha/Omega and actual bootstrap source midpoint.
          const Real rho = Real(1) + Real(0.1) * x * x * std::sin(theta);
          constexpr Real s = Real(5e-11), alpha = Real(39478417600000.), omega = Real(-6283185310000.);
          constexpr Real w = s * omega, response = s * s * alpha / (Real(1) + w * w);
          const std::array<Real, 4> tensor{
              r * (Real(1) + response * rho), response * w * rho,
              -response * w * rho, (Real(1) + response * rho) / r};
          for (int slot = 0; slot < 4; ++slot)
            tensors[slot].values[level][static_cast<std::size_t>(j * (16 << level) + i)] = tensor[slot];
        }
  std::vector<std::array<std::unique_ptr<Field>, 4>> coefficients(level_count);
  std::vector<std::unique_ptr<Field>> rhs(level_count), initial(level_count), solution(level_count), image(level_count);
  std::vector<LevelBinding<2, Field::memory_space>> bindings(level_count);
  auto offset = [](const Box<2>& box, int i, int j) {
    return static_cast<std::size_t>(i - box.lo[0]) +
        static_cast<std::size_t>(j - box.lo[1]) * static_cast<std::size_t>(box.length(0));
  };
  auto fill_reference = [&](Field& field, const PartialDiskReference& reference, int level) {
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      auto& fab = field.fab(local); auto host = fab.create_host_mirror();
      const auto grown = fab.grown_box();
      for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
        for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
          host(offset(grown, i, j)) = reference.sample(level, i, j);
      fab.copy_from_host(host);
    }
  };
  const elliptic::nd::CartesianTensorStencilOptions options{
      .zero_flux_faces = 1u, .dirichlet_faces = 2u, .arithmetic_diagonal = true};
  auto apply_original = [&](int level) {
    for (std::size_t local = 0; local < solution[level]->local_size(); ++local) {
      std::array<FieldView<const Real, 2>, 4> fields{};
      for (int slot = 0; slot < 4; ++slot)
        fields[slot] = std::as_const(coefficients[level][slot]->fab(local)).view();
      const auto op = elliptic::nd::make_cartesian_tensor_operator<
          elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
          std::as_const(solution[level]->fab(local)).view(),
          elliptic::nd::split_cartesian_tensor_coefficients<2>(fields), geometry[level], options);
      auto out = image[level]->fab(local).view();
      for_each_cell(solution[level]->box(local), KOKKOS_LAMBDA(const Index<2>& index) { out(index, 0) = op.image(index); });
    }
    Kokkos::fence();
  };
  for (int level = 0; level < level_count; ++level) {
    for (auto* group : {&rhs, &image})
      (*group)[level] = std::make_unique<Field>(layouts[level], distributions[level], rank, 1, Extent<2>{});
    for (auto* group : {&initial, &solution})
      (*group)[level] = std::make_unique<Field>(layouts[level], distributions[level], rank, 1, Extent<2>{1, 1});
    initial[level]->set_val(Real(0));
    fill_reference(*solution[level], exact, level);
    for (int slot = 0; slot < 4; ++slot) {
      coefficients[level][slot] = std::make_unique<Field>(layouts[level], distributions[level], rank, 1, Extent<2>{1, 1});
      fill_reference(*coefficients[level][slot], tensors[slot], level);
    }
    apply_original(level);
    for (std::size_t local = 0; local < rhs[level]->local_size(); ++local) {
      auto destination = rhs[level]->fab(local).view();
      auto source = std::as_const(image[level]->fab(local)).view();
      for_each_cell(rhs[level]->box(local), KOKKOS_LAMBDA(const Index<2>& index) {
        destination(index, 0) = source(index, 0);
      });
    }
    Kokkos::fence();
    bindings[level] = {&geometry[level], &boundary[level],
        {coefficients[level][0].get(), coefficients[level][1].get(), coefficients[level][2].get(), coefficients[level][3].get()},
        rhs[level].get(), initial[level].get(), solution[level].get()};
  }
  std::vector<amr::RefinementRatio<2>> ratios(level_count - 1, amr::RefinementRatio<2>{std::array<int, 2>{2, 2}});
  FullTensorCompositeFac<2> solver(bindings, ratios, lane, options,
      CoarseCorrectionMethod::gmres, 64, CoarsePreconditionerKind::polar_poisson);
  Controls controls;
  controls.maximum_iterations = 300;
  controls.relative_tolerance = Real(1e-10); controls.absolute_tolerance = Real(1e-12);
  controls.fine_sweeps = 64; controls.correction_damping = Real(0.5);
  controls.coarse_method = CoarseCorrectionMethod::gmres; controls.coarse_restart = 64;
  controls.coarse_preconditioner = CoarsePreconditionerKind::polar_poisson;
  controls.coarse_cycles = 512; controls.coarse_relative_tolerance = Real(1e-12);
  controls.coarse_absolute_tolerance = Real(0);
  PartialMappedDiskFacMeasurement result;
  result.remote_parent_gather = solver.has_remote_parent_gather();
  result.report = solver.solve(controls, lane);

  PartialDiskReference measured = exact;
  for (int level = 0; level < level_count; ++level) {
    std::fill(measured.values[level].begin(), measured.values[level].end(), Real(0));
    for (std::size_t local = 0; local < solution[level]->local_size(); ++local) {
      const auto& fab = solution[level]->fab(local); auto host = fab.create_host_mirror(); fab.copy_to_host(host);
      const auto valid = fab.box();
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          const Real value = host(offset(fab.grown_box(), i, j));
          result.all_values_finite = result.all_values_finite && std::isfinite(value);
          measured.values[level][static_cast<std::size_t>(j * (16 << level) + i)] = value;
        }
    }
#ifdef POPS_HAS_MPI
    if (lane.size() > 1) {
      std::vector<Real> global(measured.values[level].size());
      if (MPI_Allreduce(measured.values[level].data(), global.data(), static_cast<int>(global.size()),
                        mpi_real_datatype(), MPI_SUM, lane.native_handle()) != MPI_SUCCESS)
        throw std::runtime_error("partial disk reference gather failed");
      measured.values[level].swap(global);
    }
#endif
    // Reconstruct ghosts independently from returned valid cells, then reapply the original A.
    fill_reference(*solution[level], measured, level);
    apply_original(level);
  }
  result.active_cells.resize(level_count, 0);
  for (int level = 0; level < level_count; ++level) {
    long local_active = 0;
    for (std::size_t local = 0; local < solution[level]->local_size(); ++local) {
      const auto& out = image[level]->fab(local); auto oh = out.create_host_mirror(); out.copy_to_host(oh);
      const auto& right = rhs[level]->fab(local); auto rh = right.create_host_mirror(); right.copy_to_host(rh);
      const auto valid = out.box();
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
          if (!exact.covered(level, i, j)) {
            ++local_active;
            result.all_values_finite = result.all_values_finite &&
                std::isfinite(rh(offset(right.grown_box(), i, j))) &&
                std::isfinite(oh(offset(out.grown_box(), i, j)));
            result.reference = std::max(result.reference, std::abs(rh(offset(right.grown_box(), i, j))));
            result.independent_residual = std::max(result.independent_residual,
                std::abs(rh(offset(right.grown_box(), i, j)) - oh(offset(out.grown_box(), i, j))));
            result.maximum_error = std::max(result.maximum_error,
                std::abs(measured.sample(level, i, j) - exact.sample(level, i, j)));
          }
      for (int slot = 0; slot < 4; ++slot) {
        const auto& coefficient = coefficients[level][slot]->fab(local);
        auto host = coefficient.create_host_mirror(); coefficient.copy_to_host(host);
        const auto grown = coefficient.grown_box();
        for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
          for (int i = grown.lo[0]; i <= grown.hi[0]; ++i) {
            const Real value = host(offset(grown, i, j));
            result.all_values_finite = result.all_values_finite && std::isfinite(value);
            result.coefficient_scale = std::max(result.coefficient_scale, std::abs(value));
            result.coefficient_ghost_difference = std::max(result.coefficient_ghost_difference,
                std::abs(value - tensors[slot].sample(level, i, j)));
          }
      }
    }
    result.active_cells[level] = all_reduce_sum(local_active, lane);
  }
  result.all_values_finite = all_reduce_max(result.all_values_finite ? 0L : 1L, lane) == 0;
  result.reference = all_reduce_max(result.reference, lane);
  result.independent_residual = all_reduce_max(result.independent_residual, lane);
  result.maximum_error = all_reduce_max(result.maximum_error, lane);
  result.coefficient_scale = all_reduce_max(result.coefficient_scale, lane);
  result.coefficient_ghost_difference = all_reduce_max(result.coefficient_ghost_difference, lane);
  return result;
}

}  // namespace pops::test
