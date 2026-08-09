// Exact-ranked GPU validation for the currently supported GeometricMG operator family.
//
// One source is compiled for exactly one POPS_NATIVE_DIM specialization.  It solves the
// manufactured Dirichlet problem
//
//   (-Laplacian + reaction) phi = f,
//   phi(x) = product_axis sin(pi x_axis),
//
// on three successively refined Cartesian meshes.  The harness proves that the same generic
// GeometricMG<kNativeDimension> algorithm executes on the selected Kokkos device, converges, and
// retains second-order spatial accuracy.  It deliberately does not claim support for the retired
// variable-diagonal, tensor, or embedded-boundary operator families.

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;
constexpr pops::Real kReaction = pops::Real(12);
using Field = pops::MultiFab<kDim>;
using Solver = pops::elliptic::mg::GeometricMG<kDim>;

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

pops::Index<kDim> index_from_ordinal(const pops::Box<kDim>& box, std::size_t ordinal) {
  pops::Index<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

std::size_t storage_ordinal(const pops::Box<kDim>& storage, const pops::Index<kDim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < kDim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(storage.length(axis));
  }
  return result;
}

pops::EllipticBuildRequest<kDim> make_request(int cells) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  const pops::Geometry<kDim> geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{}, filled<pops::RealVector<kDim>>(pops::Real(1)));
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  const pops::mesh::RankSpace<kDim> rank_space{pops::Index<kDim>{},
                                               filled<pops::Extent<kDim>>(std::int64_t{1})};
  const pops::mesh::Distribution<kDim> distribution =
      pops::mesh::Distribution<kDim>::replicated(layout, rank_space);

  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * kDim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);

  return {geometry,
          layout,
          distribution,
          pops::Index<kDim>{},
          pops::PhysicalBoundaryConditions<kDim>{pops::BoundaryTopology<kDim>::physical(), faces,
                                                 spacing},
          pops::Extent<kDim>{},
          filled<pops::Extent<kDim>>(std::int64_t{1}),
          {layout.size(), 0}};
}

pops::Real manufactured_value(const pops::Geometry<kDim>& geometry,
                              const pops::Index<kDim>& index) {
  const pops::Real pi = std::acos(pops::Real(-1));
  pops::Real result = pops::Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    result *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

void fill_manufactured_rhs(Field& rhs, const pops::Geometry<kDim>& geometry) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real eigenvalue = static_cast<pops::Real>(kDim) * pi * pi + kReaction;
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      host(storage_ordinal(storage, index)) = eigenvalue * manufactured_value(geometry, index);
    }
    fab.copy_from_host(host);
  }
}

struct Result {
  double linf = 0;
  int cycles = 0;
  bool solved = false;
  std::string reason;
  std::vector<double> solution;
};

Result solve_mms(int cells, bool retain_solution) {
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-10);
  options.absolute_tolerance = pops::Real(1e-12);
  options.maximum_cycles = 120;
  options.bottom_sweeps = 60;
  options.reaction = kReaction;

  Solver solver(make_request(cells), options);
  solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                           pops::PreparedVectorDistribution<kDim>::replicated());
  solver.phi().set_val(pops::Real(0));
  fill_manufactured_rhs(solver.rhs(), solver.geom());
  const pops::SolveReport report = solver.solve();
  Kokkos::fence();

  Result result;
  result.cycles = report.iters;
  result.solved = report.solved();
  result.reason = report.reason;
  if (retain_solution)
    result.solution.reserve(static_cast<std::size_t>(solver.geom().domain().numPts()));

  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    const auto& fab = solver.phi().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      const double value = static_cast<double>(host(storage_ordinal(storage, index)));
      result.linf =
          std::max(result.linf,
                   std::abs(value - static_cast<double>(manufactured_value(solver.geom(), index))));
      if (retain_solution)
        result.solution.push_back(value);
    }
  }
  return result;
}

void dump_binary(const std::string& path, const std::vector<double>& values, int& failures) {
  if (path.empty())
    return;
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    std::printf("FAIL cannot open %s\n", path.c_str());
    ++failures;
    return;
  }
  const std::size_t written = std::fwrite(values.data(), sizeof(double), values.size(), file);
  std::fclose(file);
  if (written != values.size()) {
    std::printf("FAIL incomplete dump %s\n", path.c_str());
    ++failures;
    return;
  }
  std::printf("  dump %s (%zu doubles)\n", path.c_str(), values.size());
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  std::string dump_prefix;
  for (int argument = 1; argument < argc; ++argument)
    if (std::strncmp(argv[argument], "--dump=", 7) == 0)
      dump_prefix = argv[argument] + 7;

  int failures = 0;
  const auto check = [&](bool condition, const char* message) {
    if (!condition) {
      std::printf("FAIL %s\n", message);
      ++failures;
    }
  };

  const Result coarse = solve_mms(8, false);
  const Result medium = solve_mms(16, true);
  const Result fine = solve_mms(32, false);
  const double coarse_to_medium = coarse.linf / medium.linf;
  const double medium_to_fine = medium.linf / fine.linf;
  const char* execution_space = Kokkos::DefaultExecutionSpace::name();
  std::printf(
      "[GeometricMG exact] exec=%s dim=%d reaction=%.3g cycles(8/16/32)=%d/%d/%d "
      "Linf=%.17g/%.17g/%.17g ratios=%.4f/%.4f\n",
      execution_space, kDim, static_cast<double>(kReaction), coarse.cycles, medium.cycles,
      fine.cycles, coarse.linf, medium.linf, fine.linf, coarse_to_medium, medium_to_fine);

  check(coarse.solved, coarse.reason.c_str());
  check(medium.solved, medium.reason.c_str());
  check(fine.solved, fine.reason.c_str());
  check(coarse.linf > medium.linf && medium.linf > fine.linf,
        "manufactured error decreases under refinement");
  check(coarse_to_medium > 3.2 && coarse_to_medium < 4.8,
        "second-order ratio from 8 to 16 cells per axis");
  check(medium_to_fine > 3.2 && medium_to_fine < 4.8,
        "second-order ratio from 16 to 32 cells per axis");

  dump_binary(
      dump_prefix.empty() ? "" : dump_prefix + "_d" + std::to_string(kDim) + "_reaction16.bin",
      medium.solution, failures);

  if (failures == 0)
    std::printf("OK gpu_epm_validate (exec=%s dim=%d)\n", execution_space, kDim);
  return failures == 0 ? 0 : 1;
}
