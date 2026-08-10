// Exact distributed FFT provider gate for every Cartesian native rank. The historical executable
// name is retained so CI keeps exercising np={1,2,4}. No rank materializes a global field and no
// System remap hides an invalid decomposition.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int kCells = 16;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

static_assert(pops::PoissonFFTCapabilities<1>::available);
static_assert(pops::PoissonFFTCapabilities<2>::available);
static_assert(pops::PoissonFFTCapabilities<3>::available);

template <int Dim>
pops::EllipticBuildRequest<Dim> fft_request(bool canonical_slabs) {
  const int ranks = pops::n_ranks();
  if (ranks < 1 || kCells % ranks != 0)
    throw std::invalid_argument("FFT provider test extent must divide the communicator size");

  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  pops::RealVector<Dim> lower{};
  pops::RealVector<Dim> upper{};
  pops::Extent<Dim> rank_extent{};
  pops::Extent<Dim> rhs_ghosts{};
  pops::Extent<Dim> phi_ghosts{};
  std::array<bool, Dim> periodic{};
  for (int axis = 0; axis < Dim; ++axis) {
    hi[axis] = kCells - 1;
    upper[axis] = pops::Real(1);
    rank_extent[axis] = axis == Dim - 1 ? ranks : 1;
    phi_ghosts[axis] = 1;
    periodic[axis] = true;
  }
  const pops::Box<Dim> domain{lo, hi};
  const pops::Geometry<Dim> geometry = pops::Geometry<Dim>::from_bounds(domain, lower, upper);
  const pops::mesh::RankSpace<Dim> rank_space{lo, rank_extent};

  std::vector<pops::Box<Dim>> boxes;
  std::vector<pops::Index<Dim>> owners;
  if (canonical_slabs) {
    const int local_last = kCells / ranks;
    boxes.reserve(static_cast<std::size_t>(ranks));
    owners.reserve(static_cast<std::size_t>(ranks));
    for (int rank = 0; rank < ranks; ++rank) {
      auto slab_lo = lo;
      auto slab_hi = hi;
      slab_lo[Dim - 1] = rank * local_last;
      slab_hi[Dim - 1] = (rank + 1) * local_last - 1;
      boxes.emplace_back(slab_lo, slab_hi);
      owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
    }
  } else {
    boxes.push_back(domain);
    owners.push_back(rank_space.coordinate(0));
  }

  pops::mesh::BoxArray<Dim> layout(std::move(boxes));
  const pops::mesh::Distribution<Dim> distribution =
      pops::mesh::Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
  std::array<pops::PhysicalBoundaryFace, 2 * Dim> faces{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
  return {geometry,
          std::move(layout),
          distribution,
          rank_space.coordinate(static_cast<std::size_t>(pops::my_rank())),
          pops::PhysicalBoundaryConditions<Dim>{
              pops::BoundaryTopology<Dim>::axis_periodic(periodic), faces, spacing},
          rhs_ghosts,
          phi_ghosts,
          {static_cast<std::size_t>(ranks), pairs}};
}

template <int Dim>
std::size_t storage_ordinal(const pops::Box<Dim>& storage, const pops::Index<Dim>& index) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(storage.length(axis));
  }
  return ordinal;
}

template <int Dim>
pops::Index<Dim> valid_index(const pops::Box<Dim>& valid, std::size_t ordinal) {
  pops::Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(valid.length(axis));
    index[axis] = valid.lo[axis] + static_cast<int>(ordinal % extent);
    ordinal /= extent;
  }
  return index;
}

template <int Dim>
void fill_manufactured_rhs(pops::MultiFab<Dim>& rhs, const pops::Geometry<Dim>& geometry) {
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(valid.numPts());
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      const auto index = valid_index(valid, ordinal);
      host(storage_ordinal(storage, index)) =
          std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(Dim - 1, index[Dim - 1]));
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
pops::Real manufactured_error(const pops::PoissonFFTSolver<Dim>& solver) {
  const pops::Real dx = solver.geom().spacing(Dim - 1);
  const pops::Real theta = pops::Real(2) * kPi / pops::Real(kCells);
  const pops::Real eigenvalue = pops::Real(2) * (pops::Real(1) - std::cos(theta)) / (dx * dx);
  pops::Real local_error = 0;
  const auto& field = solver.phi();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(valid.numPts());
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      const auto index = valid_index(valid, ordinal);
      const pops::Real rhs =
          std::sin(pops::Real(2) * kPi * solver.geom().cell_coordinate(Dim - 1, index[Dim - 1]));
      local_error =
          std::max(local_error, std::abs(host(storage_ordinal(storage, index)) - rhs / eigenvalue));
    }
  }
  return pops::all_reduce_max(local_error);
}

template <int Dim>
bool verify_exact_provider() {
  auto request = fft_request<Dim>(true);
  const auto expected = pops::PoissonFFTSolver<Dim>::expected_operator_contract(request);
  pops::Real cell_measure = pops::Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    cell_measure *= request.geometry.spacing(axis);
  pops::PoissonFFTSolver<Dim> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<Dim>>(
      std::move(request), pops::PoissonFFTFactory<Dim>{});
  solver.install_nullspace(pops::constant_mean_zero_nullspace<Dim>(
                               "periodic-mpi-fft", "test-mpi-system-fft", cell_measure),
                           pops::PreparedVectorDistribution<Dim>::distributed());
  bool valid =
      solver.prepared_operator_contract().exact_fingerprint() == expected.exact_fingerprint();

  fill_manufactured_rhs(solver.rhs(), solver.geom());
  const pops::SolveReport first = solver.solve();
  const pops::Real first_error = manufactured_error(solver);
  const pops::SolveReport second = solver.solve();
  const pops::Real second_error = manufactured_error(solver);
  valid = valid && first.solved() && second.solved() && first.residual_norm < pops::Real(1e-9) &&
          second.residual_norm < pops::Real(1e-9) && first_error < pops::Real(1e-12) &&
          second_error < pops::Real(1e-12);

  if (pops::n_ranks() > 1) {
    bool rejected = false;
    try {
      auto remap_request = fft_request<Dim>(false);
      auto forbidden = pops::make_elliptic_solver<pops::PoissonFFTSolver<Dim>>(
          std::move(remap_request), pops::PoissonFFTFactory<Dim>{});
      (void)forbidden;
    } catch (const std::exception&) {
      rejected = true;
    }
    valid = valid && rejected;
  }
  return valid;
}

using NativeSystem = pops::System<pops::kNativeDimension>;
using NativeSystemConfig = pops::SystemConfig<pops::kNativeDimension>;
using NativeGas = pops::nd::IdealGasEuler<pops::kNativeDimension>;
using NativeModel = pops::CompositeModel<NativeGas, pops::NoSource, pops::NoElliptic>;

NativeSystemConfig system_fft_config() {
  constexpr int Dim = pops::kNativeDimension;
  const int ranks = pops::n_ranks();
  NativeSystemConfig config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = kCells;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  if (kCells % ranks != 0)
    throw std::invalid_argument("System FFT test extent must divide the communicator size");
  const pops::Box<Dim> domain = config.index_domain();
  const int local_last = kCells / ranks;
  config.boxes.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank) {
    auto lo = domain.lo;
    auto hi = domain.hi;
    lo[Dim - 1] = rank * local_last;
    hi[Dim - 1] = (rank + 1) * local_last - 1;
    config.boxes.emplace_back(lo, hi);
  }
  return config;
}

bool verify_system_fft_materialization() {
  constexpr int Dim = pops::kNativeDimension;
  NativeSystem system(system_fft_config());
  constexpr const char* slot = "test.mpi-system-fft/field";
  constexpr const char* field = "fft-potential";
  NativeModel model{};
  model.hyp = NativeGas::prepare(pops::Real(1.4));
  system.install_block_state_route("gas", "test.mpi-system-fft/gas/state@1");
  pops::add_compiled_model(system, "gas", std::move(model), "minmod", "rusanov", "conservative",
                           "explicit", 1.4);
  bool rejected_nonempty_options = false;
  try {
    system.register_configured_field_solver_provider(
        "fft", "test.mpi-system-fft/invalid-options",
        pops::PreparedProviderOptions{"pops.system.fft-discrete-exact-rank-options.empty@2",
                                      {{"unexpected", 1.0}}});
  } catch (const std::invalid_argument&) {
    rejected_nonempty_options = true;
  }
  if (!rejected_nonempty_options)
    return false;
  system.register_configured_field_solver_provider(
      "fft", slot,
      pops::PreparedProviderOptions{"pops.system.fft-discrete-exact-rank-options.empty@2", {}});
  system.set_field_solver_plan(slot, "test.mpi-system-fft/plan@1", "test.mpi-system-fft/rhs@1",
                               "test.mpi-system-fft/output@1", "gas", field,
                               {"test.mpi-system-fft/rhs@1"}, {"gas"}, {"charge"}, {1.0}, slot);
  system.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                      "test.mpi-system-fft/periodic",
                                      "test.mpi-system-fft/topology@1");
  const std::vector<std::string> periodic_faces(static_cast<std::size_t>(2 * Dim), "periodic");
  const std::vector<double> zero_faces(static_cast<std::size_t>(2 * Dim), 0.0);
  system.set_field_boundary_plan(slot, periodic_faces, zero_faces, zero_faces, zero_faces);

  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentKey output_key{"test.mpi-system-fft", "field", field, "potential"};
  const AuxiliaryComponentContract output_contract{"cell-average", "cell", "unitless", "field",
                                                   "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.mpi-system-fft/field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{output_key, output_contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  system.register_elliptic_field("gas", field, {output_key}, 1);
  system.set_block_elliptic_field(
      "gas", field, [](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) {
        for (std::size_t local = 0; local < rhs.local_size(); ++local) {
          const pops::Box<Dim> valid = rhs.fab(local).box();
          const auto view = rhs.fab(local).view();
          pops::for_each_cell(valid, [=] POPS_HD(const pops::Index<Dim>& cell) {
            const pops::Real coordinate =
                (static_cast<pops::Real>(cell[Dim - 1]) + pops::Real(0.5)) /
                static_cast<pops::Real>(kCells);
            view(cell, 0) = Kokkos::sin(pops::Real(2) * kPi * coordinate);
          });
        }
      });

  const std::size_t cells = static_cast<std::size_t>(std::pow(kCells, Dim));
  std::vector<double> state(static_cast<std::size_t>(NativeModel::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[cell] = 1.0;
    state[static_cast<std::size_t>(Dim + 1) * cells + cell] = 2.5;
  }
  system.set_state("gas", state);
  const pops::SolveReport report =
      pops::consume_solve_outcome(system.solve_fields_from_state(field, 0, system.block_state(0)));
  const std::vector<double> potential = system.auxiliary_component(output_key);
  bool finite_nonzero = !potential.empty();
  double maximum = 0.0;
  for (const double value : potential) {
    finite_nonzero = finite_nonzero && std::isfinite(value);
    maximum = std::max(maximum, std::abs(value));
  }
  return report.solved() && report.evaluations == 1 && finite_nonzero && maximum > 1.0e-8;
}

int run_exact_mpi_fft_provider(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  pops::reset_poisson_fft_direct_dft_fallback_count();
  long local_failures = 0;
  local_failures += verify_exact_provider<1>() ? 0 : 1;
  local_failures += verify_exact_provider<2>() ? 0 : 1;
  local_failures += verify_exact_provider<3>() ? 0 : 1;
  local_failures += verify_system_fft_materialization() ? 0 : 1;
  local_failures += pops::poisson_fft_direct_dft_fallback_count() == 0 ? 0 : 1;
  const long failures = pops::all_reduce_sum(local_failures);
  if (failures == 0 && pops::my_rank() == 0)
    std::printf("OK test_mpi_system_fft exact provider 1D/2D/3D (np=%d)\n", pops::n_ranks());
  pops::comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_system_fft, exact_ranked_provider_has_no_single_box_remap) {
  EXPECT_EQ(pops::test::RunTestBody(&run_exact_mpi_fft_provider, "test_mpi_system_fft"), 0);
}
