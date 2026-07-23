#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

ModelSpec analytic_scalar_model() {
  ModelSpec model;
  model.transport = "exb";
  model.source = "none";
  model.elliptic = "charge";
  return model;
}

int run_analytic_level_set_collective_preflight(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  long local_failures = 0;
  const auto require = [&local_failures, rank](bool condition, const char* message) {
    if (!condition) {
      std::cerr << "analytic level-set MPI check failed on rank " << rank << ": " << message
                << '\n';
      ++local_failures;
    }
  };

  require(n_ranks() == 2, "the regression must run with exactly two MPI ranks");

  // A Cartesian System currently owns one global patch, distributed round-robin to rank zero. This
  // probe mirrors that exact layout and proves that only one rank samples the invalid expression;
  // the other rank can reject only through the native collective preflight.
  const Box2D domain = Box2D::from_extents(20, 20);
  MultiFab ownership_probe(BoxArray(std::vector<Box2D>{domain}),
                           DistributionMapping(1, n_ranks()), 1, 1);
  const long local_sampler = ownership_probe.local_size() == 1 ? 1L : 0L;
  require(all_reduce_sum(local_sampler) == 1,
          "the invalid expression must be sampled on exactly one rank");
  require(local_sampler == (rank == 0 ? 1L : 0L),
          "the single Cartesian patch must be owned by rank zero");

  System system(SystemConfig{20, 1.0, Periodicity{false, false}});
  system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0},
                                "staircase", 0.2, 1e-5, 0.1);
  const std::vector<double> before = system.disc_mask();
  const auto active = std::count(before.begin(), before.end(), 1.0);
  require(active > 0 && active < static_cast<std::ptrdiff_t>(before.size()),
          "the committed reference mask must contain active and inactive cells");

  // Both requests are locally valid, but rank one changes the geometry mode and one exact binary64
  // literal. The collective request preflight must reject before replacing the committed mask.
  bool geometry_mismatch_rejected = false;
  std::string geometry_mismatch_message;
  try {
    system.set_analytic_level_set(
        {"x", "constant", "sub"}, {0.0, rank == 0 ? 0.45 : 0.55, 0.0},
        rank == 0 ? "staircase" : "cutcell", 0.2, 1e-5, 0.1);
  } catch (const std::runtime_error& error) {
    geometry_mismatch_rejected = true;
    geometry_mismatch_message = error.what();
  }
  require(all_reduce_sum(geometry_mismatch_rejected ? 1L : 0L) == n_ranks(),
          "all ranks must reject a rank-dependent analytic geometry request");
  require(geometry_mismatch_rejected &&
              geometry_mismatch_message.find("differs across MPI ranks") != std::string::npos,
          "rank-dependent geometry rejection must identify exact MPI disagreement");
  const std::vector<double> after_geometry_mismatch = system.disc_mask();
  require(all_reduce_sum(after_geometry_mismatch == before ? 0L : 1L) == 0,
          "rank-dependent geometry rejection must preserve the committed mask");

  // (x - x) / 0 is non-finite at every sampled point. Only rank zero owns sample points, yet every
  // rank must receive the same domain_error after the native MPI all-reduce, without deadlock.
  const std::vector<std::string> invalid_ops{"x", "x", "sub", "constant", "div"};
  const std::vector<double> invalid_literals{0.0, 0.0, 0.0, 0.0, 0.0};
  bool rejected = false;
  std::string rejection_message;
  try {
    system.set_analytic_level_set(invalid_ops, invalid_literals, "cutcell", 0.3, 2e-5, 0.2);
  } catch (const std::domain_error& error) {
    rejected = true;
    rejection_message = error.what();
  } catch (const std::exception& error) {
    rejection_message = std::string("wrong exception type: ") + error.what();
  }

  require(all_reduce_sum(rejected ? 1L : 0L) == n_ranks(),
          "all ranks must reject the rank-local non-finite sample");
  require(rejected && rejection_message.find("non-finite") != std::string::npos,
          "every rank must report the analytic finite-value contract");

  const std::vector<double> after = system.disc_mask();
  const long local_partial_publication = before == after ? 0L : 1L;
  require(all_reduce_sum(local_partial_publication) == 0,
          "failed replacement must preserve the previously committed global mask on every rank");

  // Every rank supplies a locally valid scalar program, but rank one changes one binary64 literal.
  // Exact request consensus must reject before either rank writes the System state.
  System expression_system(SystemConfig{12, 1.0, Periodicity{true, true}});
  expression_system.add_block("plasma", analytic_scalar_model());
  bool mismatch_rejected = false;
  std::string mismatch_message;
  try {
    expression_system.set_analytic_expression_state(
        "plasma", "cell", "cell", "conservative_cell_average", {{"constant"}},
        {{rank == 0 ? 0.25 : 0.5}});
  } catch (const std::runtime_error& error) {
    mismatch_rejected = true;
    mismatch_message = error.what();
  }
  require(all_reduce_sum(mismatch_rejected ? 1L : 0L) == n_ranks(),
          "all ranks must reject a rank-dependent analytic state payload");
  require(mismatch_rejected &&
              mismatch_message.find("differs across MPI ranks") != std::string::npos,
          "rank-dependent analytic state rejection must identify exact MPI disagreement");

  // Rejection is pre-publication: the same System remains usable by an identical request.
  bool valid_state_installed = true;
  try {
    expression_system.set_analytic_expression_state(
        "plasma", "cell", "cell", "conservative_cell_average", {{"constant"}}, {{0.75}});
  } catch (const std::exception& error) {
    valid_state_installed = false;
    std::cerr << "valid analytic state failed after mismatch on rank " << rank << ": "
              << error.what() << '\n';
  }
  require(all_reduce_sum(valid_state_installed ? 1L : 0L) == n_ranks(),
          "a rejected rank mismatch must not poison later System materialization");

  // The AMR registration path has no halo yet, but it feeds later collective hierarchy setup. Rank
  // one supplies an unknown opcode while rank zero has a valid program: both ranks must leave the
  // registration without publishing either the provider or its block binding.
  AmrSystemConfig amr_config;
  amr_config.n = 12;
  amr_config.L = 1.0;
  amr_config.level_count = 1;
  amr_config.explicit_bootstrap = true;
  amr_config.regrid_every = 0;
  AmrSystem amr(amr_config);
  amr.add_block("plasma", analytic_scalar_model());
  const std::string subject = "case::plasma::state::U";
  amr.register_bootstrap_transfer_route(
      "case::plasma::initial::prolongation", {subject}, "test::analytic-provider", "cell",
      "cell", "conservative", "dense", "prolongation", "conservative_linear", 2, {1}, 2,
      2);
  bool malformed_rejected = false;
  std::string malformed_message;
  try {
    amr.register_analytic_expression(
        subject, "plasma", "cell", "cell",
        {{rank == 0 ? "constant" : "not-an-analytic-opcode"}}, {{1.0}});
  } catch (const std::runtime_error& error) {
    malformed_rejected = true;
    malformed_message = error.what();
  }
  require(all_reduce_sum(malformed_rejected ? 1L : 0L) == n_ranks(),
          "one malformed local AMR payload must reject collectively on every rank");
  require(malformed_rejected &&
              malformed_message.find("rank-local analytic validation failed collectively") !=
                  std::string::npos,
          "collective AMR rejection must identify a rank-local validation failure");

  bool valid_amr_registered = true;
  try {
    amr.register_analytic_expression(subject, "plasma", "cell", "cell", {{"constant"}},
                                     {{1.0}});
  } catch (const std::exception& error) {
    valid_amr_registered = false;
    std::cerr << "valid AMR analytic registration failed after malformed payload on rank " << rank
              << ": " << error.what() << '\n';
  }
  require(all_reduce_sum(valid_amr_registered ? 1L : 0L) == n_ranks(),
          "a rejected local AMR error must not leak a partial registration");

  const long failures = all_reduce_sum(local_failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_system_analytic_level_set, CollectiveAnalyticRequestsRejectBeforePublication) {
  EXPECT_EQ(pops::test::RunTestBody(&run_analytic_level_set_collective_preflight,
                                    "test_mpi_system_analytic_level_set"),
            0);
}
