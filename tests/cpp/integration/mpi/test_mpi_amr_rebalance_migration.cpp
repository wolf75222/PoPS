// Accepted-boundary AMR owner migration: a collective RebalanceDecision redistributes one live
// fine level without changing its scientific boxes, clocks, values or regrid counter. The Program
// context must rematerialize topology-qualified history/flux authority and stale or malformed
// decisions must fail before any accepted byte changes.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using namespace pops::runtime::program;

namespace {

ModelSpec exb_spec() {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 1.0;
  spec.B0 = 1.0;
  return spec;
}

std::vector<ResourceEstimate> uniform_estimates(const AmrRuntime& runtime, int level) {
  std::vector<ResourceEstimate> estimates(runtime.level_owner_ranks(level).size());
  for (ResourceEstimate& estimate : estimates) {
    estimate.topology_epoch = runtime.topology_epoch();
    estimate.materialization_generation = runtime.topology_materialization_generation();
    estimate.samples = 1;
    estimate.cell_updates = 1;
    estimate.compute_nanoseconds = 1000;
    estimate.memory_bytes = 64;
    estimate.resident_bytes = 64;
  }
  return estimates;
}

RebalancePolicy migration_policy() {
  RebalancePolicy policy;
  policy.minimum_improvement_ppm = 0;
  policy.amortization_steps = 100;
  policy.migration_bandwidth_bytes_per_second = 1'000'000'000'000LL;
  policy.per_patch_migration_latency_nanoseconds = 0;
  return policy;
}

AmrProgramRankOwnership ownership_snapshot(const AmrRuntime& runtime) {
  AmrProgramRankOwnership ownership;
  ownership.rank_count = n_ranks();
  ownership.level_patch_owners.reserve(static_cast<std::size_t>(runtime.nlev()));
  for (int level = 0; level < runtime.nlev(); ++level)
    ownership.level_patch_owners.push_back(runtime.level_owner_ranks(level));
  return ownership;
}

std::vector<std::vector<std::uint8_t>> gather_program_payloads(
    const std::vector<std::uint8_t>& local) {
  std::string payload;
  payload.reserve(local.size());
  for (const std::uint8_t byte : local)
    payload.push_back(static_cast<char>(byte));
  const std::vector<std::string> gathered = WorldCommunicator::world().allgather_bytes(payload);
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(gathered.size());
  for (const std::string& rank_payload : gathered) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(rank_payload.size());
    for (const char byte : rank_payload)
      bytes.push_back(static_cast<std::uint8_t>(byte));
    result.push_back(std::move(bytes));
  }
  return result;
}

int run_mpi_amr_rebalance_migration(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  const int ranks = n_ranks();
  long failures = 0;
  if (ranks < 2) {
    if (rank == 0)
      std::printf("FAIL test_mpi_amr_rebalance_migration requires at least two ranks\n");
    comm_finalize();
    return 1;
  }

  AmrSystemConfig config;
  config.n = 8;
  config.L = 1.0;
  config.level_count = 2;
  config.regrid_every = 0;
  config.periodicity = {true, true};
  AmrSystem system(config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.add_block("tracer", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
  system.set_poisson("charge_density", "geometric_mg", "periodic");
  std::vector<double> density(static_cast<std::size_t>(config.n * config.n), 1.0);
  for (int j = 0; j < config.n; ++j)
    for (int i = 0; i < config.n; ++i)
      density[static_cast<std::size_t>(j * config.n + i)] +=
          0.1 * std::sin(2.0 * 3.14159265358979323846 * (i + 0.5) / config.n);
  system.set_density("tracer", density);
  test::install_prepared_threshold_union(system, {{"tracer", "n", 1.0e29}});
  const std::vector<PatchBox> fine_boxes{
      {1, 4, 4, 7, 7}, {1, 8, 4, 11, 7}, {1, 4, 8, 7, 11}, {1, 8, 8, 11, 11}};
  const auto context = test::install_forward_euler_program_context(system, [&](AmrSystem& built) {
    built.rebuild_hierarchy(fine_boxes, std::vector<int>(fine_boxes.size(), 0));
  });
  system.step(1.0e-3);

  AmrRuntime& runtime = *system.engine();
  if (runtime.nlev() != 2) {
    ++failures;
  } else {
    constexpr int fine_level = 1;
    const std::vector<double> state_before = system.block_level_state_global("tracer", fine_level);
    const std::uint64_t program_revision_before = system.program_accepted_state_revision();
    const double time_before = system.time();
    const int step_before = system.macro_step();
    const int regrid_before = runtime.regrid_count();
    const std::uint64_t epoch_before = runtime.topology_epoch();
    const std::uint64_t generation_before = runtime.topology_materialization_generation();

    const AmrProgramAcceptedState accepted_before =
        deserialize_amr_program_accepted_state(system.program_accepted_state());
    failures += accepted_before.accepted_flux_ledger.empty();
    failures += accepted_before.accepted_sync.empty();

    RebalanceDecision decision = runtime.decide_rebalance(
        fine_level, uniform_estimates(runtime, fine_level), migration_policy());
    failures += !decision.accepted || decision.reason != RebalanceReason::NetBenefit;
    const std::vector<int> proposed = decision.proposed_mapping.ranks();
    const AmrProgramRankOwnership source_ownership = ownership_snapshot(runtime);
    AmrProgramRankOwnership target_ownership = source_ownership;
    target_ownership.level_patch_owners[static_cast<std::size_t>(fine_level)] = proposed;
    AmrProgramAcceptedState expected_state =
        deserialize_amr_program_accepted_state(rematerialize_amr_program_accepted_state_bytes(
            gather_program_payloads(system.program_accepted_state()), source_ownership,
            target_ownership, rank));
    expected_state.accepted_flux_ledger.clear();
    expected_state.accepted_interface_flux_ledger.clear();
    expected_state.accepted_sync.clear();
    const std::vector<std::uint8_t> expected_program =
        serialize_amr_program_accepted_state(expected_state);
    bool applied = false;
    try {
      applied = context->apply_rebalance_decision(fine_level, decision);
    } catch (const std::exception& error) {
      if (rank == 0)
        std::printf("rebalance migration threw: %s\n", error.what());
      ++failures;
    }
    failures += !applied;
    failures += runtime.level_owner_ranks(fine_level) != proposed;
    failures += runtime.topology_epoch() != epoch_before + 1;
    failures += runtime.topology_materialization_generation() <= generation_before;
    failures += runtime.regrid_count() != regrid_before;
    failures += system.time() != time_before || system.macro_step() != step_before;
    failures += system.block_level_state_global("tracer", fine_level) != state_before;
    failures += context->history_flux_topology_epoch() != runtime.topology_epoch();
    failures += system.program_accepted_state() != expected_program;

    const AmrProgramAcceptedState migrated =
        deserialize_amr_program_accepted_state(system.program_accepted_state());
    failures += migrated.level_clocks.size() != 2;
    failures += !migrated.accepted_flux_ledger.empty();
    failures += !migrated.accepted_interface_flux_ledger.empty();
    failures += !migrated.accepted_sync.empty();
    failures += system.program_accepted_state_revision() != program_revision_before + 1;

    const std::vector<std::uint8_t> stable_program = system.program_accepted_state();
    const std::uint64_t stable_program_revision = system.program_accepted_state_revision();
    const std::uint64_t stable_epoch = runtime.topology_epoch();
    const std::uint64_t stable_generation = runtime.topology_materialization_generation();
    const std::vector<int> stable_owners = runtime.level_owner_ranks(fine_level);
    bool stale_rejected = false;
    try {
      static_cast<void>(context->apply_rebalance_decision(fine_level, decision));
    } catch (const std::invalid_argument&) {
      stale_rejected = true;
    }
    failures += !stale_rejected;
    failures += runtime.topology_epoch() != stable_epoch;
    failures += runtime.topology_materialization_generation() != stable_generation;
    failures += runtime.level_owner_ranks(fine_level) != stable_owners;
    failures += system.program_accepted_state() != stable_program;
    failures += system.program_accepted_state_revision() != stable_program_revision;

    RebalanceDecision malformed = runtime.decide_rebalance(
        fine_level, uniform_estimates(runtime, fine_level), migration_policy());
    malformed.source_contract.push_back('x');
    malformed.exact_contract = pops::detail::exact_rebalance_decision(malformed);
    bool malformed_rejected = false;
    try {
      static_cast<void>(context->apply_rebalance_decision(fine_level, malformed));
    } catch (const std::invalid_argument&) {
      malformed_rejected = true;
    }
    failures += !malformed_rejected;
    failures += runtime.topology_epoch() != stable_epoch;
    failures += runtime.topology_materialization_generation() != stable_generation;
    failures += system.program_accepted_state() != stable_program;
    failures += system.program_accepted_state_revision() != stable_program_revision;

    const RebalanceDecision refusal = runtime.decide_rebalance(
        fine_level, uniform_estimates(runtime, fine_level), migration_policy());
    failures += refusal.accepted || refusal.reason != RebalanceReason::MappingUnchanged;
    try {
      failures += context->apply_rebalance_decision(fine_level, refusal);
    } catch (const std::exception& error) {
      if (rank == 0)
        std::printf("unchanged rebalance refusal threw: %s\n", error.what());
      ++failures;
    }
    failures += runtime.topology_epoch() != stable_epoch;
    failures += runtime.topology_materialization_generation() != stable_generation;
    failures += system.program_accepted_state() != stable_program;
    failures += system.program_accepted_state_revision() != stable_program_revision;

    try {
      system.step(1.0e-3);
    } catch (const std::exception& error) {
      if (rank == 0)
        std::printf("post-rebalance step threw: %s\n", error.what());
      ++failures;
    }
    failures += !(system.time() > time_before) || system.macro_step() <= step_before;
    const std::vector<double> resumed_state = system.block_level_state_global("tracer", fine_level);
    for (const double value : resumed_state)
      failures += !std::isfinite(value);
  }

  failures = all_reduce_sum(failures);
  if (rank == 0)
    std::printf("%s test_mpi_amr_rebalance_migration (np=%d)\n", failures == 0 ? "OK" : "FAIL",
                ranks);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_amr_rebalance_migration, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_mpi_amr_rebalance_migration, "test_mpi_amr_rebalance_migration"),
      0);
}
