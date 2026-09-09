#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>

#include <array>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int Dim = pops::kNativeDimension;
using System = pops::AmrSystem<Dim>;
using namespace pops;
using namespace pops::runtime::system;

struct FieldConsumerModel : nd::ScalarAdvection<Dim> {
  using Base = nd::ScalarAdvection<Dim>;
  using State = typename Base::State;
  static constexpr int n_providers = 3;
  static PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.program-field/three-input-source", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    Base::serialize_exact_parameters(contract);
    contract.text("third+2first-4second");
  }
  POPS_HD State source(const State&, const ProviderValues<3>& values) const {
    State result{};
    result[0] = values[0] + Real(2) * values[1] - Real(4) * values[2];
    return result;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

std::array<AuxiliaryComponentKey, 3> install_outputs(System& system) {
  std::array<AuxiliaryComponentKey, 3> keys{{{"tests.fields", "field", "tuple", "first"},
                                             {"tests.fields", "field", "tuple", "second"},
                                             {"tests.fields", "field", "tuple", "third"}}};
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field-tuple",
                                            "scalar"};
  std::vector<AuxiliaryOutput<Dim>> outputs;
  for (const auto& key : keys)
    outputs.push_back({key, contract, shape});
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.fields/consumed-tuple",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      outputs,
      {}});
  AuxiliaryConsumerProviderPlan<Dim> consumer;
  consumer.consumer_qid = "tests.fields/physical-source";
  // Deliberately permute the compact consumer slots independently of provider storage order.
  consumer.values = {{{keys[2], contract, shape}, 0},
                     {{keys[0], contract, shape}, 1},
                     {{keys[1], contract, shape}, 2}};
  system.install_auxiliary_consumer_plan(std::move(consumer));
  system.seal_auxiliary_providers();
  return keys;
}

runtime::multiblock::BoundaryEvaluationPoint point(int level, int stage = 0) {
  return {.clock = "tests.fields/clock",
          .tick = 0,
          .level = level,
          .substep = 0,
          .stage = stage,
          .stage_fraction = {0, 1},
          .dt = 0.01,
          .physical_time = 0.0};
}

std::vector<System::ProgramFieldLevel> publication(std::vector<MultiFab<Dim>>& fields,
                                                   const std::array<AuxiliaryComponentKey, 3>& keys,
                                                   int stage = 0) {
  std::vector<System::ProgramFieldLevel> result;
  constexpr std::array<int, 3> components{2, 0, 3};
  for (std::size_t level = 0; level < fields.size(); ++level) {
    System::ProgramFieldLevel row;
    row.point = point(static_cast<int>(level), stage);
    for (int i = 0; i < 3; ++i)
      row.components.push_back(
          {keys[i], "tests.fields/consumed-tuple", &fields[level], components[i]});
    result.push_back(std::move(row));
  }
  return result;
}

void fill_tuple(std::vector<MultiFab<Dim>>& fields, Real offset) {
  for (std::size_t level = 0; level < fields.size(); ++level)
    for (std::size_t patch = 0; patch < fields[level].local_size(); ++patch) {
      const auto values = fields[level].fab(patch).view();
      const Real base = offset + Real(10) * Real(level);
      for_each_cell(fields[level].box(patch), [=] POPS_HD(const Index<Dim>& cell) {
        values(cell, 2) = base + Real(1);
        values(cell, 0) = base + Real(2);
        values(cell, 3) = base + Real(3);
        values(cell, 1) = Real(1000);  // unused tuple component must never leak into the source
      });
    }
}

void verify_native_source(System& system, Real offset, int stage = 0) {
  for (int level = 0; level < system.n_levels(); ++level) {
    auto& state = system.prepared_amr_block_state(0, level);
    MultiFab<Dim> rhs(state.layout(), state.distribution(), state.local_rank(), state.ncomp(),
                      state.ghosts());
    rhs.set_val(Real(0));
    system.prepared_amr_block_level_source_into_at(0, point(level, stage), state, rhs);
    const Real expected = Real(-3) - Real(10 * level) - offset;
    Real error = Real(0);
    for (std::size_t patch = 0; patch < rhs.local_size(); ++patch) {
      const auto actual = std::as_const(rhs).fab(patch).view();
      error += for_each_cell_reduce_sum(rhs.box(patch), [=] POPS_HD(const Index<Dim>& cell) {
        return std::abs(actual(cell, 0) - expected);
      });
    }
    EXPECT_NEAR(all_reduce_sum(error), Real(0), Real(1e-11));
  }
}

TEST(AmrProgramFieldPublication, ThreeFieldsReachPreparedSourcesAtEveryLevelAndRollback) {
  AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 0;
  config.periodicity.fill(true);
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    cells *= 8;
    config.transition_buffers.front()[axis] = 0;
    config.transition_lookaheads.front()[axis] = 0;
  }
  System system(config);
  test::install_amr_runtime_authority(system, "tests.fields/publication-runtime");
  const auto keys = install_outputs(system);
  constexpr const char* state_identity = "tests.fields/transport-state";
  system.install_block_state_route("transport", state_identity);
  add_compiled_model<Dim>(system, "transport", FieldConsumerModel{}, "none", "rusanov",
                          "conservative", "euler", static_cast<double>(kPhysicalDefaultGamma), 1, 1,
                          {}, {}, 0.0, static_cast<double>(kWenoEpsilon), false,
                          "tests.fields/physical-source");
  std::vector<double> initial(cells, 1.0);
  initial[cells / 2] = 2.0;
  system.set_conservative_state("transport", initial);
  test::install_prepared_threshold_union(
      system,
      {{"transport", "scalar", 1.5, test::PreparedThresholdRelation::Above, state_identity}},
      "tests.fields/partial-refinement");
  system.set_program_block_map({0});
  system.refresh_prepared_amr_levels();
  ASSERT_EQ(system.n_levels(), 2);
  const auto& coarse_coverage = system.prepared_amr_block_level_coverage_mask(0, 0);
  Real active = Real(0), covered = Real(0);
  for (std::size_t patch = 0; patch < coarse_coverage.local_size(); ++patch) {
    const auto mask = coarse_coverage.fab(patch).view();
    active += for_each_cell_reduce_sum(
        coarse_coverage.box(patch), [=] POPS_HD(const Index<Dim>& cell) { return mask(cell, 0); });
    covered += for_each_cell_reduce_sum(
        coarse_coverage.box(patch),
        [=] POPS_HD(const Index<Dim>& cell) { return Real(1) - mask(cell, 0); });
  }
  ASSERT_GT(all_reduce_sum(active), Real(0));
  ASSERT_GT(all_reduce_sum(covered), Real(0));
  std::vector<MultiFab<Dim>> fields;
  for (int level = 0; level < system.n_levels(); ++level) {
    const auto& state = system.prepared_amr_block_state(0, level);
    fields.emplace_back(state.layout(), state.distribution(), state.local_rank(), 4, Extent<Dim>{});
  }
  fill_tuple(fields, Real(0));
  system.publish_program_field_components("tests.fields/accepted-tuple", publication(fields, keys));
  verify_native_source(system, Real(0));
  // The manifest hashes the complete grown carriers, including coarse/fine and periodic ghosts.
  const auto accepted = system.checkpoint_rank_local_carrier_manifest();
  const auto metadata = system.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(metadata.size(), 2U);

  // Even a no-write rollback rebuilds the graph. It must preserve every accepted halo byte,
  // independently of the publication transaction's own candidate rejection path.
  system.begin_step_transaction();
  system.rollback_step_transaction();
  EXPECT_EQ(system.checkpoint_rank_local_carrier_manifest(), accepted);
  verify_native_source(system, Real(0));

  // The coarse candidate is valid and different; the fine NaN must reject the entire batch.
  fill_tuple(fields, Real(20));
  fields[1].set_val(std::numeric_limits<Real>::quiet_NaN());
  EXPECT_THROW(system.publish_program_field_components("tests.fields/rejected-tuple",
                                                       publication(fields, keys, 1)),
               std::runtime_error);
  EXPECT_EQ(system.checkpoint_rank_local_carrier_manifest(), accepted);
  const auto after_rejection = system.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(after_rejection.size(), metadata.size());
  for (std::size_t level = 0; level < metadata.size(); ++level)
    EXPECT_EQ(serialize_auxiliary_checkpoint_state(after_rejection[level]),
              serialize_auxiliary_checkpoint_state(metadata[level]));
  verify_native_source(system, Real(0));

  fill_tuple(fields, Real(30));
  auto incomplete = publication(fields, keys);
  incomplete.pop_back();
  EXPECT_THROW(system.publish_program_field_components("tests.fields/incomplete", incomplete),
               std::invalid_argument);
  EXPECT_EQ(system.checkpoint_rank_local_carrier_manifest(), accepted);

  // The new path participates in the facade's pre-existing step rollback journal.
  system.begin_step_transaction();
  system.publish_program_field_components("tests.fields/provisional-tuple",
                                          publication(fields, keys, 2));
  verify_native_source(system, Real(30), 2);
  system.rollback_step_transaction();
  EXPECT_EQ(system.checkpoint_rank_local_carrier_manifest(), accepted);
  verify_native_source(system, Real(0));

  // The reconstructed candidate carriers and borrowed consumer views must also support retry.
  system.begin_step_transaction();
  system.publish_program_field_components("tests.fields/retried-tuple",
                                          publication(fields, keys, 2));
  verify_native_source(system, Real(30), 2);
  system.commit_step_transaction();
  system.finalize_step_transaction();
  const auto retried = system.checkpoint_rank_local_carrier_manifest();
  EXPECT_NE(retried, accepted);
  system.begin_step_transaction();
  system.rollback_step_transaction();
  EXPECT_EQ(system.checkpoint_rank_local_carrier_manifest(), retried);
  verify_native_source(system, Real(30), 2);
}
}  // namespace
