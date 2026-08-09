#include <gtest/gtest.h>

#include <pops/runtime/system/exact_aux_registry.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pops::ExactContractBuilder;
using pops::PreparedProvider;
using pops::PreparedProviderIdentity;
using pops::runtime::system::AuxiliaryCentering;
using pops::runtime::system::AuxiliaryComponentContract;
using pops::runtime::system::AuxiliaryComponentKey;
using pops::runtime::system::AuxiliaryDependency;
using pops::runtime::system::AuxiliaryEvaluationEvent;
using pops::runtime::system::AuxiliaryEvaluationPoint;
using pops::runtime::system::AuxiliaryEvaluationPolicy;
using pops::runtime::system::AuxiliaryFreshness;
using pops::runtime::system::AuxiliaryOutput;
using pops::runtime::system::AuxiliaryProviderKind;
using pops::runtime::system::AuxiliarySpaceKind;
using pops::runtime::system::AuxiliaryValueKind;
using pops::runtime::system::ExactAuxiliaryRegistry;
using pops::runtime::system::PreparedAuxiliaryProvider;

template <int Dim>
AuxiliaryComponentContract<Dim> contract(std::string layout = "compact") {
  AuxiliaryComponentContract<Dim> result;
  result.representation = "cell-average";
  result.centering = AuxiliaryCentering::cell;
  result.unit = "unitless";
  result.layout = std::move(layout);
  result.value_kind = AuxiliaryValueKind::scalar;
  result.value_components = 1;
  result.spatial_rank = Dim;
  for (int axis = 0; axis < Dim; ++axis)
    result.halo[axis] = 1;
  return result;
}

inline AuxiliaryComponentKey key(std::string owner, std::string space, int component) {
  return {std::move(owner), AuxiliarySpaceKind::auxiliary, std::move(space), component};
}

template <int Dim>
AuxiliaryOutput<Dim> output(std::string owner, std::string space, int component, std::size_t slot) {
  return {key(std::move(owner), std::move(space), component), contract<Dim>(), slot};
}

template <int Dim>
AuxiliaryDependency<Dim> dependency(const AuxiliaryOutput<Dim>& output) {
  return {output.key, output.contract};
}

template <int Dim>
struct RecordingNativeLaunch {
  std::shared_ptr<std::vector<std::string>> calls;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.exact-aux.native-launch", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& exact) const {
    exact.text("test.exact-aux.native-launch").scalar(std::uint32_t{1});
  }

  void operator()(const pops::runtime::system::AuxiliaryKernelLaunchContext<Dim>& context) const {
    calls->push_back(context.point.clock + ":" + std::to_string(context.candidate_generation));
  }
};

template <int Dim>
PreparedAuxiliaryProvider<Dim> input(std::string identity, AuxiliaryOutput<Dim> result,
                                     AuxiliaryEvaluationPolicy policy = {
                                         AuxiliaryEvaluationEvent::initialization,
                                         AuxiliaryFreshness::once}) {
  return {std::move(identity), AuxiliaryProviderKind::input, policy, {std::move(result)}, {}};
}

template <int Dim>
PreparedAuxiliaryProvider<Dim> derived(std::string identity, AuxiliaryOutput<Dim> result,
                                       std::vector<AuxiliaryDependency<Dim>> dependencies,
                                       std::shared_ptr<std::vector<std::string>> calls,
                                       AuxiliaryEvaluationPolicy policy = {
                                           AuxiliaryEvaluationEvent::before_residual,
                                           AuxiliaryFreshness::evaluation}) {
  using Provider = PreparedAuxiliaryProvider<Dim>;
  using Launcher = typename Provider::launcher_type;
  return {std::move(identity),
          AuxiliaryProviderKind::derived,
          policy,
          {std::move(result)},
          std::move(dependencies),
          Launcher(RecordingNativeLaunch<Dim>{std::move(calls)})};
}

AuxiliaryEvaluationPoint point(std::string clock, std::uint64_t accepted_step,
                               AuxiliaryEvaluationEvent event, std::uint64_t layout_generation = 0,
                               int level = 0, int substep = 0, int stage = 0) {
  return {std::move(clock), accepted_step, layout_generation, level, substep, stage, event};
}

template <int Dim>
void verifies_empty_and_compact_registry() {
  ExactAuxiliaryRegistry<Dim> empty;
  empty.seal();
  EXPECT_EQ(empty.provider_count(), 0U);
  EXPECT_EQ(empty.slot_count(), 0U);
  EXPECT_TRUE(empty.topological_order().empty());

  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("input-a", output<Dim>("owner-a", "space-a", 0, 0)));
  registry.add(input<Dim>("input-b", output<Dim>("owner-b", "space-b", 3, 1)));
  registry.seal();
  EXPECT_EQ(registry.provider_count(), 2U);
  EXPECT_EQ(registry.slot_count(), 2U);
  EXPECT_EQ(registry.slot_of(key("owner-b", "space-b", 3)), 1U);
  EXPECT_THROW(registry.add(input<Dim>("late", output<Dim>("late", "space", 0, 2))),
               std::logic_error);
}

TEST(ExactAuxiliaryRegistryNd, EmptyAndCompactPacksAreRankGeneric) {
  verifies_empty_and_compact_registry<1>();
  verifies_empty_and_compact_registry<2>();
  verifies_empty_and_compact_registry<3>();
}

template <int Dim>
void verifies_structural_rejections() {
  const auto first = output<Dim>("owner", "space", 0, 0);
  ExactAuxiliaryRegistry<Dim> duplicate;
  duplicate.add(input<Dim>("first", first));
  duplicate.add(input<Dim>("second", {first.key, first.contract, 1}));
  EXPECT_THROW(duplicate.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<Dim> sparse;
  sparse.add(input<Dim>("sparse", output<Dim>("owner", "sparse", 0, 1)));
  EXPECT_THROW(sparse.seal(), std::invalid_argument);

  auto calls = std::make_shared<std::vector<std::string>>();
  const auto left = output<Dim>("owner", "cycle", 0, 0);
  const auto right = output<Dim>("owner", "cycle", 1, 1);
  ExactAuxiliaryRegistry<Dim> cyclic;
  cyclic.add(derived<Dim>("left", left, {dependency(right)}, calls));
  cyclic.add(derived<Dim>("right", right, {dependency(left)}, calls));
  EXPECT_THROW(cyclic.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<Dim> missing;
  missing.add(derived<Dim>("missing", output<Dim>("owner", "missing", 0, 0),
                           {{key("missing-owner", "missing-space", 3), contract<Dim>()}}, calls));
  EXPECT_THROW(missing.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<Dim> mismatch;
  mismatch.add(input<Dim>("source", output<Dim>("owner", "mismatch", 0, 0)));
  auto wrong = contract<Dim>("different-layout");
  mismatch.add(derived<Dim>("consumer", output<Dim>("owner", "mismatch", 1, 1),
                            {{key("owner", "mismatch", 0), wrong}}, calls));
  EXPECT_THROW(mismatch.seal(), std::invalid_argument);

  auto invalid_rank = contract<Dim>();
  invalid_rank.spatial_rank = Dim + 1;
  EXPECT_THROW((PreparedAuxiliaryProvider<Dim>{
                   "invalid-rank",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                   {{key("owner", "bad-rank", 0), invalid_rank, 0}},
                   {}}),
               std::invalid_argument);
}

TEST(ExactAuxiliaryRegistryNd, RejectsDuplicateCycleMissingMismatchSparseAndRank) {
  verifies_structural_rejections<1>();
  verifies_structural_rejections<2>();
  verifies_structural_rejections<3>();
}

template <int Dim>
void verifies_topology_and_transaction() {
  auto calls = std::make_shared<std::vector<std::string>>();
  const auto seed = output<Dim>("owner", "data", 0, 0);
  const auto result = output<Dim>("owner", "data", 1, 1);
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("seed", seed));
  registry.add(derived<Dim>("derive", result, {dependency(seed)}, calls));
  registry.seal();
  ASSERT_EQ(registry.topological_order(), (std::vector<std::size_t>{0, 1}));

  {
    auto candidate = registry.begin_publication(
        point("macro", 0, AuxiliaryEvaluationEvent::initialization, 4, 2, 3, 5));
    EXPECT_EQ(candidate.candidate_generation(), 1U);
    candidate.stage_external("seed");
    candidate.launch_ready_native();
    EXPECT_TRUE(calls->empty());
    candidate.reject();
  }
  EXPECT_EQ(registry.accepted_generation(), 0U);
  EXPECT_FALSE(registry.last_accepted_point("seed").has_value());

  {
    auto candidate = registry.begin_publication(
        point("macro", 0, AuxiliaryEvaluationEvent::initialization, 4, 2, 3, 5));
    candidate.stage_external("seed");
    candidate.accept();
  }
  EXPECT_EQ(registry.accepted_generation(), 1U);
  ASSERT_TRUE(registry.last_accepted_point("seed").has_value());
  EXPECT_EQ(*registry.last_accepted_point("seed"),
            point("macro", 0, AuxiliaryEvaluationEvent::initialization, 4, 2, 3, 5));

  const auto residual_point =
      point("macro", 1, AuxiliaryEvaluationEvent::before_residual, 4, 2, 3, 6);
  {
    auto candidate = registry.begin_publication(residual_point);
    candidate.launch_ready_native();
    ASSERT_EQ(calls->size(), 1U);
    EXPECT_EQ((*calls)[0], "macro:2");
    candidate.accept();
  }
  EXPECT_EQ(registry.accepted_generation(), 2U);
  ASSERT_TRUE(registry.last_accepted_point("derive").has_value());
  EXPECT_EQ(*registry.last_accepted_point("derive"), residual_point);

  EXPECT_NO_THROW(registry.require_collective_contract(registry.collective_contract()));
  EXPECT_THROW(registry.require_collective_contract("not-the-same-contract"), std::runtime_error);
}

TEST(ExactAuxiliaryRegistryNd, OrdersNativeLaunchesAndPublishesOnlyCompleteCandidates) {
  verifies_topology_and_transaction<1>();
  verifies_topology_and_transaction<2>();
  verifies_topology_and_transaction<3>();
}

TEST(ExactAuxiliaryRegistryNd, RejectsInvalidProviderClassAndHaloBeforePublication) {
  auto calls = std::make_shared<std::vector<std::string>>();
  EXPECT_THROW((PreparedAuxiliaryProvider<2>{
                   "derived-without-launch",
                   AuxiliaryProviderKind::derived,
                   {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::evaluation},
                   {output<2>("owner", "derived", 0, 0)},
                   {}}),
               std::invalid_argument);
  EXPECT_THROW((PreparedAuxiliaryProvider<2>{
                   "input-with-launch",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                   {output<2>("owner", "input", 0, 0)},
                   {},
                   PreparedAuxiliaryProvider<2>::launcher_type(RecordingNativeLaunch<2>{calls})}),
               std::invalid_argument);

  auto bad_halo = contract<3>();
  bad_halo.halo[2] = -1;
  EXPECT_THROW((PreparedAuxiliaryProvider<3>{
                   "negative-halo",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                   {{key("owner", "halo", 0), bad_halo, 0}},
                   {}}),
               std::invalid_argument);
}

}  // namespace
