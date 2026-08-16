#include <gtest/gtest.h>

#include <pops/runtime/system/exact_aux_registry.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pops::ExactContractBuilder;
using pops::PreparedProviderIdentity;
using pops::runtime::system::AuxiliaryComponentContract;
using pops::runtime::system::AuxiliaryBoundaryPolicy;
using pops::runtime::system::AuxiliaryConsumerProviderPlan;
using pops::runtime::system::AuxiliaryConsumerValue;
using pops::runtime::system::AuxiliaryComponentKey;
using pops::runtime::system::AuxiliaryDependency;
using pops::runtime::system::AuxiliaryEvaluationEvent;
using pops::runtime::system::AuxiliaryEvaluationPoint;
using pops::runtime::system::AuxiliaryEvaluationPolicy;
using pops::runtime::system::AuxiliaryFreshness;
using pops::runtime::system::AuxiliaryOutput;
using pops::runtime::system::AuxiliaryProviderKind;
using pops::runtime::system::AuxiliaryStorageShape;
using pops::runtime::system::ExactAuxiliaryRegistry;
using pops::runtime::system::PreparedAuxiliaryProvider;

inline AuxiliaryComponentContract contract(std::string layout = "compact") {
  AuxiliaryComponentContract result;
  result.representation = "cell-average";
  result.centering = "cell";
  result.unit = "unitless";
  result.layout = std::move(layout);
  result.value_kind = "scalar";
  return result;
}

template <int Dim>
AuxiliaryStorageShape<Dim> shape() {
  AuxiliaryStorageShape<Dim> result;
  result.spatial_rank = Dim;
  result.value_components = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result.halo[axis] = 1;
  return result;
}

inline AuxiliaryComponentKey key(std::string owner_qid, std::string space_kind,
                                 std::string space_name, std::string component) {
  return {std::move(owner_qid), std::move(space_kind), std::move(space_name), std::move(component)};
}

template <int Dim>
AuxiliaryOutput<Dim> output(std::string owner_qid, std::string space_name, std::string component,
                            std::size_t /*package_local_order*/) {
  return {key(std::move(owner_qid), "auxiliary", std::move(space_name), std::move(component)),
          contract(), shape<Dim>()};
}

template <int Dim>
AuxiliaryDependency<Dim> dependency(const AuxiliaryOutput<Dim>& output) {
  return {output.key, output.contract, output.shape};
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
  return {std::move(clock), accepted_step, layout_generation, level, substep, stage, 0, event};
}

template <int Dim>
void verifies_empty_and_compact_registry() {
  ExactAuxiliaryRegistry<Dim> empty;
  empty.seal();
  EXPECT_EQ(empty.provider_count(), 0U);
  EXPECT_EQ(empty.slot_count(), 0U);
  EXPECT_TRUE(empty.topological_order().empty());

  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("input-a", output<Dim>("owner-a", "space-a", "0", 0)));
  registry.add(input<Dim>("input-b", output<Dim>("owner-b", "space-b", "3", 1)));
  registry.seal();
  EXPECT_EQ(registry.provider_count(), 2U);
  EXPECT_EQ(registry.slot_count(), 2U);
  EXPECT_EQ(registry.address_of(key("owner-b", "auxiliary", "space-b", "3")).component, 1U);
  EXPECT_THROW(registry.add(input<Dim>("late", output<Dim>("late", "space", "0", 2))),
               std::logic_error);
}

TEST(ExactAuxiliaryRegistryNd, EmptyAndCompactPacksAreRankGeneric) {
  verifies_empty_and_compact_registry<1>();
  verifies_empty_and_compact_registry<2>();
  verifies_empty_and_compact_registry<3>();
}

template <int Dim>
void verifies_structural_rejections() {
  const auto first = output<Dim>("owner", "space", "0", 0);
  ExactAuxiliaryRegistry<Dim> duplicate;
  duplicate.add(input<Dim>("first", first));
  duplicate.add(input<Dim>("second", {first.key, first.contract, first.shape}));
  EXPECT_THROW(duplicate.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<Dim> independent_packages;
  // Both independently generated packages naturally start their local declaration order at zero.
  // The global registry must assign compact storage from ComponentKeys rather than reject that.
  independent_packages.add(input<Dim>("package-a", output<Dim>("owner-a", "space", "0", 0)));
  independent_packages.add(input<Dim>("package-b", output<Dim>("owner-b", "space", "0", 0)));
  EXPECT_NO_THROW(independent_packages.seal());
  EXPECT_EQ(independent_packages.slot_count(), 2U);

  auto calls = std::make_shared<std::vector<std::string>>();
  const auto left = output<Dim>("owner", "cycle", "0", 0);
  const auto right = output<Dim>("owner", "cycle", "1", 1);
  ExactAuxiliaryRegistry<Dim> cyclic;
  cyclic.add(derived<Dim>("left", left, {dependency(right)}, calls));
  cyclic.add(derived<Dim>("right", right, {dependency(left)}, calls));
  EXPECT_THROW(cyclic.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<Dim> missing;
  missing.add(derived<Dim>(
      "missing", output<Dim>("owner", "missing", "0", 0),
      {{key("missing-owner", "auxiliary", "missing-space", "3"), contract(), shape<Dim>()}},
      calls));
  EXPECT_THROW(missing.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<Dim> mismatch;
  mismatch.add(input<Dim>("source", output<Dim>("owner", "mismatch", "0", 0)));
  auto wrong = contract("different-layout");
  mismatch.add(derived<Dim>("consumer", output<Dim>("owner", "mismatch", "1", 1),
                            {{key("owner", "auxiliary", "mismatch", "0"), wrong, shape<Dim>()}},
                            calls));
  EXPECT_THROW(mismatch.seal(), std::invalid_argument);

  auto invalid_rank = shape<Dim>();
  invalid_rank.spatial_rank = Dim + 1;
  EXPECT_THROW((PreparedAuxiliaryProvider<Dim>{
                   "invalid-rank",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                   {{key("owner", "auxiliary", "bad-rank", "0"), contract(), invalid_rank}},
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
  const auto seed = output<Dim>("owner", "data", "0", 0);
  const auto result = output<Dim>("owner", "data", "1", 1);
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("seed", seed));
  registry.add(derived<Dim>("derive", result, {dependency(seed)}, calls));
  registry.seal();
  ASSERT_EQ(registry.topological_order().size(), 2U);
  EXPECT_EQ(registry.provider(registry.topological_order()[0]).identity(), "seed");
  EXPECT_EQ(registry.provider(registry.topological_order()[1]).identity(), "derive");

  {
    auto candidate = registry.begin_publication(
        point("macro", 0, AuxiliaryEvaluationEvent::initialization, 4, 2, 3, 5));
    EXPECT_EQ(candidate.candidate_generation(), 1U);
    candidate.stage_external("seed");
    candidate.launch_ready_native();
    EXPECT_EQ(calls->size(), 1U);
    candidate.reject();
  }
  EXPECT_EQ(registry.accepted_generation(), 0U);
  EXPECT_FALSE(registry.last_accepted_point("seed").has_value());

  {
    auto candidate = registry.begin_publication(
        point("macro", 0, AuxiliaryEvaluationEvent::initialization, 4, 2, 3, 5));
    candidate.stage_external("seed");
    candidate.launch_ready_native();
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
    ASSERT_EQ(calls->size(), 3U);
    EXPECT_EQ((*calls)[2], "macro:2");
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

template <int Dim>
void verifies_explicit_dirty_input_uses_transaction_authority() {
  const auto value = output<Dim>("owner", "restaged", "value", 0);
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("restaged-input", value));
  registry.seal();

  {
    auto candidate = registry.begin_publication(
        point("initial", 0, AuxiliaryEvaluationEvent::initialization, 0, 0, 0, 0));
    EXPECT_TRUE(candidate.requires_staging("restaged-input"));
    candidate.stage_external("restaged-input");
    candidate.accept();
  }
  {
    auto candidate = registry.begin_publication(
        point("unchanged", 1, AuxiliaryEvaluationEvent::before_residual, 0, 0, 0, 1));
    EXPECT_FALSE(candidate.requires_staging("restaged-input"));
    candidate.accept();
  }
  {
    auto candidate = registry.begin_publication(
        point("restaged", 2, AuxiliaryEvaluationEvent::before_residual, 0, 0, 0, 2),
        {"restaged-input"});
    EXPECT_TRUE(candidate.requires_staging("restaged-input"));
    candidate.stage_external("restaged-input");
    candidate.accept();
  }
}

TEST(ExactAuxiliaryRegistryNd, ExplicitDirtyInputIsRepublishedInOneTwoAndThreeDimensions) {
  verifies_explicit_dirty_input_uses_transaction_authority<1>();
  verifies_explicit_dirty_input_uses_transaction_authority<2>();
  verifies_explicit_dirty_input_uses_transaction_authority<3>();
}

TEST(ExactAuxiliaryRegistryNd, RejectsInvalidProviderClassAndHaloBeforePublication) {
  auto calls = std::make_shared<std::vector<std::string>>();
  EXPECT_THROW((PreparedAuxiliaryProvider<2>{
                   "derived-without-launch",
                   AuxiliaryProviderKind::derived,
                   {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::evaluation},
                   {output<2>("owner", "derived", "0", 0)},
                   {}}),
               std::invalid_argument);
  EXPECT_THROW((PreparedAuxiliaryProvider<2>{
                   "input-with-launch",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                   {output<2>("owner", "input", "0", 0)},
                   {},
                   PreparedAuxiliaryProvider<2>::launcher_type(RecordingNativeLaunch<2>{calls})}),
               std::invalid_argument);

  auto bad_halo = shape<3>();
  bad_halo.halo[2] = -1;
  EXPECT_THROW((PreparedAuxiliaryProvider<3>{
                   "negative-halo",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                   {{key("owner", "auxiliary", "halo", "0"), contract(), bad_halo}},
                   {}}),
               std::invalid_argument);
}

TEST(ExactAuxiliaryRegistryNd, MirrorsOptionalProviderPackContractFields) {
  AuxiliaryComponentContract absent_optional_fields{"cell-average", "cell", std::nullopt, "compact",
                                                    std::nullopt};
  EXPECT_NO_THROW(absent_optional_fields.validate());

  auto empty_unit = absent_optional_fields;
  empty_unit.unit = "";
  EXPECT_THROW(empty_unit.validate(), std::invalid_argument);

  auto empty_value_kind = absent_optional_fields;
  empty_value_kind.value_kind = "";
  EXPECT_THROW(empty_value_kind.validate(), std::invalid_argument);
}

template <int Dim>
void verifies_consumer_local_slots_resolve_independently_of_storage_slots() {
  const auto first = output<Dim>("owner", "provider", "first", 0);
  const auto second = output<Dim>("owner", "provider", "second", 1);
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("first-input", first));
  registry.add(input<Dim>("second-input", second));
  registry.add_consumer_plan({"consumer-a", {{dependency(second), 0}, {dependency(first), 1}}});
  registry.seal();

  const auto& plan = registry.consumer_plan("consumer-a");
  ASSERT_EQ(plan.value_count(), 2U);
  EXPECT_EQ(plan.values[0].consumer_slot, 0U);
  EXPECT_EQ(plan.values[0].address.component, 1U);
  EXPECT_EQ(plan.values[1].consumer_slot, 1U);
  EXPECT_EQ(plan.values[1].address.component, 0U);
}

TEST(ExactAuxiliaryRegistryNd, ResolvesConsumerSlotsWithoutPhysicalOrGlobalAlias) {
  verifies_consumer_local_slots_resolve_independently_of_storage_slots<1>();
  verifies_consumer_local_slots_resolve_independently_of_storage_slots<2>();
  verifies_consumer_local_slots_resolve_independently_of_storage_slots<3>();
}

TEST(ExactAuxiliaryRegistryNd, RejectsDuplicateOrSparseConsumerSlots) {
  const auto input_output = output<2>("owner", "provider", "input", 0);
  ExactAuxiliaryRegistry<2> duplicate;
  duplicate.add(input<2>("input", input_output));
  duplicate.add_consumer_plan(
      {"consumer", {{dependency(input_output), 0}, {dependency(input_output), 0}}});
  EXPECT_THROW(duplicate.seal(), std::invalid_argument);

  ExactAuxiliaryRegistry<2> sparse;
  sparse.add(input<2>("input", input_output));
  sparse.add_consumer_plan({"consumer", {{dependency(input_output), 1}}});
  EXPECT_THROW(sparse.seal(), std::invalid_argument);
}

template <int Dim>
void verifies_provider_storage_binding_is_compact_and_group_qualified() {
  using pops::Box;
  using pops::Extent;
  using pops::Index;
  using pops::MultiFab;
  using pops::mesh::Distribution;
  using pops::mesh::RankSpace;
  using pops::mesh::BoxArray;
  using pops::runtime::system::AuxiliaryStorageGroups;
  using pops::runtime::system::ResolvedAuxiliaryConsumerPlan;

  Index<Dim> lower{};
  Index<Dim> upper{};
  Extent<Dim> one_rank{};
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = 3;
    one_rank[axis] = 1;
    ghosts[axis] = 1;
  }
  const Box<Dim> domain{lower, upper};
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution =
      Distribution<Dim>::replicated(layout, RankSpace<Dim>(Index<Dim>{}, one_rank));
  MultiFab<Dim> state(layout, distribution, Index<Dim>{}, 1, ghosts);

  AuxiliaryStorageGroups<Dim> groups;
  groups.groups.emplace("provider/group-a",
                        MultiFab<Dim>(layout, distribution, Index<Dim>{}, 2, ghosts));
  groups.groups.emplace("provider/group-b",
                        MultiFab<Dim>(layout, distribution, Index<Dim>{}, 1, ghosts));

  const auto first = output<Dim>("owner-a", "provider-a", "first", 0);
  const auto second = output<Dim>("owner-b", "provider-b", "second", 1);
  ResolvedAuxiliaryConsumerPlan<Dim> plan{
      "consumer/exact",
      {{second.key, second.contract, second.shape, {"provider/group-b", 0}, 0},
       {first.key, first.contract, first.shape, {"provider/group-a", 1}, 1}}};

  // The local packing deliberately reverses producer declaration order and crosses two storage
  // groups.  Compact consumer slots must be independent from both global component and group.
  const auto view = pops::runtime::system::bind_provider_storage_view<Dim, 2>(&plan, &groups, 0);
  EXPECT_EQ(view.storage[0].data, groups.find("provider/group-b")->fab(0).view().data);
  EXPECT_EQ(view.storage_components[0], 0);
  EXPECT_EQ(view.storage[1].data, groups.find("provider/group-a")->fab(0).view().data);
  EXPECT_EQ(view.storage_components[1], 1);
  EXPECT_NO_THROW((pops::runtime::system::require_pointwise_provider_groups<Dim, 2>(
      state, &groups, &plan, "test provider binding")));

  auto invalid_component = plan;
  invalid_component.values[0].address.component = 1;
  EXPECT_THROW(((void)pops::runtime::system::bind_provider_storage_view<Dim, 2>(&invalid_component,
                                                                                &groups, 0)),
               std::invalid_argument);

  auto non_scalar = plan;
  non_scalar.values[1].shape.value_components = 2;
  EXPECT_THROW((pops::runtime::system::require_pointwise_provider_groups<Dim, 2>(
                   state, &groups, &non_scalar, "test provider binding")),
               std::invalid_argument);

  // Provider-free consumers never inspect a plan or a storage carrier owned by another consumer.
  EXPECT_NO_THROW(
      ((void)pops::runtime::system::bind_provider_storage_view<Dim, 0>(nullptr, nullptr, 0)));
  EXPECT_NO_THROW((pops::runtime::system::require_pointwise_provider_groups<Dim, 0>(
      state, &groups, &plan, "test provider-free binding")));
}

TEST(ExactAuxiliaryRegistryNd, ProviderStorageBindingIsExactAcrossRanksAndGroups) {
  verifies_provider_storage_binding_is_compact_and_group_qualified<1>();
  verifies_provider_storage_binding_is_compact_and_group_qualified<2>();
  verifies_provider_storage_binding_is_compact_and_group_qualified<3>();
}

template <int Dim>
void verifies_transactional_auxiliary_ghosts() {
  using pops::BoundaryTopology;
  using pops::Box;
  using pops::Extent;
  using pops::Geometry;
  using pops::Index;
  using pops::MultiFab;
  using pops::Real;
  using pops::RealVector;
  using pops::mesh::BoxArray;
  using pops::mesh::Distribution;
  using pops::mesh::RankSpace;
  using pops::runtime::system::AuxiliaryStorageGroups;

  Index<Dim> lower{};
  Index<Dim> upper{};
  Extent<Dim> one_rank{};
  Extent<Dim> ghosts{};
  RealVector<Dim> physical_lower{};
  RealVector<Dim> physical_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = 2;
    one_rank[axis] = 1;
    ghosts[axis] = 1;
    physical_upper[axis] = Real(3);
  }
  const Box<Dim> domain{lower, upper};
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution =
      Distribution<Dim>::replicated(layout, RankSpace<Dim>(Index<Dim>{}, one_rank));
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, physical_lower, physical_upper);

  auto declared = output<Dim>("ghost-owner", "ghost-space", "value", 0);
  declared.boundary = {AuxiliaryBoundaryPolicy::Kind::dirichlet, Real(7)};
  auto secondary = output<Dim>("ghost-owner", "ghost-space-secondary", "value", 1);
  // A distinct layout contract makes this a second resolved storage group while retaining the
  // same dimension-generic halo geometry.
  secondary.contract.layout = "compact-secondary";
  secondary.boundary = {AuxiliaryBoundaryPolicy::Kind::dirichlet, Real(11)};
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("ghost-input", declared));
  registry.add(input<Dim>("ghost-input-secondary", secondary));
  registry.seal();
  AuxiliaryStorageGroups<Dim> groups;
  ASSERT_EQ(registry.storage_groups().size(), 2U);
  for (const auto& resolved : registry.storage_groups())
    groups.groups.emplace(resolved.identity,
                          MultiFab<Dim>(layout, distribution, Index<Dim>{},
                                        static_cast<int>(resolved.component_count), ghosts));

  const auto populate = [&](const AuxiliaryOutput<Dim>& output, int boundary_axis, Real base) {
    const auto address = registry.address_of(output.key);
    auto& field = *groups.find(address.group);
    auto& fab = field.fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (int ordinal = 0; ordinal < 3; ++ordinal) {
      Index<Dim> index{};
      index[boundary_axis] = ordinal;
      host(pops::runtime::system::marshaling::storage_ordinal(
          fab, index, static_cast<int>(address.component))) = base + Real(ordinal);
    }
    fab.copy_from_host(host);
  };

  const auto expect_boundary = [&](const AuxiliaryOutput<Dim>& output, int boundary_axis,
                                   Real expected) {
    const auto address = registry.address_of(output.key);
    auto& fab = groups.find(address.group)->fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    Index<Dim> ghost{};
    ghost[boundary_axis] = -1;
    EXPECT_EQ(host(pops::runtime::system::marshaling::storage_ordinal(
                  fab, ghost, static_cast<int>(address.component))),
              expected);
  };

  const auto inject_nonfinite_ghost = [&](const AuxiliaryOutput<Dim>& output, int boundary_axis) {
    const auto address = registry.address_of(output.key);
    auto& fab = groups.find(address.group)->fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    Index<Dim> ghost{};
    ghost[boundary_axis] = -1;
    host(pops::runtime::system::marshaling::storage_ordinal(
        fab, ghost, static_cast<int>(address.component))) = std::numeric_limits<Real>::quiet_NaN();
    fab.copy_from_host(host);
  };

  for (int boundary_axis = 0; boundary_axis < Dim; ++boundary_axis) {
    populate(declared, boundary_axis, Real(1));
    populate(secondary, boundary_axis, Real(9));

    std::array<bool, Dim> periodic{};
    periodic.fill(true);
    pops::runtime::system::refresh_auxiliary_group_ghosts(
        groups, registry, domain, geometry, BoundaryTopology<Dim>::axis_periodic(periodic));
    expect_boundary(declared, boundary_axis, Real(3));
    expect_boundary(secondary, boundary_axis, Real(11));

    periodic.fill(false);
    pops::runtime::system::refresh_auxiliary_group_ghosts(
        groups, registry, domain, geometry, BoundaryTopology<Dim>::axis_periodic(periodic));
    expect_boundary(declared, boundary_axis, Real(13));
    expect_boundary(secondary, boundary_axis, Real(13));

    inject_nonfinite_ghost(declared, boundary_axis);
    EXPECT_THROW(pops::runtime::system::require_finite_auxiliary_groups(
                     groups, nullptr, "serial auxiliary ghost proof"),
                 std::runtime_error);
  }
}

TEST(ExactAuxiliaryRegistryNd, TransactionalProviderGhostsAreExactInOneTwoAndThreeDimensions) {
  verifies_transactional_auxiliary_ghosts<1>();
  verifies_transactional_auxiliary_ghosts<2>();
  verifies_transactional_auxiliary_ghosts<3>();
}

template <int Dim>
void verifies_external_field_publication_defers_and_then_refreshes_dependents() {
  const auto field_output = output<Dim>("field/owner", "electric", "potential", 0);
  const auto derived_output = output<Dim>("model/owner", "electric", "acceleration", 1);
  auto launches = std::make_shared<std::vector<std::string>>();

  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "field-output-a",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::accepted_step},
      {field_output},
      {}});
  registry.add(
      derived<Dim>("derived-b", derived_output, {dependency(field_output)}, launches,
                   {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::evaluation}));
  registry.add_consumer_plan({"consumer/b", {{dependency(derived_output), 0}}});
  registry.seal();

  EXPECT_EQ(registry.dependent_provider_identities({"field-output-a"}),
            (std::vector<std::string>{"derived-b"}));

  const auto first = point("clock", 0, AuxiliaryEvaluationEvent::before_residual);
  {
    auto publication = registry.begin_external_publication(first, {"field-output-a"});
    publication.stage_external("field-output-a");
    publication.launch_ready_native();
    EXPECT_TRUE(launches->empty())
        << "an external field publish must not eagerly launch its downstream derived provider";
    publication.accept();
  }
  EXPECT_TRUE(registry.last_accepted_point("field-output-a").has_value());
  EXPECT_FALSE(registry.last_accepted_point("derived-b").has_value());

  const auto second = point("clock", 1, AuxiliaryEvaluationEvent::before_residual);
  {
    auto incomplete = registry.begin_external_publication(second, {"field-output-a"});
    EXPECT_THROW(incomplete.validate_complete(), std::logic_error)
        << "a forced external root cannot publish without its exact candidate";
    incomplete.reject();
  }

  {
    auto refresh = registry.begin_publication(second, {}, {"consumer/b"});
    refresh.launch_ready_native();
    EXPECT_TRUE(launches->empty())
        << "the derived provider waits for its due external prerequisite";
    refresh.stage_external("field-output-a");
    refresh.launch_ready_native();
    ASSERT_EQ(launches->size(), 1U);
    EXPECT_EQ((*launches)[0], "clock:2");
    refresh.accept();
  }
  EXPECT_EQ(*registry.last_accepted_point("derived-b"), second);
}

TEST(ExactAuxiliaryRegistryNd,
     ExternalFieldPublicationDefersDependentsUntilAnExactConsumerRefreshesThem) {
  verifies_external_field_publication_defers_and_then_refreshes_dependents<1>();
  verifies_external_field_publication_defers_and_then_refreshes_dependents<2>();
  verifies_external_field_publication_defers_and_then_refreshes_dependents<3>();
}

template <int Dim>
ExactAuxiliaryRegistry<Dim> accepted_registry_for_checkpoint(
    std::shared_ptr<std::vector<std::string>> launches) {
  auto input_output = output<Dim>("checkpoint/input-owner", "input", "density", 0);
  auto field_output = output<Dim>("checkpoint/field-owner", "field", "potential", 1);
  field_output.contract = contract("field-layout");
  const auto derived_output = output<Dim>("checkpoint/derived-owner", "derived", "force", 2);

  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("checkpoint/input", input_output));
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "checkpoint/field",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {field_output},
      {}});
  registry.add(derived<Dim>("checkpoint/derived", derived_output, {dependency(input_output)},
                            std::move(launches)));
  registry.seal();

  const auto initialization =
      point("checkpoint-clock", 0, AuxiliaryEvaluationEvent::initialization);
  auto initial_publication = registry.begin_publication(initialization);
  initial_publication.stage_external("checkpoint/input");
  initial_publication.stage_external("checkpoint/field");
  initial_publication.launch_ready_native();
  initial_publication.accept();

  const auto residual = point("checkpoint-clock", 1, AuxiliaryEvaluationEvent::before_residual);
  auto residual_publication = registry.begin_publication(residual);
  residual_publication.launch_ready_native();
  residual_publication.accept();
  return registry;
}

template <int Dim>
pops::runtime::system::AuxiliaryStorageGroups<Dim> storage_for_checkpoint(
    const pops::runtime::system::AuxiliaryCheckpointAcceptedState<Dim>& state) {
  using pops::Box;
  using pops::Extent;
  using pops::Index;
  using pops::MultiFab;
  using pops::mesh::BoxArray;
  using pops::mesh::Distribution;
  using pops::mesh::RankSpace;
  using pops::runtime::system::AuxiliaryStorageGroups;

  Index<Dim> lower{};
  Index<Dim> upper{};
  Extent<Dim> one_rank{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = 2;
    one_rank[axis] = 1;
  }
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{{lower, upper}});
  const auto distribution =
      Distribution<Dim>::replicated(layout, RankSpace<Dim>(Index<Dim>{}, one_rank));
  AuxiliaryStorageGroups<Dim> storage;
  for (const auto& group : state.groups) {
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = group.shape.halo[axis];
    storage.groups.emplace(group.identity,
                           MultiFab<Dim>(layout, distribution, Index<Dim>{},
                                         static_cast<int>(group.component_count), ghosts));
  }
  return storage;
}

template <int Dim>
void verifies_auxiliary_checkpoint_is_exact_and_restart_atomic() {
  using pops::runtime::system::capture_auxiliary_checkpoint_state;
  using pops::runtime::system::deserialize_auxiliary_checkpoint_state;
  using pops::runtime::system::require_auxiliary_checkpoint_storage;
  using pops::runtime::system::restore_auxiliary_checkpoint_state;
  using pops::runtime::system::serialize_auxiliary_checkpoint_state;

  auto launches = std::make_shared<std::vector<std::string>>();
  const auto accepted = accepted_registry_for_checkpoint<Dim>(launches);
  ASSERT_EQ(accepted.accepted_generation(), 2U);
  const auto image = capture_auxiliary_checkpoint_state(accepted);
  ASSERT_EQ(image.groups.size(), 2U);
  ASSERT_EQ(image.components.size(), 3U);
  ASSERT_EQ(image.providers.size(), 3U);
  EXPECT_EQ(
      deserialize_auxiliary_checkpoint_state<Dim>(serialize_auxiliary_checkpoint_state(image)),
      image);

  auto storage = storage_for_checkpoint<Dim>(image);
  EXPECT_NO_THROW(require_auxiliary_checkpoint_storage(image, storage));
  storage.groups.erase(image.groups.front().identity);
  EXPECT_THROW(require_auxiliary_checkpoint_storage(image, storage), std::invalid_argument);

  auto restarted_launches = std::make_shared<std::vector<std::string>>();
  // A fresh sealed registry has no accepted history; restore must publish the image atomically.
  ExactAuxiliaryRegistry<Dim> empty_history;
  const auto input_output = output<Dim>("checkpoint/input-owner", "input", "density", 0);
  auto field_output = output<Dim>("checkpoint/field-owner", "field", "potential", 1);
  field_output.contract = contract("field-layout");
  const auto derived_output = output<Dim>("checkpoint/derived-owner", "derived", "force", 2);
  empty_history.add(input<Dim>("checkpoint/input", input_output));
  empty_history.add(PreparedAuxiliaryProvider<Dim>{
      "checkpoint/field",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {field_output},
      {}});
  empty_history.add(derived<Dim>("checkpoint/derived", derived_output, {dependency(input_output)},
                                 restarted_launches));
  empty_history.seal();
  restore_auxiliary_checkpoint_state(image, empty_history);
  EXPECT_EQ(empty_history.accepted_generation(), image.accepted_generation);
  for (std::size_t provider = 0; provider < image.providers.size(); ++provider)
    EXPECT_EQ(empty_history.last_accepted_point(image.providers[provider].identity),
              image.providers[provider].accepted_point);

  auto wrong_key = image;
  wrong_key.components.front().key.component = "wrong";
  EXPECT_THROW(restore_auxiliary_checkpoint_state(wrong_key, empty_history), std::invalid_argument);
  EXPECT_EQ(empty_history.accepted_generation(), image.accepted_generation);
}

TEST(ExactAuxiliaryRegistryNd, CheckpointPersistsExactGroupsKeysShapesAndAcceptedGeneration) {
  verifies_auxiliary_checkpoint_is_exact_and_restart_atomic<1>();
  verifies_auxiliary_checkpoint_is_exact_and_restart_atomic<2>();
  verifies_auxiliary_checkpoint_is_exact_and_restart_atomic<3>();
}

template <int Dim>
std::vector<std::uint8_t> blank_contract_empty_auxiliary_checkpoint() {
  namespace detail = pops::runtime::system::auxiliary_checkpoint_detail;
  detail::Writer out;
  out.raw(detail::kMagic);
  out.i32(Dim);
  out.string({});
  out.u64(0);
  out.size(0);
  out.size(0);
  out.size(0);
  return std::move(out).take();
}

template <int Dim>
void verifies_empty_auxiliary_checkpoint_attestation() {
  using pops::runtime::system::attest_empty_auxiliary_checkpoint_state;

  ExactAuxiliaryRegistry<Dim> empty;
  empty.seal();
  const auto empty_image = pops::runtime::system::serialize_auxiliary_checkpoint_state(
      pops::runtime::system::capture_auxiliary_checkpoint_state(empty));
  const auto proof = attest_empty_auxiliary_checkpoint_state<Dim>(empty_image);
  EXPECT_EQ(proof.dimension, Dim);
  EXPECT_EQ(proof.registry_contract, empty.collective_contract());
  EXPECT_EQ(proof.accepted_generation, 0U);
  EXPECT_EQ(proof.groups, 0U);
  EXPECT_EQ(proof.components, 0U);
  EXPECT_EQ(proof.providers, 0U);

  auto accepted_empty_publication = empty.begin_publication(
      point("empty-attestation", 0, AuxiliaryEvaluationEvent::initialization));
  accepted_empty_publication.accept();
  ASSERT_EQ(empty.accepted_generation(), 1U);
  const auto accepted_empty_image = pops::runtime::system::serialize_auxiliary_checkpoint_state(
      pops::runtime::system::capture_auxiliary_checkpoint_state(empty));
  const auto accepted_proof = attest_empty_auxiliary_checkpoint_state<Dim>(accepted_empty_image);
  EXPECT_EQ(accepted_proof.accepted_generation, 1U);
  EXPECT_EQ(accepted_proof.registry_contract, empty.collective_contract());
  EXPECT_EQ(accepted_proof.groups, 0U);
  EXPECT_EQ(accepted_proof.components, 0U);
  EXPECT_EQ(accepted_proof.providers, 0U);

  ExactAuxiliaryRegistry<Dim> provider_free_consumer;
  provider_free_consumer.add_consumer_plan({"consumer/provider-free", {}});
  provider_free_consumer.seal();
  EXPECT_EQ(provider_free_consumer.provider_count(), 0U);
  const auto provider_free_image = pops::runtime::system::serialize_auxiliary_checkpoint_state(
      pops::runtime::system::capture_auxiliary_checkpoint_state(provider_free_consumer));
  const auto provider_free_proof =
      attest_empty_auxiliary_checkpoint_state<Dim>(provider_free_image);
  EXPECT_NE(provider_free_proof.registry_contract, empty.collective_contract());
  EXPECT_EQ(provider_free_proof.accepted_generation, 0U);
  EXPECT_EQ(provider_free_proof.groups, 0U);
  EXPECT_EQ(provider_free_proof.components, 0U);
  EXPECT_EQ(provider_free_proof.providers, 0U);

  auto exhausted = pops::runtime::system::capture_auxiliary_checkpoint_state(empty);
  exhausted.accepted_generation = std::numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(static_cast<void>(attest_empty_auxiliary_checkpoint_state<Dim>(
                   pops::runtime::system::serialize_auxiliary_checkpoint_state(exhausted))),
               std::runtime_error);

  EXPECT_THROW(static_cast<void>(attest_empty_auxiliary_checkpoint_state<Dim>(
                   blank_contract_empty_auxiliary_checkpoint<Dim>())),
               std::exception);
  const auto accepted =
      accepted_registry_for_checkpoint<Dim>(std::make_shared<std::vector<std::string>>());
  const auto nonempty = pops::runtime::system::serialize_auxiliary_checkpoint_state(
      pops::runtime::system::capture_auxiliary_checkpoint_state(accepted));
  EXPECT_THROW(static_cast<void>(attest_empty_auxiliary_checkpoint_state<Dim>(nonempty)),
               std::runtime_error);
  constexpr int WrongDim = Dim == 1 ? 2 : 1;
  ExactAuxiliaryRegistry<WrongDim> wrong_dimension_registry;
  wrong_dimension_registry.seal();
  const auto wrong_dimension = pops::runtime::system::serialize_auxiliary_checkpoint_state(
      pops::runtime::system::capture_auxiliary_checkpoint_state(wrong_dimension_registry));
  EXPECT_THROW(static_cast<void>(attest_empty_auxiliary_checkpoint_state<Dim>(wrong_dimension)),
               std::runtime_error);
}

TEST(ExactAuxiliaryRegistryNd, EmptyAuxiliaryCheckpointAttestationIsExactAndFailClosed) {
  verifies_empty_auxiliary_checkpoint_attestation<1>();
  verifies_empty_auxiliary_checkpoint_attestation<2>();
  verifies_empty_auxiliary_checkpoint_attestation<3>();
}

}  // namespace
