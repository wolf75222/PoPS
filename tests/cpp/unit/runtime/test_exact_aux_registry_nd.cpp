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

namespace {
template <int Dim>
void verifies_lazy_checkpoint_invalidations() {
  using namespace pops::runtime::system;
  auto calls = std::make_shared<std::vector<std::string>>();
  auto registry = accepted_registry_for_checkpoint<Dim>(calls);
  const auto initial_launches = calls->size();
  const auto clean = capture_auxiliary_checkpoint_state(registry);
  const auto clean_bytes = serialize_auxiliary_checkpoint_state(clean);
  EXPECT_EQ(clean_bytes[7], '2');
  EXPECT_EQ(serialize_auxiliary_checkpoint_state(deserialize_auxiliary_checkpoint_state<Dim>(clean_bytes)), clean_bytes);
  const auto stale = capture_auxiliary_checkpoint_state(registry, {"checkpoint/derived"});
  const auto stale_bytes = serialize_auxiliary_checkpoint_state(stale);
  EXPECT_EQ(stale_bytes[7], '3');
  const auto decoded = deserialize_auxiliary_checkpoint_state<Dim>(stale_bytes);
  EXPECT_EQ(decoded, stale);
  EXPECT_EQ(decoded.providers, clean.providers);
  EXPECT_EQ(calls->size(), initial_launches); // capture/codec never evaluates a lazy provider
  auto restored = registry;
  restore_auxiliary_checkpoint_state(decoded, restored);
  EXPECT_EQ(capture_auxiliary_checkpoint_state(restored, decoded.invalidated_providers), stale);
  EXPECT_EQ(calls->size(), initial_launches);
  {
    auto rejected = restored.begin_publication(point("checkpoint-clock", 2, AuxiliaryEvaluationEvent::before_residual), decoded.invalidated_providers);
    rejected.launch_ready_native();
    rejected.reject();
  }
  EXPECT_EQ(capture_auxiliary_checkpoint_state(restored, decoded.invalidated_providers), stale);
  {
    auto retry = restored.begin_publication(point("checkpoint-clock", 2, AuxiliaryEvaluationEvent::before_residual), decoded.invalidated_providers);
    retry.launch_ready_native(); retry.accept();
  }
  EXPECT_EQ(calls->size(), initial_launches + 2);
  EXPECT_EQ(capture_auxiliary_checkpoint_state(restored).invalidated_providers.size(), 0U);
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(registry, {"checkpoint/input"}), std::invalid_argument);
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(registry, {"foreign"}), std::invalid_argument);
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(registry, {"checkpoint/derived", "checkpoint/derived"}), std::invalid_argument);
  auto bad = stale;
  for (auto& provider : bad.providers)
    if (provider.identity == "checkpoint/derived") provider.accepted_point.reset();
  EXPECT_THROW((void)serialize_auxiliary_checkpoint_state(bad), std::invalid_argument);
  auto empty_extension = clean_bytes;
  empty_extension[7] = '3'; empty_extension.insert(empty_extension.end(), 8, 0);
  EXPECT_THROW((void)deserialize_auxiliary_checkpoint_state<Dim>(empty_extension), std::runtime_error);
  auto unsupported = clean_bytes; unsupported[7] = '4';
  EXPECT_THROW((void)deserialize_auxiliary_checkpoint_state<Dim>(unsupported), std::runtime_error);
  ExactAuxiliaryRegistry<Dim> initial;
  auto unaccepted_output = output<Dim>("initial/owner", "gain", "value", 0);
  initial.add(derived<Dim>("initial/derived", unaccepted_output, {}, calls));
  initial.seal();
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(initial, {"initial/derived"}), std::invalid_argument);
  auto truncated = stale_bytes; truncated.pop_back();
  EXPECT_THROW((void)deserialize_auxiliary_checkpoint_state<Dim>(truncated), std::runtime_error);
  auto foreign = stale; foreign.registry_contract += "foreign";
  const auto before = capture_auxiliary_checkpoint_state(restored);
  EXPECT_THROW(restore_auxiliary_checkpoint_state(foreign, restored), std::invalid_argument);
  EXPECT_EQ(capture_auxiliary_checkpoint_state(restored), before);
}
TEST(ExactAuxiliaryRegistryNd, LazyInvalidationsRoundTripWithoutPublishingAndRefreshOnlyOnConsumer) {
  verifies_lazy_checkpoint_invalidations<1>();
  verifies_lazy_checkpoint_invalidations<2>();
  verifies_lazy_checkpoint_invalidations<3>();
}

template <int Dim>
void verifies_unpublished_dependents_are_due_without_stale_publication() {
  using namespace pops::runtime::system;
  auto calls = std::make_shared<std::vector<std::string>>();
  const auto source = output<Dim>("arbitrary/root-owner", "field", "load", 0);
  const auto used = output<Dim>("arbitrary/used-owner", "derived", "used", 1);
  const auto dormant = output<Dim>("arbitrary/dormant-owner", "derived", "dormant", 2);
  ExactAuxiliaryRegistry<Dim> registry;
  const AuxiliaryEvaluationPolicy once{AuxiliaryEvaluationEvent::before_residual,
                                      AuxiliaryFreshness::once};
  registry.add(PreparedAuxiliaryProvider<Dim>{"root", AuxiliaryProviderKind::field_output,
                                            once, {source}, {}});
  registry.add(derived<Dim>("used", used, {dependency(source)}, calls, once));
  registry.add(derived<Dim>("dormant", dormant, {dependency(source)}, calls, once));
  registry.add_consumer_plan({"read-used", {{dependency(used), 0}}});
  registry.add_consumer_plan({"read-dormant", {{dependency(dormant), 0}}});
  registry.seal();
  const auto at = point("arbitrary-clock", 0, AuxiliaryEvaluationEvent::before_residual);
  {
    auto external = registry.begin_external_publication(at, {"root"});
    external.stage_external("root");
    external.accept();
  }
  EXPECT_EQ(registry.dependent_provider_identities({"root"}).size(), 2U);
  EXPECT_TRUE(registry.accepted_dependent_provider_identities({"root"}).empty());
  {
    auto read = registry.begin_publication(at, {}, {"read-used"});
    read.launch_ready_native();
    read.accept();
  }
  EXPECT_EQ(calls->size(), 1U);
  EXPECT_FALSE(registry.last_accepted_point("dormant"));
  const auto invalidated = registry.accepted_dependent_provider_identities({"root"});
  EXPECT_EQ(invalidated, (std::vector<std::string>{"used"}));
  {
    auto external = registry.begin_external_publication(at, {"root"});
    external.stage_external("root");
    external.accept();
  }
  const auto checkpoint = capture_auxiliary_checkpoint_state(registry, invalidated);
  const auto bytes = serialize_auxiliary_checkpoint_state(checkpoint);
  EXPECT_EQ(bytes[7], '3');
  auto restored = registry;
  restore_auxiliary_checkpoint_state(deserialize_auxiliary_checkpoint_state<Dim>(bytes), restored);
  EXPECT_EQ(capture_auxiliary_checkpoint_state(restored, invalidated), checkpoint);
  EXPECT_EQ(calls->size(), 1U) << "checkpoint must not evaluate a dormant provider";
  {
    auto read = restored.begin_publication(at, invalidated, {"read-dormant"});
    EXPECT_TRUE(read.requires_staging("used")) << "accepted stale image remains forced";
    EXPECT_TRUE(read.requires_staging("dormant")) << "first read needs no dirty marker";
    read.launch_ready_native();
    read.reject();
  }
  EXPECT_FALSE(restored.last_accepted_point("dormant"));
  EXPECT_EQ(capture_auxiliary_checkpoint_state(restored, invalidated), checkpoint);
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(registry, {"dormant"}),
               std::invalid_argument);
}

TEST(ExactAuxiliaryRegistryNd, UnpublishedDependentsRemainDueWithoutCheckpointInvalidations) {
  verifies_unpublished_dependents_are_due_without_stale_publication<1>();
  verifies_unpublished_dependents_are_due_without_stale_publication<2>();
  verifies_unpublished_dependents_are_due_without_stale_publication<3>();
}

// Registry selection/provenance witnesses only. These do not substitute for AMR
// accepted publication, per-level acknowledgement, carrier transport or rollback.
template <int Dim>
struct DrainPhysicalNativeLaunch {
  std::shared_ptr<std::vector<AuxiliaryEvaluationPoint>> calls;
  std::string expected_clock;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.exact-aux.drain-physical-launch", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& exact) const {
    exact.text(expected_clock);
  }
  void operator()(const pops::runtime::system::AuxiliaryKernelLaunchContext<Dim>& context) const {
    calls->push_back(context.point);
    (void)context.point.require_physical_time(expected_clock);
  }
};

AuxiliaryEvaluationPoint drain_physical_point(int stage, pops::amr::Rational fraction,
                                             double time) {
  auto requested = point("counterexample.physical-clock", 0,
                         AuxiliaryEvaluationEvent::before_field_solve, 0, 0, 0, stage);
  pops::runtime::multiblock::BoundaryEvaluationPoint actual;
  actual.clock = requested.clock; actual.tick = 0; actual.level = 0;
  actual.substep = 0; actual.stage = stage;
  actual.stage_fraction = fraction; actual.dt = 1.; actual.physical_time = time;
  requested.qualify_physical_evaluation(actual);
  return requested;
}

AuxiliaryEvaluationPoint drain_diagnostic_point() {
  return point("counterexample.legacy-topology", 0,
               AuxiliaryEvaluationEvent::after_regrid, 7);
}

template <int Dim>
struct DrainRegistryFixture {
  std::shared_ptr<std::vector<AuxiliaryEvaluationPoint>> physical_calls =
      std::make_shared<std::vector<AuxiliaryEvaluationPoint>>();
  std::shared_ptr<std::vector<std::string>> legacy_calls =
      std::make_shared<std::vector<std::string>>();
  ExactAuxiliaryRegistry<Dim> registry;

  explicit DrainRegistryFixture(bool legacy_depends_on_physical) {
    using Provider = PreparedAuxiliaryProvider<Dim>;
    using Launcher = typename Provider::launcher_type;
    const auto physical = output<Dim>("physical-owner", "aux", "physical", 0);
    const auto legacy = output<Dim>("legacy-owner", "aux", "legacy", 1);
    registry.add(Provider{"physical", AuxiliaryProviderKind::derived,
      AuxiliaryEvaluationPolicy{std::vector<AuxiliaryEvaluationEvent>{
        AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryEvaluationEvent::after_regrid},
        AuxiliaryFreshness::evaluation}, {physical}, {},
      Launcher(DrainPhysicalNativeLaunch<Dim>{
        physical_calls, "counterexample.physical-clock"})});
    std::vector<AuxiliaryDependency<Dim>> dependencies;
    if (legacy_depends_on_physical) dependencies.push_back(dependency(physical));
    registry.add(derived<Dim>("legacy", legacy, std::move(dependencies), legacy_calls,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::once}));
    registry.add_consumer_plan({"read-physical", {{dependency(physical), 0}}});
    registry.add_consumer_plan({"read-legacy", {{dependency(legacy), 0}}});
    registry.seal();
  }

  void publish_initial() {
    auto publication = registry.begin_publication(
        drain_physical_point(1, {1, 4}, .25), {}, {"read-physical", "read-legacy"});
    publication.launch_ready_native();
    publication.accept();
  }
};

template <int Dim>
void verifies_drain_leaves_unrelated_clean_physical_provider_accepted() {
  DrainRegistryFixture<Dim> fixture(false);
  fixture.publish_initial();
  const auto accepted = fixture.registry.last_accepted_point("physical");
  ASSERT_EQ(fixture.physical_calls->size(), 1U);
  const auto diagnostic = drain_diagnostic_point();
  {
    // Control: ordinary publication would select the clean physical provider by
    // its authored after_regrid/evaluation policy, even with only legacy forced.
    auto normal = fixture.registry.begin_publication(diagnostic, {"legacy"});
    EXPECT_TRUE(normal.requires_staging("physical"));
    normal.reject();
  }
  {
    auto drain = fixture.registry.begin_invalidated_publication(diagnostic, {"legacy"});
    EXPECT_FALSE(drain.requires_staging("physical"));
    EXPECT_TRUE(drain.requires_staging("legacy"));
    drain.launch_ready_native();
    drain.accept();
  }
  EXPECT_EQ(fixture.physical_calls->size(), 1U);
  EXPECT_EQ(fixture.legacy_calls->size(), 2U);
  EXPECT_EQ(fixture.registry.last_accepted_point("physical"), accepted);
  EXPECT_EQ(*fixture.registry.last_accepted_point("legacy"), diagnostic);
}

TEST(ExactAuxiliaryRegistryNd, InvalidationDrainDoesNotReevaluateUnrelatedCleanPhysicalProvider) {
  verifies_drain_leaves_unrelated_clean_physical_provider_accepted<1>();
  verifies_drain_leaves_unrelated_clean_physical_provider_accepted<2>();
  verifies_drain_leaves_unrelated_clean_physical_provider_accepted<3>();
}

template <int Dim>
void verifies_drain_uses_clean_accepted_physical_prerequisite() {
  DrainRegistryFixture<Dim> fixture(true);
  fixture.publish_initial();
  const auto accepted = fixture.registry.last_accepted_point("physical");
  ASSERT_TRUE(accepted.has_value());
  const auto diagnostic = drain_diagnostic_point();
  auto drain = fixture.registry.begin_invalidated_publication(diagnostic, {"legacy"});
  EXPECT_FALSE(drain.requires_staging("physical"));
  EXPECT_TRUE(drain.requires_staging("legacy"));
  drain.launch_ready_native();
  EXPECT_NO_THROW(drain.validate_complete());
  drain.accept();
  EXPECT_EQ(fixture.physical_calls->size(), 1U);
  EXPECT_EQ(fixture.legacy_calls->size(), 2U);
  EXPECT_EQ(fixture.registry.last_accepted_point("physical"), accepted);
  EXPECT_EQ(*fixture.registry.last_accepted_point("legacy"), diagnostic);
}

TEST(ExactAuxiliaryRegistryNd, InvalidationDrainBorrowsCleanPhysicalPrerequisiteAtAcceptedProvenance) {
  verifies_drain_uses_clean_accepted_physical_prerequisite<1>();
  verifies_drain_uses_clean_accepted_physical_prerequisite<2>();
  verifies_drain_uses_clean_accepted_physical_prerequisite<3>();
}

template <int Dim>
void verifies_dirty_physical_provider_refuses_unqualified_drain() {
  DrainRegistryFixture<Dim> fixture(true);
  fixture.publish_initial();
  const auto accepted = fixture.registry.accepted_points();
  const auto generation = fixture.registry.accepted_generation();
  auto same_clock_without_payload = drain_diagnostic_point();
  same_clock_without_payload.clock = "counterexample.physical-clock";
  for (const auto& requested : std::vector<AuxiliaryEvaluationPoint>{
         drain_diagnostic_point(), same_clock_without_payload}) {
    auto drain = fixture.registry.begin_invalidated_publication(requested, {"physical", "legacy"});
    EXPECT_TRUE(drain.requires_staging("physical"));
    EXPECT_THROW(drain.launch_ready_native(), std::invalid_argument);
    EXPECT_THROW(drain.validate_complete(), std::logic_error);
    drain.reject();
    EXPECT_EQ(fixture.registry.accepted_points(), accepted);
    EXPECT_EQ(fixture.registry.accepted_generation(), generation);
  }
  EXPECT_EQ(fixture.physical_calls->size(), 3U) << "both absent-payload paths must reach the guard";
  EXPECT_EQ(fixture.legacy_calls->size(), 1U) << "a dependent cannot run after its prerequisite fails";
  EXPECT_EQ(fixture.registry.accepted_points(), accepted);
  EXPECT_EQ(fixture.registry.accepted_generation(), generation);
}

TEST(ExactAuxiliaryRegistryNd, DirtyPhysicalInvalidationDrainRetainsExactPhysicalClockGuard) {
  verifies_dirty_physical_provider_refuses_unqualified_drain<1>();
  verifies_dirty_physical_provider_refuses_unqualified_drain<2>();
  verifies_dirty_physical_provider_refuses_unqualified_drain<3>();
}

template <int Dim>
void verifies_normal_exact_consumer_rechecks_physical_freshness() {
  DrainRegistryFixture<Dim> fixture(true);
  fixture.publish_initial();
  ASSERT_EQ(fixture.physical_calls->size(), 1U);
  auto after_regrid = drain_physical_point(3, {7, 8}, .875);
  after_regrid.event = AuxiliaryEvaluationEvent::after_regrid;
  for (const auto& next : std::vector<AuxiliaryEvaluationPoint>{
         drain_physical_point(2, {3, 4}, .75), after_regrid}) {
    auto consumer = fixture.registry.begin_publication(next, {}, {"read-legacy"});
    EXPECT_TRUE(consumer.requires_staging("physical"));
    EXPECT_TRUE(consumer.requires_staging("legacy")) << "due dependency propagation remains active";
    consumer.launch_ready_native();
    consumer.accept();
    EXPECT_EQ(fixture.physical_calls->back(), next);
    EXPECT_EQ(*fixture.registry.last_accepted_point("physical"), next);
    EXPECT_EQ(*fixture.registry.last_accepted_point("legacy"), next);
  }
  EXPECT_EQ(fixture.physical_calls->size(), 3U);
  EXPECT_EQ(fixture.legacy_calls->size(), 3U);
}

TEST(ExactAuxiliaryRegistryNd, NormalExactConsumerStillReevaluatesCleanPhysicalProviderAtNewPoint) {
  verifies_normal_exact_consumer_rechecks_physical_freshness<1>();
  verifies_normal_exact_consumer_rechecks_physical_freshness<2>();
  verifies_normal_exact_consumer_rechecks_physical_freshness<3>();
}

template <int Dim>
void verifies_drain_cannot_invent_accepted_prerequisite() {
  DrainRegistryFixture<Dim> fixture(true);
  EXPECT_THROW((void)fixture.registry.begin_invalidated_publication(
      drain_diagnostic_point(), {"legacy"}), std::logic_error);
  EXPECT_TRUE(fixture.physical_calls->empty());
  EXPECT_TRUE(fixture.legacy_calls->empty());
  EXPECT_EQ(fixture.registry.accepted_generation(), 0U);
  EXPECT_FALSE(fixture.registry.last_accepted_point("physical"));
  EXPECT_FALSE(fixture.registry.last_accepted_point("legacy"));
  // Refusal must leave the registry available for a real qualified publication.
  EXPECT_NO_THROW(fixture.publish_initial());
  EXPECT_EQ(fixture.registry.accepted_generation(), 1U);
}

TEST(ExactAuxiliaryRegistryNd, InvalidationDrainRefusesUnpublishedCleanPrerequisiteBeforeLaunch) {
  verifies_drain_cannot_invent_accepted_prerequisite<1>();
  verifies_drain_cannot_invent_accepted_prerequisite<2>();
  verifies_drain_cannot_invent_accepted_prerequisite<3>();
}

template <int Dim>
void verifies_rejected_drain_preserves_accepted_provenance_and_retry() {
  DrainRegistryFixture<Dim> fixture(true);
  fixture.publish_initial();
  const auto accepted = fixture.registry.accepted_points();
  const auto physical_accepted = fixture.registry.last_accepted_point("physical");
  const auto generation = fixture.registry.accepted_generation();
  const auto diagnostic = drain_diagnostic_point();
  {
    auto rejected = fixture.registry.begin_invalidated_publication(diagnostic, {"legacy"});
    rejected.launch_ready_native();
    EXPECT_NO_THROW(rejected.validate_complete());
    rejected.reject();
  }
  EXPECT_EQ(fixture.registry.accepted_points(), accepted);
  EXPECT_EQ(fixture.registry.accepted_generation(), generation);
  {
    auto retry = fixture.registry.begin_invalidated_publication(diagnostic, {"legacy"});
    EXPECT_TRUE(retry.requires_staging("legacy"));
    EXPECT_FALSE(retry.requires_staging("physical"));
    retry.launch_ready_native();
    retry.accept();
  }
  EXPECT_EQ(fixture.physical_calls->size(), 1U);
  EXPECT_EQ(fixture.legacy_calls->size(), 3U);
  EXPECT_EQ(fixture.registry.accepted_generation(), generation + 1);
  EXPECT_EQ(*fixture.registry.last_accepted_point("legacy"), diagnostic);
  EXPECT_EQ(fixture.registry.last_accepted_point("physical"), physical_accepted);
}

TEST(ExactAuxiliaryRegistryNd, RejectedInvalidationDrainDoesNotPublishOrConsumeRetryWork) {
  verifies_rejected_drain_preserves_accepted_provenance_and_retry<1>();
  verifies_rejected_drain_preserves_accepted_provenance_and_retry<2>();
  verifies_rejected_drain_preserves_accepted_provenance_and_retry<3>();
}
} // namespace

#include <pops/runtime/system/auxiliary_checkpoint_capacity.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <bit>
namespace {
template <int Dim>
ExactAuxiliaryRegistry<Dim> cold_capacity_registry() {
  auto calls = std::make_shared<std::vector<std::string>>();
  auto seed = output<Dim>("capacity/seed-owner", "input", "value", 0);
  auto dormant = output<Dim>("capacity/dormant-owner", "derived", "value", 1);
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(input<Dim>("seed", seed));
  registry.add(derived<Dim>("dormant", dormant, {dependency(seed)}, calls));
  registry.add_consumer_plan({"read-dormant", {{dependency(dormant), 0}}});
  registry.seal();
  return registry;
}
template <int Dim>
void capacity_counterexamples() {
  using namespace pops::runtime::system;
  auto registry = cold_capacity_registry<Dim>();
  const auto clean = capture_auxiliary_checkpoint_state(registry);
  EXPECT_THROW(require_no_pending_auxiliary_input_checkpoint(registry, {"seed"}), std::logic_error);
  EXPECT_THROW(require_no_pending_auxiliary_input_checkpoint(registry, {"foreign"}),
               std::invalid_argument);
  EXPECT_NO_THROW(require_no_pending_auxiliary_input_checkpoint(registry, {}));
  EXPECT_TRUE(current_accepted_auxiliary_checkpoint_observation(registry, {"seed", "dormant"})
                  .invalidated_providers.empty());
  EXPECT_THROW((void)current_accepted_auxiliary_checkpoint_observation(registry, {"foreign"}),
               std::invalid_argument);
  EXPECT_THROW((void)current_accepted_auxiliary_checkpoint_observation(registry, {"seed", "seed"}),
               std::invalid_argument);
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(registry, {"seed"}), std::invalid_argument);
  EXPECT_THROW((void)capture_auxiliary_checkpoint_state(registry, {"dormant"}),
               std::invalid_argument);
  EXPECT_FALSE(registry.last_accepted_point("seed"));
  EXPECT_FALSE(registry.last_accepted_point("dormant"));
  auto restored = registry;
  restore_auxiliary_checkpoint_state(clean, restored);
  auto retry = restored.begin_publication(point("p", 0, AuxiliaryEvaluationEvent::before_residual),
                                          {"seed"}, {"read-dormant"});
  EXPECT_TRUE(retry.requires_staging("seed"));
  EXPECT_TRUE(retry.requires_staging("dormant"));
  retry.reject();
  EXPECT_FALSE(restored.last_accepted_point("seed"));
  EXPECT_FALSE(restored.last_accepted_point("dormant"));
}
TEST(ExactAuxiliaryCapacityNd, ColdPendingIdentitiesAreNotAcceptedInvalidations) {
  capacity_counterexamples<1>();
  capacity_counterexamples<2>();
  capacity_counterexamples<3>();
}
template <int Dim>
void future_owned_clock_bound() {
  using namespace pops::runtime::system;
  auto registry = cold_capacity_registry<Dim>();
  const std::string long_clock(4096, 'q');
  pops::runtime::program::ProgramOwnedClockManifest manifest{
      "installed-owner", "p", {"p", long_clock}};
  const auto initial_size =
      serialize_auxiliary_checkpoint_state(capture_auxiliary_checkpoint_state(registry)).size();
  const auto bound = program_auxiliary_metadata_capacity(registry, manifest);
  EXPECT_GT(bound, initial_size);
  auto initialization = registry.begin_external_publication(
      point("p", 0, AuxiliaryEvaluationEvent::initialization), {"seed"});
  initialization.stage_external("seed");
  initialization.launch_ready_native();
  initialization.accept();
  EXPECT_FALSE(registry.last_accepted_point("dormant"));
  const auto dormant_image = capture_auxiliary_checkpoint_state(registry);
  auto restored = registry;
  restore_auxiliary_checkpoint_state(dormant_image, restored);
  auto publication = restored.begin_publication(
      point(long_clock, 1, AuxiliaryEvaluationEvent::before_residual), {}, {"read-dormant"});
  EXPECT_TRUE(publication.requires_staging("dormant"));
  publication.launch_ready_native();
  publication.accept();
  ASSERT_TRUE(restored.last_accepted_point("dormant"));
  EXPECT_EQ(restored.last_accepted_point("dormant")->clock, long_clock);
  const auto image = capture_auxiliary_checkpoint_state(restored, {"dormant"});
  EXPECT_EQ(current_accepted_auxiliary_checkpoint_observation(restored, {"seed", "dormant"}),
            image);
  EXPECT_LE(serialize_auxiliary_checkpoint_state(image).size(), bound);
  EXPECT_EQ(program_auxiliary_metadata_capacity(restored, manifest), bound);
  // Raw System/registry clocks remain unbounded. An unrelated longer raw clock is valid,
  // but invalidates any claim that the original Program-owned capacity bound covers it.
  const std::string raw_clock(bound + 1, 'r');
  auto raw = restored.begin_publication(
      point(raw_clock, 2, AuxiliaryEvaluationEvent::before_residual), {}, {"read-dormant"});
  raw.launch_ready_native();
  raw.accept();
  const auto raw_image = capture_auxiliary_checkpoint_state(restored);
  EXPECT_GT(serialize_auxiliary_checkpoint_state(raw_image).size(), bound);
  EXPECT_GT(program_auxiliary_metadata_capacity(restored, manifest), bound);
}
TEST(ExactAuxiliaryCapacityNd, ActualLongClockPublicationAndInvalidationFitOwnedReserve) {
  future_owned_clock_bound<1>();
  future_owned_clock_bound<2>();
  future_owned_clock_bound<3>();
}
TEST(ExactAuxiliaryCapacityNd, IncompleteManifestAndOverflowFailClosed) {
  pops::runtime::program::ProgramOwnedClockManifest incomplete{"", "p", {"p"}};
  EXPECT_THROW(incomplete.validate(), std::invalid_argument);
  auto unsupported = incomplete;
  unsupported.contract_version =
      static_cast<pops::runtime::program::ProgramOwnedClockManifestVersion>(2);
  EXPECT_THROW(unsupported.validate(), std::invalid_argument);
  pops::runtime::program::ProgramOwnedClockManifest foreign{"owner", "p", {"q"}};
  EXPECT_THROW(foreign.validate(), std::invalid_argument);
  pops::runtime::program::ProgramOwnedClockManifest duplicate{"owner", "p", {"p", "p"}};
  EXPECT_THROW(duplicate.validate(), std::invalid_argument);
  EXPECT_THROW(pops::runtime::system::checked_auxiliary_capacity_add(
                   std::numeric_limits<std::size_t>::max(), 1),
               std::overflow_error);
}
}  // namespace

namespace {
template <int Dim>
void accepted_input_cache_restore() {
  using namespace pops::runtime::system;
  auto registry = cold_capacity_registry<Dim>();
  auto publication = registry.begin_external_publication(
      point("p", 0, AuxiliaryEvaluationEvent::initialization), {"seed"});
  publication.stage_external("seed");
  publication.accept();
  auto image = capture_auxiliary_checkpoint_state(registry);
  for (auto& group : image.groups) {
    group.payload.resize(group.component_count * 2);
    for (std::size_t i = 0; i < group.payload.size(); ++i)
      group.payload[i] = i % 2 ? -0.0 : 7.0;
  }
  auto cache = restored_accepted_auxiliary_inputs(image, 2);
  ASSERT_EQ(cache.size(), 1U);
  const auto input_component =
      std::find_if(image.components.begin(), image.components.end(), [](const auto& component) {
        return component.provider_kind == AuxiliaryProviderKind::input;
      });
  ASSERT_NE(input_component, image.components.end());
  const auto expected = input_component->key.exact_key();
  ASSERT_EQ(cache.at(expected).size(), 2U);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(cache.at(expected)[0]), std::bit_cast<std::uint64_t>(7.0));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(cache.at(expected)[1]),
            std::bit_cast<std::uint64_t>(-0.0));
  // Accepted-only restore replaces a newer staged cache rather than leaking it into the restart.
  std::map<std::string, std::vector<double>> staged{{expected, {99.0, 98.0}}};
  staged.swap(cache);
  EXPECT_EQ(staged.at(expected)[0], 7.0);
  EXPECT_EQ(cache.at(expected)[0], 99.0);  // retired owner remains intact until commit completes
  auto cold = capture_auxiliary_checkpoint_state(cold_capacity_registry<Dim>());
  for (auto& group : cold.groups)
    group.payload.resize(group.component_count * 2, 0.0);
  EXPECT_TRUE(restored_accepted_auxiliary_inputs(cold, 2).empty());
  EXPECT_FALSE(cold.providers[0].accepted_point);
  EXPECT_FALSE(cold.providers[1].accepted_point);
  image.groups[0].payload.pop_back();
  EXPECT_THROW((void)restored_accepted_auxiliary_inputs(image, 2), std::invalid_argument);
}
TEST(ExactAuxiliaryCapacityNd, AcceptedRestoreCacheOwnsBytesAndLeavesColdProvidersUninitialized) {
  accepted_input_cache_restore<1>();
  accepted_input_cache_restore<2>();
  accepted_input_cache_restore<3>();
}
template <int Dim>
void manifest_install_rollback() {
  pops::runtime::program::ProgramRuntimeState<Dim> state;
  const pops::runtime::program::ProgramOwnedClockManifest original{
      "owned-install", "main", {"main", std::string(4096, 's')}, 7};
  state.step_install_generation_ = 7;
  state.checkpoint_metadata_.uniform_auxiliary_clocks = original;
  auto snapshot = state.capture_artifact_step_install();
  state.install_unverified_step([](double) {});
  EXPECT_TRUE(state.checkpoint_metadata_.uniform_auxiliary_clocks.owner_identity.empty());
  state.rollback_artifact_step_install(std::move(snapshot));
  EXPECT_EQ(state.checkpoint_metadata_.uniform_auxiliary_clocks, original);
  EXPECT_EQ(state.step_install_generation_, 7U);
  EXPECT_NO_THROW(
      state.checkpoint_metadata_.uniform_auxiliary_clocks.require_owned(std::string(4096, 's')));
  EXPECT_THROW(state.checkpoint_metadata_.uniform_auxiliary_clocks.require_owned("foreign"),
               std::invalid_argument);
}
TEST(ExactAuxiliaryCapacityNd, InstalledClockManifestSharesExistingFullInstallRollback) {
  manifest_install_rollback<1>();
  manifest_install_rollback<2>();
  manifest_install_rollback<3>();
}
}  // namespace
