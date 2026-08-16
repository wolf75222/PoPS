#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pops;
using namespace pops::mesh;
using namespace pops::runtime::system;

AuxiliaryComponentContract contract() {
  return {"cell-average", "cell", "unitless", "mpi-auxiliary-ghost", "scalar"};
}

template <int Dim>
struct RankFailingLaunch {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.auxiliary-rank-failure", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& exact) const {
    exact.text("test.mpi.auxiliary-rank-failure").scalar(std::uint32_t{1});
  }
  void operator()(const AuxiliaryKernelLaunchContext<Dim>&) const {
    if (my_rank() == 0)
      throw std::runtime_error("injected rank-local auxiliary launch failure");
  }
};

template <int Dim>
void expect_rank_local_launch_failure(int rank, int ranks, int split_axis,
                                      const ExecutionLane& lane) {
  constexpr int patch_count = 2;
  Index<Dim> upper{};
  Extent<Dim> rank_extent{};
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = axis == split_axis ? 2 * patch_count - 1 : 1;
    rank_extent[axis] = axis == split_axis ? ranks : 1;
    ghosts[axis] = 1;
  }
  const Box<Dim> domain{Index<Dim>{}, upper};
  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  for (int owner = 0; owner < patch_count; ++owner) {
    Index<Dim> lower{};
    Index<Dim> patch_upper = upper;
    Index<Dim> coordinate{};
    lower[split_axis] = 2 * owner;
    patch_upper[split_axis] = lower[split_axis] + 1;
    coordinate[split_axis] = owner;
    boxes.push_back(Box<Dim>{lower, patch_upper});
    owners.push_back(coordinate);
  }
  const BoxArray<Dim> layout(std::move(boxes));
  const Distribution<Dim> distribution = Distribution<Dim>::partitioned(
      layout, RankSpace<Dim>(Index<Dim>{}, rank_extent), std::move(owners));
  Index<Dim> local_rank{};
  local_rank[split_axis] = rank;

  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = ghosts[axis];
  AuxiliaryOutput<Dim> output{{"mpi.failure", "derived", "carrier", "value"}, contract(), shape};
  using Provider = PreparedAuxiliaryProvider<Dim>;
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(Provider{"mpi.failure.provider",
                        AuxiliaryProviderKind::derived,
                        {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
                        {output},
                        {},
                        typename Provider::launcher_type(RankFailingLaunch<Dim>{})});
  registry.seal();

  AuxiliaryStorageGroups<Dim> accepted;
  const std::string identity = registry.storage_groups().front().identity;
  accepted.groups.emplace(identity, MultiFab<Dim>(layout, distribution, local_rank, 1, ghosts));
  AuxiliaryStorageGroups<Dim> candidate = accepted;
  bool threw = false;
  try {
    auto publication = registry.begin_publication(AuxiliaryEvaluationPoint{
        "mpi-failure", 0, 0, 0, 0, 0, 0, AuxiliaryEvaluationEvent::initialization});
    publication.launch_ready_native(
        {&accepted, &candidate}, [&](const auto&, std::exception_ptr local_error) {
          if (all_reduce_max(local_error ? 1L : 0L, lane.communicator()) != 0)
            throw std::runtime_error("rank-local auxiliary launch failed collectively before halo");
          ADD_FAILURE() << "rank-local failure barrier unexpectedly allowed a ghost exchange";
        });
    publication.accept();
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_min(threw ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(threw ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(registry.accepted_generation(), 0U);
  EXPECT_FALSE(registry.last_accepted_point("mpi.failure.provider").has_value());
}

template <int Dim>
void expect_preparation_failure_is_collective(int rank, int ranks, int split_axis,
                                              const ExecutionLane& lane) {
  constexpr int patch_count = 2;
  Index<Dim> upper{};
  Extent<Dim> rank_extent{};
  Extent<Dim> ghosts{};
  RealVector<Dim> lower{};
  RealVector<Dim> physical_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = axis == split_axis ? 2 * patch_count - 1 : 1;
    rank_extent[axis] = axis == split_axis ? ranks : 1;
    ghosts[axis] = 1;
    physical_upper[axis] = static_cast<Real>(upper[axis] + 1);
  }
  const Box<Dim> domain{Index<Dim>{}, upper};
  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  for (int owner = 0; owner < patch_count; ++owner) {
    Index<Dim> patch_lower{};
    Index<Dim> patch_upper = upper;
    Index<Dim> coordinate{};
    patch_lower[split_axis] = 2 * owner;
    patch_upper[split_axis] = patch_lower[split_axis] + 1;
    coordinate[split_axis] = owner;
    boxes.push_back(Box<Dim>{patch_lower, patch_upper});
    owners.push_back(coordinate);
  }
  const BoxArray<Dim> layout(std::move(boxes));
  const Distribution<Dim> distribution = Distribution<Dim>::partitioned(
      layout, RankSpace<Dim>(Index<Dim>{}, rank_extent), std::move(owners));
  Index<Dim> local_rank{};
  local_rank[split_axis] = rank;

  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = ghosts[axis];
  AuxiliaryOutput<Dim> output{{"mpi.preflight", "input", "carrier", "value"}, contract(), shape};
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "mpi.preflight.input",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {output},
      {}});
  registry.seal();

  // Only rank zero carries an extra unregistered component.  The factory must reduce that local
  // policy error before creating any HaloExchange, so rank one cannot enter remote setup alone.
  AuxiliaryStorageGroups<Dim> groups;
  groups.groups.emplace(registry.storage_groups().front().identity,
                        MultiFab<Dim>(layout, distribution, local_rank, rank == 0 ? 2 : 1, ghosts));
  std::array<bool, Dim> periodic{};
  periodic.fill(false);
  bool threw = false;
  try {
    [[maybe_unused]] auto prepared = prepare_auxiliary_ghost_transport(
        groups, registry, domain, Geometry<Dim>::from_bounds(domain, lower, physical_upper),
        BoundaryTopology<Dim>::axis_periodic(periodic), &lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_min(threw ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(threw ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(registry.accepted_generation(), 0U);
  EXPECT_FALSE(registry.last_accepted_point("mpi.preflight.input").has_value());
}

template <int Dim>
void expect_group_contract_failures_are_collective(int rank, int ranks, int split_axis,
                                                   const ExecutionLane& lane) {
  constexpr int patch_count = 2;
  Index<Dim> upper{};
  Extent<Dim> rank_extent{};
  Extent<Dim> ghosts{};
  RealVector<Dim> lower{};
  RealVector<Dim> physical_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = axis == split_axis ? 2 * patch_count - 1 : 1;
    rank_extent[axis] = axis == split_axis ? ranks : 1;
    ghosts[axis] = 1;
    physical_upper[axis] = static_cast<Real>(upper[axis] + 1);
  }
  const Box<Dim> domain{Index<Dim>{}, upper};
  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  for (int owner = 0; owner < patch_count; ++owner) {
    Index<Dim> patch_lower{};
    Index<Dim> patch_upper = upper;
    Index<Dim> coordinate{};
    patch_lower[split_axis] = 2 * owner;
    patch_upper[split_axis] = patch_lower[split_axis] + 1;
    coordinate[split_axis] = owner;
    boxes.push_back(Box<Dim>{patch_lower, patch_upper});
    owners.push_back(coordinate);
  }
  const BoxArray<Dim> layout(std::move(boxes));
  const Distribution<Dim> distribution = Distribution<Dim>::partitioned(
      layout, RankSpace<Dim>(Index<Dim>{}, rank_extent), std::move(owners));
  Index<Dim> local_rank{};
  local_rank[split_axis] = rank;

  AuxiliaryStorageShape<Dim> primary_shape;
  AuxiliaryStorageShape<Dim> secondary_shape;
  for (int axis = 0; axis < Dim; ++axis) {
    primary_shape.halo[axis] = ghosts[axis];
    secondary_shape.halo[axis] = ghosts[axis];
  }
  secondary_shape.halo[split_axis] = 2;
  AuxiliaryOutput<Dim> primary{
      {"mpi.groups", "input", "primary", "value"}, contract(), primary_shape};
  AuxiliaryComponentContract secondary_contract = contract();
  secondary_contract.layout = "mpi-auxiliary-ghost-secondary";
  AuxiliaryOutput<Dim> secondary{
      {"mpi.groups", "input", "secondary", "value"}, secondary_contract, secondary_shape};
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "mpi.groups.primary",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {primary},
      {}});
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "mpi.groups.secondary",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {secondary},
      {}});
  registry.seal();
  std::array<bool, Dim> periodic{};
  periodic.fill(false);
  const auto make_groups = [&] {
    AuxiliaryStorageGroups<Dim> result;
    for (const auto& resolved : registry.storage_groups()) {
      Extent<Dim> resolved_ghosts{};
      for (int axis = 0; axis < Dim; ++axis)
        resolved_ghosts[axis] = resolved.shape.halo[axis];
      result.groups.emplace(
          resolved.identity,
          MultiFab<Dim>(layout, distribution, local_rank,
                        static_cast<int>(resolved.component_count), resolved_ghosts));
    }
    return result;
  };
  const auto expect_collective_rejection = [&](AuxiliaryStorageGroups<Dim>& groups) {
    bool threw = false;
    try {
      [[maybe_unused]] auto prepared = prepare_auxiliary_ghost_transport(
          groups, registry, domain, Geometry<Dim>::from_bounds(domain, lower, physical_upper),
          BoundaryTopology<Dim>::axis_periodic(periodic), &lane);
    } catch (const std::exception&) {
      threw = true;
    }
    EXPECT_EQ(all_reduce_min(threw ? 1L : 0L, lane.communicator()), 1L);
    EXPECT_EQ(all_reduce_max(threw ? 1L : 0L, lane.communicator()), 1L);
  };

  auto missing = make_groups();
  if (rank == 0)
    missing.groups.erase(missing.groups.begin());
  expect_collective_rejection(missing);

  auto reordered = make_groups();
  if (rank == 0) {
    auto first = reordered.groups.begin();
    auto second = std::next(first);
    std::swap(first->second, second->second);
  }
  expect_collective_rejection(reordered);
}

template <int Dim>
void expect_partitioned_auxiliary_ghosts(int rank, int ranks, int split_axis,
                                         const ExecutionLane& lane) {
  constexpr int patch_count = 2;
  Index<Dim> domain_upper{};
  Extent<Dim> rank_extent{};
  Extent<Dim> ghosts{};
  RealVector<Dim> physical_lower{};
  RealVector<Dim> physical_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    domain_upper[axis] = axis == split_axis ? 2 * patch_count - 1 : 2;
    rank_extent[axis] = axis == split_axis ? ranks : 1;
    ghosts[axis] = 1;
    physical_upper[axis] = static_cast<Real>(domain_upper[axis] + 1);
  }
  const Box<Dim> domain{Index<Dim>{}, domain_upper};

  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  boxes.reserve(static_cast<std::size_t>(patch_count));
  owners.reserve(static_cast<std::size_t>(patch_count));
  for (int owner = 0; owner < patch_count; ++owner) {
    Index<Dim> lower{};
    Index<Dim> upper = domain_upper;
    Index<Dim> coordinate{};
    lower[split_axis] = 2 * owner;
    upper[split_axis] = lower[split_axis] + 1;
    coordinate[split_axis] = owner;
    boxes.push_back(Box<Dim>{lower, upper});
    owners.push_back(coordinate);
  }
  const BoxArray<Dim> layout(std::move(boxes));
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution =
      Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
  Index<Dim> local_rank{};
  local_rank[split_axis] = rank;
  std::array<bool, Dim> periodic{};
  periodic.fill(false);
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(periodic);
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, physical_lower, physical_upper);

  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = ghosts[axis];
  AuxiliaryOutput<Dim> output{{"mpi.auxiliary", "input", "carrier", "value"},
                              contract(),
                              shape,
                              {AuxiliaryBoundaryPolicy::Kind::dirichlet, Real(5)}};
  ExactAuxiliaryRegistry<Dim> registry;
  registry.add(PreparedAuxiliaryProvider<Dim>{
      "mpi.auxiliary.input",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {output},
      {}});
  registry.seal();

  AuxiliaryStorageGroups<Dim> groups;
  const std::string identity = registry.storage_groups().front().identity;
  groups.groups.emplace(identity, MultiFab<Dim>(layout, distribution, local_rank, 1, ghosts));
  MultiFab<Dim>& field = *groups.find(identity);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    marshaling::for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
      Real value = Real(0);
      Real scale = Real(1);
      for (int axis = 0; axis < Dim; ++axis) {
        value += scale * static_cast<Real>(index[axis]);
        scale *= Real(10);
      }
      host(marshaling::storage_ordinal(fab, index, 0)) = value;
    });
    fab.copy_from_host(host);
  }

  refresh_auxiliary_group_ghosts(groups, registry, domain, geometry, topology, &lane);

  if (rank < patch_count) {
    const auto& fab = field.fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    Index<Dim> remote{};
    remote[split_axis] = rank == 0 ? 2 : 1;
    Real remote_value = Real(0);
    Real remote_scale = Real(1);
    for (int axis = 0; axis < Dim; ++axis) {
      remote_value += remote_scale * static_cast<Real>(remote[axis]);
      remote_scale *= Real(10);
    }
    EXPECT_EQ(host(marshaling::storage_ordinal(fab, remote, 0)), remote_value);

    Index<Dim> physical{};
    physical[split_axis] = rank == 0 ? -1 : 2 * patch_count;
    const int boundary = rank == 0 ? 0 : 2 * patch_count - 1;
    Real boundary_value = Real(boundary);
    for (int axis = 0; axis < split_axis; ++axis)
      boundary_value *= Real(10);
    EXPECT_EQ(host(marshaling::storage_ordinal(fab, physical, 0)), Real(10) - boundary_value);
  }
}

template <int Dim>
void expect_every_partition_axis(int rank, int ranks, const ExecutionLane& lane) {
  for (int split_axis = 0; split_axis < Dim; ++split_axis) {
    expect_partitioned_auxiliary_ghosts<Dim>(rank, ranks, split_axis, lane);
    expect_rank_local_launch_failure<Dim>(rank, ranks, split_axis, lane);
    expect_preparation_failure_is_collective<Dim>(rank, ranks, split_axis, lane);
    expect_group_contract_failures_are_collective<Dim>(rank, ranks, split_axis, lane);
  }
}

int run_mpi_auxiliary_ghost_fill(int argc, char** argv) {
  comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    const int rank = my_rank();
    const int ranks = n_ranks();
    if (ranks != 2 && ranks != 3) {
      ADD_FAILURE() << "MPI auxiliary ghost proof requires two or three ranks";
    } else {
      auto lane = ExecutionLane::duplicate_world_collectively("test.mpi.auxiliary-ghosts");
      expect_every_partition_axis<1>(rank, ranks, lane);
      expect_every_partition_axis<2>(rank, ranks, lane);
      expect_every_partition_axis<3>(rank, ranks, lane);
    }
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_auxiliary_ghost_fill, ExchangesRemoteAndPhysicalGhostsInOneTwoAndThreeDimensions) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_auxiliary_ghost_fill, "test_mpi_auxiliary_ghost_fill"),
            0);
}
