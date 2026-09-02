/// @file
/// @brief Transactional exact-rank halo and physical-boundary population for provider groups.
///
/// Auxiliary components deliberately do not share a global scratch slab.  This helper therefore
/// prepares one authenticated Cartesian halo schedule per resolved storage group, then applies
/// each component's declared physical policy.  The same implementation is used for InputAux,
/// DerivedAux, and field-output storage in both uniform and AMR runtimes.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <pops/runtime/system/exact_aux_registry.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::system {
namespace auxiliary_ghost_detail {

inline void rethrow_collective_failure(std::exception_ptr local_error, const ExecutionLane* lane,
                                       const char* message) {
  const long failed = local_error ? 1L : 0L;
  const long collective =
      lane ? all_reduce_max(failed, lane->communicator()) : all_reduce_max(failed);
  if (collective == 0)
    return;
  if ((!lane || lane->size() == 1) && local_error)
    std::rethrow_exception(local_error);
  throw std::runtime_error(message);
}

inline std::size_t checked_multiply(std::size_t left, std::size_t right, const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                     const BoundaryTopology<Dim>& topology) {
  const std::size_t patches = field.layout().size();
  const std::size_t pairs =
      checked_multiply(patches, patches, "auxiliary halo patch-pair budget overflows size_t");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t axis_images = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && field.ghosts()[axis] != 0) {
      const std::int64_t extent = domain.length(axis);
      if (extent <= 0)
        throw std::invalid_argument("auxiliary halo domain must be non-empty on periodic axes");
      const std::int64_t wraps = 1 + (field.ghosts()[axis] - 1) / extent;
      if (wraps < 0 ||
          static_cast<std::uint64_t>(wraps) > (std::numeric_limits<std::size_t>::max() - 1u) / 2u)
        throw std::overflow_error("auxiliary halo periodic-image budget overflows size_t");
      axis_images += 2u * static_cast<std::size_t>(wraps);
    }
    images = checked_multiply(images, axis_images,
                              "auxiliary halo periodic-image product overflows size_t");
  }
  const std::size_t work =
      checked_multiply(pairs, images, "auxiliary halo image work budget overflows size_t");
  const std::size_t jobs = checked_multiply(work, static_cast<std::size_t>(2 * Dim),
                                            "auxiliary halo job budget overflows size_t");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("auxiliary halo domain must be non-empty");
  const std::size_t elements = checked_multiply(
      checked_multiply(jobs, static_cast<std::size_t>(signed_cells),
                       "auxiliary halo element budget overflows size_t"),
      static_cast<std::size_t>(field.ncomp()), "auxiliary halo component budget overflows size_t");
  const std::size_t peers =
      checked_multiply(patches, std::size_t{2}, "auxiliary halo peer budget overflows size_t");
  return HaloScheduleBudget{mesh::BoxArrayValidationBudget{patches, pairs},
                            work,
                            jobs,
                            images,
                            peers,
                            elements,
                            elements,
                            elements};
}

template <int Dim>
constexpr std::size_t physical_boundary_entries() {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= 3;
  return result - 1;
}

template <int Dim>
PhysicalBoundaryConditions<Dim> physical_conditions(const BoundaryTopology<Dim>& topology,
                                                    const Geometry<Dim>& geometry,
                                                    const AuxiliaryBoundaryPolicy& policy) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      auto& law = faces[static_cast<std::size_t>(face.ordinal())];
      if (topology.is_periodic(face))
        continue;
      // ``inherit_topology`` inherits periodic transport from BoundaryTopology.  On a physical
      // face its neutral scalar continuation is first-order extrapolation: it is the only
      // formula-free completion which remains defined for an arbitrary provider value.
      if (policy.kind == AuxiliaryBoundaryPolicy::Kind::dirichlet) {
        law.kind = PhysicalBoundaryKind::dirichlet;
        law.value = *policy.value;
      } else {
        law.kind = PhysicalBoundaryKind::constant_extrapolation;
      }
    }
  }
  return PhysicalBoundaryConditions<Dim>{topology, faces, spacing};
}

template <int Dim>
std::vector<AuxiliaryBoundaryPolicy> component_policies(const ExactAuxiliaryRegistry<Dim>& registry,
                                                        std::string_view group_identity,
                                                        int component_count) {
  if (component_count < 1)
    throw std::invalid_argument("auxiliary group has no components for boundary preparation");
  std::vector<AuxiliaryBoundaryPolicy> result(static_cast<std::size_t>(component_count));
  std::vector<bool> assigned(static_cast<std::size_t>(component_count), false);
  for (std::size_t provider_index = 0; provider_index < registry.provider_count(); ++provider_index)
    for (const auto& output : registry.provider(provider_index).outputs()) {
      const auto address = registry.address_of(output.key);
      if (address.group != group_identity)
        continue;
      if (address.component >= result.size() || assigned[address.component])
        throw std::logic_error("auxiliary registry has an invalid resolved group component");
      result[address.component] = output.boundary;
      assigned[address.component] = true;
    }
  for (const bool value : assigned)
    if (!value)
      throw std::logic_error("auxiliary group has no boundary policy for one resolved component");
  return result;
}

template <int Dim>
std::string exact_group_binding_contract(const AuxiliaryStorageGroups<Dim>& groups,
                                         const ExactAuxiliaryRegistry<Dim>& registry,
                                         const Box<Dim>& domain,
                                         const BoundaryTopology<Dim>& topology) {
  const auto& resolved = registry.storage_groups();
  if (groups.groups.size() != resolved.size())
    throw std::invalid_argument("auxiliary carrier differs from its resolved storage-group set");
  ExactContractBuilder exact;
  exact.text("pops.auxiliary-ghost-group-binding")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(resolved.size()));
  for (int axis = 0; axis < Dim; ++axis)
    exact.scalar(std::int64_t{domain.lo[axis]})
        .scalar(std::int64_t{domain.hi[axis]})
        .scalar(topology.kind(Face<Dim>{axis, BoundarySide::lower}))
        .scalar(topology.kind(Face<Dim>{axis, BoundarySide::upper}));
  for (const auto& declaration : resolved) {
    const MultiFab<Dim>* group = groups.find(declaration.identity);
    if (group == nullptr || declaration.component_count != static_cast<std::size_t>(group->ncomp()))
      throw std::invalid_argument(
          "auxiliary carrier group identity or component count differs from its registry");
    for (int axis = 0; axis < Dim; ++axis)
      if (declaration.shape.halo[axis] != group->ghosts()[axis])
        throw std::invalid_argument(
            "auxiliary carrier group ghosts differ from its resolved storage shape");
    exact.text(declaration.identity);
    declaration.shape.serialize_exact(exact);
    exact.scalar(std::int32_t{group->ncomp()});
    for (int axis = 0; axis < Dim; ++axis)
      exact.scalar(std::int64_t{group->ghosts()[axis]});
    exact.sequence(group->layout().boxes(), [](ExactContractBuilder& item, const Box<Dim>& box) {
      for (int axis = 0; axis < Dim; ++axis)
        item.scalar(std::int64_t{box.lo[axis]}).scalar(std::int64_t{box.hi[axis]});
    });
    const auto& distribution = group->distribution();
    exact.scalar(distribution.mode());
    for (int axis = 0; axis < Dim; ++axis)
      exact.scalar(std::int64_t{distribution.rank_space().origin()[axis]})
          .scalar(std::int64_t{distribution.rank_space().extent()[axis]});
    exact.sequence(distribution.owners(), [](ExactContractBuilder& item, const Index<Dim>& owner) {
      for (int axis = 0; axis < Dim; ++axis)
        item.scalar(std::int64_t{owner[axis]});
    });
  }
  return std::move(exact).release();
}

inline bool exact_contract_agrees(std::string_view contract, const ExecutionLane* lane) {
  if (lane)
    return all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("pops.auxiliary-ghost-group-binding"), contract}}, *lane);
  return all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("pops.auxiliary-ghost-group-binding"), contract}});
}

}  // namespace auxiliary_ghost_detail

/// Reject a candidate containing non-finite values anywhere in its valid or ghost image.
/// Host-mirror allocation/copy is itself rank-local and fallible, so every rank completes that
/// inspection phase (or reports its failure) before the communicator enters the value consensus.
template <int Dim>
void require_finite_auxiliary_groups(const AuxiliaryStorageGroups<Dim>& groups,
                                     const ExecutionLane* lane, std::string_view label) {
  long local_nonfinite = 0;
  std::exception_ptr local_error;
  try {
    for (const auto& [_, carrier] : groups.groups)
      for (std::size_t local = 0; local < carrier.local_size(); ++local) {
        const Fab<Dim>& fab = carrier.fab(local);
        auto host = fab.create_host_mirror();
        fab.copy_to_host(host);
        marshaling::for_each_host_index(fab.grown_box(), [&](const Index<Dim>& index, std::size_t) {
          for (int component = 0; component < carrier.ncomp(); ++component)
            if (!std::isfinite(
                    static_cast<double>(host(marshaling::storage_ordinal(fab, index, component)))))
              local_nonfinite = 1;
        });
      }
  } catch (...) {
    local_error = std::current_exception();
  }
  auxiliary_ghost_detail::rethrow_collective_failure(
      local_error, lane, "auxiliary candidate finite-value inspection failed collectively");
  const long nonfinite = lane ? all_reduce_max(local_nonfinite, lane->communicator())
                              : all_reduce_max(local_nonfinite);
  if (nonfinite != 0)
    throw std::runtime_error(std::string(label) +
                             " rejected: candidate valid/ghost image contains non-finite values");
}

template <int Dim>
class PreparedAuxiliaryPhysicalBoundaries final {
 public:
  struct Group {
    std::string identity;
    std::vector<PreparedPhysicalBoundary<Dim>> components;
  };

  PreparedAuxiliaryPhysicalBoundaries() = default;
  PreparedAuxiliaryPhysicalBoundaries(std::vector<Group> groups, const ExecutionLane* lane,
                                      std::string binding_contract)
      : groups_(std::move(groups)), lane_(lane), binding_contract_(std::move(binding_contract)) {}

  [[nodiscard]] std::string_view collective_contract() const noexcept { return binding_contract_; }

  void execute(AuxiliaryStorageGroups<Dim>& groups) const {
    std::exception_ptr binding_error;
    try {
      if (groups.groups.size() != groups_.size())
        throw std::invalid_argument("prepared auxiliary physical group set changed");
      for (const Group& prepared : groups_) {
        const auto* group = groups.find(prepared.identity);
        if (group == nullptr ||
            prepared.components.size() != static_cast<std::size_t>(group->ncomp()))
          throw std::invalid_argument("prepared auxiliary physical binding changed");
      }
    } catch (...) {
      binding_error = std::current_exception();
    }
    auxiliary_ghost_detail::rethrow_collective_failure(
        binding_error, lane_, "prepared auxiliary physical binding failed collectively");
    for (const Group& prepared : groups_) {
      std::exception_ptr physical_error;
      try {
        MultiFab<Dim>* group = groups.find(prepared.identity);
        for (int component = 0; component < group->ncomp(); ++component)
          fill_physical_boundary(*group, prepared.components[static_cast<std::size_t>(component)],
                                 component, 1);
        ::pops::device_fence();
      } catch (...) {
        physical_error = std::current_exception();
      }
      auxiliary_ghost_detail::rethrow_collective_failure(
          physical_error, lane_, "prepared auxiliary physical boundary phase failed collectively");
    }
  }

 private:
  std::vector<Group> groups_;
  const ExecutionLane* lane_ = nullptr;
  std::string binding_contract_;
};

template <int Dim>
PreparedAuxiliaryPhysicalBoundaries<Dim> prepare_auxiliary_physical_boundaries(
    const AuxiliaryStorageGroups<Dim>& groups, const ExactAuxiliaryRegistry<Dim>& registry,
    const Box<Dim>& domain, const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
    const ExecutionLane* lane = nullptr) {
  using authority_type = PreparedAuxiliaryPhysicalBoundaries<Dim>;
  std::string binding_contract;
  std::vector<typename authority_type::Group> prepared;
  std::exception_ptr local_error;
  try {
    binding_contract =
        auxiliary_ghost_detail::exact_group_binding_contract(groups, registry, domain, topology);
    ExactContractBuilder exact;
    exact.text("pops.auxiliary-physical-boundaries")
        .scalar(std::uint32_t{1})
        .bytes(binding_contract);
    prepared.reserve(groups.groups.size());
    for (const auto& [identity, group] : groups.groups) {
      typename authority_type::Group item;
      item.identity = identity;
      const auto policies =
          auxiliary_ghost_detail::component_policies(registry, identity, group.ncomp());
      exact.text(identity).sequence(
          policies, [](ExactContractBuilder& item, const AuxiliaryBoundaryPolicy& policy) {
            policy.serialize_exact(item);
          });
      item.components.reserve(static_cast<std::size_t>(group.ncomp()));
      for (int component = 0; component < group.ncomp(); ++component) {
        const auto conditions = auxiliary_ghost_detail::physical_conditions(
            topology, geometry, policies[static_cast<std::size_t>(component)]);
        item.components.push_back(prepare_physical_boundary(
            domain, group.ghosts(), conditions,
            BoundaryScheduleBudget{auxiliary_ghost_detail::physical_boundary_entries<Dim>()}));
      }
      prepared.push_back(std::move(item));
    }
    binding_contract = std::move(exact).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  auxiliary_ghost_detail::rethrow_collective_failure(
      local_error, lane, "auxiliary physical-boundary preparation failed collectively");
  if (!auxiliary_ghost_detail::exact_contract_agrees(binding_contract, lane))
    throw std::invalid_argument(
        "auxiliary physical-boundary group contract differs across communicator ranks");
  return authority_type{std::move(prepared), lane, std::move(binding_contract)};
}

template <int Dim>
class PreparedAuxiliaryGhostTransport final {
 public:
  struct Runtime {
    std::optional<HaloSchedule<Dim>> schedule;
    std::optional<HaloExchange<Dim>> exchange;
  };

  struct Group {
    std::string identity;
    std::shared_ptr<Runtime> runtime;
    std::vector<PreparedPhysicalBoundary<Dim>> physical;
  };

  PreparedAuxiliaryGhostTransport() = default;
  PreparedAuxiliaryGhostTransport(std::vector<Group> groups, const ExecutionLane* lane,
                                  std::string binding_contract)
      : groups_(std::move(groups)), lane_(lane), binding_contract_(std::move(binding_contract)) {}

  PreparedAuxiliaryGhostTransport(const PreparedAuxiliaryGhostTransport&) = delete;
  PreparedAuxiliaryGhostTransport& operator=(const PreparedAuxiliaryGhostTransport&) = delete;
  PreparedAuxiliaryGhostTransport(PreparedAuxiliaryGhostTransport&&) noexcept = default;
  PreparedAuxiliaryGhostTransport& operator=(PreparedAuxiliaryGhostTransport&&) noexcept = default;

  void execute(AuxiliaryStorageGroups<Dim>& groups) const {
    std::exception_ptr local_error;
    try {
      if (groups.groups.size() != groups_.size())
        throw std::invalid_argument("prepared auxiliary ghost transport group set changed");
      for (const Group& prepared : groups_) {
        const auto* group = groups.find(prepared.identity);
        if (group == nullptr || !prepared.runtime || !prepared.runtime->schedule ||
            prepared.physical.size() != static_cast<std::size_t>(group->ncomp()))
          throw std::invalid_argument("prepared auxiliary ghost transport binding changed");
        prepared.runtime->schedule->authenticate(*group);
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    auxiliary_ghost_detail::rethrow_collective_failure(
        local_error, lane_, "prepared auxiliary ghost transport binding failed collectively");
    for (const Group& prepared : groups_) {
      auto* group = groups.find(prepared.identity);
      std::exception_ptr halo_error;
      try {
        if (prepared.runtime->exchange) {
          if (lane_ == nullptr)
            throw std::logic_error("prepared auxiliary ghost exchange lost its owning lane");
          fill_boundary(*group, *prepared.runtime->exchange, *lane_);
        } else {
          fill_boundary(*group, *prepared.runtime->schedule);
        }
      } catch (...) {
        halo_error = std::current_exception();
      }
      auxiliary_ghost_detail::rethrow_collective_failure(
          halo_error, lane_, "prepared auxiliary halo phase failed collectively");
      std::exception_ptr physical_error;
      try {
        for (int component = 0; component < group->ncomp(); ++component)
          fill_physical_boundary(*group, prepared.physical[static_cast<std::size_t>(component)],
                                 component, 1);
        ::pops::device_fence();
      } catch (...) {
        physical_error = std::current_exception();
      }
      auxiliary_ghost_detail::rethrow_collective_failure(
          physical_error, lane_, "prepared auxiliary physical boundary phase failed collectively");
    }
  }

 private:
  std::vector<Group> groups_;
  const ExecutionLane* lane_ = nullptr;
  std::string binding_contract_;
};

/// Prepare every storage-group schedule, exchange workspace, and component boundary law before
/// any candidate writes or MPI transport.  Local construction failures are reduced before the
/// first HaloExchange constructor, so one bad rank cannot strand a peer in point-to-point setup.
template <int Dim>
PreparedAuxiliaryGhostTransport<Dim> prepare_auxiliary_ghost_transport(
    const AuxiliaryStorageGroups<Dim>& groups, const ExactAuxiliaryRegistry<Dim>& registry,
    const Box<Dim>& domain, const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
    const ExecutionLane* lane = nullptr) {
  using transport_type = PreparedAuxiliaryGhostTransport<Dim>;
  std::vector<typename transport_type::Group> prepared;
  std::string binding_contract;
  std::exception_ptr local_error;
  try {
    binding_contract =
        auxiliary_ghost_detail::exact_group_binding_contract(groups, registry, domain, topology);
    prepared.reserve(groups.groups.size());
    for (const auto& [identity, group] : groups.groups) {
      typename transport_type::Group item;
      item.identity = identity;
      item.runtime = std::make_shared<typename transport_type::Runtime>();
      item.runtime->schedule.emplace(prepare_halo_schedule(
          group, domain, topology,
          auxiliary_ghost_detail::exact_halo_budget(group, domain, topology)));
      const auto policies =
          auxiliary_ghost_detail::component_policies(registry, identity, group.ncomp());
      item.physical.reserve(static_cast<std::size_t>(group.ncomp()));
      for (int component = 0; component < group.ncomp(); ++component) {
        const auto conditions = auxiliary_ghost_detail::physical_conditions(
            topology, geometry, policies[static_cast<std::size_t>(component)]);
        item.physical.push_back(prepare_physical_boundary(
            domain, group.ghosts(), conditions,
            BoundaryScheduleBudget{auxiliary_ghost_detail::physical_boundary_entries<Dim>()}));
      }
      prepared.push_back(std::move(item));
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  auxiliary_ghost_detail::rethrow_collective_failure(
      local_error, lane, "auxiliary ghost transport preparation failed collectively");
  if (!auxiliary_ghost_detail::exact_contract_agrees(binding_contract, lane))
    throw std::invalid_argument(
        "auxiliary ghost ordered group contract differs across communicator ranks");

  // HaloExchange has its own allocation and contract consensus. It is deliberately entered only
  // after every rank has completed the entirely local schedule/physical-boundary preflight above.
  std::uint64_t exchange_generation = 1;
  for (auto& item : prepared) {
    const long remote_any =
        lane ? all_reduce_max(item.runtime->schedule->has_remote_jobs() ? 1L : 0L,
                              lane->communicator())
             : all_reduce_max(item.runtime->schedule->has_remote_jobs() ? 1L : 0L);
    std::exception_ptr exchange_error;
    try {
      if (remote_any != 0) {
        if (lane == nullptr || !lane->active() || !lane->owns_communicator())
          throw std::logic_error(
              "auxiliary ghost transport has remote jobs but no active owning ExecutionLane");
        item.runtime->exchange.emplace(
            *item.runtime->schedule, *lane,
            HaloExchangeContext{.context_generation = exchange_generation,
                                .schedule_generation = exchange_generation});
      }
    } catch (...) {
      exchange_error = std::current_exception();
    }
    auxiliary_ghost_detail::rethrow_collective_failure(
        exchange_error, lane, "auxiliary halo exchange preparation failed collectively");
    if (exchange_generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("auxiliary halo exchange generation overflows uint64_t");
    ++exchange_generation;
  }
  return transport_type{std::move(prepared), lane, std::move(binding_contract)};
}

/// Convenience entry point for standalone/tests. Runtime paths retain a prepared transport and
/// invoke `execute` repeatedly instead.
template <int Dim>
void refresh_auxiliary_group_ghosts(AuxiliaryStorageGroups<Dim>& groups,
                                    const ExactAuxiliaryRegistry<Dim>& registry,
                                    const Box<Dim>& domain, const Geometry<Dim>& geometry,
                                    const BoundaryTopology<Dim>& topology,
                                    const ExecutionLane* lane = nullptr) {
  auto prepared =
      prepare_auxiliary_ghost_transport(groups, registry, domain, geometry, topology, lane);
  prepared.execute(groups);
}

}  // namespace pops::runtime::system
