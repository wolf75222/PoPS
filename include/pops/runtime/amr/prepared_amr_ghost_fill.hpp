/// @file
/// @brief Final prepared sparse AMR ghost population in exact dimensions 1, 2, and 3.

#pragma once

#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/coarse_fine_ghost_schedule.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

struct AmrGhostFillBudget {
  CoarseFineGhostScheduleBudget coarse_fine{};
  HaloScheduleBudget same_level{};
};

template <int Dim>
struct AmrGhostFillPreparation {
  int fine_level = -1;
  Box<Dim> coarse_domain{};
  Box<Dim> fine_domain{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  ::pops::amr::transfer::TransferKind interpolation_kind =
      ::pops::amr::transfer::TransferKind::CoarseFineGhostInterpolation;
  BoundaryTopology<Dim> topology{};
  std::uint64_t topology_generation = 0;
  std::uint64_t materialization_generation = 0;
  std::string field_identity{};
  AmrGhostFillBudget budget{};
};

namespace prepared_amr_ghost_detail {

inline std::uint64_t exchange_generation(std::uint64_t generation, const char* name) {
  if (generation == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(std::string(name) + " cannot be mapped to an exchange generation");
  return generation + 1;
}

template <int Dim>
void append_index(ExactContractBuilder& contract, const Index<Dim>& index) {
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int64_t{index[axis]});
}

template <int Dim>
void append_box(ExactContractBuilder& contract, const Box<Dim>& box) {
  append_index(contract, box.lo);
  append_index(contract, box.hi);
}

template <int Dim>
void append_extent(ExactContractBuilder& contract, const Extent<Dim>& extent) {
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int64_t{extent[axis]});
}

template <int Dim>
void append_distribution(ExactContractBuilder& contract,
                         const mesh::Distribution<Dim>& distribution) {
  contract.scalar(distribution.mode());
  append_index(contract, distribution.rank_space().origin());
  append_extent(contract, distribution.rank_space().extent());
  contract.sequence(distribution.layout().boxes(),
                    [](ExactContractBuilder& item, const Box<Dim>& box) { append_box(item, box); });
  contract.sequence(distribution.owners(), [](ExactContractBuilder& item, const Index<Dim>& owner) {
    append_index(item, owner);
  });
}

template <int Dim>
std::string exact_contract(const CoarseFineGhostSchedule<Dim>& schedule,
                           const BoundaryTopology<Dim>& topology,
                           const AmrGhostFillPreparation<Dim>& preparation,
                           std::string_view lane_identity) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-amr-ghost-fill")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .scalar(std::int32_t{preparation.fine_level})
      .scalar(preparation.interpolation_kind)
      .text(preparation.field_identity)
      .text(lane_identity)
      .scalar(preparation.topology_generation)
      .scalar(preparation.materialization_generation);
  append_box(contract, preparation.coarse_domain);
  append_box(contract, preparation.fine_domain);
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int32_t{preparation.ratio[axis]});
  append_extent(contract, schedule.ghosts());
  contract.scalar(std::int32_t{schedule.ncomp()});
  append_distribution(contract, schedule.coarse_distribution());
  append_distribution(contract, schedule.fine_distribution());
  for (const auto& face : topology.faces())
    contract.scalar(std::int32_t{face.face.axis})
        .scalar(face.face.side)
        .scalar(face.kind)
        .scalar(std::int32_t{face.partner.axis})
        .scalar(face.partner.side);
  contract.sequence(schedule.patch_plans(), [](ExactContractBuilder& item, const auto& patch) {
    item.scalar(static_cast<std::uint64_t>(patch.fine_patch));
    append_box(item, patch.coarse_staging_region);
    item.sequence(patch.fine_destination_regions,
                  [](ExactContractBuilder& region, const auto& interpolation) {
                    append_box(region, interpolation.destination);
                    append_index(region, interpolation.periodic_source_from_destination);
                  });
  });
  contract.sequence(schedule.canonical_jobs(), [](ExactContractBuilder& item, const auto& job) {
    item.scalar(static_cast<std::uint64_t>(job.coarse_patch))
        .scalar(static_cast<std::uint64_t>(job.fine_patch));
    append_index(item, job.destination_rank);
    append_box(item, job.coarse_region);
    append_index(item, job.source_from_destination);
    item.scalar(static_cast<std::uint64_t>(job.elements));
  });
  return std::move(contract).release();
}

template <int Dim, class MemorySpace>
std::vector<const Real*> storage_identity(const MultiFab<Dim, MemorySpace>& field) {
  std::vector<const Real*> result;
  result.reserve(field.local_size());
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result.push_back(field.fab(local).view().data);
  return result;
}

template <int Dim, class MemorySpace>
bool storage_matches(const MultiFab<Dim, MemorySpace>& field,
                     const std::vector<const Real*>& identity) noexcept {
  try {
    if (field.local_size() != identity.size())
      return false;
    for (std::size_t local = 0; local < field.local_size(); ++local)
      if (field.fab(local).view().data != identity[local])
        return false;
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace prepared_amr_ghost_detail

/// Prepared sparse-level ghost operation.
///
/// Execution has one fixed semantic order:
/// 1. gather the complete parent interpolation stencil into private per-child-patch storage;
/// 2. interpolate only child ghosts that remain inside the physical fine domain;
/// 3. overwrite same-level and periodic overlaps with the exact local/MPI halo schedule;
/// 4. leave every child ghost outside a physical face untouched for the boundary provider.
///
/// The object is copyable only as a shared immutable provider handle.  Its execution state is
/// serialized by the borrowed ExecutionLane; concurrent calls require separately prepared lanes.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedAmrGhostFill {
  static_assert(Dim >= 1 && Dim <= 3, "PreparedAmrGhostFill only supports dimensions 1, 2, and 3");
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "PreparedAmrGhostFill requires DefaultExecutionSpace access to MemorySpace");

 public:
  using field_type = MultiFab<Dim, MemorySpace>;

  PreparedAmrGhostFill() = default;

  [[nodiscard]] static PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.amr.sparse-ghost-fill", 2};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    require_prepared_();
    contract.bytes(state_->exact_contract);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(state_); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return state_ ? std::string_view(state_->exact_contract) : std::string_view{};
  }
  [[nodiscard]] int fine_level() const {
    require_prepared_();
    return state_->preparation.fine_level;
  }
  [[nodiscard]] bool has_remote_parent_jobs() const {
    require_prepared_();
    return state_->coarse_fine->has_remote_jobs();
  }
  [[nodiscard]] bool has_remote_same_level_jobs() const {
    require_prepared_();
    return state_->same_level->has_remote_jobs();
  }

  void operator()(field_type& fine,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    require_prepared_();
    if (point.level != state_->preparation.fine_level)
      throw std::invalid_argument("prepared AMR ghost provider received another hierarchy level");
    execute(fine, state_->preparation.topology_generation,
            state_->preparation.materialization_generation, *state_->lane);
  }

  void execute(field_type& fine, std::uint64_t topology_generation,
               std::uint64_t materialization_generation, const ExecutionLane& lane) const {
    require_prepared_();
    state_->execute(fine, topology_generation, materialization_generation, lane);
  }

 private:
  struct InterpolationSlot {
    Box<Dim> destination{};
    ::pops::amr::transfer::IndexMapping<Dim> mapping{};
    std::optional<::pops::amr::transfer::PreparedTransfer<Dim>> transfer{};
  };

  struct ScratchPatch {
    std::size_t fine_patch = 0;
    Fab<Dim, MemorySpace> coarse{};
    std::vector<InterpolationSlot> interpolations{};
  };

  using job_type = CoarseFineGhostJob<Dim>;
  using peer_plan_type = CoarseFineGhostPeerPlan<Dim>;
  using device_buffer_type = Kokkos::View<Real*, MemorySpace>;
  using pinned_buffer_type = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;
  using execution_index_type = std::int64_t;
  using execution_policy =
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<execution_index_type>>;

  struct KernelJob {
    int lower[Dim]{};
    execution_index_type extent[Dim]{};
    execution_index_type cells = 0;
    execution_index_type offset = 0;
    execution_index_type elements = 0;
    int source_from_destination[Dim]{};
  };

  struct PackKernel {
    device_buffer_type buffer{};
    FieldView<const Real, Dim> source{};
    KernelJob job{};

    POPS_HD void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells);
      execution_index_type cell = element % job.cells;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] = static_cast<int>(static_cast<execution_index_type>(job.lower[axis]) +
                                       cell % job.extent[axis] + job.source_from_destination[axis]);
        cell /= job.extent[axis];
      }
      buffer(job.offset + element) = source(index, component);
    }
  };

  struct UnpackKernel {
    device_buffer_type buffer{};
    FieldView<Real, Dim> destination{};
    KernelJob job{};

    POPS_HD void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells);
      execution_index_type cell = element % job.cells;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] = static_cast<int>(static_cast<execution_index_type>(job.lower[axis]) +
                                       cell % job.extent[axis]);
        cell /= job.extent[axis];
      }
      destination(index, component) = buffer(job.offset + element);
    }
  };

  struct PeerStorage {
    Index<Dim> coordinate{};
    int mpi_rank = 0;
    const peer_plan_type* send = nullptr;
    const peer_plan_type* receive = nullptr;
    device_buffer_type device_send{};
    device_buffer_type device_receive{};
    pinned_buffer_type host_send{};
    pinned_buffer_type host_receive{};
  };

  struct State {
    const field_type* coarse = nullptr;
    const ExecutionLane* lane = nullptr;
    std::optional<ExecutionLane::ImmutableBorrow> lane_borrow{};
    AmrGhostFillPreparation<Dim> preparation{};
    std::optional<CoarseFineGhostSchedule<Dim>> coarse_fine{};
    std::optional<HaloSchedule<Dim>> same_level{};
    std::optional<HaloExchange<Dim, MemorySpace>> same_level_exchange{};
    bool remote_parent_collective = false;
    std::vector<ScratchPatch> scratch{};
    std::vector<std::size_t> scratch_by_fine_patch{};
    device_buffer_type local_buffer{};
    std::vector<PeerStorage> peers{};
    std::vector<const Real*> coarse_storage{};
    std::string exact_contract{};
    bool sealed = false;
#ifdef POPS_HAS_MPI
    std::vector<MPI_Request> receive_requests{};
    std::vector<MPI_Request> send_requests{};
    std::vector<MPI_Status> receive_statuses{};
#endif

    static constexpr std::size_t no_scratch = std::numeric_limits<std::size_t>::max();

    void prepare_metadata(const field_type& coarse_field, field_type& fine_field,
                          AmrGhostFillPreparation<Dim> requested,
                          const ExecutionLane& requested_lane) {
      bool lane_ready = !requested_lane.identity().empty();
#ifdef POPS_HAS_MPI
      lane_ready = lane_ready && requested_lane.active();
#endif
      if (requested.fine_level < 1 || requested.field_identity.empty() || !lane_ready)
        throw std::invalid_argument(
            "prepared AMR ghost fill requires a fine level, field identity, and active lane");
      if (coarse_field.rank_space().size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          requested_lane.size() != static_cast<int>(coarse_field.rank_space().size()) ||
          requested_lane.rank() !=
              static_cast<int>(coarse_field.rank_space().linear_rank(fine_field.local_rank())))
        throw std::invalid_argument(
            "prepared AMR ghost fill lane differs from the field process-coordinate space");
      coarse = &coarse_field;
      lane = &requested_lane;
      preparation = std::move(requested);
      const auto provider =
          ::pops::amr::transfer::TransferProvider<Dim, ::pops::amr::transfer::Centering::Cell>(
              preparation.interpolation_kind);
      const auto capabilities = provider.capabilities();
      if (preparation.interpolation_kind !=
              ::pops::amr::transfer::TransferKind::CoarseFineGhostInterpolation &&
          preparation.interpolation_kind !=
              ::pops::amr::transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation)
        throw std::invalid_argument(
            "prepared AMR ghost fill requires an authenticated coarse/fine interpolation kind");
      coarse_fine.emplace(coarse_field, fine_field, preparation.coarse_domain,
                          preparation.fine_domain, preparation.ratio, preparation.topology,
                          capabilities.source_stencil_radius, preparation.budget.coarse_fine);
      same_level.emplace(
          prepare_halo_schedule(fine_field, preparation.fine_domain, preparation.topology,
                                HaloLayoutCoverage::sparse_level, preparation.budget.same_level));
      exact_contract = prepared_amr_ghost_detail::exact_contract(
          *coarse_fine, preparation.topology, preparation, requested_lane.identity());
    }

    void prepare_storage(field_type& fine_field) {
      lane_borrow.emplace(lane->borrow_immutably());
      scratch_by_fine_patch.assign(coarse_fine->fine_layout().size(), no_scratch);
      scratch.reserve(fine_field.local_size());
      for (const auto& plan : coarse_fine->patch_plans()) {
        if (!fine_field.contains_local(plan.fine_patch) || plan.coarse_staging_region.empty())
          continue;
        scratch_by_fine_patch[plan.fine_patch] = scratch.size();
        ScratchPatch patch{};
        patch.fine_patch = plan.fine_patch;
        patch.coarse =
            Fab<Dim, MemorySpace>(plan.coarse_staging_region, fine_field.ncomp(), Extent<Dim>{});
        patch.interpolations.reserve(plan.fine_destination_regions.size());
        scratch.push_back(std::move(patch));
      }

      for (const auto& plan : coarse_fine->patch_plans()) {
        const std::size_t scratch_index = scratch_by_fine_patch[plan.fine_patch];
        if (scratch_index == no_scratch)
          continue;
        ScratchPatch& patch = scratch[scratch_index];
        for (const auto& region : plan.fine_destination_regions) {
          ::pops::amr::transfer::IndexMapping<Dim> mapping{};
          mapping.coarse_origin = preparation.coarse_domain.lo;
          for (int axis = 0; axis < Dim; ++axis) {
            const std::int64_t origin =
                static_cast<std::int64_t>(preparation.fine_domain.lo[axis]) -
                region.periodic_source_from_destination[axis];
            if (origin < std::numeric_limits<int>::min() ||
                origin > std::numeric_limits<int>::max())
              throw std::overflow_error(
                  "prepared AMR ghost periodic interpolation origin exceeds native indices");
            mapping.fine_origin[axis] = static_cast<int>(origin);
          }
          patch.interpolations.push_back(
              InterpolationSlot{region.destination, mapping, std::nullopt});
        }
      }

      local_buffer = device_buffer_type("pops_amr_parent_local", coarse_fine->local_elements());
      prepare_peers();
      coarse_storage = prepared_amr_ghost_detail::storage_identity(*coarse);
    }

    void prepare_peers() {
      const auto& ranks = coarse_fine->fine_distribution().rank_space();
      const auto attach = [this, &ranks](const peer_plan_type& plan, bool is_send) {
        const std::size_t linear = ranks.linear_rank(plan.peer);
        if (linear > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            plan.elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::overflow_error("prepared AMR ghost peer payload exceeds MPI int range");
        auto found = std::find_if(peers.begin(), peers.end(), [linear](const PeerStorage& peer) {
          return peer.mpi_rank == static_cast<int>(linear);
        });
        if (found == peers.end()) {
          peers.push_back(PeerStorage{plan.peer, static_cast<int>(linear)});
          found = std::prev(peers.end());
        }
        const peer_plan_type*& slot = is_send ? found->send : found->receive;
        if (slot != nullptr)
          throw std::logic_error("prepared AMR ghost fill has a duplicate peer plan");
        slot = &plan;
      };
      peers.reserve(coarse_fine->send_plans().size() + coarse_fine->receive_plans().size());
      for (const auto& plan : coarse_fine->send_plans())
        attach(plan, true);
      for (const auto& plan : coarse_fine->receive_plans())
        attach(plan, false);
      std::sort(peers.begin(), peers.end(), [](const PeerStorage& left, const PeerStorage& right) {
        return left.mpi_rank < right.mpi_rank;
      });
      for (PeerStorage& peer : peers) {
        const std::size_t sends = peer.send == nullptr ? 0 : peer.send->elements;
        const std::size_t receives = peer.receive == nullptr ? 0 : peer.receive->elements;
        peer.device_send = device_buffer_type("pops_amr_parent_device_send", sends);
        peer.device_receive = device_buffer_type("pops_amr_parent_device_receive", receives);
        peer.host_send = pinned_buffer_type("pops_amr_parent_host_send", sends);
        peer.host_receive = pinned_buffer_type("pops_amr_parent_host_receive", receives);
      }
#ifdef POPS_HAS_MPI
      receive_requests.resize(coarse_fine->receive_plans().size(), MPI_REQUEST_NULL);
      send_requests.resize(coarse_fine->send_plans().size(), MPI_REQUEST_NULL);
      receive_statuses.resize(coarse_fine->receive_plans().size());
#endif
    }

    KernelJob lower(const job_type& job) const {
      if (job.elements == 0 ||
          job.elements >
              static_cast<std::size_t>(std::numeric_limits<execution_index_type>::max()) ||
          job.offset > static_cast<std::size_t>(std::numeric_limits<execution_index_type>::max()) ||
          job.elements % static_cast<std::size_t>(coarse_fine->ncomp()) != 0)
        throw std::overflow_error("prepared AMR ghost job exceeds the execution index range");
      KernelJob result{};
      result.cells = static_cast<execution_index_type>(
          job.elements / static_cast<std::size_t>(coarse_fine->ncomp()));
      result.offset = static_cast<execution_index_type>(job.offset);
      result.elements = static_cast<execution_index_type>(job.elements);
      for (int axis = 0; axis < Dim; ++axis) {
        result.lower[axis] = job.coarse_region.lo[axis];
        result.extent[axis] = job.coarse_region.length(axis);
        result.source_from_destination[axis] = job.source_from_destination[axis];
      }
      return result;
    }

    void pack(const std::vector<job_type>& jobs, device_buffer_type buffer) const {
      for (const job_type& job : jobs) {
        const KernelJob lowered = lower(job);
        Kokkos::parallel_for(
            "pops_amr_parent_pack", execution_policy(0, lowered.elements),
            PackKernel{buffer, coarse->fab_global(job.coarse_patch).view(), lowered});
      }
    }

    void unpack(const std::vector<job_type>& jobs, device_buffer_type buffer) {
      for (const job_type& job : jobs) {
        const std::size_t scratch_index = scratch_by_fine_patch.at(job.fine_patch);
        if (scratch_index == no_scratch)
          throw std::logic_error("prepared AMR ghost receive has no local scratch owner");
        const KernelJob lowered = lower(job);
        Kokkos::parallel_for("pops_amr_parent_unpack", execution_policy(0, lowered.elements),
                             UnpackKernel{buffer, scratch[scratch_index].coarse.view(), lowered});
      }
    }

    bool gate(long failure) const noexcept {
      try {
        return all_reduce_max(failure, lane->communicator()) == 0;
      } catch (...) {
        return false;
      }
    }

    void require_binding(field_type& requested_fine, std::uint64_t topology_generation,
                         std::uint64_t materialization_generation,
                         const ExecutionLane& requested_lane) const {
      long invalid = sealed || &requested_lane != lane ||
                             topology_generation != preparation.topology_generation ||
                             materialization_generation != preparation.materialization_generation ||
                             !prepared_amr_ghost_detail::storage_matches(*coarse, coarse_storage)
                         ? 1L
                         : 0L;
      try {
        coarse_fine->authenticate(*coarse, requested_fine);
        same_level->authenticate(requested_fine);
      } catch (...) {
        invalid = 1;
      }
      if (!gate(invalid))
        throw std::invalid_argument(
            "prepared AMR ghost fill is stale for the requested hierarchy materialization");
    }

    void rebind_parent_interpolations(field_type& requested_fine) {
      long binding_failure = 0;
      std::exception_ptr binding_error;
      try {
        const auto provider =
            ::pops::amr::transfer::TransferProvider<Dim, ::pops::amr::transfer::Centering::Cell>(
                preparation.interpolation_kind);
        const auto components = ::pops::amr::transfer::ComponentRange{0, 0, coarse_fine->ncomp()};
        for (ScratchPatch& patch : scratch) {
          const auto source = std::as_const(patch.coarse).view();
          auto destination = requested_fine.fab_global(patch.fine_patch).view();
          for (InterpolationSlot& slot : patch.interpolations)
            slot.transfer.emplace(provider.prepare(source, destination, slot.destination,
                                                   preparation.ratio, slot.mapping, components));
        }
      } catch (...) {
        binding_failure = 1;
        binding_error = std::current_exception();
      }
      if (!gate(binding_failure)) {
        release_parent_interpolations();
        if (binding_error)
          std::rethrow_exception(binding_error);
        throw std::runtime_error("prepared AMR ghost interpolation rebinding failed collectively");
      }
    }

    void release_parent_interpolations() noexcept {
      for (ScratchPatch& patch : scratch)
        for (InterpolationSlot& slot : patch.interpolations)
          slot.transfer.reset();
    }

    void gather_parent() {
      long staging_failure = 0;
      try {
        pack(coarse_fine->local_jobs(), local_buffer);
        for (PeerStorage& peer : peers)
          if (peer.send != nullptr) {
            pack(peer.send->jobs, peer.device_send);
            Kokkos::deep_copy(peer.host_send, peer.device_send);
          }
        Kokkos::fence();
      } catch (...) {
        staging_failure = 1;
      }
      if (!gate(staging_failure))
        throw std::runtime_error("prepared AMR ghost parent packing failed before MPI publication");

#ifdef POPS_HAS_MPI
      if (remote_parent_collective)
        exchange_parent();
#else
      if (remote_parent_collective)
        throw std::logic_error("distributed AMR ghost fill requires an MPI build");
#endif

      long unpack_failure = 0;
      try {
        unpack(coarse_fine->local_jobs(), local_buffer);
        for (PeerStorage& peer : peers)
          if (peer.receive != nullptr)
            unpack(peer.receive->jobs, peer.device_receive);
        Kokkos::fence();
      } catch (...) {
        unpack_failure = 1;
      }
      if (!gate(unpack_failure))
        throw std::runtime_error(
            "prepared AMR ghost parent staging failed before child publication");
    }

#ifdef POPS_HAS_MPI
    static int wait_all(std::vector<MPI_Request>& requests) noexcept {
      if (requests.empty())
        return MPI_SUCCESS;
      if (requests.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return MPI_ERR_COUNT;
      return MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }

    bool drain() noexcept {
      bool safe = true;
      for (MPI_Request& request : receive_requests)
        if (request != MPI_REQUEST_NULL && MPI_Cancel(&request) != MPI_SUCCESS)
          safe = false;
      if (wait_all(send_requests) != MPI_SUCCESS || wait_all(receive_requests) != MPI_SUCCESS)
        safe = false;
      const auto null = [](MPI_Request request) { return request == MPI_REQUEST_NULL; };
      return safe && std::all_of(send_requests.begin(), send_requests.end(), null) &&
             std::all_of(receive_requests.begin(), receive_requests.end(), null);
    }

    [[noreturn]] void unsafe_exchange_failure(const char* message) {
      sealed = true;
      if (!drain())
        std::terminate();
      throw std::runtime_error(message);
    }

    void exchange_parent() {
      if (!lane->owns_communicator())
        throw std::logic_error(
            "distributed AMR ghost fill requires an owning duplicated ExecutionLane");
      std::fill(receive_requests.begin(), receive_requests.end(), MPI_REQUEST_NULL);
      std::fill(send_requests.begin(), send_requests.end(), MPI_REQUEST_NULL);
      std::size_t receive_index = 0;
      int post_code = MPI_SUCCESS;
      for (PeerStorage& peer : peers)
        if (peer.receive != nullptr) {
          if (post_code == MPI_SUCCESS)
            post_code =
                MPI_Irecv(peer.host_receive.data(), static_cast<int>(peer.receive->elements),
                          MPI_DOUBLE, peer.mpi_rank, ExecutionLane::parallel_copy_message_tag,
                          lane->native_handle(), &receive_requests[receive_index]);
          ++receive_index;
        }
      if (!gate(post_code == MPI_SUCCESS ? 0L : 1L))
        unsafe_exchange_failure("prepared AMR ghost receive posting failed collectively");

      std::size_t send_index = 0;
      post_code = MPI_SUCCESS;
      for (PeerStorage& peer : peers)
        if (peer.send != nullptr) {
          if (post_code == MPI_SUCCESS)
            post_code =
                MPI_Isend(peer.host_send.data(), static_cast<int>(peer.send->elements), MPI_DOUBLE,
                          peer.mpi_rank, ExecutionLane::parallel_copy_message_tag,
                          lane->native_handle(), &send_requests[send_index]);
          ++send_index;
        }
      if (!gate(post_code == MPI_SUCCESS ? 0L : 1L))
        unsafe_exchange_failure("prepared AMR ghost send posting failed collectively");

      int wait_code = wait_all(send_requests);
      if (wait_code == MPI_SUCCESS && !receive_requests.empty())
        wait_code = MPI_Waitall(static_cast<int>(receive_requests.size()), receive_requests.data(),
                                receive_statuses.data());
      if (wait_code == MPI_SUCCESS) {
        std::size_t status = 0;
        for (const PeerStorage& peer : peers)
          if (peer.receive != nullptr) {
            int count = MPI_UNDEFINED;
            const MPI_Status& received = receive_statuses[status++];
            if (MPI_Get_count(&received, MPI_DOUBLE, &count) != MPI_SUCCESS ||
                received.MPI_SOURCE != peer.mpi_rank ||
                received.MPI_TAG != ExecutionLane::parallel_copy_message_tag ||
                count != static_cast<int>(peer.receive->elements)) {
              wait_code = MPI_ERR_OTHER;
              break;
            }
          }
      }
      if (!gate(wait_code == MPI_SUCCESS ? 0L : 1L))
        unsafe_exchange_failure(
            "prepared AMR ghost receive authentication or MPI wait failed collectively");

      long copy_failure = 0;
      try {
        for (PeerStorage& peer : peers)
          if (peer.receive != nullptr)
            Kokkos::deep_copy(peer.device_receive, peer.host_receive);
        Kokkos::fence();
      } catch (...) {
        copy_failure = 1;
      }
      if (!gate(copy_failure)) {
        sealed = true;
        throw std::runtime_error(
            "prepared AMR ghost receive staging failed collectively after a safe MPI drain");
      }
    }
#endif

    void interpolate_parent_ghosts() {
      long publication_failure = 0;
      try {
        for (ScratchPatch& patch : scratch)
          for (const InterpolationSlot& slot : patch.interpolations) {
            if (!slot.transfer)
              throw std::logic_error(
                  "prepared AMR ghost interpolation was not rebound to its destination");
            for_each_cell(slot.transfer->destination_region(), *slot.transfer);
          }
        Kokkos::fence();
      } catch (...) {
        publication_failure = 1;
      }
      if (!gate(publication_failure)) {
        sealed = true;
        std::terminate();
      }
    }

    void execute(field_type& requested_fine, std::uint64_t topology_generation,
                 std::uint64_t materialization_generation, const ExecutionLane& requested_lane) {
      require_binding(requested_fine, topology_generation, materialization_generation,
                      requested_lane);
      rebind_parent_interpolations(requested_fine);
      try {
        gather_parent();
        if (same_level_exchange)
          same_level_exchange->begin(requested_fine, requested_lane);
        interpolate_parent_ghosts();
        if (same_level_exchange)
          same_level_exchange->complete(requested_fine, requested_lane);
        else
          fill_boundary(requested_fine, *same_level);
      } catch (...) {
        release_parent_interpolations();
        throw;
      }
      release_parent_interpolations();
    }
  };

  explicit PreparedAmrGhostFill(std::shared_ptr<State> state) : state_(std::move(state)) {}

  void require_prepared_() const {
    if (!state_)
      throw std::logic_error("cannot use an empty prepared AMR ghost fill");
  }

  template <int D, class M>
  friend PreparedAmrGhostFill<D, M> prepare_amr_ghost_fill(const MultiFab<D, M>&, MultiFab<D, M>&,
                                                           AmrGhostFillPreparation<D>,
                                                           const ExecutionLane&);

  std::shared_ptr<State> state_{};
};

/// Collective preparation boundary. All potentially rank-local allocation failures are reduced
/// before the next collective object is materialized.
template <int Dim, class MemorySpace>
PreparedAmrGhostFill<Dim, MemorySpace> prepare_amr_ghost_fill(
    const MultiFab<Dim, MemorySpace>& coarse, MultiFab<Dim, MemorySpace>& fine,
    AmrGhostFillPreparation<Dim> preparation, const ExecutionLane& lane) {
  using provider_type = PreparedAmrGhostFill<Dim, MemorySpace>;
  using state_type = typename provider_type::State;
  std::shared_ptr<state_type> state;
  long metadata_failure = 0;
  std::exception_ptr metadata_error;
  try {
    state = std::make_shared<state_type>();
    state->prepare_metadata(coarse, fine, std::move(preparation), lane);
  } catch (...) {
    metadata_failure = 1;
    metadata_error = std::current_exception();
  }
  if (all_reduce_max(metadata_failure, lane.communicator()) != 0) {
    if (metadata_error)
      std::rethrow_exception(metadata_error);
    throw std::runtime_error(
        "prepared AMR ghost metadata or schedule construction failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs({{std::string_view("pops-prepared-amr-ghost-fill"),
                                                  std::string_view(state->exact_contract)}},
                                                lane.communicator()))
    throw std::invalid_argument(
        "prepared AMR ghost exact topology/materialization contract differs across ranks");
  state->remote_parent_collective =
      all_reduce_max(state->coarse_fine->has_remote_jobs() ? 1L : 0L, lane.communicator()) != 0;

  long allocation_failure = 0;
  std::exception_ptr allocation_error;
  try {
    state->prepare_storage(fine);
  } catch (...) {
    allocation_failure = 1;
    allocation_error = std::current_exception();
  }
  if (all_reduce_max(allocation_failure, lane.communicator()) != 0) {
    if (allocation_error)
      std::rethrow_exception(allocation_error);
    throw std::runtime_error("prepared AMR ghost reusable storage allocation failed collectively");
  }

  const long remote_same_level_any =
      all_reduce_max(state->same_level->has_remote_jobs() ? 1L : 0L, lane.communicator());
  if (remote_same_level_any != 0) {
    HaloExchangeContext context{};
    context.context_generation = prepared_amr_ghost_detail::exchange_generation(
        state->preparation.topology_generation, "AMR topology generation");
    context.schedule_generation = prepared_amr_ghost_detail::exchange_generation(
        state->preparation.materialization_generation, "AMR materialization generation");
    state->same_level_exchange.emplace(*state->same_level, lane, context);
  }
  return provider_type(std::move(state));
}

}  // namespace pops::runtime::amr
