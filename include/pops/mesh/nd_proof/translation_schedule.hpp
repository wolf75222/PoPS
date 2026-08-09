/// @file
/// @brief Private MPI-free translation schedule proof over authenticated ND MultiFab metadata.

#pragma once

#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/nd_proof/local_neighbors.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::mesh::nd_proof {

struct TranslationScheduleBudget {
  std::size_t global_jobs;
  std::size_t peer_plans;
  std::size_t local_elements;
  std::size_t send_elements;
  std::size_t receive_elements;
  LocalNeighborWorkBudget neighbor;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class TranslationSchedule {
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "TranslationSchedule requires DefaultExecutionSpace access to MemorySpace");

 public:
  using execution_space = Kokkos::DefaultExecutionSpace;
  using execution_index_type = std::int64_t;
  using execution_policy =
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<execution_index_type>>;
  using rank_type = Index<Dim>;
  using layout_type = ::pops::mesh::BoxArray<Dim>;
  using distribution_type = ::pops::mesh::Distribution<Dim>;
  using rank_space_type = ::pops::mesh::RankSpace<Dim>;
  using multifab_type = ::pops::MultiFab<Dim, MemorySpace>;
  using buffer_type = Kokkos::View<Real*, MemorySpace>;

  struct Job {
    std::size_t ordinal = 0;
    std::size_t source_box = 0;
    std::size_t destination_box = 0;
    Box<Dim> destination_region{};
    std::array<std::int64_t, Dim> source_from_destination{};
    std::size_t offset = 0;
    std::size_t elements = 0;

    bool operator==(const Job&) const = default;
  };

  struct PeerPlan {
    rank_type peer{};
    std::vector<Job> jobs{};
    std::size_t elements = 0;

    bool operator==(const PeerPlan&) const = default;
  };

  /// Offset-free identity of one job in the globally canonical neighbor sequence.
  struct CanonicalJob {
    std::size_t ordinal = 0;
    std::size_t source_box = 0;
    std::size_t destination_box = 0;
    Box<Dim> destination_region{};
    std::array<std::int64_t, Dim> source_from_destination{};
    std::size_t elements = 0;

    bool operator==(const CanonicalJob&) const = default;
  };

  TranslationSchedule(const layout_type& layout, const distribution_type& distribution,
                      const Box<Dim>& domain, const PeriodicTopology<Dim>& topology,
                      Extent<Dim> ghosts, int ncomp, int first_component, int component_count,
                      rank_type local_rank, const Extent<Dim>& hash_bins, BoxHashBudget hash_budget,
                      TranslationScheduleBudget budget)
      : layout_(layout),
        distribution_(distribution),
        domain_(domain),
        topology_(topology),
        ghosts_(ghosts),
        ncomp_(ncomp),
        first_(first_component),
        count_(component_count),
        local_rank_(local_rank) {
    validate_metadata();

    LocalNeighborWorkBudget neighbor_budget = budget.neighbor;
    neighbor_budget.jobs = std::min(neighbor_budget.jobs, budget.global_jobs);
    const std::vector<LocalNeighborJob<Dim>> neighbors = enumerate_local_translation_neighbors(
        layout_, domain_, ghosts_, topology_, hash_bins, hash_budget, neighbor_budget);
    if (neighbors.size() > budget.global_jobs)
      throw std::length_error("nd_proof::TranslationSchedule global jobs exceed budget");

    // Phase one: validate every applicable job and every aggregate before materializing state.
    std::vector<PlannedJob> planned;
    reserve_exact(planned, neighbors.size(), "nd_proof::TranslationSchedule planned jobs");
    std::vector<CanonicalJob> planned_global_jobs;
    reserve_exact(planned_global_jobs, neighbors.size(),
                  "nd_proof::TranslationSchedule canonical global jobs");
    std::vector<PlannedPeer> planned_peers;
    std::size_t planned_local_jobs = 0;
    std::size_t planned_send_peers = 0;
    std::size_t planned_receive_peers = 0;
    std::size_t planned_local_elements = 0;
    std::size_t planned_send_elements = 0;
    std::size_t planned_receive_elements = 0;

    for (std::size_t ordinal = 0; ordinal < neighbors.size(); ++ordinal) {
      Job job = make_validated_job(neighbors[ordinal], ordinal);
      if (planned_global_jobs.size() >= planned_global_jobs.max_size())
        throw std::length_error(
            "nd_proof::TranslationSchedule canonical global job capacity exceeded");
      planned_global_jobs.push_back(CanonicalJob{job.ordinal, job.source_box, job.destination_box,
                                                 job.destination_region,
                                                 job.source_from_destination, job.elements});
      const JobKind kind = classify(job);
      if (kind == JobKind::irrelevant)
        continue;

      PlannedJob entry{std::move(job), kind, rank_type{}, 0};
      if (kind == JobKind::local) {
        checked_increment(planned_local_jobs,
                          "nd_proof::TranslationSchedule local job count overflows size_t");
        if (planned_local_jobs > planned.max_size())
          throw std::length_error("nd_proof::TranslationSchedule local job capacity exceeded");
        checked_add_into(planned_local_elements, entry.job.elements, budget.local_elements,
                         "nd_proof::TranslationSchedule local elements exceed budget");
        require_execution_count(
            planned_local_elements,
            "nd_proof::TranslationSchedule local prefix exceeds execution range");
      } else {
        entry.peer = peer_for(entry.job, kind);
        entry.peer_index = find_or_add_peer(entry.peer, kind, planned_peers, budget.peer_plans,
                                            planned_send_peers, planned_receive_peers);
        PlannedPeer& peer = planned_peers[entry.peer_index];
        checked_increment(peer.jobs,
                          "nd_proof::TranslationSchedule peer job count overflows size_t");
        checked_add_into(peer.elements, entry.job.elements, std::numeric_limits<std::size_t>::max(),
                         "nd_proof::TranslationSchedule peer elements overflow size_t");
        require_execution_count(
            peer.elements, "nd_proof::TranslationSchedule peer prefix exceeds execution range");
        if (kind == JobKind::send)
          checked_add_into(planned_send_elements, entry.job.elements, budget.send_elements,
                           "nd_proof::TranslationSchedule send elements exceed budget");
        else
          checked_add_into(planned_receive_elements, entry.job.elements, budget.receive_elements,
                           "nd_proof::TranslationSchedule receive elements exceed budget");
      }
      if (planned.size() >= planned.max_size())
        throw std::length_error("nd_proof::TranslationSchedule planned job capacity exceeded");
      planned.push_back(std::move(entry));
    }

    // Phase two: reserve exact capacities, assign checked prefixes, then publish all state at once.
    std::vector<Job> materialized_local;
    reserve_exact(materialized_local, planned_local_jobs,
                  "nd_proof::TranslationSchedule local jobs");
    std::vector<PeerPlan> materialized_send;
    std::vector<PeerPlan> materialized_receive;
    reserve_exact(materialized_send, planned_send_peers,
                  "nd_proof::TranslationSchedule send plans");
    reserve_exact(materialized_receive, planned_receive_peers,
                  "nd_proof::TranslationSchedule receive plans");
    materialize_peers(planned_peers, materialized_send, materialized_receive);

    std::size_t local_offset = 0;
    for (const PlannedJob& entry : planned) {
      Job job = entry.job;
      if (entry.kind == JobKind::local) {
        job.offset = local_offset;
        checked_add_into(local_offset, job.elements, planned_local_elements,
                         "nd_proof::TranslationSchedule local prefix exceeds plan");
        require_execution_count(
            local_offset, "nd_proof::TranslationSchedule local prefix exceeds execution range");
        materialized_local.push_back(std::move(job));
        continue;
      }
      PlannedPeer& peer = planned_peers[entry.peer_index];
      std::vector<PeerPlan>& plans =
          entry.kind == JobKind::send ? materialized_send : materialized_receive;
      PeerPlan& plan = plans[peer.materialized_index];
      job.offset = plan.elements;
      checked_add_into(plan.elements, job.elements, peer.elements,
                       "nd_proof::TranslationSchedule peer prefix exceeds plan");
      require_execution_count(plan.elements,
                              "nd_proof::TranslationSchedule peer prefix exceeds execution range");
      plan.jobs.push_back(std::move(job));
    }
    if (local_offset != planned_local_elements)
      throw std::logic_error("nd_proof::TranslationSchedule local materialization mismatch");
    validate_materialized_peers(materialized_send, planned_peers, JobKind::send);
    validate_materialized_peers(materialized_receive, planned_peers, JobKind::receive);
    sort_peers(materialized_send);
    sort_peers(materialized_receive);

    local_ = std::move(materialized_local);
    send_ = std::move(materialized_send);
    receive_ = std::move(materialized_receive);
    canonical_global_jobs_ = std::move(planned_global_jobs);
    local_elements_ = planned_local_elements;
    send_elements_ = planned_send_elements;
    receive_elements_ = planned_receive_elements;
    global_job_count_ = canonical_global_jobs_.size();
  }

  const layout_type& layout() const noexcept { return layout_; }
  const distribution_type& distribution() const noexcept { return distribution_; }
  const Box<Dim>& domain() const noexcept { return domain_; }
  const PeriodicTopology<Dim>& topology() const noexcept { return topology_; }
  const Extent<Dim>& ghosts() const noexcept { return ghosts_; }
  int ncomp() const noexcept { return ncomp_; }
  int first_component() const noexcept { return first_; }
  int component_count() const noexcept { return count_; }
  const rank_type& local_rank() const noexcept { return local_rank_; }

  const std::vector<Job>& local_jobs() const noexcept { return local_; }
  const std::vector<PeerPlan>& send_plans() const noexcept { return send_; }
  const std::vector<PeerPlan>& receive_plans() const noexcept { return receive_; }
  std::size_t global_job_count() const noexcept { return global_job_count_; }
  const std::vector<CanonicalJob>& canonical_global_jobs() const noexcept {
    return canonical_global_jobs_;
  }
  std::size_t local_job_count() const noexcept { return local_.size(); }
  std::size_t send_plan_count() const noexcept { return send_.size(); }
  std::size_t receive_plan_count() const noexcept { return receive_.size(); }
  std::size_t local_elements() const noexcept { return local_elements_; }
  std::size_t send_elements() const noexcept { return send_elements_; }
  std::size_t receive_elements() const noexcept { return receive_elements_; }

  const PeerPlan& send_plan(const rank_type& peer) const { return find_peer(send_, peer, "send"); }
  const PeerPlan& receive_plan(const rank_type& peer) const {
    return find_peer(receive_, peer, "receive");
  }

  /// Validates the exact MultiFab identity without launching a kernel or accessing field storage.
  void validate_fields(const multifab_type& fields) const { authenticate(fields); }

  void replay(multifab_type& fields) const {
    authenticate(fields);
    for (const Job& job : local_)
      copy(fields, job);
    Kokkos::fence();
  }

  void pack(const multifab_type& fields, const rank_type& peer, buffer_type buffer) const {
    authenticate(fields);
    const PeerPlan& plan = send_plan(peer);
    check_buffer(plan, buffer);
    for (const Job& job : plan.jobs)
      pack_job(fields, job, buffer);
    Kokkos::fence();
  }

  void unpack(multifab_type& fields, const rank_type& peer, buffer_type buffer) const {
    authenticate(fields);
    const PeerPlan& plan = receive_plan(peer);
    check_buffer(plan, buffer);
    for (const Job& job : plan.jobs)
      unpack_job(fields, job, buffer);
    Kokkos::fence();
  }

 private:
  enum class JobKind { irrelevant, local, send, receive };

  struct PlannedPeer {
    rank_type peer{};
    JobKind kind = JobKind::irrelevant;
    std::size_t jobs = 0;
    std::size_t elements = 0;
    std::size_t materialized_index = 0;
  };

  struct PlannedJob {
    Job job{};
    JobKind kind = JobKind::irrelevant;
    rank_type peer{};
    std::size_t peer_index = 0;
  };

  void validate_metadata() const {
    if (domain_.empty() || !distribution_.matches_layout(layout_))
      throw std::invalid_argument(
          "nd_proof::TranslationSchedule requires an exact non-empty layout identity");
    if (!distribution_.rank_space().contains(local_rank_) || ncomp_ < 1 || first_ < 0 ||
        count_ < 1 || first_ > ncomp_ - count_)
      throw std::invalid_argument("nd_proof::TranslationSchedule metadata is invalid");
    for (int axis = 0; axis < Dim; ++axis)
      if (ghosts_[axis] < 0)
        throw std::invalid_argument("nd_proof::TranslationSchedule ghosts must be non-negative");
    topology_.validate(domain_);
  }

  static void checked_increment(std::size_t& total, const char* operation) {
    if (total == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error(operation);
    ++total;
  }

  static void checked_add_into(std::size_t& total, std::size_t value, std::size_t limit,
                               const char* operation) {
    if (total > limit || value > limit - total)
      throw std::length_error(operation);
    total += value;
  }

  static void require_execution_count(std::size_t value, const char* operation) {
    if (value > static_cast<std::size_t>(std::numeric_limits<execution_index_type>::max()))
      throw std::overflow_error(operation);
  }

  template <class T>
  static void reserve_exact(std::vector<T>& values, std::size_t capacity, const char* operation) {
    if (capacity > values.max_size())
      throw std::length_error(operation);
    values.reserve(capacity);
  }

  Job make_validated_job(const LocalNeighborJob<Dim>& neighbor, std::size_t ordinal) const {
    if (neighbor.source_box >= layout_.size() || neighbor.destination_box >= layout_.size() ||
        neighbor.destination_region.empty())
      throw std::invalid_argument("nd_proof::TranslationSchedule neighbor metadata is invalid");
    const Box<Dim> grown_destination =
        periodicity_detail::grow_box(layout_[neighbor.destination_box], ghosts_);
    if (!grown_destination.contains(neighbor.destination_region))
      throw std::invalid_argument(
          "nd_proof::TranslationSchedule destination region is outside destination ghosts");
    const Box<Dim> source_region = periodicity_detail::translate_box<Dim>(
        neighbor.destination_region, neighbor.source_from_destination_translation,
        "nd_proof::TranslationSchedule source translation overflows int64_t");
    if (!layout_[neighbor.source_box].contains(source_region))
      throw std::invalid_argument(
          "nd_proof::TranslationSchedule translated source region is outside source valid box");
    return Job{ordinal,
               neighbor.source_box,
               neighbor.destination_box,
               neighbor.destination_region,
               neighbor.source_from_destination_translation,
               0,
               checked_elements(neighbor.destination_region)};
  }

  std::size_t checked_elements(const Box<Dim>& box) const {
    const std::int64_t cells = box.numPts();
    if (cells <= 0 || static_cast<std::uint64_t>(cells) > std::numeric_limits<std::size_t>::max() /
                                                              static_cast<std::size_t>(count_))
      throw std::overflow_error("nd_proof::TranslationSchedule element count overflows size_t");
    if (cells > std::numeric_limits<execution_index_type>::max() / count_)
      throw std::overflow_error(
          "nd_proof::TranslationSchedule element count exceeds execution index range");
    return static_cast<std::size_t>(cells) * static_cast<std::size_t>(count_);
  }

  JobKind classify(const Job& job) const {
    if (distribution_.replicated())
      return JobKind::local;
    const bool source_local = distribution_.owner(job.source_box) == local_rank_;
    const bool destination_local = distribution_.owner(job.destination_box) == local_rank_;
    if (source_local && destination_local)
      return JobKind::local;
    if (source_local)
      return JobKind::send;
    if (destination_local)
      return JobKind::receive;
    return JobKind::irrelevant;
  }

  rank_type peer_for(const Job& job, JobKind kind) const {
    if (kind == JobKind::send)
      return distribution_.owner(job.destination_box);
    if (kind == JobKind::receive)
      return distribution_.owner(job.source_box);
    throw std::logic_error("nd_proof::TranslationSchedule local jobs have no peer");
  }

  static std::size_t find_or_add_peer(const rank_type& peer, JobKind kind,
                                      std::vector<PlannedPeer>& peers, std::size_t budget,
                                      std::size_t& send_count, std::size_t& receive_count) {
    for (std::size_t index = 0; index < peers.size(); ++index)
      if (peers[index].kind == kind && peers[index].peer == peer)
        return index;
    if (peers.size() >= budget || peers.size() >= peers.max_size())
      throw std::length_error("nd_proof::TranslationSchedule peer plans exceed budget");
    if (kind == JobKind::send)
      checked_increment(send_count,
                        "nd_proof::TranslationSchedule send peer count overflows size_t");
    else if (kind == JobKind::receive)
      checked_increment(receive_count,
                        "nd_proof::TranslationSchedule receive peer count overflows size_t");
    else
      throw std::logic_error("nd_proof::TranslationSchedule invalid peer plan kind");
    peers.push_back(PlannedPeer{peer, kind});
    return peers.size() - 1;
  }

  static void materialize_peers(std::vector<PlannedPeer>& peers, std::vector<PeerPlan>& send,
                                std::vector<PeerPlan>& receive) {
    for (PlannedPeer& peer : peers) {
      std::vector<PeerPlan>& plans = peer.kind == JobKind::send ? send : receive;
      peer.materialized_index = plans.size();
      plans.push_back(PeerPlan{peer.peer});
      reserve_exact(plans.back().jobs, peer.jobs,
                    "nd_proof::TranslationSchedule peer job capacity exceeded");
    }
  }

  static void validate_materialized_peers(const std::vector<PeerPlan>& plans,
                                          const std::vector<PlannedPeer>& peers, JobKind kind) {
    for (const PlannedPeer& peer : peers) {
      if (peer.kind != kind)
        continue;
      const PeerPlan& plan = plans[peer.materialized_index];
      if (plan.elements != peer.elements || plan.jobs.size() != peer.jobs)
        throw std::logic_error("nd_proof::TranslationSchedule peer materialization mismatch");
    }
  }

  void sort_peers(std::vector<PeerPlan>& plans) const {
    std::sort(plans.begin(), plans.end(), [this](const PeerPlan& left, const PeerPlan& right) {
      return distribution_.rank_space().linear_rank(left.peer) <
             distribution_.rank_space().linear_rank(right.peer);
    });
  }

  const PeerPlan& find_peer(const std::vector<PeerPlan>& plans, const rank_type& peer,
                            const char* direction) const {
    const auto found = std::find_if(plans.begin(), plans.end(),
                                    [&](const PeerPlan& plan) { return plan.peer == peer; });
    if (found == plans.end())
      throw std::invalid_argument(std::string("nd_proof::TranslationSchedule has no ") + direction +
                                  " plan for peer coordinate");
    return *found;
  }

  void authenticate(const multifab_type& fields) const {
    if (!(fields.layout() == layout_) || !(fields.distribution() == distribution_) ||
        fields.local_rank() != local_rank_ || fields.ghosts() != ghosts_ ||
        fields.ncomp() != ncomp_)
      throw std::invalid_argument("nd_proof::TranslationSchedule MultiFab identity is stale");
  }

  static void check_buffer(const PeerPlan& plan, const buffer_type& buffer) {
    if (buffer.extent(0) != plan.elements)
      throw std::invalid_argument(
          "nd_proof::TranslationSchedule buffer size does not match peer plan");
  }

  struct KernelJob {
    int destination_lower[Dim]{};
    execution_index_type destination_extent[Dim]{};
    std::int64_t source_translation[Dim]{};
    int first_component = 0;
    int component_count = 0;
    execution_index_type cells_per_component = 0;
    execution_index_type offset = 0;
    execution_index_type elements = 0;
  };

  struct CopyKernel {
    FieldView<Real, Dim> destination{};
    FieldView<const Real, Dim> source{};
    KernelJob job{};

    KOKKOS_FUNCTION void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells_per_component);
      execution_index_type cell = element % job.cells_per_component;
      Index<Dim> destination_index{};
      Index<Dim> source_index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::int64_t coordinate =
            job.destination_lower[axis] + cell % job.destination_extent[axis];
        destination_index.values[axis] = static_cast<int>(coordinate);
        const std::int64_t translated = coordinate + job.source_translation[axis];
        source_index.values[axis] = static_cast<int>(translated);
        cell /= job.destination_extent[axis];
      }
      destination(destination_index, job.first_component + component) =
          source(source_index, job.first_component + component);
    }
  };

  struct PackKernel {
    buffer_type buffer{};
    FieldView<const Real, Dim> source{};
    KernelJob job{};

    KOKKOS_FUNCTION void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells_per_component);
      execution_index_type cell = element % job.cells_per_component;
      Index<Dim> source_index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::int64_t coordinate =
            job.destination_lower[axis] + cell % job.destination_extent[axis];
        const std::int64_t translated = coordinate + job.source_translation[axis];
        source_index.values[axis] = static_cast<int>(translated);
        cell /= job.destination_extent[axis];
      }
      buffer(job.offset + element) = source(source_index, job.first_component + component);
    }
  };

  struct UnpackKernel {
    buffer_type buffer{};
    FieldView<Real, Dim> destination{};
    KernelJob job{};

    KOKKOS_FUNCTION void operator()(execution_index_type element) const {
      const int component = static_cast<int>(element / job.cells_per_component);
      execution_index_type cell = element % job.cells_per_component;
      Index<Dim> destination_index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const std::int64_t coordinate =
            job.destination_lower[axis] + cell % job.destination_extent[axis];
        destination_index.values[axis] = static_cast<int>(coordinate);
        cell /= job.destination_extent[axis];
      }
      destination(destination_index, job.first_component + component) =
          buffer(job.offset + element);
    }
  };

  KernelJob lower_kernel_job(const Job& job) const {
    require_execution_count(job.elements,
                            "nd_proof::TranslationSchedule job exceeds execution index range");
    require_execution_count(
        job.offset, "nd_proof::TranslationSchedule job offset exceeds execution index range");
    KernelJob result{};
    result.first_component = first_;
    result.component_count = count_;
    result.cells_per_component =
        static_cast<execution_index_type>(job.elements / static_cast<std::size_t>(count_));
    result.offset = static_cast<execution_index_type>(job.offset);
    result.elements = static_cast<execution_index_type>(job.elements);
    for (int axis = 0; axis < Dim; ++axis) {
      result.destination_lower[axis] = job.destination_region.lo[axis];
      result.destination_extent[axis] = job.destination_region.length(axis);
      result.source_translation[axis] = job.source_from_destination[axis];
    }
    return result;
  }

  void copy(multifab_type& fields, const Job& job) const {
    const FieldView<const Real, Dim> source =
        static_cast<const multifab_type&>(fields).fab_global(job.source_box).view();
    const FieldView<Real, Dim> destination = fields.fab_global(job.destination_box).view();
    const KernelJob kernel_job = lower_kernel_job(job);
    Kokkos::parallel_for("pops_nd_translation_copy", execution_policy(0, kernel_job.elements),
                         CopyKernel{destination, source, kernel_job});
  }

  void pack_job(const multifab_type& fields, const Job& job, buffer_type buffer) const {
    const FieldView<const Real, Dim> source = fields.fab_global(job.source_box).view();
    const KernelJob kernel_job = lower_kernel_job(job);
    Kokkos::parallel_for("pops_nd_translation_pack", execution_policy(0, kernel_job.elements),
                         PackKernel{buffer, source, kernel_job});
  }

  void unpack_job(multifab_type& fields, const Job& job, buffer_type buffer) const {
    const FieldView<Real, Dim> destination = fields.fab_global(job.destination_box).view();
    const KernelJob kernel_job = lower_kernel_job(job);
    Kokkos::parallel_for("pops_nd_translation_unpack", execution_policy(0, kernel_job.elements),
                         UnpackKernel{buffer, destination, kernel_job});
  }

  layout_type layout_{};
  distribution_type distribution_{};
  Box<Dim> domain_{};
  PeriodicTopology<Dim> topology_{};
  Extent<Dim> ghosts_{};
  int ncomp_ = 0;
  int first_ = 0;
  int count_ = 0;
  rank_type local_rank_{};
  std::vector<Job> local_{};
  std::vector<PeerPlan> send_{};
  std::vector<PeerPlan> receive_{};
  std::size_t local_elements_ = 0;
  std::size_t send_elements_ = 0;
  std::size_t receive_elements_ = 0;
  std::vector<CanonicalJob> canonical_global_jobs_{};
  std::size_t global_job_count_ = 0;
};

}  // namespace pops::mesh::nd_proof
