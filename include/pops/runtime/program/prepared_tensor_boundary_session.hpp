/// @file
/// @brief Provider-authenticated boundary session for one Cartesian tensor Krylov operator.

#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops::runtime::program {

/// Exact ownership carried by one provider-qualified tensor boundary transaction.
struct PreparedTensorBoundaryAuthority {
  const void* program_owner = nullptr;
  const void* runtime_owner = nullptr;
  std::uintptr_t block_owner_identity = 0;
  std::string runtime_lane_identity;
  int program_block = -1;
  int runtime_block = -1;
  int level = -1;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
};

/// One immutable Cartesian tensor boundary transport prepared outside the Krylov hot loop.
///
/// Unlike PreparedScalarBoundarySession, this session never synthesizes a generic physical law.
/// It retains the exact PhysicalBoundaryConditions included in the hierarchy tensor provider's
/// authenticated build request. The owning prepared operator session refreshes its exact
/// evaluation-point value before each solve without replacing any spatial authority.
template <int Dim>
class PreparedTensorBoundarySession {
 private:
  struct UninitializedTag {};

 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedTensorBoundarySession only supports dimensions 1, 2, and 3");
  using field_type = MultiFab<Dim>;
  using point_type = runtime::multiblock::BoundaryEvaluationPoint;

  static std::shared_ptr<PreparedTensorBoundarySession> prepare(
      const Geometry<Dim>& geometry, PhysicalBoundaryConditions<Dim> conditions,
      const field_type& prototype, const ExecutionLane& lane, std::uint64_t generation,
      PreparedTensorBoundaryAuthority authority, const point_type& point) {
    std::shared_ptr<PreparedTensorBoundarySession> session;
    std::exception_ptr local_error;
    try {
      session = std::shared_ptr<PreparedTensorBoundarySession>(
          new PreparedTensorBoundarySession(UninitializedTag{}, geometry, std::move(conditions),
                                            lane, generation, std::move(authority), point));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program tensor boundary allocation failed collectively");
    }
    session->initialize_(prototype);
    return session;
  }

  PreparedTensorBoundarySession(const PreparedTensorBoundarySession&) = delete;
  PreparedTensorBoundarySession& operator=(const PreparedTensorBoundarySession&) = delete;
  PreparedTensorBoundarySession(PreparedTensorBoundarySession&&) = delete;
  PreparedTensorBoundarySession& operator=(PreparedTensorBoundarySession&&) = delete;

  const Geometry<Dim>& geometry() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& conditions() const noexcept { return conditions_; }
  const BoundaryTopology<Dim>& topology() const noexcept { return conditions_.topology(); }
  const PreparedTensorBoundaryAuthority& authority() const noexcept { return authority_; }
  const point_type& point() const noexcept { return point_; }
  const ExecutionLane& lane() const noexcept { return *lane_; }
  std::uint64_t generation() const noexcept { return generation_; }

  /// Collectively refresh the exact solve point without exposing a partially published value.
  void refresh_point(const point_type& point) {
    std::exception_ptr local_error;
    try {
      require_point_(point);
      if (point.clock != point_.clock || point.level != point_.level ||
          point.stage != point_.stage || point.stage_fraction != point_.stage_fraction ||
          point.graph_identity != point_.graph_identity ||
          point.rate_identity != point_.rate_identity ||
          point.application_identity != point_.application_identity)
        throw std::invalid_argument(
            "Program tensor boundary refresh changed its exact evaluation identity");
      require_point_capacity_(point);
      std::memset(&refresh_numeric_, 0, sizeof(refresh_numeric_));
      refresh_numeric_ = {point.tick,
                          point.level,
                          point.substep,
                          point.stage,
                          point.stage_fraction.numerator,
                          point.stage_fraction.denominator,
                          point.dt,
                          point.physical_time};
      std::memcpy(refresh_numeric_bytes_.data(), &refresh_numeric_, sizeof(refresh_numeric_));
      refresh_pairs_[0].second = point.clock;
      refresh_pairs_[1].second = point.graph_identity;
      refresh_pairs_[2].second = point.rate_identity;
      refresh_pairs_[3].second = point.application_identity;
      refresh_pairs_[4].second =
          std::string_view(refresh_numeric_bytes_.data(), refresh_numeric_bytes_.size());
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program tensor boundary point refresh failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            std::span<const ExactOrderedBytePair>(refresh_pairs_), *lane_))
      throw std::invalid_argument(
          "Program tensor boundary point refresh differs between prepared-lane ranks");
    copy_point_into_prepared_(point);
  }

  /// Rank-local, allocation-free validation used before the owning context enters consensus.
  void authenticate_field(const field_type& field) const {
    if (!schedule_ || !physical_)
      throw std::logic_error("Program tensor boundary session is not initialized");
    schedule_->authenticate(field);
    physical_->authenticate(field);
  }

  /// Execute the already-authenticated transport. No storage is materialized in this call.
  void fill(field_type& field) const {
    if (exchange_)
      exchange_->execute(field, *lane_);
    else
      pops::fill_boundary(field, *schedule_);
    fill_physical_boundary(field, *physical_);
  }

 private:
  PreparedTensorBoundarySession(UninitializedTag, const Geometry<Dim>& geometry,
                                PhysicalBoundaryConditions<Dim> conditions,
                                const ExecutionLane& lane, std::uint64_t generation,
                                PreparedTensorBoundaryAuthority authority, const point_type& point)
      : geometry_(geometry),
        conditions_(std::move(conditions)),
        authority_(std::move(authority)),
        point_(point),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        generation_(generation) {
    // The copied value may retain only its logical size.  The session is a cold object, so prime
    // the complete externally prepared envelope before it can be refreshed by a candidate step.
    prime_point_capacity_(point);
    refresh_pairs_[0].first = "program-tensor-boundary-clock-v1";
    refresh_pairs_[1].first = "program-tensor-boundary-graph-v1";
    refresh_pairs_[2].first = "program-tensor-boundary-rate-v1";
    refresh_pairs_[3].first = "program-tensor-boundary-application-v1";
    refresh_pairs_[4].first = "program-tensor-boundary-numeric-v1";
  }

  struct RefreshNumericWitness {
    std::int64_t tick = 0;
    std::int32_t level = 0;
    std::int32_t substep = 0;
    std::int32_t stage = 0;
    std::int64_t stage_numerator = 0;
    std::int64_t stage_denominator = 1;
    double dt = 0.0;
    double physical_time = 0.0;
  };

  static void require_string_capacity_(const std::string& destination, std::string_view source) {
    if (destination.capacity() < source.size())
      throw std::logic_error("Program tensor boundary point exceeds its prepared capacity");
  }

  void prime_point_capacity_(const point_type& source) {
    const auto prime = [](std::string& destination, const std::string& input) {
      if (destination.capacity() < input.capacity())
        destination.reserve(input.capacity());
    };
    prime(point_.clock, source.clock);
    prime(point_.graph_identity, source.graph_identity);
    prime(point_.rate_identity, source.rate_identity);
    prime(point_.application_identity, source.application_identity);
  }

  void require_point_capacity_(const point_type& source) const {
    require_string_capacity_(point_.clock, source.clock);
    require_string_capacity_(point_.graph_identity, source.graph_identity);
    require_string_capacity_(point_.rate_identity, source.rate_identity);
    require_string_capacity_(point_.application_identity, source.application_identity);
  }

  void copy_point_into_prepared_(const point_type& source) noexcept {
    point_.clock.assign(source.clock);
    point_.tick = source.tick;
    point_.level = source.level;
    point_.substep = source.substep;
    point_.stage = source.stage;
    point_.stage_fraction = source.stage_fraction;
    point_.dt = source.dt;
    point_.physical_time = source.physical_time;
    point_.graph_identity.assign(source.graph_identity);
    point_.rate_identity.assign(source.rate_identity);
    point_.application_identity.assign(source.application_identity);
  }

  void initialize_(const field_type& prototype) {
    std::exception_ptr local_error;
    std::string local_contract;
    try {
      if (generation_ == 0 || authority_.program_owner == nullptr ||
          authority_.runtime_owner == nullptr || authority_.block_owner_identity == 0 ||
          authority_.runtime_lane_identity.empty() || authority_.program_block < 0 ||
          authority_.runtime_block < 0 || authority_.level < 0)
        throw std::invalid_argument("Program tensor boundary authority is incomplete");
      require_point_(point_);
      for (int axis = 0; axis < Dim; ++axis)
        if (conditions_.spacing()[axis] != geometry_.spacing(axis))
          throw std::invalid_argument(
              "Program tensor boundary spacing differs from provider geometry");
      if (lane_->size() != static_cast<int>(prototype.rank_space().size()) ||
          lane_->rank() !=
              static_cast<int>(prototype.rank_space().linear_rank(prototype.local_rank())))
        throw std::invalid_argument(
            "Program tensor boundary lane differs from the prototype rank space");
      schedule_.emplace(prepare_halo_schedule(
          prototype, geometry_.domain(), conditions_.topology(),
          scalar_boundary_detail::halo_budget(prototype, geometry_, conditions_.topology())));
      physical_.emplace(prepare_physical_boundary(
          geometry_.domain(), prototype.ghosts(), conditions_,
          BoundaryScheduleBudget{scalar_boundary_detail::boundary_entry_budget(
              geometry_, prototype.ghosts(), conditions_.topology())}));
      ExactContractBuilder contract;
      contract.text("pops.program.tensor-boundary-session")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(std::int32_t{authority_.program_block})
          .scalar(std::int32_t{authority_.runtime_block})
          .scalar(std::int32_t{authority_.level})
          .scalar(authority_.topology_epoch)
          .scalar(authority_.materialization_generation)
          .scalar(generation_)
          .text(lane_->identity())
          .text(authority_.runtime_lane_identity);
      append_point_contract_(contract, point_);
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(geometry_.domain().lo[axis])
            .scalar(geometry_.domain().hi[axis])
            .scalar(geometry_.lower()[axis])
            .scalar(geometry_.upper()[axis])
            .scalar(conditions_.spacing()[axis])
            .scalar(prototype.rank_space().origin()[axis])
            .scalar(prototype.rank_space().extent()[axis])
            .scalar(prototype.ghosts()[axis]);
      for (int axis = 0; axis < Dim; ++axis)
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          const auto& topology = conditions_.topology().at(face);
          const PhysicalBoundaryFace& law = conditions_.at(face);
          contract.scalar(topology.kind)
              .scalar(topology.partner.axis)
              .scalar(topology.partner.side)
              .scalar(law.kind)
              .scalar(law.value)
              .scalar(law.alpha)
              .scalar(law.beta);
        }
      contract.scalar(prototype.distribution().mode())
          .scalar(std::int32_t{prototype.ncomp()})
          .sequence(prototype.layout().boxes(),
                    [](ExactContractBuilder& item, const Box<Dim>& patch) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                    })
          .sequence(prototype.distribution().owners(),
                    [](ExactContractBuilder& item, const Index<Dim>& owner) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(owner[axis]);
                    });
      local_contract = std::move(contract).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program tensor boundary preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("program-tensor-boundary"), std::string_view(local_contract)}},
            *lane_))
      throw std::invalid_argument(
          "Program tensor boundary authority differs between prepared-lane ranks");

    const bool distributed = all_reduce_max(schedule_->has_remote_jobs() ? 1L : 0L, *lane_) != 0;
    if (!distributed)
      return;
    local_error = nullptr;
    try {
      HaloExchangeContext context{};
      context.context_generation = generation_;
      context.schedule_generation = generation_;
      exchange_.emplace(*schedule_, *lane_, context);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Program tensor boundary exchange allocation failed collectively");
    }
  }

  void require_point_(const point_type& point) const {
    if (point.level != authority_.level || point.clock.empty() || point.tick < 0 ||
        point.substep < 0 || point.stage < 0 || !std::isfinite(point.dt) || !(point.dt > 0.0) ||
        !std::isfinite(point.physical_time) || point.stage_fraction.denominator <= 0 ||
        point.stage_fraction.numerator < 0 ||
        point.stage_fraction.numerator > point.stage_fraction.denominator)
      throw std::invalid_argument("Program tensor boundary evaluation point is invalid");
  }

  static void append_point_contract_(ExactContractBuilder& contract, const point_type& point) {
    contract.text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time)
        .text(point.graph_identity)
        .text(point.rate_identity)
        .text(point.application_identity);
  }

  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> conditions_;
  PreparedTensorBoundaryAuthority authority_;
  point_type point_;
  std::optional<HaloSchedule<Dim>> schedule_;
  std::optional<PreparedPhysicalBoundary<Dim>> physical_;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::uint64_t generation_ = 0;
  mutable std::optional<HaloExchange<Dim>> exchange_;
  RefreshNumericWitness refresh_numeric_{};
  std::array<char, sizeof(RefreshNumericWitness)> refresh_numeric_bytes_{};
  std::array<ExactOrderedBytePair, 5> refresh_pairs_{};
};

}  // namespace pops::runtime::program
