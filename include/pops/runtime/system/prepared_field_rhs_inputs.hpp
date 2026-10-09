/// Host-only, invocation-scoped authority for a pointwise State/Auxiliary Field RHS.
#pragma once

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>
#include <pops/runtime/program/program_value_authority.hpp>
#include <Kokkos_Core.hpp>

#include <atomic>
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pops {
template <int Dim> class System;
template <int Dim> class AmrSystem;
}

namespace pops::runtime::system {

/// Separate accepted-carrier authority for native bootstrap/rematerialization. The runtime
/// seals actual registry membership and its session; callers cannot supply a pointer or proof.
template <int Dim>
class NativeAcceptedFieldSource {
 public:
  void validate() const {
    if (!validate_) throw std::logic_error("native accepted Field source is empty");
    validate_();
  }
  [[nodiscard]] const MultiFab<Dim>& state() const { validate(); return *state_; }
  [[nodiscard]] const multiblock::BoundaryEvaluationPoint& birth_point() const { validate(); return point_; }
  [[nodiscard]] int runtime_block() const { validate(); return block_; }
  [[nodiscard]] int level() const { validate(); return point_.level; }
 private:
  friend class ::pops::System<Dim>;
  friend class ::pops::AmrSystem<Dim>;
  NativeAcceptedFieldSource(std::shared_ptr<const MultiFab<Dim>> state,
      multiblock::BoundaryEvaluationPoint point, int block, std::function<void()> validate)
      : state_(std::move(state)), point_(std::move(point)), block_(block), validate_(std::move(validate)) {
    if (!state_ || block_ < 0 || point_.level < 0)
      throw std::invalid_argument("native accepted Field source lacks its actual carrier");
    this->validate();
  }
  std::shared_ptr<const MultiFab<Dim>> state_;
  multiblock::BoundaryEvaluationPoint point_;
  int block_ = -1;
  std::function<void()> validate_;
};

/// @2: a native context joins an opaque producer value to one exact Field edge.
/// The source birth point is retained separately from the consumer evaluation point.
template <int Dim>
class FieldSolveRequest {
 public:
  using authority_type = program::ProgramValueAuthority<Dim>;
  struct Source {
    std::variant<typename authority_type::Value, typename authority_type::AcceptedValue,
                 NativeAcceptedFieldSource<Dim>> proof;
    std::function<void()> validate_field_edge;
    void validate() const {
      std::visit([](const auto& value) { value.validate(); }, proof);
      if (!validate_field_edge)
        throw std::logic_error("Field solve source has no native Field-edge authority");
      validate_field_edge();
    }
    [[nodiscard]] const MultiFab<Dim>& state() const {
      validate();
      return std::visit([](const auto& value) -> const MultiFab<Dim>& { return value.state(); }, proof);
    }
    [[nodiscard]] int block() const {
      validate(); return std::visit([](const auto& value) { return value.runtime_block(); }, proof);
    }
    [[nodiscard]] int level() const {
      validate(); return std::visit([](const auto& value) { return value.level(); }, proof);
    }
    [[nodiscard]] std::string source_identity() const {
      validate();
      if (const auto* value = std::get_if<typename authority_type::Value>(&proof))
        return "pops.ProgramValueAuthority@1/ssa/" + std::to_string(value->source_ssa());
      if (std::holds_alternative<NativeAcceptedFieldSource<Dim>>(proof))
        return "pops.NativeAcceptedFieldSource@1";
      return "pops.ProgramValueAuthority@1/accepted";
    }
  };
  struct LegacySource {
    int block = -1, level = -1;
    std::shared_ptr<const MultiFab<Dim>> state;
  };
  [[nodiscard]] bool has_source(int block, int level) const {
    return std::any_of(sources_.begin(), sources_.end(), [&](const auto& source) {
      return source.block() == block && source.level() == level;
    });
  }
  [[nodiscard]] bool has_legacy_source(int block, int level) const noexcept {
    return std::any_of(legacy_sources_.begin(), legacy_sources_.end(), [&](const auto& source) {
      return source.block == block && source.level == level;
    });
  }
  [[nodiscard]] const MultiFab<Dim>& selected_state(int block, int level) const {
    for (const auto& source : sources_)
      if (source.block() == block && source.level() == level) return source.state();
    const MultiFab<Dim>* selected = nullptr;
    for (const auto& source : legacy_sources_)
      if (source.block == block && source.level == level) {
        if (!source.state || selected)
          throw std::invalid_argument("legacy Field source is empty or duplicated");
        selected = source.state.get();
      }
    if (!selected) throw std::invalid_argument("Field request lacks a selected source image");
    return *selected;
  }
  [[nodiscard]] const std::string& field() const noexcept { return field_; }
  [[nodiscard]] const multiblock::BoundaryEvaluationPoint& point() const noexcept { return point_; }
  [[nodiscard]] const std::vector<Source>& sources() const noexcept { return sources_; }
  [[nodiscard]] const Source& source(int block, int level) const {
    const Source* selected = nullptr;
    for (const auto& source : sources_)
      if (source.block() == block && source.level() == level) {
        if (selected) throw std::invalid_argument("Field request contains duplicate native sources");
        selected = &source;
      }
    if (!selected) throw std::invalid_argument("Field request lacks its actual native source level");
    return *selected;
  }
 private:
  friend class ::pops::System<Dim>;
  friend class ::pops::AmrSystem<Dim>;
  friend class program::ProgramContext<Dim>;
  template <int, class> friend class program::AmrProgramContext;
  FieldSolveRequest(std::string field, multiblock::BoundaryEvaluationPoint point,
                    std::vector<Source> sources, std::vector<LegacySource> legacy_sources = {})
      : field_(std::move(field)), point_(std::move(point)), sources_(std::move(sources)), legacy_sources_(std::move(legacy_sources)) {
    if (field_.empty() || sources_.empty())
      throw std::invalid_argument("native FieldSolveRequest@2 is incomplete");
    for (const auto& source : sources_) source.validate();
  }
  std::string field_;
  multiblock::BoundaryEvaluationPoint point_;
  std::vector<Source> sources_;
  std::vector<LegacySource> legacy_sources_;
};

inline FieldLogicalTimePoint field_rhs_logical_point(
    const multiblock::BoundaryEvaluationPoint& point) {
  FieldLogicalTimePoint logical;
  logical.time = static_cast<Real>(point.physical_time);
  logical.dt = static_cast<Real>(point.dt);
  logical.stage_slot = point.stage;
  logical.level = point.level;
  logical.step = point.tick;
  logical.substep = point.substep;
  logical.stage_fraction_numerator = point.stage_fraction.numerator;
  logical.stage_fraction_denominator = point.stage_fraction.denominator;
  return logical;
}

/// Runtime-owned quarantine. A failed fence never destroys allocations still used by a kernel.
/// The next admitted invocation drains this image before issuing a new lease.
class FieldRhsExecutionRecovery {
 public:
  void retain(std::shared_ptr<void> owner, std::function<void()> complete = {}) {
    owners_.push_back({std::move(owner), std::move(complete)});
  }
  void drain() {
    Kokkos::fence();
    for (const auto& owner : owners_) if (owner.complete) owner.complete();
    owners_.clear();
  }
  ~FieldRhsExecutionRecovery() noexcept {
    if (owners_.empty()) return;
    try { drain(); }
    catch (...) {
      // The backend could not prove completion even during runtime destruction. Preserve
      // allocations for backend teardown instead of releasing them under pending work.
      (void)new decltype(owners_)(std::move(owners_));
    }
  }
 private:
  struct Retained { std::shared_ptr<void> owner; std::function<void()> complete; };
  std::vector<Retained> owners_;
};

/// No public constructor accepts pointers or proof metadata. Only the owning native runtime
/// issues this value after its exact Field-source/point/level and collective preparation checks.
/// Copies retain storage, but never extend the right to execute beyond the issuing invocation.
template <int Dim>
class PreparedFieldRhsInputs {
 public:
  using field_type = MultiFab<Dim>;
  using execution_space = Kokkos::DefaultExecutionSpace;
  static constexpr unsigned input_contract_version = 2;

  [[nodiscard]] const field_type& state() const { require_live_(); return *state_; }
  [[nodiscard]] const Geometry<Dim>& frame() const { require_live_(); return frame_; }
  [[nodiscard]] const FieldLogicalTimePoint& point() const { require_live_(); return point_; }
  [[nodiscard]] const multiblock::BoundaryEvaluationPoint& boundary_point() const {
    require_live_(); return boundary_point_;
  }
  [[nodiscard]] int level() const { require_live_(); return point_.level; }
  [[nodiscard]] const std::string& binding_identity() const { require_live_(); return binding_; }
  [[nodiscard]] const std::string& source_identity() const { require_live_(); return source_; }
  [[nodiscard]] const std::string& consumer_qid() const { require_live_(); return consumer_; }
  [[nodiscard]] std::uint64_t topology_epoch() const { require_live_(); return topology_; }
  [[nodiscard]] std::uint64_t materialization_generation() const {
    require_live_(); return generation_;
  }
  [[nodiscard]] execution_space execution() const { require_live_(); return execution_; }
  [[nodiscard]] const ExecutionLane& lane() const { require_live_(); return *lane_; }
  /// The retained immutable lane is available for the refusal vote even if this input is stale.
  [[nodiscard]] const ExecutionLane& consensus_lane() const noexcept { return *lane_; }

  /// Register scratch/model/view owners before launching; the runtime releases them after drain.
  void retain_execution_owner(std::shared_ptr<void> owner) const {
    require_live_(); recovery_->retain(std::move(owner));
  }

  template <int Count>
  void validate_provider_inputs() const {
    require_live_();
    if (count_ != static_cast<std::size_t>(Count))
      throw std::invalid_argument("Field RHS provider count differs from its qualified input");
    require_pointwise_provider_groups<Dim, Count>(*state_, groups_.get(), plan_.get(),
                                                 "prepared Field RHS");
    for (std::size_t local = 0; local < state_->local_size(); ++local)
      (void)bind_provider_storage_view<Dim, Count>(plan_.get(), groups_.get(), local);
  }

  template <int Count>
  [[nodiscard]] ProviderStorageView<Dim, Count> provider_values_view(std::size_t local) const {
    require_live_();
    if (count_ != static_cast<std::size_t>(Count) || local >= state_->local_size())
      throw std::invalid_argument("Field RHS provider view differs from its qualified patch");
    return bind_provider_storage_view<Dim, Count>(plan_.get(), groups_.get(), local);
  }

 private:
  friend class ::pops::System<Dim>;
  friend class ::pops::AmrSystem<Dim>;

  struct InvocationLease {
    std::atomic<bool> active{true};
    std::atomic<bool> revoked{false};
  };

  PreparedFieldRhsInputs(
      std::shared_ptr<const field_type> state,
      std::shared_ptr<const AuxiliaryStorageGroups<Dim>> groups,
      std::shared_ptr<const ResolvedAuxiliaryConsumerPlan<Dim>> plan,
      Geometry<Dim> frame, FieldLogicalTimePoint point,
      multiblock::BoundaryEvaluationPoint boundary_point,
      std::string binding, std::string source, std::string consumer, std::size_t count,
      std::uint64_t topology, std::uint64_t generation, execution_space execution,
      std::shared_ptr<const ExecutionLane> lane, const std::shared_ptr<InvocationLease>& lease,
      std::function<void()> validate,
      std::shared_ptr<FieldRhsExecutionRecovery> recovery)
      : state_(std::move(state)), groups_(std::move(groups)), plan_(std::move(plan)),
        frame_(std::move(frame)), point_(point), boundary_point_(std::move(boundary_point)),
        binding_(std::move(binding)), source_(std::move(source)), consumer_(std::move(consumer)),
        count_(count), topology_(topology), generation_(generation), execution_(execution),
        lane_(std::move(lane)), lane_borrow_(std::make_shared<ExecutionLane::ImmutableBorrow>(lane_->borrow_immutably())),
        lease_(lease), validate_(std::move(validate)), recovery_(std::move(recovery)) {
    if (!state_ || !lease || !validate_ || !recovery_ || binding_.empty() || source_.empty() ||
        point_.level < 0 || point_.level != boundary_point_.level ||
        (count_ != 0 && (!groups_ || !plan_ || consumer_.empty() ||
                        plan_->consumer_qid != consumer_ || plan_->value_count() != count_)))
      throw std::invalid_argument("Field RHS input lacks its exact native issuer contract");
    require_live_();
  }

  void require_live_() const {
    const auto lease = lease_.lock();
    if (!lease || !lease->active.load(std::memory_order_acquire) ||
        lease->revoked.load(std::memory_order_acquire))
      throw std::logic_error("prepared Field RHS invocation is no longer live");
    validate_();
  }

  std::shared_ptr<const field_type> state_;
  std::shared_ptr<const AuxiliaryStorageGroups<Dim>> groups_;
  std::shared_ptr<const ResolvedAuxiliaryConsumerPlan<Dim>> plan_;
  Geometry<Dim> frame_;
  FieldLogicalTimePoint point_;
  multiblock::BoundaryEvaluationPoint boundary_point_;
  std::string binding_, source_, consumer_;
  std::size_t count_;
  std::uint64_t topology_, generation_;
  execution_space execution_;
  std::shared_ptr<const ExecutionLane> lane_;
  std::shared_ptr<ExecutionLane::ImmutableBorrow> lane_borrow_;
  std::weak_ptr<InvocationLease> lease_;
  std::function<void()> validate_;
  std::shared_ptr<FieldRhsExecutionRecovery> recovery_;
};

template <int Dim>
using FieldRhsCallbackV2 = std::function<void(const PreparedFieldRhsInputs<Dim>&, MultiFab<Dim>&)>;

}  // namespace pops::runtime::system
