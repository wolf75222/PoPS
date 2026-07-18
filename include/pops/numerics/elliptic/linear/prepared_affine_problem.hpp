#pragma once

/// @file
/// @brief Prepared, snapshot-authenticated affine linear problems.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/prepared_vector_metric.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/prepared_provider_consensus.hpp>

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

class PreparedAffineLinearProblem;
namespace runtime::program {
class ProgramContext;
class AmrProgramContext;
}  // namespace runtime::program
namespace detail {
struct PreparedProblemAccess;

/// Process-local identity of one concrete provider source.  Exact contracts authenticate semantic
/// agreement across ranks; this opaque shared-owner identity answers the separate lifecycle
/// question "can this already materialized session be refreshed in place?" without exposing or
/// comparing a raw address.  Keeping a weak owner also prevents allocator address reuse from making
/// two successive provider implementations appear identical.
class PreparedProviderSourceIdentity {
 public:
  PreparedProviderSourceIdentity() = default;
  explicit PreparedProviderSourceIdentity(const std::shared_ptr<const void>& owner) noexcept
      : owner_(owner) {}

  [[nodiscard]] explicit operator bool() const noexcept {
    const std::weak_ptr<const void> empty;
    return owner_.owner_before(empty) || empty.owner_before(owner_);
  }

  friend bool operator==(const PreparedProviderSourceIdentity& left,
                         const PreparedProviderSourceIdentity& right) noexcept {
    return !left.owner_.owner_before(right.owner_) && !right.owner_.owner_before(left.owner_);
  }

 private:
  std::weak_ptr<const void> owner_{};
};

struct MaterializePreparedNullspaceBasisKernel {
  Array4 values;
  ConstArray4 mask;
  ConstArray4 coverage;
  int component;
  bool masked;
  bool covered;
  POPS_HD void operator()(int i, int j) const {
    const Real basis = masked ? mask(i, j, 0) : Real(1);
    values(i, j, component) = basis * (covered ? coverage(i, j, 0) : Real(1));
  }
};

template <class Distribution>
void preflight_prepared_nullspace_provider(const MultiFab& layout, bool singular,
                                           const FieldNullspacePlan& plan, int first_level,
                                           const Distribution& distribution,
                                           const ExecutionLane& lane) {
  FieldNullspacePreflightPayload payload;
  long metadata_failure_local = 0;
  try {
    payload.append_scalar(static_cast<std::uint8_t>(singular));
    payload.append_scalar(first_level);
    payload.append_plan(plan, distribution);
    validate_field_nullspace_plan_locally(payload, plan);
    if (singular) {
      const bool level_resolved = first_level >= 0;
      payload.require(level_resolved && !plan.bases.empty() &&
                      plan.gauges.size() == plan.bases.size());
      payload.append_layout(&layout, distribution);
    } else {
      payload.require(plan.bases.empty() && plan.gauges.empty());
    }
  } catch (...) {
    metadata_failure_local = 1;
  }
  if (all_reduce_max(metadata_failure_local, lane) != 0)
    throw std::runtime_error(
        "prepared nullspace metadata construction failed on at least one communicator rank");
  finish_field_nullspace_preflight(payload, FieldNullspaceCollectiveBoundary::Preparation, lane);

  // The common preflight above must establish one uniform singular/nonsingular branch before any
  // branch-specific layout or exact-value collective. Otherwise one rank could enter a nullspace
  // validation reduction while another rank returns through the nonsingular path.
  if (singular) {
    std::vector<char, comm_allocator<char>> validation;
    long validation_allocation_failure_local = 0;
    try {
      validation.assign(distribution.validation_scratch_byte_count(), char{0});
    } catch (...) {
      validation_allocation_failure_local = 1;
    }
    if (all_reduce_max(validation_allocation_failure_local, lane) != 0)
      throw std::runtime_error(
          "prepared nullspace validation scratch allocation failed on at least one "
          "communicator rank");

    distribution.require_collective_layout(layout, "prepared nullspace solved vector", lane);
    distribution.require_exact_values(layout, validation, "prepared nullspace solved vector", lane);
    for (const FieldNullspaceBasis& basis : plan.bases) {
      for (const auto& mask : basis.masks) {
        if (!mask)
          continue;
        distribution.require_collective_layout(*mask, "prepared nullspace basis mask", lane);
        distribution.require_exact_values(*mask, validation, "prepared nullspace basis mask", lane);
      }
      for (const auto& coverage : basis.coverage) {
        if (!coverage)
          continue;
        distribution.require_collective_layout(*coverage, "prepared nullspace coverage mask", lane);
        distribution.require_exact_values(*coverage, validation, "prepared nullspace coverage mask",
                                          lane);
      }
    }
  }
}
}  // namespace detail

/// Trusted host-side matrix-free callback. The callback must overwrite every valid output cell and
/// owns any typed halo/boundary fill required before it reads input ghosts. Device work remains
/// inside the Kokkos-backed kernels called by the function; no callback is copied or constructed
/// inside an iteration. The trusted bridge catches a returned exception and publishes deterministic
/// NaNs instead of adding an MPI control reduction to every apply. A callback that enters MPI must
/// nevertheless complete the same collective trace on every rank; no wrapper can repair a trace
/// abandoned from inside the callback.
using ApplyFn = std::function<void(MultiFab& out, const MultiFab& in)>;
using PreparedResourceFn = std::function<void()>;
using PreparedAllocationCountFn = std::function<std::size_t()>;

/// Concurrency guaranteed by an affine-operator provider across independently prepared sessions.
/// Independent providers own all mutable state per session. Exclusive providers remain fully
/// prepared and allocation-stable but borrow an external mutable context that permits only one
/// active solve invocation per PreparedAffineLinearProblem.
enum class PreparedOperatorConcurrency : std::uint8_t { Independent = 0, Exclusive = 1 };

/// Allocation-free outcome of one hot operator/preconditioner application. A failed status is
/// sticky in its materialized session and workspace until the next reserved solve starts. Callbacks
/// still execute after a local failure so every rank preserves the provider's MPI trace; PoPS
/// re-publishes NaN after each downstream callback so finite output cannot erase the failure.
enum class PreparedApplyStatus : std::uint8_t { Success = 0, Failure = 1 };

inline constexpr bool prepared_apply_succeeded(PreparedApplyStatus status) noexcept {
  return status == PreparedApplyStatus::Success;
}

inline constexpr bool prepared_operator_concurrency_is_valid(
    PreparedOperatorConcurrency value) noexcept {
  return value == PreparedOperatorConcurrency::Independent ||
         value == PreparedOperatorConcurrency::Exclusive;
}

/// One workspace-private affine-operator execution session. Stateful matrix-free operators own
/// their mutable scratch and communication caches inside this object; no session is reused by a
/// second KrylovWorkspace.
template <class Session>
concept PreparedAffineOperatorSessionSource =
    std::movable<std::remove_cvref_t<Session>> &&
    requires(std::remove_cvref_t<Session>& session, MultiFab& out, const MultiFab& in) {
      { session.prepare() } -> std::same_as<void>;
      { session.apply(out, in) } noexcept -> std::same_as<PreparedApplyStatus>;
      { session.allocation_count() } -> std::convertible_to<std::size_t>;
    };

class PreparedAffineOperatorSession {
 public:
  PreparedAffineOperatorSession() = default;

  template <PreparedAffineOperatorSessionSource Session>
  explicit PreparedAffineOperatorSession(Session session)
      : implementation_(std::make_unique<Model<std::remove_cvref_t<Session>>>(std::move(session))) {
  }

  PreparedAffineOperatorSession(PreparedAffineOperatorSession&&) noexcept = default;
  PreparedAffineOperatorSession& operator=(PreparedAffineOperatorSession&&) noexcept = default;
  PreparedAffineOperatorSession(const PreparedAffineOperatorSession&) = delete;
  PreparedAffineOperatorSession& operator=(const PreparedAffineOperatorSession&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(implementation_);
  }
  void prepare() {
    require_initialized_();
    implementation_->prepare();
  }
  [[nodiscard]] PreparedApplyStatus apply(MultiFab& out, const MultiFab& in) const noexcept {
    if (!implementation_) {
      apply_status_ = PreparedApplyStatus::Failure;
      publish_failure_(out);
      return apply_status_;
    }
    const PreparedApplyStatus current = implementation_->apply(out, in);
    if (!prepared_apply_succeeded(current))
      apply_status_ = PreparedApplyStatus::Failure;
    if (!prepared_apply_succeeded(apply_status_))
      publish_failure_(out);
    return apply_status_;
  }
  [[nodiscard]] std::size_t allocation_count() const {
    require_initialized_();
    return implementation_->allocation_count();
  }
  void reset_apply_status() noexcept { apply_status_ = PreparedApplyStatus::Success; }
  [[nodiscard]] PreparedApplyStatus apply_status() const noexcept { return apply_status_; }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual void prepare() = 0;
    [[nodiscard]] virtual PreparedApplyStatus apply(MultiFab&, const MultiFab&) const noexcept = 0;
    [[nodiscard]] virtual std::size_t allocation_count() const = 0;
  };

  template <PreparedAffineOperatorSessionSource Session>
  class Model final : public Concept {
   public:
    explicit Model(Session session) : session_(std::move(session)) {}
    void prepare() override { session_.prepare(); }
    [[nodiscard]] PreparedApplyStatus apply(MultiFab& out,
                                            const MultiFab& in) const noexcept override {
      return session_.apply(out, in);
    }
    [[nodiscard]] std::size_t allocation_count() const override {
      return static_cast<std::size_t>(session_.allocation_count());
    }

   private:
    mutable Session session_;
  };

  void require_initialized_() const {
    if (!implementation_)
      throw std::logic_error("prepared affine operator session is not initialized");
  }

  static void publish_failure_(MultiFab& out) noexcept {
    // Continuing with stale finite output is forbidden. If the backend itself cannot write the
    // deterministic poison, noexcept terminates rather than silently accepting wrong science.
    out.set_val(std::numeric_limits<Real>::quiet_NaN());
  }

  std::unique_ptr<Concept> implementation_;
  mutable PreparedApplyStatus apply_status_ = PreparedApplyStatus::Success;
};

struct PreparedAffineOperatorSessionCallbacks {
  PreparedResourceFn prepare{};
  ApplyFn apply{};
  PreparedAllocationCountFn allocation_count{};
};

using PreparedAffineOperatorSessionFactory =
    std::function<PreparedAffineOperatorSessionCallbacks(const ExecutionLane&)>;

template <class Source>
concept PreparedAffineOperatorSource =
    std::copy_constructible<std::remove_cvref_t<Source>> &&
    requires(const std::remove_cvref_t<Source>& source, ExactContractBuilder& contract,
             const ExecutionLane& lane) {
      {
        std::remove_cvref_t<Source>::provider_identity()
      } noexcept -> std::same_as<PreparedProviderIdentity>;
      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
      requires PreparedAffineOperatorSessionSource<decltype(source.make_session(lane))>;
    };

/// Authenticated factory for fresh affine-operator sessions. The provider is immutable and may be
/// shared. An Independent provider owns all mutable apply state in each returned session; an
/// Exclusive trusted extension may instead borrow one external mutable execution context and is
/// protected by the prepared problem's invocation reservation.
class PreparedAffineOperatorProvider {
 public:
  PreparedAffineOperatorProvider() = default;
  PreparedAffineOperatorProvider(const PreparedAffineOperatorProvider&) = default;
  PreparedAffineOperatorProvider(PreparedAffineOperatorProvider&&) noexcept = default;
  PreparedAffineOperatorProvider& operator=(const PreparedAffineOperatorProvider&) = default;
  PreparedAffineOperatorProvider& operator=(PreparedAffineOperatorProvider&&) noexcept = default;

  template <class Source>
    requires(!std::same_as<std::remove_cvref_t<Source>, PreparedAffineOperatorProvider> &&
             PreparedAffineOperatorSource<Source>)
  explicit PreparedAffineOperatorProvider(Source source)
      : implementation_(std::make_shared<Model<std::remove_cvref_t<Source>>>(std::move(source))) {}

  /// Explicit native/plugin trust boundary. Each callback pair must be freshly materialized. The
  /// concurrency argument states whether its mutable execution state is session-private or borrowed
  /// from one exclusive external context.
  [[nodiscard]] static PreparedAffineOperatorProvider trusted_extension(
      PreparedProviderIdentity identity, std::string exact_parameters,
      PreparedAffineOperatorSessionFactory make_session,
      PreparedOperatorConcurrency concurrency = PreparedOperatorConcurrency::Independent) {
    PreparedAffineOperatorProvider provider;
    provider.implementation_ = std::make_shared<TrustedModel>(identity, std::move(exact_parameters),
                                                              std::move(make_session), concurrency);
    return provider;
  }

  /// Convenience for a callback whose implementation and every captured object are explicitly
  /// guaranteed reentrant by its author. The persistent-field witness is mandatory even when it
  /// returns zero; the core never infers "stateless" from a callback shape. Stateful callbacks that
  /// need fresh per-workspace captures must use trusted_extension instead.
  [[nodiscard]] static PreparedAffineOperatorProvider trusted_reentrant(
      ApplyFn apply, PreparedAllocationCountFn allocation_count) {
    if (!apply)
      throw std::invalid_argument("reentrant affine operator requires an apply callback");
    if (!allocation_count)
      throw std::invalid_argument(
          "reentrant affine operator requires an exact persistent-field count");
    return trusted_extension(
        {"pops.affine-operator.reentrant-callback", 1}, {},
        [apply = std::move(apply),
         allocation_count = std::move(allocation_count)](const ExecutionLane&) {
          return PreparedAffineOperatorSessionCallbacks{{}, apply, allocation_count};
        });
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(implementation_);
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return implementation_ ? implementation_->collective_contract() : std::string_view{};
  }
  [[nodiscard]] PreparedOperatorConcurrency concurrency() const noexcept {
    return implementation_ ? implementation_->concurrency()
                           : PreparedOperatorConcurrency::Exclusive;
  }
  [[nodiscard]] PreparedAffineOperatorSession make_session(const ExecutionLane& lane) const {
    if (!implementation_)
      throw std::logic_error("cannot create a session from an empty affine operator provider");
    return implementation_->make_session(lane);
  }

 private:
  friend struct detail::PreparedProblemAccess;

  [[nodiscard]] detail::PreparedProviderSourceIdentity source_identity_() const noexcept {
    return detail::PreparedProviderSourceIdentity(std::shared_ptr<const void>(implementation_));
  }

  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
    [[nodiscard]] virtual PreparedOperatorConcurrency concurrency() const noexcept = 0;
    [[nodiscard]] virtual PreparedAffineOperatorSession make_session(
        const ExecutionLane&) const = 0;
  };

  static std::string make_contract_(PreparedProviderIdentity identity,
                                    std::string_view exact_parameters,
                                    PreparedOperatorConcurrency concurrency) {
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument("prepared affine operator source requires an exact identity");
    if (!prepared_operator_concurrency_is_valid(concurrency))
      throw std::invalid_argument("prepared affine operator has an invalid concurrency contract");
    ExactContractBuilder contract;
    contract.text("pops.prepared-affine-operator-source")
        .scalar(std::uint32_t{2})
        .text(identity.name)
        .scalar(identity.version)
        .scalar(concurrency)
        .bytes(exact_parameters);
    return std::move(contract).release();
  }

  template <PreparedAffineOperatorSource Source>
  class Model final : public Concept {
   public:
    explicit Model(Source source) : source_(std::move(source)) {
      ExactContractBuilder parameters;
      source_.serialize_exact_parameters(parameters);
      collective_contract_ = make_contract_(Source::provider_identity(), parameters.view(),
                                            PreparedOperatorConcurrency::Independent);
    }
    std::string_view collective_contract() const noexcept override { return collective_contract_; }
    PreparedOperatorConcurrency concurrency() const noexcept override {
      return PreparedOperatorConcurrency::Independent;
    }
    PreparedAffineOperatorSession make_session(const ExecutionLane& lane) const override {
      return PreparedAffineOperatorSession(source_.make_session(lane));
    }

   private:
    Source source_;
    std::string collective_contract_;
  };

  class TrustedModel final : public Concept {
   public:
    TrustedModel(PreparedProviderIdentity identity, std::string exact_parameters,
                 PreparedAffineOperatorSessionFactory make_session,
                 PreparedOperatorConcurrency concurrency)
        : make_session_(std::move(make_session)),
          collective_contract_(make_contract_(identity, exact_parameters, concurrency)),
          concurrency_(concurrency) {
      if (!make_session_)
        throw std::invalid_argument(
            "prepared affine operator extension requires a session factory");
    }
    std::string_view collective_contract() const noexcept override { return collective_contract_; }
    PreparedOperatorConcurrency concurrency() const noexcept override { return concurrency_; }
    PreparedAffineOperatorSession make_session(const ExecutionLane& lane) const override {
      PreparedAffineOperatorSessionCallbacks callbacks = make_session_(lane);
      if (!callbacks.apply)
        throw std::logic_error(
            "prepared affine operator session factory returned no apply callback");
      if (!callbacks.allocation_count)
        throw std::logic_error(
            "prepared affine operator session factory did not report persistent storage");
      struct TrustedSession {
        PreparedResourceFn prepare_callback;
        ApplyFn apply_callback;
        PreparedAllocationCountFn allocation_count_callback;
        void prepare() {
          if (prepare_callback)
            prepare_callback();
        }
        [[nodiscard]] PreparedApplyStatus apply(MultiFab& out, const MultiFab& in) noexcept {
          try {
            apply_callback(out, in);
            return PreparedApplyStatus::Success;
          } catch (...) {
            out.set_val(std::numeric_limits<Real>::quiet_NaN());
            return PreparedApplyStatus::Failure;
          }
        }
        [[nodiscard]] std::size_t allocation_count() const { return allocation_count_callback(); }
      };
      return PreparedAffineOperatorSession(TrustedSession{std::move(callbacks.prepare),
                                                          std::move(callbacks.apply),
                                                          std::move(callbacks.allocation_count)});
    }

   private:
    PreparedAffineOperatorSessionFactory make_session_;
    std::string collective_contract_;
    PreparedOperatorConcurrency concurrency_ = PreparedOperatorConcurrency::Independent;
  };

  std::shared_ptr<const Concept> implementation_;
};

/// One workspace-private preconditioner execution session. A session may own mutable caches, but it
/// is never shared between Krylov workspaces. Preparation happens at bind time and every apply after
/// that boundary reuses the same storage.
template <class Session>
concept PreparedLinearPreconditionerSessionSource =
    std::movable<std::remove_cvref_t<Session>> &&
    requires(std::remove_cvref_t<Session>& session, MultiFab& out, const MultiFab& in) {
      { session.prepare() } -> std::same_as<void>;
      { session.apply(out, in) } noexcept -> std::same_as<PreparedApplyStatus>;
      { session.allocation_count() } -> std::convertible_to<std::size_t>;
    };

class PreparedLinearPreconditionerSession {
 public:
  PreparedLinearPreconditionerSession() = default;

  template <PreparedLinearPreconditionerSessionSource Session>
  explicit PreparedLinearPreconditionerSession(Session session)
      : implementation_(std::make_unique<Model<std::remove_cvref_t<Session>>>(std::move(session))) {
  }

  PreparedLinearPreconditionerSession(PreparedLinearPreconditionerSession&&) noexcept = default;
  PreparedLinearPreconditionerSession& operator=(PreparedLinearPreconditionerSession&&) noexcept =
      default;
  PreparedLinearPreconditionerSession(const PreparedLinearPreconditionerSession&) = delete;
  PreparedLinearPreconditionerSession& operator=(const PreparedLinearPreconditionerSession&) =
      delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(implementation_);
  }
  void prepare() {
    require_initialized_();
    implementation_->prepare();
  }
  [[nodiscard]] PreparedApplyStatus apply(MultiFab& out, const MultiFab& in) const noexcept {
    if (!implementation_) {
      apply_status_ = PreparedApplyStatus::Failure;
      publish_failure_(out);
      return apply_status_;
    }
    const PreparedApplyStatus current = implementation_->apply(out, in);
    if (!prepared_apply_succeeded(current))
      apply_status_ = PreparedApplyStatus::Failure;
    if (!prepared_apply_succeeded(apply_status_))
      publish_failure_(out);
    return apply_status_;
  }
  [[nodiscard]] std::size_t allocation_count() const {
    require_initialized_();
    return implementation_->allocation_count();
  }
  void reset_apply_status() noexcept { apply_status_ = PreparedApplyStatus::Success; }
  [[nodiscard]] PreparedApplyStatus apply_status() const noexcept { return apply_status_; }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual void prepare() = 0;
    [[nodiscard]] virtual PreparedApplyStatus apply(MultiFab&, const MultiFab&) const noexcept = 0;
    [[nodiscard]] virtual std::size_t allocation_count() const = 0;
  };

  template <PreparedLinearPreconditionerSessionSource Session>
  class Model final : public Concept {
   public:
    explicit Model(Session session) : session_(std::move(session)) {}
    void prepare() override { session_.prepare(); }
    [[nodiscard]] PreparedApplyStatus apply(MultiFab& out,
                                            const MultiFab& in) const noexcept override {
      return session_.apply(out, in);
    }
    [[nodiscard]] std::size_t allocation_count() const override {
      return static_cast<std::size_t>(session_.allocation_count());
    }

   private:
    mutable Session session_;
  };

  void require_initialized_() const {
    if (!implementation_)
      throw std::logic_error("prepared linear preconditioner session is not initialized");
  }

  static void publish_failure_(MultiFab& out) noexcept {
    out.set_val(std::numeric_limits<Real>::quiet_NaN());
  }

  std::unique_ptr<Concept> implementation_;
  mutable PreparedApplyStatus apply_status_ = PreparedApplyStatus::Success;
};

struct PreparedLinearPreconditionerSessionCallbacks {
  PreparedResourceFn prepare{};
  ApplyFn apply{};
  PreparedAllocationCountFn allocation_count{};
};

using PreparedLinearPreconditionerSessionFactory =
    std::function<PreparedLinearPreconditionerSessionCallbacks(const ExecutionLane&)>;

/// One exact provider owns a factory for fresh execution sessions. Requiring an explicit factory,
/// rather than copying an apply callback, makes workspace isolation a real protocol guarantee: a
/// stateful implementation must decide how each session receives independent mutable state.
template <class Source>
concept PreparedLinearPreconditionerSource =
    std::copy_constructible<std::remove_cvref_t<Source>> &&
    requires(const std::remove_cvref_t<Source>& source, ExactContractBuilder& contract,
             const ExecutionLane& lane) {
      {
        std::remove_cvref_t<Source>::provider_identity()
      } noexcept -> std::same_as<PreparedProviderIdentity>;
      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
      requires PreparedLinearPreconditionerSessionSource<decltype(source.make_session(lane))>;
    };

class PreparedLinearPreconditionerProvider {
 public:
  PreparedLinearPreconditionerProvider() = default;
  PreparedLinearPreconditionerProvider(const PreparedLinearPreconditionerProvider&) = default;
  PreparedLinearPreconditionerProvider(PreparedLinearPreconditionerProvider&&) noexcept = default;
  PreparedLinearPreconditionerProvider& operator=(const PreparedLinearPreconditionerProvider&) =
      default;
  PreparedLinearPreconditionerProvider& operator=(PreparedLinearPreconditionerProvider&&) noexcept =
      default;

  template <class Source>
    requires(!std::same_as<std::remove_cvref_t<Source>, PreparedLinearPreconditionerProvider> &&
             PreparedLinearPreconditionerSource<Source>)
  explicit PreparedLinearPreconditionerProvider(Source source)
      : implementation_(std::make_shared<Model<std::remove_cvref_t<Source>>>(std::move(source))) {}

  /// Explicit plugin/ABI trust boundary when a concrete source type cannot cross the boundary.
  /// The single exact-parameter contract covers the session factory. Every factory invocation must
  /// return a fresh prepare/apply state pair; sharing mutable state across returned sessions violates
  /// this explicit trust boundary.
  [[nodiscard]] static PreparedLinearPreconditionerProvider trusted_extension(
      PreparedProviderIdentity identity, std::string exact_parameters,
      PreparedLinearPreconditionerSessionFactory make_session) {
    PreparedLinearPreconditionerProvider provider;
    provider.implementation_ = std::make_shared<TrustedModel>(identity, std::move(exact_parameters),
                                                              std::move(make_session));
    return provider;
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(implementation_);
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return implementation_ ? implementation_->collective_contract() : std::string_view{};
  }
  [[nodiscard]] PreparedLinearPreconditionerSession make_session(const ExecutionLane& lane) const {
    if (!implementation_)
      throw std::logic_error(
          "cannot create a session from an empty linear preconditioner provider");
    return implementation_->make_session(lane);
  }

 private:
  friend struct detail::PreparedProblemAccess;

  [[nodiscard]] detail::PreparedProviderSourceIdentity source_identity_() const noexcept {
    return detail::PreparedProviderSourceIdentity(std::shared_ptr<const void>(implementation_));
  }

  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
    [[nodiscard]] virtual PreparedLinearPreconditionerSession make_session(
        const ExecutionLane&) const = 0;
  };

  static std::string make_contract_(PreparedProviderIdentity identity,
                                    std::string_view exact_parameters) {
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument(
          "prepared linear preconditioner source requires an exact identity");
    ExactContractBuilder contract;
    contract.text("pops.prepared-linear-preconditioner-source")
        .scalar(std::uint32_t{1})
        .text(identity.name)
        .scalar(identity.version)
        .bytes(exact_parameters);
    return std::move(contract).release();
  }

  template <PreparedLinearPreconditionerSource Source>
  class Model final : public Concept {
   public:
    explicit Model(Source source) : source_(std::move(source)) {
      ExactContractBuilder parameters;
      source_.serialize_exact_parameters(parameters);
      collective_contract_ = make_contract_(Source::provider_identity(), parameters.view());
    }
    std::string_view collective_contract() const noexcept override { return collective_contract_; }
    PreparedLinearPreconditionerSession make_session(const ExecutionLane& lane) const override {
      return PreparedLinearPreconditionerSession(source_.make_session(lane));
    }

   private:
    Source source_;
    std::string collective_contract_;
  };

  class TrustedModel final : public Concept {
   public:
    TrustedModel(PreparedProviderIdentity identity, std::string exact_parameters,
                 PreparedLinearPreconditionerSessionFactory make_session)
        : make_session_(std::move(make_session)),
          collective_contract_(make_contract_(identity, exact_parameters)) {
      if (!make_session_)
        throw std::invalid_argument(
            "prepared linear preconditioner extension requires a session factory");
    }
    std::string_view collective_contract() const noexcept override { return collective_contract_; }
    PreparedLinearPreconditionerSession make_session(const ExecutionLane& lane) const override {
      PreparedLinearPreconditionerSessionCallbacks callbacks = make_session_(lane);
      if (!callbacks.apply)
        throw std::logic_error(
            "prepared linear preconditioner session factory returned no apply callback");
      if (!callbacks.allocation_count)
        throw std::logic_error(
            "prepared linear preconditioner session factory did not report persistent storage");
      struct TrustedSession {
        PreparedResourceFn prepare_callback;
        ApplyFn apply_callback;
        PreparedAllocationCountFn allocation_count_callback;
        void prepare() {
          if (prepare_callback)
            prepare_callback();
        }
        [[nodiscard]] PreparedApplyStatus apply(MultiFab& out, const MultiFab& in) noexcept {
          try {
            apply_callback(out, in);
            return PreparedApplyStatus::Success;
          } catch (...) {
            out.set_val(std::numeric_limits<Real>::quiet_NaN());
            return PreparedApplyStatus::Failure;
          }
        }
        [[nodiscard]] std::size_t allocation_count() const { return allocation_count_callback(); }
      };
      return PreparedLinearPreconditionerSession(
          TrustedSession{std::move(callbacks.prepare), std::move(callbacks.apply),
                         std::move(callbacks.allocation_count)});
    }

   private:
    PreparedLinearPreconditionerSessionFactory make_session_;
    std::string collective_contract_;
  };

  std::shared_ptr<const Concept> implementation_;
};

enum class LinearOperatorProperty : std::uint32_t {
  kNone = 0,
  kSymmetric = 1u << 0u,
  kPositiveDefinite = 1u << 1u,
  kPositiveDefiniteOnNullspaceComplement = 1u << 2u,
};

constexpr std::uint32_t operator_property_bit(LinearOperatorProperty property) {
  return static_cast<std::uint32_t>(property);
}

/// Authenticated mathematical properties. Global positive definiteness and positive definiteness
/// on a declared nullspace complement are mutually exclusive, both require symmetry, and CG
/// requires the certificate coherent with the problem's explicit nullspace policy.
struct LinearOperatorProperties {
  std::uint32_t bits = 0;

  static constexpr LinearOperatorProperties general() { return {}; }
  static constexpr LinearOperatorProperties symmetric() {
    return {operator_property_bit(LinearOperatorProperty::kSymmetric)};
  }
  static constexpr LinearOperatorProperties symmetric_positive_definite() {
    return {operator_property_bit(LinearOperatorProperty::kSymmetric) |
            operator_property_bit(LinearOperatorProperty::kPositiveDefinite)};
  }
  static constexpr LinearOperatorProperties symmetric_positive_definite_on_nullspace_complement() {
    return {operator_property_bit(LinearOperatorProperty::kSymmetric) |
            operator_property_bit(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement)};
  }
  constexpr bool has(LinearOperatorProperty property) const {
    return (bits & operator_property_bit(property)) != 0;
  }
  constexpr bool certifies_spd() const {
    return has(LinearOperatorProperty::kSymmetric) &&
           has(LinearOperatorProperty::kPositiveDefinite);
  }
  constexpr bool certifies_cg(bool declared_nullspace) const {
    return declared_nullspace
               ? has(LinearOperatorProperty::kSymmetric) &&
                     has(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement)
               : certifies_spd();
  }
  constexpr bool valid() const {
    constexpr std::uint32_t known =
        operator_property_bit(LinearOperatorProperty::kSymmetric) |
        operator_property_bit(LinearOperatorProperty::kPositiveDefinite) |
        operator_property_bit(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement);
    const bool positive_definite = has(LinearOperatorProperty::kPositiveDefinite);
    const bool complement_positive =
        has(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement);
    return (bits & ~known) == 0 && !(positive_definite && complement_positive) &&
           (!(positive_definite || complement_positive) || has(LinearOperatorProperty::kSymmetric));
  }
};

/// Required nullspace contract of a prepared problem. `preserving(plan)` is an authenticated
/// install-time certificate that both A_lin and the installed preconditioner map the declared
/// gauge-fixed subspace into itself. The generic algorithms therefore need no method-specific
/// projection branch or collective in their hot loops. Preparation still validates the complete
/// FieldNullspacePlan against the concrete layout; a missing/incomplete gauge fails before solve.
class PreparedNullspacePolicy {
 public:
  static PreparedNullspacePolicy nonsingular() { return PreparedNullspacePolicy(); }

  static PreparedNullspacePolicy preserving(FieldNullspacePlan plan, int first_level = 0) {
    if (plan.empty())
      throw std::invalid_argument(
          "a preserving prepared nullspace policy requires a non-empty FieldNullspacePlan");
    if (first_level < 0)
      throw std::invalid_argument("prepared nullspace first level must be non-negative");
    PreparedNullspacePolicy result;
    result.plan_ = std::move(plan);
    result.first_level_ = first_level;
    result.singular_ = true;
    return result;
  }

  bool singular() const { return singular_; }

  /// Public/direct preparation deliberately always revalidates. It is not a collective cache
  /// boundary: callers that invoke this policy themselves may not have completed the fixed
  /// PreparedAffineLinearProblem consensus preflight, so a rank-local early return could split
  /// the Gram reduction below.
  void prepare(const MultiFab& layout, const PreparedVectorDistribution& distribution) {
    const ExecutionLane lane = ExecutionLane::world();
    prepare(layout, PreparedVectorMetric::euclidean(layout, distribution), lane);
  }

  void prepare(const MultiFab& layout, const PreparedVectorMetric& metric) {
    const ExecutionLane lane = ExecutionLane::world();
    prepare(layout, metric, lane);
  }

  void prepare(const MultiFab& layout, const PreparedVectorMetric& metric,
               const ExecutionLane& lane) {
    prepared_ = false;
    detail::preflight_prepared_nullspace_provider(layout, singular_, plan_, first_level_,
                                                  metric.distribution(), lane);
    if (!singular_) {
      basis_vectors_.clear();
      basis_metric_gram_factor_.clear();
      prepared_ = true;
      return;
    }
    // Complete every rank-local operation that may throw or allocate before the first metric
    // callback. A rank must never leave this preparation branch while peers have already entered
    // a provider-owned collective. Publish the new certificate only after all collective work and
    // factorization have succeeded everywhere.
    std::vector<MultiFab> candidate_basis_vectors;
    std::vector<double> candidate_gram_factor;
    std::vector<double> metric_scratch;
    long local_staging_failure = 0;
    try {
      if (plan_.identity.empty() || plan_.layout_identity.empty())
        throw std::invalid_argument(
            "prepared nullspace policy requires exact plan and layout identities");
      if (plan_.gauges.size() != plan_.bases.size())
        throw std::invalid_argument(
            "prepared nullspace policy must gauge every declared basis exactly once");
      for (const FieldGaugeConstraint& gauge : plan_.gauges)
        if (!std::isfinite(static_cast<double>(gauge.value)))
          throw std::invalid_argument("prepared nullspace gauge values must be finite");
      if (!metric.compatible_with(layout, metric.distribution()))
        throw std::invalid_argument(
            "prepared nullspace metric disagrees with the single-field vector space");

      candidate_basis_vectors.reserve(plan_.bases.size());
      for (const FieldNullspaceBasis& basis : plan_.bases) {
        detail::validate_basis_layout(layout, basis.mask(first_level_), basis);
        candidate_basis_vectors.emplace_back(layout.box_array(), layout.dmap(), layout.ncomp(),
                                             layout.n_grow());
        MultiFab& materialized = candidate_basis_vectors.back();
        materialized.share_halo_cache_from(layout);
        detail::PreparedFieldAlgebra::zero(materialized);
        const MultiFab* mask = basis.mask(first_level_);
        const MultiFab* coverage = basis.coverage_mask(first_level_);
        detail::validate_mask_layout(layout, coverage, "coverage");
        for (int local = 0; local < materialized.local_size(); ++local) {
          const ConstArray4 mask_values =
              mask == nullptr ? ConstArray4{} : mask->fab(local).const_array();
          const ConstArray4 coverage_values =
              coverage == nullptr ? ConstArray4{} : coverage->fab(local).const_array();
          for_each_cell(materialized.box(local),
                        detail::MaterializePreparedNullspaceBasisKernel{
                            materialized.fab(local).array(), mask_values, coverage_values,
                            basis.field_component, mask != nullptr, coverage != nullptr});
        }
      }

      const std::size_t basis_count = candidate_basis_vectors.size();
      candidate_gram_factor.assign(detail::checked_field_nullspace_collective_product(
                                       basis_count, basis_count, "prepared nullspace Gram matrix"),
                                   0.0);
      metric_scratch.assign(metric.reduction_scratch_value_count(), 0.0);
      for (std::size_t left = 0; left < plan_.bases.size(); ++left) {
        for (std::size_t right = left; right < plan_.bases.size(); ++right) {
          if (plan_.bases[left].field_component == plan_.bases[right].field_component &&
              plan_.bases[left].measure(first_level_) != plan_.bases[right].measure(first_level_))
            throw std::invalid_argument(
                "prepared nullspace bases disagree on the single-field cell measure");
        }
      }
    } catch (...) {
      local_staging_failure = 1;
    }
    if (all_reduce_max(local_staging_failure, lane) != 0)
      throw std::runtime_error(
          "prepared nullspace local certificate staging failed on at least one communicator rank");

    const std::size_t basis_count = candidate_basis_vectors.size();
    for (std::size_t left = 0; left < basis_count; ++left) {
      for (std::size_t right = left; right < basis_count; ++right) {
        double overlap = std::numeric_limits<double>::quiet_NaN();
        long metric_callback_failure_local = 0;
        try {
          overlap = static_cast<double>(metric.nullspace_inner_product(
              candidate_basis_vectors[left], candidate_basis_vectors[right],
              plan_.bases[left].measure(first_level_), metric_scratch, lane));
        } catch (...) {
          metric_callback_failure_local = 1;
        }
        if (all_reduce_max(metric_callback_failure_local, lane) != 0)
          throw std::runtime_error(
              "prepared nullspace metric callback failed on at least one communicator rank");
        candidate_gram_factor[left * basis_count + right] = overlap;
        candidate_gram_factor[right * basis_count + left] = overlap;
      }
    }

    long factorization_failure_local = 0;
    try {
      detail::factor_field_nullspace_gram(candidate_gram_factor, basis_count,
                                          "prepared nullspace Gram matrix");
    } catch (...) {
      factorization_failure_local = 1;
    }
    if (all_reduce_max(factorization_failure_local, lane) != 0)
      throw std::runtime_error(
          "prepared nullspace Gram factorization failed on at least one communicator rank");

    basis_vectors_ = std::move(candidate_basis_vectors);
    basis_metric_gram_factor_ = std::move(candidate_gram_factor);
    prepared_ = true;
  }

  void require_compatible(const MultiFab& normalized_rhs, const PreparedVectorMetric& metric,
                          std::span<double> metric_scratch, const ExecutionLane& lane) const {
    require_prepared();
    if (!singular_)
      return;
    for (std::size_t index = 0; index < basis_vectors_.size(); ++index) {
      const Real measure = plan_.bases[index].measure(first_level_);
      const Real moment = metric.nullspace_inner_product(normalized_rhs, basis_vectors_[index],
                                                         measure, metric_scratch, lane);
      const Real absolute = metric.nullspace_absolute_inner_product(
          normalized_rhs, basis_vectors_[index], measure, metric_scratch, lane);
      if (!std::isfinite(static_cast<double>(moment)) ||
          !std::isfinite(static_cast<double>(absolute)))
        throw FieldNullspaceInvalidEvaluation(
            "field RHS has a non-finite compatibility moment for nullspace basis '" +
            plan_.bases[index].identity + "'; silent projection is forbidden");
      const Real scale = absolute > Real(1) ? absolute : Real(1);
      const Real tolerance = Real(128) * std::numeric_limits<Real>::epsilon() * scale;
      if (std::abs(moment) > tolerance)
        throw FieldNullspaceIncompatibleRhs("field RHS is incompatible with nullspace basis '" +
                                            plan_.bases[index].identity +
                                            "'; silent projection is forbidden");
    }
  }

  void apply_gauge(MultiFab& iterate, const PreparedVectorMetric& metric,
                   std::span<double> gauge_coefficients, std::span<double> metric_scratch,
                   const ExecutionLane& lane) const {
    require_prepared();
    if (!singular_)
      return;
    if (gauge_coefficients.size() < basis_vectors_.size())
      throw std::invalid_argument("prepared nullspace gauge scratch is too small");
    for (std::size_t index = 0; index < basis_vectors_.size(); ++index) {
      const std::size_t gauge = detail::gauge_index(plan_, plan_.bases[index].identity);
      if (gauge == plan_.gauges.size())
        throw std::logic_error("prepared nullspace gauge does not cover every basis");
      gauge_coefficients[index] = static_cast<double>(metric.nullspace_inner_product(
          iterate, basis_vectors_[index], plan_.bases[index].measure(first_level_), metric_scratch,
          lane));
    }
    detail::solve_field_nullspace_gram(basis_metric_gram_factor_, basis_vectors_.size(),
                                       gauge_coefficients);
    for (std::size_t index = 0; index < basis_vectors_.size(); ++index) {
      const std::size_t gauge = detail::gauge_index(plan_, plan_.bases[index].identity);
      const Real coefficient =
          static_cast<Real>(gauge_coefficients[index]) - plan_.gauges[gauge].value;
      if (!std::isfinite(static_cast<double>(coefficient)))
        throw FieldNullspaceInvalidEvaluation(
            "field gauge produced a non-finite metric coefficient");
      detail::PreparedFieldAlgebra::axpy(iterate, -coefficient, basis_vectors_[index]);
    }
  }

 private:
  friend class PreparedAffineLinearProblem;
  friend struct detail::PreparedProblemAccess;

  PreparedNullspacePolicy() = default;

  /// This route is private to PreparedAffineLinearProblem. Its caller has just established the
  /// exact fixed collective contract, including the immutable plan, layout, prepared bit, and
  /// persistent metric-basis capacity on every rank. Only that consensus authorizes reusing the Gram
  /// certificate; public prepare() above remains deliberately uncached.
  void prepare_after_collective_preflight_(const MultiFab& layout,
                                           const PreparedVectorMetric& metric,
                                           const ExecutionLane& lane) {
    if (!prepared_)
      prepare(layout, metric, lane);
  }

  /// A failed collective preparation stage may have completed the local certificate on only a
  /// subset of ranks. The owner calls this on every rank after its failure reduction so the next
  /// attempt starts from one uniform, re-preparable state.
  void invalidate_collective_certificate_() noexcept { prepared_ = false; }

  void require_prepared() const {
    if (!prepared_)
      throw std::logic_error("prepared nullspace policy was used before problem preparation");
  }

  FieldNullspacePlan plan_{};
  std::array<std::uint64_t, 4> plan_fingerprint_{};
  std::vector<MultiFab> basis_vectors_{};
  std::vector<double> basis_metric_gram_factor_{};
  int first_level_ = 0;
  bool singular_ = false;
  bool prepared_ = false;
};

/// Scale-safe physical reference used by the authored stopping criterion.  The field returned with
/// this value is divided by `reference_norm` only for the one nullspace-compatibility check; the
/// iterative equation chooses a separate scale from the actual warm-start residual.
struct PreparedEquationReference {
  Real reference_norm = Real(0);
};

/// Exact memory/layout requirement of one prepared solve route.
struct KrylovFootprint {
  int components = 1;
  int input_ghosts = 0;
  bool preconditioned = false;

  friend bool operator==(const KrylovFootprint&, const KrylovFootprint&) = default;
};

/// Allocation-free identity of one operator evaluation. The 256-bit authority is emitted from the
/// canonical Program/IR identity. The remaining fields authenticate the actual runtime evaluation
/// point and topology. Binary64 values travel as exact bit patterns, never rounded text.
using OperatorFingerprint = std::array<std::uint64_t, 4>;

namespace detail {

/// Capability issued only by a compiled-Program runtime context.  Direct/native extension
/// callbacks cannot disable the verified apply path merely by selecting a public enum value.
class AuthenticatedProgramApplyToken {
 public:
  AuthenticatedProgramApplyToken(const AuthenticatedProgramApplyToken&) = default;
  AuthenticatedProgramApplyToken& operator=(const AuthenticatedProgramApplyToken&) = default;

 private:
  explicit AuthenticatedProgramApplyToken(OperatorFingerprint authority) : authority_(authority) {
    if (std::all_of(authority_.begin(), authority_.end(),
                    [](std::uint64_t word) { return word == 0; }))
      throw std::invalid_argument("authenticated Program operator authority must be non-zero");
  }

  friend class ::pops::runtime::program::ProgramContext;
  friend class ::pops::runtime::program::AmrProgramContext;
  friend class ::pops::PreparedAffineLinearProblem;

  OperatorFingerprint authority_{};
};

}  // namespace detail

struct OperatorEvaluationSnapshot {
  OperatorFingerprint authority{};
  std::uint64_t revision = 0;
  std::int64_t macro_step = 0;
  std::int64_t stage_numerator = 0;
  std::int64_t stage_denominator = 1;
  std::uint64_t dt_bits = 0;
  std::uint64_t physical_time_bits = 0;
  std::uint64_t topology_revision = 0;
  OperatorFingerprint topology{};
  OperatorFingerprint resources{};

  bool valid() const {
    const bool nonzero_authority =
        authority[0] != 0 || authority[1] != 0 || authority[2] != 0 || authority[3] != 0;
    const auto any_nonzero = [](const OperatorFingerprint& fingerprint) {
      return fingerprint[0] != 0 || fingerprint[1] != 0 || fingerprint[2] != 0 ||
             fingerprint[3] != 0;
    };
    const double dt = std::bit_cast<double>(dt_bits);
    const double time = std::bit_cast<double>(physical_time_bits);
    return nonzero_authority && revision != 0 && macro_step >= 0 && stage_numerator >= 0 &&
           stage_denominator > 0 && stage_numerator <= stage_denominator && std::isfinite(dt) &&
           dt >= 0.0 && std::isfinite(time) && topology_revision != 0 && any_nonzero(topology) &&
           any_nonzero(resources);
  }
  friend bool operator==(const OperatorEvaluationSnapshot&,
                         const OperatorEvaluationSnapshot&) = default;
};

/// Rank-local observation of the current evaluation identity. A probe must not issue MPI
/// collectives: PreparedAffineLinearProblem owns the identically ordered consensus operation and
/// converts a probe failure on any one rank into the same exception on every rank.
using OperatorSnapshotProbe = std::function<OperatorEvaluationSnapshot()>;

namespace detail {

struct PreparedProblemAccess;

inline std::uint64_t fingerprint_mix_word(std::uint64_t hash, std::uint64_t value) {
  constexpr std::uint64_t prime = 1099511628211ull;
  for (unsigned byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8u)) & 0xffu;
    hash *= prime;
  }
  return hash;
}

inline OperatorFingerprint fingerprint_seed() {
  return {1469598103934665603ull, 1099511628211ull, 7809847782465536322ull, 9650029242287828579ull};
}

inline void fingerprint_mix(OperatorFingerprint& hash, std::uint64_t value) {
  constexpr std::array<std::uint64_t, 4> domain_separators{
      0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull, 0x94d049bb133111ebull, 0xd6e8feb86659fd93ull};
  for (std::size_t lane = 0; lane < hash.size(); ++lane)
    hash[lane] = fingerprint_mix_word(hash[lane], value ^ domain_separators[lane]);
}

inline void fingerprint_mix(OperatorFingerprint& hash, std::string_view value) {
  fingerprint_mix(hash, static_cast<std::uint64_t>(value.size()));
  for (const unsigned char byte : value)
    fingerprint_mix(hash, static_cast<std::uint64_t>(byte));
}

inline OperatorFingerprint layout_fingerprint(
    const MultiFab& value,
    PreparedVectorDistribution ownership = PreparedVectorDistribution::Distributed) {
  OperatorFingerprint hash = fingerprint_seed();
  fingerprint_mix(hash, ownership.collective_contract());
  fingerprint_mix(hash, ownership.layout_contract(value));
  return hash;
}

inline void fingerprint_geometry(OperatorFingerprint& hash, const Geometry& geometry) {
  fingerprint_mix(hash, "cartesian");
  for (int axis = 0; axis < 2; ++axis) {
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.lo[axis])));
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.hi[axis])));
  }
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.xlo));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.xhi));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.ylo));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.yhi));
}

inline void fingerprint_geometry(OperatorFingerprint& hash, const PolarGeometry& geometry) {
  fingerprint_mix(hash, "polar");
  for (int axis = 0; axis < 2; ++axis) {
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.lo[axis])));
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.hi[axis])));
  }
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.r_min));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.r_max));
}

inline void fingerprint_boundary(OperatorFingerprint& hash, const BCRec& boundary) {
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.xlo));
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.xhi));
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.ylo));
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.yhi));
  const std::array<Real, 14> values{boundary.xlo_val,   boundary.xhi_val,   boundary.ylo_val,
                                    boundary.yhi_val,   boundary.xlo_alpha, boundary.xlo_beta,
                                    boundary.xhi_alpha, boundary.xhi_beta,  boundary.ylo_alpha,
                                    boundary.ylo_beta,  boundary.yhi_alpha, boundary.yhi_beta,
                                    boundary.dx,        boundary.dy};
  for (const Real value : values)
    fingerprint_mix(hash, std::bit_cast<std::uint64_t>(value));
}

inline std::array<char, 19 * sizeof(std::uint64_t)> snapshot_consensus_payload(
    const OperatorEvaluationSnapshot& snapshot) {
  std::array<std::uint64_t, 19> words{};
  std::size_t index = 0;
  const auto append_fingerprint = [&words, &index](const OperatorFingerprint& fingerprint) {
    for (const std::uint64_t word : fingerprint)
      words[index++] = word;
  };
  append_fingerprint(snapshot.authority);
  words[index++] = snapshot.revision;
  words[index++] = std::bit_cast<std::uint64_t>(snapshot.macro_step);
  words[index++] = std::bit_cast<std::uint64_t>(snapshot.stage_numerator);
  words[index++] = std::bit_cast<std::uint64_t>(snapshot.stage_denominator);
  words[index++] = snapshot.dt_bits;
  words[index++] = snapshot.physical_time_bits;
  words[index++] = snapshot.topology_revision;
  append_fingerprint(snapshot.topology);
  append_fingerprint(snapshot.resources);

  std::array<char, sizeof(words)> payload{};
  std::memcpy(payload.data(), words.data(), payload.size());
  return payload;
}

/// Fixed-capacity, allocation-free consensus payload used at public Krylov boundaries. Bytewise
/// min/max is intentional: it preserves exact binary64 bits and detects disagreement without
/// interpreting unsigned fingerprints as signed MPI integers.
struct KrylovCollectivePayload {
  static constexpr std::size_t kCapacity = 2048;
  static constexpr std::size_t kUsableCapacity = kCapacity - 1u;
  static constexpr std::size_t kFingerprintBytes = 4u * sizeof(std::uint64_t);
  static constexpr std::size_t kSnapshotBytes = 19u * sizeof(std::uint64_t);

  // The largest current sequence is generic_krylov's solve preflight: prepared-problem state
  // (including its observed snapshot), workspace state, controls, two field contracts, and the
  // alias bit. Keeping this arithmetic here makes the fixed capacity a compile-time contract
  // instead of a rank-local overflow path. Update it with any new payload append sequence.
  static constexpr std::size_t kPreparedProblemContractBytes =
      sizeof(std::uint32_t) + 3u * sizeof(int) + 8u * sizeof(std::uint8_t) + kSnapshotBytes +
      4u * kFingerprintBytes + 2u * sizeof(std::uint8_t) + sizeof(std::uint64_t) +
      kFingerprintBytes + sizeof(std::uint8_t) + sizeof(std::uint64_t) + 2u * kFingerprintBytes +
      sizeof(std::uint8_t) + kSnapshotBytes + kFingerprintBytes + sizeof(std::uint64_t);
  static constexpr std::size_t kPreparedProblemAccessBytes =
      kPreparedProblemContractBytes + sizeof(std::uint8_t) + kSnapshotBytes;
  static constexpr std::size_t kWorkspaceStateBytes = 3u * kFingerprintBytes + 3u * sizeof(int) +
                                                      4u * sizeof(std::uint8_t) +
                                                      8u * sizeof(std::uint64_t) + kSnapshotBytes;
  static constexpr std::size_t kControlsBytes =
      kFingerprintBytes + 3u * sizeof(std::uint64_t) + 2u * sizeof(int) + sizeof(std::uint32_t);
  static constexpr std::size_t kFieldContractBytes = kFingerprintBytes + 2u * sizeof(int);
  static constexpr std::size_t kMaximumKnownPayloadBytes =
      kPreparedProblemAccessBytes + kWorkspaceStateBytes + kControlsBytes +
      2u * kFieldContractBytes + sizeof(std::uint8_t);
  static_assert(kMaximumKnownPayloadBytes <= kUsableCapacity);

  std::array<char, kCapacity> bytes{};
  std::size_t used = 0;

  template <class T>
  void append(const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) <= kUsableCapacity);
    // The last byte is a collective overflow witness and is never payload storage.  A future
    // schema extension that forgets to update kMaximumKnownPayloadBytes therefore fails uniformly
    // after the same min/max reductions instead of terminating or writing past the fixed buffer.
    if (bytes.back() != 0 || used > kUsableCapacity - sizeof(T)) {
      bytes.back() = 1;
      return;
    }
    std::memcpy(bytes.data() + used, &value, sizeof(T));
    used += sizeof(T);
  }

  void append(const OperatorFingerprint& value) noexcept {
    for (const std::uint64_t word : value)
      append(word);
  }

  void append(const OperatorEvaluationSnapshot& value) noexcept {
    append(value.authority);
    append(value.revision);
    append(value.macro_step);
    append(value.stage_numerator);
    append(value.stage_denominator);
    append(value.dt_bits);
    append(value.physical_time_bits);
    append(value.topology_revision);
    append(value.topology);
    append(value.resources);
  }
};

inline bool collective_payload_agrees(const KrylovCollectivePayload& local,
                                      const ExecutionLane& lane) {
  auto minimum = local.bytes;
  auto maximum = local.bytes;
  all_reduce_min_inplace(minimum.data(), minimum.size(), lane);
  all_reduce_max_inplace(maximum.data(), maximum.size(), lane);
  if (maximum.back() != 0)
    throw std::logic_error(
        "prepared Krylov collective payload exceeded its fixed internal capacity");
  return minimum == maximum;
}

inline bool collective_payload_agrees(const KrylovCollectivePayload& local) {
  const ExecutionLane lane = ExecutionLane::world();
  return collective_payload_agrees(local, lane);
}

/// Rank-local outcome of a fallible prepared-problem constructor. Runtime failure deliberately has
/// the highest severity: if one rank rejects an authored contract while another cannot materialize
/// its local resources, every rank must report the execution failure rather than invite a retry of
/// a construction that did not complete everywhere.
enum class PreparedProblemConstructionFailure : long {
  None = 0,
  InvalidArgument = 1,
  RuntimeFailure = 2,
};

}  // namespace detail

/// A fixed preconditioner prepared separately from the affine operator. Its raw callback may contain
/// an affine physical-boundary response; prepare() captures the exact zero response and apply()
/// subtracts it. The preparation hook must allocate/build all native state before the first apply;
/// apply() refuses an unbound or changed snapshot, so no lazy initialization can hide inside a
/// Krylov iteration.
class PreparedLinearPreconditioner {
 public:
  static PreparedLinearPreconditioner identity() { return PreparedLinearPreconditioner(); }

  PreparedLinearPreconditioner() { initialize_provider_fingerprint_(); }
  PreparedLinearPreconditioner(const PreparedLinearPreconditioner&) = delete;
  PreparedLinearPreconditioner& operator=(const PreparedLinearPreconditioner&) = delete;
  PreparedLinearPreconditioner(PreparedLinearPreconditioner&&) noexcept = default;
  PreparedLinearPreconditioner& operator=(PreparedLinearPreconditioner&&) noexcept = default;
  explicit PreparedLinearPreconditioner(
      const MultiFab& prototype, PreparedLinearPreconditionerProvider provider,
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed)
      : provider_(std::move(provider)),
        zero_(prototype.box_array(), prototype.dmap(), prototype.ncomp(), prototype.n_grow()),
        constant_(prototype.box_array(), prototype.dmap(), prototype.ncomp(), prototype.n_grow()),
        layout_(detail::layout_fingerprint(prototype, vector_distribution)),
        vector_distribution_(vector_distribution),
        vector_distribution_layout_valid_(
            detail::field_distribution_layout_matches(prototype, vector_distribution)) {
    if (!provider_)
      throw std::invalid_argument(
          "PreparedLinearPreconditioner requires a non-empty authenticated provider");
    zero_.share_halo_cache_from(prototype);
    constant_.share_halo_cache_from(prototype);
    initialize_provider_fingerprint_();
  }

  bool is_identity() const { return !static_cast<bool>(provider_); }
  bool compatible_with(const MultiFab& prototype,
                       PreparedVectorDistribution vector_distribution) const {
    return is_identity() || (vector_distribution_ == vector_distribution &&
                             layout_ == detail::layout_fingerprint(prototype, vector_distribution));
  }

  void prepare(const OperatorEvaluationSnapshot& snapshot, const ExecutionLane& lane) {
    snapshot_.reset();
    if (!snapshot.valid())
      throw std::invalid_argument("PreparedLinearPreconditioner received an invalid snapshot");
    if (!is_identity()) {
      // The provider source is immutable for this object. Materialize its private session once and
      // refresh that same state for every subsequent snapshot. Construction remains a two-phase
      // collective boundary so one rank never enters a session's possibly-collective prepare while
      // another rank is still unwinding a failed factory.
      const bool reuse_session = all_reduce_min(session_ ? 1L : 0L, lane) != 0;
      if (!reuse_session) {
        // A prior failure may have invalidated only one rank before its owning problem completed
        // the common failure gate. Never let some ranks reuse while peers enter a fresh factory.
        session_ = PreparedLinearPreconditionerSession{};
        PreparedLinearPreconditionerSession candidate;
        long construction_failure_local = 0;
        try {
          candidate = provider_.make_session(lane);
        } catch (...) {
          construction_failure_local = 1;
        }
        if (all_reduce_max(construction_failure_local, lane) != 0) {
          session_ = PreparedLinearPreconditionerSession{};
          throw std::runtime_error(
              "prepared preconditioner session construction failed on at least one communicator "
              "rank");
        }
        session_ = std::move(candidate);
      }

      std::size_t persistent_field_count = 0;
      long preparation_failure_local = 0;
      try {
        session_.reset_apply_status();
        session_.prepare();
        // A physical-BC preconditioner can itself be affine. Evaluate its exact zero response once
        // after all resources are materialized, then subtract it from every search-direction apply.
        // This is the preconditioner analogue of A_lin(v) = A(v) - A(0), and it also warms every
        // callback/halo resource before the iteration begins.
        detail::PreparedFieldAlgebra::zero(zero_);
        detail::PreparedFieldAlgebra::zero(constant_);
      } catch (...) {
        preparation_failure_local = 1;
      }
      if (all_reduce_max(preparation_failure_local, lane) != 0) {
        session_ = PreparedLinearPreconditionerSession{};
        throw std::runtime_error(
            "prepared preconditioner session preparation failed on at least one communicator "
            "rank");
      }

      const PreparedApplyStatus probe_status = session_.apply(constant_, zero_);
      if (all_reduce_max(prepared_apply_succeeded(probe_status) ? 0L : 1L, lane) != 0) {
        session_ = PreparedLinearPreconditionerSession{};
        throw std::runtime_error(
            "prepared preconditioner failed its zero-response probe on at least one "
            "communicator rank");
      }

      long allocation_count_failure_local = 0;
      try {
        persistent_field_count = session_.allocation_count();
        if (persistent_field_count > static_cast<std::size_t>(std::numeric_limits<long>::max()))
          throw std::overflow_error(
              "prepared preconditioner session field count exceeds collective capacity");
      } catch (...) {
        allocation_count_failure_local = 1;
      }
      if (all_reduce_max(allocation_count_failure_local, lane) != 0) {
        session_ = PreparedLinearPreconditionerSession{};
        throw std::runtime_error(
            "prepared preconditioner session allocation-count query failed on at least one "
            "communicator rank");
      }
      const long collective_field_count = static_cast<long>(persistent_field_count);
      if (all_reduce_min(collective_field_count, lane) !=
          all_reduce_max(collective_field_count, lane)) {
        session_ = PreparedLinearPreconditionerSession{};
        throw std::runtime_error(
            "prepared preconditioner session field count differs between communicator ranks");
      }
    }
    snapshot_ = snapshot;
  }

  [[nodiscard]] PreparedApplyStatus apply(MultiFab& out, const MultiFab& in,
                                          const OperatorEvaluationSnapshot& snapshot,
                                          const ExecutionLane& lane) const {
    if (!snapshot_ || *snapshot_ != snapshot)
      throw std::logic_error("prepared preconditioner snapshot changed without preparation");
    if (out.shares_storage_with(in))
      throw std::invalid_argument("prepared preconditioner output must not alias its input");
    if (is_identity()) {
      detail::PreparedFieldAlgebra::copy(out, in);
      return PreparedApplyStatus::Success;
    } else {
      (void)lane;
      if (!session_)
        throw std::logic_error("prepared preconditioner has no materialized execution session");
      const PreparedApplyStatus status = session_.apply(out, in);
      detail::PreparedFieldAlgebra::axpy(out, Real(-1), constant_);
      return status;
    }
  }

 private:
  friend class PreparedAffineLinearProblem;
  friend struct detail::PreparedProblemAccess;

  void invalidate_collective_preparation_() noexcept {
    snapshot_.reset();
    session_ = PreparedLinearPreconditionerSession{};
  }

  void initialize_provider_fingerprint_() {
    ExactContractBuilder contract;
    contract.optional_collective_contract(provider_);
    provider_fingerprint_ = detail::fingerprint_seed();
    detail::fingerprint_mix(provider_fingerprint_, contract.view());
  }

  PreparedLinearPreconditionerProvider provider_{};
  MultiFab zero_{};
  MultiFab constant_{};
  OperatorFingerprint layout_{};
  PreparedVectorDistribution vector_distribution_ = PreparedVectorDistribution::Distributed;
  bool vector_distribution_layout_valid_ = true;
  OperatorFingerprint provider_fingerprint_{};
  PreparedLinearPreconditionerSession session_{};
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
};

/// Owns one affine operator evaluation A(u), its exact constant c=A(0), and the frozen resources
/// captured for that evaluation. Fields/callbacks are created by the constructor; prepare() is the
/// explicit resource-materialization boundary where frozen coefficients, halo/MPI capacities and
/// preconditioners may be warmed before it evaluates A(0). No lazy work may escape into iteration.
class PreparedAffineLinearProblem {
 public:
  static_assert(std::is_nothrow_move_constructible_v<PreparedLinearPreconditioner>);
  static_assert(std::is_nothrow_move_constructible_v<PreparedNullspacePolicy>);
  static_assert(std::is_nothrow_move_constructible_v<OperatorSnapshotProbe>);
  static_assert(std::is_nothrow_move_constructible_v<PreparedVectorDistribution>);
  static_assert(std::is_nothrow_move_constructible_v<PreparedVectorMetric>);

  PreparedAffineLinearProblem(
      const MultiFab& prototype, PreparedAffineOperatorProvider operator_provider,
      PreparedLinearPreconditioner preconditioner, LinearOperatorProperties properties,
      KrylovFootprint footprint, PreparedNullspacePolicy nullspace_policy,
      OperatorSnapshotProbe snapshot_probe, PreparedResourceFn freeze_resources = {},
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : PreparedAffineLinearProblem(
            ExecutionCommunicator::world(), "pops.prepared-affine-problem", prototype,
            std::move(operator_provider), std::move(preconditioner), properties, footprint,
            std::move(nullspace_policy), std::move(snapshot_probe), std::move(freeze_resources),
            std::move(vector_distribution), std::move(metric)) {}

  /// Materialize a prepared problem on an authenticated communicator. `lane_identity` is the
  /// caller-owned stable identity of this problem within the parent's canonical creation order.
  /// Every numerical/control collective subsequently stays inside the duplicated lane.
  PreparedAffineLinearProblem(
      const ExecutionCommunicator& execution_communicator, std::string_view lane_identity,
      const MultiFab& prototype, PreparedAffineOperatorProvider operator_provider,
      PreparedLinearPreconditioner preconditioner, LinearOperatorProperties properties,
      KrylovFootprint footprint, PreparedNullspacePolicy nullspace_policy,
      OperatorSnapshotProbe snapshot_probe, PreparedResourceFn freeze_resources = {},
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : operator_provider_(std::move(operator_provider)),
        preparation_lane_(
            ExecutionLane::duplicate_collectively(execution_communicator, lane_identity)),
        preconditioner_(std::move(preconditioner)),
        properties_(properties),
        footprint_(footprint),
        nullspace_policy_(std::move(nullspace_policy)),
        snapshot_probe_(std::move(snapshot_probe)),
        freeze_resources_(std::move(freeze_resources)),
        vector_distribution_(std::move(vector_distribution)),
        metric_(std::move(metric)) {
    // The lane already exists, so every fallible rank-local materialization/validation below is
    // captured before one common gate. If only one rank rejects its local layout or runs out of
    // memory, all ranks leave the constructor together and can free the duplicated communicator in
    // the same order. Every rank observes the same exception type and message, so callers cannot
    // accidentally choose divergent recovery paths from a rank-local implementation detail.
    detail::PreparedProblemConstructionFailure local_construction_failure =
        detail::PreparedProblemConstructionFailure::None;
    try {
      if (!metric_)
        metric_ = PreparedVectorMetric::euclidean(prototype, vector_distribution_);
      zero_ = MultiFab(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                       footprint.input_ghosts);
      constant_ = MultiFab(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                           footprint.input_ghosts);
      layout_ = detail::layout_fingerprint(prototype, vector_distribution_);
      vector_distribution_layout_valid_ =
          detail::field_distribution_layout_matches(prototype, vector_distribution_);
      zero_.share_halo_cache_from(prototype);
      constant_.share_halo_cache_from(prototype);
      if (!operator_provider_)
        throw std::invalid_argument(
            "PreparedAffineLinearProblem requires an affine operator provider");
      if (!properties_.valid())
        throw std::invalid_argument("PreparedAffineLinearProblem received incoherent properties");
      if (nullspace_policy_.singular() &&
          properties_.has(LinearOperatorProperty::kPositiveDefinite))
        throw std::invalid_argument(
            "a prepared singular operator cannot carry a global positive-definite certificate");
      if (!nullspace_policy_.singular() &&
          properties_.has(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement))
        throw std::invalid_argument(
            "a nullspace-complement certificate requires a prepared nullspace policy");
      if (footprint_.components != prototype.ncomp() || footprint_.components < 1 ||
          footprint_.input_ghosts < 0 || footprint_.input_ghosts != prototype.n_grow())
        throw std::invalid_argument(
            "PreparedAffineLinearProblem footprint disagrees with prototype");
      if (footprint_.preconditioned != !preconditioner_.is_identity())
        throw std::invalid_argument(
            "PreparedAffineLinearProblem footprint disagrees with preconditioner presence");
      if (!preconditioner_.compatible_with(prototype, vector_distribution_))
        throw std::invalid_argument(
            "PreparedAffineLinearProblem preconditioner layout disagrees with prototype");
      if (!snapshot_probe_)
        throw std::invalid_argument("PreparedAffineLinearProblem requires a snapshot probe");
      if (!metric_.compatible_with(prototype, vector_distribution_))
        throw std::invalid_argument(
            "PreparedAffineLinearProblem metric disagrees with the solved vector space");
      metric_fingerprint_ = detail::fingerprint_seed();
      detail::fingerprint_mix(metric_fingerprint_, metric_.collective_contract());
      operator_provider_fingerprint_ = detail::fingerprint_seed();
      detail::fingerprint_mix(operator_provider_fingerprint_,
                              operator_provider_.collective_contract());
      nullspace_policy_.plan_fingerprint_ = compute_nullspace_plan_fingerprint_();
    } catch (const std::invalid_argument&) {
      local_construction_failure = detail::PreparedProblemConstructionFailure::InvalidArgument;
    } catch (...) {
      local_construction_failure = detail::PreparedProblemConstructionFailure::RuntimeFailure;
    }
    const auto collective_construction_failure =
        static_cast<detail::PreparedProblemConstructionFailure>(
            all_reduce_max(static_cast<long>(local_construction_failure), preparation_lane_));
    if (collective_construction_failure ==
        detail::PreparedProblemConstructionFailure::InvalidArgument) {
      throw std::invalid_argument(
          "PreparedAffineLinearProblem received invalid construction arguments on at least one "
          "communicator rank");
    }
    if (collective_construction_failure ==
        detail::PreparedProblemConstructionFailure::RuntimeFailure) {
      throw std::runtime_error(
          "PreparedAffineLinearProblem construction failed on at least one communicator rank");
    }
  }

  PreparedAffineLinearProblem(
      const MultiFab& prototype, PreparedAffineOperatorProvider operator_provider,
      PreparedLinearPreconditioner preconditioner, LinearOperatorProperties properties,
      KrylovFootprint footprint, PreparedNullspacePolicy nullspace_policy,
      OperatorSnapshotProbe snapshot_probe, PreparedResourceFn freeze_resources,
      detail::AuthenticatedProgramApplyToken authenticated_program,
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : PreparedAffineLinearProblem(
            prototype, std::move(operator_provider), std::move(preconditioner), properties,
            footprint, std::move(nullspace_policy), std::move(snapshot_probe),
            std::move(freeze_resources), std::move(vector_distribution), std::move(metric)) {
    authenticated_program_authority_ = authenticated_program.authority_;
  }

  PreparedAffineLinearProblem(
      const ExecutionCommunicator& execution_communicator, std::string_view lane_identity,
      const MultiFab& prototype, PreparedAffineOperatorProvider operator_provider,
      PreparedLinearPreconditioner preconditioner, LinearOperatorProperties properties,
      KrylovFootprint footprint, PreparedNullspacePolicy nullspace_policy,
      OperatorSnapshotProbe snapshot_probe, PreparedResourceFn freeze_resources,
      detail::AuthenticatedProgramApplyToken authenticated_program,
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : PreparedAffineLinearProblem(execution_communicator, lane_identity, prototype,
                                    std::move(operator_provider), std::move(preconditioner),
                                    properties, footprint, std::move(nullspace_policy),
                                    std::move(snapshot_probe), std::move(freeze_resources),
                                    std::move(vector_distribution), std::move(metric)) {
    authenticated_program_authority_ = authenticated_program.authority_;
  }

  void prepare(const OperatorEvaluationSnapshot& snapshot) {
    require_collective_prepare_contract_();
    if (all_reduce_max(active_solve_reservations_.load(std::memory_order_acquire) != 0 ? 1L : 0L,
                       preparation_lane_) != 0)
      throw std::logic_error(
          "PreparedAffineLinearProblem cannot be prepared while a solve invocation is active");
    snapshot_.reset();
    try {
      // This helper performs its own rank-local allocation catch and preparation-lane failure
      // reduction. It therefore completes before the snapshot probe enters its first collective.
      auto distribution_validation = make_distribution_validation_scratch_();
      std::size_t operator_persistent_field_count = 0;
      require_collective_prepare_snapshot_(snapshot);
      const long authenticated_authority_failure =
          all_reduce_max(authenticated_program_authority_.has_value() &&
                                 snapshot.authority != *authenticated_program_authority_
                             ? 1L
                             : 0L,
                         preparation_lane_);
      if (authenticated_authority_failure != 0)
        throw std::invalid_argument(
            "compiled Program operator authority disagrees with its authenticated apply token");
      run_collective_prepare_stage_(PrepareStage::kFreeze, [&] {
        if (freeze_resources_)
          freeze_resources_();
      });
      require_collective_snapshot_match_(snapshot, PrepareStage::kFreeze);

      // The provider source is immutable for this problem. Reuse its materialized session across
      // snapshots, but make the reuse decision collective: after any prior partial failure every
      // rank either refreshes the existing state or every rank rematerializes it.
      const bool reuse_operator_session =
          all_reduce_min(operator_session_ ? 1L : 0L, preparation_lane_) != 0;
      if (!reuse_operator_session) {
        PreparedAffineOperatorSession candidate;
        long operator_session_failure_local = 0;
        try {
          candidate = operator_provider_.make_session(preparation_lane_);
        } catch (...) {
          operator_session_failure_local = 1;
        }
        if (all_reduce_max(operator_session_failure_local, preparation_lane_) != 0)
          throw std::runtime_error(
              "prepared affine operator session construction failed on at least one communicator "
              "rank");
        operator_session_ = std::move(candidate);
      }

      run_collective_prepare_stage_(PrepareStage::kOperatorSession, [&] {
        operator_session_.reset_apply_status();
        operator_session_.prepare();
        operator_persistent_field_count = operator_session_.allocation_count();
        if (operator_persistent_field_count >
            static_cast<std::size_t>(std::numeric_limits<long>::max()))
          throw std::overflow_error(
              "prepared affine operator session field count exceeds collective capacity");
      });
      require_collective_session_field_count_(operator_persistent_field_count,
                                              "prepared affine operator session");
      require_collective_snapshot_match_(snapshot, PrepareStage::kOperatorSession);
      prepare_nullspace_collectively_(snapshot);
      run_collective_prepare_stage_(PrepareStage::kPreconditioner,
                                    [&] { preconditioner_.prepare(snapshot, preparation_lane_); });
      require_collective_snapshot_match_(snapshot, PrepareStage::kPreconditioner);
      if (!preconditioner_.is_identity())
        vector_distribution_.require_exact_values(
            preconditioner_.constant_, distribution_validation, "prepared preconditioner constant",
            preparation_lane_);
      run_collective_prepare_stage_(PrepareStage::kOperatorConstant, [&] {
        detail::PreparedFieldAlgebra::zero(zero_);
        if (!prepared_apply_succeeded(operator_session_.apply(constant_, zero_)))
          throw std::runtime_error("prepared affine operator failed its zero-response probe");
        operator_persistent_field_count = operator_session_.allocation_count();
        if (operator_persistent_field_count >
            static_cast<std::size_t>(std::numeric_limits<long>::max()))
          throw std::overflow_error(
              "prepared affine operator session field count exceeds collective capacity");
      });
      require_collective_session_field_count_(operator_persistent_field_count,
                                              "prepared affine operator session");
      require_collective_snapshot_match_(snapshot, PrepareStage::kOperatorConstant);
      vector_distribution_.require_exact_values(constant_, distribution_validation,
                                                "prepared operator constant", preparation_lane_);
      snapshot_ = snapshot;
    } catch (...) {
      // A failed refresh may have mutated opaque provider state on only a subset of ranks. Retaining
      // it would make the next prepare non-deterministic, so poison the whole execution-session set
      // uniformly and force fresh factories on the retry.
      invalidate_execution_sessions_();
      throw;
    }
  }

  bool prepared() const { return snapshot_.has_value(); }
  const OperatorEvaluationSnapshot& snapshot() const {
    require_prepared_local_("PreparedAffineLinearProblem::snapshot");
    return *snapshot_;
  }
  const OperatorFingerprint& layout_fingerprint() const { return layout_; }
  const PreparedVectorDistribution& vector_distribution() const { return vector_distribution_; }
  const PreparedVectorMetric& metric() const { return metric_; }
  const LinearOperatorProperties& properties() const { return properties_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  const MultiFab& constant_term() const {
    require_prepared_local_("PreparedAffineLinearProblem::constant_term");
    return constant_;
  }
  bool has_preconditioner() const { return !preconditioner_.is_identity(); }
  bool has_nullspace() const { return nullspace_policy_.singular(); }

  void require_nullspace_compatible(const MultiFab& normalized_rhs) const {
    require_current();
    require_collective_arguments_(vector_field_failure_(normalized_rhs),
                                  "PreparedAffineLinearProblem::require_nullspace_compatible");
    auto metric_scratch = make_metric_scratch_();
    nullspace_policy_.require_compatible(normalized_rhs, metric_, metric_scratch,
                                         preparation_lane_);
  }

  void apply_nullspace_gauge(MultiFab& iterate) const {
    require_current();
    require_collective_arguments_(operator_field_failure_(iterate),
                                  "PreparedAffineLinearProblem::apply_nullspace_gauge");
    auto metric_scratch = make_metric_scratch_();
    auto gauge_scratch = make_gauge_scratch_();
    nullspace_policy_.apply_gauge(iterate, metric_, gauge_scratch, metric_scratch,
                                  preparation_lane_);
  }

  void effective_rhs(MultiFab& out, const MultiFab& rhs) const {
    require_current();
    long failure = std::max(vector_field_failure_(rhs), operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, rhs));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::effective_rhs");
    auto validation_scratch = make_distribution_validation_scratch_();
    require_public_replica_input_(rhs, validation_scratch, "prepared effective rhs input");
    effective_rhs_prepared_(out, rhs);
  }

  /// A_lin(v) = A(v) - A(0), valid for search directions even when boundaries/sources make A affine.
  void apply_linear(MultiFab& out, const MultiFab& direction) const {
    require_current();
    long failure = std::max(operator_field_failure_(direction), operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, direction));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::apply_linear");
    auto validation_scratch = make_distribution_validation_scratch_();
    require_public_replica_input_(direction, validation_scratch, "prepared linear operator input");
    apply_linear_prepared_(out, direction, validation_scratch);
  }

  /// Scientific residual R(u) = b - A(u), never a preconditioned or Arnoldi estimate.
  void true_residual(MultiFab& out, const MultiFab& rhs, const MultiFab& iterate) const {
    require_current();
    long failure = std::max(vector_field_failure_(rhs), operator_field_failure_(iterate));
    failure = std::max(failure, operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, rhs));
    failure = std::max(failure, distinct_storage_failure_(out, iterate));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::true_residual");
    auto validation_scratch = make_distribution_validation_scratch_();
    require_public_replica_input_(rhs, validation_scratch, "prepared residual rhs");
    require_public_replica_input_(iterate, validation_scratch, "prepared residual iterate");
    true_residual_prepared_(out, rhs, iterate, validation_scratch);
  }

  void apply_preconditioner(MultiFab& out, const MultiFab& in) const {
    require_current();
    long failure = std::max(operator_field_failure_(in), operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, in));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::apply_preconditioner");
    auto validation_scratch = make_distribution_validation_scratch_();
    require_public_replica_input_(in, validation_scratch, "prepared preconditioner input");
    apply_preconditioner_prepared_(out, in, validation_scratch);
  }

  /// The delivered prepared metric is one global L2 product over every component and rank. Keeping
  /// it on the problem (rather than inside individual algorithms) gives every method and report one
  /// authority and leaves a narrow metric-provider seam for a future weighted/composite route.
  Real inner_product(const MultiFab& left, const MultiFab& right) const {
    require_current();
    require_collective_arguments_(
        std::max(vector_field_failure_(left), vector_field_failure_(right)),
        "PreparedAffineLinearProblem::inner_product");
    auto validation_scratch = make_distribution_validation_scratch_();
    require_public_replica_input_(left, validation_scratch, "prepared inner-product left input");
    require_public_replica_input_(right, validation_scratch, "prepared inner-product right input");
    auto metric_scratch = make_metric_scratch_();
    return metric_.inner_product(left, right, metric_scratch, preparation_lane_);
  }

  Real residual_norm(const MultiFab& value) const {
    require_current();
    require_collective_arguments_(vector_field_failure_(value),
                                  "PreparedAffineLinearProblem::residual_norm");
    auto validation_scratch = make_distribution_validation_scratch_();
    require_public_replica_input_(value, validation_scratch, "prepared residual-norm input");
    auto metric_scratch = make_metric_scratch_();
    return metric_.norm(value, metric_scratch, preparation_lane_);
  }

 private:
  static constexpr long kAliasFailure = 1;
  static constexpr long kGhostFailure = 2;
  static constexpr long kVectorSpaceFailure = 3;

  long vector_field_failure_(const MultiFab& value) const noexcept {
    return PureFieldAlgebra::same_vector_space(value, zero_) ? 0 : kVectorSpaceFailure;
  }

  long operator_field_failure_(const MultiFab& value) const noexcept {
    const long vector_failure = vector_field_failure_(value);
    if (vector_failure != 0)
      return vector_failure;
    return value.n_grow() == footprint_.input_ghosts ? 0 : kGhostFailure;
  }

  static long distinct_storage_failure_(const MultiFab& out, const MultiFab& in) noexcept {
    return out.shares_storage_with(in) ? kAliasFailure : 0;
  }

  void require_collective_arguments_(long local_failure, const char* where) const {
    const long collective_failure = all_reduce_max(local_failure, preparation_lane_);
    if (collective_failure == kVectorSpaceFailure)
      throw std::invalid_argument(std::string(where) + ": incompatible vector space");
    if (collective_failure == kGhostFailure)
      throw std::invalid_argument(std::string(where) + ": incompatible ghost footprint");
    if (collective_failure == kAliasFailure)
      throw std::invalid_argument(std::string(where) + ": output aliases an input field");
  }

  void require_prepared_local_(const char* where) const {
    if (!snapshot_)
      throw std::logic_error(std::string(where) + ": problem is not prepared");
  }

  void require_current(const ExecutionLane& lane) const {
    OperatorEvaluationSnapshot observed{};
    bool probe_failed = false;
    try {
      observed = snapshot_probe_();
    } catch (...) {
      probe_failed = true;
    }
    long local_failure = 0;
    if (!snapshot_)
      local_failure = 3;
    else if (probe_failed)
      local_failure = 2;
    else if (observed != *snapshot_)
      local_failure = 1;
    const long collective_failure = all_reduce_max(local_failure, lane);
    if (collective_failure == 3)
      throw std::logic_error(
          "PreparedAffineLinearProblem is not prepared on every communicator rank");
    if (collective_failure == 2)
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    if (collective_failure == 1)
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
  }

  void require_current() const { require_current(preparation_lane_); }

  std::vector<double> make_metric_scratch_() const {
    std::vector<double> scratch;
    long local_failure = 0;
    try {
      scratch.assign(metric_.reduction_scratch_value_count(), 0.0);
    } catch (...) {
      local_failure = 1;
    }
    if (all_reduce_max(local_failure, preparation_lane_) != 0)
      throw std::runtime_error(
          "prepared vector metric scratch allocation failed on at least one communicator rank");
    return scratch;
  }

  std::vector<double> make_gauge_scratch_() const {
    std::vector<double> scratch;
    long local_failure = 0;
    try {
      scratch.assign(nullspace_policy_.basis_vectors_.size(), 0.0);
    } catch (...) {
      local_failure = 1;
    }
    if (all_reduce_max(local_failure, preparation_lane_) != 0)
      throw std::runtime_error(
          "prepared nullspace gauge scratch allocation failed on at least one communicator rank");
    return scratch;
  }

  std::vector<char, comm_allocator<char>> make_distribution_validation_scratch_() const {
    std::vector<char, comm_allocator<char>> scratch;
    long local_failure = 0;
    try {
      scratch.assign(vector_distribution_.validation_scratch_byte_count(), char{0});
    } catch (...) {
      local_failure = 1;
    }
    if (all_reduce_max(local_failure, preparation_lane_) != 0)
      throw std::runtime_error(
          "prepared vector-distribution validation scratch allocation failed on at least one "
          "communicator rank");
    return scratch;
  }

  friend struct detail::PreparedProblemAccess;

  enum class PrepareStage : std::uint8_t {
    kFreeze,
    kOperatorSession,
    kNullspace,
    kPreconditioner,
    kOperatorConstant,
  };

  struct NullspaceCertificateIdentity {
    std::uint64_t topology_revision = 0;
    OperatorFingerprint topology{};
    OperatorFingerprint resources{};

    friend bool operator==(const NullspaceCertificateIdentity&,
                           const NullspaceCertificateIdentity&) = default;
  };

  static NullspaceCertificateIdentity nullspace_certificate_identity_(
      const OperatorEvaluationSnapshot& snapshot) noexcept {
    return {snapshot.topology_revision, snapshot.topology, snapshot.resources};
  }

  OperatorFingerprint compute_nullspace_plan_fingerprint_() const {
    OperatorFingerprint hash = detail::fingerprint_seed();
    const FieldNullspacePlan& plan = nullspace_policy_.plan_;
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(
                                      static_cast<std::int64_t>(nullspace_policy_.first_level_)));
    detail::fingerprint_mix(hash, plan.identity);
    detail::fingerprint_mix(hash, plan.layout_identity);
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(plan.bases.size()));
    for (const FieldNullspaceBasis& basis : plan.bases) {
      detail::fingerprint_mix(hash, basis.identity);
      detail::fingerprint_mix(hash, basis.provenance);
      detail::fingerprint_mix(hash, basis.recipe_identity);
      detail::fingerprint_mix(
          hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(basis.field_component)));
      detail::fingerprint_mix(hash, static_cast<std::uint64_t>(basis.masks.size()));
      for (std::size_t level = 0; level < basis.masks.size(); ++level) {
        const auto& mask = basis.masks[level];
        detail::fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<bool>(mask)));
        if (mask) {
          for (const std::uint64_t word : detail::layout_fingerprint(*mask, vector_distribution_))
            detail::fingerprint_mix(hash, word);
        }
      }
      detail::fingerprint_mix(hash, static_cast<std::uint64_t>(basis.coverage.size()));
      for (std::size_t level = 0; level < basis.coverage.size(); ++level) {
        const auto& coverage = basis.coverage[level];
        detail::fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<bool>(coverage)));
        if (coverage) {
          for (const std::uint64_t word :
               detail::layout_fingerprint(*coverage, vector_distribution_))
            detail::fingerprint_mix(hash, word);
        }
      }
      detail::fingerprint_mix(hash, static_cast<std::uint64_t>(basis.cell_measure.size()));
      for (const Real measure : basis.cell_measure)
        detail::fingerprint_mix(hash, std::bit_cast<std::uint64_t>(measure));
    }
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(plan.gauges.size()));
    for (const FieldGaugeConstraint& gauge : plan.gauges) {
      detail::fingerprint_mix(hash, gauge.basis_identity);
      detail::fingerprint_mix(hash, std::bit_cast<std::uint64_t>(gauge.value));
    }
    return hash;
  }

  void append_collective_contract_(detail::KrylovCollectivePayload& payload) const noexcept {
    payload.append(properties_.bits);
    payload.append(footprint_.components);
    payload.append(footprint_.input_ghosts);
    payload.append(static_cast<std::uint8_t>(footprint_.preconditioned));
    payload.append(static_cast<std::uint8_t>(preconditioner_.is_identity()));
    payload.append(static_cast<std::uint8_t>(preconditioner_.snapshot_.has_value()));
    payload.append(preconditioner_.snapshot_.value_or(OperatorEvaluationSnapshot{}));
    payload.append(static_cast<std::uint8_t>(authenticated_program_authority_.has_value()));
    payload.append(authenticated_program_authority_.value_or(OperatorFingerprint{}));
    payload.append(static_cast<std::uint8_t>(vector_distribution_layout_valid_));
    payload.append(static_cast<std::uint8_t>(preconditioner_.vector_distribution_layout_valid_));
    payload.append(preconditioner_.provider_fingerprint_);
    payload.append(operator_provider_fingerprint_);
    payload.append(preconditioner_.layout_);
    payload.append(layout_);
    payload.append(static_cast<std::uint8_t>(nullspace_policy_.singular_));
    payload.append(static_cast<std::uint8_t>(nullspace_policy_.prepared_));
    payload.append(static_cast<std::uint64_t>(nullspace_policy_.basis_vectors_.size()));
    payload.append(nullspace_policy_.plan_fingerprint_);
    payload.append(static_cast<std::uint8_t>(nullspace_certificate_identity_cache_.has_value()));
    const NullspaceCertificateIdentity nullspace_identity =
        nullspace_certificate_identity_cache_.value_or(NullspaceCertificateIdentity{});
    payload.append(nullspace_identity.topology_revision);
    payload.append(nullspace_identity.topology);
    payload.append(nullspace_identity.resources);
    payload.append(static_cast<std::uint8_t>(snapshot_.has_value()));
    payload.append(snapshot_.value_or(OperatorEvaluationSnapshot{}));
    payload.append(metric_fingerprint_);
    payload.append(static_cast<std::uint64_t>(metric_.robust_payload_width()));
  }

  void require_collective_prepare_contract_() const {
    detail::KrylovCollectivePayload payload;
    append_collective_contract_(payload);
    const bool agrees = detail::collective_payload_agrees(payload, preparation_lane_);
    const bool metric_agrees = all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("pops.prepared-vector-metric"), metric_.collective_contract()}},
        preparation_lane_);
    require_prepared_provider_collective_consensus(preconditioner_.provider_, preparation_lane_);
    require_prepared_provider_collective_consensus(operator_provider_, preparation_lane_);
    const bool valid_layout =
        vector_distribution_layout_valid_ &&
        (preconditioner_.is_identity() || preconditioner_.vector_distribution_layout_valid_);
    const long invalid_layout = all_reduce_max(valid_layout ? 0L : 1L, preparation_lane_);
    if (invalid_layout != 0)
      throw std::invalid_argument(
          "prepared vector-distribution provider rejected the authenticated layout");
    if (!metric_agrees)
      throw std::invalid_argument(
          "prepared vector metric contract differs across communicator ranks");
    if (!agrees)
      throw std::logic_error(
          "prepared affine problem contract differs across communicator ranks before preparation");
    detail::require_collective_field_distribution_layout(
        zero_, vector_distribution_, "PreparedAffineLinearProblem::prepare", preparation_lane_);
    if (!preconditioner_.is_identity())
      detail::require_collective_field_distribution_layout(
          preconditioner_.zero_, vector_distribution_,
          "PreparedAffineLinearProblem::prepare(preconditioner)", preparation_lane_);
  }

  template <class Operation>
  void run_collective_prepare_stage_(PrepareStage stage, Operation&& operation) {
    long local_failure = 0;
    try {
      std::forward<Operation>(operation)();
    } catch (...) {
      local_failure = 1;
    }
    if (all_reduce_max(local_failure, preparation_lane_) == 0)
      return;
    switch (stage) {
      case PrepareStage::kFreeze:
        throw std::logic_error("prepared resource freeze failed on at least one communicator rank");
      case PrepareStage::kNullspace:
        throw std::logic_error("prepared nullspace setup failed on at least one communicator rank");
      case PrepareStage::kOperatorSession:
        throw std::logic_error(
            "prepared affine operator session setup failed on at least one communicator rank");
      case PrepareStage::kPreconditioner:
        throw std::logic_error(
            "prepared preconditioner setup failed on at least one communicator rank");
      case PrepareStage::kOperatorConstant:
        throw std::logic_error(
            "prepared affine constant evaluation failed on at least one communicator rank");
    }
    throw std::logic_error("unknown prepared affine setup stage");
  }

  void prepare_nullspace_collectively_(const OperatorEvaluationSnapshot& snapshot) {
    // The prepared bit, plan fingerprint and cache identity have already passed the fixed
    // collective contract. A change in the authenticated topology/resource identity invalidates
    // the Gram certificate even when the provider keeps the same field allocation; time/stage-only
    // changes retain the cold certificate and never add work to a Krylov iteration.
    const NullspaceCertificateIdentity identity = nullspace_certificate_identity_(snapshot);
    if (nullspace_policy_.prepared_ && nullspace_certificate_identity_cache_ == identity)
      return;
    nullspace_policy_.invalidate_collective_certificate_();
    nullspace_certificate_identity_cache_.reset();
    try {
      run_collective_prepare_stage_(PrepareStage::kNullspace, [&] {
        nullspace_policy_.prepare_after_collective_preflight_(zero_, metric_, preparation_lane_);
      });
      nullspace_certificate_identity_cache_ = identity;
    } catch (...) {
      // run_collective_prepare_stage_ has completed its common failure reduction. Clear the
      // certificate everywhere before exposing the uniform failure, otherwise a rank that
      // completed the Gram check could make the next preflight disagree with a failing rank.
      nullspace_policy_.invalidate_collective_certificate_();
      nullspace_certificate_identity_cache_.reset();
      throw;
    }
  }

  void require_collective_prepare_snapshot_(const OperatorEvaluationSnapshot& expected) const {
    long local_failure = expected.valid() ? 0 : 3;
    try {
      const OperatorEvaluationSnapshot observed = snapshot_probe_();
      if (local_failure == 0 && observed != expected)
        local_failure = 1;
    } catch (...) {
      if (local_failure == 0)
        local_failure = 2;
    }

    auto minimum_payload = detail::snapshot_consensus_payload(expected);
    auto maximum_payload = minimum_payload;
    all_reduce_min_inplace(minimum_payload.data(), minimum_payload.size(), preparation_lane_);
    all_reduce_max_inplace(maximum_payload.data(), maximum_payload.size(), preparation_lane_);
    const long collective_failure = all_reduce_max(local_failure, preparation_lane_);

    if (collective_failure == 3)
      throw std::invalid_argument(
          "PreparedAffineLinearProblem received an invalid snapshot on at least one "
          "communicator rank");
    if (collective_failure == 2)
      throw std::logic_error(
          "operator snapshot probe failed before preparation on at least one communicator rank");
    if (collective_failure == 1)
      throw std::logic_error(
          "operator snapshot changed before preparation on at least one communicator rank");
    if (minimum_payload != maximum_payload)
      throw std::logic_error(
          "operator snapshot differs across communicator ranks before preparation");
  }

  void require_collective_snapshot_match_(const OperatorEvaluationSnapshot& expected,
                                          PrepareStage stage) const {
    long local_failure = 0;
    try {
      if (snapshot_probe_() != expected)
        local_failure = 1;
    } catch (...) {
      local_failure = 2;
    }
    const long collective_failure = all_reduce_max(local_failure, preparation_lane_);
    if (collective_failure == 2) {
      if (stage == PrepareStage::kFreeze)
        throw std::logic_error(
            "operator snapshot probe failed during resource preparation on at least one "
            "communicator rank");
      if (stage == PrepareStage::kPreconditioner)
        throw std::logic_error(
            "operator snapshot probe failed during preconditioner preparation on at least one "
            "communicator rank");
      throw std::logic_error(
          "operator snapshot probe failed during operator preparation on at least one "
          "communicator rank");
    }
    if (collective_failure == 1) {
      if (stage == PrepareStage::kFreeze)
        throw std::logic_error(
            "operator snapshot changed during resource preparation on at least one communicator "
            "rank");
      if (stage == PrepareStage::kPreconditioner)
        throw std::logic_error(
            "operator snapshot changed during preconditioner preparation on at least one "
            "communicator rank");
      throw std::logic_error(
          "operator snapshot changed during operator preparation on at least one communicator "
          "rank");
    }
  }

  void require_hot_apply_ready_() const {
    // Public direct-call methods authenticate immediately below. Workspace calls have already passed
    // the reserved solve preflight and therefore need neither a snapshot probe nor an MPI control
    // reduction for each matvec, irrespective of provider origin.
    if (!snapshot_)
      throw std::logic_error("authenticated prepared operator was used before preparation");
  }

  void invalidate_execution_sessions_() noexcept {
    snapshot_.reset();
    operator_session_ = PreparedAffineOperatorSession{};
    preconditioner_.invalidate_collective_preparation_();
  }

  void require_collective_session_field_count_(std::size_t count, const char* provider) const {
    const long collective_count = static_cast<long>(count);
    if (all_reduce_min(collective_count, preparation_lane_) !=
        all_reduce_max(collective_count, preparation_lane_))
      throw std::runtime_error(std::string(provider) +
                               " field count differs between communicator ranks");
  }

  template <class Operation>
  void run_external_apply_(Operation&& operation, const ExecutionLane& lane) const {
    long local_failure = 0;
    try {
      if (!prepared_apply_succeeded(std::forward<Operation>(operation)()))
        local_failure = 4;
    } catch (...) {
      local_failure = 3;
    }
    OperatorEvaluationSnapshot observed{};
    try {
      observed = snapshot_probe_();
      if (local_failure == 0 && (!snapshot_ || observed != *snapshot_))
        local_failure = 1;
    } catch (...) {
      if (local_failure == 0)
        local_failure = 2;
    }
    const long collective_failure = all_reduce_max(local_failure, lane);
    if (collective_failure == 3)
      throw std::logic_error(
          "external prepared operator callback failed on at least one communicator rank");
    if (collective_failure == 4)
      throw std::logic_error(
          "external prepared operator application reported failure on at least one communicator "
          "rank");
    if (collective_failure == 2)
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    if (collective_failure == 1)
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
  }

  void effective_rhs_prepared_(MultiFab& out, const MultiFab& rhs) const {
    require_hot_apply_ready_();
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), constant_);
  }

  PreparedEquationReference prepare_compatibility_rhs_prepared_(MultiFab& out, const MultiFab& rhs,
                                                                std::span<double> metric_scratch,
                                                                const ExecutionLane& lane) const {
    require_hot_apply_ready_();
    // Form the exact floating-point R(0)=b-A(0) first. The scale-safe norm below avoids squaring
    // overflow/underflow without globally rescaling b and A(0) before subtraction, which could
    // erase a small but representable residual on cells far below an unrelated global maximum.
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), constant_);
    const Real reference = metric_.norm(out, metric_scratch, lane);
    if (!std::isfinite(static_cast<double>(reference)))
      return {reference};
    if (reference > Real(0)) {
      // Divide directly instead of materializing 1/reference, which can overflow for a finite
      // subnormal reference.
      detail::PreparedFieldAlgebra::divide(out, reference);
      return {reference};
    }
    return {Real(0)};
  }

  void apply_linear_prepared_(MultiFab& out, const MultiFab& direction,
                              std::span<char> validation_scratch) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { return operator_session_.apply(out, direction); }, preparation_lane_);
    require_verified_replica_output_(out, validation_scratch, "prepared linear operator output",
                                     preparation_lane_);
    detail::PreparedFieldAlgebra::axpy(out, Real(-1), constant_);
  }

  void apply_linear_normalized_prepared_(MultiFab& out, const MultiFab& direction,
                                         Real equation_scale,
                                         std::span<char> validation_scratch) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { return operator_session_.apply(out, direction); }, preparation_lane_);
    require_verified_replica_output_(out, validation_scratch, "prepared normalized operator output",
                                     preparation_lane_);
    detail::PreparedFieldAlgebra::normalized_difference(out, out, constant_, equation_scale);
  }

  void true_residual_prepared_(MultiFab& out, const MultiFab& rhs, const MultiFab& iterate,
                               std::span<char> validation_scratch) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { return operator_session_.apply(out, iterate); }, preparation_lane_);
    require_verified_replica_output_(out, validation_scratch, "prepared residual operator output",
                                     preparation_lane_);
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), out);
  }

  void apply_preconditioner_prepared_(MultiFab& out, const MultiFab& in,
                                      std::span<char> validation_scratch) const {
    require_hot_apply_ready_();
    run_external_apply_(
        [&] { return preconditioner_.apply(out, in, *snapshot_, preparation_lane_); },
        preparation_lane_);
    require_verified_replica_output_(out, validation_scratch, "prepared preconditioner output",
                                     preparation_lane_);
  }

  [[nodiscard]] PreparedApplyStatus apply_workspace_preconditioner_prepared_(
      PreparedLinearPreconditionerSession& session, const MultiFab& affine_constant, MultiFab& out,
      const MultiFab& in) const {
    require_hot_apply_ready_();
    const PreparedApplyStatus status = session.apply(out, in);
    detail::PreparedFieldAlgebra::axpy(out, Real(-1), affine_constant);
    return status;
  }

  [[nodiscard]] PreparedApplyStatus apply_workspace_linear_prepared_(
      PreparedAffineOperatorSession& session, MultiFab& out, const MultiFab& direction,
      Real equation_scale) const {
    require_hot_apply_ready_();
    const PreparedApplyStatus status = session.apply(out, direction);
    detail::PreparedFieldAlgebra::normalized_difference(out, out, constant_, equation_scale);
    return status;
  }

  [[nodiscard]] PreparedApplyStatus workspace_true_residual_prepared_(
      PreparedAffineOperatorSession& session, MultiFab& out, const MultiFab& rhs,
      const MultiFab& iterate) const {
    require_hot_apply_ready_();
    const PreparedApplyStatus status = session.apply(out, iterate);
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), out);
    return status;
  }

  void require_verified_replica_output_(const MultiFab& output, std::span<char> validation_scratch,
                                        const char* where, const ExecutionLane& lane) const {
    if (authenticated_program_authority_)
      return;
    vector_distribution_.require_exact_values(output, validation_scratch, where, lane);
  }

  void require_public_replica_input_(const MultiFab& input, std::span<char> validation_scratch,
                                     const char* where) const {
    vector_distribution_.require_exact_values(input, validation_scratch, where, preparation_lane_);
  }

  Real inner_product_prepared_(const MultiFab& left, const MultiFab& right,
                               std::span<double> metric_scratch, const ExecutionLane& lane) const {
    return metric_.inner_product(left, right, metric_scratch, lane);
  }

  Real local_inner_product_prepared_(const MultiFab& left, const MultiFab& right) const {
    return metric_.local_inner_product(left, right);
  }

  void local_robust_inner_product_payload_prepared_(const MultiFab& left, const MultiFab& right,
                                                    std::span<double> payload) const {
    metric_.local_robust_inner_product_payload(left, right, payload);
  }

  Real inner_product_from_global_robust_payload_prepared_(std::span<const double> payload) const {
    return metric_.inner_product_from_global_robust_payload(payload);
  }

  Real residual_norm_prepared_(const MultiFab& value, std::span<double> metric_scratch,
                               const ExecutionLane& lane) const {
    return metric_.norm(value, metric_scratch, lane);
  }

  PreparedAffineOperatorProvider operator_provider_;
  ExecutionLane preparation_lane_;
  PreparedAffineOperatorSession operator_session_{};
  PreparedLinearPreconditioner preconditioner_;
  LinearOperatorProperties properties_{};
  KrylovFootprint footprint_{};
  PreparedNullspacePolicy nullspace_policy_;
  OperatorSnapshotProbe snapshot_probe_;
  PreparedResourceFn freeze_resources_;
  std::optional<OperatorFingerprint> authenticated_program_authority_;
  PreparedVectorDistribution vector_distribution_ = PreparedVectorDistribution::Distributed;
  PreparedVectorMetric metric_;
  MultiFab zero_;
  MultiFab constant_;
  OperatorFingerprint layout_{};
  OperatorFingerprint metric_fingerprint_{};
  OperatorFingerprint operator_provider_fingerprint_{};
  bool vector_distribution_layout_valid_ = true;
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
  std::optional<NullspaceCertificateIdentity> nullspace_certificate_identity_cache_{};
  mutable std::atomic<std::size_t> active_solve_reservations_{0};
};

namespace detail {

/// Private, allocation-free access used only after solve_prepared_affine has authenticated the
/// caller-owned iterate/RHS and KrylovWorkspace against the prepared problem. Public direct calls on
/// PreparedAffineLinearProblem retain their complete defensive layout and exact replica-value
/// validation.
struct PreparedProblemAccess {
  static const ExecutionLane& preparation_lane(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.preparation_lane_;
  }

  static bool try_reserve_solve(const PreparedAffineLinearProblem& problem) noexcept {
    if (problem.operator_provider_.concurrency() == PreparedOperatorConcurrency::Exclusive) {
      std::size_t expected = 0;
      return problem.active_solve_reservations_.compare_exchange_strong(
          expected, 1, std::memory_order_acq_rel, std::memory_order_acquire);
    }
    std::size_t current = problem.active_solve_reservations_.load(std::memory_order_acquire);
    while (current != std::numeric_limits<std::size_t>::max()) {
      if (problem.active_solve_reservations_.compare_exchange_weak(
              current, current + 1, std::memory_order_acq_rel, std::memory_order_acquire))
        return true;
    }
    return false;
  }

  static void release_solve(const PreparedAffineLinearProblem& problem) noexcept {
    problem.active_solve_reservations_.fetch_sub(1, std::memory_order_acq_rel);
  }

  static long append_collective_state(const PreparedAffineLinearProblem& problem,
                                      KrylovCollectivePayload& payload) noexcept {
    problem.append_collective_contract_(payload);

    OperatorEvaluationSnapshot observed{};
    bool probe_failed = false;
    try {
      observed = problem.snapshot_probe_();
    } catch (...) {
      probe_failed = true;
    }
    payload.append(static_cast<std::uint8_t>(probe_failed));
    payload.append(observed);
    if (!problem.snapshot_)
      return 3;
    if (probe_failed)
      return 2;
    return observed == *problem.snapshot_ ? 0 : 1;
  }

  static const std::optional<OperatorEvaluationSnapshot>& stored_snapshot(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.snapshot_;
  }

  static void require_current(const PreparedAffineLinearProblem& problem,
                              const ExecutionLane& lane) {
    problem.require_current(lane);
  }

  static PreparedVectorDistribution vector_distribution(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.vector_distribution_;
  }

  static const PreparedVectorMetric& metric(const PreparedAffineLinearProblem& problem) noexcept {
    return problem.metric_;
  }

  static const OperatorFingerprint& metric_fingerprint(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.metric_fingerprint_;
  }

  static std::size_t nullspace_basis_count(const PreparedAffineLinearProblem& problem) noexcept {
    return problem.nullspace_policy_.basis_vectors_.size();
  }

  static PreparedProviderSourceIdentity operator_source_identity(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.operator_provider_.source_identity_();
  }

  static PreparedProviderSourceIdentity preconditioner_source_identity(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.preconditioner_.provider_.source_identity_();
  }

  static PreparedLinearPreconditionerSession make_preconditioner_session(
      const PreparedAffineLinearProblem& problem, const ExecutionLane& lane) {
    if (problem.preconditioner_.is_identity())
      return {};
    return problem.preconditioner_.provider_.make_session(lane);
  }

  static PreparedAffineOperatorSession make_operator_session(
      const PreparedAffineLinearProblem& problem, const ExecutionLane& lane) {
    return problem.operator_provider_.make_session(lane);
  }

  static bool matches_vector_space(const PreparedAffineLinearProblem& problem,
                                   const MultiFab& value) noexcept {
    return PureFieldAlgebra::same_vector_space(problem.zero_, value);
  }

  static PreparedEquationReference prepare_compatibility_rhs(
      const PreparedAffineLinearProblem& problem, MultiFab& out, const MultiFab& rhs,
      std::span<double> metric_scratch, const ExecutionLane& lane) {
    return problem.prepare_compatibility_rhs_prepared_(out, rhs, metric_scratch, lane);
  }
  static void require_nullspace_compatible(const PreparedAffineLinearProblem& problem,
                                           const MultiFab& normalized_rhs,
                                           std::span<double> metric_scratch,
                                           const ExecutionLane& lane) {
    problem.nullspace_policy_.require_compatible(normalized_rhs, problem.metric_, metric_scratch,
                                                 lane);
  }
  static void apply_nullspace_gauge(const PreparedAffineLinearProblem& problem, MultiFab& iterate,
                                    std::span<double> gauge_scratch,
                                    std::span<double> metric_scratch, const ExecutionLane& lane) {
    problem.nullspace_policy_.apply_gauge(iterate, problem.metric_, gauge_scratch, metric_scratch,
                                          lane);
  }
  static PreparedApplyStatus apply_linear(const PreparedAffineLinearProblem& problem,
                                          PreparedAffineOperatorSession& session, MultiFab& out,
                                          const MultiFab& direction, Real equation_scale) {
    return problem.apply_workspace_linear_prepared_(session, out, direction, equation_scale);
  }
  static PreparedApplyStatus true_residual_physical(const PreparedAffineLinearProblem& problem,
                                                    PreparedAffineOperatorSession& session,
                                                    MultiFab& out, const MultiFab& rhs,
                                                    const MultiFab& iterate) {
    return problem.workspace_true_residual_prepared_(session, out, rhs, iterate);
  }
  static PreparedApplyStatus apply_preconditioner(const PreparedAffineLinearProblem& problem,
                                                  PreparedLinearPreconditionerSession& session,
                                                  const MultiFab& affine_constant, MultiFab& out,
                                                  const MultiFab& in) {
    return problem.apply_workspace_preconditioner_prepared_(session, affine_constant, out, in);
  }
  static Real inner_product(const PreparedAffineLinearProblem& problem, const MultiFab& left,
                            const MultiFab& right, std::span<double> metric_scratch,
                            const ExecutionLane& lane) {
    return problem.inner_product_prepared_(left, right, metric_scratch, lane);
  }
  static Real local_inner_product(const PreparedAffineLinearProblem& problem, const MultiFab& left,
                                  const MultiFab& right) {
    return problem.local_inner_product_prepared_(left, right);
  }
  static void local_robust_inner_product_payload(const PreparedAffineLinearProblem& problem,
                                                 const MultiFab& left, const MultiFab& right,
                                                 std::span<double> payload) {
    problem.local_robust_inner_product_payload_prepared_(left, right, payload);
  }
  static Real inner_product_from_global_robust_payload(const PreparedAffineLinearProblem& problem,
                                                       std::span<const double> payload) {
    return problem.inner_product_from_global_robust_payload_prepared_(payload);
  }
  static Real residual_norm(const PreparedAffineLinearProblem& problem, const MultiFab& value,
                            std::span<double> metric_scratch, const ExecutionLane& lane) {
    return problem.residual_norm_prepared_(value, metric_scratch, lane);
  }
};

}  // namespace detail

}  // namespace pops
