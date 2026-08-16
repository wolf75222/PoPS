#pragma once

/// @file
/// @brief Typed, fallible variable inversion with a reusable backend workspace.

#include <pops/core/foundation/types.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/identity/prepared_provider.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <concepts>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace pops {

/// Exact allocation requirement of one prepared inversion algorithm.
struct InversionWorkspaceBudget {
  std::size_t bytes = 0;
  std::size_t alignment = alignof(std::max_align_t);

  [[nodiscard]] constexpr bool valid() const noexcept {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
  }
};

/// Non-owning device-clean view passed to the inversion source on every attempt.
struct InversionWorkspaceView {
  std::byte* data = nullptr;
  std::size_t size = 0;

  template <class Value>
  POPS_HD Value* as(std::size_t offset = 0) const noexcept {
    if (offset > size || sizeof(Value) > size - offset)
      return nullptr;
    auto* pointer = data + offset;
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(Value) != 0)
      return nullptr;
    return reinterpret_cast<Value*>(pointer);
  }
};

/// Backend allocation owned once by a prepared inversion and reused by every attempt.
///
/// Under Kokkos this is memory in the default execution space.  The CPU specialization is the
/// execution space itself, not a staging mirror.  No allocation, resize, or host copy occurs in
/// `view()` or in the prepared provider's retry path.
class InversionDeviceWorkspace final {
 public:
  explicit InversionDeviceWorkspace(InversionWorkspaceBudget budget) : budget_(budget) {
    if (!budget_.valid())
      throw std::invalid_argument("inversion workspace alignment must be a power of two");
    if (budget_.bytes > std::numeric_limits<std::size_t>::max() - (budget_.alignment - 1))
      throw std::length_error("inversion workspace budget overflows addressable storage");
    const std::size_t allocation_bytes =
        budget_.bytes + (budget_.bytes == 0 ? 0 : budget_.alignment - 1);
#if defined(POPS_HAS_KOKKOS)
    detail::ensure_kokkos_initialized();
    allocation_ = Allocation("pops.variable_inversion.workspace", allocation_bytes);
    base_ = aligned_(reinterpret_cast<std::byte*>(allocation_.data()), budget_.alignment);
#else
    if (allocation_bytes != 0) {
      allocation_.reset(new std::byte[allocation_bytes]);
      base_ = aligned_(allocation_.get(), budget_.alignment);
    }
#endif
  }

  InversionDeviceWorkspace(const InversionDeviceWorkspace&) = delete;
  InversionDeviceWorkspace& operator=(const InversionDeviceWorkspace&) = delete;
  InversionDeviceWorkspace(InversionDeviceWorkspace&&) noexcept = default;
  InversionDeviceWorkspace& operator=(InversionDeviceWorkspace&&) noexcept = default;

  [[nodiscard]] InversionWorkspaceView view() noexcept { return {base_, budget_.bytes}; }
  [[nodiscard]] InversionWorkspaceBudget budget() const noexcept { return budget_; }
  [[nodiscard]] const void* allocation_identity() const noexcept { return base_; }

 private:
  static std::byte* aligned_(std::byte* pointer, std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return nullptr;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto aligned = (address + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    return reinterpret_cast<std::byte*>(aligned);
  }

#if defined(POPS_HAS_KOKKOS)
  // Attempts are host calls. Keep the reusable allocation Kokkos-owned while ensuring that a
  // CUDA/HIP default execution space never hands host recovery an inaccessible device pointer.
  using Allocation = Kokkos::View<unsigned char*, Kokkos::HostSpace>;
  Allocation allocation_{};
#else
  std::unique_ptr<std::byte[]> allocation_{};
#endif
  InversionWorkspaceBudget budget_{};
  std::byte* base_ = nullptr;
};

/// Complete type and identity description of one variable inversion problem.
template <int Dim, class State, class ProviderInputs, class PrimitiveCandidate, class Failure>
class VariableInversionProblem final {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "variable inversion dimension must be 1, 2, or 3");
  static_assert(std::is_enum_v<Failure>, "variable inversion failures must be a typed enum");

  static constexpr int dimension = Dim;
  using state_type = State;
  using provider_inputs_type = ProviderInputs;
  using candidate_type = PrimitiveCandidate;
  using failure_type = Failure;

  VariableInversionProblem(std::string state_identity, std::string provider_inputs_identity,
                           std::string candidate_identity, std::string failure_identity,
                           InversionWorkspaceBudget workspace)
      : state_identity_(nonempty_(std::move(state_identity), "state")),
        provider_inputs_identity_(
            nonempty_(std::move(provider_inputs_identity), "provider inputs")),
        candidate_identity_(nonempty_(std::move(candidate_identity), "candidate")),
        failure_identity_(nonempty_(std::move(failure_identity), "failure")),
        workspace_(workspace) {
    if (!workspace_.valid())
      throw std::invalid_argument("inversion workspace alignment must be a power of two");
  }

  [[nodiscard]] InversionWorkspaceBudget workspace_budget() const noexcept { return workspace_; }
  [[nodiscard]] std::string_view state_identity() const noexcept { return state_identity_; }
  [[nodiscard]] std::string_view provider_inputs_identity() const noexcept {
    return provider_inputs_identity_;
  }
  [[nodiscard]] std::string_view candidate_identity() const noexcept { return candidate_identity_; }
  [[nodiscard]] std::string_view failure_identity() const noexcept { return failure_identity_; }

  void serialize_exact(ExactContractBuilder& contract) const {
    contract.text("pops.variable-inversion-problem")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(state_identity_)
        .text(provider_inputs_identity_)
        .text(candidate_identity_)
        .text(failure_identity_)
        .scalar(static_cast<std::uint64_t>(workspace_.bytes))
        .scalar(static_cast<std::uint64_t>(workspace_.alignment));
  }

  [[nodiscard]] std::string exact_contract() const {
    ExactContractBuilder contract;
    serialize_exact(contract);
    return std::move(contract).release();
  }

 private:
  static std::string nonempty_(std::string value, std::string_view role) {
    if (value.empty())
      throw std::invalid_argument("variable inversion " + std::string(role) +
                                  " identity must not be empty");
    return value;
  }

  std::string state_identity_;
  std::string provider_inputs_identity_;
  std::string candidate_identity_;
  std::string failure_identity_;
  InversionWorkspaceBudget workspace_{};
};

/// Device-clean result returned by a concrete closed-form or iterative source.
template <class Candidate, class Failure>
struct InversionResult {
  static_assert(std::is_trivially_copyable_v<Candidate>,
                "device inversion candidates must be trivially copyable");
  static_assert(std::is_enum_v<Failure>, "device inversion failure must be a typed enum");

  bool candidate_available = false;
  Candidate candidate{};
  Failure failure{};

  POPS_HD static InversionResult success(Candidate value) { return {true, value, Failure{}}; }
  POPS_HD static InversionResult fail(Failure reason) { return {false, Candidate{}, reason}; }
};

template <class Candidate, class Failure>
class ConsumedInversion final {
 public:
  explicit ConsumedInversion(Candidate candidate) : value_(std::move(candidate)) {}
  explicit ConsumedInversion(Failure failure) : value_(failure) {}

  [[nodiscard]] bool succeeded() const noexcept {
    return std::holds_alternative<Candidate>(value_);
  }
  [[nodiscard]] const Candidate& candidate() const { return std::get<Candidate>(value_); }
  [[nodiscard]] Failure failure() const { return std::get<Failure>(value_); }

 private:
  std::variant<Candidate, Failure> value_;
};

/// One-shot detached outcome.  Nothing is published by construction or consumption.
///
/// The input is const and the candidate lives only in this object.  A failed attempt contains no
/// readable candidate.  Consumers must explicitly consume the result once before deciding whether
/// and where to publish a successful value.
template <class Candidate, class Failure>
class [[nodiscard]] InversionOutcome final {
 public:
  explicit InversionOutcome(InversionResult<Candidate, Failure> result)
      : result_(std::move(result)) {}

  InversionOutcome(const InversionOutcome&) = delete;
  InversionOutcome& operator=(const InversionOutcome&) = delete;
  InversionOutcome(InversionOutcome&& other) noexcept
      : result_(std::move(other.result_)), consumed_(std::exchange(other.consumed_, true)) {}
  InversionOutcome& operator=(InversionOutcome&&) noexcept = delete;

  [[nodiscard]] bool succeeded() const {
    require_open_();
    return result_.candidate_available;
  }
  [[nodiscard]] Failure failure() const {
    require_open_();
    if (result_.candidate_available)
      throw std::logic_error("successful inversion outcome has no failure");
    return result_.failure;
  }

  ConsumedInversion<Candidate, Failure> consume() {
    require_open_();
    consumed_ = true;
    if (result_.candidate_available)
      return ConsumedInversion<Candidate, Failure>(std::move(result_.candidate));
    return ConsumedInversion<Candidate, Failure>(result_.failure);
  }

 private:
  void require_open_() const {
    if (consumed_)
      throw std::logic_error("variable inversion outcome has already been consumed");
  }

  InversionResult<Candidate, Failure> result_;
  bool consumed_ = false;
};

namespace inversion_detail {

template <class Source, class Problem>
concept SourceFor =
    std::copy_constructible<Source> &&
    requires(const Source& source, const typename Problem::state_type& state,
             const typename Problem::provider_inputs_type& inputs, InversionWorkspaceView workspace,
             ExactContractBuilder& contract) {
      { Source::provider_identity() } noexcept -> std::same_as<PreparedProviderIdentity>;
      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
      {
        source(state, inputs, workspace)
      } -> std::same_as<
          InversionResult<typename Problem::candidate_type, typename Problem::failure_type>>;
    };

}  // namespace inversion_detail

/// Prepared, authenticated inversion source with one stable backend allocation.
template <class Problem, class Source>
  requires inversion_detail::SourceFor<Source, Problem>
class PreparedVariableInversion final {
 public:
  using Candidate = typename Problem::candidate_type;
  using Failure = typename Problem::failure_type;

  PreparedVariableInversion(Problem problem, Source source)
      : problem_(std::move(problem)),
        source_(std::move(source)),
        workspace_(problem_.workspace_budget()) {
    if constexpr (requires { Source::dimension; })
      static_assert(Source::dimension == Problem::dimension,
                    "variable inversion source dimension differs from its problem");
    const PreparedProviderIdentity identity = Source::provider_identity();
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument("variable inversion provider identity is incomplete");
    implementation_ = std::string(identity.name);
    implementation_version_ = identity.version;
    ExactContractBuilder parameters;
    source_.serialize_exact_parameters(parameters);
    ExactContractBuilder problem_contract;
    problem_.serialize_exact(problem_contract);
    ExactContractBuilder contract;
    contract.text("pops.prepared-variable-inversion")
        .scalar(std::uint32_t{1})
        .text(identity.name)
        .scalar(identity.version)
        .bytes(problem_contract.view())
        .bytes(parameters.view());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] InversionOutcome<Candidate, Failure> attempt(
      const typename Problem::state_type& state,
      const typename Problem::provider_inputs_type& inputs) {
    return InversionOutcome<Candidate, Failure>(source_(state, inputs, workspace_.view()));
  }

  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }
  [[nodiscard]] std::string_view implementation() const noexcept { return implementation_; }
  [[nodiscard]] std::uint64_t implementation_version() const noexcept {
    return implementation_version_;
  }
  [[nodiscard]] const Problem& problem() const noexcept { return problem_; }
  [[nodiscard]] const void* workspace_allocation_identity() const noexcept {
    return workspace_.allocation_identity();
  }
  [[nodiscard]] InversionWorkspaceBudget workspace_budget() const noexcept {
    return workspace_.budget();
  }

 private:
  Problem problem_;
  Source source_;
  InversionDeviceWorkspace workspace_;
  std::string implementation_;
  std::uint64_t implementation_version_ = 0;
  std::string collective_contract_;
};

}  // namespace pops
