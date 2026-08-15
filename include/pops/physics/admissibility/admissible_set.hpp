#pragma once

/// @file
/// @brief Model-declared, device-clean admissibility constraints without physical-name policy.

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pops {

enum class AdmissibilityConstraintKind : std::uint8_t {
  kFinite = 1,
  kPositive = 2,
  kRealizability = 3,
  kCustomInequality = 4,
};

struct AdmissibilityResult {
  bool accepted = true;
  std::uint32_t constraint_index = 0;
  AdmissibilityConstraintKind kind = AdmissibilityConstraintKind::kFinite;
  std::uint32_t diagnostic_code = 0;

  POPS_HD static constexpr AdmissibilityResult accept() noexcept { return {}; }
  POPS_HD static constexpr AdmissibilityResult reject(std::uint32_t index,
                                                      AdmissibilityConstraintKind rejected_kind,
                                                      std::uint32_t code) noexcept {
    return {false, index, rejected_kind, code};
  }
};

namespace admissibility_detail {

POPS_HD inline bool finite(Real value) noexcept {
  constexpr Real infinity = std::numeric_limits<Real>::infinity();
  return value == value && value != infinity && value != -infinity;
}

template <class Constraint, class Candidate>
concept ConstraintFor = requires(const Constraint& constraint, const Candidate& candidate,
                                 ExactContractBuilder& contract) {
  { Constraint::kind } -> std::convertible_to<AdmissibilityConstraintKind>;
  { constraint.diagnostic_code() } -> std::same_as<std::uint32_t>;
  { constraint(candidate) } -> std::same_as<bool>;
  { constraint.serialize_exact_parameters(contract) } -> std::same_as<void>;
};

template <class Predicate, class Candidate>
concept PredicateFor = std::copy_constructible<Predicate> &&
                       requires(const Predicate& predicate, const Candidate& candidate,
                                ExactContractBuilder& contract) {
                         {
                           Predicate::provider_identity()
                         } noexcept -> std::same_as<PreparedProviderIdentity>;
                         { predicate.serialize_exact_parameters(contract) } -> std::same_as<void>;
                         { predicate(candidate) } -> std::same_as<bool>;
                       };

template <std::size_t Index, class Constraint>
struct ConstraintSlot {
  Constraint value;

  POPS_HD const Constraint& get() const noexcept { return value; }
};

template <class Indices, class... Constraints>
struct ConstraintPack;

template <std::size_t... Indices, class... Constraints>
struct ConstraintPack<std::index_sequence<Indices...>, Constraints...>
    : ConstraintSlot<Indices, Constraints>... {
  explicit ConstraintPack(Constraints... constraints)
      : ConstraintSlot<Indices, Constraints>{std::move(constraints)}... {}

  template <std::size_t Index>
  POPS_HD const auto& get() const noexcept {
    using Constraint = std::tuple_element_t<Index, std::tuple<Constraints...>>;
    return static_cast<const ConstraintSlot<Index, Constraint>&>(*this).get();
  }
};

}  // namespace admissibility_detail

/// Require a compile-time component interval to contain only finite values.
template <int First, int Count>
class FiniteComponents final {
 public:
  static_assert(First >= 0 && Count > 0, "finite component interval must be non-empty");
  static constexpr AdmissibilityConstraintKind kind = AdmissibilityConstraintKind::kFinite;

  explicit constexpr FiniteComponents(std::uint32_t diagnostic_code) noexcept
      : diagnostic_code_(diagnostic_code) {}

  template <class Candidate>
  POPS_HD bool operator()(const Candidate& candidate) const noexcept {
    static_assert(First + Count <= Candidate::size(),
                  "finite constraint interval is outside the candidate");
    for (int component = First; component < First + Count; ++component)
      if (!admissibility_detail::finite(candidate[component]))
        return false;
    return true;
  }

  [[nodiscard]] constexpr std::uint32_t diagnostic_code() const noexcept {
    return diagnostic_code_;
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("pops.admissibility.finite-components")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{First})
        .scalar(std::int32_t{Count})
        .scalar(diagnostic_code_);
  }

 private:
  std::uint32_t diagnostic_code_;
};

/// Require one component to be strictly greater than a model-owned lower bound.
template <int Component>
class PositiveComponent final {
 public:
  static_assert(Component >= 0, "positive component index cannot be negative");
  static constexpr AdmissibilityConstraintKind kind = AdmissibilityConstraintKind::kPositive;

  PositiveComponent(Real lower_bound, std::uint32_t diagnostic_code)
      : lower_bound_(lower_bound), diagnostic_code_(diagnostic_code) {
    if (!admissibility_detail::finite(lower_bound_))
      throw std::invalid_argument("positive constraint lower bound must be finite");
  }

  template <class Candidate>
  POPS_HD bool operator()(const Candidate& candidate) const noexcept {
    static_assert(Component < Candidate::size(),
                  "positive constraint component is outside the candidate");
    const Real value = candidate[Component];
    return admissibility_detail::finite(value) && value > lower_bound_;
  }

  [[nodiscard]] constexpr std::uint32_t diagnostic_code() const noexcept {
    return diagnostic_code_;
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("pops.admissibility.positive-component")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Component})
        .scalar(lower_bound_)
        .scalar(diagnostic_code_);
  }

 private:
  Real lower_bound_;
  std::uint32_t diagnostic_code_;
};

/// Provider-owned inequality classified as realizability or a custom model constraint.
template <AdmissibilityConstraintKind Kind, class Predicate>
  requires(Kind == AdmissibilityConstraintKind::kRealizability ||
           Kind == AdmissibilityConstraintKind::kCustomInequality)
class ModelInequality final {
 public:
  static constexpr AdmissibilityConstraintKind kind = Kind;

  ModelInequality(Predicate predicate, std::uint32_t diagnostic_code)
      : predicate_(std::move(predicate)), diagnostic_code_(diagnostic_code) {}

  template <class Candidate>
    requires admissibility_detail::PredicateFor<Predicate, Candidate>
  POPS_HD bool operator()(const Candidate& candidate) const {
    return predicate_(candidate);
  }

  [[nodiscard]] constexpr std::uint32_t diagnostic_code() const noexcept {
    return diagnostic_code_;
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    const PreparedProviderIdentity identity = Predicate::provider_identity();
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument("model inequality provider identity is incomplete");
    ExactContractBuilder parameters;
    predicate_.serialize_exact_parameters(parameters);
    contract
        .text(Kind == AdmissibilityConstraintKind::kRealizability
                  ? "pops.admissibility.realizability"
                  : "pops.admissibility.custom-inequality")
        .scalar(std::uint32_t{1})
        .text(identity.name)
        .scalar(identity.version)
        .scalar(diagnostic_code_)
        .bytes(parameters.view());
  }

 private:
  Predicate predicate_;
  std::uint32_t diagnostic_code_;
};

template <class Predicate>
using RealizabilityConstraint =
    ModelInequality<AdmissibilityConstraintKind::kRealizability, Predicate>;

template <class Predicate>
using CustomInequality = ModelInequality<AdmissibilityConstraintKind::kCustomInequality, Predicate>;

/// Ordered model authority for every candidate admissibility decision.
///
/// Diagnostic codes must be non-zero and unique within the set.  Evaluation is allocation-free and
/// reports the first failing declaration in exact serialized order.
template <class... Constraints>
class AdmissibleSet final {
 public:
  static_assert(sizeof...(Constraints) > 0, "an admissible set must declare a constraint");
  static_assert((std::is_trivially_copyable_v<Constraints> && ...),
                "device admissibility constraints must be trivially copyable");

  explicit AdmissibleSet(Constraints... constraints) : constraints_(std::move(constraints)...) {
    validate_codes_<0>();
  }

  template <class Candidate>
    requires(admissibility_detail::ConstraintFor<Constraints, Candidate> && ...)
  POPS_HD AdmissibilityResult evaluate(const Candidate& candidate) const {
    return evaluate_<0>(candidate);
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    contract.text("pops.admissible-set")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint32_t>(sizeof...(Constraints)));
    serialize_<0>(contract);
  }

  [[nodiscard]] std::string exact_contract() const {
    ExactContractBuilder contract;
    serialize_exact(contract);
    return std::move(contract).release();
  }

 private:
  template <std::size_t Index>
  void validate_codes_() const {
    if constexpr (Index < sizeof...(Constraints)) {
      const std::uint32_t code = constraints_.template get<Index>().diagnostic_code();
      if (code == 0)
        throw std::invalid_argument("admissibility diagnostic codes must be non-zero");
      validate_code_against_<Index, Index + 1>(code);
      validate_codes_<Index + 1>();
    }
  }

  template <std::size_t Owner, std::size_t Other>
  void validate_code_against_(std::uint32_t code) const {
    if constexpr (Other < sizeof...(Constraints)) {
      if (code == constraints_.template get<Other>().diagnostic_code())
        throw std::invalid_argument("admissibility diagnostic code collision");
      validate_code_against_<Owner, Other + 1>(code);
    }
  }

  template <std::size_t Index, class Candidate>
  POPS_HD AdmissibilityResult evaluate_(const Candidate& candidate) const {
    const auto& constraint = constraints_.template get<Index>();
    if (!constraint(candidate))
      return AdmissibilityResult::reject(static_cast<std::uint32_t>(Index),
                                         std::remove_cvref_t<decltype(constraint)>::kind,
                                         constraint.diagnostic_code());
    if constexpr (Index + 1 < sizeof...(Constraints))
      return evaluate_<Index + 1>(candidate);
    return AdmissibilityResult::accept();
  }

  template <std::size_t Index>
  void serialize_(ExactContractBuilder& contract) const {
    if constexpr (Index < sizeof...(Constraints)) {
      ExactContractBuilder constraint;
      constraints_.template get<Index>().serialize_exact_parameters(constraint);
      contract.bytes(constraint.view());
      serialize_<Index + 1>(contract);
    }
  }

  admissibility_detail::ConstraintPack<std::index_sequence_for<Constraints...>, Constraints...>
      constraints_;
};

template <class... Constraints>
AdmissibleSet(Constraints...) -> AdmissibleSet<Constraints...>;

}  // namespace pops
