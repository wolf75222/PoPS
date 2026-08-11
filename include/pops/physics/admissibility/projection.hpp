#pragma once

/// @file
/// @brief Explicit authenticated projection producing only detached candidates.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/physics/admissibility/admissible_set.hpp>
#include <pops/physics/admissibility/enforcement_schedule.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {

template <class Candidate>
struct ProjectionResult {
  Candidate candidate{};
  bool changed = false;
};

/// Observable result of an explicit projection request.
template <class Candidate>
class [[nodiscard]] ProjectedCandidate final {
 public:
  ProjectedCandidate(Candidate candidate, bool changed)
      : candidate_(std::move(candidate)), changed_(changed) {}

  [[nodiscard]] bool changed() const noexcept { return changed_; }
  [[nodiscard]] Candidate consume() && { return std::move(candidate_); }

 private:
  Candidate candidate_;
  bool changed_ = false;
};

namespace projection_detail {

template <class Source, class Candidate, class Inputs>
concept SourceFor = std::copy_constructible<Source> &&
                    requires(const Source& source, const Candidate& candidate, const Inputs& inputs,
                             ExactContractBuilder& contract) {
                      {
                        Source::provider_identity()
                      } noexcept -> std::same_as<PreparedProviderIdentity>;
                      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
                      { source(candidate, inputs) } -> std::same_as<ProjectionResult<Candidate>>;
                    };

}  // namespace projection_detail

/// Prepared projection authority. Calling `project` is the sole projection act and is observable
/// through `ProjectedCandidate::changed`; the input is const and publication remains external.
template <int Dim, class Candidate, class Inputs, class Source>
  requires projection_detail::SourceFor<Source, Candidate, Inputs>
class ProjectionProvider final {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "projection provider dimension must be 1, 2, or 3");
  static constexpr int dimension = Dim;

  ProjectionProvider(std::string candidate_identity, std::string inputs_identity, Source source)
      : candidate_identity_(std::move(candidate_identity)),
        inputs_identity_(std::move(inputs_identity)),
        source_(std::move(source)) {
    if constexpr (requires { Source::dimension; })
      static_assert(Source::dimension == Dim,
                    "projection source dimension differs from its provider");
    if (candidate_identity_.empty() || inputs_identity_.empty())
      throw std::invalid_argument("projection candidate/input identities must not be empty");
    const PreparedProviderIdentity identity = Source::provider_identity();
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument("projection provider identity is incomplete");
    implementation_ = std::string(identity.name);
    implementation_version_ = identity.version;
    ExactContractBuilder parameters;
    source_.serialize_exact_parameters(parameters);
    ExactContractBuilder contract;
    contract.text("pops.projection-provider")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(candidate_identity_)
        .text(inputs_identity_)
        .text(identity.name)
        .scalar(identity.version)
        .bytes(parameters.view());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] ProjectedCandidate<Candidate> project(const Candidate& candidate,
                                                      const Inputs& inputs) const {
    ProjectionResult<Candidate> projected = source_(candidate, inputs);
    return ProjectedCandidate<Candidate>(std::move(projected.candidate), projected.changed);
  }

  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }
  [[nodiscard]] std::string_view implementation() const noexcept { return implementation_; }
  [[nodiscard]] std::uint64_t implementation_version() const noexcept {
    return implementation_version_;
  }
  [[nodiscard]] std::string_view candidate_identity() const noexcept { return candidate_identity_; }
  [[nodiscard]] std::string_view inputs_identity() const noexcept { return inputs_identity_; }

 private:
  std::string candidate_identity_;
  std::string inputs_identity_;
  Source source_;
  std::string implementation_;
  std::uint64_t implementation_version_ = 0;
  std::string collective_contract_;
};

/// Detached outcome of one schedule-controlled admissibility enforcement.
template <class Candidate>
struct EnforcementResult {
  Candidate candidate{};
  AdmissibilityResult admissibility = AdmissibilityResult::accept();
  bool checked = false;
  bool projection_attempted = false;
  bool projection_changed = false;
};

/// Exact authority joining an ordered admissible set, its schedule, and an optional real
/// projection provider. Absence is represented by the type ``std::nullptr_t``; it is not an
/// identity projection. A schedule requesting projection without a provider is rejected while the
/// authority is prepared, before any candidate can be evaluated or published.
template <class Candidate, class Inputs, class Admissibility, class Projection = std::nullptr_t>
class PreparedAdmissibilityEnforcement final {
 public:
  static constexpr bool has_projection = !std::same_as<Projection, std::nullptr_t>;

  PreparedAdmissibilityEnforcement(Admissibility admissibility, EnforcementSchedule schedule)
    requires(!has_projection)
      : admissibility_(std::move(admissibility)), schedule_(std::move(schedule)) {
    validate_schedule_();
    build_contract_();
  }

  PreparedAdmissibilityEnforcement(Admissibility admissibility, EnforcementSchedule schedule,
                                   Projection projection)
    requires(has_projection)
      : admissibility_(std::move(admissibility)),
        schedule_(std::move(schedule)),
        projection_(std::move(projection)) {
    validate_schedule_();
    build_contract_();
  }

  [[nodiscard]] EnforcementResult<Candidate> enforce(const Candidate& candidate,
                                                     const Inputs& inputs,
                                                     EnforcementPhase phase) const {
    EnforcementResult<Candidate> result;
    result.candidate = candidate;
    const EnforcementRule rule = schedule_.at(phase);
    if (!rule.check)
      return result;

    result.checked = true;
    result.admissibility = admissibility_.evaluate(result.candidate);
    if (!result.admissibility.accepted && rule.project_if_invalid) {
      if constexpr (has_projection) {
        auto projected = projection_.project(result.candidate, inputs);
        result.projection_attempted = true;
        result.projection_changed = projected.changed();
        result.candidate = std::move(projected).consume();
        result.admissibility = admissibility_.evaluate(result.candidate);
      } else {
        throw std::logic_error(
            "admissibility enforcement escaped preparation without a projection provider");
      }
    }
    return result;
  }

  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

 private:
  void validate_schedule_() const {
    if constexpr (!has_projection)
      for (std::size_t phase = 0; phase < EnforcementSchedule::phase_count; ++phase)
        if (schedule_.at(static_cast<EnforcementPhase>(phase)).project_if_invalid)
          throw std::invalid_argument(
              "admissibility enforcement schedule requests an absent projection provider");
  }

  void build_contract_() {
    ExactContractBuilder admissibility;
    admissibility_.serialize_exact(admissibility);
    ExactContractBuilder schedule;
    schedule_.serialize_exact(schedule);
    ExactContractBuilder contract;
    contract.text("pops.prepared-admissibility-enforcement")
        .scalar(std::uint32_t{1})
        .bytes(admissibility.view())
        .bytes(schedule.view())
        .presence(has_projection);
    if constexpr (has_projection)
      contract.bytes(projection_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  Admissibility admissibility_;
  EnforcementSchedule schedule_;
  [[no_unique_address]] Projection projection_{};
  std::string collective_contract_;
};

}  // namespace pops
