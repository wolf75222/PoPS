#pragma once

/// @file
/// @brief Explicit authenticated projection producing only detached candidates.

#include <pops/core/identity/prepared_provider.hpp>

#include <concepts>
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

}  // namespace pops
