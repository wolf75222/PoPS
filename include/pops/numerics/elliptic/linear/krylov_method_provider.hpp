#pragma once

/// @file
/// @brief Exact, extensible provider protocol for prepared Krylov methods.

#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/numerics/elliptic/linear/prepared_affine_problem.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <map>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace pops {

class KrylovWorkspace;
struct KrylovControls;
class PreparedKrylovSolveContext;

inline int max_krylov_batched_basis_extent(std::size_t robust_payload_width) noexcept {
  if (robust_payload_width == 0 ||
      robust_payload_width > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
    return 0;
  return (std::numeric_limits<int>::max() - 1) /
             (2 * static_cast<int>(robust_payload_width)) -
         1;
}

/// Universal stopping controls offered to every prepared iterative method. Method-specific
/// controls never grow this structure: they travel in the provider-owned PreparedProviderOptions
/// authenticated by PreparedKrylovMethod.
struct KrylovMethodControls {
  Real rel_tol = Real(1e-8);
  Real abs_tol = Real(0);
  int max_iterations = 1;
};

/// Exact vector-space facts needed before persistent method storage is allocated.
struct KrylovWorkspaceRequest {
  KrylovFootprint footprint{};
  PreparedVectorDistribution distribution = PreparedVectorDistribution::Distributed;
  std::size_t robust_payload_width = 0;
};

/// Generic persistent storage requested by one method provider.  Providers receive stable indexed
/// pools rather than owning hidden allocations, so all storage is materialized before iteration.
struct KrylovWorkspaceRequirements {
  std::size_t field_count = 0;
  std::size_t real_count = 0;
  std::size_t scaled_scalar_count = 0;
  std::size_t collective_value_count = 0;
  /// Maximum number of scientific values reduced by one provider call. The vector-distribution
  /// provider turns this capacity into its own opaque scratch requirement.
  std::size_t reduction_value_capacity = 0;
  std::size_t state_word_count = 0;
  std::size_t initial_residual_field = 0;

  [[nodiscard]] bool valid() const noexcept {
    return field_count > 0 && initial_residual_field < field_count;
  }

  friend bool operator==(const KrylovWorkspaceRequirements&,
                         const KrylovWorkspaceRequirements&) = default;
};

/// Authenticated mathematical and storage facts offered to a method at the solve boundary.
struct KrylovMethodProblemFacts {
  LinearOperatorProperties properties{};
  KrylovFootprint footprint{};
  PreparedVectorDistribution distribution = PreparedVectorDistribution::Distributed;
  std::size_t robust_payload_width = 0;
  bool has_nullspace = false;
  bool has_preconditioner = false;
};

/// Allocation-free provider validation result.  Code zero means accepted.  A provider owns both
/// its nonzero codes and static diagnostic text; the common MPI gate never throws rank-locally.
struct KrylovMethodValidation {
  std::uint32_t code = 0;
  std::string_view reason{};

  [[nodiscard]] constexpr bool accepted() const noexcept { return code == 0; }
  static constexpr KrylovMethodValidation accept() noexcept { return {}; }
  static constexpr KrylovMethodValidation reject(std::uint32_t code,
                                                 std::string_view reason) noexcept {
    return {code, reason};
  }
};

/// Native extension protocol for one prepared iterative method.  Builtins and external providers
/// implement this same interface.  The virtual solve call occurs exactly once per solve; iteration
/// remains inside the provider and never pays a name lookup or virtual dispatch in its hot loop.
class PreparedKrylovMethodProvider {
 public:
  virtual ~PreparedKrylovMethodProvider() = default;

  [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t interface_version() const noexcept = 0;
  [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;

  [[nodiscard]] virtual KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept = 0;
  [[nodiscard]] virtual KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts,
      const PreparedProviderOptions& options) const noexcept = 0;
  [[nodiscard]] virtual KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request,
      const PreparedProviderOptions& options) const = 0;
  /// Execute one complete recurrence. If the provider enters MPI it must execute the same complete
  /// collective trace on every rank before returning or throwing; the common boundary can make a
  /// post-trace failure uniform, but cannot repair an abandoned collective.
  [[nodiscard]] virtual SolveReport solve(
      PreparedKrylovSolveContext& context,
      const PreparedProviderOptions& options) const = 0;
};

/// Immutable prepared handle. Equality is semantic and exact, never pointer- or name-based.
class PreparedKrylovMethod {
 public:
  PreparedKrylovMethod() = default;

  explicit PreparedKrylovMethod(std::shared_ptr<const PreparedKrylovMethodProvider> provider,
                                PreparedProviderOptions options)
      : provider_(std::move(provider)), options_(std::move(options)) {
    if (!provider_)
      throw std::invalid_argument("prepared Krylov method provider must not be null");
    if (provider_->identity().empty() || provider_->interface_version() == 0 ||
        provider_->collective_contract().empty())
      throw std::invalid_argument("prepared Krylov method provider requires exact identities");
    ExactContractBuilder exact;
    exact.text("pops.prepared-krylov-method")
        .scalar(std::uint32_t{1})
        .text(provider_->identity())
        .scalar(provider_->interface_version())
        .bytes(provider_->collective_contract())
        .bytes(options_.exact_contract());
    collective_contract_ = std::move(exact).release();
    fingerprint_ = detail::fingerprint_seed();
    detail::fingerprint_mix(fingerprint_, collective_contract_);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(provider_); }
  [[nodiscard]] std::string_view identity() const noexcept {
    return provider_ ? provider_->identity() : std::string_view{};
  }
  [[nodiscard]] std::uint64_t interface_version() const noexcept {
    return provider_ ? provider_->interface_version() : 0;
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }
  [[nodiscard]] const OperatorFingerprint& fingerprint() const noexcept { return fingerprint_; }
  [[nodiscard]] const PreparedProviderOptions& options() const noexcept { return options_; }

  [[nodiscard]] KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls) const noexcept {
    return provider_ ? provider_->validate_controls(controls, options_)
                     : KrylovMethodValidation::reject(1, "no prepared Krylov method provider");
  }
  [[nodiscard]] KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts) const noexcept {
    return provider_ ? provider_->validate_problem(facts, options_)
                     : KrylovMethodValidation::reject(1, "no prepared Krylov method provider");
  }
  [[nodiscard]] KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request) const {
    if (!provider_)
      throw std::invalid_argument("prepared Krylov method provider is empty");
    KrylovWorkspaceRequirements result = provider_->workspace_requirements(request, options_);
    if (!result.valid())
      throw std::invalid_argument("prepared Krylov method provider returned an invalid workspace");
    return result;
  }
  [[nodiscard]] SolveReport solve(PreparedKrylovSolveContext& context) const {
    if (!provider_)
      throw std::invalid_argument("prepared Krylov method provider is empty");
    return provider_->solve(context, options_);
  }

  friend bool operator==(const PreparedKrylovMethod& left,
                         const PreparedKrylovMethod& right) noexcept {
    return left.collective_contract_ == right.collective_contract_;
  }

 private:
  std::shared_ptr<const PreparedKrylovMethodProvider> provider_{};
  PreparedProviderOptions options_{};
  std::string collective_contract_{};
  OperatorFingerprint fingerprint_{};
};

/// Append-only native registry.  Resolving a fifth method returns the same prepared handle type as
/// resolving a builtin; consumers never branch on the registered identity.
class PreparedKrylovMethodRegistry {
 public:
  void add(std::shared_ptr<const PreparedKrylovMethodProvider> provider) {
    if (!provider || provider->identity().empty() || provider->interface_version() == 0 ||
        provider->collective_contract().empty())
      throw std::invalid_argument("prepared Krylov method provider requires exact identities");
    const std::string identity(provider->identity());
    if (!providers_.emplace(identity, std::move(provider)).second)
      throw std::invalid_argument("duplicate prepared Krylov method provider '" + identity + "'");
  }

  [[nodiscard]] PreparedKrylovMethod resolve(std::string_view identity,
                                             PreparedProviderOptions options) const {
    const auto found = providers_.find(std::string(identity));
    if (found == providers_.end())
      throw std::invalid_argument("unknown prepared Krylov method provider '" +
                                  std::string(identity) + "'");
    return PreparedKrylovMethod(found->second, std::move(options));
  }

 private:
  std::map<std::string, std::shared_ptr<const PreparedKrylovMethodProvider>> providers_{};
};

namespace detail {

inline KrylovMethodValidation validate_common_krylov_controls(
    const KrylovMethodControls& controls) noexcept {
  if (controls.max_iterations <= 0)
    return KrylovMethodValidation::reject(10, "max_iterations must be positive");
  if (!std::isfinite(static_cast<double>(controls.rel_tol)) || controls.rel_tol < Real(0) ||
      controls.rel_tol >= Real(1))
    return KrylovMethodValidation::reject(11, "rel_tol must be finite and in [0, 1)");
  if (!std::isfinite(static_cast<double>(controls.abs_tol)) || controls.abs_tol < Real(0))
    return KrylovMethodValidation::reject(12, "abs_tol must be finite and non-negative");
  if (controls.rel_tol == Real(0) && controls.abs_tol == Real(0))
    return KrylovMethodValidation::reject(13, "at least one stopping tolerance must be non-zero");
  return KrylovMethodValidation::accept();
}

inline constexpr std::string_view kCgOptionsSchema = "pops.krylov.cg.options@1";
inline constexpr std::string_view kBicgstabOptionsSchema = "pops.krylov.bicgstab.options@1";
inline constexpr std::string_view kGmresOptionsSchema = "pops.krylov.gmres.options@1";
inline constexpr std::string_view kRichardsonOptionsSchema =
    "pops.krylov.richardson.options@1";

inline bool empty_options(const PreparedProviderOptions& options,
                          std::string_view schema) noexcept {
  return options.schema_identity == schema && options.values.empty();
}

inline const std::int64_t* exact_int_option(const PreparedProviderOptions& options,
                                            std::string_view schema,
                                            std::string_view key) noexcept {
  if (options.schema_identity != schema || options.values.size() != 1)
    return nullptr;
  const auto& entry = *options.values.begin();
  return entry.first == key ? std::get_if<std::int64_t>(&entry.second) : nullptr;
}

inline const double* exact_real_option(const PreparedProviderOptions& options,
                                       std::string_view schema,
                                       std::string_view key) noexcept {
  if (options.schema_identity != schema || options.values.size() != 1)
    return nullptr;
  const auto& entry = *options.values.begin();
  return entry.first == key ? std::get_if<double>(&entry.second) : nullptr;
}

inline KrylovMethodValidation validate_generic_problem_facts(
    const KrylovMethodProblemFacts& facts) noexcept {
  if (!facts.properties.valid())
    return KrylovMethodValidation::reject(20, "operator properties are incoherent");
  if (!field_distribution_is_valid(facts.distribution))
    return KrylovMethodValidation::reject(21, "vector distribution is invalid");
  if (facts.footprint.components < 1 || facts.footprint.input_ghosts < 0 ||
      facts.footprint.preconditioned != facts.has_preconditioner)
    return KrylovMethodValidation::reject(22, "prepared footprint is incoherent");
  if (facts.robust_payload_width == 0)
    return KrylovMethodValidation::reject(23, "vector metric has no robust reduction payload");
  return KrylovMethodValidation::accept();
}

inline std::size_t checked_krylov_product(std::size_t left, std::size_t right,
                                          std::string_view quantity) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::length_error("prepared Krylov " + std::string(quantity) + " size overflows");
  return left * right;
}

inline std::size_t checked_krylov_sum(std::size_t left, std::size_t right,
                                      std::string_view quantity) {
  if (left > std::numeric_limits<std::size_t>::max() - right)
    throw std::length_error("prepared Krylov " + std::string(quantity) + " size overflows");
  return left + right;
}

class CgKrylovMethodProvider final : public PreparedKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override { return "pops.krylov.cg"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override { return "pops.krylov.cg@1"; }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (!empty_options(options, kCgOptionsSchema))
      return KrylovMethodValidation::reject(14, "CG options contract is invalid");
    return validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts,
      const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    if (facts.has_preconditioner)
      return KrylovMethodValidation::reject(24, "CG has no prepared preconditioner slot");
    if (!facts.properties.certifies_cg(facts.has_nullspace))
      return KrylovMethodValidation::reject(
          25, "CG requires the authenticated positive-definite certificate for its nullspace");
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request,
      const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("CG workspace requires no preconditioner");
    return {.field_count = 4, .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions& options) const override;
};

class BicgstabKrylovMethodProvider final : public PreparedKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override { return "pops.krylov.bicgstab"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.krylov.bicgstab@1";
  }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (!empty_options(options, kBicgstabOptionsSchema))
      return KrylovMethodValidation::reject(14, "BiCGStab options contract is invalid");
    return validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts,
      const PreparedProviderOptions&) const noexcept override {
    return validate_generic_problem_facts(facts);
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request,
      const PreparedProviderOptions&) const override {
    return {.field_count = request.footprint.preconditioned ? 9u : 7u,
            .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions& options) const override;
};

class GmresKrylovMethodProvider final : public PreparedKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override { return "pops.krylov.gmres"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override { return "pops.krylov.gmres@1"; }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (const KrylovMethodValidation common = validate_common_krylov_controls(controls);
        !common.accepted())
      return common;
    const std::int64_t* restart = exact_int_option(options, kGmresOptionsSchema, "restart");
    if (restart == nullptr || *restart < 1)
      return KrylovMethodValidation::reject(14, "restart must be positive");
    if (*restart >
        max_krylov_batched_basis_extent(PreparedFieldAlgebra::kRobustDotPayloadWidth))
      return KrylovMethodValidation::reject(
          26, "restart exceeds the native batched collective capacity");
    return KrylovMethodValidation::accept();
  }
  KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts,
      const PreparedProviderOptions& options) const noexcept override {
    if (const KrylovMethodValidation common = validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    const std::int64_t* restart = exact_int_option(options, kGmresOptionsSchema, "restart");
    if (restart == nullptr || *restart < 1 ||
        *restart > max_krylov_batched_basis_extent(facts.robust_payload_width))
      return KrylovMethodValidation::reject(
          26, "restart exceeds the vector metric's batched collective capacity");
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request,
      const PreparedProviderOptions& options) const override {
    const std::int64_t* prepared_restart =
        exact_int_option(options, kGmresOptionsSchema, "restart");
    const int restart = prepared_restart == nullptr ? 0 : static_cast<int>(*prepared_restart);
    if (restart < 1 || restart > max_krylov_batched_basis_extent(request.robust_payload_width))
      throw std::invalid_argument(
          "GMRES workspace restart exceeds the vector metric's batched collective capacity");
    const std::size_t extent = static_cast<std::size_t>(restart);
    const std::size_t h = checked_krylov_product(extent + 1u, extent, "Hessenberg");
    const std::size_t real_count = checked_krylov_sum(h, 4u * extent + 1u, "scalar");
    const std::size_t scaled_count = checked_krylov_sum(h, 2u * extent + 1u, "scaled scalar");
    const std::size_t ordinary_count = extent + 1u;
    const std::size_t collective_count = checked_krylov_product(
        ordinary_count, request.robust_payload_width + 1u, "collective payload");
    const std::size_t reduction_capacity = checked_krylov_product(
        ordinary_count, request.robust_payload_width, "distribution reduction");
    return {.field_count = extent + (request.footprint.preconditioned ? 4u : 3u),
            .real_count = real_count,
            .scaled_scalar_count = scaled_count,
            .collective_value_count = collective_count,
            .reduction_value_capacity = reduction_capacity,
            .initial_residual_field = extent + 2u};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions& options) const override;
};

class RichardsonKrylovMethodProvider final : public PreparedKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override { return "pops.krylov.richardson"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.krylov.richardson@1";
  }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (const KrylovMethodValidation common = validate_common_krylov_controls(controls);
        !common.accepted())
      return common;
    const double* relaxation =
        exact_real_option(options, kRichardsonOptionsSchema, "relaxation");
    if (relaxation == nullptr || !std::isfinite(*relaxation) || *relaxation <= 0.0)
      return KrylovMethodValidation::reject(15, "relaxation must be finite and positive");
    return KrylovMethodValidation::accept();
  }
  KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts,
      const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    if (facts.has_preconditioner)
      return KrylovMethodValidation::reject(
          24, "Richardson has no prepared preconditioner slot");
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request,
      const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("Richardson workspace requires no preconditioner");
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions& options) const override;
};

}  // namespace detail

/// Builtin presets. Their providers are registered through the same append-only registry used by
/// extensions; these functions are conveniences, not a closed dispatch vocabulary.
[[nodiscard]] PreparedKrylovMethod cg_krylov_method();
[[nodiscard]] PreparedKrylovMethod bicgstab_krylov_method();
[[nodiscard]] PreparedKrylovMethod gmres_krylov_method(int restart = 30);
[[nodiscard]] PreparedKrylovMethod richardson_krylov_method(Real relaxation = Real(1));
[[nodiscard]] std::shared_ptr<PreparedKrylovMethodRegistry>
make_default_krylov_method_provider_registry();

}  // namespace pops
