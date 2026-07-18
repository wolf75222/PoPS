#pragma once

/// @file
/// @brief Provider-neutral prepared hierarchy tensor-solver protocol for compiled AMR Programs.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/solve_report_consensus.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

class AmrRuntime;

namespace runtime::program {

struct HierarchyTensorSolveControls {
  Real relative_tolerance = Real(0);
  Real absolute_tolerance = Real(0);
  int maximum_iterations = 0;
};

struct HierarchyTensorSolverBuildRequest {
  AmrRuntime* runtime = nullptr;
  int block = -1;
  int components = 0;
  int levels = 0;
  std::vector<bool> level_populated;
  std::vector<FieldDistribution> level_distributions;
  std::string plan_identity;
  std::string operator_contract_identity;
  std::vector<std::string> assembly_field_slots;
  std::string solution_field_slot;
  PreparedProviderOptions options;
};

/// Provider-selected execution path for the materialized request.  This is deliberately a small
/// protocol distinction, not a solver-family enumeration: the core only needs to know whether the
/// separately prepared level-local Krylov program owns execution, or whether the selected hierarchy
/// provider owns storage, solve and publication itself.
enum class HierarchyTensorSolverExecutionPath : std::uint8_t {
  PreparedKrylovFallback,
  DirectProvider,
};

/// Fully materialized tensor solve.  The Program context owns only this interface; concrete solver
/// storage, redistribution schedules and iteration policy remain provider-private.
class PreparedHierarchyTensorSolver {
 public:
  virtual ~PreparedHierarchyTensorSolver() = default;
  [[nodiscard]] virtual std::string_view provider_identity() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t provider_version() const noexcept = 0;
  [[nodiscard]] virtual std::string_view exact_prepared_contract() const noexcept = 0;
  [[nodiscard]] virtual HierarchyTensorSolverExecutionPath execution_path() const noexcept = 0;
  /// Resolve a provider-owned prepared field slot. The core treats the stable identity as opaque;
  /// a new operator envelope may add slots without modifying this protocol.
  virtual MultiFab& assembly_target(std::string_view field_slot_identity, int level) = 0;
  virtual MultiFab& solution(int level) = 0;
  virtual void stage_initial_guess(int level, const MultiFab* guess) = 0;
  /// Return only after recomputing the true residual defined by exact_prepared_contract(). The core
  /// validates the report structure collectively; scientific verification cannot be reconstructed
  /// outside the provider because the operator and publication storage are intentionally opaque.
  virtual SolveReport solve(const HierarchyTensorSolveControls& controls) = 0;
};

class HierarchyTensorSolverProvider {
 public:
  virtual ~HierarchyTensorSolverProvider() = default;
  [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t interface_version() const noexcept = 0;
  [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
  /// Opaque, inspectable declarations owned by this provider. The registry authenticates their exact
  /// bytes but never assigns semantics to individual entries.
  [[nodiscard]] virtual std::vector<std::string> capability_contracts() const = 0;
  [[nodiscard]] virtual PreparedProviderOptions default_options() const = 0;
  [[nodiscard]] virtual PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept = 0;
  [[nodiscard]] virtual PreparedProviderSupport supports(
      const HierarchyTensorSolverBuildRequest& request) const noexcept = 0;
  [[nodiscard]] virtual PreparedProviderSupport accepts_execution(
      const HierarchyTensorSolverBuildRequest& request,
      HierarchyTensorSolverExecutionPath execution) const noexcept = 0;
  [[nodiscard]] virtual std::string expected_prepared_contract(
      const HierarchyTensorSolverBuildRequest& request) const = 0;
  [[nodiscard]] virtual std::unique_ptr<PreparedHierarchyTensorSolver> prepare(
      const HierarchyTensorSolverBuildRequest& request) const = 0;
};

/// Execute one provider-owned hierarchy solve behind a collective publication boundary.  The exact
/// operator and scientific residual remain provider-owned, while the core guarantees that an
/// exception, malformed report or rank-divergent report rejects publication on every rank.
inline SolveReport solve_prepared_hierarchy_tensor_collectively(
    PreparedHierarchyTensorSolver& solver, const HierarchyTensorSolveControls& controls) {
  const bool invalid_controls =
      !std::isfinite(controls.relative_tolerance) || controls.relative_tolerance < Real(0) ||
      !std::isfinite(controls.absolute_tolerance) || controls.absolute_tolerance < Real(0) ||
      controls.maximum_iterations < 0;
  if (all_reduce_max(invalid_controls ? 1L : 0L) != 0)
    throw std::invalid_argument("hierarchy tensor solve controls are invalid");

  SolveReport report;
  bool solve_failed = false;
  try {
    report = solver.solve(controls);
  } catch (...) {
    solve_failed = true;
  }
  if (all_reduce_max(solve_failed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "hierarchy tensor-solver provider failed on at least one MPI rank");
  const bool malformed =
      !solve_report_is_publishable(report, controls.maximum_iterations);
  if (all_reduce_max(malformed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "hierarchy tensor-solver provider published a malformed SolveReport");
  ExactSolveReportConsensusScratch report_consensus;
  if (!report_consensus.agrees(report))
    throw std::runtime_error(
        "hierarchy tensor-solver provider report differs between MPI ranks");
  return report;
}

inline std::string exact_hierarchy_tensor_solver_provider_declaration(
    const HierarchyTensorSolverProvider& provider) {
  std::vector<std::string> capabilities = provider.capability_contracts();
  std::sort(capabilities.begin(), capabilities.end());
  if (std::any_of(capabilities.begin(), capabilities.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::adjacent_find(capabilities.begin(), capabilities.end()) != capabilities.end())
    throw std::invalid_argument(
        "hierarchy tensor-solver provider capabilities require unique exact identities");
  ExactContractBuilder contract;
  contract.text("pops.hierarchy.tensor-solver-provider-declaration")
      .scalar(std::uint32_t{1})
      .text(provider.identity())
      .scalar(provider.interface_version())
      .text(provider.collective_contract())
      .sequence(capabilities,
                [](ExactContractBuilder& item, const std::string& capability) {
                  item.text(capability);
                })
      .bytes(provider.default_options().exact_contract());
  return std::move(contract).release();
}

class HierarchyTensorSolverProviderRegistry {
 public:
  void add(std::shared_ptr<const HierarchyTensorSolverProvider> provider) {
    if (!provider || provider->identity().empty() || provider->interface_version() == 0 ||
        provider->collective_contract().empty())
      throw std::invalid_argument("hierarchy tensor-solver provider requires exact identities");
    const std::string identity(provider->identity());
    // Validate and canonicalize the complete declaration before mutating the append-only registry.
    // A malformed provider must not reserve its identity and poison a later valid registration.
    (void)exact_hierarchy_tensor_solver_provider_declaration(*provider);
    if (!providers_.emplace(identity, std::move(provider)).second)
      throw std::invalid_argument("duplicate hierarchy tensor-solver provider identity '" +
                                  identity + "'");
  }

  /// Install a provider from a compiled Program component. Program installation is collective, but
  /// each rank owns an independent registry; authenticate the complete declaration before mutating
  /// any of them. Reinstalling the exact same declaration is idempotent because one component may be
  /// referenced by multiple level-program instances. A same-name/different-contract replacement is
  /// always rejected.
  void add_collectively(std::shared_ptr<const HierarchyTensorSolverProvider> provider) {
    std::string identity;
    std::string declaration;
    bool local_invalid = false;
    try {
      if (!provider || provider->identity().empty() || provider->interface_version() == 0 ||
          provider->collective_contract().empty())
        throw std::invalid_argument("hierarchy tensor-solver provider requires exact identities");
      identity = std::string(provider->identity());
      declaration = exact_hierarchy_tensor_solver_provider_declaration(*provider);
    } catch (...) {
      local_invalid = true;
    }
    if (all_reduce_max(local_invalid || declaration.empty() ? 1L : 0L) != 0)
      throw std::runtime_error(
          "hierarchy tensor-solver component published an invalid provider declaration");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"hierarchy-tensor-component-provider", declaration}}))
      throw std::runtime_error(
          "hierarchy tensor-solver component provider differs across MPI ranks");

    const auto existing = providers_.find(identity);
    const long local_exists = existing == providers_.end() ? 0L : 1L;
    if (all_reduce_min(local_exists) != all_reduce_max(local_exists))
      throw std::runtime_error(
          "hierarchy tensor-solver provider registry differs across MPI ranks");
    if (local_exists != 0) {
      std::string existing_declaration;
      bool existing_invalid = false;
      try {
        existing_declaration =
            exact_hierarchy_tensor_solver_provider_declaration(*existing->second);
      } catch (...) {
        existing_invalid = true;
      }
      const bool existing_mismatch = existing_declaration != declaration;
      if (all_reduce_max(existing_invalid || existing_mismatch ? 1L : 0L) != 0)
        throw std::invalid_argument("conflicting hierarchy tensor-solver provider identity '" +
                                    identity + "'");
      return;
    }
    providers_.emplace(std::move(identity), std::move(provider));
  }

  [[nodiscard]] std::shared_ptr<const HierarchyTensorSolverProvider> resolve(
      std::string_view identity) const {
    const auto found = providers_.find(std::string(identity));
    if (found == providers_.end())
      throw std::invalid_argument("unknown hierarchy tensor-solver provider '" +
                                  std::string(identity) + "'");
    return found->second;
  }

 private:
  std::map<std::string, std::shared_ptr<const HierarchyTensorSolverProvider>> providers_;
};

inline std::unique_ptr<PreparedHierarchyTensorSolver>
prepare_hierarchy_tensor_solver_collectively(
    const HierarchyTensorSolverProviderRegistry& registry, std::string_view provider_identity,
    HierarchyTensorSolverBuildRequest request) {
  std::shared_ptr<const HierarchyTensorSolverProvider> provider;
  std::string declaration_contract;
  std::string option_support_contract;
  std::string request_support_contract;
  std::string expected_contract;
  PreparedProviderSupport option_support;
  PreparedProviderSupport request_support;
  bool declaration_failed = false;
  try {
    provider = registry.resolve(provider_identity);
    declaration_contract = exact_hierarchy_tensor_solver_provider_declaration(*provider);
    option_support = provider->accepts_options(request.options);
    request_support = provider->supports(request);
    option_support_contract = exact_prepared_provider_support(option_support);
    request_support_contract = exact_prepared_provider_support(request_support);
    if (option_support.accepted() && request_support.accepted())
      expected_contract = provider->expected_prepared_contract(request);
  } catch (...) {
    declaration_failed = true;
  }
  if (all_reduce_max(declaration_failed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "hierarchy tensor-solver provider support inspection failed on at least one MPI rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"hierarchy-tensor-provider", declaration_contract},
           {"hierarchy-tensor-option-support", option_support_contract},
           {"hierarchy-tensor-request-support", request_support_contract},
           {"hierarchy-tensor-expected-contract", expected_contract}}))
    throw std::runtime_error(
        "hierarchy tensor-solver declaration or support decision differs across MPI ranks");
  if (!option_support.accepted())
    throw std::invalid_argument(
        "hierarchy tensor-solver provider rejected options (code " +
        std::to_string(option_support.code) + "): " + std::string(option_support.reason));
  if (!request_support.accepted())
    throw std::invalid_argument(
        "hierarchy tensor-solver provider rejected request (code " +
        std::to_string(request_support.code) + "): " + std::string(request_support.reason));
  if (all_reduce_max(expected_contract.empty() ? 1L : 0L) != 0)
    throw std::runtime_error(
        "hierarchy tensor-solver provider accepted the request without an exact contract");

  std::unique_ptr<PreparedHierarchyTensorSolver> prepared;
  bool preparation_failed = false;
  try {
    prepared = provider->prepare(request);
  } catch (...) {
    preparation_failed = true;
  }
  if (all_reduce_max(preparation_failed || !prepared ? 1L : 0L) != 0)
    throw std::runtime_error("hierarchy tensor-solver preparation failed on at least one rank");
  PreparedProviderSupport execution_support;
  std::string execution_support_contract;
  bool execution_inspection_failed = false;
  try {
    execution_support = provider->accepts_execution(request, prepared->execution_path());
    execution_support_contract = exact_prepared_provider_support(execution_support);
  } catch (...) {
    execution_inspection_failed = true;
  }
  if (all_reduce_max(execution_inspection_failed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "hierarchy tensor-solver execution support inspection failed on at least one MPI rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"hierarchy-tensor-execution-support", execution_support_contract}}))
    throw std::runtime_error(
        "hierarchy tensor-solver execution support differs across MPI ranks");
  if (!execution_support.accepted())
    throw std::invalid_argument(
        "hierarchy tensor-solver provider rejected prepared execution path (code " +
        std::to_string(execution_support.code) + "): " +
        std::string(execution_support.reason));
  const bool mismatch = prepared->provider_identity() != provider->identity() ||
                        prepared->provider_version() != provider->interface_version() ||
                        prepared->exact_prepared_contract() != expected_contract;
  if (all_reduce_max(mismatch ? 1L : 0L) != 0)
    throw std::runtime_error("hierarchy tensor-solver published an invalid prepared contract");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"hierarchy-tensor-actual-contract", prepared->exact_prepared_contract()}}))
    throw std::runtime_error("prepared hierarchy tensor solver differs across MPI ranks");
  return prepared;
}

std::shared_ptr<HierarchyTensorSolverProviderRegistry>
make_default_hierarchy_tensor_solver_provider_registry();

}  // namespace runtime::program
}  // namespace pops
