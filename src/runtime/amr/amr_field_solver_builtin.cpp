#include <pops/runtime/amr/amr_runtime.hpp>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/runtime/system/system_poisson_options.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pops {
namespace {

struct GeometricMgAmrOptions {
  GeometricMgOptions mg;
  CompositeFacOptions fac;
};

template <class Value>
Value solver_option(const AmrFieldSolverOptions& options, std::string_view name) {
  const auto found = options.values.find(std::string(name));
  if (found == options.values.end() || !std::holds_alternative<Value>(found->second))
    throw std::invalid_argument("AMR field solver option '" + std::string(name) +
                                "' is missing or has the wrong type");
  return std::get<Value>(found->second);
}

GeometricMgAmrOptions decode_options(const AmrFieldSolverOptions& options) {
  if (options.schema_identity != "pops.amr.field-solver-options.geometric-mg@1" ||
      options.values.size() != 16)
    throw std::invalid_argument("invalid geometric-MG AMR field solver option schema");
  GeometricMgAmrOptions decoded;
  decoded.mg.abs_tol = static_cast<Real>(solver_option<double>(options, "mg.abs_tol"));
  decoded.mg.rel_tol = static_cast<Real>(solver_option<double>(options, "mg.rel_tol"));
  decoded.mg.max_cycles =
      static_cast<int>(solver_option<std::int64_t>(options, "mg.max_cycles"));
  decoded.mg.min_coarse =
      static_cast<int>(solver_option<std::int64_t>(options, "mg.min_coarse"));
  decoded.mg.nu1 = static_cast<int>(solver_option<std::int64_t>(options, "mg.pre_smooth"));
  decoded.mg.nu2 = static_cast<int>(solver_option<std::int64_t>(options, "mg.post_smooth"));
  decoded.mg.nbottom =
      static_cast<int>(solver_option<std::int64_t>(options, "mg.bottom_sweeps"));
  decoded.mg.coarse_threshold =
      static_cast<int>(solver_option<std::int64_t>(options, "mg.coarse_threshold"));
  decoded.fac.max_iters =
      static_cast<int>(solver_option<std::int64_t>(options, "fac.max_iters"));
  decoded.fac.fine_sweeps =
      static_cast<int>(solver_option<std::int64_t>(options, "fac.fine_sweeps"));
  decoded.fac.rel_tol = static_cast<Real>(solver_option<double>(options, "fac.rel_tol"));
  decoded.fac.abs_tol = static_cast<Real>(solver_option<double>(options, "fac.abs_tol"));
  decoded.fac.coarse_rel_tol =
      static_cast<Real>(solver_option<double>(options, "fac.coarse_rel_tol"));
  decoded.fac.coarse_abs_tol =
      static_cast<Real>(solver_option<double>(options, "fac.coarse_abs_tol"));
  decoded.fac.coarse_cycles =
      static_cast<int>(solver_option<std::int64_t>(options, "fac.coarse_cycles"));
  decoded.fac.verbose = solver_option<bool>(options, "fac.verbose");
  return decoded;
}

void validate_options(const GeometricMgAmrOptions& options) {
  const auto& mg = options.mg;
  if (!std::isfinite(static_cast<double>(mg.abs_tol)) || mg.abs_tol < Real(0) ||
      !std::isfinite(static_cast<double>(mg.rel_tol)) || mg.rel_tol <= Real(0) ||
      mg.max_cycles < 1 || mg.min_coarse < 1 || mg.nu1 < 0 || mg.nu2 < 0 || mg.nbottom < 0 ||
      mg.coarse_threshold < 0)
    throw std::invalid_argument("invalid geometric-MG AMR field solver options");
  const auto& fac = options.fac;
  if (fac.max_iters < 1 || fac.fine_sweeps < 1 || fac.coarse_cycles < 1 ||
      !std::isfinite(static_cast<double>(fac.rel_tol)) || fac.rel_tol <= Real(0) ||
      fac.rel_tol >= Real(1) || !std::isfinite(static_cast<double>(fac.abs_tol)) ||
      fac.abs_tol < Real(0) || !std::isfinite(static_cast<double>(fac.coarse_rel_tol)) ||
      fac.coarse_rel_tol <= Real(0) || fac.coarse_rel_tol >= Real(1) ||
      !std::isfinite(static_cast<double>(fac.coarse_abs_tol)) || fac.coarse_abs_tol < Real(0))
    throw std::invalid_argument("invalid composite-FAC AMR field solver options");
}

enum class HierarchyPolicy : std::uint8_t { LevelLocal, Composite };

std::optional<HierarchyPolicy> decode_hierarchy_policy(
    const AmrFieldHierarchyPolicyAuthority& authority) noexcept {
  if (authority.interface_version != 1 ||
      authority.options.schema_identity != "pops.field-hierarchy.options.empty@1" ||
      !authority.options.values.empty())
    return std::nullopt;
  if (authority.policy_id == "pops.field-hierarchy.level-local")
    return HierarchyPolicy::LevelLocal;
  if (authority.policy_id == "pops.field-hierarchy.composite")
    return HierarchyPolicy::Composite;
  return std::nullopt;
}

class PreparedGeometricMgFieldSolver final : public AmrPreparedFieldSolver {
 public:
  PreparedGeometricMgFieldSolver(const AmrFieldSolverBuildRequest& request,
                                 std::string contract)
      : contract_(std::move(contract)), plan_(request.plan), options_(decode_options(
                                                               request.plan.solver_options)) {
    validate_options(options_);
    const auto policy = decode_hierarchy_policy(request.plan.hierarchy_policy);
    if (!policy)
      throw std::invalid_argument("geometric-MG received an unknown hierarchy-policy authority");
    const bool composite =
        *policy == HierarchyPolicy::Composite && request.hierarchy.nlev() > 1;
    if (composite) {
      distributions_.assign(static_cast<std::size_t>(request.hierarchy.nlev()),
                            FieldDistribution::Distributed);
      distributions_.front() = FieldDistribution::Replicated;
      std::vector<BoxArray> fine_boxes;
      fine_boxes.reserve(request.hierarchy.ba.size() - 1);
      for (std::size_t level = 1; level < request.hierarchy.ba.size(); ++level)
        fine_boxes.push_back(request.hierarchy.ba[level]);
      fac_ = std::make_unique<CompositeFacPoisson>(request.geometry, request.hierarchy.ba.front(),
                                                   request.boundary, fine_boxes);
      fac_->set_options(options_.fac);
      if (request.plan.has_reaction)
        fac_->set_reaction(request.plan.reaction);
      if (request.plan.has_boundary_kernel)
        fac_->set_boundary_kernel(request.plan.boundary_kernel, request.plan.boundary_context);
      if (request.plan.has_newton)
        fac_->set_field_nonlinear_options(request.plan.newton);
      return;
    }

    const int levels = *policy == HierarchyPolicy::LevelLocal ? request.hierarchy.nlev() : 1;
    distributions_.reserve(static_cast<std::size_t>(levels));
    level_solvers_.reserve(static_cast<std::size_t>(levels));
    int refinement = 1;
    for (int level = 0; level < levels; ++level) {
      const Geometry geometry = request.geometry.refine(refinement);
      const auto index = static_cast<std::size_t>(level);
      auto solver = std::make_unique<GeometricMG>(
          geometry, request.hierarchy.ba[index], request.hierarchy.dm[index], request.boundary,
          request.active, options_.mg.min_coarse, options_.mg.nu1, options_.mg.nu2,
          options_.mg.nbottom, options_.mg.coarse_threshold,
          level == 0 && request.replicated_coarse ? FieldDistribution::Replicated
                                                  : FieldDistribution::Distributed);
      distributions_.push_back(level == 0 && request.replicated_coarse
                                   ? FieldDistribution::Replicated
                                   : FieldDistribution::Distributed);
      solver->set_abs_tol(options_.mg.abs_tol);
      if (request.plan.has_reaction)
        solver->set_reaction(constant_scalar_field_provider(request.plan.reaction));
      if (request.plan.has_boundary_kernel)
        solver->set_boundary_kernel(request.plan.boundary_kernel, request.plan.boundary_context);
      if (request.plan.has_newton)
        solver->set_field_newton_options(request.plan.newton);
      level_solvers_.push_back(std::move(solver));
      if (index < request.hierarchy.refinement_ratios.size())
        refinement *= request.hierarchy.refinement_ratios[index];
    }
  }

  [[nodiscard]] std::string_view provider_identity() const noexcept override {
    return "geometric_mg";
  }
  [[nodiscard]] std::string_view exact_prepared_contract() const noexcept override {
    return contract_;
  }
  [[nodiscard]] bool couples_hierarchy_levels() const noexcept override {
    return static_cast<bool>(fac_);
  }
  [[nodiscard]] int level_count() const noexcept override {
    return fac_ ? fac_->n_levels() : static_cast<int>(level_solvers_.size());
  }
  [[nodiscard]] FieldDistribution level_distribution(int level) const override {
    return distributions_.at(static_cast<std::size_t>(level));
  }
  MultiFab& rhs_level(int level) override {
    return fac_ ? fac_->rhs_level(level)
                : level_solvers_.at(static_cast<std::size_t>(level))->rhs();
  }
  MultiFab& phi_level(int level) override {
    return fac_ ? fac_->phi_level(level)
                : level_solvers_.at(static_cast<std::size_t>(level))->phi();
  }
  void set_boundary_context(const FieldBoundaryExecutionContext& context) override {
    if (fac_) {
      fac_->set_boundary_context(context);
      return;
    }
    for (auto& solver : level_solvers_)
      solver->set_boundary_context(context);
  }
  SolveReport solve() override {
    if (fac_) {
      try {
        fac_->solve();
      } catch (...) {
        report_ = fac_->last_solve_report();
        throw;
      }
      report_ = fac_->last_solve_report();
      return report_;
    }
    for (auto& solver : level_solvers_) {
      try {
        if (plan_.has_boundary_kernel && plan_.boundary_kernel.observes_iteration)
          solver->solve();
        else
          solver->solve(options_.mg.rel_tol, options_.mg.max_cycles, options_.mg.abs_tol);
      } catch (...) {
        report_ = solver->last_solve_report();
        throw;
      }
      report_ = solver->last_solve_report();
      if (!report_.solved())
        return report_;
    }
    return report_;
  }
  [[nodiscard]] const SolveReport& last_solve_report() const noexcept override { return report_; }

 private:
  std::string contract_;
  AmrFieldSolveConfig plan_;
  GeometricMgAmrOptions options_;
  std::vector<std::unique_ptr<GeometricMG>> level_solvers_;
  std::vector<FieldDistribution> distributions_;
  std::unique_ptr<CompositeFacPoisson> fac_;
  SolveReport report_{};
};

class GeometricMgFieldSolverProvider final : public AmrFieldSolverProvider {
 public:
  [[nodiscard]] std::string_view identity() const noexcept override { return "geometric_mg"; }
  [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
  [[nodiscard]] std::string_view collective_contract() const noexcept override {
    return "pops.amr.field-solver.geometric-mg@1";
  }
  [[nodiscard]] std::vector<std::string> capability_contracts() const override {
    return {
        "pops.amr.field-solver.geometric-mg.active-region@1",
        "pops.amr.field-solver.geometric-mg.composite-hierarchy@1",
        "pops.amr.field-solver.geometric-mg.distributed-coarse@1",
        "pops.amr.field-solver.geometric-mg.dynamic-boundary@1",
        "pops.amr.field-solver.geometric-mg.exact-preparation@1",
        "pops.amr.field-solver.geometric-mg.level-local-hierarchy@1",
        "pops.amr.field-solver.geometric-mg.nonlinear-boundary@1",
        "pops.amr.field-solver.geometric-mg.reaction@1",
        "pops.amr.field-solver.geometric-mg.replicated-coarse@1",
    };
  }
  [[nodiscard]] AmrFieldSolverOptions default_field_options() const override {
    return geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
  }
  [[nodiscard]] std::optional<AmrFieldHierarchyPolicyAuthority> default_hierarchy_policy(
      std::string_view use_contract_identity) const override {
    if (use_contract_identity != "pops.amr.field-solver-use.default@1")
      return std::nullopt;
    return AmrFieldHierarchyPolicyAuthority{
        "pops.field-hierarchy.level-local",
        1,
        {"pops.field-hierarchy.options.empty@1", {}},
    };
  }
  [[nodiscard]] PreparedProviderSupport accepts_options(
      const AmrFieldSolverOptions& options) const noexcept override {
    try {
      validate_options(decode_options(options));
      return PreparedProviderSupport::accept();
    } catch (...) {
      return PreparedProviderSupport::reject(
          1, "geometric multigrid options do not match the provider schema");
    }
  }
  [[nodiscard]] PreparedProviderSupport supports(
      const AmrFieldSolverBuildRequest& request) const noexcept override {
    if (!accepts_options(request.plan.solver_options).accepted())
      return PreparedProviderSupport::reject(10, "field solver options are incompatible");
    const auto policy = decode_hierarchy_policy(request.plan.hierarchy_policy);
    if (!policy)
      return PreparedProviderSupport::reject(11, "hierarchy policy is not implemented by this provider");
    const bool composite = *policy == HierarchyPolicy::Composite;
    const bool level_local = *policy == HierarchyPolicy::LevelLocal;
    const bool default_field =
        request.use_contract_identity == "pops.amr.field-solver-use.default@1";
    const bool named_field =
        request.use_contract_identity == "pops.amr.field-solver-use.named@1";
    if (!default_field && !named_field)
      return PreparedProviderSupport::reject(12, "field-solver use contract is unsupported");
    if (default_field && (!level_local || request.hierarchy.nlev() != 1))
      return PreparedProviderSupport::reject(
          13, "default field solve requires this provider's one-level policy");
    if (composite && request.hierarchy.nlev() > 1 &&
        (!request.replicated_coarse || static_cast<bool>(request.active)))
      return PreparedProviderSupport::reject(
          14, "composite hierarchy cannot represent this coarse distribution or active region");
    if (level_local && request.hierarchy.nlev() > 1 &&
        (request.plan.has_boundary_kernel || request.plan.has_newton))
      return PreparedProviderSupport::reject(
          15, "multi-level local hierarchy cannot represent dynamic or nonlinear boundaries");
    return PreparedProviderSupport::accept();
  }
  [[nodiscard]] std::string expected_prepared_contract(
      const AmrFieldSolverBuildRequest& request) const override {
    return make_amr_field_solver_contract(identity(), request);
  }
  [[nodiscard]] std::unique_ptr<AmrPreparedFieldSolver> build(
      const AmrFieldSolverBuildRequest& request) const override {
    return std::make_unique<PreparedGeometricMgFieldSolver>(
        request, expected_prepared_contract(request));
  }
};

}  // namespace

std::shared_ptr<AmrFieldSolverProviderRegistry> make_default_amr_field_solver_registry() {
  auto registry = std::make_shared<AmrFieldSolverProviderRegistry>();
  registry->add(std::make_shared<GeometricMgFieldSolverProvider>());
  return registry;
}

}  // namespace pops
