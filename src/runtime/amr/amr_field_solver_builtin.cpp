#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>

#include <algorithm>
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

namespace pops::runtime::amr {
namespace {

struct BuiltinOptions {
  elliptic::mg::GeometricMultigridOptions mg;
  CompositeFacOptions fac;
};

template <class Value>
Value option(const PreparedProviderOptions& options, std::string_view name) {
  const auto found = options.values.find(std::string(name));
  if (found == options.values.end() || !std::holds_alternative<Value>(found->second))
    throw std::invalid_argument("geometric MG option '" + std::string(name) +
                                "' is missing or has the wrong type");
  return std::get<Value>(found->second);
}

template <int Dim>
BuiltinOptions decode_options(const PreparedProviderOptions& options) {
  if (options.schema_identity != "pops.amr.field-solver-options.geometric-mg@1" ||
      options.values.size() != 16)
    throw std::invalid_argument("invalid exact-ranked geometric MG option schema");
  BuiltinOptions result;
  result.mg.absolute_tolerance = static_cast<Real>(option<double>(options, "mg.abs_tol"));
  result.mg.relative_tolerance = static_cast<Real>(option<double>(options, "mg.rel_tol"));
  result.mg.maximum_cycles = static_cast<int>(option<std::int64_t>(options, "mg.max_cycles"));
  result.mg.minimum_coarse_extent =
      static_cast<int>(option<std::int64_t>(options, "mg.min_coarse"));
  result.mg.pre_sweeps = static_cast<int>(option<std::int64_t>(options, "mg.pre_smooth"));
  result.mg.post_sweeps = static_cast<int>(option<std::int64_t>(options, "mg.post_smooth"));
  result.mg.bottom_sweeps = static_cast<int>(option<std::int64_t>(options, "mg.bottom_sweeps"));
  result.mg.coarse_cell_threshold =
      static_cast<int>(option<std::int64_t>(options, "mg.coarse_threshold"));
  result.fac.max_iters = static_cast<int>(option<std::int64_t>(options, "fac.max_iters"));
  result.fac.fine_sweeps = static_cast<int>(option<std::int64_t>(options, "fac.fine_sweeps"));
  result.fac.rel_tol = static_cast<Real>(option<double>(options, "fac.rel_tol"));
  result.fac.abs_tol = static_cast<Real>(option<double>(options, "fac.abs_tol"));
  result.fac.coarse_rel_tol = static_cast<Real>(option<double>(options, "fac.coarse_rel_tol"));
  result.fac.coarse_abs_tol = static_cast<Real>(option<double>(options, "fac.coarse_abs_tol"));
  result.fac.coarse_cycles = static_cast<int>(option<std::int64_t>(options, "fac.coarse_cycles"));
  result.fac.verbose = option<bool>(options, "fac.verbose");
  elliptic::mg::detail::validate_options<Dim>(result.mg);
  elliptic::mg::detail::validate_fac_options(result.fac);
  return result;
}

template <int Dim, class MemorySpace>
class BuiltinExactAmrFieldSolver final : public ExactAmrFieldSolver<Dim, MemorySpace> {
 public:
  using base_type = ExactAmrFieldSolver<Dim, MemorySpace>;
  using field_type = typename base_type::field_type;
  using request_type = ExactAmrFieldSolverBuildRequest<Dim>;

  BuiltinExactAmrFieldSolver(const request_type& request, std::string contract,
                             BuiltinOptions options)
      : contract_(std::move(contract)), mode_(request.mode), options_(options) {
    if (mode_ == ExactFieldHierarchyMode::composite) {
      if (request.composite_route == ExactFieldCompositeRoute::partitioned) {
        partitioned_composite_ =
            std::make_unique<elliptic::amr::CompositeFacPoisson<Dim, MemorySpace>>(
                request.hierarchy, options_.fac, request.reaction);
      } else {
        elliptic::mg::CompositeFacBuildRequest<Dim> legacy;
        legacy.levels = request.hierarchy.levels;
        legacy.ratios = request.hierarchy.ratios;
        replicated_composite_ =
            std::make_unique<elliptic::mg::CompositeFacPoisson<Dim, MemorySpace>>(
                std::move(legacy), options_.fac, request.reaction);
      }
      return;
    }
    local_.reserve(request.hierarchy.levels.size());
    for (const auto& level : request.hierarchy.levels) {
      auto controls = options_.mg;
      controls.reaction = request.reaction;
      local_.push_back(
          std::make_unique<elliptic::mg::GeometricMG<Dim, MemorySpace>>(level, controls));
    }
  }

  std::string_view provider_identity() const noexcept override { return "geometric_mg"; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  bool couples_hierarchy_levels() const noexcept override {
    return static_cast<bool>(partitioned_composite_) || static_cast<bool>(replicated_composite_);
  }
  int level_count() const noexcept override {
    if (partitioned_composite_)
      return partitioned_composite_->n_levels();
    if (replicated_composite_)
      return replicated_composite_->n_levels();
    return static_cast<int>(local_.size());
  }
  field_type& rhs_level(int level) override {
    if (partitioned_composite_)
      return partitioned_composite_->rhs_level(level);
    if (replicated_composite_)
      return replicated_composite_->rhs_level(level);
    return local_.at(static_cast<std::size_t>(level))->rhs();
  }
  field_type& candidate_level(int level) override {
    if (partitioned_composite_)
      return partitioned_composite_->phi_level(level);
    if (replicated_composite_)
      return replicated_composite_->phi_level(level);
    return local_.at(static_cast<std::size_t>(level))->phi();
  }
  const field_type& candidate_level(int level) const override {
    if (partitioned_composite_)
      return partitioned_composite_->phi_level(level);
    if (replicated_composite_)
      return replicated_composite_->phi_level(level);
    return local_.at(static_cast<std::size_t>(level))->phi();
  }
  void install_newton(FieldNewtonOptions options) override {
    if (partitioned_composite_) {
      partitioned_composite_->install_newton(options);
      return;
    }
    if (replicated_composite_) {
      replicated_composite_->install_newton(options);
      return;
    }
    for (auto& solver : local_)
      solver->install_newton(options);
  }
  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) override {
    if (partitioned_composite_) {
      partitioned_composite_->install_boundary_kernel(std::move(kernel));
      return;
    }
    if (replicated_composite_) {
      replicated_composite_->install_boundary_kernel(std::move(kernel));
      return;
    }
    for (auto& solver : local_)
      solver->install_boundary_kernel(kernel);
  }
  void set_boundary_contexts(std::vector<FieldBoundaryExecutionContext<Dim>> contexts) override {
    if (contexts.size() != static_cast<std::size_t>(level_count()))
      throw std::invalid_argument(
          "exact AMR field solver requires one boundary context per live level");
    if (partitioned_composite_) {
      partitioned_composite_->set_boundary_contexts(std::move(contexts));
      return;
    }
    if (replicated_composite_) {
      replicated_composite_->set_boundary_contexts(std::move(contexts));
      return;
    }
    for (std::size_t level = 0; level < local_.size(); ++level)
      local_[level]->set_boundary_context(contexts[level]);
  }
  void install_nullspace(
      PreparedFieldNullspace<Dim> prepared,
      std::vector<PreparedVectorDistribution<Dim>> level_distributions) override {
    if (!nullspace_contract_.empty())
      throw std::logic_error("exact AMR field nullspace authority is already installed");
    if (prepared.provider_identity.empty() || prepared.provider_version == 0 ||
        prepared.exact_prepared_contract.empty())
      throw std::invalid_argument(
          "exact AMR field nullspace authority requires authenticated provider metadata");
    if (level_distributions.size() != static_cast<std::size_t>(level_count()))
      throw std::invalid_argument(
          "exact AMR field nullspace authority requires one distribution per level");

    if (partitioned_composite_) {
      partitioned_composite_->install_nullspace(std::move(prepared.plan),
                                                std::move(level_distributions));
    } else if (replicated_composite_) {
      replicated_composite_->install_nullspace(std::move(prepared.plan),
                                               std::move(level_distributions));
    } else {
      std::vector<FieldNullspacePlan<Dim>> plans;
      plans.reserve(local_.size());
      for (std::size_t level = 0; level < local_.size(); ++level)
        plans.push_back(level_local_plan_(prepared.plan, level));
      for (std::size_t level = 0; level < local_.size(); ++level)
        local_[level]->install_nullspace(std::move(plans[level]),
                                         std::move(level_distributions[level]));
    }
    nullspace_contract_ = std::move(prepared.exact_prepared_contract);
  }
  int maximum_iterations() const noexcept override {
    if (partitioned_composite_)
      return partitioned_composite_->maximum_iterations();
    if (replicated_composite_)
      return replicated_composite_->maximum_iterations();
    int result = 0;
    for (const auto& solver : local_)
      result = std::max(result, solver->maximum_iterations());
    return result;
  }
  SolveReport solve() override {
    if (nullspace_contract_.empty())
      throw std::logic_error("exact AMR field solve has no prepared nullspace authority");
    if (partitioned_composite_)
      return partitioned_composite_->solve();
    if (replicated_composite_)
      return replicated_composite_->solve();
    SolveReport result;
    result.mark_solved("geometric_mg_empty_level_set");
    for (auto& solver : local_) {
      result = solver->solve();
      if (!result.solved())
        return result;
    }
    return result;
  }

 private:
  static FieldNullspacePlan<Dim> level_local_plan_(const FieldNullspacePlan<Dim>& hierarchy_plan,
                                                   std::size_t level) {
    if (hierarchy_plan.empty())
      return {};
    FieldNullspacePlan<Dim> result;
    result.identity = hierarchy_plan.identity + ":level-local:" + std::to_string(level);
    result.layout_identity =
        hierarchy_plan.layout_identity + ":level-local:" + std::to_string(level);
    result.gauges = hierarchy_plan.gauges;
    result.bases.reserve(hierarchy_plan.bases.size());
    for (const FieldNullspaceBasis<Dim>& source : hierarchy_plan.bases) {
      FieldNullspaceBasis<Dim> basis;
      basis.identity = source.identity;
      basis.provenance = source.provenance;
      basis.recipe_identity = source.recipe_identity + ":level-local:" + std::to_string(level);
      basis.field_component = source.field_component;
      if (!source.masks.empty()) {
        if (level >= source.masks.size() || !source.masks[level])
          throw std::invalid_argument(
              "exact AMR level-local nullspace basis is missing its level mask");
        basis.masks.push_back(source.masks[level]);
      }
      basis.cell_measure.push_back(source.measure(static_cast<int>(level)));
      result.bases.push_back(std::move(basis));
    }
    return result;
  }

  std::string contract_;
  std::string nullspace_contract_;
  ExactFieldHierarchyMode mode_ = ExactFieldHierarchyMode::level_local;
  BuiltinOptions options_{};
  std::vector<std::unique_ptr<elliptic::mg::GeometricMG<Dim, MemorySpace>>> local_{};
  std::unique_ptr<elliptic::amr::CompositeFacPoisson<Dim, MemorySpace>> partitioned_composite_{};
  std::unique_ptr<elliptic::mg::CompositeFacPoisson<Dim, MemorySpace>> replicated_composite_{};
};

template <int Dim, class MemorySpace>
class BuiltinExactAmrFieldSolverProvider final
    : public ExactAmrFieldSolverProvider<Dim, MemorySpace> {
 public:
  using request_type = ExactAmrFieldSolverBuildRequest<Dim>;
  using solver_type = ExactAmrFieldSolver<Dim, MemorySpace>;

  std::string_view identity() const noexcept override { return "geometric_mg"; }
  std::string_view collective_contract() const noexcept override {
    return "pops.amr.field-solver.geometric-mg.exact-ranked@4";
  }
  PreparedProviderSupport supports(const request_type& request) const noexcept override {
    try {
      const BuiltinOptions decoded = decode_options<Dim>(request.provider_options);
      (void)decoded;
      if (request.hierarchy.levels.empty() ||
          request.hierarchy.ratios.size() + 1 != request.hierarchy.levels.size())
        return PreparedProviderSupport::reject(1, "AMR hierarchy is incomplete");
      if (!std::isfinite(static_cast<double>(request.reaction)) || request.reaction < Real(0))
        return PreparedProviderSupport::reject(2, "reaction coefficient is invalid");
      if (request.mode == ExactFieldHierarchyMode::level_local) {
        for (const auto& level : request.hierarchy.levels)
          if (!level.boxes.tiles_exactly(level.geometry.domain(), level.layout_budget))
            return PreparedProviderSupport::reject(
                3, "level-local geometric MG requires a complete uniform level");
      } else if (request.composite_route == ExactFieldCompositeRoute::partitioned) {
        for (std::size_t level = 1; level < request.hierarchy.levels.size(); ++level)
          if (request.hierarchy.levels[level].distribution.replicated())
            return PreparedProviderSupport::reject(14,
                                                   "partitioned composite FAC permits a replicated "
                                                   "root only before partitioned fine levels");
      } else {
        for (const auto& level : request.hierarchy.levels)
          if (!level.distribution.replicated())
            return PreparedProviderSupport::reject(
                15, "replicated composite hierarchy requires replicated levels");
      }
      return PreparedProviderSupport::accept();
    } catch (const std::exception& error) {
      return PreparedProviderSupport::reject(5, error.what());
    } catch (...) {
      return PreparedProviderSupport::reject(6, "geometric MG provider validation failed");
    }
  }
  std::string expected_prepared_contract(const request_type& request) const override {
    return make_exact_amr_field_solver_contract(identity(), request);
  }
  std::unique_ptr<solver_type> build(const request_type& request) const override {
    const PreparedProviderSupport decision = supports(request);
    if (!decision.accepted())
      throw std::invalid_argument(std::string(decision.reason));
    return std::make_unique<BuiltinExactAmrFieldSolver<Dim, MemorySpace>>(
        request, expected_prepared_contract(request),
        decode_options<Dim>(request.provider_options));
  }
};

}  // namespace

template <int Dim, class MemorySpace>
std::shared_ptr<const ExactAmrFieldSolverProvider<Dim, MemorySpace>>
make_builtin_exact_amr_field_solver_provider() {
  return std::make_shared<BuiltinExactAmrFieldSolverProvider<Dim, MemorySpace>>();
}

template POPS_EXPORT std::shared_ptr<const ExactAmrFieldSolverProvider<
    kNativeDimension, typename Kokkos::DefaultExecutionSpace::memory_space>>
make_builtin_exact_amr_field_solver_provider<
    kNativeDimension, typename Kokkos::DefaultExecutionSpace::memory_space>();

}  // namespace pops::runtime::amr
