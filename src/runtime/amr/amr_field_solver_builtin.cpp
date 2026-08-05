#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>

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
  result.mg.bottom_sweeps =
      static_cast<int>(option<std::int64_t>(options, "mg.bottom_sweeps"));
  result.mg.coarse_cell_threshold =
      static_cast<int>(option<std::int64_t>(options, "mg.coarse_threshold"));
  result.fac.max_iters = static_cast<int>(option<std::int64_t>(options, "fac.max_iters"));
  result.fac.fine_sweeps = static_cast<int>(option<std::int64_t>(options, "fac.fine_sweeps"));
  result.fac.rel_tol = static_cast<Real>(option<double>(options, "fac.rel_tol"));
  result.fac.abs_tol = static_cast<Real>(option<double>(options, "fac.abs_tol"));
  result.fac.coarse_rel_tol =
      static_cast<Real>(option<double>(options, "fac.coarse_rel_tol"));
  result.fac.coarse_abs_tol =
      static_cast<Real>(option<double>(options, "fac.coarse_abs_tol"));
  result.fac.coarse_cycles =
      static_cast<int>(option<std::int64_t>(options, "fac.coarse_cycles"));
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
      composite_ = std::make_unique<elliptic::mg::CompositeFacPoisson<Dim, MemorySpace>>(
          request.hierarchy, options_.fac, request.reaction);
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
    return static_cast<bool>(composite_);
  }
  int level_count() const noexcept override {
    return composite_ ? composite_->n_levels() : static_cast<int>(local_.size());
  }
  field_type& rhs_level(int level) override {
    return composite_ ? composite_->rhs_level(level)
                      : local_.at(static_cast<std::size_t>(level))->rhs();
  }
  field_type& candidate_level(int level) override {
    return composite_ ? composite_->phi_level(level)
                      : local_.at(static_cast<std::size_t>(level))->phi();
  }
  const field_type& candidate_level(int level) const override {
    return composite_ ? composite_->phi_level(level)
                      : local_.at(static_cast<std::size_t>(level))->phi();
  }
  int maximum_iterations() const noexcept override {
    return composite_ ? options_.fac.max_iters : options_.mg.maximum_cycles;
  }
  SolveReport solve() override {
    if (composite_)
      return composite_->solve();
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
  std::string contract_;
  ExactFieldHierarchyMode mode_ = ExactFieldHierarchyMode::level_local;
  BuiltinOptions options_{};
  std::vector<std::unique_ptr<elliptic::mg::GeometricMG<Dim, MemorySpace>>> local_{};
  std::unique_ptr<elliptic::mg::CompositeFacPoisson<Dim, MemorySpace>> composite_{};
};

template <int Dim, class MemorySpace>
class BuiltinExactAmrFieldSolverProvider final
    : public ExactAmrFieldSolverProvider<Dim, MemorySpace> {
 public:
  using request_type = ExactAmrFieldSolverBuildRequest<Dim>;
  using solver_type = ExactAmrFieldSolver<Dim, MemorySpace>;

  std::string_view identity() const noexcept override { return "geometric_mg"; }
  std::string_view collective_contract() const noexcept override {
    return "pops.amr.field-solver.geometric-mg.exact-ranked@2";
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
      } else if (n_ranks() > 1) {
        for (const auto& level : request.hierarchy.levels)
          if (!level.distribution.replicated())
            return PreparedProviderSupport::reject(
                4, "composite FAC distributed inter-level transfers are unavailable");
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
        request, expected_prepared_contract(request), decode_options<Dim>(request.provider_options));
  }
};

}  // namespace

template <int Dim, class MemorySpace>
std::shared_ptr<const ExactAmrFieldSolverProvider<Dim, MemorySpace>>
make_builtin_exact_amr_field_solver_provider() {
  return std::make_shared<BuiltinExactAmrFieldSolverProvider<Dim, MemorySpace>>();
}

template POPS_EXPORT std::shared_ptr<
    const ExactAmrFieldSolverProvider<kNativeDimension,
                                     typename Kokkos::DefaultExecutionSpace::memory_space>>
make_builtin_exact_amr_field_solver_provider<
    kNativeDimension, typename Kokkos::DefaultExecutionSpace::memory_space>();

}  // namespace pops::runtime::amr
