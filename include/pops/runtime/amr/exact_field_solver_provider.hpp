/// @file
/// @brief Public exact-ranked provider contract for AMR elliptic solvers.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/runtime/export.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

enum class ExactFieldHierarchyMode : unsigned char { level_local = 0, composite = 1 };

template <int Dim>
struct ExactAmrFieldSolverBuildRequest {
  static_assert(Dim >= 1 && Dim <= 3,
                "ExactAmrFieldSolverBuildRequest only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using hierarchy_type = elliptic::mg::CompositeFacBuildRequest<Dim>;

  hierarchy_type hierarchy;
  ExactFieldHierarchyMode mode = ExactFieldHierarchyMode::level_local;
  PreparedProviderOptions provider_options;
  Real reaction = Real(0);
  std::string use_contract;
  std::string spatial_contract;
};

template <int Dim>
std::string make_exact_amr_field_solver_contract(
    std::string_view provider_identity, const ExactAmrFieldSolverBuildRequest<Dim>& request) {
  if (provider_identity.empty() || request.use_contract.empty() || request.spatial_contract.empty())
    throw std::invalid_argument(
        "exact AMR field solver contract requires provider, use, and spatial identities");
  ExactContractBuilder contract;
  contract.text("pops.amr.exact-field-solver")
      .scalar(std::uint32_t{3})
      .scalar(std::int32_t{Dim})
      .text(provider_identity)
      .scalar(request.mode)
      .text(request.use_contract)
      .bytes(request.spatial_contract)
      .bytes(request.provider_options.exact_contract())
      .scalar(request.reaction)
      .bytes(elliptic::mg::detail::fac_hierarchy_contract(request.hierarchy));
  return std::move(contract).release();
}

/// Prepared solver ownership remains rank-specialized. Runtime polymorphism selects a provider,
/// never a dimension, layout type, or field representation.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class ExactAmrFieldSolver {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim, MemorySpace>;

  virtual ~ExactAmrFieldSolver() = default;
  virtual std::string_view provider_identity() const noexcept = 0;
  virtual std::string_view exact_prepared_contract() const noexcept = 0;
  virtual bool couples_hierarchy_levels() const noexcept = 0;
  virtual int level_count() const noexcept = 0;
  virtual field_type& rhs_level(int level) = 0;
  virtual field_type& candidate_level(int level) = 0;
  virtual const field_type& candidate_level(int level) const = 0;
  virtual void install_newton(FieldNewtonOptions options) = 0;
  virtual void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) = 0;
  virtual void set_boundary_contexts(std::vector<FieldBoundaryExecutionContext<Dim>> contexts) = 0;
  virtual void install_nullspace(
      PreparedFieldNullspace<Dim> prepared,
      std::vector<PreparedVectorDistribution<Dim>> level_distributions) = 0;
  virtual int maximum_iterations() const noexcept = 0;
  virtual SolveReport solve() = 0;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class ExactAmrFieldSolverProvider {
 public:
  using request_type = ExactAmrFieldSolverBuildRequest<Dim>;
  using solver_type = ExactAmrFieldSolver<Dim, MemorySpace>;

  virtual ~ExactAmrFieldSolverProvider() = default;
  virtual std::string_view identity() const noexcept = 0;
  virtual std::uint64_t interface_version() const noexcept { return 3; }
  virtual std::string_view collective_contract() const noexcept = 0;
  virtual PreparedProviderSupport supports(const request_type& request) const noexcept = 0;
  virtual std::string expected_prepared_contract(const request_type& request) const = 0;
  virtual std::unique_ptr<solver_type> build(const request_type& request) const = 0;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class ExactAmrFieldSolverRegistry {
 public:
  using provider_type = ExactAmrFieldSolverProvider<Dim, MemorySpace>;

  void add(std::shared_ptr<const provider_type> provider) {
    if (!provider || provider->identity().empty() || provider->interface_version() == 0 ||
        provider->collective_contract().empty())
      throw std::invalid_argument("exact AMR field solver registry requires a complete provider");
    for (const auto& existing : providers_)
      if (existing->identity() == provider->identity())
        throw std::invalid_argument("duplicate exact AMR field solver provider identity");
    providers_.push_back(std::move(provider));
  }

  std::shared_ptr<const provider_type> find(std::string_view identity) const {
    for (const auto& provider : providers_)
      if (provider->identity() == identity)
        return provider;
    return {};
  }

 private:
  std::vector<std::shared_ptr<const provider_type>> providers_{};
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
POPS_EXPORT std::shared_ptr<const ExactAmrFieldSolverProvider<Dim, MemorySpace>>
make_builtin_exact_amr_field_solver_provider();

}  // namespace pops::runtime::amr
