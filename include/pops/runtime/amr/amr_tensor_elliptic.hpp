/// @file
/// @brief Exact-ranked hierarchy tensor-elliptic storage and prepared-kernel provider.

#pragma once

#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/runtime/amr/tensor_composite_fac_2d.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pops::runtime::program {

namespace tensor_elliptic_detail {

inline constexpr std::string_view kCompositeTensorProvider = "pops.hierarchy.composite-tensor-fac";
inline constexpr std::string_view kCompositeTensorOptionSchema =
    "pops.hierarchy.composite-tensor-fac.options@2";
inline constexpr std::string_view kScalarTensorEllipticRank2Contract =
    "pops.operator.scalar-tensor-elliptic.rank2@2";

struct TensorFacControls {
  std::optional<int> fine_sweeps;
  std::optional<Real> coarse_relative_tolerance;
  std::optional<Real> coarse_absolute_tolerance;
  std::optional<int> coarse_cycles;
  std::optional<bool> verbose;
};

inline void validate_controls(const TensorFacControls& controls) {
  if (controls.fine_sweeps && *controls.fine_sweeps <= 0)
    throw std::invalid_argument("tensor FAC fine_sweeps must be positive");
  if (controls.coarse_relative_tolerance &&
      (!std::isfinite(static_cast<double>(*controls.coarse_relative_tolerance)) ||
       *controls.coarse_relative_tolerance <= Real(0) ||
       *controls.coarse_relative_tolerance >= Real(1)))
    throw std::invalid_argument("tensor FAC coarse relative tolerance must lie in (0, 1)");
  if (controls.coarse_absolute_tolerance &&
      (!std::isfinite(static_cast<double>(*controls.coarse_absolute_tolerance)) ||
       *controls.coarse_absolute_tolerance < Real(0)))
    throw std::invalid_argument("tensor FAC coarse absolute tolerance must be non-negative");
  if (controls.coarse_cycles && *controls.coarse_cycles <= 0)
    throw std::invalid_argument("tensor FAC coarse_cycles must be positive");
}

inline TensorFacControls decode_controls(const PreparedProviderOptions& options) {
  if (options.schema_identity != kCompositeTensorOptionSchema)
    throw std::invalid_argument("tensor FAC options use an unsupported exact schema");
  TensorFacControls controls;
  for (const auto& [key, value] : options.values) {
    if (key == "fac.fine_sweeps" || key == "fac.coarse_cycles") {
      if (!std::holds_alternative<std::int64_t>(value))
        throw std::invalid_argument("tensor FAC integer option has the wrong wire type");
      const std::int64_t raw = std::get<std::int64_t>(value);
      if (raw <= 0 || raw > std::numeric_limits<int>::max())
        throw std::invalid_argument("tensor FAC integer option is outside the native range");
      if (key == "fac.fine_sweeps")
        controls.fine_sweeps = static_cast<int>(raw);
      else
        controls.coarse_cycles = static_cast<int>(raw);
    } else if (key == "fac.coarse_rel_tol" || key == "fac.coarse_abs_tol") {
      if (!std::holds_alternative<double>(value))
        throw std::invalid_argument("tensor FAC tolerance option has the wrong wire type");
      if (key == "fac.coarse_rel_tol")
        controls.coarse_relative_tolerance = static_cast<Real>(std::get<double>(value));
      else
        controls.coarse_absolute_tolerance = static_cast<Real>(std::get<double>(value));
    } else if (key == "fac.verbose") {
      if (!std::holds_alternative<bool>(value))
        throw std::invalid_argument("tensor FAC verbose option has the wrong wire type");
      controls.verbose = std::get<bool>(value);
    } else {
      throw std::invalid_argument("unknown tensor FAC option '" + key + "'");
    }
  }
  validate_controls(controls);
  return controls;
}

inline PreparedProviderOptions default_options() {
  PreparedProviderOptions options;
  options.schema_identity = std::string(kCompositeTensorOptionSchema);
  return options;
}

inline std::string coefficient_slot(int row_axis, int column_axis) {
  if (row_axis < 0 || column_axis < 0)
    throw std::invalid_argument("tensor coefficient slot axes must be non-negative");
  return "pops.tensor-elliptic.coefficient." + std::to_string(row_axis) + "." +
         std::to_string(column_axis);
}

template <int Dim>
std::vector<std::string> assembly_slots() {
  std::vector<std::string> slots;
  slots.reserve(static_cast<std::size_t>(Dim * Dim + 2));
  for (int row = 0; row < Dim; ++row)
    for (int column = 0; column < Dim; ++column)
      slots.push_back(coefficient_slot(row, column));
  slots.emplace_back("pops.tensor-elliptic.rhs");
  slots.emplace_back("pops.tensor-elliptic.flux");
  return slots;
}

template <int Dim>
struct CopyValidKernel {
  FieldView<Real, Dim> destination;
  FieldView<const Real, Dim> source;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, component) = source(index, component);
  }
};

template <int Dim, class MemorySpace>
void copy_valid(MultiFab<Dim, MemorySpace>& destination, const MultiFab<Dim, MemorySpace>& source) {
  if (destination.layout() != source.layout() ||
      destination.distribution() != source.distribution() ||
      destination.local_rank() != source.local_rank() || destination.ncomp() != source.ncomp())
    throw std::invalid_argument("tensor elliptic copy requires one exact vector space");
  destination.set_val(Real(0));
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const FieldView<Real, Dim> output = destination.fab(local).view();
    const FieldView<const Real, Dim> input = std::as_const(source.fab(local)).view();
    for (int component = 0; component < destination.ncomp(); ++component)
      for_each_cell(destination.box(local), CopyValidKernel<Dim>{output, input, component});
  }
  Kokkos::fence();
}

template <int Dim>
Extent<Dim> one_ghost() {
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  return ghosts;
}

}  // namespace tensor_elliptic_detail

/// Exact fields presented to one prepared tensor kernel at one hierarchy level.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct HierarchyTensorLevelFields {
  using field_type = MultiFab<Dim, MemorySpace>;

  const Geometry<Dim>* geometry = nullptr;
  const PhysicalBoundaryConditions<Dim>* boundary = nullptr;
  std::array<field_type*, static_cast<std::size_t>(Dim* Dim)> coefficients{};
  field_type* rhs = nullptr;
  field_type* flux = nullptr;
  field_type* initial_guess = nullptr;
  field_type* solution = nullptr;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct HierarchyTensorSolveInvocation {
  static constexpr int dimension = Dim;

  const HierarchyTensorSolverBuildRequest<Dim>* request = nullptr;
  std::span<const HierarchyTensorLevelFields<Dim, MemorySpace>> levels;
  HierarchyTensorSolveControls solve_controls;
  tensor_elliptic_detail::TensorFacControls fac_controls;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
using PreparedHierarchyTensorKernel =
    PreparedProvider<SolveReport(const HierarchyTensorSolveInvocation<Dim, MemorySpace>&)>;

/// Storage driver for an exact prepared tensor kernel.
///
/// The mathematical builtin contract is intentionally rank two. The class remains templated so a
/// provider declaration for Dim=1/3 can be compiled and rejected during preparation without a
/// runtime dimension tag; only the provider's `if constexpr (Dim == 2)` route instantiates it.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrTensorElliptic final : public PreparedHierarchyTensorSolver<Dim, MemorySpace> {
 public:
  static_assert(Dim == 2,
                "the registered full-tensor hierarchy operator is mathematically rank two");
  using request_type = HierarchyTensorSolverBuildRequest<Dim>;
  using field_type = MultiFab<Dim, MemorySpace>;
  using kernel_type = PreparedHierarchyTensorKernel<Dim, MemorySpace>;
  using level_fields_type = HierarchyTensorLevelFields<Dim, MemorySpace>;

  AmrTensorElliptic(request_type request, kernel_type kernel,
                    tensor_elliptic_detail::TensorFacControls controls,
                    std::string prepared_contract)
      : request_(std::move(request)),
        kernel_(std::move(kernel)),
        controls_(std::move(controls)),
        prepared_contract_(std::move(prepared_contract)),
        slots_(tensor_elliptic_detail::assembly_slots<Dim>()) {
    hierarchy_tensor_detail::validate_request(request_);
    tensor_elliptic_detail::validate_controls(controls_);
    if (prepared_contract_.empty() || request_.components != 1)
      throw std::invalid_argument("rank-two tensor elliptic preparation is incomplete");
    levels_.reserve(request_.levels.size());
    for (const auto& level : request_.levels)
      levels_.emplace_back(level);
    invocation_levels_.reserve(levels_.size());
    for (std::size_t level = 0; level < levels_.size(); ++level) {
      LevelBuffers& storage = levels_[level];
      level_fields_type fields;
      fields.geometry = &request_.levels[level].geometry;
      fields.boundary = &request_.levels[level].boundary;
      for (std::size_t coefficient = 0; coefficient < fields.coefficients.size(); ++coefficient)
        fields.coefficients[coefficient] = &storage.coefficients[coefficient];
      fields.rhs = &storage.rhs;
      fields.flux = &storage.flux;
      fields.initial_guess = &storage.initial_guess;
      fields.solution = &storage.solution;
      invocation_levels_.push_back(fields);
    }
    if (!kernel_ && request_.levels.size() > 1) {
      std::vector<tensor_fac_2d::LevelBinding<MemorySpace>> bindings;
      bindings.reserve(levels_.size());
      for (std::size_t level = 0; level < levels_.size(); ++level) {
        tensor_fac_2d::LevelBinding<MemorySpace> binding;
        binding.geometry = &request_.levels[level].geometry;
        binding.boundary = &request_.levels[level].boundary;
        for (std::size_t coefficient = 0; coefficient < binding.coefficients.size(); ++coefficient)
          binding.coefficients[coefficient] = &levels_[level].coefficients[coefficient];
        binding.rhs = &levels_[level].rhs;
        binding.initial_guess = &levels_[level].initial_guess;
        binding.solution = &levels_[level].solution;
        bindings.push_back(binding);
      }
      tensor_fac_ = std::make_unique<tensor_fac_2d::TensorCompositeFac<MemorySpace>>(
          std::span<const tensor_fac_2d::LevelBinding<MemorySpace>>(bindings), request_.ratios);
    }
  }

  std::string_view provider_identity() const noexcept override {
    return tensor_elliptic_detail::kCompositeTensorProvider;
  }
  std::uint64_t provider_version() const noexcept override { return 2; }
  std::string_view exact_prepared_contract() const noexcept override { return prepared_contract_; }
  HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    return request_.levels.size() == 1 ? HierarchyTensorSolverExecutionPath::PreparedKrylovFallback
                                       : HierarchyTensorSolverExecutionPath::DirectProvider;
  }
  int level_count() const noexcept override { return static_cast<int>(levels_.size()); }

  bool owns_execution_lane() const noexcept {
    return tensor_fac_ && tensor_fac_->owns_execution_lane();
  }
  bool has_remote_same_level_halo() const noexcept {
    return tensor_fac_ && tensor_fac_->has_remote_same_level_halo();
  }
  bool has_remote_parent_gather() const noexcept {
    return tensor_fac_ && tensor_fac_->has_remote_parent_gather();
  }
  bool has_remote_fine_restriction() const noexcept {
    return tensor_fac_ && tensor_fac_->has_remote_fine_restriction();
  }
  bool uses_replicated_parent_restriction() const noexcept {
    return tensor_fac_ && tensor_fac_->uses_replicated_parent_restriction();
  }

  field_type& assembly_target(std::string_view slot, int level) override {
    LevelBuffers& storage = level_at_(level);
    for (std::size_t coefficient = 0; coefficient < static_cast<std::size_t>(Dim * Dim);
         ++coefficient)
      if (slot == slots_[coefficient])
        return storage.coefficients[coefficient];
    if (slot == "pops.tensor-elliptic.rhs")
      return storage.rhs;
    if (slot == "pops.tensor-elliptic.flux")
      return storage.flux;
    throw std::invalid_argument("rank-two tensor elliptic received an unknown field slot");
  }

  field_type& solution(int level) override { return level_at_(level).solution; }

  void stage_initial_guess(int level, const field_type* guess) override {
    LevelBuffers& storage = level_at_(level);
    if (guess == nullptr)
      storage.initial_guess.set_val(Real(0));
    else
      tensor_elliptic_detail::copy_valid(storage.initial_guess, *guess);
  }

 protected:
  SolveReport solve(const HierarchyTensorSolveControls& controls) override {
    if (kernel_)
      return kernel_(HierarchyTensorSolveInvocation<Dim, MemorySpace>{
          &request_, std::span<const level_fields_type>(invocation_levels_), controls, controls_});
    if (!tensor_fac_)
      throw std::logic_error("flat tensor preparation delegates execution to prepared Krylov");
    CompositeFacOptions defaults;
    tensor_fac_2d::Controls resolved;
    resolved.relative_tolerance = controls.relative_tolerance;
    resolved.absolute_tolerance = controls.absolute_tolerance;
    resolved.maximum_iterations = controls.maximum_iterations;
    resolved.fine_sweeps = controls_.fine_sweeps.value_or(defaults.fine_sweeps);
    resolved.coarse_relative_tolerance =
        controls_.coarse_relative_tolerance.value_or(defaults.coarse_rel_tol);
    resolved.coarse_absolute_tolerance =
        controls_.coarse_absolute_tolerance.value_or(defaults.coarse_abs_tol);
    resolved.coarse_cycles = controls_.coarse_cycles.value_or(defaults.coarse_cycles);
    return tensor_fac_->solve(resolved);
  }

 private:
  struct LevelBuffers {
    std::array<field_type, static_cast<std::size_t>(Dim* Dim)> coefficients;
    field_type rhs;
    field_type flux;
    field_type initial_guess;
    field_type solution;

    explicit LevelBuffers(const HierarchyTensorLevelBuildRequest<Dim>& level)
        : coefficients(make_coefficients_(level)),
          rhs(level.layout, level.distribution, level.local_rank, 1, Extent<Dim>{}),
          flux(level.layout, level.distribution, level.local_rank, Dim,
               tensor_elliptic_detail::one_ghost<Dim>()),
          initial_guess(level.layout, level.distribution, level.local_rank, 1,
                        tensor_elliptic_detail::one_ghost<Dim>()),
          solution(level.layout, level.distribution, level.local_rank, 1,
                   tensor_elliptic_detail::one_ghost<Dim>()) {}

   private:
    static std::array<field_type, static_cast<std::size_t>(Dim* Dim)> make_coefficients_(
        const HierarchyTensorLevelBuildRequest<Dim>& level) {
      std::array<field_type, static_cast<std::size_t>(Dim * Dim)> result;
      for (field_type& coefficient : result)
        coefficient = field_type(level.layout, level.distribution, level.local_rank, 1,
                                 tensor_elliptic_detail::one_ghost<Dim>());
      return result;
    }
  };

  LevelBuffers& level_at_(int level) {
    if (level < 0 || static_cast<std::size_t>(level) >= levels_.size())
      throw std::out_of_range("rank-two tensor elliptic level is outside its hierarchy");
    return levels_[static_cast<std::size_t>(level)];
  }

  request_type request_;
  kernel_type kernel_;
  tensor_elliptic_detail::TensorFacControls controls_;
  std::string prepared_contract_;
  std::vector<std::string> slots_;
  std::vector<LevelBuffers> levels_;
  std::vector<level_fields_type> invocation_levels_;
  std::unique_ptr<tensor_fac_2d::TensorCompositeFac<MemorySpace>> tensor_fac_{};
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class CompositeTensorHierarchyProvider final
    : public HierarchyTensorSolverProvider<Dim, MemorySpace> {
 public:
  using request_type = HierarchyTensorSolverBuildRequest<Dim>;
  using solver_type = PreparedHierarchyTensorSolver<Dim, MemorySpace>;
  using kernel_type = PreparedHierarchyTensorKernel<Dim, MemorySpace>;

  explicit CompositeTensorHierarchyProvider(kernel_type kernel = {}) : kernel_(std::move(kernel)) {
    ExactContractBuilder contract;
    contract.text("pops.hierarchy.composite-tensor-provider")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .optional_collective_contract(kernel_);
    collective_contract_ = std::move(contract).release();
  }

  std::string_view identity() const noexcept override {
    return tensor_elliptic_detail::kCompositeTensorProvider;
  }
  std::uint64_t interface_version() const noexcept override { return 2; }
  std::string_view collective_contract() const noexcept override { return collective_contract_; }
  std::vector<std::string> capability_contracts() const override {
    return {"pops.hierarchy.composite-tensor-fac.exact-rank@2",
            "pops.hierarchy.composite-tensor-fac.flat-krylov@2",
            "pops.hierarchy.composite-tensor-fac.preallocated-publication@2",
            "pops.hierarchy.composite-tensor-fac.refined-full-tensor-fac@2",
            "pops.hierarchy.composite-tensor-fac.partitioned-mpi@2",
            "pops.hierarchy.composite-tensor-fac.rank2-only@2"};
  }
  PreparedProviderOptions default_options() const override {
    return tensor_elliptic_detail::default_options();
  }
  PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    try {
      (void)tensor_elliptic_detail::decode_controls(options);
      (void)options.exact_contract();
      return PreparedProviderSupport::accept();
    } catch (...) {
      return PreparedProviderSupport::reject(1, "tensor FAC options are invalid");
    }
  }
  PreparedProviderSupport supports(const request_type& request) const noexcept override {
    if constexpr (Dim != 2) {
      (void)request;
      return PreparedProviderSupport::reject(
          10, "the registered full-tensor hierarchy operator is compile-time rank two");
    } else {
      try {
        hierarchy_tensor_detail::validate_request(request);
        if (request.components != 1)
          return PreparedProviderSupport::reject(12, "rank-two tensor operator is scalar");
        if (request.operator_contract_identity !=
            tensor_elliptic_detail::kScalarTensorEllipticRank2Contract)
          return PreparedProviderSupport::reject(13, "tensor operator contract is incompatible");
        if (request.assembly_field_slots != tensor_elliptic_detail::assembly_slots<Dim>() ||
            request.solution_field_slot != "pops.tensor-elliptic.solution")
          return PreparedProviderSupport::reject(14, "tensor field-slot contract is incompatible");
        if (!accepts_options(request.options).accepted())
          return PreparedProviderSupport::reject(15, "tensor provider options are incompatible");
        return PreparedProviderSupport::accept();
      } catch (...) {
        return PreparedProviderSupport::reject(16, "tensor hierarchy metadata is invalid");
      }
    }
  }
  PreparedProviderSupport accepts_execution(
      const request_type& request,
      HierarchyTensorSolverExecutionPath execution) const noexcept override {
    const PreparedProviderSupport request_support = supports(request);
    if (!request_support.accepted())
      return request_support;
    const auto expected = request.levels.size() == 1
                              ? HierarchyTensorSolverExecutionPath::PreparedKrylovFallback
                              : HierarchyTensorSolverExecutionPath::DirectProvider;
    return execution == expected
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(
                     17, "tensor provider execution does not match its exact hierarchy depth");
  }
  std::string expected_prepared_contract(const request_type& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("tensor hierarchy provider rejected the build request");
    ExactContractBuilder contract;
    contract.text("pops.hierarchy.composite-tensor-prepared")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .bytes(hierarchy_tensor_detail::request_contract(request))
        .optional_collective_contract(kernel_);
    return std::move(contract).release();
  }
  std::unique_ptr<solver_type> prepare(const request_type& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("tensor hierarchy provider rejected the build request");
    if constexpr (Dim == 2) {
      return std::make_unique<AmrTensorElliptic<Dim, MemorySpace>>(
          request, kernel_, tensor_elliptic_detail::decode_controls(request.options),
          expected_prepared_contract(request));
    } else {
      throw std::logic_error("non-rank-two tensor provider reached materialization");
    }
  }

 private:
  kernel_type kernel_;
  std::string collective_contract_;
};

template <int Dim, class MemorySpace>
std::shared_ptr<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>
make_default_hierarchy_tensor_solver_provider_registry() {
  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>();
  registry->add(std::make_shared<CompositeTensorHierarchyProvider<Dim, MemorySpace>>());
  return registry;
}

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
std::shared_ptr<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>
make_hierarchy_tensor_solver_provider_registry(
    PreparedHierarchyTensorKernel<Dim, MemorySpace> kernel) {
  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>();
  registry->add(
      std::make_shared<CompositeTensorHierarchyProvider<Dim, MemorySpace>>(std::move(kernel)));
  return registry;
}

}  // namespace pops::runtime::program
