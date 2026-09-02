/// @file
/// @brief Exact-ranked hierarchy tensor-elliptic storage and prepared-kernel provider.

#pragma once

#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/runtime/amr/tensor_composite_fac.hpp>
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
inline constexpr std::string_view kScalarTensorEllipticContract =
    "pops.operator.scalar-tensor-elliptic.exact-rank@3";

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

inline void checked_add_resident_storage(std::uint64_t& total, std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    throw std::overflow_error("tensor elliptic resident storage size overflows uint64");
  total += value;
}

template <class Value>
std::uint64_t vector_storage_bytes(const std::vector<Value>& values) {
  if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(Value))
    throw std::overflow_error("tensor elliptic resident vector storage overflows uint64");
  return static_cast<std::uint64_t>(values.capacity()) * sizeof(Value);
}

inline std::uint64_t external_string_storage_bytes(const std::string& value) noexcept {
  const auto object_begin = reinterpret_cast<std::uintptr_t>(&value);
  const auto object_end = object_begin + sizeof(value);
  const auto data = reinterpret_cast<std::uintptr_t>(value.data());
  return data >= object_begin && data < object_end
             ? 0
             : static_cast<std::uint64_t>(value.capacity()) + 1U;
}

template <int Dim>
std::uint64_t request_resident_storage_bytes(
    const HierarchyTensorSolverBuildRequest<Dim>& request) {
  std::uint64_t total = 0;
  checked_add_resident_storage(total, vector_storage_bytes(request.levels));
  for (const auto& level : request.levels) {
    checked_add_resident_storage(total, vector_storage_bytes(level.layout.boxes()));
    checked_add_resident_storage(total, vector_storage_bytes(level.distribution.layout().boxes()));
    checked_add_resident_storage(total, vector_storage_bytes(level.distribution.owners()));
  }
  checked_add_resident_storage(total, vector_storage_bytes(request.ratios));
  checked_add_resident_storage(total, external_string_storage_bytes(request.plan_identity));
  checked_add_resident_storage(total,
                               external_string_storage_bytes(request.operator_contract_identity));
  checked_add_resident_storage(total, vector_storage_bytes(request.assembly_field_slots));
  for (const std::string& slot : request.assembly_field_slots)
    checked_add_resident_storage(total, external_string_storage_bytes(slot));
  checked_add_resident_storage(total, external_string_storage_bytes(request.solution_field_slot));
  checked_add_resident_storage(total,
                               external_string_storage_bytes(request.options.schema_identity));
  for (const auto& option : request.options.values) {
    // std::map node allocator overhead is deliberately outside the logical-storage contract.  The
    // value object and every externally allocated string it retains are stable provider payload.
    checked_add_resident_storage(total, sizeof(option));
    checked_add_resident_storage(total, external_string_storage_bytes(option.first));
    if (const auto* string_value = std::get_if<std::string>(&option.second))
      checked_add_resident_storage(total, external_string_storage_bytes(*string_value));
  }
  return total;
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
  ::pops::device_fence();
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
  const ExecutionLane* execution_lane = nullptr;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
using PreparedHierarchyTensorKernel =
    PreparedProvider<SolveReport(const HierarchyTensorSolveInvocation<Dim, MemorySpace>&)>;

/// Storage driver for an exact prepared tensor kernel.
///
/// The builtin carries one compile-time dimension through allocation, preparation, and execution.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrTensorElliptic final : public PreparedHierarchyTensorSolver<Dim, MemorySpace> {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "tensor FAC supports dimensions one through three");
  using request_type = HierarchyTensorSolverBuildRequest<Dim>;
  using field_type = MultiFab<Dim, MemorySpace>;
  using kernel_type = PreparedHierarchyTensorKernel<Dim, MemorySpace>;
  using level_fields_type = HierarchyTensorLevelFields<Dim, MemorySpace>;

 private:
  struct LevelBuffers;

 public:
  AmrTensorElliptic(request_type request, kernel_type kernel,
                    tensor_elliptic_detail::TensorFacControls controls,
                    std::string prepared_contract, const ExecutionLane& lane)
      : request_(std::move(request)),
        kernel_(std::move(kernel)),
        controls_(std::move(controls)),
        prepared_contract_(std::move(prepared_contract)),
        slots_(tensor_elliptic_detail::assembly_slots<Dim>()) {
    hierarchy_tensor_detail::validate_request(request_);
    tensor_elliptic_detail::validate_controls(controls_);
    if (prepared_contract_.empty() || request_.components != 1)
      throw std::invalid_argument("dimension-generic tensor elliptic preparation is incomplete");
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
      std::vector<tensor_fac::LevelBinding<Dim, MemorySpace>> bindings;
      bindings.reserve(levels_.size());
      for (std::size_t level = 0; level < levels_.size(); ++level) {
        tensor_fac::LevelBinding<Dim, MemorySpace> binding;
        binding.geometry = &request_.levels[level].geometry;
        binding.boundary = &request_.levels[level].boundary;
        for (std::size_t coefficient = 0; coefficient < binding.coefficients.size(); ++coefficient)
          binding.coefficients[coefficient] = &levels_[level].coefficients[coefficient];
        binding.rhs = &levels_[level].rhs;
        binding.initial_guess = &levels_[level].initial_guess;
        binding.solution = &levels_[level].solution;
        bindings.push_back(binding);
      }
      tensor_fac_ = std::make_unique<tensor_fac::FullTensorCompositeFac<Dim, MemorySpace>>(
          std::span<const tensor_fac::LevelBinding<Dim, MemorySpace>>(bindings), request_.ratios,
          lane);
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

  bool borrows_execution_lane() const noexcept {
    return tensor_fac_ && tensor_fac_->borrows_execution_lane();
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

  /// Object-array size charged by a configured provider ceiling.  The buffers themselves remain
  /// private, but the provider must account their vector element array before materialization.
  [[nodiscard]] static constexpr std::uint64_t configured_level_buffer_object_bytes() noexcept {
    return sizeof(LevelBuffers);
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
    throw std::invalid_argument("dimension-generic tensor elliptic received an unknown field slot");
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
  PreparedResidentStorage derived_resident_storage() const override;

  SolveReport solve(const HierarchyTensorSolveControls& controls,
                    const ExecutionLane& lane) override {
    if (kernel_)
      return kernel_(HierarchyTensorSolveInvocation<Dim, MemorySpace>{
          &request_, std::span<const level_fields_type>(invocation_levels_), controls, controls_,
          &lane});
    if (!tensor_fac_)
      throw std::logic_error("flat tensor preparation delegates execution to prepared Krylov");
    CompositeFacOptions defaults;
    tensor_fac::Controls resolved;
    resolved.relative_tolerance = controls.relative_tolerance;
    resolved.absolute_tolerance = controls.absolute_tolerance;
    resolved.maximum_iterations = controls.maximum_iterations;
    resolved.fine_sweeps = controls_.fine_sweeps.value_or(defaults.fine_sweeps);
    resolved.coarse_relative_tolerance =
        controls_.coarse_relative_tolerance.value_or(defaults.coarse_rel_tol);
    resolved.coarse_absolute_tolerance =
        controls_.coarse_absolute_tolerance.value_or(defaults.coarse_abs_tol);
    resolved.coarse_cycles = controls_.coarse_cycles.value_or(defaults.coarse_cycles);
    return tensor_fac_->solve(resolved, lane);
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
      throw std::out_of_range("dimension-generic tensor elliptic level is outside its hierarchy");
    return levels_[static_cast<std::size_t>(level)];
  }

  request_type request_;
  kernel_type kernel_;
  tensor_elliptic_detail::TensorFacControls controls_;
  std::string prepared_contract_;
  std::vector<std::string> slots_;
  std::vector<LevelBuffers> levels_;
  std::vector<level_fields_type> invocation_levels_;
  std::unique_ptr<tensor_fac::FullTensorCompositeFac<Dim, MemorySpace>> tensor_fac_{};
};

template <int Dim, class MemorySpace>
PreparedResidentStorage AmrTensorElliptic<Dim, MemorySpace>::derived_resident_storage() const {
  // A PreparedProvider callback is type-erased behind std::function.  It is intentionally
  // unknown until that extension protocol supplies its own logical-storage hook.
  if (!this->preparation_sealed() || !kernel_.resident_storage().is_exact())
    return PreparedResidentStorage::unknown();

  std::uint64_t total = sizeof(AmrTensorElliptic<Dim, MemorySpace>);
  tensor_elliptic_detail::checked_add_resident_storage(
      total, tensor_elliptic_detail::request_resident_storage_bytes(request_));
  tensor_elliptic_detail::checked_add_resident_storage(
      total, tensor_elliptic_detail::external_string_storage_bytes(prepared_contract_));
  tensor_elliptic_detail::checked_add_resident_storage(
      total, tensor_elliptic_detail::vector_storage_bytes(slots_));
  for (const std::string& slot : slots_)
    tensor_elliptic_detail::checked_add_resident_storage(
        total, tensor_elliptic_detail::external_string_storage_bytes(slot));
  tensor_elliptic_detail::checked_add_resident_storage(
      total, tensor_elliptic_detail::vector_storage_bytes(levels_));
  for (const LevelBuffers& level : levels_) {
    for (const field_type& coefficient : level.coefficients)
      tensor_elliptic_detail::checked_add_resident_storage(total,
                                                           coefficient.resident_storage_bytes());
    for (const field_type* field : {&level.rhs, &level.flux, &level.initial_guess, &level.solution})
      tensor_elliptic_detail::checked_add_resident_storage(total, field->resident_storage_bytes());
  }
  tensor_elliptic_detail::checked_add_resident_storage(
      total, tensor_elliptic_detail::vector_storage_bytes(invocation_levels_));
  if (tensor_fac_) {
    const PreparedResidentStorage fac_storage = tensor_fac_->resident_storage();
    if (!fac_storage.is_exact())
      return PreparedResidentStorage::unknown();
    tensor_elliptic_detail::checked_add_resident_storage(
        total, sizeof(tensor_fac::FullTensorCompositeFac<Dim, MemorySpace>));
    tensor_elliptic_detail::checked_add_resident_storage(total, fac_storage.bytes);
  }
  return PreparedResidentStorage::exact(total);
}

namespace tensor_elliptic_detail {

inline void checked_add_configured_storage(std::uint64_t& total, std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    throw std::overflow_error("tensor elliptic configured storage size overflows uint64");
  total += value;
}

inline std::uint64_t checked_configured_storage_product(std::uint64_t left, std::uint64_t right) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::overflow_error("tensor elliptic configured storage product overflows uint64");
  return left * right;
}

inline std::uint64_t configured_external_string_storage(std::uint64_t characters) {
  std::uint64_t result = characters;
  checked_add_configured_storage(result, 1U);
  return result;
}

template <int Dim>
std::uint64_t configured_power(std::uint64_t base) {
  std::uint64_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result = checked_configured_storage_product(result, base);
  return result;
}

template <int Dim, class MemorySpace>
std::uint64_t configured_multifab_storage(std::uint64_t cells, std::uint64_t patches,
                                          int components, bool one_ghost_layer) {
  if (cells == 0 || patches == 0 || components < 1)
    throw std::invalid_argument("tensor elliptic configured field capacity is incomplete");
  std::uint64_t total = 0;
  const std::uint64_t growth = one_ghost_layer ? configured_power<Dim>(3) : 1U;
  // The envelope authenticates a level domain and a finite patch count, not one already
  // materialized decomposition.  Charge every permitted patch as a full-domain image: that is
  // conservative for the disjoint FAC path and remains a valid bound for a flat candidate whose
  // input layout has not yet been normalized by FAC validation.
  const std::uint64_t patch_cells = checked_configured_storage_product(cells, patches);
  const std::uint64_t values =
      checked_configured_storage_product(checked_configured_storage_product(patch_cells, growth),
                                         static_cast<std::uint64_t>(components));
  checked_add_configured_storage(
      total, checked_configured_storage_product(values, static_cast<std::uint64_t>(sizeof(Real))));
  // Mirror MultiFab::resident_storage_bytes(): two BoxArray images, one owner image, two index
  // maps, and the local Fab array.  A configured rank can own every patch, so global P is the
  // only safe detached bound.
  for (const std::size_t element_size :
       {sizeof(Box<Dim>), sizeof(Box<Dim>), sizeof(Index<Dim>), sizeof(std::size_t),
        sizeof(std::size_t), sizeof(Fab<Dim, MemorySpace>)})
    checked_add_configured_storage(total, checked_configured_storage_product(
                                              patches, static_cast<std::uint64_t>(element_size)));
  return total;
}

template <int Dim>
std::uint64_t configured_request_contract_bytes(
    const HierarchyTensorConfiguredStorageRequest<Dim>& request) {
  constexpr std::uint64_t kFrameBytes = 9;
  constexpr std::uint64_t kScalarFrameBytes = 20;
  constexpr std::uint64_t kSequenceCountBytes = 8;
  const std::uint64_t levels = static_cast<std::uint64_t>(request.level_cell_bounds.size());
  std::uint64_t total = 0;
  const auto text = [&total](std::string_view value) {
    checked_add_configured_storage(total, kFrameBytes + static_cast<std::uint64_t>(value.size()));
  };
  const auto scalar = [&total](std::uint64_t count = 1U) {
    checked_add_configured_storage(total,
                                   checked_configured_storage_product(count, kScalarFrameBytes));
  };
  const auto sequence = [&total](std::uint64_t count, std::uint64_t item_bytes) {
    checked_add_configured_storage(total, kFrameBytes + kSequenceCountBytes);
    checked_add_configured_storage(
        total, checked_configured_storage_product(count, kFrameBytes + item_bytes));
  };

  text("pops.hierarchy.tensor-solver-request");
  scalar(4);  // version, dimension, block, components
  text(request.plan_identity);
  text(request.operator_contract_identity);
  checked_add_configured_storage(total, kFrameBytes + kSequenceCountBytes);
  for (const std::string& slot : request.assembly_field_slots)
    checked_add_configured_storage(
        total, kFrameBytes + kFrameBytes + static_cast<std::uint64_t>(slot.size()));
  text(request.solution_field_slot);
  const std::string options_contract = request.options.exact_contract();
  checked_add_configured_storage(total,
                                 kFrameBytes + static_cast<std::uint64_t>(options_contract.size()));
  scalar();  // level count
  for (std::size_t level = 0; level < request.level_cell_bounds.size(); ++level) {
    const std::uint64_t patches = request.patch_bounds[level];
    scalar(static_cast<std::uint64_t>(16 * Dim + 1));
    // BoxArray and owner sequences retain their own outer frame, count, and nested item frame.
    sequence(patches, checked_configured_storage_product(static_cast<std::uint64_t>(2 * Dim),
                                                         kScalarFrameBytes));
    sequence(patches, checked_configured_storage_product(static_cast<std::uint64_t>(Dim),
                                                         kScalarFrameBytes));
  }
  scalar();  // ratio count
  scalar(checked_configured_storage_product(levels - 1U, static_cast<std::uint64_t>(Dim)));
  return total;
}

template <int Dim>
std::uint64_t configured_prepared_contract_bytes(
    const HierarchyTensorConfiguredStorageRequest<Dim>& request) {
  constexpr std::uint64_t kFrameBytes = 9;
  constexpr std::uint64_t kScalarFrameBytes = 20;
  const std::uint64_t request_contract = configured_request_contract_bytes(request);
  std::uint64_t total = 0;
  checked_add_configured_storage(
      total,
      kFrameBytes + static_cast<std::uint64_t>(
                        std::string_view{"pops.hierarchy.composite-tensor-prepared"}.size()));
  checked_add_configured_storage(total, checked_configured_storage_product(2U, kScalarFrameBytes));
  checked_add_configured_storage(total, kFrameBytes + request_contract);
  // PreparedProvider::optional_collective_contract(empty) stores a one-byte presence frame.
  checked_add_configured_storage(total, kFrameBytes + 1U);
  return total;
}

template <int Dim>
std::uint64_t configured_request_resident_storage(
    const HierarchyTensorConfiguredStorageRequest<Dim>& request) {
  const std::uint64_t levels = static_cast<std::uint64_t>(request.level_cell_bounds.size());
  std::uint64_t total = 0;
  checked_add_configured_storage(
      total,
      checked_configured_storage_product(
          levels, static_cast<std::uint64_t>(sizeof(HierarchyTensorLevelBuildRequest<Dim>))));
  for (std::size_t level = 0; level < request.level_cell_bounds.size(); ++level) {
    const std::uint64_t patches = request.patch_bounds[level];
    checked_add_configured_storage(
        total,
        checked_configured_storage_product(patches, static_cast<std::uint64_t>(sizeof(Box<Dim>))));
    checked_add_configured_storage(
        total,
        checked_configured_storage_product(patches, static_cast<std::uint64_t>(sizeof(Box<Dim>))));
    checked_add_configured_storage(
        total, checked_configured_storage_product(patches,
                                                  static_cast<std::uint64_t>(sizeof(Index<Dim>))));
  }
  checked_add_configured_storage(
      total,
      checked_configured_storage_product(
          levels - 1U, static_cast<std::uint64_t>(sizeof(::pops::amr::RefinementRatio<Dim>))));
  for (const std::string_view value : {std::string_view{request.plan_identity},
                                       std::string_view{request.operator_contract_identity},
                                       std::string_view{request.solution_field_slot},
                                       std::string_view{request.options.schema_identity}})
    checked_add_configured_storage(
        total, configured_external_string_storage(static_cast<std::uint64_t>(value.size())));
  checked_add_configured_storage(
      total, checked_configured_storage_product(
                 static_cast<std::uint64_t>(request.assembly_field_slots.size()),
                 static_cast<std::uint64_t>(sizeof(std::string))));
  for (const std::string& slot : request.assembly_field_slots)
    checked_add_configured_storage(
        total, configured_external_string_storage(static_cast<std::uint64_t>(slot.size())));
  using option_entry = std::pair<const std::string, PreparedProviderOptionValue>;
  for (const auto& option : request.options.values) {
    checked_add_configured_storage(total, sizeof(option_entry));
    checked_add_configured_storage(
        total, configured_external_string_storage(static_cast<std::uint64_t>(option.first.size())));
    if (const auto* value = std::get_if<std::string>(&option.second))
      checked_add_configured_storage(
          total, configured_external_string_storage(static_cast<std::uint64_t>(value->size())));
  }
  return total;
}

template <int Dim, class MemorySpace>
std::uint64_t composite_configured_storage_upper_bound(
    const HierarchyTensorConfiguredStorageRequest<Dim>& request) {
  hierarchy_tensor_detail::validate_configured_storage_request(request);
  const std::uint64_t levels = static_cast<std::uint64_t>(request.level_cell_bounds.size());
  std::uint64_t total = sizeof(AmrTensorElliptic<Dim, MemorySpace>);
  checked_add_configured_storage(total, configured_request_resident_storage(request));
  checked_add_configured_storage(
      total, configured_external_string_storage(configured_prepared_contract_bytes(request)));
  checked_add_configured_storage(
      total, checked_configured_storage_product(
                 static_cast<std::uint64_t>(request.assembly_field_slots.size()),
                 static_cast<std::uint64_t>(sizeof(std::string))));
  for (const std::string& slot : request.assembly_field_slots)
    checked_add_configured_storage(
        total, configured_external_string_storage(static_cast<std::uint64_t>(slot.size())));
  checked_add_configured_storage(
      total,
      checked_configured_storage_product(
          levels, AmrTensorElliptic<Dim, MemorySpace>::configured_level_buffer_object_bytes()));
  for (std::size_t level = 0; level < request.level_cell_bounds.size(); ++level) {
    const std::uint64_t cells = request.level_cell_bounds[level];
    const std::uint64_t patches = request.patch_bounds[level];
    // coefficients, flux, initial guess, and solution carry one ghost layer; RHS is valid-only.
    for (int field = 0; field < Dim * Dim + Dim + 2; ++field)
      checked_add_configured_storage(
          total, configured_multifab_storage<Dim, MemorySpace>(cells, patches, 1, true));
    checked_add_configured_storage(
        total, configured_multifab_storage<Dim, MemorySpace>(cells, patches, 1, false));
  }
  checked_add_configured_storage(
      total, checked_configured_storage_product(
                 levels,
                 static_cast<std::uint64_t>(sizeof(HierarchyTensorLevelFields<Dim, MemorySpace>))));
  if (levels > 1U) {
    using fac_type = tensor_fac::FullTensorCompositeFac<Dim, MemorySpace>;
    checked_add_configured_storage(total, sizeof(fac_type));
    checked_add_configured_storage(total, fac_type::configured_resident_storage_upper_bound(
                                              request.level_cell_bounds, request.patch_bounds,
                                              request.parent_child_pair_bounds, request.rank_bound,
                                              request.execution_lane_identity));
  }
  // PreparedHierarchyTensorSolver owns two solution rollback images in addition to the concrete
  // provider image above.
  checked_add_configured_storage(
      total, checked_configured_storage_product(
                 checked_configured_storage_product(2U, levels),
                 static_cast<std::uint64_t>(sizeof(MultiFab<Dim, MemorySpace>))));
  for (std::size_t level = 0; level < request.level_cell_bounds.size(); ++level)
    for (int image = 0; image < 2; ++image)
      checked_add_configured_storage(
          total, configured_multifab_storage<Dim, MemorySpace>(
                     request.level_cell_bounds[level], request.patch_bounds[level], 1, true));
  return total;
}

}  // namespace tensor_elliptic_detail

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class CompositeTensorHierarchyProvider final
    : public HierarchyTensorSolverProvider<Dim, MemorySpace> {
 public:
  using request_type = HierarchyTensorSolverBuildRequest<Dim>;
  using configured_storage_request_type = HierarchyTensorConfiguredStorageRequest<Dim>;
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
    return {"pops.hierarchy.composite-tensor-fac.exact-rank",
            "pops.hierarchy.composite-tensor-fac.flat-krylov",
            "pops.hierarchy.composite-tensor-fac.preallocated-publication",
            "pops.hierarchy.composite-tensor-fac.refined-full-tensor-fac",
            "pops.hierarchy.composite-tensor-fac.partitioned-mpi",
            "pops.hierarchy.composite-tensor-fac.full-tensor-nd@3"};
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
    try {
      hierarchy_tensor_detail::validate_request(request);
      if (request.components != 1)
        return PreparedProviderSupport::reject(12, "dimension-generic tensor operator is scalar");
      if (request.operator_contract_identity !=
          tensor_elliptic_detail::kScalarTensorEllipticContract)
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
  [[nodiscard]] HierarchyTensorConfiguredStorageLimit configured_storage_limit(
      const configured_storage_request_type& request) const override {
    hierarchy_tensor_detail::validate_configured_storage_request(request);
    if (request.provider_identity != identity() ||
        request.provider_interface_version != interface_version() || request.components != 1 ||
        request.operator_contract_identity !=
            tensor_elliptic_detail::kScalarTensorEllipticContract ||
        request.assembly_field_slots != tensor_elliptic_detail::assembly_slots<Dim>() ||
        request.solution_field_slot != "pops.tensor-elliptic.solution" ||
        !accepts_options(request.options).accepted() || !kernel_.resident_storage().is_exact())
      return HierarchyTensorConfiguredStorageLimit::unknown();
    return HierarchyTensorConfiguredStorageLimit::exact(
        tensor_elliptic_detail::composite_configured_storage_upper_bound<Dim, MemorySpace>(
            request));
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
  std::unique_ptr<solver_type> prepare(const request_type& request,
                                       const ExecutionLane& lane) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("tensor hierarchy provider rejected the build request");
    return std::make_unique<AmrTensorElliptic<Dim, MemorySpace>>(
        request, kernel_, tensor_elliptic_detail::decode_controls(request.options),
        expected_prepared_contract(request), lane);
  }

 private:
  kernel_type kernel_;
  std::string collective_contract_;
};

template <int Dim, class MemorySpace>
std::shared_ptr<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>
make_default_hierarchy_tensor_solver_provider_registry(const ExecutionLane& lane) {
  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>();
  registry->add(std::make_shared<CompositeTensorHierarchyProvider<Dim, MemorySpace>>(), lane);
  return registry;
}

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
std::shared_ptr<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>
make_hierarchy_tensor_solver_provider_registry(
    PreparedHierarchyTensorKernel<Dim, MemorySpace> kernel, const ExecutionLane& lane) {
  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>();
  registry->add(
      std::make_shared<CompositeTensorHierarchyProvider<Dim, MemorySpace>>(std::move(kernel)),
      lane);
  return registry;
}

}  // namespace pops::runtime::program
