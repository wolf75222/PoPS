#pragma once

/// @file
/// @brief Persistent allocation-free evaluator for one prepared field-nullspace plan.

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// Prepared scientific evaluator for one immutable nullspace plan and vector space.
///
/// Construction authenticates the complete plan, materializes the dense Gram factor, and allocates
/// every reduction/replica buffer. Compatibility and gauge application then perform no host or
/// communication-buffer allocation. The workspace is intentionally mutable and single-solve-at-a-
/// time, like KrylovWorkspace; callers that execute concurrent solves own one workspace per solve.
template <int Dim>
class FieldNullspaceWorkspace {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "FieldNullspaceWorkspace only supports dimensions 1, 2, and 3");

  FieldNullspaceWorkspace(FieldNullspacePlan<Dim> plan, std::vector<const MultiFab<Dim>*> layouts,
                          std::vector<PreparedVectorDistribution<Dim>> distributions,
                          const ExecutionLane& lane, int first_level = 0)
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        plan_(std::move(plan)),
        layouts_(std::move(layouts)),
        distributions_(std::move(distributions)),
        first_level_(first_level) {
    initialize_(true);
  }

 private:
  struct CollectivelyPreflighted final {};

  FieldNullspaceWorkspace(FieldNullspacePlan<Dim> plan, std::vector<const MultiFab<Dim>*> layouts,
                          std::vector<PreparedVectorDistribution<Dim>> distributions,
                          const ExecutionLane& lane, int first_level, CollectivelyPreflighted)
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        plan_(std::move(plan)),
        layouts_(std::move(layouts)),
        distributions_(std::move(distributions)),
        first_level_(first_level) {
    initialize_(false);
  }

  void initialize_(bool validate_basis) {
    if (validate_basis)
      validate_field_nullspace_basis<Dim>(
          layouts_, plan_, std::span<const PreparedVectorDistribution<Dim>>(distributions_), *lane_,
          first_level_);
    basis_count_ = plan_.bases.size();
    if (basis_count_ == 0)
      return;
    gram_value_count_ = detail::checked_field_nullspace_collective_product(
        basis_count_, basis_count_, "prepared field-nullspace Gram matrix");
    compatibility_value_count_ = detail::checked_field_nullspace_collective_product(
        basis_count_, std::size_t{2}, "prepared field-nullspace compatibility moments");
    value_capacity_ = std::max({basis_count_, gram_value_count_, compatibility_value_count_});

    long allocation_failed = 0;
    try {
      const std::size_t level_value_count = detail::checked_field_nullspace_collective_product(
          layouts_.size(), value_capacity_, "prepared field-nullspace level moments");
      level_values_.assign(level_value_count, 0.0);
      reduced_values_.assign(value_capacity_, 0.0);
      gram_factor_.assign(gram_value_count_, 0.0);
      coefficients_.assign(basis_count_, 0.0);
      std::size_t validation_capacity = 0;
      std::size_t reduction_capacity = 0;
      for (const PreparedVectorDistribution<Dim>& distribution : distributions_) {
        validation_capacity =
            std::max(validation_capacity, distribution.validation_scratch_byte_count());
        reduction_capacity = std::max(reduction_capacity,
                                      distribution.reduction_scratch_value_count(value_capacity_));
      }
      validation_scratch_.assign(validation_capacity, char{0});
      reduction_scratch_.assign(reduction_capacity, 0.0);
    } catch (...) {
      allocation_failed = 1;
    }
    if (all_reduce_max(allocation_failed, *lane_) != 0) {
      clear_storage_();
      throw std::runtime_error(
          "field-nullspace workspace allocation failed on at least one communicator rank");
    }
    assemble_gram_factor_();
  }

 public:
  /// Construct after a collective preflight and a collectively agreed raw allocation.
  ///
  /// A plain ``make_unique`` is unsafe here: one rank could fail allocating the object before
  /// another enters the constructor's validation collectives.  This factory establishes the
  /// allocation decision on every rank first, then lets every rank follow the same constructor
  /// trace.  It is the required construction boundary for prepared runtime workspaces.
  [[nodiscard]] static std::unique_ptr<FieldNullspaceWorkspace> prepare_collectively(
      FieldNullspacePlan<Dim> plan, std::vector<const MultiFab<Dim>*> layouts,
      std::vector<PreparedVectorDistribution<Dim>> distributions, const ExecutionLane& lane,
      int first_level = 0) {
    preflight_field_nullspace_fields(
        layouts, plan, std::span<const PreparedVectorDistribution<Dim>>(distributions), first_level,
        detail::FieldNullspaceCollectiveBoundary::BasisValidation, lane);

    void* raw = nullptr;
    long raw_allocation_failed = 0;
    try {
      raw = ::operator new(sizeof(FieldNullspaceWorkspace));
    } catch (...) {
      raw_allocation_failed = 1;
    }
    if (all_reduce_max(raw_allocation_failed, lane) != 0) {
      ::operator delete(raw);
      throw std::runtime_error(
          "field-nullspace workspace object allocation failed on at least one communicator rank");
    }

    FieldNullspaceWorkspace* workspace = nullptr;
    long construction_failed = 0;
    try {
      workspace = ::new (raw)
          FieldNullspaceWorkspace(std::move(plan), std::move(layouts), std::move(distributions),
                                  lane, first_level, CollectivelyPreflighted{});
    } catch (...) {
      construction_failed = 1;
    }
    // ``FieldNullspaceWorkspace`` performs only preflight-authenticated collective phases after
    // the raw-allocation agreement above.  Converge its remaining local construction outcome
    // before allowing a successfully constructed rank to retain the workspace; otherwise a
    // following caller collective could strand a peer which saw an allocation/validation error.
    if (all_reduce_max(construction_failed, lane) != 0) {
      if (workspace != nullptr)
        workspace->~FieldNullspaceWorkspace();
      ::operator delete(raw);
      throw std::runtime_error(
          "field-nullspace workspace construction failed on at least one communicator rank");
    }
    return std::unique_ptr<FieldNullspaceWorkspace>(workspace);
  }

  FieldNullspaceWorkspace(const FieldNullspaceWorkspace&) = delete;
  FieldNullspaceWorkspace& operator=(const FieldNullspaceWorkspace&) = delete;
  FieldNullspaceWorkspace(FieldNullspaceWorkspace&&) noexcept = default;
  FieldNullspaceWorkspace& operator=(FieldNullspaceWorkspace&&) noexcept = default;

  [[nodiscard]] const FieldNullspacePlan<Dim>& plan() const noexcept { return plan_; }
  [[nodiscard]] int first_level() const noexcept { return first_level_; }
  [[nodiscard]] std::size_t validation_scratch_byte_count() const noexcept {
    return validation_scratch_.size();
  }
  [[nodiscard]] std::size_t reduction_scratch_value_count() const noexcept {
    return reduction_scratch_.size();
  }

  /// Logical dynamic storage retained by this prepared evaluator, excluding the workspace object.
  ///
  /// The masks are shared through the plan, so each distinct materialized mask is charged once.
  /// Allocator/control-block bookkeeping remains outside the logical resident-storage contract,
  /// while each retained ``MultiFab`` object and its field metadata/payload are included.
  [[nodiscard]] PreparedResidentStorage resident_storage() const {
    const auto add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("field-nullspace workspace storage size overflows uint64");
      total += value;
    };
    const auto vector_bytes = [](const auto& values) -> std::uint64_t {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
        throw std::overflow_error("field-nullspace workspace vector storage overflows uint64");
      return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
    };
    const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
      const auto object_begin = reinterpret_cast<std::uintptr_t>(&value);
      const auto object_end = object_begin + sizeof(value);
      const auto data = reinterpret_cast<std::uintptr_t>(value.data());
      if (data >= object_begin && data < object_end)
        return 0;
      if (value.capacity() == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("field-nullspace workspace string storage overflows size_t");
      return static_cast<std::uint64_t>(value.capacity()) + 1U;
    };

    std::uint64_t total = 0;
    add(total, external_string_bytes(plan_.identity));
    add(total, external_string_bytes(plan_.layout_identity));
    add(total, vector_bytes(plan_.bases));
    add(total, vector_bytes(plan_.gauges));

    std::vector<const MultiFab<Dim>*> unique_masks;
    for (const FieldNullspaceBasis<Dim>& basis : plan_.bases) {
      add(total, external_string_bytes(basis.identity));
      add(total, external_string_bytes(basis.provenance));
      add(total, external_string_bytes(basis.recipe_identity));
      add(total, vector_bytes(basis.masks));
      add(total, vector_bytes(basis.coverage));
      add(total, vector_bytes(basis.cell_measure));
      for (const auto& mask : basis.masks)
        if (mask != nullptr &&
            std::find(unique_masks.begin(), unique_masks.end(), mask.get()) == unique_masks.end())
          unique_masks.push_back(mask.get());
      for (const auto& coverage : basis.coverage)
        if (coverage != nullptr && std::find(unique_masks.begin(), unique_masks.end(),
                                             coverage.get()) == unique_masks.end())
          unique_masks.push_back(coverage.get());
    }
    for (const FieldGaugeConstraint& gauge : plan_.gauges)
      add(total, external_string_bytes(gauge.basis_identity));
    for (const MultiFab<Dim>* mask : unique_masks) {
      add(total, sizeof(MultiFab<Dim>));
      add(total, mask->resident_storage_bytes());
    }

    add(total, vector_bytes(layouts_));
    add(total, vector_bytes(distributions_));
    for (const PreparedVectorDistribution<Dim>& distribution : distributions_) {
      const PreparedResidentStorage storage = distribution.resident_storage();
      if (!storage.is_exact())
        return PreparedResidentStorage::unknown();
      add(total, storage.bytes);
    }
    add(total, vector_bytes(level_values_));
    add(total, vector_bytes(reduced_values_));
    add(total, vector_bytes(gram_factor_));
    add(total, vector_bytes(coefficients_));
    add(total, vector_bytes(validation_scratch_));
    add(total, vector_bytes(reduction_scratch_));
    return PreparedResidentStorage::exact(total);
  }

  /// Returns the persistent witness [dot(rhs,b_0), abs(rhs*b_0), ...]. The span remains valid until
  /// the next operation on this workspace.
  std::span<const double> require_compatible(std::span<const MultiFab<Dim>* const> rhs_levels) {
    if (basis_count_ == 0)
      return {};
    require_hot_fields_(rhs_levels, "field nullspace compatibility");
    clear_level_values_(compatibility_value_count_);
    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const FieldNullspaceBasis<Dim>& basis = plan_.bases[basis_index];
      for (std::size_t level = 0; level < rhs_levels.size(); ++level) {
        const MultiFab<Dim>& rhs = *rhs_levels[level];
        const int resolved_level = first_level_ + static_cast<int>(level);
        const MultiFab<Dim>* mask = basis.mask(resolved_level);
        const MultiFab<Dim>* coverage = basis.coverage_mask(resolved_level);
        const Real measure = basis.measure(resolved_level);
        for (std::size_t local = 0; local < rhs.local_size(); ++local) {
          const FieldView<const Real, Dim> values = rhs.fab(local).view();
          const FieldView<const Real, Dim> mask_values =
              mask == nullptr ? FieldView<const Real, Dim>{} : mask->fab(local).view();
          const FieldView<const Real, Dim> coverage_values =
              coverage == nullptr ? FieldView<const Real, Dim>{} : coverage->fab(local).view();
          level_value_(level, 2 * basis_index) += static_cast<double>(for_each_cell_reduce_sum(
              rhs.box(local), detail::FieldBasisMomentKernel<Dim>{
                                  values, mask_values, coverage_values, basis.field_component,
                                  mask != nullptr, coverage != nullptr, measure}));
          level_value_(level, 2 * basis_index + 1) += static_cast<double>(for_each_cell_reduce_sum(
              rhs.box(local), detail::FieldBasisAbsMomentKernel<Dim>{
                                  values, mask_values, coverage_values, basis.field_component,
                                  mask != nullptr, coverage != nullptr, measure}));
        }
      }
    }
    reduce_levels_(compatibility_value_count_, "field nullspace compatibility moments");
    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const double moment = reduced_values_[2 * basis_index];
      const double absolute = reduced_values_[2 * basis_index + 1];
      if (!std::isfinite(moment) || !std::isfinite(absolute))
        throw FieldNullspaceInvalidEvaluation(
            "field RHS has a non-finite prepared nullspace compatibility moment");
      const double tolerance =
          128.0 * std::numeric_limits<Real>::epsilon() * (absolute > 1.0 ? absolute : 1.0);
      if (std::abs(moment) > tolerance)
        throw FieldNullspaceIncompatibleRhs(
            "field RHS is incompatible with prepared nullspace basis '" +
            plan_.bases[basis_index].identity + "'; silent projection is forbidden");
    }
    return std::span<const double>(reduced_values_.data(), compatibility_value_count_);
  }

  std::span<const double> require_compatible(const MultiFab<Dim>& rhs) {
    const std::array<const MultiFab<Dim>*, 1> levels{&rhs};
    return require_compatible(levels);
  }

  void apply_gauge(std::span<MultiFab<Dim>* const> phi_levels) {
    if (basis_count_ == 0 || plan_.gauges.empty())
      return;
    require_hot_fields_(phi_levels, "field nullspace gauge");
    clear_level_values_(basis_count_);
    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const FieldNullspaceBasis<Dim>& basis = plan_.bases[basis_index];
      for (std::size_t level = 0; level < phi_levels.size(); ++level) {
        MultiFab<Dim>& phi = *phi_levels[level];
        const int resolved_level = first_level_ + static_cast<int>(level);
        const MultiFab<Dim>* mask = basis.mask(resolved_level);
        const MultiFab<Dim>* coverage = basis.coverage_mask(resolved_level);
        for (std::size_t local = 0; local < phi.local_size(); ++local) {
          const FieldView<const Real, Dim> values = std::as_const(phi.fab(local)).view();
          const FieldView<const Real, Dim> mask_values =
              mask == nullptr ? FieldView<const Real, Dim>{} : mask->fab(local).view();
          const FieldView<const Real, Dim> coverage_values =
              coverage == nullptr ? FieldView<const Real, Dim>{} : coverage->fab(local).view();
          level_value_(level, basis_index) += static_cast<double>(for_each_cell_reduce_sum(
              phi.box(local),
              detail::FieldBasisMomentKernel<Dim>{
                  values, mask_values, coverage_values, basis.field_component, mask != nullptr,
                  coverage != nullptr, basis.measure(resolved_level)}));
        }
      }
    }
    reduce_levels_(basis_count_, "field nullspace gauge moments");
    std::copy_n(reduced_values_.begin(), basis_count_, coefficients_.begin());
    detail::solve_field_nullspace_gram(gram_factor_, basis_count_, coefficients_);

    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const FieldNullspaceBasis<Dim>& basis = plan_.bases[basis_index];
      const std::size_t gauge = detail::gauge_index(plan_, basis.identity);
      if (gauge == plan_.gauges.size())
        throw std::logic_error("prepared field-nullspace gauge does not cover every basis");
      const Real coefficient =
          static_cast<Real>(coefficients_[basis_index]) - plan_.gauges[gauge].value;
      if (!std::isfinite(static_cast<double>(coefficient)))
        throw FieldNullspaceInvalidEvaluation(
            "prepared field-nullspace gauge produced a non-finite coefficient");
      for (std::size_t level = 0; level < phi_levels.size(); ++level) {
        MultiFab<Dim>& phi = *phi_levels[level];
        const int resolved_level = first_level_ + static_cast<int>(level);
        const MultiFab<Dim>* mask = basis.mask(resolved_level);
        const MultiFab<Dim>* coverage = basis.coverage_mask(resolved_level);
        for (std::size_t local = 0; local < phi.local_size(); ++local) {
          const FieldView<const Real, Dim> mask_values =
              mask == nullptr ? FieldView<const Real, Dim>{} : mask->fab(local).view();
          const FieldView<const Real, Dim> coverage_values =
              coverage == nullptr ? FieldView<const Real, Dim>{} : coverage->fab(local).view();
          for_each_cell(phi.box(local), detail::ShiftFieldBasisKernel<Dim>{
                                            phi.fab(local).view(), mask_values, coverage_values,
                                            basis.field_component, mask != nullptr,
                                            coverage != nullptr, coefficient});
        }
      }
    }
  }

  void apply_gauge(MultiFab<Dim>& phi) {
    const std::array<MultiFab<Dim>*, 1> levels{&phi};
    apply_gauge(levels);
  }

 private:
  template <class Pointer>
  void require_hot_fields_(std::span<Pointer const> fields, const char* where) {
    long invalid_local = fields.size() == layouts_.size() ? 0L : 1L;
    try {
      const std::size_t count = std::min(fields.size(), layouts_.size());
      for (std::size_t level = 0; level < count; ++level) {
        const MultiFab<Dim>* field = fields[level];
        const MultiFab<Dim>* prepared = layouts_[level];
        const bool structural_match = field != nullptr && prepared != nullptr &&
                                      field->ncomp() == prepared->ncomp() &&
                                      detail::field_nullspace_layouts_match(*field, *prepared);
        if (!structural_match || !distributions_[level].layout_matches(*field))
          invalid_local = 1;
      }
    } catch (...) {
      invalid_local = 1;
    }
    if (all_reduce_max(invalid_local, *lane_) != 0)
      throw std::invalid_argument(std::string(where) +
                                  " valid-cell layout differs from its prepared vector space");
    for (std::size_t level = 0; level < fields.size(); ++level) {
      const std::size_t bytes = distributions_[level].validation_scratch_byte_count();
      distributions_[level].require_exact_values(
          *fields[level], std::span<char>(validation_scratch_.data(), bytes), where, *lane_);
    }
  }

  void assemble_gram_factor_() {
    clear_level_values_(gram_value_count_);
    for (std::size_t left = 0; left < basis_count_; ++left) {
      for (std::size_t right = left; right < basis_count_; ++right) {
        if (plan_.bases[left].field_component != plan_.bases[right].field_component)
          continue;
        for (std::size_t level = 0; level < layouts_.size(); ++level) {
          const MultiFab<Dim>& layout = *layouts_[level];
          const int resolved_level = first_level_ + static_cast<int>(level);
          const MultiFab<Dim>* left_mask = plan_.bases[left].mask(resolved_level);
          const MultiFab<Dim>* right_mask = plan_.bases[right].mask(resolved_level);
          const MultiFab<Dim>* left_coverage = plan_.bases[left].coverage_mask(resolved_level);
          const MultiFab<Dim>* right_coverage = plan_.bases[right].coverage_mask(resolved_level);
          for (std::size_t local = 0; local < layout.local_size(); ++local) {
            const FieldView<const Real, Dim> left_values =
                left_mask == nullptr ? FieldView<const Real, Dim>{} : left_mask->fab(local).view();
            const FieldView<const Real, Dim> right_values = right_mask == nullptr
                                                                ? FieldView<const Real, Dim>{}
                                                                : right_mask->fab(local).view();
            const FieldView<const Real, Dim> left_coverage_values =
                left_coverage == nullptr ? FieldView<const Real, Dim>{}
                                         : left_coverage->fab(local).view();
            const FieldView<const Real, Dim> right_coverage_values =
                right_coverage == nullptr ? FieldView<const Real, Dim>{}
                                          : right_coverage->fab(local).view();
            level_value_(level, left * basis_count_ + right) +=
                static_cast<double>(for_each_cell_reduce_sum(
                    layout.box(local),
                    detail::FieldBasisGramKernel<Dim>{
                        left_values, right_values, left_coverage_values, right_coverage_values,
                        left_mask != nullptr, right_mask != nullptr, left_coverage != nullptr,
                        right_coverage != nullptr, plan_.bases[left].measure(resolved_level)}));
          }
        }
        for (std::size_t level = 0; level < layouts_.size(); ++level)
          level_value_(level, right * basis_count_ + left) =
              level_value_(level, left * basis_count_ + right);
      }
    }
    reduce_levels_(gram_value_count_, "prepared field-nullspace Gram matrix");
    std::copy_n(reduced_values_.begin(), gram_value_count_, gram_factor_.begin());
    detail::factor_field_nullspace_gram(gram_factor_, basis_count_,
                                        "prepared field-nullspace Gram matrix");
  }

  void clear_level_values_(std::size_t width) {
    for (std::size_t level = 0; level < layouts_.size(); ++level)
      std::fill_n(level_values_.begin() + level * value_capacity_, width, 0.0);
  }

  double& level_value_(std::size_t level, std::size_t index) {
    return level_values_[level * value_capacity_ + index];
  }

  void reduce_levels_(std::size_t width, const char* quantity) {
    std::fill_n(reduced_values_.begin(), width, 0.0);
    for (std::size_t level = 0; level < layouts_.size(); ++level) {
      std::span<double> values(level_values_.data() + level * value_capacity_, width);
      const std::size_t scratch_count = distributions_[level].reduction_scratch_value_count(width);
      distributions_[level].reduce_sum_values(
          values, std::span<double>(reduction_scratch_.data(), scratch_count), quantity, *lane_);
      for (std::size_t index = 0; index < width; ++index)
        reduced_values_[index] += values[index];
    }
  }

  void clear_storage_() noexcept {
    level_values_.clear();
    reduced_values_.clear();
    gram_factor_.clear();
    coefficients_.clear();
    validation_scratch_.clear();
    reduction_scratch_.clear();
  }

  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  FieldNullspacePlan<Dim> plan_;
  std::vector<const MultiFab<Dim>*> layouts_;
  std::vector<PreparedVectorDistribution<Dim>> distributions_;
  std::vector<double> level_values_;
  std::vector<double> reduced_values_;
  std::vector<double> gram_factor_;
  std::vector<double> coefficients_;
  std::vector<char, comm_allocator<char>> validation_scratch_;
  std::vector<double, comm_allocator<double>> reduction_scratch_;
  std::size_t basis_count_ = 0;
  std::size_t gram_value_count_ = 0;
  std::size_t compatibility_value_count_ = 0;
  std::size_t value_capacity_ = 0;
  int first_level_ = 0;
};

}  // namespace pops
