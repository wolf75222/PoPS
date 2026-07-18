#pragma once

/// @file
/// @brief Persistent allocation-free evaluator for one prepared field-nullspace plan.

#include <pops/core/foundation/allocator.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

/// Prepared scientific evaluator for one immutable nullspace plan and vector space.
///
/// Construction authenticates the complete plan, materializes the dense Gram factor, and allocates
/// every reduction/replica buffer. Compatibility and gauge application then perform no host or
/// communication-buffer allocation. The workspace is intentionally mutable and single-solve-at-a-
/// time, like KrylovWorkspace; callers that execute concurrent solves own one workspace per solve.
class FieldNullspaceWorkspace {
 public:
  FieldNullspaceWorkspace(FieldNullspacePlan plan, std::vector<const MultiFab*> layouts,
                          std::vector<PreparedVectorDistribution> distributions,
                          int first_level = 0)
      : plan_(std::move(plan)),
        layouts_(std::move(layouts)),
        distributions_(std::move(distributions)),
        first_level_(first_level) {
    validate_field_nullspace_basis(layouts_, plan_, distributions_, first_level_);
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
      level_values_.assign(layouts_.size() * value_capacity_, 0.0);
      reduced_values_.assign(value_capacity_, 0.0);
      gram_factor_.assign(gram_value_count_, 0.0);
      coefficients_.assign(basis_count_, 0.0);
      std::size_t validation_capacity = 0;
      std::size_t reduction_capacity = 0;
      for (const PreparedVectorDistribution& distribution : distributions_) {
        validation_capacity =
            std::max(validation_capacity, distribution.validation_scratch_byte_count());
        reduction_capacity = std::max(
            reduction_capacity, distribution.reduction_scratch_value_count(value_capacity_));
      }
      validation_scratch_.assign(validation_capacity, char{0});
      reduction_scratch_.assign(reduction_capacity, 0.0);
    } catch (...) {
      allocation_failed = 1;
    }
    if (all_reduce_max(allocation_failed) != 0) {
      clear_storage_();
      throw std::runtime_error(
          "field-nullspace workspace allocation failed on at least one communicator rank");
    }
    assemble_gram_factor_();
  }

  FieldNullspaceWorkspace(const FieldNullspaceWorkspace&) = delete;
  FieldNullspaceWorkspace& operator=(const FieldNullspaceWorkspace&) = delete;
  FieldNullspaceWorkspace(FieldNullspaceWorkspace&&) noexcept = default;
  FieldNullspaceWorkspace& operator=(FieldNullspaceWorkspace&&) noexcept = default;

  [[nodiscard]] const FieldNullspacePlan& plan() const noexcept { return plan_; }
  [[nodiscard]] int first_level() const noexcept { return first_level_; }
  [[nodiscard]] std::size_t validation_scratch_byte_count() const noexcept {
    return validation_scratch_.size();
  }
  [[nodiscard]] std::size_t reduction_scratch_value_count() const noexcept {
    return reduction_scratch_.size();
  }

  /// Returns the persistent witness [dot(rhs,b_0), abs(rhs*b_0), ...]. The span remains valid until
  /// the next operation on this workspace.
  std::span<const double> require_compatible(
      std::span<const MultiFab* const> rhs_levels) {
    if (basis_count_ == 0)
      return {};
    require_hot_fields_(rhs_levels, "field nullspace compatibility");
    clear_level_values_(compatibility_value_count_);
    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const FieldNullspaceBasis& basis = plan_.bases[basis_index];
      for (std::size_t level = 0; level < rhs_levels.size(); ++level) {
        const MultiFab& rhs = *rhs_levels[level];
        const int resolved_level = first_level_ + static_cast<int>(level);
        const MultiFab* mask = basis.mask(resolved_level);
        const MultiFab* coverage = basis.coverage_mask(resolved_level);
        const Real measure = basis.measure(resolved_level);
        for (int local = 0; local < rhs.local_size(); ++local) {
          const ConstArray4 values = rhs.fab(local).const_array();
          const ConstArray4 mask_values =
              mask == nullptr ? ConstArray4{} : mask->fab(local).const_array();
          const ConstArray4 coverage_values =
              coverage == nullptr ? ConstArray4{} : coverage->fab(local).const_array();
          level_value_(level, 2 * basis_index) += static_cast<double>(reduce_sum_cell(
              rhs.box(local),
              detail::FieldBasisMomentKernel{values, mask_values, coverage_values,
                                             basis.field_component, mask != nullptr,
                                             coverage != nullptr, measure}));
          level_value_(level, 2 * basis_index + 1) += static_cast<double>(reduce_sum_cell(
              rhs.box(local),
              detail::FieldBasisAbsMomentKernel{values, mask_values, coverage_values,
                                                basis.field_component, mask != nullptr,
                                                coverage != nullptr, measure}));
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
      const double tolerance = 128.0 * std::numeric_limits<Real>::epsilon() *
                               (absolute > 1.0 ? absolute : 1.0);
      if (std::abs(moment) > tolerance)
        throw FieldNullspaceIncompatibleRhs(
            "field RHS is incompatible with prepared nullspace basis '" +
            plan_.bases[basis_index].identity + "'; silent projection is forbidden");
    }
    return std::span<const double>(reduced_values_.data(), compatibility_value_count_);
  }

  std::span<const double> require_compatible(const MultiFab& rhs) {
    const std::array<const MultiFab*, 1> levels{&rhs};
    return require_compatible(levels);
  }

  void apply_gauge(std::span<MultiFab* const> phi_levels) {
    if (basis_count_ == 0 || plan_.gauges.empty())
      return;
    require_hot_fields_(phi_levels, "field nullspace gauge");
    clear_level_values_(basis_count_);
    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const FieldNullspaceBasis& basis = plan_.bases[basis_index];
      for (std::size_t level = 0; level < phi_levels.size(); ++level) {
        MultiFab& phi = *phi_levels[level];
        const int resolved_level = first_level_ + static_cast<int>(level);
        const MultiFab* mask = basis.mask(resolved_level);
        const MultiFab* coverage = basis.coverage_mask(resolved_level);
        for (int local = 0; local < phi.local_size(); ++local) {
          const ConstArray4 values = phi.fab(local).const_array();
          const ConstArray4 mask_values =
              mask == nullptr ? ConstArray4{} : mask->fab(local).const_array();
          const ConstArray4 coverage_values =
              coverage == nullptr ? ConstArray4{} : coverage->fab(local).const_array();
          level_value_(level, basis_index) += static_cast<double>(reduce_sum_cell(
              phi.box(local),
              detail::FieldBasisMomentKernel{
                  values, mask_values, coverage_values, basis.field_component, mask != nullptr,
                  coverage != nullptr, basis.measure(resolved_level)}));
        }
      }
    }
    reduce_levels_(basis_count_, "field nullspace gauge moments");
    std::copy_n(reduced_values_.begin(), basis_count_, coefficients_.begin());
    detail::solve_field_nullspace_gram(gram_factor_, basis_count_, coefficients_);

    for (std::size_t basis_index = 0; basis_index < basis_count_; ++basis_index) {
      const FieldNullspaceBasis& basis = plan_.bases[basis_index];
      const std::size_t gauge = detail::gauge_index(plan_, basis.identity);
      if (gauge == plan_.gauges.size())
        throw std::logic_error("prepared field-nullspace gauge does not cover every basis");
      const Real coefficient =
          static_cast<Real>(coefficients_[basis_index]) - plan_.gauges[gauge].value;
      if (!std::isfinite(static_cast<double>(coefficient)))
        throw FieldNullspaceInvalidEvaluation(
            "prepared field-nullspace gauge produced a non-finite coefficient");
      for (std::size_t level = 0; level < phi_levels.size(); ++level) {
        MultiFab& phi = *phi_levels[level];
        const int resolved_level = first_level_ + static_cast<int>(level);
        const MultiFab* mask = basis.mask(resolved_level);
        const MultiFab* coverage = basis.coverage_mask(resolved_level);
        for (int local = 0; local < phi.local_size(); ++local) {
          const ConstArray4 mask_values =
              mask == nullptr ? ConstArray4{} : mask->fab(local).const_array();
          const ConstArray4 coverage_values =
              coverage == nullptr ? ConstArray4{} : coverage->fab(local).const_array();
          for_each_cell(
              phi.box(local),
              detail::ShiftFieldBasisKernel{phi.fab(local).array(), mask_values, coverage_values,
                                            basis.field_component, mask != nullptr,
                                            coverage != nullptr, coefficient});
        }
      }
    }
  }

  void apply_gauge(MultiFab& phi) {
    const std::array<MultiFab*, 1> levels{&phi};
    apply_gauge(levels);
  }

 private:
  template <class Pointer>
  void require_hot_fields_(std::span<Pointer const> fields, const char* where) {
    long invalid_local = fields.size() == layouts_.size() ? 0L : 1L;
    try {
      const std::size_t count = std::min(fields.size(), layouts_.size());
      for (std::size_t level = 0; level < count; ++level) {
        const MultiFab* field = fields[level];
        const MultiFab* prepared = layouts_[level];
        const bool structural_match =
            field != nullptr && prepared != nullptr && field->ncomp() == prepared->ncomp() &&
            field->box_array().boxes() == prepared->box_array().boxes() &&
            field->dmap().ranks() == prepared->dmap().ranks();
        if (!structural_match || !distributions_[level].layout_matches(*field))
          invalid_local = 1;
      }
    } catch (...) {
      invalid_local = 1;
    }
    if (all_reduce_max(invalid_local) != 0)
      throw std::invalid_argument(std::string(where) +
                                  " valid-cell layout differs from its prepared vector space");
    for (std::size_t level = 0; level < fields.size(); ++level) {
      const std::size_t bytes = distributions_[level].validation_scratch_byte_count();
      distributions_[level].require_exact_values(
          *fields[level], std::span<char>(validation_scratch_.data(), bytes), where);
    }
  }

  void assemble_gram_factor_() {
    clear_level_values_(gram_value_count_);
    for (std::size_t left = 0; left < basis_count_; ++left) {
      for (std::size_t right = left; right < basis_count_; ++right) {
        if (plan_.bases[left].field_component != plan_.bases[right].field_component)
          continue;
        for (std::size_t level = 0; level < layouts_.size(); ++level) {
          const MultiFab& layout = *layouts_[level];
          const int resolved_level = first_level_ + static_cast<int>(level);
          const MultiFab* left_mask = plan_.bases[left].mask(resolved_level);
          const MultiFab* right_mask = plan_.bases[right].mask(resolved_level);
          const MultiFab* left_coverage = plan_.bases[left].coverage_mask(resolved_level);
          const MultiFab* right_coverage = plan_.bases[right].coverage_mask(resolved_level);
          for (int local = 0; local < layout.local_size(); ++local) {
            const ConstArray4 left_values =
                left_mask == nullptr ? ConstArray4{} : left_mask->fab(local).const_array();
            const ConstArray4 right_values =
                right_mask == nullptr ? ConstArray4{} : right_mask->fab(local).const_array();
            const ConstArray4 left_coverage_values =
                left_coverage == nullptr ? ConstArray4{}
                                         : left_coverage->fab(local).const_array();
            const ConstArray4 right_coverage_values =
                right_coverage == nullptr ? ConstArray4{}
                                          : right_coverage->fab(local).const_array();
            level_value_(level, left * basis_count_ + right) +=
                static_cast<double>(reduce_sum_cell(
                    layout.box(local),
                    detail::FieldBasisGramKernel{
                        left_values, right_values, left_coverage_values, right_coverage_values,
                        left_mask != nullptr, right_mask != nullptr, left_coverage != nullptr,
                        right_coverage != nullptr,
                        plan_.bases[left].measure(resolved_level)}));
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
      const std::size_t scratch_count =
          distributions_[level].reduction_scratch_value_count(width);
      distributions_[level].reduce_sum_values(
          values, std::span<double>(reduction_scratch_.data(), scratch_count), quantity);
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

  FieldNullspacePlan plan_;
  std::vector<const MultiFab*> layouts_;
  std::vector<PreparedVectorDistribution> distributions_;
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
