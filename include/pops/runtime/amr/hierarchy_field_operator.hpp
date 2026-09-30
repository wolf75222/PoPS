#pragma once

#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>

namespace pops::runtime::program {

/// Optional apply-only capability. A solver without this interface cannot supply an
/// original nonlinear field residual. This interface never solves or publishes.
/// It leaves the existing hierarchy solver provider contract and virtual table intact.
template <int Dim>
class PreparedHierarchyFieldOperator {
 public:
  using field_type = MultiFab<Dim>;
  using hierarchy_type = std::vector<field_type>;
  static constexpr std::string_view identity = "pops.hierarchy.original-field-operator@1";
  virtual ~PreparedHierarchyFieldOperator() = default;
  virtual std::string_view original_field_contract() const noexcept = 0;
  virtual const ExecutionLane& original_field_execution_lane() const noexcept = 0;
  virtual int original_field_levels() const noexcept = 0;
  virtual const field_type& original_field_layout(int level) const = 0;
  virtual const field_type& original_field_active_cells(int level) const = 0;
  virtual Real original_field_cell_measure(int level) const = 0;
  virtual std::uint64_t original_field_preparation_generation() const noexcept = 0;
  /// Freeze actual coefficients through the provider's native preparation.
  virtual void prepare_original_field_operator() = 0;
  /// Apply the entire coupled conservative composite operator, including interfaces.
  virtual void apply_original_field_operator(const hierarchy_type&, hierarchy_type&) = 0;
  virtual void synchronize_original_field_candidate(hierarchy_type&) = 0;
  virtual Real original_field_dot(const hierarchy_type&, const hierarchy_type&) const = 0;
};

template <int Dim>
PreparedHierarchyFieldOperator<Dim>& require_original_field_operator(
    PreparedHierarchyTensorSolver<Dim>& solver, const ExecutionLane& lane) {
  PreparedHierarchyFieldOperator<Dim>* result = nullptr;
  std::exception_ptr error;
  try {
    result = dynamic_cast<PreparedHierarchyFieldOperator<Dim>*>(&solver);
    if (!result || &result->original_field_execution_lane() != &lane ||
        result->original_field_levels() != solver.level_count() ||
        result->original_field_contract() != solver.exact_prepared_contract())
      throw std::logic_error(
          "hierarchy provider has no prepared original-field operator capability");
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "original field operator capability");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string(PreparedHierarchyFieldOperator<Dim>::identity),
            std::string(result->original_field_contract())}},
          lane))
    throw std::logic_error("original field operator contract differs across execution ranks");
  return *result;
}
}  // namespace pops::runtime::program
