/// @file
/// @brief Dimension-generic block registry and state marshaling for System.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/system/system_block_closures.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

/// Ordered storage for every equation block of one compile-time-ranked System.
///
/// The registry owns fields and the exact prepared hyperbolic boundary authority selected for each
/// block. Polar, embedded-boundary, elliptic and shared-interface implementations attach through
/// capability-qualified providers; none of their two-dimensional storage types enters this core.
template <int Dim>
class SystemBlockStore {
  static_assert(Dim >= 1 && Dim <= 3, "SystemBlockStore only supports dimensions 1, 2, and 3");

 public:
  using field_type = MultiFab<Dim>;
  using boundary_type = PreparedHyperbolicBoundary<Dim>;
  using evaluation_point = runtime::multiblock::BoundaryEvaluationPoint;
  using CellConvert = std::function<void(const double* input, double* output)>;
  using CellRecovery = std::function<RecoveryReport(const double* input, double* output)>;
  using CellBatchRecovery = UniformCellRecovery;
  using Residual = std::function<void(field_type&, field_type&)>;
  using ConstResidual = std::function<void(const field_type&, field_type&)>;
  using PointResidual = std::function<void(const evaluation_point&, field_type&, field_type&)>;
  using PreparedPointResidual =
      std::function<void(const evaluation_point&, field_type&, field_type&, const boundary_type&)>;
  using PointJvp =
      std::function<void(const evaluation_point&, field_type&, const field_type&, field_type&)>;
  using PreparedPointJvp = std::function<void(
      const evaluation_point&, field_type&, const field_type&, field_type&, const boundary_type&)>;
  using PointStatePreparation = std::function<void(const evaluation_point&, field_type&)>;
  using PreparedPointStatePreparation =
      std::function<void(const evaluation_point&, field_type&, const boundary_type&)>;

  struct ResidualFamily {
    PointResidual full;
    PointResidual flux_only;
    PointResidual without_prepared_interfaces;
    PointResidual flux_only_without_prepared_interfaces;
    PointResidual core;
    PointResidual flux_only_core;
    PreparedPointResidual full_prepared;
    PreparedPointResidual flux_only_full_prepared;
    PreparedPointResidual core_prepared;
    PreparedPointResidual flux_only_core_prepared;
  };

  using InterfaceProvider = SystemInterfaceProvider<Dim>;

  struct BlockState {
    std::string name;
    field_type U;
    int ncomp = 0;
    int substeps = 1;
    bool evolve = true;
    int stride = 1;
    double gamma = 1.0;
    Residual rhs_into;
    std::function<Real(const field_type&)> max_speed;
    ConstResidual add_poisson_rhs;

    VariableSet cons_vars;
    VariableSet prim_vars;
    CellConvert prim_to_cons;
    CellRecovery cons_to_prim;
    std::function<void(const field_type&, Real&, Index<Dim>&)> hotspot;
    std::function<Real(const field_type&)> source_frequency;
    std::function<Real(const field_type&)> stability_dt;
    std::function<void(field_type&)> project;
    std::function<void(field_type&)> project_masked;
    Residual rhs_flux_only;
    std::map<std::string, ConstResidual> named_poisson_rhs;
    Residual source_only;
    Residual source_only_masked;

    PointResidual rhs_at_point;
    PointResidual rhs_flux_only_at_point;
    PointResidual rhs_without_prepared_interfaces;
    PointResidual rhs_flux_only_without_prepared_interfaces;
    PointResidual rhs_core_at_point;
    PointResidual rhs_flux_only_core_at_point;
    PointResidual boundary_residual_at_point;
    PointJvp boundary_jvp_at_point;
    PreparedPointResidual rhs_core_at_point_prepared;
    PreparedPointResidual rhs_flux_only_core_at_point_prepared;
    PreparedPointResidual boundary_residual_at_point_prepared;
    PreparedPointJvp boundary_jvp_at_point_prepared;
    PointStatePreparation prepare_generated_state_at_point;
    PreparedPointStatePreparation prepare_generated_state_at_point_prepared;
    ResidualFamily staircase_residuals;
    ResidualFamily cutcell_residuals;

    /// The one immutable model-qualified transport-boundary authority for this block.
    std::shared_ptr<const boundary_type> boundary;
    std::string state_identity;
    CellBatchRecovery batch_cons_to_prim;
  };

  std::vector<BlockState> blocks;

  BlockState& find(const std::string& name) {
    for (BlockState& block : blocks)
      if (block.name == name)
        return block;
    throw std::runtime_error("System: unknown block '" + name + "'");
  }

  const BlockState& find(const std::string& name) const {
    for (const BlockState& block : blocks)
      if (block.name == name)
        return block;
    throw std::runtime_error("System: unknown block '" + name + "'");
  }

  int index(const std::string& name) const {
    for (std::size_t index = 0; index < blocks.size(); ++index)
      if (blocks[index].name == name)
        return static_cast<int>(index);
    throw std::runtime_error("System: unknown block '" + name + "'");
  }

  int size() const noexcept { return static_cast<int>(blocks.size()); }

  void install_interface_provider(InterfaceProvider provider) {
    if (interface_provider_)
      throw std::runtime_error("System shared-interface provider is already installed");
    if (!provider.evaluate_rhs || !provider.evaluate_core || !provider.evaluation_count ||
        !provider.has_interfaces || !provider.discard)
      throw std::invalid_argument(
          "System shared-interface provider must implement the complete ranked contract");
    interface_provider_ = std::move(provider);
  }

  std::vector<std::string> names() const {
    std::vector<std::string> result;
    result.reserve(blocks.size());
    for (const BlockState& block : blocks)
      result.push_back(block.name);
    return result;
  }

  /// Evaluate one sparse, simultaneous block batch. A capability-qualified shared-interface
  /// provider may replace this route before bind; the generic registry never reconstructs a 2D
  /// interface from axis names.
  void evaluate_rhs_with_interfaces(const evaluation_point& point,
                                    const std::vector<field_type*>& states,
                                    const std::vector<field_type*>& residuals,
                                    const std::vector<int>& flux_only = {}) {
    validate_batch_(states, residuals, flux_only);
    if (interface_provider_) {
      interface_provider_->evaluate_rhs(point, states, residuals, flux_only);
      return;
    }
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if (states[block] == nullptr)
        continue;
      const bool only_flux = !flux_only.empty() && flux_only[block] != 0;
      evaluate_rhs_core(point, block, *states[block], *residuals[block], only_flux);
    }
  }

  void evaluate_rhs_core_with_interfaces(const evaluation_point& point,
                                         const std::vector<field_type*>& states,
                                         const std::vector<field_type*>& residuals,
                                         const std::vector<int>& flux_only = {}) {
    validate_batch_(states, residuals, flux_only);
    if (interface_provider_) {
      interface_provider_->evaluate_core(point, states, residuals, flux_only);
      return;
    }
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if (states[block] == nullptr)
        continue;
      evaluate_rhs_core(point, block, *states[block], *residuals[block],
                        !flux_only.empty() && flux_only[block] != 0);
    }
  }

  void evaluate_rhs_core(const evaluation_point& point, std::size_t block, field_type& state,
                         field_type& residual, bool flux_only) {
    BlockState& selected = at_(block);
    if (selected.boundary) {
      PreparedPointResidual& closure = flux_only ? selected.rhs_flux_only_core_at_point_prepared
                                                 : selected.rhs_core_at_point_prepared;
      if (!closure)
        throw std::runtime_error("System block '" + selected.name +
                                 "' lacks its prepared hyperbolic residual provider");
      closure(point, state, residual, *selected.boundary);
      return;
    }
    PointResidual& closure =
        flux_only ? selected.rhs_flux_only_core_at_point : selected.rhs_core_at_point;
    if (!closure)
      throw std::runtime_error("System block '" + selected.name +
                               "' lacks a dimension-qualified core residual provider");
    closure(point, state, residual);
  }

  void evaluate_rhs_core_prepared(const evaluation_point& point, std::size_t block,
                                  field_type& state, field_type& residual, bool flux_only,
                                  const boundary_type& boundary) {
    BlockState& selected = at_(block);
    PreparedPointResidual& closure = flux_only ? selected.rhs_flux_only_core_at_point_prepared
                                               : selected.rhs_core_at_point_prepared;
    if (!closure)
      throw std::runtime_error("System block '" + selected.name +
                               "' lacks a prepared hyperbolic core residual provider");
    closure(point, state, residual, boundary);
  }

  /// Prepare one state for a generated stencil through the block's retained halo/boundary
  /// authority.  The generated Program supplies only the exact evaluation point and ranked field;
  /// it never re-derives a schedule or assumes a two-dimensional boundary table.
  void prepare_generated_state(const evaluation_point& point, std::size_t block,
                               field_type& state) {
    BlockState& selected = at_(block);
    if (selected.boundary) {
      if (!selected.prepare_generated_state_at_point_prepared)
        throw std::runtime_error("System block '" + selected.name +
                                 "' lacks its prepared generated-state provider");
      selected.prepare_generated_state_at_point_prepared(point, state, *selected.boundary);
      return;
    }
    if (!selected.prepare_generated_state_at_point)
      throw std::runtime_error("System block '" + selected.name +
                               "' lacks a dimension-qualified generated-state provider");
    selected.prepare_generated_state_at_point(point, state);
  }

  std::size_t interface_evaluation_count(const std::string& identity, int level) const {
    return interface_provider_ ? interface_provider_->evaluation_count(identity, level) : 0;
  }
  bool has_interfaces(int block) const {
    return interface_provider_ && interface_provider_->has_interfaces(block);
  }
  void discard_interface_fluxes() {
    if (!interface_provider_)
      return;
    interface_provider_->discard();
    interface_provider_.reset();
  }

  std::vector<double> copy_comp0(const field_type& field) const {
    return copy_components_(field, 1);
  }

  std::vector<double> copy_state(const field_type& field, int components) const {
    if (components != field.ncomp())
      throw std::invalid_argument("System state component count differs from its field");
    return copy_components_(field, components);
  }

  void write_state(field_type& field, int components, const std::vector<double>& input) {
    if (components != field.ncomp())
      throw std::invalid_argument("System state component count differs from its field");
    if (field.local_size() == 0)
      return;
    Fab<Dim>& fab = field.fab(0);
    const Box<Dim>& valid = fab.box();
    const std::size_t cells = checked_cells_(valid);
    if (cells > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(components) ||
        input.size() != cells * static_cast<std::size_t>(components))
      throw std::runtime_error(
          "System::set_state size differs from components times the local ranked box");

    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& grown = fab.grown_box();
    const Extent<Dim> grown_extents = grown.extent();
    const std::size_t component_stride = checked_cells_(grown);
    std::size_t input_offset = 0;
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < cells; ++linear) {
        const Index<Dim> index = unflatten_(valid, linear);
        const std::size_t storage = static_cast<std::size_t>(component) * component_stride +
                                    offset_(index, grown.lo, grown_extents);
        host(storage) = static_cast<Real>(input[input_offset++]);
      }
    fab.copy_from_host(host);
  }

 private:
  BlockState& at_(std::size_t block) {
    if (block >= blocks.size())
      throw std::out_of_range("System block index is out of range");
    return blocks[block];
  }

  void validate_batch_(const std::vector<field_type*>& states,
                       const std::vector<field_type*>& residuals,
                       const std::vector<int>& flux_only) const {
    if (states.size() != blocks.size() || residuals.size() != blocks.size() ||
        (!flux_only.empty() && flux_only.size() != blocks.size()))
      throw std::invalid_argument("System residual batch differs from the block registry");
    for (std::size_t block = 0; block < blocks.size(); ++block)
      if ((states[block] == nullptr) != (residuals[block] == nullptr))
        throw std::invalid_argument(
            "System sparse residual batch has only one state/output pointer");
  }

  std::vector<double> copy_components_(const field_type& field, int components) const {
    if (components < 1 || components > field.ncomp())
      throw std::invalid_argument("System requested an invalid component count");
    if (field.local_size() == 0)
      return {};
    const Fab<Dim>& fab = field.fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const Extent<Dim> grown_extents = grown.extent();
    const std::size_t cells = checked_cells_(valid);
    const std::size_t component_stride = checked_cells_(grown);
    std::vector<double> result;
    result.reserve(cells * static_cast<std::size_t>(components));
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < cells; ++linear) {
        const Index<Dim> index = unflatten_(valid, linear);
        const std::size_t storage = static_cast<std::size_t>(component) * component_stride +
                                    offset_(index, grown.lo, grown_extents);
        result.push_back(static_cast<double>(host(storage)));
      }
    return result;
  }

  static std::size_t checked_cells_(const Box<Dim>& box) {
    const std::int64_t cells = box.numPts();
    if (cells < 0 || static_cast<std::uint64_t>(cells) > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("System ranked box exceeds the host marshaling budget");
    return static_cast<std::size_t>(cells);
  }

  static Index<Dim> unflatten_(const Box<Dim>& box, std::size_t linear) {
    Index<Dim> index{};
    const Extent<Dim> extents = box.extent();
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t extent = static_cast<std::size_t>(extents[axis]);
      index[axis] = box.lo[axis] + static_cast<int>(linear % extent);
      linear /= extent;
    }
    return index;
  }

  static std::size_t offset_(const Index<Dim>& index, const Index<Dim>& origin,
                             const Extent<Dim>& extents) {
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      offset += static_cast<std::size_t>(index[axis] - origin[axis]) * stride;
      stride *= static_cast<std::size_t>(extents[axis]);
    }
    return offset;
  }

  std::optional<InterfaceProvider> interface_provider_;
};

}  // namespace pops
