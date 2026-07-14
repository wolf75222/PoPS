/// @file
/// @brief Executable, immutable boundary authority prepared before block construction.
///
/// A PreparedBoundaryPlan is the executable native transport-face lowering of one resolved
/// GhostProducerPlan. Resolution and string/Handle dispatch happen once during installation, never
/// in a face-cell loop. The executed order is:
///
///   same-level/MPI + axis-aligned periodic -> physical faces.
///
/// AMR coarse/fine production remains the prepared AMRTransfer authority invoked by AmrRuntime
/// immediately before this plan. Qualified GhostBoundary and FieldBoundaryClosure components are
/// authenticated and prepared at installation, then called as typed bulk tables at the scheduler's
/// exact BoundaryEvaluationPoint. There is no Python callback or runtime component selection.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/boundary_component_executor.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// Native boundary plan captured by every block closure.  Component BCs permit systems with
/// different Dirichlet data per conservative component while the topology (periodic vs physical)
/// remains common to the state.
class PreparedBoundaryPlan {
 public:
  PreparedBoundaryPlan() = default;

  PreparedBoundaryPlan(std::string identity, int required_depth, std::vector<BCRec> component_bc,
                       std::vector<int> omitted_face_ordinals = {},
                       std::string state_identity = {})
      : identity_(std::move(identity)), required_depth_(required_depth),
        component_bc_(std::move(component_bc)), state_identity_(std::move(state_identity)) {
    for (const int face : omitted_face_ordinals) {
      if (face < 0 || face >= 4 || omitted_faces_[static_cast<std::size_t>(face)])
        throw std::invalid_argument(
            "PreparedBoundaryPlan omitted interface faces must be unique ordinals 0..3");
      omitted_faces_[static_cast<std::size_t>(face)] = true;
    }
    validate_base();
  }

  const std::string& identity() const { return identity_; }
  const std::string& state_identity() const { return state_identity_; }
  int required_depth() const { return required_depth_; }
  int ncomp() const { return static_cast<int>(component_bc_.size()); }
  /// Validate that an already allocated state can execute this prepared plan.  Installation-time
  /// consumers use the same invariant as the fill path, without performing or probing a fill.
  void validate_state_layout(const MultiFab& state) const { validate_for(state); }
  bool has_omitted_faces() const {
    return std::any_of(omitted_faces_.begin(), omitted_faces_.end(), [](bool value) {
      return value;
    });
  }
  bool omits_face(int axis, int side) const {
    if (axis < 0 || axis >= 2 || (side != -1 && side != 1))
      throw std::invalid_argument("PreparedBoundaryPlan face selector is invalid");
    return omitted_faces_[static_cast<std::size_t>(2 * axis + (side > 0 ? 1 : 0))];
  }

  const BCRec& component_bc(int comp) const {
    if (comp < 0 || comp >= ncomp())
      throw std::runtime_error("PreparedBoundaryPlan component index out of range");
    return component_bc_[static_cast<std::size_t>(comp)];
  }

  void install_ghost_component(PreparedBoundaryComponentSpec spec,
                               std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(ghost_components_, std::move(spec), std::move(component));
  }
  void install_residual_component(PreparedBoundaryComponentSpec spec,
                                  std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(residual_components_, std::move(spec), std::move(component));
  }
  void install_jvp_component(PreparedBoundaryComponentSpec spec,
                             std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(jvp_components_, std::move(spec), std::move(component));
  }

  bool has_component_boundaries() const {
    return !ghost_components_.empty() || !residual_components_.empty() ||
           !jvp_components_.empty();
  }

  /// Whether this plan owns an executable residual/JVP pair for an implicit operator.  A partial
  /// installation is an invalid native state, never a false capability that can be silently ignored.
  bool has_boundary_linearization() const {
    if (residual_components_.empty() && jvp_components_.empty()) return false;
    std::vector<PreparedBoundaryComponentSpec> residuals;
    std::vector<PreparedBoundaryComponentSpec> jvps;
    residuals.reserve(residual_components_.size());
    jvps.reserve(jvp_components_.size());
    for (const auto& component : residual_components_) residuals.push_back(component->spec());
    for (const auto& component : jvp_components_) jvps.push_back(component->spec());
    validate_linearization_bijection(residuals, jvps);
    return true;
  }

  /// Defensive native authentication for the typed residual/JVP installation. The common
  /// dependency contract must identify exactly one endpoint of each operation; operation-specific
  /// target, direction and output identities are then checked explicitly. No JVP can satisfy two
  /// residuals and no extra JVP can remain installed.
  static void validate_linearization_bijection(
      const std::vector<PreparedBoundaryComponentSpec>& residuals,
      const std::vector<PreparedBoundaryComponentSpec>& jvps) {
    if (residuals.size() != jvps.size())
      throw std::runtime_error(
          "PreparedBoundaryPlan has an unpaired residual/JVP installation");
    std::vector<bool> consumed(jvps.size(), false);
    for (const auto& residual : residuals) {
      validate_linearization_endpoint_(residual, false);
      std::size_t match = jvps.size();
      for (std::size_t index = 0; index < jvps.size(); ++index) {
        if (!same_linearization_contract_(residual, jvps[index])) continue;
        if (match != jvps.size())
          throw std::runtime_error(
              "PreparedBoundaryPlan has duplicate JVPs for one exact residual contract");
        match = index;
      }
      if (match == jvps.size())
        throw std::runtime_error(
            "PreparedBoundaryPlan residual/JVP identities do not form executable pairs");
      validate_linearization_endpoint_(jvps[match], true);
      if (consumed[match])
        throw std::runtime_error(
            "PreparedBoundaryPlan reuses one JVP for multiple residuals");
      if (residual.target_identity == jvps[match].target_identity)
        throw std::runtime_error(
            "PreparedBoundaryPlan residual/JVP targets must be distinct typed identities");
      consumed[match] = true;
    }
    if (std::find(consumed.begin(), consumed.end(), false) != consumed.end())
      throw std::runtime_error("PreparedBoundaryPlan has an orphan JVP installation");
  }

  std::vector<std::string> required_field_identities() const {
    std::vector<std::string> result;
    for (const auto& component : ghost_components_)
      append_unique_(result, component->spec().fields);
    for (const auto& component : residual_components_)
      append_unique_(result, component->spec().fields);
    for (const auto& component : jvp_components_)
      append_unique_(result, component->spec().fields);
    return result;
  }

  std::vector<std::string> required_direction_identities() const {
    std::vector<std::string> result;
    for (const auto& component : jvp_components_)
      append_unique_(result, component->spec().directions);
    return result;
  }

  std::vector<std::string> residual_output_identities() const {
    std::vector<std::string> result;
    for (const auto& component : residual_components_)
      append_unique_(result, component->spec().outputs);
    return result;
  }

  std::vector<std::string> jvp_output_identities() const {
    std::vector<std::string> result;
    for (const auto& component : jvp_components_)
      append_unique_(result, component->spec().outputs);
    return result;
  }

  Periodicity periodicity() const {
    validate_topology();
    const BCRec& bc = component_bc_.front();
    return Periodicity{bc.xlo == BCType::Periodic, bc.ylo == BCType::Periodic};
  }

  /// Same-level/MPI and axis-aligned periodic production are performed by the memoized native halo
  /// schedule. Physical data is then applied per component. AmrRuntime executes the resolved
  /// AMRTransfer coarse/fine producer immediately before this closure on refined levels.
  void fill_same_level_and_physical(MultiFab& state, const Box2D& domain) const {
    if (has_component_boundaries())
      throw std::runtime_error(
          "PreparedBoundaryPlan native components require an exact BoundaryEvaluationPoint");
    validate_for(state);
    fill_boundary(state, domain, periodicity());
    for (int comp = 0; comp < state.ncomp(); ++comp)
      fill_physical_bc(state, domain, component_bc(comp), comp);
  }

  /// Point-qualified production route used by every compiled Program residual.  Built-in faces run
  /// first, then explicitly bound GhostBoundary providers/resolvers overwrite their exact regions.
  void fill_same_level_and_physical(
      MultiFab& state, const MultiFab* auxiliary, const Geometry& geometry,
      const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    validate_for(state);
    fill_boundary(state, geometry.domain, periodicity());
    for (int comp = 0; comp < state.ncomp(); ++comp)
      fill_physical_bc(state, geometry.domain, component_bc(comp), comp);
    for (const auto& component : ghost_components_)
      detail::apply_ghost_component(
          *component, state, auxiliary, geometry, required_depth_, point);
  }

  /// N-ary twin: every qualified dependency is supplied by exact Handle identity.  This is the
  /// extension seam for multi-state/multi-field providers; the one-state overload above is merely a
  /// convenience adapter and never fabricates aliases for missing identities.
  void fill_same_level_and_physical(
      MultiFab& state, const detail::BoundaryFieldRegistry& fields,
      const Geometry& geometry,
      const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    validate_for(state);
    fill_boundary(state, geometry.domain, periodicity());
    for (int comp = 0; comp < state.ncomp(); ++comp)
      fill_physical_bc(state, geometry.domain, component_bc(comp), comp);
    for (const auto& component : ghost_components_)
      detail::apply_ghost_component(
          *component, state, fields, geometry, required_depth_, point);
  }

  /// Add every qualified residual contribution after the volume/face residual has been assembled.
  void add_residual(const runtime::multiblock::BoundaryEvaluationPoint& point,
                    const MultiFab& state, const MultiFab* auxiliary,
                    const Geometry& geometry, MultiFab& residual) const {
    for (const auto& component : residual_components_)
      detail::apply_field_component(
          *component, state, nullptr, auxiliary, geometry, residual, point);
  }

  void add_residual(const runtime::multiblock::BoundaryEvaluationPoint& point,
                    const detail::BoundaryFieldRegistry& fields,
                    const Geometry& geometry) const {
    for (const auto& component : residual_components_)
      detail::apply_field_component(*component, fields, geometry, point);
  }

  /// Exact JVP twin used by implicit solvers; residual/JVP bindings are authenticated as a pair by
  /// the resolved Python IR before either reaches this plan.
  void apply_jvp(const runtime::multiblock::BoundaryEvaluationPoint& point,
                 const MultiFab& state, const MultiFab& direction,
                 const MultiFab* auxiliary, const Geometry& geometry,
                 MultiFab& output) const {
    for (const auto& component : jvp_components_)
      detail::apply_field_component(
          *component, state, &direction, auxiliary, geometry, output, point);
  }

  void apply_jvp(const runtime::multiblock::BoundaryEvaluationPoint& point,
                 const detail::BoundaryFieldRegistry& fields,
                 const Geometry& geometry) const {
    for (const auto& component : jvp_components_)
      detail::apply_field_component(*component, fields, geometry, point);
  }

 private:
  std::string identity_;
  int required_depth_ = 0;
  std::vector<BCRec> component_bc_;
  std::array<bool, 4> omitted_faces_{{false, false, false, false}};
  std::string state_identity_;
  std::vector<std::shared_ptr<PreparedGhostBoundaryComponent>> ghost_components_;
  std::vector<std::shared_ptr<PreparedFieldBoundaryResidualComponent>> residual_components_;
  std::vector<std::shared_ptr<PreparedFieldBoundaryJvpComponent>> jvp_components_;

  template <class Component>
  static void install_typed_(
      std::vector<std::shared_ptr<Component>>& destination,
      PreparedBoundaryComponentSpec spec,
      std::shared_ptr<component::LoadedComponent> component) {
    for (const auto& installed : destination)
      if (installed->spec().target_identity == spec.target_identity &&
          installed->spec().region.identity == spec.region.identity)
        throw std::runtime_error(
            "PreparedBoundaryPlan duplicate typed target/region component binding");
    destination.push_back(std::make_shared<Component>(
        std::move(spec), std::move(component)));
  }

  static void append_unique_(std::vector<std::string>& destination,
                             const std::vector<std::string>& source) {
    for (const auto& identity : source)
      if (std::find(destination.begin(), destination.end(), identity) == destination.end())
        destination.push_back(identity);
    std::sort(destination.begin(), destination.end());
  }

  static bool same_region_(const PreparedBoundaryRegion& left,
                           const PreparedBoundaryRegion& right) {
    return left.kind == right.kind && left.dimension == right.dimension &&
           left.codimension == right.codimension && left.axes == right.axes &&
           left.sides == right.sides && left.identity == right.identity;
  }

  static bool same_linearization_contract_(const PreparedBoundaryComponentSpec& residual,
                                           const PreparedBoundaryComponentSpec& jvp) {
    return residual.component_id == jvp.component_id &&
           residual.manifest_identity == jvp.manifest_identity &&
           residual.interface_version == jvp.interface_version &&
           residual.producer_identity == jvp.producer_identity &&
           residual.state_identity == jvp.state_identity &&
           residual.ghost_identity == jvp.ghost_identity &&
           residual.layout_identity == jvp.layout_identity &&
           same_region_(residual.region, jvp.region) && residual.states == jvp.states &&
           residual.fields == jvp.fields && residual.parameter_ids == jvp.parameter_ids &&
           residual.parameter_values == jvp.parameter_values && residual.rate == jvp.rate &&
           residual.nonlinear_iterate == jvp.nonlinear_iterate &&
           residual.parameters_json == jvp.parameters_json &&
           residual.target_json == jvp.target_json &&
           same_execution_(residual.execution, jvp.execution);
  }

  static bool same_execution_(
      const std::shared_ptr<const component::PreparedExecutionContextV1>& left,
      const std::shared_ptr<const component::PreparedExecutionContextV1>& right) {
    if (!left || !right) return !left && !right;
    const PopsExecutionContextV1 a = left->view();
    const PopsExecutionContextV1 b = right->view();
    const auto same_text = [](const char* x, const char* y) {
      return x != nullptr && y != nullptr && std::string_view(x) == std::string_view(y);
    };
    return a.context_version == b.context_version &&
           same_text(a.execution_identity, b.execution_identity) &&
           a.memory_space == b.memory_space &&
           same_text(a.backend_identity, b.backend_identity) &&
           same_text(a.device_identity, b.device_identity) &&
           a.scalar_type == b.scalar_type && a.storage_precision == b.storage_precision &&
           a.compute_precision == b.compute_precision &&
           a.accumulation_precision == b.accumulation_precision &&
           a.reduction_precision == b.reduction_precision &&
           a.stream_handle == b.stream_handle &&
           same_text(a.stream_identity, b.stream_identity) &&
           a.communicator_f_handle == b.communicator_f_handle &&
           a.communicator_datatype_f_handle == b.communicator_datatype_f_handle &&
           same_text(a.communicator_identity, b.communicator_identity) &&
           same_text(a.communicator_datatype_identity, b.communicator_datatype_identity);
  }

  static void validate_linearization_endpoint_(const PreparedBoundaryComponentSpec& spec,
                                               bool jvp) {
    if (spec.target_identity.empty() || spec.outputs.size() != 1 ||
        spec.outputs.front().empty())
      throw std::runtime_error(
          "PreparedBoundaryPlan residual/JVP target and output identities must be exact");
    const std::vector<std::string> expected_directions =
        jvp ? std::vector<std::string>{spec.state_identity} : std::vector<std::string>{};
    if (spec.directions != expected_directions)
      throw std::runtime_error(
          "PreparedBoundaryPlan residual/JVP direction identities are not executable");
  }

  static std::array<BCType, 4> face_types(const BCRec& bc) {
    return {bc.xlo, bc.xhi, bc.ylo, bc.yhi};
  }

  void validate_base() const {
    if (identity_.empty())
      throw std::runtime_error("PreparedBoundaryPlan requires a canonical identity");
    if (required_depth_ < 1)
      throw std::runtime_error("PreparedBoundaryPlan required depth must be >= 1");
    if (component_bc_.empty())
      throw std::runtime_error("PreparedBoundaryPlan requires one BC record per component");
    validate_topology();
  }

  void validate_topology() const {
    if (component_bc_.empty())
      throw std::runtime_error("PreparedBoundaryPlan has no component BCs");
    const auto expected = face_types(component_bc_.front());
    for (std::size_t comp = 1; comp < component_bc_.size(); ++comp) {
      const auto actual = face_types(component_bc_[comp]);
      for (std::size_t face = 0; face < actual.size(); ++face) {
        const bool expected_periodic = expected[face] == BCType::Periodic;
        const bool actual_periodic = actual[face] == BCType::Periodic;
        if (expected_periodic != actual_periodic)
          throw std::runtime_error(
              "PreparedBoundaryPlan periodic/physical topology differs between components");
      }
    }
    if ((expected[0] == BCType::Periodic) != (expected[1] == BCType::Periodic) ||
        (expected[2] == BCType::Periodic) != (expected[3] == BCType::Periodic))
      throw std::runtime_error(
          "axis-aligned PreparedBoundaryPlan requires periodic faces in complete axis pairs");
  }

  void validate_for(const MultiFab& state) const {
    if (state.ncomp() != ncomp())
      throw std::runtime_error("PreparedBoundaryPlan component count does not match block state");
    if (state.n_grow() < required_depth_)
      throw std::runtime_error("PreparedBoundaryPlan stencil depth exceeds allocated ghosts");
  }
};

}  // namespace pops
