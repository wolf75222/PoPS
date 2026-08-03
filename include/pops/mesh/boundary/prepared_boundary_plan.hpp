/// @file
/// @brief Executable boundary authority finalized before numerical execution.
///
/// A PreparedBoundaryPlan is the executable native transport-face lowering of one resolved
/// GhostProducerPlan. Resolution, one-time model conversion, component preparation and
/// string/Handle dispatch happen before a session is retained, never in a face-cell loop. The
/// executed order is:
///
///   same-level/MPI + prepared periodic identifications -> physical faces.
///
/// AMR coarse/fine production remains the prepared AMRTransfer authority invoked by AmrRuntime
/// immediately before this plan. Qualified GhostBoundary and FieldBoundaryClosure components are
/// authenticated at installation and prepared into one private state per ExecutionLane session,
/// then called as typed bulk tables at the scheduler's exact BoundaryEvaluationPoint. There is no
/// Python callback or runtime component selection.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/boundary_component_executor.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

class PreparedBoundaryPlan;
class PreparedBoundaryReadView;

/// Opaque slot for one exact state dependency of an immutable prepared-plan revision.
class PreparedBoundaryStateRead final {
 public:
  PreparedBoundaryStateRead(const PreparedBoundaryStateRead&) = default;
  PreparedBoundaryStateRead& operator=(const PreparedBoundaryStateRead&) = default;

 private:
  friend class PreparedBoundaryPlan;
  friend class PreparedBoundaryReadView;

  PreparedBoundaryStateRead(const PreparedBoundaryPlan* owner, std::size_t slot,
                            std::size_t component_revision)
      : owner_(owner), slot_(slot), component_revision_(component_revision) {}

  const PreparedBoundaryPlan* owner_;
  std::size_t slot_;
  std::size_t component_revision_;
};

/// Opaque slot for one exact solved-field dependency of an immutable prepared-plan revision.
class PreparedBoundaryFieldRead final {
 public:
  PreparedBoundaryFieldRead(const PreparedBoundaryFieldRead&) = default;
  PreparedBoundaryFieldRead& operator=(const PreparedBoundaryFieldRead&) = default;

 private:
  friend class PreparedBoundaryPlan;
  friend class PreparedBoundaryReadView;

  PreparedBoundaryFieldRead(const PreparedBoundaryPlan* owner, std::size_t slot,
                            std::size_t component_revision)
      : owner_(owner), slot_(slot), component_revision_(component_revision) {}

  const PreparedBoundaryPlan* owner_;
  std::size_t slot_;
  std::size_t component_revision_;
};

/// Exact read-only storage dependencies owned directly by a boundary plan.
///
/// Native block closures may consume qualified states or fields even when that read is not owned
/// by a dynamically loaded boundary component.  Declaring those reads here keeps the AMR/System
/// storage registry closed and allocation-free without reverting to the historical "bind every
/// state" behavior.
struct PreparedBoundaryReadDependencies {
  std::vector<std::string> states;
  std::vector<std::string> fields;
};

namespace detail {

inline void append_boundary_request_u64(std::string& payload, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    payload.push_back(static_cast<char>((value >> shift) & UINT64_C(0xff)));
}

inline void append_boundary_request_size(std::string& payload, std::size_t value) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
      throw std::length_error("boundary request exceeds canonical uint64 length capacity");
  }
  append_boundary_request_u64(payload, static_cast<std::uint64_t>(value));
}

inline void append_boundary_request_bytes(std::string& payload, std::string_view value) {
  append_boundary_request_size(payload, value.size());
  if (!value.empty())
    payload.append(value.data(), value.size());
}

inline void append_boundary_request_int(std::string& payload, int value) {
  append_boundary_request_u64(payload,
                              static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
}

inline void append_boundary_request_strings(std::string& payload,
                                            const std::vector<std::string>& values) {
  append_boundary_request_size(payload, values.size());
  for (const auto& value : values)
    append_boundary_request_bytes(payload, value);
}

inline void append_boundary_request_reals(std::string& payload, const std::vector<double>& values) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559,
                "boundary request consensus requires IEEE-754 binary64");
  append_boundary_request_size(payload, values.size());
  for (double value : values)
    append_boundary_request_u64(payload, std::bit_cast<std::uint64_t>(value));
}

/// Exact byte identity of every argument that can affect one prepared boundary plan.  Analytic
/// opcode/literal rows are included here as well as fixed-state and topology metadata so a
/// collectively prepared plan cannot diverge through a non-program field.
inline std::string canonical_prepared_boundary_plan_request(
    std::string_view name, std::string_view identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles,
    const std::vector<int>& omitted_interface_faces, std::string_view state_identity,
    const PreparedBoundaryReadDependencies& read_dependencies,
    const std::vector<PeriodicIdentification2D>& periodic_identifications,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  std::string payload;
  append_boundary_request_bytes(payload, "pops.prepared-boundary-plan.request.v1");
  append_boundary_request_bytes(payload, name);
  append_boundary_request_bytes(payload, identity);
  append_boundary_request_int(payload, required_depth);
  append_boundary_request_strings(payload, face_types);
  append_boundary_request_reals(payload, face_values);
  append_boundary_request_strings(payload, face_identities);
  append_boundary_request_strings(payload, component_roles);
  append_boundary_request_size(payload, omitted_interface_faces.size());
  for (int face : omitted_interface_faces)
    append_boundary_request_int(payload, face);
  append_boundary_request_bytes(payload, state_identity);
  append_boundary_request_strings(payload, read_dependencies.states);
  append_boundary_request_strings(payload, read_dependencies.fields);
  append_boundary_request_size(payload, periodic_identifications.size());
  for (const auto& periodic : periodic_identifications) {
    append_boundary_request_int(payload, periodic.source_face);
    append_boundary_request_int(payload, periodic.target_face);
    for (int axis : periodic.permutation)
      append_boundary_request_int(payload, axis);
    for (int sign : periodic.signs)
      append_boundary_request_int(payload, sign);
  }
  append_boundary_request_strings(payload, face_representations);
  append_boundary_request_strings(payload, face_converter_identities);
  append_boundary_request_size(payload, face_analytic_opcodes.size());
  for (const auto& row : face_analytic_opcodes)
    append_boundary_request_strings(payload, row);
  append_boundary_request_size(payload, face_analytic_literals.size());
  for (const auto& row : face_analytic_literals)
    append_boundary_request_reals(payload, row);
  append_boundary_request_strings(payload, face_analytic_clocks);
  return payload;
}

}  // namespace detail

/// Native boundary plan captured by every block closure. The built-in physical-face authority is
/// one model-aware hyperbolic plan; field/elliptic BCRec data is not a transport semantic here.
class PreparedBoundaryPlan {
 private:
  struct BoundaryRecoveryWorkspace {
    std::vector<Real> snapshot;
    std::vector<double> conserved;
    std::vector<double> primitive;
    bool prepared = false;
  };

 public:
  using CharacteristicNoInflowFill = std::function<void(MultiFab&, const Box2D&, CommunicatorView)>;
  /// Move-only, lane-bound executable state for this immutable plan.
  ///
  /// Session construction is the sole materialization point for component-owned native state. Its
  /// invocation methods perform no component preparation, lookup or lazy cache insertion.
  class Session final {
   public:
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept
        : plan_(std::exchange(other.plan_, nullptr)),
          lane_(std::exchange(other.lane_, nullptr)),
          component_revision_(std::exchange(other.component_revision_, 0)),
          ghost_components_(std::move(other.ghost_components_)),
          flux_components_(std::move(other.flux_components_)),
          residual_components_(std::move(other.residual_components_)),
          jvp_components_(std::move(other.jvp_components_)),
          ghost_workspaces_(std::move(other.ghost_workspaces_)),
          flux_workspaces_(std::move(other.flux_workspaces_)),
          residual_workspaces_(std::move(other.residual_workspaces_)),
          jvp_workspaces_(std::move(other.jvp_workspaces_)),
          recovery_workspace_(std::move(other.recovery_workspace_)) {}
    Session& operator=(Session&& other) noexcept {
      if (this != &other) {
        plan_ = std::exchange(other.plan_, nullptr);
        lane_ = std::exchange(other.lane_, nullptr);
        component_revision_ = std::exchange(other.component_revision_, 0);
        ghost_components_ = std::move(other.ghost_components_);
        flux_components_ = std::move(other.flux_components_);
        residual_components_ = std::move(other.residual_components_);
        jvp_components_ = std::move(other.jvp_components_);
        ghost_workspaces_ = std::move(other.ghost_workspaces_);
        flux_workspaces_ = std::move(other.flux_workspaces_);
        residual_workspaces_ = std::move(other.residual_workspaces_);
        jvp_workspaces_ = std::move(other.jvp_workspaces_);
        recovery_workspace_ = std::move(other.recovery_workspace_);
      }
      return *this;
    }
    ~Session() = default;

    /// Refuse use after the owning plan's component table changed or after this session moved.
    void validate_current() const { validate_current_(); }

    void fill_same_level_and_physical(MultiFab& state, const Box2D& domain) const;
    void fill_same_level_and_physical(MultiFab& state, const Geometry& geometry) const;
    void fill_same_level_and_physical(
        MultiFab& state, const Geometry& geometry,
        const runtime::multiblock::BoundaryEvaluationPoint& point) const;
    void fill_same_level_and_physical(
        MultiFab& state, const detail::BoundaryFieldRegistry& fields, const Geometry& geometry,
        const runtime::multiblock::BoundaryEvaluationPoint& point) const;
    void transform_fluxes(const runtime::multiblock::BoundaryEvaluationPoint& point,
                          const MultiFab& state, const detail::BoundaryFieldRegistry& fields,
                          const Geometry& geometry, MultiFab& fx, MultiFab& fy) const;
    void add_residual(const runtime::multiblock::BoundaryEvaluationPoint& point,
                      const detail::BoundaryFieldRegistry& fields, const Geometry& geometry) const;
    void apply_jvp(const runtime::multiblock::BoundaryEvaluationPoint& point,
                   const detail::BoundaryFieldRegistry& fields, const Geometry& geometry) const;

    /// Materialize all geometry/layout-dependent executor storage before entering a numerical loop.
    /// The registry is already bound to exact storage for the corresponding operation, so every
    /// string-to-slot resolution and every vector allocation happens here, not in fill/apply.
    void prepare_ghost_executor(const MultiFab& prototype,
                                const detail::BoundaryFieldRegistry& fields,
                                const Geometry& geometry);
    void prepare_flux_executor(const MultiFab& prototype,
                               const detail::BoundaryFieldRegistry& fields,
                               const Geometry& geometry);
    void prepare_residual_executor(const detail::BoundaryFieldRegistry& fields,
                                   const Geometry& geometry);
    void prepare_jvp_executor(const detail::BoundaryFieldRegistry& fields,
                              const Geometry& geometry);
    /// Allocate the exact lane-private transaction storage used to validate physical ghost traces.
    /// Production GridContext sessions call this once while binding their prototype, never lazily
    /// from a numerical fill.
    void prepare_trace_recovery_workspace(const MultiFab& prototype);

   private:
    friend class PreparedBoundaryPlan;

    Session(const PreparedBoundaryPlan& plan, const ExecutionLane& lane);
    void validate_current_() const;
    // One-shot adapters are private and explicitly control-only.  Production callers bind the
    // stable registry, prepare executor workspaces once and use the public registry overloads.
    void fill_same_level_and_physical_control(
        MultiFab& state, const MultiFab* auxiliary, const Geometry& geometry,
        const runtime::multiblock::BoundaryEvaluationPoint& point) const;
    void transform_fluxes_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  const MultiFab& state, const MultiFab* auxiliary,
                                  const Geometry& geometry, MultiFab& fx, MultiFab& fy) const;
    void add_residual_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                              const MultiFab& state, const MultiFab* auxiliary,
                              const Geometry& geometry, MultiFab& residual) const;
    void apply_jvp_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                           const MultiFab& state, const MultiFab& direction,
                           const MultiFab* auxiliary, const Geometry& geometry,
                           MultiFab& output) const;

    const PreparedBoundaryPlan* plan_ = nullptr;
    const ExecutionLane* lane_ = nullptr;
    std::size_t component_revision_ = 0;
    std::vector<PreparedGhostBoundaryComponent::Session> ghost_components_;
    std::vector<PreparedBoundaryFluxComponent::Session> flux_components_;
    std::vector<PreparedFieldBoundaryResidualComponent::Session> residual_components_;
    std::vector<PreparedFieldBoundaryJvpComponent::Session> jvp_components_;
    mutable std::vector<detail::PreparedGhostBoundaryWorkspace> ghost_workspaces_;
    mutable std::vector<detail::PreparedBoundaryFluxWorkspace> flux_workspaces_;
    mutable std::vector<detail::PreparedFieldBoundaryWorkspace> residual_workspaces_;
    mutable std::vector<detail::PreparedFieldBoundaryWorkspace> jvp_workspaces_;
    mutable BoundaryRecoveryWorkspace recovery_workspace_;
  };

  PreparedBoundaryPlan() = default;

  PreparedBoundaryPlan(std::string identity, int required_depth,
                       PreparedHyperbolicBoundary<2> hyperbolic_boundary,
                       std::vector<int> omitted_face_ordinals = {}, std::string state_identity = {},
                       PreparedBoundaryReadDependencies read_dependencies = {},
                       std::vector<PeriodicIdentification2D> periodic_identifications = {})
      : identity_(std::move(identity)),
        required_depth_(required_depth),
        hyperbolic_boundary_(std::move(hyperbolic_boundary)),
        state_identity_(std::move(state_identity)),
        read_dependencies_(std::move(read_dependencies)),
        periodic_identifications_(std::move(periodic_identifications)) {
    for (const int face : omitted_face_ordinals) {
      if (face < 0 || face >= 4 || omitted_faces_[static_cast<std::size_t>(face)])
        throw std::invalid_argument(
            "PreparedBoundaryPlan omitted interface faces must be unique ordinals 0..3");
      omitted_faces_[static_cast<std::size_t>(face)] = true;
    }
    for (int face = 0; face < 4; ++face)
      if (omitted_faces_[static_cast<std::size_t>(face)] &&
          (hyperbolic_boundary_.face(face / 2, face % 2 == 0 ? -1 : 1).law ==
               HyperbolicBoundaryLaw::NoFlux ||
           hyperbolic_boundary_.face(face / 2, face % 2 == 0 ? -1 : 1).law ==
               HyperbolicBoundaryLaw::CharacteristicNoInflow))
        throw std::invalid_argument(
            "a prepared interface face cannot also be a physical no-flux/characteristic boundary");
    validate_base();
  }

  const std::string& identity() const { return identity_; }
  const std::string& state_identity() const { return state_identity_; }
  int required_depth() const { return required_depth_; }
  int ncomp() const { return hyperbolic_boundary_.ncomp(); }
  const PreparedHyperbolicBoundary<2>& hyperbolic_boundary() const { return hyperbolic_boundary_; }
  bool requires_fixed_state_conversion() const {
    return hyperbolic_boundary_.requires_fixed_state_conversion();
  }
  /// Complete one model-dependent preparation step before any execution session is retained.
  ///
  /// The revision increment invalidates any session or dependency token created too early instead
  /// of allowing it to observe a changed numerical table.
  void prepare_fixed_state_conversion(
      const std::function<void(const double*, double*)>& primitive_to_conservative) {
    if (!requires_fixed_state_conversion())
      return;
    hyperbolic_boundary_ =
        hyperbolic_boundary_.with_converted_fixed_states(primitive_to_conservative);
    ++component_revision_;
  }
  /// Attach the exact block-model conservative-to-primitive recovery used to authenticate every
  /// produced physical ghost trace before it becomes visible to a reconstruction kernel.
  void prepare_trace_recovery(
      std::function<RecoveryReport(const double*, double*)> conservative_to_primitive) {
    if (!conservative_to_primitive)
      throw std::invalid_argument(
          "PreparedBoundaryPlan trace recovery requires a prepared block-model authority");
    if (trace_recovery_)
      throw std::logic_error("PreparedBoundaryPlan trace recovery authority is already finalized");
    trace_recovery_ = std::move(conservative_to_primitive);
    ++component_revision_;
  }
  bool has_trace_recovery() const noexcept { return static_cast<bool>(trace_recovery_); }
  bool requires_characteristic_no_inflow() const noexcept {
    return hyperbolic_boundary_.has_characteristic_no_inflow();
  }
  void prepare_characteristic_no_inflow(CharacteristicNoInflowFill fill) {
    if (!requires_characteristic_no_inflow())
      throw std::logic_error(
          "PreparedBoundaryPlan cannot install an unrequested characteristic provider");
    if (!fill)
      throw std::invalid_argument(
          "PreparedBoundaryPlan characteristic no-inflow requires an executable model provider");
    if (characteristic_no_inflow_fill_)
      throw std::logic_error(
          "PreparedBoundaryPlan characteristic no-inflow provider is already finalized");
    characteristic_no_inflow_fill_ = std::move(fill);
    ++component_revision_;
  }
  const std::vector<PeriodicIdentification2D>& periodic_identifications() const noexcept {
    return periodic_identifications_;
  }
  bool has_mapped_periodicity() const noexcept { return has_mapped_periodicity_(); }
  std::optional<Periodicity> axis_aligned_periodicity() const noexcept {
    const bool xlo = hyperbolic_boundary_.face(0, -1).law == HyperbolicBoundaryLaw::Periodic;
    const bool xhi = hyperbolic_boundary_.face(0, 1).law == HyperbolicBoundaryLaw::Periodic;
    const bool ylo = hyperbolic_boundary_.face(1, -1).law == HyperbolicBoundaryLaw::Periodic;
    const bool yhi = hyperbolic_boundary_.face(1, 1).law == HyperbolicBoundaryLaw::Periodic;
    if (xlo != xhi || ylo != yhi)
      return std::nullopt;
    return Periodicity{xlo, ylo};
  }
  /// Validate that an already allocated state can execute this prepared plan.  Installation-time
  /// consumers use the same invariant as the fill path, without performing or probing a fill.
  void validate_state_layout(const MultiFab& state) const { validate_for(state); }
  bool has_omitted_faces() const {
    return std::any_of(omitted_faces_.begin(), omitted_faces_.end(),
                       [](bool value) { return value; });
  }
  bool has_zero_flux_faces() const noexcept {
    for (int axis = 0; axis < 2; ++axis)
      for (const int side : {-1, 1})
        if (zeroes_face(axis, side))
          return true;
    return false;
  }
  bool zeroes_face(int axis, int side) const {
    if (axis < 0 || axis >= 2 || (side != -1 && side != 1))
      throw std::invalid_argument("PreparedBoundaryPlan face selector is invalid");
    return hyperbolic_boundary_.face(axis, side).law == HyperbolicBoundaryLaw::NoFlux;
  }
  bool omits_face(int axis, int side) const {
    if (axis < 0 || axis >= 2 || (side != -1 && side != 1))
      throw std::invalid_argument("PreparedBoundaryPlan face selector is invalid");
    return omitted_faces_[static_cast<std::size_t>(2 * axis + (side > 0 ? 1 : 0))];
  }

  void install_ghost_component(PreparedBoundaryComponentSpec spec,
                               std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(ghost_components_, std::move(spec), std::move(component));
    ++component_revision_;
  }
  void install_flux_component(PreparedBoundaryComponentSpec spec,
                              std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(flux_components_, std::move(spec), std::move(component));
    ++component_revision_;
  }
  void install_residual_component(PreparedBoundaryComponentSpec spec,
                                  std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(residual_components_, std::move(spec), std::move(component));
    ++component_revision_;
  }
  void install_jvp_component(PreparedBoundaryComponentSpec spec,
                             std::shared_ptr<component::LoadedComponent> component) {
    install_typed_(jvp_components_, std::move(spec), std::move(component));
    ++component_revision_;
  }

  /// Materialize all component states once for a numerical execution lane.
  [[nodiscard]] Session make_session(const ExecutionLane& lane) const {
    return Session(*this, lane);
  }

  bool has_component_boundaries() const {
    return !ghost_components_.empty() || !flux_components_.empty() ||
           !residual_components_.empty() || !jvp_components_.empty();
  }
  bool has_flux_transformations() const noexcept { return !flux_components_.empty(); }

  /// The built-in hyperbolic laws fill every ghost layer allocated by the state. A dynamically
  /// loaded ghost component is prepared only for this plan's authenticated required_depth(), so it
  /// keeps the bounded-depth contract even though residual/JVP-only components do not affect fill.
  bool fills_all_allocated_physical_ghosts() const noexcept { return ghost_components_.empty(); }

  /// Whether this plan owns an executable residual/JVP pair for an implicit operator.  A partial
  /// installation is an invalid native state, never a false capability that can be silently ignored.
  bool has_boundary_linearization() const {
    if (residual_components_.empty() && jvp_components_.empty())
      return false;
    std::vector<PreparedBoundaryComponentSpec> residuals;
    std::vector<PreparedBoundaryComponentSpec> jvps;
    residuals.reserve(residual_components_.size());
    jvps.reserve(jvp_components_.size());
    for (const auto& component : residual_components_)
      residuals.push_back(component->spec());
    for (const auto& component : jvp_components_)
      jvps.push_back(component->spec());
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
      throw std::runtime_error("PreparedBoundaryPlan has an unpaired residual/JVP installation");
    std::vector<bool> consumed(jvps.size(), false);
    for (const auto& residual : residuals) {
      validate_linearization_endpoint_(residual, false);
      std::size_t match = jvps.size();
      for (std::size_t index = 0; index < jvps.size(); ++index) {
        if (!same_linearization_contract_(residual, jvps[index]))
          continue;
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
        throw std::runtime_error("PreparedBoundaryPlan reuses one JVP for multiple residuals");
      if (residual.target_identity == jvps[match].target_identity)
        throw std::runtime_error(
            "PreparedBoundaryPlan residual/JVP targets must be distinct typed identities");
      consumed[match] = true;
    }
    if (std::find(consumed.begin(), consumed.end(), false) != consumed.end())
      throw std::runtime_error("PreparedBoundaryPlan has an orphan JVP installation");
  }

  std::vector<std::string> required_field_identities() const {
    std::vector<std::string> result = read_dependencies_.fields;
    for (const auto& component : ghost_components_)
      append_unique_(result, component->spec().fields);
    for (const auto& component : flux_components_)
      append_unique_(result, component->spec().fields);
    for (const auto& component : residual_components_)
      append_unique_(result, component->spec().fields);
    for (const auto& component : jvp_components_)
      append_unique_(result, component->spec().fields);
    return result;
  }

  std::vector<std::string> required_state_identities() const {
    std::vector<std::string> result = read_dependencies_.states;
    for (const auto& component : ghost_components_)
      append_unique_(result, component->spec().states);
    for (const auto& component : flux_components_)
      append_unique_(result, component->spec().states);
    for (const auto& component : residual_components_)
      append_unique_(result, component->spec().states);
    for (const auto& component : jvp_components_)
      append_unique_(result, component->spec().states);
    return result;
  }

  /// Resolve a qualified state dependency once while constructing a native closure. The returned
  /// token is owner- and revision-bound, so a session cannot silently consume a same-index slot
  /// from another plan or from a subsequently modified component table.
  [[nodiscard]] PreparedBoundaryStateRead prepare_state_read(std::string_view identity) const {
    const auto identities = required_state_identities();
    const auto found = std::find(identities.begin(), identities.end(), identity);
    if (identity.empty() || found == identities.end())
      throw std::invalid_argument("PreparedBoundaryPlan has no declared state read dependency '" +
                                  std::string(identity) + "'");
    return PreparedBoundaryStateRead(
        this, static_cast<std::size_t>(std::distance(identities.begin(), found)),
        component_revision_);
  }

  /// Field twin of prepare_state_read().
  [[nodiscard]] PreparedBoundaryFieldRead prepare_field_read(std::string_view identity) const {
    const auto identities = required_field_identities();
    const auto found = std::find(identities.begin(), identities.end(), identity);
    if (identity.empty() || found == identities.end())
      throw std::invalid_argument("PreparedBoundaryPlan has no declared field read dependency '" +
                                  std::string(identity) + "'");
    return PreparedBoundaryFieldRead(
        this, static_cast<std::size_t>(std::distance(identities.begin(), found)),
        component_revision_);
  }

  std::vector<std::string> all_output_identities() const {
    auto result = residual_output_identities();
    append_unique_(result, jvp_output_identities());
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
    const auto result = axis_aligned_periodicity();
    if (!result)
      throw std::logic_error(
          "axis-permuted periodic topology has no per-axis runtime Periodicity projection");
    return *result;
  }

  /// Same-level/MPI and prepared periodic production are performed by the memoized native halo
  /// schedule. Physical data is then applied per component. AmrRuntime executes the resolved
  /// AMRTransfer coarse/fine producer immediately before this closure on refined levels.
  void fill_same_level_and_physical(MultiFab& state, const Box2D& domain) const {
    if (has_component_boundaries())
      throw std::runtime_error(
          "PreparedBoundaryPlan native components require an exact BoundaryEvaluationPoint");
    validate_for(state);
    auto physical_preflight = hyperbolic_boundary_.preflight_physical(state, domain);
    BoundaryRecoveryWorkspace workspace;
    fill_with_trace_recovery_transaction_(
        state, domain, world_communicator_view(), workspace, false, [&] {
          fill_native_halos_(state, domain);
          fill_prepared_physical_(state, domain, world_communicator_view(),
                                  std::move(physical_preflight));
        });
  }

  void fill_same_level_and_physical(MultiFab& state, const Box2D& domain,
                                    const ExecutionLane& lane) const {
    if (has_component_boundaries())
      throw std::runtime_error(
          "PreparedBoundaryPlan native components require an exact BoundaryEvaluationPoint");
    validate_for(state);
    auto physical_preflight = hyperbolic_boundary_.preflight_physical(state, domain);
    BoundaryRecoveryWorkspace workspace;
    fill_with_trace_recovery_transaction_(
        state, domain, lane.communicator(), workspace, false, [&] {
          fill_native_halos_(state, domain, lane);
          fill_prepared_physical_(state, domain, lane.communicator(),
                                  std::move(physical_preflight));
        });
  }

  void fill_same_level_and_physical(MultiFab& state, const Geometry& geometry) const {
    if (has_component_boundaries())
      throw std::runtime_error(
          "PreparedBoundaryPlan native components require an exact BoundaryEvaluationPoint");
    validate_for(state);
    auto physical_preflight = hyperbolic_boundary_.preflight_physical(state, geometry);
    BoundaryRecoveryWorkspace workspace;
    fill_with_trace_recovery_transaction_(
        state, geometry.domain, world_communicator_view(), workspace, false, [&] {
          fill_native_halos_(state, geometry.domain);
          fill_prepared_physical_(state, geometry.domain, world_communicator_view(),
                                  std::move(physical_preflight));
        });
  }

  void fill_same_level_and_physical(MultiFab& state, const Geometry& geometry,
                                    const ExecutionLane& lane) const {
    if (has_component_boundaries())
      throw std::runtime_error(
          "PreparedBoundaryPlan native components require an exact BoundaryEvaluationPoint");
    validate_for(state);
    auto physical_preflight =
        hyperbolic_boundary_.preflight_physical(state, geometry, lane.communicator());
    BoundaryRecoveryWorkspace workspace;
    fill_with_trace_recovery_transaction_(
        state, geometry.domain, lane.communicator(), workspace, false, [&] {
          fill_native_halos_(state, geometry.domain, lane);
          fill_prepared_physical_(state, geometry.domain, lane.communicator(),
                                  std::move(physical_preflight));
        });
  }

  /// One-shot control/diagnostic adapter. It materializes a fresh component session and workspace;
  /// compiled numerical routes retain Session and must not call this method.
  void fill_same_level_and_physical_control(
      MultiFab& state, const MultiFab* auxiliary, const Geometry& geometry,
      const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    const auto lane = ExecutionLane::world(identity_, "::boundary-control");
    auto session = make_session(lane);
    session.prepare_trace_recovery_workspace(state);
    session.fill_same_level_and_physical_control(state, auxiliary, geometry, point);
  }

  void fill_same_level_and_physical_control(
      MultiFab& state, const MultiFab* auxiliary, const Geometry& geometry,
      const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
    // Honest control-path convenience: callers that execute repeatedly retain make_session(lane)
    // and invoke it directly, avoiding preparation and allocation in the numerical hot path.
    auto session = make_session(lane);
    session.prepare_trace_recovery_workspace(state);
    session.fill_same_level_and_physical_control(state, auxiliary, geometry, point);
  }

  /// N-ary control twin: every qualified dependency is supplied by exact Handle identity.  The
  /// one-state overload above remains a convenience adapter and never fabricates aliases.
  void fill_same_level_and_physical_control(
      MultiFab& state, const detail::BoundaryFieldRegistry& fields, const Geometry& geometry,
      const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    const auto lane = ExecutionLane::world(identity_, "::boundary-control");
    auto session = make_session(lane);
    session.prepare_trace_recovery_workspace(state);
    session.prepare_ghost_executor(state, fields, geometry);
    session.fill_same_level_and_physical(state, fields, geometry, point);
  }

  void fill_same_level_and_physical_control(
      MultiFab& state, const detail::BoundaryFieldRegistry& fields, const Geometry& geometry,
      const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
    auto session = make_session(lane);
    session.prepare_trace_recovery_workspace(state);
    session.prepare_ghost_executor(state, fields, geometry);
    session.fill_same_level_and_physical(state, fields, geometry, point);
  }

  /// One-shot control adapter for the qualified residual contribution.
  void add_residual_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            const MultiFab& state, const MultiFab* auxiliary,
                            const Geometry& geometry, MultiFab& residual) const {
    const auto lane = ExecutionLane::world(identity_, "::boundary-control");
    auto session = make_session(lane);
    session.add_residual_control(point, state, auxiliary, geometry, residual);
  }

  void add_residual_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            const MultiFab& state, const MultiFab* auxiliary,
                            const Geometry& geometry, MultiFab& residual,
                            const ExecutionLane& lane) const {
    auto session = make_session(lane);
    session.add_residual_control(point, state, auxiliary, geometry, residual);
  }

  void add_residual_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            const detail::BoundaryFieldRegistry& fields,
                            const Geometry& geometry) const {
    const auto lane = ExecutionLane::world(identity_, "::boundary-control");
    auto session = make_session(lane);
    session.prepare_residual_executor(fields, geometry);
    session.add_residual(point, fields, geometry);
  }

  void add_residual_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            const detail::BoundaryFieldRegistry& fields, const Geometry& geometry,
                            const ExecutionLane& lane) const {
    auto session = make_session(lane);
    session.prepare_residual_executor(fields, geometry);
    session.add_residual(point, fields, geometry);
  }

  /// One-shot control adapter for an exact JVP; production implicit solvers use a retained Session.
  void apply_jvp_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                         const MultiFab& state, const MultiFab& direction,
                         const MultiFab* auxiliary, const Geometry& geometry,
                         MultiFab& output) const {
    const auto lane = ExecutionLane::world(identity_, "::boundary-control");
    auto session = make_session(lane);
    session.apply_jvp_control(point, state, direction, auxiliary, geometry, output);
  }

  void apply_jvp_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                         const MultiFab& state, const MultiFab& direction,
                         const MultiFab* auxiliary, const Geometry& geometry, MultiFab& output,
                         const ExecutionLane& lane) const {
    auto session = make_session(lane);
    session.apply_jvp_control(point, state, direction, auxiliary, geometry, output);
  }

  void apply_jvp_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                         const detail::BoundaryFieldRegistry& fields,
                         const Geometry& geometry) const {
    const auto lane = ExecutionLane::world(identity_, "::boundary-control");
    auto session = make_session(lane);
    session.prepare_jvp_executor(fields, geometry);
    session.apply_jvp(point, fields, geometry);
  }

  void transform_fluxes_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                const MultiFab& state, const MultiFab* auxiliary,
                                const Geometry& geometry, MultiFab& fx, MultiFab& fy,
                                const ExecutionLane& lane = ExecutionLane::world()) const {
    if (!has_flux_transformations())
      return;
    auto session = make_session(lane);
    session.transform_fluxes_control(point, state, auxiliary, geometry, fx, fy);
  }

  void apply_jvp_control(const runtime::multiblock::BoundaryEvaluationPoint& point,
                         const detail::BoundaryFieldRegistry& fields, const Geometry& geometry,
                         const ExecutionLane& lane) const {
    auto session = make_session(lane);
    session.prepare_jvp_executor(fields, geometry);
    session.apply_jvp(point, fields, geometry);
  }

 private:
  friend class PreparedBoundaryReadView;

  std::string identity_;
  int required_depth_ = 0;
  PreparedHyperbolicBoundary<2> hyperbolic_boundary_;
  std::array<bool, 4> omitted_faces_{{false, false, false, false}};
  std::string state_identity_;
  PreparedBoundaryReadDependencies read_dependencies_;
  std::vector<PeriodicIdentification2D> periodic_identifications_;
  std::vector<std::shared_ptr<PreparedGhostBoundaryComponent>> ghost_components_;
  std::vector<std::shared_ptr<PreparedBoundaryFluxComponent>> flux_components_;
  std::vector<std::shared_ptr<PreparedFieldBoundaryResidualComponent>> residual_components_;
  std::vector<std::shared_ptr<PreparedFieldBoundaryJvpComponent>> jvp_components_;
  std::function<RecoveryReport(const double*, double*)> trace_recovery_;
  CharacteristicNoInflowFill characteristic_no_inflow_fill_;
  std::size_t component_revision_ = 0;

  template <class Component>
  static void install_typed_(std::vector<std::shared_ptr<Component>>& destination,
                             PreparedBoundaryComponentSpec spec,
                             std::shared_ptr<component::LoadedComponent> component) {
    for (const auto& installed : destination)
      if (installed->spec().target_identity == spec.target_identity &&
          installed->spec().region.identity == spec.region.identity)
        throw std::runtime_error(
            "PreparedBoundaryPlan duplicate typed target/region component binding");
    destination.push_back(std::make_shared<Component>(std::move(spec), std::move(component)));
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
    if (!left || !right)
      return !left && !right;
    const PopsExecutionContextV1 a = left->view();
    const PopsExecutionContextV1 b = right->view();
    const auto same_text = [](const char* x, const char* y) {
      return x != nullptr && y != nullptr && std::string_view(x) == std::string_view(y);
    };
    return a.context_version == b.context_version &&
           same_text(a.execution_identity, b.execution_identity) &&
           a.memory_space == b.memory_space && same_text(a.backend_identity, b.backend_identity) &&
           same_text(a.device_identity, b.device_identity) && a.scalar_type == b.scalar_type &&
           a.storage_precision == b.storage_precision &&
           a.compute_precision == b.compute_precision &&
           a.accumulation_precision == b.accumulation_precision &&
           a.reduction_precision == b.reduction_precision && a.stream_handle == b.stream_handle &&
           same_text(a.stream_identity, b.stream_identity) &&
           a.communicator_f_handle == b.communicator_f_handle &&
           a.communicator_datatype_f_handle == b.communicator_datatype_f_handle &&
           same_text(a.communicator_identity, b.communicator_identity) &&
           same_text(a.communicator_datatype_identity, b.communicator_datatype_identity);
  }

  static void validate_linearization_endpoint_(const PreparedBoundaryComponentSpec& spec,
                                               bool jvp) {
    if (spec.target_identity.empty() || spec.outputs.size() != 1 || spec.outputs.front().empty())
      throw std::runtime_error(
          "PreparedBoundaryPlan residual/JVP target and output identities must be exact");
    const std::vector<std::string> expected_directions =
        jvp ? std::vector<std::string>{spec.state_identity} : std::vector<std::string>{};
    if (spec.directions != expected_directions)
      throw std::runtime_error(
          "PreparedBoundaryPlan residual/JVP direction identities are not executable");
  }

  bool has_mapped_periodicity_() const noexcept {
    return std::any_of(periodic_identifications_.begin(), periodic_identifications_.end(),
                       [](const PeriodicIdentification2D& identification) {
                         return !identification.is_translation_identity();
                       });
  }

  void fill_native_halos_(MultiFab& state, const Box2D& domain) const {
    if (has_mapped_periodicity_())
      fill_mapped_boundary(state, domain, periodic_identifications_);
    else
      fill_boundary(state, domain, periodicity());
  }

  void fill_native_halos_(MultiFab& state, const Box2D& domain, const ExecutionLane& lane) const {
    if (has_mapped_periodicity_())
      fill_mapped_boundary(state, domain, lane, periodic_identifications_);
    else
      fill_boundary(state, domain, lane, periodicity());
  }

  template <class PhysicalPreflight>
  void fill_prepared_physical_(MultiFab& state, const Box2D& domain, CommunicatorView communicator,
                               PhysicalPreflight&& physical_preflight) const {
    hyperbolic_boundary_.fill_physical_preflighted(
        state, std::forward<PhysicalPreflight>(physical_preflight));
    if (requires_characteristic_no_inflow()) {
      if (!characteristic_no_inflow_fill_)
        throw std::logic_error(
            "characteristic no-inflow reached execution without its exact block-model provider");
      characteristic_no_inflow_fill_(state, domain, communicator);
    }
  }

  static std::size_t ghost_snapshot_value_count_(const MultiFab& state) {
    std::size_t cells = 0;
    for (int local = 0; local < state.local_size(); ++local) {
      const std::int64_t ghost_cells =
          state.fab(local).grown_box().num_cells() - state.box(local).num_cells();
      if (ghost_cells < 0)
        throw std::logic_error("PreparedBoundaryPlan observed an invalid grown state box");
      const auto count = static_cast<std::size_t>(ghost_cells);
      if (count > std::numeric_limits<std::size_t>::max() - cells)
        throw std::length_error("PreparedBoundaryPlan ghost transaction exceeds size_t");
      cells += count;
    }
    const auto components = static_cast<std::size_t>(state.ncomp());
    if (components != 0 && cells > std::numeric_limits<std::size_t>::max() / components)
      throw std::length_error("PreparedBoundaryPlan ghost transaction exceeds size_t");
    return cells * components;
  }

  void prepare_boundary_recovery_workspace_(const MultiFab& prototype,
                                            BoundaryRecoveryWorkspace& workspace) const {
    validate_for(prototype);
    const bool recover_traces = trace_recovery_ && has_physical_trace_faces_();
    const bool rollback_characteristics = requires_characteristic_no_inflow();
    if (!recover_traces && !rollback_characteristics) {
      workspace = {};
      return;
    }
    workspace.snapshot.resize(ghost_snapshot_value_count_(prototype));
    if (recover_traces) {
      workspace.conserved.resize(static_cast<std::size_t>(prototype.ncomp()));
      workspace.primitive.resize(static_cast<std::size_t>(prototype.ncomp()));
    } else {
      workspace.conserved.clear();
      workspace.primitive.clear();
    }
    workspace.prepared = true;
  }

  bool has_physical_trace_faces_() const {
    for (int face = 0; face < 4; ++face)
      if (const auto law = hyperbolic_boundary_.face(face / 2, face % 2 == 0 ? -1 : 1).law;
          detail::is_physical_hyperbolic_law(law) ||
          law == HyperbolicBoundaryLaw::CharacteristicNoInflow)
        return true;
    return false;
  }

  static void snapshot_ghost_values_(MultiFab& state, std::vector<Real>& snapshot) {
    state.sync_host();
    std::size_t cursor = 0;
    for (int local = 0; local < state.local_size(); ++local) {
      const Fab2D& fab = state.fab(local);
      const Box2D valid = fab.box();
      const Box2D grown = fab.grown_box();
      for (int component = 0; component < state.ncomp(); ++component)
        for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
          for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
            if (!valid.contains(i, j))
              snapshot[cursor++] = fab(i, j, component);
    }
    if (cursor != snapshot.size())
      throw std::logic_error("PreparedBoundaryPlan ghost snapshot size changed after preparation");
    state.sync_device();
  }

  static void restore_ghost_values_(MultiFab& state, const std::vector<Real>& snapshot) {
    state.sync_host();
    std::size_t cursor = 0;
    for (int local = 0; local < state.local_size(); ++local) {
      Fab2D& fab = state.fab(local);
      const Box2D valid = fab.box();
      const Box2D grown = fab.grown_box();
      for (int component = 0; component < state.ncomp(); ++component)
        for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
          for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
            if (!valid.contains(i, j))
              fab(i, j, component) = snapshot[cursor++];
    }
    if (cursor != snapshot.size())
      throw std::logic_error("PreparedBoundaryPlan ghost restore size changed after preparation");
    state.sync_device();
  }

  template <class Visitor>
  void for_each_physical_trace_cell_(const MultiFab& state, const Box2D& domain,
                                     Visitor&& visitor) const {
    const int depth = state.n_grow();
    const auto physical = [this](int face) {
      const auto law = hyperbolic_boundary_.face(face / 2, face % 2 == 0 ? -1 : 1).law;
      return detail::is_physical_hyperbolic_law(law) ||
             law == HyperbolicBoundaryLaw::CharacteristicNoInflow;
    };
    const auto visit = [&visitor](const Fab2D& fab, const Box2D& region) {
      for (int j = region.lo[1]; j <= region.hi[1]; ++j)
        for (int i = region.lo[0]; i <= region.hi[0]; ++i)
          visitor(fab, i, j);
    };

    for (int local = 0; local < state.local_size(); ++local) {
      const Fab2D& fab = state.fab(local);
      const Box2D valid = fab.box();
      int tangential_lo = valid.lo[1] - depth;
      int tangential_hi = valid.hi[1] + depth;
      if (hyperbolic_boundary_.face(1, -1).law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, domain.lo[1]);
      if (hyperbolic_boundary_.face(1, 1).law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, domain.hi[1]);
      if (physical(0) && valid.lo[0] == domain.lo[0])
        visit(fab, Box2D{{domain.lo[0] - depth, tangential_lo}, {domain.lo[0] - 1, tangential_hi}});
      if (physical(1) && valid.hi[0] == domain.hi[0])
        visit(fab, Box2D{{domain.hi[0] + 1, tangential_lo}, {domain.hi[0] + depth, tangential_hi}});

      tangential_lo = valid.lo[0] - depth;
      tangential_hi = valid.hi[0] + depth;
      if (hyperbolic_boundary_.face(0, -1).law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, domain.lo[0]);
      if (hyperbolic_boundary_.face(0, 1).law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, domain.hi[0]);
      if (physical(2) && valid.lo[1] == domain.lo[1])
        visit(fab, Box2D{{tangential_lo, domain.lo[1] - depth}, {tangential_hi, domain.lo[1] - 1}});
      if (physical(3) && valid.hi[1] == domain.hi[1])
        visit(fab, Box2D{{tangential_lo, domain.hi[1] + 1}, {tangential_hi, domain.hi[1] + depth}});
    }
  }

  void require_recoverable_physical_traces_(const MultiFab& state, const Box2D& domain,
                                            CommunicatorView communicator,
                                            BoundaryRecoveryWorkspace& workspace) const {
    state.sync_host();
    long local_failures = 0;
    for_each_physical_trace_cell_(state, domain, [&](const Fab2D& fab, int i, int j) {
      for (int component = 0; component < state.ncomp(); ++component)
        workspace.conserved[static_cast<std::size_t>(component)] = fab(i, j, component);
      std::fill(workspace.primitive.begin(), workspace.primitive.end(),
                std::numeric_limits<double>::quiet_NaN());
      try {
        const RecoveryReport report =
            trace_recovery_(workspace.conserved.data(), workspace.primitive.data());
        const bool finite = std::all_of(workspace.conserved.begin(), workspace.conserved.end(),
                                        [](double value) { return std::isfinite(value); }) &&
                            std::all_of(workspace.primitive.begin(), workspace.primitive.end(),
                                        [](double value) { return std::isfinite(value); });
        if (!report.publication_permitted() || !finite)
          ++local_failures;
      } catch (...) {
        // A rank-local provider exception is data, not control flow: every peer still reaches the
        // one collective verdict before the transaction either commits or restores its snapshot.
        ++local_failures;
      }
    });
    const long failures = all_reduce_sum(local_failures, communicator);
    if (failures != 0)
      throw std::runtime_error(
          "PreparedBoundaryPlan prepared variable recovery rejected physical boundary traces "
          "before publication (failed cells=" +
          std::to_string(failures) + ")");
    state.sync_device();
  }

  template <class Fill>
  void fill_with_trace_recovery_transaction_(MultiFab& state, const Box2D& domain,
                                             CommunicatorView communicator,
                                             BoundaryRecoveryWorkspace& workspace,
                                             bool require_prepared_workspace, Fill&& fill) const {
    const bool recover_traces = trace_recovery_ && has_physical_trace_faces_();
    const bool rollback_characteristics = requires_characteristic_no_inflow();
    if (!recover_traces && !rollback_characteristics) {
      std::forward<Fill>(fill)();
      return;
    }
    if (!workspace.prepared) {
      if (require_prepared_workspace)
        throw std::logic_error(
            "PreparedBoundaryPlan trace recovery workspace was not materialized before execution");
      prepare_boundary_recovery_workspace_(state, workspace);
    }
    if (workspace.snapshot.size() != ghost_snapshot_value_count_(state) ||
        (recover_traces && (workspace.conserved.size() != static_cast<std::size_t>(state.ncomp()) ||
                            workspace.primitive.size() != static_cast<std::size_t>(state.ncomp()))))
      throw std::logic_error(
          "PreparedBoundaryPlan trace recovery workspace does not match the execution layout");

    snapshot_ghost_values_(state, workspace.snapshot);
    try {
      std::forward<Fill>(fill)();
      if (recover_traces)
        require_recoverable_physical_traces_(state, domain, communicator, workspace);
    } catch (...) {
      device_fence();
      restore_ghost_values_(state, workspace.snapshot);
      throw;
    }
  }

  void validate_base() const {
    if (identity_.empty())
      throw std::runtime_error("PreparedBoundaryPlan requires a canonical identity");
    if (required_depth_ < 1)
      throw std::runtime_error("PreparedBoundaryPlan required depth must be >= 1");
    if (hyperbolic_boundary_.ncomp() < 1)
      throw std::runtime_error(
          "PreparedBoundaryPlan requires one model-aware component transform per state component");
    validate_read_dependencies_(read_dependencies_.states, "state");
    validate_read_dependencies_(read_dependencies_.fields, "field");
    validate_topology();
  }

  static void validate_read_dependencies_(const std::vector<std::string>& identities,
                                          std::string_view kind) {
    std::vector<std::string_view> seen;
    seen.reserve(identities.size());
    for (const auto& identity : identities) {
      if (identity.empty())
        throw std::runtime_error("PreparedBoundaryPlan has an empty declared " + std::string(kind) +
                                 " dependency");
      if (std::find(seen.begin(), seen.end(), identity) != seen.end())
        throw std::runtime_error("PreparedBoundaryPlan has a duplicate declared " +
                                 std::string(kind) + " dependency");
      seen.push_back(identity);
    }
  }

  void validate_topology() const {
    std::array<bool, 4> periodic{};
    for (int face = 0; face < 4; ++face)
      periodic[static_cast<std::size_t>(face)] =
          hyperbolic_boundary_.face(face / 2, face % 2 == 0 ? -1 : 1).law ==
          HyperbolicBoundaryLaw::Periodic;
    if (periodic_identifications_.empty()) {
      if (periodic[0] != periodic[1] || periodic[2] != periodic[3])
        throw std::runtime_error(
            "axis-aligned PreparedBoundaryPlan requires periodic faces in complete axis pairs");
      return;
    }

    std::array<bool, 4> claimed{{false, false, false, false}};
    for (const auto& identification : periodic_identifications_) {
      identification.validate();
      for (const int face : {identification.source_face, identification.target_face}) {
        if (claimed[static_cast<std::size_t>(face)])
          throw std::runtime_error(
              "PreparedBoundaryPlan assigns one face to multiple periodic identifications");
        claimed[static_cast<std::size_t>(face)] = true;
        if (!periodic[static_cast<std::size_t>(face)])
          throw std::runtime_error(
              "PreparedBoundaryPlan periodic identification endpoint is not a periodic face");
      }
    }
    for (std::size_t face = 0; face < periodic.size(); ++face)
      if (periodic[face] != claimed[face])
        throw std::runtime_error(
            "PreparedBoundaryPlan periodic face table differs from explicit identifications");
    if (has_mapped_periodicity_() && periodic_identifications_.size() != 1)
      throw std::runtime_error(
          "PreparedBoundaryPlan mapped periodic topology currently requires one identification; "
          "mixed periodic corners need a composed scheduler");
    if (has_mapped_periodicity_() && hyperbolic_boundary_.has_analytic_state())
      throw std::runtime_error(
          "PreparedBoundaryPlan analytic faces do not yet support mapped periodic coordinates; "
          "install an axis-aligned periodic identification or a prepared coordinate map");
    if (has_mapped_periodicity_())
      for (int component = 0; component < hyperbolic_boundary_.ncomp(); ++component)
        if (hyperbolic_boundary_.component_transform(component).parity !=
            HyperbolicComponentParity::Scalar)
          throw std::runtime_error(
              "mapped periodic topology currently supports scalar component transforms only; "
              "vector and axial states require a model-aware component map");
  }
  void validate_for(const MultiFab& state) const {
    if (state.ncomp() != ncomp())
      throw std::runtime_error("PreparedBoundaryPlan component count does not match block state");
    if (state.n_grow() < required_depth_)
      throw std::runtime_error("PreparedBoundaryPlan stencil depth exceeds allocated ghosts");
    if (requires_characteristic_no_inflow() && !characteristic_no_inflow_fill_)
      throw std::runtime_error(
          "PreparedBoundaryPlan characteristic no-inflow has no authenticated model provider");
  }
};

inline PreparedBoundaryPlan::Session::Session(const PreparedBoundaryPlan& plan,
                                              const ExecutionLane& lane)
    : plan_(&plan), lane_(&lane), component_revision_(plan.component_revision_) {
  plan.validate_base();
  ghost_components_.reserve(plan.ghost_components_.size());
  flux_components_.reserve(plan.flux_components_.size());
  residual_components_.reserve(plan.residual_components_.size());
  jvp_components_.reserve(plan.jvp_components_.size());
  for (const auto& component : plan.ghost_components_)
    ghost_components_.push_back(component->make_session(lane));
  for (const auto& component : plan.flux_components_)
    flux_components_.push_back(component->make_session(lane));
  for (const auto& component : plan.residual_components_)
    residual_components_.push_back(component->make_session(lane));
  for (const auto& component : plan.jvp_components_)
    jvp_components_.push_back(component->make_session(lane));
}

inline void PreparedBoundaryPlan::Session::validate_current_() const {
  if (plan_ == nullptr || lane_ == nullptr)
    throw std::logic_error("PreparedBoundaryPlan session is empty after move");
  if (component_revision_ != plan_->component_revision_)
    throw std::logic_error(
        "PreparedBoundaryPlan was modified after its execution session was materialized");
}

inline void PreparedBoundaryPlan::Session::prepare_trace_recovery_workspace(
    const MultiFab& prototype) {
  validate_current_();
  plan_->prepare_boundary_recovery_workspace_(prototype, recovery_workspace_);
}

inline void PreparedBoundaryPlan::Session::fill_same_level_and_physical(MultiFab& state,
                                                                        const Box2D& domain) const {
  validate_current_();
  if (!ghost_components_.empty())
    throw std::invalid_argument(
        "PreparedBoundaryPlan component session requires an exact BoundaryEvaluationPoint");
  plan_->validate_for(state);
  auto physical_preflight = plan_->hyperbolic_boundary_.preflight_physical(state, domain);
  plan_->fill_with_trace_recovery_transaction_(
      state, domain, lane_->communicator(), recovery_workspace_, true, [&] {
        plan_->fill_native_halos_(state, domain, *lane_);
        plan_->fill_prepared_physical_(state, domain, lane_->communicator(),
                                       std::move(physical_preflight));
      });
}

inline void PreparedBoundaryPlan::Session::fill_same_level_and_physical(
    MultiFab& state, const Geometry& geometry) const {
  validate_current_();
  if (!ghost_components_.empty())
    throw std::invalid_argument(
        "PreparedBoundaryPlan component session requires an exact BoundaryEvaluationPoint");
  plan_->validate_for(state);
  auto physical_preflight =
      plan_->hyperbolic_boundary_.preflight_physical(state, geometry, lane_->communicator());
  plan_->fill_with_trace_recovery_transaction_(
      state, geometry.domain, lane_->communicator(), recovery_workspace_, true, [&] {
        plan_->fill_native_halos_(state, geometry.domain, *lane_);
        plan_->fill_prepared_physical_(state, geometry.domain, lane_->communicator(),
                                       std::move(physical_preflight));
      });
}

inline void PreparedBoundaryPlan::Session::fill_same_level_and_physical(
    MultiFab& state, const Geometry& geometry,
    const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  validate_current_();
  if (!ghost_components_.empty())
    throw std::invalid_argument(
        "PreparedBoundaryPlan component session requires its prepared field registry");
  plan_->validate_for(state);
  auto physical_preflight = plan_->hyperbolic_boundary_.preflight_physical(
      state, geometry, static_cast<Real>(point.physical_time), point.clock, lane_->communicator());
  plan_->fill_with_trace_recovery_transaction_(
      state, geometry.domain, lane_->communicator(), recovery_workspace_, true, [&] {
        plan_->fill_native_halos_(state, geometry.domain, *lane_);
        plan_->fill_prepared_physical_(state, geometry.domain, lane_->communicator(),
                                       std::move(physical_preflight));
      });
}

inline void PreparedBoundaryPlan::Session::fill_same_level_and_physical_control(
    MultiFab& state, const MultiFab* auxiliary, const Geometry& geometry,
    const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  validate_current_();
  plan_->validate_for(state);
  auto physical_preflight = plan_->hyperbolic_boundary_.preflight_physical(
      state, geometry, static_cast<Real>(point.physical_time), point.clock, lane_->communicator());
  plan_->fill_with_trace_recovery_transaction_(
      state, geometry.domain, lane_->communicator(), recovery_workspace_, true, [&] {
        plan_->fill_native_halos_(state, geometry.domain, *lane_);
        plan_->fill_prepared_physical_(state, geometry.domain, lane_->communicator(),
                                       std::move(physical_preflight));
        detail::BoundaryFieldRegistry fields;
        fields.configure_states(plan_->required_state_identities());
        fields.configure_fields(plan_->required_field_identities());
        fields.begin_binding();
        const auto states = plan_->required_state_identities();
        if (states.size() != 1 || states.front() != plan_->state_identity())
          throw std::runtime_error(
              "component boundary with multiple states requires the N-ary prepared registry seam");
        fields.bind_state(states.front(), state);
        const auto dependencies = plan_->required_field_identities();
        if (!dependencies.empty()) {
          if (dependencies.size() != 1 || auxiliary == nullptr)
            throw std::runtime_error(
                "component boundary fields require the N-ary prepared registry seam");
          fields.bind_field(dependencies.front(), *auxiliary);
        }
        for (std::size_t index = 0; index < ghost_components_.size(); ++index) {
          auto workspace = detail::prepare_ghost_workspace(ghost_components_[index], state, fields,
                                                           geometry, plan_->required_depth_);
          detail::apply_ghost_component(ghost_components_[index], workspace, state, fields,
                                        geometry, point);
        }
      });
}

inline void PreparedBoundaryPlan::Session::fill_same_level_and_physical(
    MultiFab& state, const detail::BoundaryFieldRegistry& fields, const Geometry& geometry,
    const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  validate_current_();
  plan_->validate_for(state);
  auto physical_preflight = plan_->hyperbolic_boundary_.preflight_physical(
      state, geometry, static_cast<Real>(point.physical_time), point.clock, lane_->communicator());
  plan_->fill_with_trace_recovery_transaction_(
      state, geometry.domain, lane_->communicator(), recovery_workspace_, true, [&] {
        plan_->fill_native_halos_(state, geometry.domain, *lane_);
        plan_->fill_prepared_physical_(state, geometry.domain, lane_->communicator(),
                                       std::move(physical_preflight));
        if (ghost_workspaces_.size() != ghost_components_.size())
          throw std::logic_error(
              "PreparedBoundaryPlan ghost executor was not materialized before numerical "
              "execution");
        for (std::size_t index = 0; index < ghost_components_.size(); ++index)
          detail::apply_ghost_component(ghost_components_[index], ghost_workspaces_[index], state,
                                        fields, geometry, point);
      });
}

inline void PreparedBoundaryPlan::Session::transform_fluxes_control(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const MultiFab& state,
    const MultiFab* auxiliary, const Geometry& geometry, MultiFab& fx, MultiFab& fy) const {
  validate_current_();
  if (flux_components_.empty())
    return;
  detail::BoundaryFieldRegistry fields;
  fields.configure_states(plan_->required_state_identities());
  fields.configure_fields(plan_->required_field_identities());
  fields.begin_binding();
  const auto states = plan_->required_state_identities();
  if (!states.empty()) {
    if (states.size() != 1 || states.front() != plan_->state_identity())
      throw std::runtime_error(
          "post-Riemann boundary flux with multiple states requires the N-ary prepared registry "
          "seam");
    fields.bind_state(states.front(), state);
  }
  const auto dependencies = plan_->required_field_identities();
  if (!dependencies.empty()) {
    if (dependencies.size() != 1 || auxiliary == nullptr)
      throw std::runtime_error(
          "post-Riemann boundary flux fields require the N-ary prepared registry seam");
    fields.bind_field(dependencies.front(), *auxiliary);
  }
  for (const auto& component : flux_components_) {
    auto workspace = detail::prepare_boundary_flux_workspace(component, state, fields, geometry);
    detail::apply_boundary_flux_component(component, workspace, state, fields, geometry, fx, fy,
                                          point);
  }
}

inline void PreparedBoundaryPlan::Session::transform_fluxes(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const MultiFab& state,
    const detail::BoundaryFieldRegistry& fields, const Geometry& geometry, MultiFab& fx,
    MultiFab& fy) const {
  validate_current_();
  if (flux_workspaces_.size() != flux_components_.size())
    throw std::logic_error(
        "PreparedBoundaryPlan flux executor was not materialized before numerical execution");
  for (std::size_t index = 0; index < flux_components_.size(); ++index)
    detail::apply_boundary_flux_component(flux_components_[index], flux_workspaces_[index], state,
                                          fields, geometry, fx, fy, point);
}

inline void PreparedBoundaryPlan::Session::add_residual_control(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const MultiFab& state,
    const MultiFab* auxiliary, const Geometry& geometry, MultiFab& residual) const {
  validate_current_();
  detail::BoundaryFieldRegistry fields;
  fields.configure_states(plan_->required_state_identities());
  fields.configure_fields(plan_->required_field_identities());
  fields.configure_outputs(plan_->all_output_identities());
  fields.begin_binding();
  const auto states = plan_->required_state_identities();
  if (states.size() != 1 || states.front() != plan_->state_identity())
    throw std::runtime_error(
        "component boundary with multiple states requires the N-ary prepared registry seam");
  fields.bind_state(states.front(), state);
  const auto dependencies = plan_->required_field_identities();
  if (!dependencies.empty()) {
    if (dependencies.size() != 1 || auxiliary == nullptr)
      throw std::runtime_error(
          "component boundary fields require the N-ary prepared registry seam");
    fields.bind_field(dependencies.front(), *auxiliary);
  }
  const auto outputs = plan_->residual_output_identities();
  if (outputs.size() != 1)
    throw std::runtime_error("component boundary outputs require the N-ary prepared registry seam");
  fields.bind_output(outputs.front(), residual);
  for (const auto& component : residual_components_) {
    auto workspace = detail::prepare_field_workspace<PreparedBoundaryOperation::FieldResidual>(
        component, fields, geometry);
    detail::apply_field_component<PreparedBoundaryOperation::FieldResidual>(
        component, workspace, fields, geometry, point);
  }
}

inline void PreparedBoundaryPlan::Session::add_residual(
    const runtime::multiblock::BoundaryEvaluationPoint& point,
    const detail::BoundaryFieldRegistry& fields, const Geometry& geometry) const {
  validate_current_();
  if (residual_workspaces_.size() != residual_components_.size())
    throw std::logic_error(
        "PreparedBoundaryPlan residual executor was not materialized before numerical execution");
  for (std::size_t index = 0; index < residual_components_.size(); ++index)
    detail::apply_field_component<PreparedBoundaryOperation::FieldResidual>(
        residual_components_[index], residual_workspaces_[index], fields, geometry, point);
}

inline void PreparedBoundaryPlan::Session::apply_jvp_control(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const MultiFab& state,
    const MultiFab& direction, const MultiFab* auxiliary, const Geometry& geometry,
    MultiFab& output) const {
  validate_current_();
  detail::BoundaryFieldRegistry fields;
  fields.configure_states(plan_->required_state_identities());
  fields.configure_directions(plan_->required_direction_identities());
  fields.configure_fields(plan_->required_field_identities());
  fields.configure_outputs(plan_->all_output_identities());
  fields.begin_binding();
  const auto states = plan_->required_state_identities();
  if (states.size() != 1 || states.front() != plan_->state_identity())
    throw std::runtime_error(
        "component boundary with multiple states requires the N-ary prepared registry seam");
  fields.bind_state(states.front(), state);
  const auto directions = plan_->required_direction_identities();
  if (directions.size() != 1)
    throw std::runtime_error(
        "component boundary directions require the N-ary prepared registry seam");
  fields.bind_direction(directions.front(), direction);
  const auto dependencies = plan_->required_field_identities();
  if (!dependencies.empty()) {
    if (dependencies.size() != 1 || auxiliary == nullptr)
      throw std::runtime_error(
          "component boundary fields require the N-ary prepared registry seam");
    fields.bind_field(dependencies.front(), *auxiliary);
  }
  const auto outputs = plan_->jvp_output_identities();
  if (outputs.size() != 1)
    throw std::runtime_error("component boundary outputs require the N-ary prepared registry seam");
  fields.bind_output(outputs.front(), output);
  for (const auto& component : jvp_components_) {
    auto workspace = detail::prepare_field_workspace<PreparedBoundaryOperation::FieldJvp>(
        component, fields, geometry);
    detail::apply_field_component<PreparedBoundaryOperation::FieldJvp>(component, workspace, fields,
                                                                       geometry, point);
  }
}

inline void PreparedBoundaryPlan::Session::apply_jvp(
    const runtime::multiblock::BoundaryEvaluationPoint& point,
    const detail::BoundaryFieldRegistry& fields, const Geometry& geometry) const {
  validate_current_();
  if (jvp_workspaces_.size() != jvp_components_.size())
    throw std::logic_error(
        "PreparedBoundaryPlan JVP executor was not materialized before numerical execution");
  for (std::size_t index = 0; index < jvp_components_.size(); ++index)
    detail::apply_field_component<PreparedBoundaryOperation::FieldJvp>(
        jvp_components_[index], jvp_workspaces_[index], fields, geometry, point);
}

inline void PreparedBoundaryPlan::Session::prepare_ghost_executor(
    const MultiFab& prototype, const detail::BoundaryFieldRegistry& fields,
    const Geometry& geometry) {
  validate_current_();
  plan_->validate_for(prototype);
  ghost_workspaces_.clear();
  ghost_workspaces_.reserve(ghost_components_.size());
  for (const auto& component : ghost_components_)
    ghost_workspaces_.push_back(detail::prepare_ghost_workspace(component, prototype, fields,
                                                                geometry, plan_->required_depth_));
}

inline void PreparedBoundaryPlan::Session::prepare_flux_executor(
    const MultiFab& prototype, const detail::BoundaryFieldRegistry& fields,
    const Geometry& geometry) {
  validate_current_();
  plan_->validate_for(prototype);
  flux_workspaces_.clear();
  flux_workspaces_.reserve(flux_components_.size());
  for (const auto& component : flux_components_)
    flux_workspaces_.push_back(
        detail::prepare_boundary_flux_workspace(component, prototype, fields, geometry));
}

inline void PreparedBoundaryPlan::Session::prepare_residual_executor(
    const detail::BoundaryFieldRegistry& fields, const Geometry& geometry) {
  validate_current_();
  residual_workspaces_.clear();
  residual_workspaces_.reserve(residual_components_.size());
  for (const auto& component : residual_components_)
    residual_workspaces_.push_back(
        detail::prepare_field_workspace<PreparedBoundaryOperation::FieldResidual>(component, fields,
                                                                                  geometry));
}

inline void PreparedBoundaryPlan::Session::prepare_jvp_executor(
    const detail::BoundaryFieldRegistry& fields, const Geometry& geometry) {
  validate_current_();
  jvp_workspaces_.clear();
  jvp_workspaces_.reserve(jvp_components_.size());
  for (const auto& component : jvp_components_)
    jvp_workspaces_.push_back(detail::prepare_field_workspace<PreparedBoundaryOperation::FieldJvp>(
        component, fields, geometry));
}

}  // namespace pops
