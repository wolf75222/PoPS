/// @file
/// @brief Prepared model-qualified physical-boundary authority for canonical ND state storage.

#pragma once

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/embedded_boundary/characteristic.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/analytic/expression.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

enum class HyperbolicBoundaryLaw {
  Periodic,
  Extrapolate,
  FixedState,
  CharacteristicNoInflow,
  NoFlux,
  ReflectiveSlip,
  External
};

enum class HyperbolicComponentParity { Scalar, PolarVector, AxialVector };
enum class HyperbolicStateRepresentation { Conservative, Primitive };
enum class HyperbolicCornerPolicy { NotRequired };

template <int Dim>
struct HyperbolicComponentTransform {
  static_assert(Dim >= 1 && Dim <= 3);

  HyperbolicComponentParity parity = HyperbolicComponentParity::Scalar;
  int axis = -1;

  static HyperbolicComponentTransform scalar() { return {}; }
  static HyperbolicComponentTransform polar_vector(int component_axis) {
    if (component_axis < 0 || component_axis >= 3)
      throw std::invalid_argument(
          "polar boundary component axis is outside the physical x/y/z embedding");
    return {HyperbolicComponentParity::PolarVector, component_axis};
  }
  static HyperbolicComponentTransform axial_vector(int component_axis) {
    if (component_axis < 0 || component_axis >= 3)
      throw std::invalid_argument(
          "axial boundary component axis is outside the physical x/y/z embedding");
    return {HyperbolicComponentParity::AxialVector, component_axis};
  }

  POPS_HD Real reflection_sign(int normal_axis) const {
    if (parity == HyperbolicComponentParity::Scalar)
      return Real(1);
    const bool normal_component = axis == normal_axis;
    if (parity == HyperbolicComponentParity::PolarVector)
      return normal_component ? Real(-1) : Real(1);
    return normal_component ? Real(1) : Real(-1);
  }
};

template <int Dim>
struct HyperbolicFaceContext {
  static_assert(Dim >= 1 && Dim <= 3);

  int axis = 0;
  int side = -1;
  std::array<Real, Dim> coordinate{};
  std::array<Real, Dim> normal{};
  std::array<std::array<Real, Dim>, (Dim > 1 ? Dim - 1 : 0)> tangents{};
  Real metric = Real(1);
  Real area = Real(1);
  Real time = Real(0);
  const Real* runtime_parameters = nullptr;
  int runtime_parameter_count = 0;
  const Real* auxiliary_values = nullptr;
  int auxiliary_value_count = 0;
  std::uint64_t boundary_identity = 0;
};

/// Host-side authored face metadata.  Conservative analytic ScalarExpr programs evaluate on device
/// through the same `Geometry<Dim>::cell_center` already prepared on the fill session, including the
/// `z` coordinate when Dim == 3.  A domain box with origin/spacing builds that Geometry once; a
/// box without those coordinates is refused rather than inventing zeros.
struct PreparedHyperbolicFace {
  HyperbolicBoundaryLaw law = HyperbolicBoundaryLaw::Periodic;
  std::string identity;
  std::uint64_t identity_token = 0;
  std::vector<Real> fixed_state;
  HyperbolicStateRepresentation authored_representation =
      HyperbolicStateRepresentation::Conservative;
  std::string converter_identity;
  bool fixed_state_converted = true;
  std::vector<analytic::AnalyticProgram> analytic_state;
  std::string analytic_clock;
};

namespace hyperbolic_boundary_detail {

inline std::uint64_t stable_boundary_identity(std::string_view identity) {
  if (identity.empty())
    throw std::invalid_argument("hyperbolic boundary identity must be non-empty");
  std::uint64_t value = UINT64_C(1469598103934665603);
  for (const unsigned char byte : identity) {
    value ^= static_cast<std::uint64_t>(byte);
    value *= UINT64_C(1099511628211);
  }
  return value;
}

template <int Dim>
struct BoundaryTableView {
  const HyperbolicComponentTransform<Dim>* transforms = nullptr;
  const Real* fixed_values = nullptr;
  int ncomp = 0;

  POPS_HD const HyperbolicComponentTransform<Dim>& transform(int component) const {
    return transforms[component];
  }
  POPS_HD Real fixed_value(int face, int component) const {
    return fixed_values[face * ncomp + component];
  }
};

struct BoundarySample {
  int source = 0;
  Real scale = Real(1);
  Real offset = Real(0);
};

inline bool is_builtin_physical_law(HyperbolicBoundaryLaw law) {
  return law == HyperbolicBoundaryLaw::Extrapolate || law == HyperbolicBoundaryLaw::FixedState ||
         law == HyperbolicBoundaryLaw::NoFlux || law == HyperbolicBoundaryLaw::ReflectiveSlip;
}

inline const char* law_name(HyperbolicBoundaryLaw law) {
  switch (law) {
    case HyperbolicBoundaryLaw::Periodic:
      return "periodic";
    case HyperbolicBoundaryLaw::Extrapolate:
      return "extrapolate";
    case HyperbolicBoundaryLaw::FixedState:
      return "fixed_state";
    case HyperbolicBoundaryLaw::CharacteristicNoInflow:
      return "characteristic_no_inflow";
    case HyperbolicBoundaryLaw::NoFlux:
      return "no_flux";
    case HyperbolicBoundaryLaw::ReflectiveSlip:
      return "reflective_slip";
    case HyperbolicBoundaryLaw::External:
      return "external";
  }
  return "unknown";
}

template <int Dim>
POPS_HD BoundarySample sample_axis(int index, int lo, int hi, int axis, HyperbolicBoundaryLaw low,
                                   HyperbolicBoundaryLaw high, const BoundaryTableView<Dim>& table,
                                   int component) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const HyperbolicBoundaryLaw law = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (law == HyperbolicBoundaryLaw::Extrapolate || law == HyperbolicBoundaryLaw::NoFlux) {
      current = boundary;
      break;
    }

    Real face_scale = Real(1);
    Real face_offset = Real(0);
    if (law == HyperbolicBoundaryLaw::FixedState) {
      face_scale = Real(-1);
      face_offset = Real(2) * table.fixed_value(2 * axis + (below ? 0 : 1), component);
    } else if (law == HyperbolicBoundaryLaw::ReflectiveSlip) {
      face_scale = table.transform(component).reflection_sign(axis);
    } else {
      current = boundary;
      break;
    }
    offset += scale * face_offset;
    scale *= face_scale;
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
  return {static_cast<int>(current), scale, offset};
}

template <int Dim>
void validate_extension(int index, int lo, int hi, int axis, HyperbolicBoundaryLaw low,
                        HyperbolicBoundaryLaw high, const BoundaryTableView<Dim>& table,
                        int component) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const HyperbolicBoundaryLaw law = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (!is_builtin_physical_law(law))
      throw std::invalid_argument(std::string("prepared hyperbolic halo reaches a ") +
                                  law_name(law) +
                                  " face owned by another explicit boundary provider");
    if (law == HyperbolicBoundaryLaw::Extrapolate || law == HyperbolicBoundaryLaw::NoFlux)
      return;

    Real face_scale = Real(1);
    Real face_offset = Real(0);
    if (law == HyperbolicBoundaryLaw::FixedState) {
      face_scale = Real(-1);
      face_offset = Real(2) * table.fixed_value(2 * axis + (below ? 0 : 1), component);
    } else {
      face_scale = table.transform(component).reflection_sign(axis);
    }
    offset += scale * face_offset;
    scale *= face_scale;
    if (!std::isfinite(scale) || !std::isfinite(offset))
      throw std::overflow_error("prepared hyperbolic halo produced a non-finite affine extension");
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
}

template <int Axis, int Dim>
struct FillPhysicalFace {
  FieldView<Real, Dim> state{};
  BoundaryTableView<Dim> table{};
  int lo = 0;
  int hi = -1;
  HyperbolicBoundaryLaw low = HyperbolicBoundaryLaw::External;
  HyperbolicBoundaryLaw high = HyperbolicBoundaryLaw::External;

  POPS_HD void operator()(const Index<Dim>& ghost) const {
    for (int component = 0; component < table.ncomp; ++component) {
      const BoundarySample sample =
          sample_axis(ghost[Axis], lo, hi, Axis, low, high, table, component);
      Index<Dim> source = ghost;
      source[Axis] = sample.source;
      state(ghost, component) = sample.scale * state(source, component) + sample.offset;
    }
  }
};

inline constexpr int kMaxAnalyticBoundaryComponents = 32;

template <int Dim>
bool origin_spacing_usable(const RealVector<Dim>& origin, const RealVector<Dim>& spacing) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (!std::isfinite(origin[axis]) || !std::isfinite(spacing[axis]) || !(spacing[axis] > Real(0)))
      return false;
  }
  return true;
}

template <int Dim>
Geometry<Dim> geometry_from_box_origin_spacing(const Box<Dim>& domain,
                                               const RealVector<Dim>& origin,
                                               const RealVector<Dim>& spacing) {
  if (domain.empty())
    throw std::invalid_argument("prepared hyperbolic boundary domain must be non-empty");
  if (!origin_spacing_usable(origin, spacing))
    throw std::logic_error(
        "analytic hyperbolic boundary requires a requalified ND coordinate provider");
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = origin[axis] + spacing[axis] * static_cast<Real>(domain.length(axis));
  return Geometry<Dim>::from_bounds(domain, origin, upper);
}

inline analytic::AnalyticProgram compile_boundary_analytic_program(
    const std::vector<std::string>& opcodes, const std::vector<double>& literals, int dimension) {
  if (opcodes.empty() || opcodes.size() != literals.size())
    throw std::invalid_argument("analytic hyperbolic program has mismatched opcode/literal rows");
  std::vector<analytic::AnalyticToken> tokens;
  tokens.reserve(opcodes.size());
  for (std::size_t index = 0; index < opcodes.size(); ++index) {
    const analytic::AnalyticOp op = analytic::analytic_op_from_name(opcodes[index]);
    const double raw = literals[index];
    if (!std::isfinite(raw))
      throw std::invalid_argument("analytic hyperbolic token literal must be finite");
    if (op == analytic::AnalyticOp::Input &&
        (raw < 0.0 || raw >= static_cast<double>(1 + kMaxAnalyticBoundaryComponents) ||
         raw != std::floor(raw)))
      throw std::invalid_argument("analytic hyperbolic Input slot is outside the typed boundary ABI");
    tokens.push_back(analytic::AnalyticToken{op, static_cast<Real>(raw)});
  }
  analytic::AnalyticProgram program = analytic::compile_analytic_postfix(tokens);
  if (program.result_type() != analytic::AnalyticValueType::Scalar)
    throw std::invalid_argument("analytic hyperbolic program must produce one scalar value");
  if (program.required_dimension() > dimension)
    throw std::invalid_argument(
        "analytic hyperbolic program requires a higher spatial rank than the prepared boundary");
  return program;
}

template <class Model>
concept ExactPrimitiveToConservativeModel =
    requires(const Model& model, const typename Model::Prim& primitive) {
      typename Model::State;
      typename Model::Prim;
      Model::n_vars;
      { model.to_conservative(primitive) } -> std::same_as<typename Model::State>;
    };

template <int Axis, int Dim>
struct ValidateAnalyticFace {
  static_assert(Axis >= 0 && Axis < Dim);

  FieldView<const Real, Dim> state{};
  Geometry<Dim> geometry;
  analytic::AnalyticProgramView programs[kMaxAnalyticBoundaryComponents]{};
  Real time = Real(0);
  int ncomp = 0;
  int lo = 0;
  int hi = -1;
  int side = -1;

  POPS_HD Real operator()(const Index<Dim>& ghost) const {
    Real inputs[1 + kMaxAnalyticBoundaryComponents];
    inputs[0] = time;
    Index<Dim> interior = ghost;
    interior[Axis] = side < 0 ? lo : hi;
    const int input_count = 1 + ncomp;
    for (int component = 0; component < ncomp; ++component)
      inputs[1 + component] = state(interior, component);
    for (int component = 0; component < ncomp; ++component) {
      const analytic::AnalyticEvaluation evaluation = programs[component].eval_checked(
          ghost, geometry, inputs, static_cast<std::uint8_t>(input_count));
      if (!evaluation.valid || !Kokkos::isfinite(evaluation.value))
        return Real(1);
    }
    return Real(0);
  }
};

template <int Axis, int Dim>
struct FillAnalyticFace {
  static_assert(Axis >= 0 && Axis < Dim);

  FieldView<Real, Dim> state{};
  Geometry<Dim> geometry;
  analytic::AnalyticProgramView programs[kMaxAnalyticBoundaryComponents]{};
  Real time = Real(0);
  int ncomp = 0;
  int lo = 0;
  int hi = -1;
  int side = -1;

  POPS_HD void operator()(const Index<Dim>& ghost) const {
    Real inputs[1 + kMaxAnalyticBoundaryComponents];
    inputs[0] = time;
    Index<Dim> interior = ghost;
    interior[Axis] = side < 0 ? lo : hi;
    const int input_count = 1 + ncomp;
    for (int component = 0; component < ncomp; ++component)
      inputs[1 + component] = state(interior, component);
    const RealVector<Dim> coordinates = geometry.cell_center(ghost);
    for (int component = 0; component < ncomp; ++component) {
      const analytic::AnalyticEvaluation evaluation = programs[component].eval_checked(
          coordinates, inputs, static_cast<std::uint8_t>(input_count));
      const Real prescribed = evaluation.valid ? evaluation.value : std::numeric_limits<Real>::quiet_NaN();
      state(ghost, component) = Real(2) * prescribed - state(interior, component);
    }
  }
};

template <int Axis, int Dim, class Model>
struct FillAnalyticPrimitiveFace {
  static_assert(Axis >= 0 && Axis < Dim);

  Model model;
  FieldView<Real, Dim> state{};
  Geometry<Dim> geometry;
  analytic::AnalyticProgramView programs[kMaxAnalyticBoundaryComponents]{};
  Real time = Real(0);
  int ncomp = 0;
  int lo = 0;
  int hi = -1;
  int side = -1;

  POPS_HD Real operator()(const Index<Dim>& ghost) const {
    Real inputs[1 + kMaxAnalyticBoundaryComponents];
    inputs[0] = time;
    Index<Dim> interior = ghost;
    interior[Axis] = side < 0 ? lo : hi;
    const int input_count = 1 + ncomp;
    for (int component = 0; component < ncomp; ++component)
      inputs[1 + component] = state(interior, component);
    typename Model::Prim primitive{};
    for (int component = 0; component < ncomp; ++component) {
      const analytic::AnalyticEvaluation evaluation = programs[component].eval_checked(
          ghost, geometry, inputs, static_cast<std::uint8_t>(input_count));
      if (!evaluation.valid || !Kokkos::isfinite(evaluation.value))
        return Real(1);
      primitive[component] = evaluation.value;
    }
    const typename Model::State conservative = model.to_conservative(primitive);
    for (int component = 0; component < ncomp; ++component) {
      if (!Kokkos::isfinite(conservative[component]))
        return Real(1);
      state(ghost, component) = Real(2) * conservative[component] - state(interior, component);
    }
    return Real(0);
  }
};

template <int Dim>
struct ZeroBoundaryFaceFlux {
  FieldView<Real, Dim> flux{};
  int ncomp = 0;

  POPS_HD void operator()(const Index<Dim>& face) const {
    for (int component = 0; component < ncomp; ++component)
      flux(face, component) = Real(0);
  }
};

template <class Model, int Dim>
concept ExactCharacteristicNoInflowModel =
    requires(const Model& model, const typename Model::State& interior,
             const typename Model::State& reference, typename Model::State& ghost) {
      Model::dimension;
      Model::n_vars;
      Model::characteristic_no_inflow_contract_version;
      Model::characteristic_no_inflow_dimension;
      Model::characteristic_no_inflow_components;
      Model::characteristic_no_inflow_conservative;
      { model.characteristic_no_inflow(interior, reference, 0, -1, ghost) } -> std::same_as<bool>;
    } &&
    Model::dimension == Dim && Model::characteristic_no_inflow_contract_version == 1 &&
    Model::characteristic_no_inflow_dimension == Dim &&
    Model::characteristic_no_inflow_components == Model::n_vars &&
    Model::characteristic_no_inflow_conservative;

template <int Axis, int Side, int Dim, class Model>
struct FillCharacteristicNoInflowFace {
  static_assert(Axis >= 0 && Axis < Dim);
  static_assert(Side == -1 || Side == 1);

  Model model;
  FieldView<Real, Dim> state{};
  BoundaryTableView<Dim> table{};
  int boundary_index = 0;

  POPS_HD Real operator()(const Index<Dim>& ghost_index) const {
    typename Model::State interior{};
    typename Model::State reference{};
    typename Model::State ghost{};
    Index<Dim> interior_index = ghost_index;
    interior_index[Axis] = boundary_index;
    for (int component = 0; component < Model::n_vars; ++component) {
      interior[component] = state(interior_index, component);
      reference[component] = table.fixed_value(2 * Axis + (Side > 0 ? 1 : 0), component);
      if (!Kokkos::isfinite(interior[component]) || !Kokkos::isfinite(reference[component]))
        return Real(1);
    }
    if (!model.characteristic_no_inflow(interior, reference, Axis, Side, ghost))
      return Real(1);
    for (int component = 0; component < Model::n_vars; ++component)
      if (!Kokkos::isfinite(ghost[component]))
        return Real(1);
    for (int component = 0; component < Model::n_vars; ++component)
      state(ghost_index, component) = ghost[component];
    return Real(0);
  }
};

template <int Dim>
struct PublishBoundaryCandidate {
  FieldView<const Real, Dim> candidate{};
  FieldView<Real, Dim> state{};
  int ncomp = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < ncomp; ++component)
      state(index, component) = candidate(index, component);
  }
};

template <int Dim>
struct ValidateBoundaryCandidate {
  FieldView<const Real, Dim> candidate{};
  int ncomp = 0;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    for (int component = 0; component < ncomp; ++component)
      if (!Kokkos::isfinite(candidate(index, component)))
        return Real(1);
    return Real(0);
  }
};

template <int Dim>
HyperbolicComponentTransform<Dim> transform_from_semantic(VariableSemantic semantic) {
  semantic.template validate_for_dimension<Dim>();
  if (semantic.kind == VariableRoleKind::Momentum || semantic.kind == VariableRoleKind::Velocity)
    return HyperbolicComponentTransform<Dim>::polar_vector(semantic.axis);
  if (semantic.kind == VariableRoleKind::Axial)
    return HyperbolicComponentTransform<Dim>::axial_vector(semantic.axis);
  return HyperbolicComponentTransform<Dim>::scalar();
}

template <int Dim>
HyperbolicComponentTransform<Dim> transform_from_role(std::string_view role) {
  return transform_from_semantic<Dim>(role_from_name(std::string(role)));
}

inline HyperbolicBoundaryLaw law_from_token(std::string_view token) {
  if (token == "periodic")
    return HyperbolicBoundaryLaw::Periodic;
  if (token == "foextrap")
    return HyperbolicBoundaryLaw::Extrapolate;
  if (token == "dirichlet")
    return HyperbolicBoundaryLaw::FixedState;
  if (token == "characteristic_no_inflow")
    return HyperbolicBoundaryLaw::CharacteristicNoInflow;
  if (token == "no_flux")
    return HyperbolicBoundaryLaw::NoFlux;
  if (token == "slip_wall")
    return HyperbolicBoundaryLaw::ReflectiveSlip;
  if (token == "external")
    return HyperbolicBoundaryLaw::External;
  throw std::invalid_argument("unsupported prepared hyperbolic face law '" + std::string(token) +
                              "'");
}

inline HyperbolicStateRepresentation representation_from_token(std::string_view token) {
  if (token == "conservative")
    return HyperbolicStateRepresentation::Conservative;
  if (token == "primitive")
    return HyperbolicStateRepresentation::Primitive;
  throw std::invalid_argument("unsupported prepared hyperbolic representation '" +
                              std::string(token) + "'");
}

}  // namespace hyperbolic_boundary_detail

template <int Dim>
class PreparedHyperbolicBoundary {
 public:
  static_assert(Dim >= 1 && Dim <= 3);
  using Transform = HyperbolicComponentTransform<Dim>;

  template <class MemorySpace>
  class PhysicalFillPreflight final {
   public:
    PhysicalFillPreflight(const PhysicalFillPreflight&) = delete;
    PhysicalFillPreflight& operator=(const PhysicalFillPreflight&) = delete;
    PhysicalFillPreflight(PhysicalFillPreflight&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          state_(std::exchange(other.state_, nullptr)),
          domain_(other.domain_),
          geometry_(other.geometry_),
          time_(other.time_),
          ncomp_(other.ncomp_),
          ghosts_(other.ghosts_),
          local_size_(other.local_size_),
          layout_(std::move(other.layout_)),
          distribution_(std::move(other.distribution_)),
          local_rank_(other.local_rank_) {}
    PhysicalFillPreflight& operator=(PhysicalFillPreflight&& other) noexcept {
      if (this != &other) {
        owner_ = std::exchange(other.owner_, nullptr);
        state_ = std::exchange(other.state_, nullptr);
        domain_ = other.domain_;
        geometry_ = other.geometry_;
        time_ = other.time_;
        ncomp_ = other.ncomp_;
        ghosts_ = other.ghosts_;
        local_size_ = other.local_size_;
        layout_ = std::move(other.layout_);
        distribution_ = std::move(other.distribution_);
        local_rank_ = other.local_rank_;
      }
      return *this;
    }

   private:
    friend class PreparedHyperbolicBoundary;

    PhysicalFillPreflight(const PreparedHyperbolicBoundary* owner,
                          const MultiFab<Dim, MemorySpace>* state, Box<Dim> domain,
                          std::optional<Geometry<Dim>> geometry = std::nullopt,
                          Real time = Real(0))
        : owner_(owner),
          state_(state),
          domain_(domain),
          geometry_(std::move(geometry)),
          time_(time),
          ncomp_(state->ncomp()),
          ghosts_(state->ghosts()),
          local_size_(state->local_size()),
          layout_(state->layout()),
          distribution_(state->distribution()),
          local_rank_(state->local_rank()) {}

    const PreparedHyperbolicBoundary* owner_ = nullptr;
    const MultiFab<Dim, MemorySpace>* state_ = nullptr;
    Box<Dim> domain_{};
    std::optional<Geometry<Dim>> geometry_{};
    Real time_ = Real(0);
    int ncomp_ = 0;
    Extent<Dim> ghosts_{};
    std::size_t local_size_ = 0;
    typename MultiFab<Dim, MemorySpace>::layout_type layout_{};
    typename MultiFab<Dim, MemorySpace>::distribution_type distribution_{};
    Index<Dim> local_rank_{};
  };

  PreparedHyperbolicBoundary() = default;

  PreparedHyperbolicBoundary(
      std::array<PreparedHyperbolicFace, 2 * Dim> faces,
      std::vector<Transform> component_transforms,
      HyperbolicCornerPolicy corner_policy = HyperbolicCornerPolicy::NotRequired,
      bool explicit_periodic_identifications = false)
      : faces_(std::move(faces)),
        component_transforms_(std::move(component_transforms)),
        corner_policy_(corner_policy),
        explicit_periodic_identifications_(explicit_periodic_identifications) {
    validate_();
    prepare_device_tables_();
  }

  int ncomp() const { return static_cast<int>(component_transforms_.size()); }
  const PreparedHyperbolicFace& face(int axis, int side) const {
    if (axis < 0 || axis >= Dim || (side != -1 && side != 1))
      throw std::out_of_range("prepared hyperbolic face selector is outside the model dimension");
    return faces_[static_cast<std::size_t>(2 * axis + (side > 0 ? 1 : 0))];
  }
  const Transform& component_transform(int component) const {
    if (component < 0 || component >= ncomp())
      throw std::out_of_range("prepared hyperbolic component is outside the state");
    return component_transforms_[static_cast<std::size_t>(component)];
  }
  HyperbolicCornerPolicy corner_policy() const { return corner_policy_; }

  std::array<bool, Dim> periodic_axes() const {
    std::array<bool, Dim> result{};
    for (int axis = 0; axis < Dim; ++axis) {
      const bool low =
          faces_[static_cast<std::size_t>(2 * axis)].law == HyperbolicBoundaryLaw::Periodic;
      const bool high =
          faces_[static_cast<std::size_t>(2 * axis + 1)].law == HyperbolicBoundaryLaw::Periodic;
      if (low != high)
        throw std::logic_error(
            "identified periodic faces require an explicit topology provider and cannot be "
            "projected to per-axis periodicity");
      result[static_cast<std::size_t>(axis)] = low;
    }
    return result;
  }

  bool has_analytic_state() const {
    return std::any_of(faces_.begin(), faces_.end(), [](const PreparedHyperbolicFace& prepared) {
      return !prepared.analytic_state.empty();
    });
  }
  bool has_analytic_primitive() const {
    return std::any_of(faces_.begin(), faces_.end(), [](const PreparedHyperbolicFace& prepared) {
      return !prepared.analytic_state.empty() &&
             prepared.authored_representation == HyperbolicStateRepresentation::Primitive;
    });
  }
  PreparedHyperbolicBoundary with_prepared_geometry(const Geometry<Dim>& geometry) const {
    PreparedHyperbolicBoundary copy(*this);
    copy.prepared_geometry_ = geometry;
    return copy;
  }
  bool has_characteristic_no_inflow() const {
    return std::any_of(faces_.begin(), faces_.end(), [](const PreparedHyperbolicFace& prepared) {
      return prepared.law == HyperbolicBoundaryLaw::CharacteristicNoInflow;
    });
  }

  template <class Model>
  void require_model_qualified_characteristic_provider() const {
    if (!has_characteristic_no_inflow())
      return;
    if constexpr (!hyperbolic_boundary_detail::ExactCharacteristicNoInflowModel<Model, Dim>) {
      throw std::logic_error(
          "characteristic no-inflow requires the generated model's exact conservative ND "
          "provider contract");
    } else if (Model::n_vars != ncomp()) {
      throw std::invalid_argument(
          "characteristic no-inflow provider component layout differs from the boundary state");
    }
  }
  bool requires_fixed_state_conversion() const {
    return std::any_of(faces_.begin(), faces_.end(), [](const PreparedHyperbolicFace& prepared) {
      return prepared.law == HyperbolicBoundaryLaw::FixedState &&
             prepared.authored_representation == HyperbolicStateRepresentation::Primitive &&
             !prepared.fixed_state_converted && prepared.analytic_state.empty();
    });
  }

  PreparedHyperbolicBoundary with_converted_fixed_states(
      const std::function<void(const double*, double*)>& primitive_to_conservative) const {
    if (!requires_fixed_state_conversion())
      return *this;
    if (!primitive_to_conservative)
      throw std::invalid_argument(
          "primitive fixed-state boundary requires the compiled model conversion");

    auto converted_faces = faces_;
    std::vector<double> input(static_cast<std::size_t>(ncomp()));
    std::vector<double> output(static_cast<std::size_t>(ncomp()));
    for (auto& prepared : converted_faces) {
      if (prepared.law != HyperbolicBoundaryLaw::FixedState ||
          prepared.authored_representation != HyperbolicStateRepresentation::Primitive ||
          prepared.fixed_state_converted)
        continue;
      for (int component = 0; component < ncomp(); ++component)
        input[static_cast<std::size_t>(component)] =
            static_cast<double>(prepared.fixed_state[static_cast<std::size_t>(component)]);
      std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
      primitive_to_conservative(input.data(), output.data());
      if (std::any_of(output.begin(), output.end(),
                      [](double value) { return !std::isfinite(value); }))
        throw std::runtime_error(
            "primitive fixed-state boundary conversion produced a non-finite component");
      for (int component = 0; component < ncomp(); ++component) {
        const Real converted = static_cast<Real>(output[static_cast<std::size_t>(component)]);
        if (!std::isfinite(converted))
          throw std::runtime_error(
              "primitive fixed-state boundary conversion exceeds the runtime precision");
        prepared.fixed_state[static_cast<std::size_t>(component)] = converted;
      }
      prepared.fixed_state_converted = true;
    }
    auto converted = PreparedHyperbolicBoundary(std::move(converted_faces), component_transforms_,
                                                corner_policy_, explicit_periodic_identifications_);
    converted.prepared_geometry_ = prepared_geometry_;
    return converted;
  }

  template <class MemorySpace>
  PhysicalFillPreflight<MemorySpace> preflight_physical(MultiFab<Dim, MemorySpace>& state,
                                                        const Box<Dim>& domain,
                                                        Real time = Real(0)) const {
    const Geometry<Dim>* geometry = nullptr;
    if (prepared_geometry_ && prepared_geometry_->domain() == domain)
      geometry = &*prepared_geometry_;
    validate_physical_contract_(state, domain, /*allow_characteristic=*/false, geometry);
    return PhysicalFillPreflight<MemorySpace>(
        this, &state, domain, geometry == nullptr ? std::nullopt : std::optional(*geometry), time);
  }

  template <class MemorySpace>
  PhysicalFillPreflight<MemorySpace> preflight_physical(MultiFab<Dim, MemorySpace>& state,
                                                        const Geometry<Dim>& geometry,
                                                        Real time = Real(0)) const {
    validate_physical_contract_(state, geometry.domain(), /*allow_characteristic=*/false,
                                &geometry);
    return PhysicalFillPreflight<MemorySpace>(this, &state, geometry.domain(), geometry, time);
  }

  template <class MemorySpace>
  PhysicalFillPreflight<MemorySpace> preflight_physical(MultiFab<Dim, MemorySpace>& state,
                                                        const Box<Dim>& domain,
                                                        const RealVector<Dim>& origin,
                                                        const RealVector<Dim>& spacing,
                                                        Real time = Real(0)) const {
    if (hyperbolic_boundary_detail::origin_spacing_usable(origin, spacing)) {
      return preflight_physical(
          state, hyperbolic_boundary_detail::geometry_from_box_origin_spacing(domain, origin, spacing),
          time);
    }
    if (prepared_geometry_ && prepared_geometry_->domain() == domain)
      return preflight_physical(state, *prepared_geometry_, time);
    validate_physical_contract_(state, domain, /*allow_characteristic=*/false, nullptr);
    return PhysicalFillPreflight<MemorySpace>(this, &state, domain, std::nullopt, time);
  }

  template <class MemorySpace>
  void fill_physical_preflighted(MultiFab<Dim, MemorySpace>& state,
                                 PhysicalFillPreflight<MemorySpace>&& preflight) const {
    if (preflight.owner_ != this || preflight.state_ != &state ||
        preflight.ncomp_ != state.ncomp() || preflight.ghosts_ != state.ghosts() ||
        preflight.local_size_ != state.local_size() || preflight.layout_ != state.layout() ||
        preflight.distribution_ != state.distribution() ||
        preflight.local_rank_ != state.local_rank())
      throw std::logic_error(
          "prepared hyperbolic boundary received a foreign or stale physical preflight");
    preflight.owner_ = nullptr;
    fill_axes_<0>(state, preflight.domain_,
                  preflight.geometry_ ? &*preflight.geometry_ : nullptr, preflight.time_);
    device_fence();
  }

  template <class MemorySpace>
  void fill_physical(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                     Real time = Real(0)) const {
    if (has_analytic_primitive())
      throw std::logic_error(
          "analytic primitive inflow requires the compiled model's to_conservative provider");
    auto preflight = preflight_physical(state, domain, time);
    fill_physical_preflighted(state, std::move(preflight));
  }

  template <class MemorySpace>
  void fill_physical(MultiFab<Dim, MemorySpace>& state, const Geometry<Dim>& geometry,
                     Real time = Real(0)) const {
    if (has_analytic_primitive())
      throw std::logic_error(
          "analytic primitive inflow requires the compiled model's to_conservative provider");
    auto preflight = preflight_physical(state, geometry, time);
    fill_physical_preflighted(state, std::move(preflight));
  }

  template <class MemorySpace>
  void fill_physical(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                     const RealVector<Dim>& origin, const RealVector<Dim>& spacing,
                     Real time = Real(0)) const {
    if (has_analytic_primitive())
      throw std::logic_error(
          "analytic primitive inflow requires the compiled model's to_conservative provider");
    auto preflight = preflight_physical(state, domain, origin, spacing, time);
    fill_physical_preflighted(state, std::move(preflight));
  }

  /// Execute the generated model's exact characteristic provider together with the generic
  /// physical laws.  The complete candidate is materialized in Kokkos storage and validated
  /// before any value is published back to ``state``.
  template <class Model, class MemorySpace>
  void fill_physical_model_qualified(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                                     const Model& model, Real time = Real(0)) const {
    if (has_characteristic_no_inflow())
      throw std::logic_error(
          "characteristic no-inflow requires an explicit prepared ExecutionLane");
    if (has_analytic_primitive()) {
      auto preflight = preflight_physical(state, domain, time);
      fill_physical_preflighted_model_(state, std::move(preflight), model);
      return;
    }
    fill_physical(state, domain, time);
  }

  template <class Model, class MemorySpace>
  void fill_physical_model_qualified(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                                     const Model& model, const ExecutionLane& lane,
                                     MultiFab<Dim, MemorySpace>& candidate,
                                     Real time = Real(0)) const {
    fill_physical_model_qualified_(state, domain, model, lane, candidate, nullptr, time);
  }

  template <class Model, class MemorySpace>
  void fill_physical_model_qualified(MultiFab<Dim, MemorySpace>& state,
                                     const Geometry<Dim>& geometry, const Model& model,
                                     Real time = Real(0)) const {
    if (has_characteristic_no_inflow())
      throw std::logic_error(
          "characteristic no-inflow requires an explicit prepared ExecutionLane");
    if (has_analytic_primitive()) {
      auto preflight = preflight_physical(state, geometry, time);
      fill_physical_preflighted_model_(state, std::move(preflight), model);
      return;
    }
    fill_physical(state, geometry, time);
  }

  template <class Model, class MemorySpace>
  void fill_physical_model_qualified(MultiFab<Dim, MemorySpace>& state,
                                     const Geometry<Dim>& geometry, const Model& model,
                                     const ExecutionLane& lane,
                                     MultiFab<Dim, MemorySpace>& candidate,
                                     Real time = Real(0)) const {
    fill_physical_model_qualified_(state, geometry.domain(), model, lane, candidate, &geometry,
                                   time);
  }

 private:
  template <class Model, class MemorySpace>
  void fill_physical_preflighted_model_(MultiFab<Dim, MemorySpace>& state,
                                       PhysicalFillPreflight<MemorySpace>&& preflight,
                                       const Model& model) const {
    if (preflight.owner_ != this || preflight.state_ != &state ||
        preflight.ncomp_ != state.ncomp() || preflight.ghosts_ != state.ghosts() ||
        preflight.local_size_ != state.local_size() || preflight.layout_ != state.layout() ||
        preflight.distribution_ != state.distribution() ||
        preflight.local_rank_ != state.local_rank())
      throw std::logic_error(
          "prepared hyperbolic boundary received a foreign or stale physical preflight");
    if (has_analytic_primitive()) {
      if constexpr (!hyperbolic_boundary_detail::ExactPrimitiveToConservativeModel<Model>) {
        throw std::logic_error(
            "analytic primitive inflow requires the compiled model's to_conservative provider");
      } else if (Model::n_vars != ncomp()) {
        throw std::invalid_argument(
            "analytic primitive inflow component layout differs from the boundary state");
      }
    }
    preflight.owner_ = nullptr;
    fill_axes_<0>(state, preflight.domain_,
                  preflight.geometry_ ? &*preflight.geometry_ : nullptr, preflight.time_);
    if (has_analytic_primitive()) {
      if constexpr (hyperbolic_boundary_detail::ExactPrimitiveToConservativeModel<Model>) {
        const Real failure = fill_analytic_primitive_axes_<0>(
            state, preflight.domain_,
            preflight.geometry_ ? &*preflight.geometry_ : nullptr, preflight.time_, model);
        if (failure != Real(0))
          throw std::runtime_error(
              "analytic primitive inflow conversion produced a non-finite or invalid state");
      }
    }
    device_fence();
  }

  template <class Model, class MemorySpace>
  void fill_physical_model_qualified_(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                                      const Model& model, const ExecutionLane& lane,
                                      MultiFab<Dim, MemorySpace>& candidate,
                                      const Geometry<Dim>* geometry, Real time) const {
    long invalid_lane = 0;
    try {
      invalid_lane = lane.size() != static_cast<int>(state.rank_space().size()) ||
                             lane.rank() != static_cast<int>(
                                                state.rank_space().linear_rank(state.local_rank()))
                         ? 1L
                         : 0L;
    } catch (...) {
      invalid_lane = 1;
    }
    if (all_reduce_max(invalid_lane, lane.communicator()) != 0)
      throw std::invalid_argument(
          "characteristic no-inflow requires its exact prepared ExecutionLane");

    std::exception_ptr provider_error;
    try {
      require_model_qualified_characteristic_provider<Model>();
    } catch (...) {
      provider_error = std::current_exception();
    }
    if (all_reduce_max(provider_error ? 1L : 0L, lane.communicator()) != 0) {
      if (lane.size() == 1 && provider_error)
        std::rethrow_exception(provider_error);
      throw std::invalid_argument(
          "characteristic no-inflow provider contract is invalid on at least one rank");
    }

    const long characteristic = has_characteristic_no_inflow() ? 1L : 0L;
    const long minimum_characteristic = all_reduce_min(characteristic, lane.communicator());
    const long maximum_characteristic = all_reduce_max(characteristic, lane.communicator());
    if (minimum_characteristic != maximum_characteristic)
      throw std::invalid_argument(
          "characteristic no-inflow boundary selection differs between execution ranks");
    if (!has_characteristic_no_inflow()) {
      if (has_analytic_primitive()) {
        auto preflight = geometry == nullptr ? preflight_physical(state, domain, time)
                                             : preflight_physical(state, *geometry, time);
        fill_physical_preflighted_model_(state, std::move(preflight), model);
      } else if (geometry != nullptr) {
        fill_physical(state, *geometry, time);
      } else {
        fill_physical(state, domain, time);
      }
      return;
    }
    if constexpr (!hyperbolic_boundary_detail::ExactCharacteristicNoInflowModel<Model, Dim>) {
      throw std::logic_error(
          "characteristic no-inflow requires the generated model's exact conservative ND "
          "provider contract");
    } else {
      long invalid_contract = 0;
      try {
        validate_physical_contract_(state, domain, /*allow_characteristic=*/true, geometry);
      } catch (...) {
        invalid_contract = 1;
      }
      if (all_reduce_max(invalid_contract, lane.communicator()) != 0)
        throw std::invalid_argument(
            "characteristic no-inflow physical contract is invalid on at least one rank");

      std::exception_ptr staging_error;
      try {
        if (&candidate == &state || candidate.layout() != state.layout() ||
            candidate.distribution() != state.distribution() ||
            candidate.local_rank() != state.local_rank() ||
            candidate.local_size() != state.local_size() || candidate.ncomp() != state.ncomp() ||
            candidate.ghosts() != state.ghosts())
          throw std::invalid_argument(
              "characteristic no-inflow candidate differs from its prepared state contract");
        for (std::size_t local = 0; local < state.local_size(); ++local) {
          if (candidate.global_index(local) != state.global_index(local) ||
              candidate.fab(local).box() != state.fab(local).box() ||
              candidate.fab(local).grown_box() != state.fab(local).grown_box())
            throw std::invalid_argument(
                "characteristic no-inflow candidate differs from local patch storage");
          Kokkos::deep_copy(candidate.fab(local).storage(), state.fab(local).storage());
        }
        device_fence();
      } catch (...) {
        staging_error = std::current_exception();
      }
      if (all_reduce_max(staging_error ? 1L : 0L, lane.communicator()) != 0) {
        if (lane.size() == 1 && staging_error)
          std::rethrow_exception(staging_error);
        throw std::runtime_error(
            "characteristic no-inflow candidate staging failed on at least one rank");
      }

      Real failure = Real(0);
      Real candidate_failure = Real(0);
      std::exception_ptr materialization_error;
      try {
        fill_axes_<0>(candidate, domain, geometry, time);
        if (has_analytic_primitive()) {
          if constexpr (hyperbolic_boundary_detail::ExactPrimitiveToConservativeModel<Model>) {
            if (Model::n_vars != ncomp())
              throw std::invalid_argument(
                  "analytic primitive inflow component layout differs from the boundary state");
            failure = std::max(failure, fill_analytic_primitive_axes_<0>(candidate, domain, geometry,
                                                                        time, model));
          } else {
            throw std::logic_error(
                "analytic primitive inflow requires the compiled model's to_conservative provider");
          }
        }
        failure = std::max(failure, fill_characteristic_axes_<0>(candidate, domain, model));
        for (std::size_t local = 0; local < candidate.local_size(); ++local)
          candidate_failure =
              std::max(candidate_failure,
                       for_each_cell_reduce_max(
                           candidate.fab(local).grown_box(),
                           hyperbolic_boundary_detail::ValidateBoundaryCandidate<Dim>{
                               std::as_const(candidate.fab(local)).view(), candidate.ncomp()}));
        device_fence();
      } catch (...) {
        materialization_error = std::current_exception();
      }
      const long rejected =
          materialization_error || failure != Real(0) || candidate_failure != Real(0) ? 1L : 0L;
      const long collective_rejected = all_reduce_max(rejected, lane.communicator());
      if (collective_rejected != 0) {
        if (lane.size() == 1 && materialization_error)
          std::rethrow_exception(materialization_error);
        throw std::runtime_error(
            "generated characteristic no-inflow provider rejected a non-finite or invalid state");
      }

      for (std::size_t local = 0; local < state.local_size(); ++local) {
        const Fab<Dim, MemorySpace>& candidate_fab = candidate.fab(local);
        for_each_cell(state.fab(local).grown_box(),
                      hyperbolic_boundary_detail::PublishBoundaryCandidate<Dim>{
                          candidate_fab.view(), state.fab(local).view(), state.ncomp()});
      }
      device_fence();
    }
  }

 public:
  /// Apply post-Riemann physical face laws to one patch-local integrated flux field.  NoFlux is
  /// deliberately enforced here rather than by its extrapolated ghost trace: equal traces can
  /// still carry a non-zero advective flux.  All face regions are validated before the first
  /// candidate flux is changed.
  template <class MemorySpace>
  void apply_physical_flux_conditions(nd::FaceField<Dim, MemorySpace>& integrated_fluxes,
                                      const Box<Dim>& domain) const {
    if (domain.empty() || integrated_fluxes.cell_box().empty() ||
        !domain.contains(integrated_fluxes.cell_box()))
      throw std::invalid_argument(
          "prepared hyperbolic face flux patch must be non-empty and inside the domain");
    if (integrated_fluxes.ncomp() != ncomp())
      throw std::invalid_argument(
          "prepared hyperbolic face flux component count differs from the boundary model");
    validate_flux_axes_<0>(integrated_fluxes, domain);
    apply_flux_axes_<0>(integrated_fluxes, domain);
    device_fence();
  }

 private:
  template <class MemorySpace>
  void validate_physical_contract_(const MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                                   bool allow_characteristic,
                                   const Geometry<Dim>* geometry) const {
    if (requires_fixed_state_conversion() && !has_analytic_primitive())
      throw std::logic_error(
          "primitive fixed-state boundary reached execution before model conversion");
    if (has_analytic_state() && geometry == nullptr)
      throw std::logic_error(
          "analytic hyperbolic boundary requires a requalified ND coordinate provider");
    if (has_analytic_state() && geometry != nullptr && !(geometry->domain() == domain))
      throw std::invalid_argument(
          "analytic hyperbolic geometry domain differs from the prepared fill domain");
    if (has_analytic_state()) {
      for (int axis = 0; axis < Dim; ++axis) {
        if (!faces_[static_cast<std::size_t>(2 * axis)].analytic_state.empty() ||
            !faces_[static_cast<std::size_t>(2 * axis + 1)].analytic_state.empty()) {
          if (state.ghosts()[axis] > domain.length(axis))
            throw std::invalid_argument(
                "analytic ghost depth may not exceed the normal domain extent");
        }
      }
    }
    if (has_characteristic_no_inflow() && !allow_characteristic)
      throw std::logic_error(
          "characteristic no-inflow requires the generated model's exact conservative ND "
          "provider contract");
    if (domain.empty())
      throw std::invalid_argument("prepared hyperbolic boundary domain must be non-empty");
    if (state.ncomp() != ncomp())
      throw std::invalid_argument(
          "prepared hyperbolic boundary component count differs from the state");
    for (std::size_t local = 0; local < state.local_size(); ++local)
      if (!domain.contains(state.box(local)))
        throw std::invalid_argument(
            "prepared hyperbolic boundary state patch lies outside the physical domain");

    validate_axes_<0>(domain, state.ghosts());
    validate_patch_regions_<0>(state, domain, allow_characteristic);
  }

  template <int Axis, int Side>
  static Box<Dim> boundary_face_region_(const Box<Dim>& cells) {
    static_assert(Axis >= 0 && Axis < Dim);
    static_assert(Side == -1 || Side == 1);
    Box<Dim> region = cells;
    const std::int64_t coordinate = Side < 0 ? static_cast<std::int64_t>(cells.lo[Axis])
                                             : static_cast<std::int64_t>(cells.hi[Axis]) + 1;
    region.lo[Axis] = region.hi[Axis] =
        detail::checked_box_index(coordinate, "prepared hyperbolic boundary face index overflow");
    return region;
  }

  template <int Axis, class MemorySpace>
  void validate_flux_axes_(const nd::FaceField<Dim, MemorySpace>& integrated_fluxes,
                           const Box<Dim>& domain) const {
    const Box<Dim>& cells = integrated_fluxes.cell_box();
    const Box<Dim> expected = nd::face_box<Axis>(cells);
    const auto& field = integrated_fluxes.template field<Axis>();
    if (!(field.box() == expected) || field.ncomp() != ncomp())
      throw std::invalid_argument(
          "prepared hyperbolic boundary received an inconsistent face field");
    if (cells.lo[Axis] == domain.lo[Axis] &&
        faces_[static_cast<std::size_t>(2 * Axis)].law == HyperbolicBoundaryLaw::NoFlux)
      detail::require_iterable_box(boundary_face_region_<Axis, -1>(cells));
    if (cells.hi[Axis] == domain.hi[Axis] &&
        faces_[static_cast<std::size_t>(2 * Axis + 1)].law == HyperbolicBoundaryLaw::NoFlux)
      detail::require_iterable_box(boundary_face_region_<Axis, 1>(cells));
    if constexpr (Axis + 1 < Dim)
      validate_flux_axes_<Axis + 1>(integrated_fluxes, domain);
  }

  template <int Axis, class MemorySpace>
  void apply_flux_axes_(nd::FaceField<Dim, MemorySpace>& integrated_fluxes,
                        const Box<Dim>& domain) const {
    const Box<Dim>& cells = integrated_fluxes.cell_box();
    if (cells.lo[Axis] == domain.lo[Axis] &&
        faces_[static_cast<std::size_t>(2 * Axis)].law == HyperbolicBoundaryLaw::NoFlux)
      for_each_cell(boundary_face_region_<Axis, -1>(cells),
                    hyperbolic_boundary_detail::ZeroBoundaryFaceFlux<Dim>{
                        integrated_fluxes.template field<Axis>().view(), ncomp()});
    if (cells.hi[Axis] == domain.hi[Axis] &&
        faces_[static_cast<std::size_t>(2 * Axis + 1)].law == HyperbolicBoundaryLaw::NoFlux)
      for_each_cell(boundary_face_region_<Axis, 1>(cells),
                    hyperbolic_boundary_detail::ZeroBoundaryFaceFlux<Dim>{
                        integrated_fluxes.template field<Axis>().view(), ncomp()});
    if constexpr (Axis + 1 < Dim)
      apply_flux_axes_<Axis + 1>(integrated_fluxes, domain);
  }

  template <int Axis>
  void validate_axes_(const Box<Dim>& domain, const Extent<Dim>& ghosts) const {
    const auto table = table_view_();
    const HyperbolicBoundaryLaw low = faces_[static_cast<std::size_t>(2 * Axis)].law;
    const HyperbolicBoundaryLaw high = faces_[static_cast<std::size_t>(2 * Axis + 1)].law;
    for (int component = 0; component < ncomp(); ++component) {
      for (std::int64_t offset = 1; offset <= ghosts[Axis]; ++offset) {
        if (hyperbolic_boundary_detail::is_builtin_physical_law(low) &&
            faces_[static_cast<std::size_t>(2 * Axis)].analytic_state.empty())
          hyperbolic_boundary_detail::validate_extension(
              detail::checked_box_index(static_cast<std::int64_t>(domain.lo[Axis]) - offset,
                                        "prepared hyperbolic lower halo index overflow"),
              domain.lo[Axis], domain.hi[Axis], Axis, low, high, table, component);
        if (hyperbolic_boundary_detail::is_builtin_physical_law(high) &&
            faces_[static_cast<std::size_t>(2 * Axis + 1)].analytic_state.empty())
          hyperbolic_boundary_detail::validate_extension(
              detail::checked_box_index(static_cast<std::int64_t>(domain.hi[Axis]) + offset,
                                        "prepared hyperbolic upper halo index overflow"),
              domain.lo[Axis], domain.hi[Axis], Axis, low, high, table, component);
      }
    }
    if constexpr (Axis + 1 < Dim)
      validate_axes_<Axis + 1>(domain, ghosts);
  }

  template <int Axis, class MemorySpace>
  void validate_patch_regions_(const MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                               bool allow_characteristic) const {
    if (state.ghosts()[Axis] == 0) {
      if constexpr (Axis + 1 < Dim)
        validate_patch_regions_<Axis + 1>(state, domain, allow_characteristic);
      return;
    }
    const auto executable = [allow_characteristic](const PreparedHyperbolicFace& face) {
      return hyperbolic_boundary_detail::is_builtin_physical_law(face.law) ||
             !face.analytic_state.empty() ||
             (allow_characteristic && face.law == HyperbolicBoundaryLaw::CharacteristicNoInflow);
    };
    const auto& low_face = faces_[static_cast<std::size_t>(2 * Axis)];
    const auto& high_face = faces_[static_cast<std::size_t>(2 * Axis + 1)];
    for (std::size_t local = 0; local < state.local_size(); ++local) {
      const Fab<Dim, MemorySpace>& fab = state.fab(local);
      const Box<Dim> valid = fab.box();
      if (executable(low_face) && valid.lo[Axis] == domain.lo[Axis] &&
          !fab.grown_box().contains(physical_region_<Axis, -1>(valid, domain, state.ghosts())))
        throw std::invalid_argument(
            "prepared hyperbolic lower physical region exceeds the Fab ghost storage");
      if (executable(high_face) && valid.hi[Axis] == domain.hi[Axis] &&
          !fab.grown_box().contains(physical_region_<Axis, 1>(valid, domain, state.ghosts())))
        throw std::invalid_argument(
            "prepared hyperbolic upper physical region exceeds the Fab ghost storage");
    }
    if constexpr (Axis + 1 < Dim)
      validate_patch_regions_<Axis + 1>(state, domain, allow_characteristic);
  }

  template <int Axis, class Model, class MemorySpace>
  Real fill_characteristic_axes_(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                                 const Model& model) const {
    Real failure = Real(0);
    const auto table = table_view_();
    if (state.ghosts()[Axis] > 0) {
      for (std::size_t local = 0; local < state.local_size(); ++local) {
        Fab<Dim, MemorySpace>& fab = state.fab(local);
        const Box<Dim> valid = fab.box();
        if (valid.lo[Axis] == domain.lo[Axis] &&
            faces_[static_cast<std::size_t>(2 * Axis)].law ==
                HyperbolicBoundaryLaw::CharacteristicNoInflow) {
          const Box<Dim> region = physical_region_<Axis, -1>(valid, domain, state.ghosts());
          failure = std::max(
              failure,
              for_each_cell_reduce_max(
                  region,
                  hyperbolic_boundary_detail::FillCharacteristicNoInflowFace<Axis, -1, Dim, Model>{
                      model, fab.view(), table, domain.lo[Axis]}));
        }
        if (valid.hi[Axis] == domain.hi[Axis] &&
            faces_[static_cast<std::size_t>(2 * Axis + 1)].law ==
                HyperbolicBoundaryLaw::CharacteristicNoInflow) {
          const Box<Dim> region = physical_region_<Axis, 1>(valid, domain, state.ghosts());
          failure = std::max(
              failure,
              for_each_cell_reduce_max(
                  region,
                  hyperbolic_boundary_detail::FillCharacteristicNoInflowFace<Axis, 1, Dim, Model>{
                      model, fab.view(), table, domain.hi[Axis]}));
        }
      }
    }
    if constexpr (Axis + 1 < Dim)
      failure = std::max(failure, fill_characteristic_axes_<Axis + 1>(state, domain, model));
    return failure;
  }

  template <int Axis, class MemorySpace>
  void fill_axes_(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                  const Geometry<Dim>* geometry, Real time) const {
    const HyperbolicBoundaryLaw low = faces_[static_cast<std::size_t>(2 * Axis)].law;
    const HyperbolicBoundaryLaw high = faces_[static_cast<std::size_t>(2 * Axis + 1)].law;
    const auto table = table_view_();
    const std::int64_t depth = state.ghosts()[Axis];
    if (depth > 0) {
      for (std::size_t local = 0; local < state.local_size(); ++local) {
        Fab<Dim, MemorySpace>& fab = state.fab(local);
        const Box<Dim> valid = fab.box();
        const auto& low_face = faces_[static_cast<std::size_t>(2 * Axis)];
        const auto& high_face = faces_[static_cast<std::size_t>(2 * Axis + 1)];
        if (valid.lo[Axis] == domain.lo[Axis] && !low_face.analytic_state.empty() &&
            low_face.authored_representation != HyperbolicStateRepresentation::Primitive) {
          if (geometry == nullptr)
            throw std::logic_error(
                "analytic hyperbolic boundary requires a requalified ND coordinate provider");
          const Box<Dim> region = physical_region_<Axis, -1>(valid, domain, state.ghosts());
          if (!fab.grown_box().contains(region))
            throw std::logic_error(
                "prepared hyperbolic lower physical region exceeds the Fab ghost storage");
          hyperbolic_boundary_detail::ValidateAnalyticFace<Axis, Dim> validate{
              std::as_const(fab).view(), *geometry};
          hyperbolic_boundary_detail::FillAnalyticFace<Axis, Dim> fill{fab.view(), *geometry};
          validate.ncomp = fill.ncomp = ncomp();
          validate.time = fill.time = time;
          validate.lo = fill.lo = domain.lo[Axis];
          validate.hi = fill.hi = domain.hi[Axis];
          validate.side = fill.side = -1;
          for (int component = 0; component < ncomp(); ++component)
            validate.programs[component] = fill.programs[component] =
                low_face.analytic_state[static_cast<std::size_t>(component)].view();
          if (for_each_cell_reduce_max(region, validate) != Real(0))
            throw std::runtime_error("analytic hyperbolic boundary produced a non-finite value");
          for_each_cell(region, fill);
        } else if (hyperbolic_boundary_detail::is_builtin_physical_law(low) &&
                   low_face.analytic_state.empty() && valid.lo[Axis] == domain.lo[Axis]) {
          const Box<Dim> region = physical_region_<Axis, -1>(valid, domain, state.ghosts());
          if (!fab.grown_box().contains(region))
            throw std::logic_error(
                "prepared hyperbolic lower physical region exceeds the Fab ghost storage");
          for_each_cell(region,
                        hyperbolic_boundary_detail::FillPhysicalFace<Axis, Dim>{
                            fab.view(), table, domain.lo[Axis], domain.hi[Axis], low, high});
        }
        if (valid.hi[Axis] == domain.hi[Axis] && !high_face.analytic_state.empty() &&
            high_face.authored_representation != HyperbolicStateRepresentation::Primitive) {
          if (geometry == nullptr)
            throw std::logic_error(
                "analytic hyperbolic boundary requires a requalified ND coordinate provider");
          const Box<Dim> region = physical_region_<Axis, 1>(valid, domain, state.ghosts());
          if (!fab.grown_box().contains(region))
            throw std::logic_error(
                "prepared hyperbolic upper physical region exceeds the Fab ghost storage");
          hyperbolic_boundary_detail::ValidateAnalyticFace<Axis, Dim> validate{
              std::as_const(fab).view(), *geometry};
          hyperbolic_boundary_detail::FillAnalyticFace<Axis, Dim> fill{fab.view(), *geometry};
          validate.ncomp = fill.ncomp = ncomp();
          validate.time = fill.time = time;
          validate.lo = fill.lo = domain.lo[Axis];
          validate.hi = fill.hi = domain.hi[Axis];
          validate.side = fill.side = 1;
          for (int component = 0; component < ncomp(); ++component)
            validate.programs[component] = fill.programs[component] =
                high_face.analytic_state[static_cast<std::size_t>(component)].view();
          if (for_each_cell_reduce_max(region, validate) != Real(0))
            throw std::runtime_error("analytic hyperbolic boundary produced a non-finite value");
          for_each_cell(region, fill);
        } else if (hyperbolic_boundary_detail::is_builtin_physical_law(high) &&
                   high_face.analytic_state.empty() && valid.hi[Axis] == domain.hi[Axis]) {
          const Box<Dim> region = physical_region_<Axis, 1>(valid, domain, state.ghosts());
          if (!fab.grown_box().contains(region))
            throw std::logic_error(
                "prepared hyperbolic upper physical region exceeds the Fab ghost storage");
          for_each_cell(region,
                        hyperbolic_boundary_detail::FillPhysicalFace<Axis, Dim>{
                            fab.view(), table, domain.lo[Axis], domain.hi[Axis], low, high});
        }
      }
    }
    if constexpr (Axis + 1 < Dim)
      fill_axes_<Axis + 1>(state, domain, geometry, time);
  }

  template <int Axis, class Model, class MemorySpace>
  Real fill_analytic_primitive_axes_(MultiFab<Dim, MemorySpace>& state, const Box<Dim>& domain,
                                     const Geometry<Dim>* geometry, Real time,
                                     const Model& model) const {
    Real failure = Real(0);
    if (state.ghosts()[Axis] > 0) {
      if (geometry == nullptr)
        throw std::logic_error(
            "analytic hyperbolic boundary requires a requalified ND coordinate provider");
      for (std::size_t local = 0; local < state.local_size(); ++local) {
        Fab<Dim, MemorySpace>& fab = state.fab(local);
        const Box<Dim> valid = fab.box();
        const auto& low_face = faces_[static_cast<std::size_t>(2 * Axis)];
        const auto& high_face = faces_[static_cast<std::size_t>(2 * Axis + 1)];
        if (valid.lo[Axis] == domain.lo[Axis] && !low_face.analytic_state.empty() &&
            low_face.authored_representation == HyperbolicStateRepresentation::Primitive) {
          hyperbolic_boundary_detail::FillAnalyticPrimitiveFace<Axis, Dim, Model> fill{
              model, fab.view(), *geometry};
          fill.ncomp = ncomp();
          fill.time = time;
          fill.lo = domain.lo[Axis];
          fill.hi = domain.hi[Axis];
          fill.side = -1;
          for (int component = 0; component < ncomp(); ++component)
            fill.programs[component] =
                low_face.analytic_state[static_cast<std::size_t>(component)].view();
          failure = std::max(
              failure, for_each_cell_reduce_max(
                           physical_region_<Axis, -1>(valid, domain, state.ghosts()), fill));
        }
        if (valid.hi[Axis] == domain.hi[Axis] && !high_face.analytic_state.empty() &&
            high_face.authored_representation == HyperbolicStateRepresentation::Primitive) {
          hyperbolic_boundary_detail::FillAnalyticPrimitiveFace<Axis, Dim, Model> fill{
              model, fab.view(), *geometry};
          fill.ncomp = ncomp();
          fill.time = time;
          fill.lo = domain.lo[Axis];
          fill.hi = domain.hi[Axis];
          fill.side = 1;
          for (int component = 0; component < ncomp(); ++component)
            fill.programs[component] =
                high_face.analytic_state[static_cast<std::size_t>(component)].view();
          failure = std::max(
              failure, for_each_cell_reduce_max(
                           physical_region_<Axis, 1>(valid, domain, state.ghosts()), fill));
        }
      }
    }
    if constexpr (Axis + 1 < Dim)
      failure = std::max(failure, fill_analytic_primitive_axes_<Axis + 1>(state, domain, geometry,
                                                                         time, model));
    return failure;
  }

  template <int Axis, int Side>
  Box<Dim> physical_region_(const Box<Dim>& valid, const Box<Dim>& domain,
                            const Extent<Dim>& ghosts) const {
    static_assert(Side == -1 || Side == 1);
    Box<Dim> region = valid;
    for (int tangent = 0; tangent < Dim; ++tangent) {
      if (tangent == Axis)
        continue;
      region.lo[tangent] =
          detail::checked_box_index(static_cast<std::int64_t>(valid.lo[tangent]) - ghosts[tangent],
                                    "prepared hyperbolic tangent lower index overflow");
      region.hi[tangent] =
          detail::checked_box_index(static_cast<std::int64_t>(valid.hi[tangent]) + ghosts[tangent],
                                    "prepared hyperbolic tangent upper index overflow");
      if (faces_[static_cast<std::size_t>(2 * tangent)].law != HyperbolicBoundaryLaw::Periodic)
        region.lo[tangent] =
            region.lo[tangent] < domain.lo[tangent] ? domain.lo[tangent] : region.lo[tangent];
      if (faces_[static_cast<std::size_t>(2 * tangent + 1)].law != HyperbolicBoundaryLaw::Periodic)
        region.hi[tangent] =
            region.hi[tangent] > domain.hi[tangent] ? domain.hi[tangent] : region.hi[tangent];
    }
    if constexpr (Side < 0) {
      region.lo[Axis] =
          detail::checked_box_index(static_cast<std::int64_t>(domain.lo[Axis]) - ghosts[Axis],
                                    "prepared hyperbolic lower region index overflow");
      region.hi[Axis] =
          detail::checked_box_index(static_cast<std::int64_t>(domain.lo[Axis]) - 1,
                                    "prepared hyperbolic lower region boundary index overflow");
    } else {
      region.lo[Axis] =
          detail::checked_box_index(static_cast<std::int64_t>(domain.hi[Axis]) + 1,
                                    "prepared hyperbolic upper region boundary index overflow");
      region.hi[Axis] =
          detail::checked_box_index(static_cast<std::int64_t>(domain.hi[Axis]) + ghosts[Axis],
                                    "prepared hyperbolic upper region index overflow");
    }
    return region;
  }

  void validate_() const {
    if (component_transforms_.empty())
      throw std::invalid_argument(
          "prepared hyperbolic boundary requires model-qualified components");
    if (corner_policy_ != HyperbolicCornerPolicy::NotRequired)
      throw std::invalid_argument("unsupported hyperbolic corner policy");
    for (int axis = 0; axis < Dim; ++axis) {
      const auto& low = faces_[static_cast<std::size_t>(2 * axis)];
      const auto& high = faces_[static_cast<std::size_t>(2 * axis + 1)];
      if (!explicit_periodic_identifications_ && ((low.law == HyperbolicBoundaryLaw::Periodic) !=
                                                  (high.law == HyperbolicBoundaryLaw::Periodic)))
        throw std::invalid_argument(
            "prepared hyperbolic periodic topology requires complete axis pairs");
    }
    for (int ordinal = 0; ordinal < 2 * Dim; ++ordinal) {
      const auto& prepared = faces_[static_cast<std::size_t>(ordinal)];
      if (prepared.identity.empty() || prepared.identity_token == 0)
        throw std::invalid_argument("prepared hyperbolic faces require owner-qualified identities");
      if (!prepared.analytic_state.empty()) {
        if (prepared.law != HyperbolicBoundaryLaw::FixedState)
          throw std::invalid_argument("analytic hyperbolic programs require fixed-state inflow");
        if (static_cast<int>(prepared.analytic_state.size()) != ncomp())
          throw std::invalid_argument(
              "analytic hyperbolic faces must provide one program per component");
        if (std::any_of(prepared.analytic_state.begin(), prepared.analytic_state.end(),
                        [](const analytic::AnalyticProgram& program) { return program.empty(); }))
          throw std::invalid_argument("analytic hyperbolic programs must be non-empty");
        if (prepared.law == HyperbolicBoundaryLaw::CharacteristicNoInflow)
          throw std::invalid_argument(
              "characteristic no-inflow cannot carry analytic programs");
      } else if (!prepared.analytic_clock.empty()) {
        throw std::invalid_argument(
            "analytic hyperbolic Clock requires a compiled analytic program on that face");
      }
      if (prepared.law == HyperbolicBoundaryLaw::FixedState ||
          prepared.law == HyperbolicBoundaryLaw::CharacteristicNoInflow) {
        if (prepared.fixed_state.size() != component_transforms_.size() ||
            std::any_of(prepared.fixed_state.begin(), prepared.fixed_state.end(),
                        [](Real value) { return !std::isfinite(value); }))
          throw std::invalid_argument(
              "fixed-state hyperbolic boundary must provide one finite value per component");
        if (prepared.law == HyperbolicBoundaryLaw::CharacteristicNoInflow &&
            prepared.authored_representation != HyperbolicStateRepresentation::Conservative)
          throw std::invalid_argument(
              "characteristic no-inflow requires a conservative reference state");
        if (prepared.authored_representation == HyperbolicStateRepresentation::Primitive) {
          if (prepared.converter_identity.empty())
            throw std::invalid_argument(
                "primitive fixed-state boundary requires one converter identity");
        } else if (!prepared.converter_identity.empty() ||
                   (!prepared.fixed_state_converted && prepared.analytic_state.empty())) {
          throw std::invalid_argument(
              "conservative fixed-state boundary must not carry conversion metadata");
        }
      } else if (!prepared.fixed_state.empty()) {
        throw std::invalid_argument(
            "only fixed-state or characteristic no-inflow boundaries may carry component values");
      } else if (prepared.authored_representation != HyperbolicStateRepresentation::Conservative ||
                 !prepared.converter_identity.empty() || !prepared.fixed_state_converted) {
        throw std::invalid_argument(
            "only a fixed-state hyperbolic boundary may carry conversion metadata");
      }
      if (prepared.law == HyperbolicBoundaryLaw::ReflectiveSlip) {
        const int normal_axis = ordinal / 2;
        const bool owns_normal_polar =
            std::any_of(component_transforms_.begin(), component_transforms_.end(),
                        [normal_axis](const Transform& transform) {
                          return transform.parity == HyperbolicComponentParity::PolarVector &&
                                 transform.axis == normal_axis;
                        });
        if (!owns_normal_polar)
          throw std::invalid_argument(
              "reflective slip wall requires a declared normal polar-vector component");
      }
    }
  }

  void prepare_device_tables_() {
    const std::size_t components = component_transforms_.size();
    std::vector<Real> fixed(static_cast<std::size_t>(2 * Dim) * components, Real(0));
    for (int ordinal = 0; ordinal < 2 * Dim; ++ordinal) {
      const auto& source = faces_[static_cast<std::size_t>(ordinal)].fixed_state;
      if (!source.empty())
        std::copy(source.begin(), source.end(),
                  fixed.begin() + static_cast<std::ptrdiff_t>(ordinal * components));
    }
    detail::ensure_kokkos_initialized();
    device_transforms_ =
        Kokkos::View<Transform*, Kokkos::SharedSpace>("pops_nd_boundary_transforms", components);
    device_fixed_values_ =
        Kokkos::View<Real*, Kokkos::SharedSpace>("pops_nd_boundary_fixed_values", fixed.size());
    auto host_transforms = Kokkos::create_mirror_view(device_transforms_);
    auto host_fixed = Kokkos::create_mirror_view(device_fixed_values_);
    for (std::size_t index = 0; index < components; ++index)
      host_transforms(index) = component_transforms_[index];
    for (std::size_t index = 0; index < fixed.size(); ++index)
      host_fixed(index) = fixed[index];
    Kokkos::deep_copy(device_transforms_, host_transforms);
    Kokkos::deep_copy(device_fixed_values_, host_fixed);
  }

  hyperbolic_boundary_detail::BoundaryTableView<Dim> table_view_() const {
    return {device_transforms_.data(), device_fixed_values_.data(), ncomp()};
  }

  std::array<PreparedHyperbolicFace, 2 * Dim> faces_{};
  std::vector<Transform> component_transforms_;
  HyperbolicCornerPolicy corner_policy_ = HyperbolicCornerPolicy::NotRequired;
  bool explicit_periodic_identifications_ = false;
  Kokkos::View<Transform*, Kokkos::SharedSpace> device_transforms_;
  Kokkos::View<Real*, Kokkos::SharedSpace> device_fixed_values_;
  std::optional<Geometry<Dim>> prepared_geometry_{};
};

/// Sole parser from installed Python/native tables into the typed boundary authority.
template <int Dim>
PreparedHyperbolicBoundary<Dim> prepare_hyperbolic_boundary(
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, bool explicit_periodic_identifications = false,
    const std::vector<std::string>& face_representations = {},
    const std::vector<std::string>& face_converter_identities = {},
    const std::vector<std::vector<std::string>>& face_analytic_opcodes = {},
    const std::vector<std::vector<double>>& face_analytic_literals = {},
    const std::vector<std::string>& face_analytic_clocks = {}) {
  if (face_types.size() != static_cast<std::size_t>(2 * Dim) ||
      face_identities.size() != static_cast<std::size_t>(2 * Dim))
    throw std::invalid_argument(
        "prepared hyperbolic boundary requires one type and identity per oriented face");
  if (component_roles.empty() ||
      face_values.size() != component_roles.size() * static_cast<std::size_t>(2 * Dim))
    throw std::invalid_argument(
        "prepared hyperbolic boundary values must be component-major and total");
  if ((!face_representations.empty() &&
       face_representations.size() != static_cast<std::size_t>(2 * Dim)) ||
      (!face_converter_identities.empty() &&
       face_converter_identities.size() != static_cast<std::size_t>(2 * Dim)))
    throw std::invalid_argument(
        "prepared hyperbolic boundary conversion metadata must cover every oriented face");
  const bool has_analytic = !face_analytic_opcodes.empty() || !face_analytic_literals.empty() ||
                            !face_analytic_clocks.empty();
  const std::size_t face_count = static_cast<std::size_t>(2 * Dim);
  const std::size_t analytic_rows = face_count * component_roles.size();
  if (has_analytic) {
    if (face_analytic_opcodes.size() != analytic_rows ||
        face_analytic_literals.size() != analytic_rows)
      throw std::invalid_argument(
          "analytic hyperbolic tables must provide one opcode/literal row per face and component");
    if (!face_analytic_clocks.empty() && face_analytic_clocks.size() != face_count)
      throw std::invalid_argument(
          "analytic hyperbolic clocks must be empty or cover every oriented face");
  }

  std::vector<HyperbolicComponentTransform<Dim>> transforms;
  transforms.reserve(component_roles.size());
  for (const auto& role : component_roles)
    transforms.push_back(hyperbolic_boundary_detail::transform_from_role<Dim>(role));

  std::array<PreparedHyperbolicFace, 2 * Dim> faces;
  for (int ordinal = 0; ordinal < 2 * Dim; ++ordinal) {
    auto& destination = faces[static_cast<std::size_t>(ordinal)];
    destination.law =
        hyperbolic_boundary_detail::law_from_token(face_types[static_cast<std::size_t>(ordinal)]);
    destination.identity = face_identities[static_cast<std::size_t>(ordinal)];
    destination.identity_token =
        hyperbolic_boundary_detail::stable_boundary_identity(destination.identity);
    destination.authored_representation = hyperbolic_boundary_detail::representation_from_token(
        face_representations.empty()
            ? std::string_view("conservative")
            : std::string_view(face_representations[static_cast<std::size_t>(ordinal)]));
    destination.converter_identity =
        face_converter_identities.empty()
            ? std::string{}
            : face_converter_identities[static_cast<std::size_t>(ordinal)];

    if (destination.law == HyperbolicBoundaryLaw::NoFlux) {
      for (std::size_t component = 0; component < component_roles.size(); ++component)
        if (face_values[component * static_cast<std::size_t>(2 * Dim) +
                        static_cast<std::size_t>(ordinal)] != 0.0)
          throw std::invalid_argument(
              "a no-flux hyperbolic boundary cannot carry component values");
    }
    if (destination.law == HyperbolicBoundaryLaw::FixedState ||
        destination.law == HyperbolicBoundaryLaw::CharacteristicNoInflow) {
      destination.fixed_state.reserve(component_roles.size());
      for (std::size_t component = 0; component < component_roles.size(); ++component)
        destination.fixed_state.push_back(
            static_cast<Real>(face_values[component * static_cast<std::size_t>(2 * Dim) +
                                          static_cast<std::size_t>(ordinal)]));
      destination.fixed_state_converted =
          destination.authored_representation == HyperbolicStateRepresentation::Conservative;
    }
    if (has_analytic) {
      destination.analytic_clock =
          face_analytic_clocks.empty() ? std::string{}
                                       : face_analytic_clocks[static_cast<std::size_t>(ordinal)];
      std::vector<analytic::AnalyticProgram> programs;
      programs.reserve(component_roles.size());
      bool any_program = false;
      for (std::size_t component = 0; component < component_roles.size(); ++component) {
        const std::size_t row = static_cast<std::size_t>(ordinal) * component_roles.size() + component;
        if (face_analytic_opcodes[row].empty()) {
          if (!face_analytic_literals[row].empty())
            throw std::invalid_argument(
                "analytic hyperbolic empty opcode row cannot carry literals");
          continue;
        }
        any_program = true;
        programs.push_back(hyperbolic_boundary_detail::compile_boundary_analytic_program(
            face_analytic_opcodes[row], face_analytic_literals[row], Dim));
      }
      if (any_program) {
        if (programs.size() != component_roles.size())
          throw std::invalid_argument(
              "analytic hyperbolic faces must provide one program per component");
        if (destination.law != HyperbolicBoundaryLaw::FixedState)
          throw std::invalid_argument("analytic hyperbolic programs require fixed-state inflow");
        destination.analytic_state = std::move(programs);
      } else if (!destination.analytic_clock.empty()) {
        throw std::invalid_argument(
            "analytic hyperbolic Clock requires a compiled analytic program on that face");
      }
    }
  }
  return PreparedHyperbolicBoundary<Dim>(std::move(faces), std::move(transforms),
                                         HyperbolicCornerPolicy::NotRequired,
                                         explicit_periodic_identifications);
}

/// Characteristic no-inflow on a cut-cell / staircase interface normal.
///
/// This is not a Cartesian ghost-face fill: the incoming projector uses the EB area-vector
/// normal from independent face apertures, never a cell-centred axis alias of that cut.
template <int Dim, class Model, class MemorySpace>
void fill_prepared_embedded_characteristic_no_inflow(
    const Model& model, const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& phi,
    const Fab<Dim, MemorySpace>& active, const typename Model::State& reference,
    Fab<Dim, MemorySpace>& exterior, Real theta_min = kEbCutFractionFloor) {
  ::pops::nd::fill_embedded_characteristic_no_inflow(model, state, phi, active, reference, exterior,
                                                     theta_min);
}

template <int Dim, class Model, class MemorySpace>
void fill_prepared_embedded_characteristic_no_inflow(
    const Model& model, const MultiFab<Dim, MemorySpace>& state,
    const MultiFab<Dim, MemorySpace>& phi, const MultiFab<Dim, MemorySpace>& active,
    const typename Model::State& reference, MultiFab<Dim, MemorySpace>& exterior,
    Real theta_min = kEbCutFractionFloor) {
  ::pops::nd::fill_embedded_characteristic_no_inflow(model, state, phi, active, reference, exterior,
                                                     theta_min);
}

}  // namespace pops
