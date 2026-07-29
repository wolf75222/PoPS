/// @file
/// @brief One prepared, model-aware physical-boundary authority for hyperbolic state transport.
///
/// Boundary topology (periodic/external) and physical law (extrapolation, fixed state, reflective
/// slip wall) are represented independently.  Component transforms are resolved from model roles
/// before a numerical loop; face kernels therefore execute one immutable table without model
/// switches, component-index inference, Python callbacks, or per-cell allocation.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/analytic/expression.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

enum class HyperbolicBoundaryLaw { Periodic, Extrapolate, FixedState, ReflectiveSlip, External };

enum class HyperbolicComponentParity { Scalar, PolarVector, AxialVector };

/// Representation in which a fixed-state face was authored.
///
/// Native face kernels consume conservative values. Primitive data therefore remains explicitly
/// pending until the owning compiled block supplies its exact pointwise model conversion.
enum class HyperbolicStateRepresentation { Conservative, Primitive };

/// Reflection behavior of one model-qualified state component.
///
/// A polar vector reverses its normal component at a reflective plane.  An axial vector applies
/// det(R)R, so its normal component is preserved and every tangential component reverses.  Scalars
/// are even.  The axis is a three-dimensional physical-component axis, never inferred from the
/// component index.  It is intentionally independent of @p Dim: a 1D/2D mesh may evolve transverse
/// polar components or an out-of-plane axial component (the usual 2.5D case).
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

/// Exact axis-aligned face context supplied to a prepared physical law.
///
/// The pointer fields are optional device-accessible packs.  Built-in constant/extrapolation/wall
/// laws do not read them; compiled analytic providers may consume them without a Python callback.
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

/// Dimension-split FV stencils do not read double-physical corners.  Such corners are therefore
/// explicitly excluded rather than being assigned an implicit X-then-Y precedence.
enum class HyperbolicCornerPolicy { NotRequired };

namespace detail {

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
struct HyperbolicBoundaryTableView {
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

struct HyperbolicBoundarySample {
  int source;
  Real scale;
  Real offset;
};

template <int Dim>
POPS_HD inline HyperbolicBoundarySample hyperbolic_boundary_sample_1d(
    int index, int lo, int hi, int axis, HyperbolicBoundaryLaw low, HyperbolicBoundaryLaw high,
    const HyperbolicBoundaryTableView<Dim>& table, int component) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const HyperbolicBoundaryLaw law = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (law == HyperbolicBoundaryLaw::Extrapolate) {
      current = boundary;
      break;
    }

    Real face_scale = Real(1);
    Real face_offset = Real(0);
    if (law == HyperbolicBoundaryLaw::FixedState) {
      face_scale = Real(-1);
      const int face = 2 * axis + (below ? 0 : 1);
      face_offset = Real(2) * table.fixed_value(face, component);
    } else if (law == HyperbolicBoundaryLaw::ReflectiveSlip) {
      face_scale = table.transform(component).reflection_sign(axis);
    } else {
      // Installation preflight rejects an extension that can reach periodic/external ownership.
      current = boundary;
      break;
    }
    offset += scale * face_offset;
    scale *= face_scale;
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
  return {static_cast<int>(current), scale, offset};
}

inline bool is_physical_hyperbolic_law(HyperbolicBoundaryLaw law) {
  return law == HyperbolicBoundaryLaw::Extrapolate || law == HyperbolicBoundaryLaw::FixedState ||
         law == HyperbolicBoundaryLaw::ReflectiveSlip;
}

inline const char* hyperbolic_law_name(HyperbolicBoundaryLaw law) {
  switch (law) {
    case HyperbolicBoundaryLaw::Periodic:
      return "periodic";
    case HyperbolicBoundaryLaw::Extrapolate:
      return "extrapolate";
    case HyperbolicBoundaryLaw::FixedState:
      return "fixed_state";
    case HyperbolicBoundaryLaw::ReflectiveSlip:
      return "reflective_slip";
    case HyperbolicBoundaryLaw::External:
      return "external";
  }
  return "unknown";
}

template <int Dim>
inline void validate_hyperbolic_extension(int index, int lo, int hi, int axis,
                                          HyperbolicBoundaryLaw low, HyperbolicBoundaryLaw high,
                                          const HyperbolicBoundaryTableView<Dim>& table,
                                          int component) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const HyperbolicBoundaryLaw law = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (!is_physical_hyperbolic_law(law))
      throw std::invalid_argument(std::string("prepared hyperbolic halo reaches a ") +
                                  hyperbolic_law_name(law) +
                                  " face whose values belong to another topology authority");
    if (law == HyperbolicBoundaryLaw::Extrapolate)
      return;

    Real face_scale = Real(1);
    Real face_offset = Real(0);
    if (law == HyperbolicBoundaryLaw::FixedState) {
      face_scale = Real(-1);
      const int face = 2 * axis + (below ? 0 : 1);
      face_offset = Real(2) * table.fixed_value(face, component);
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

template <int Dim>
struct HyperbolicFaceXKernel {
  Array4 state;
  HyperbolicBoundaryTableView<Dim> table;
  int lo;
  int hi;
  HyperbolicBoundaryLaw low;
  HyperbolicBoundaryLaw high;

  POPS_HD void operator()(int i, int j) const {
    for (int component = 0; component < table.ncomp; ++component) {
      const auto sample = hyperbolic_boundary_sample_1d(i, lo, hi, 0, low, high, table, component);
      state(i, j, component) = sample.scale * state(sample.source, j, component) + sample.offset;
    }
  }
};

template <int Dim>
struct HyperbolicFaceYKernel {
  Array4 state;
  HyperbolicBoundaryTableView<Dim> table;
  int lo;
  int hi;
  HyperbolicBoundaryLaw low;
  HyperbolicBoundaryLaw high;

  POPS_HD void operator()(int i, int j) const {
    for (int component = 0; component < table.ncomp; ++component) {
      const auto sample = hyperbolic_boundary_sample_1d(j, lo, hi, 1, low, high, table, component);
      state(i, j, component) = sample.scale * state(i, sample.source, component) + sample.offset;
    }
  }
};

POPS_HD inline int periodic_index(int index, int lo, int hi) {
  const int count = hi - lo + 1;
  int offset = (index - lo) % count;
  if (offset < 0)
    offset += count;
  return lo + offset;
}

struct AnalyticFixedSource {
  int i = 0;
  int j = 0;
};

template <int Axis>
struct AnalyticFixedFaceEvaluator {
  static_assert(Axis == 0 || Axis == 1);

  // The analytic value is the Dirichlet trace on the physical face, not a ghost-cell sample.
  // AnalyticFixedFaceKernel applies the same affine mirror rule as a constant FixedState face.
  analytic::AnalyticProgramView program;
  Geometry geometry;
  int side;
  bool periodic_tangent;
  Real time;

  POPS_HD analytic::AnalyticEvaluation evaluate(int i, int j) const {
    int coordinate_i = i;
    int coordinate_j = j;
    if constexpr (Axis == 0) {
      if (periodic_tangent)
        coordinate_j = periodic_index(j, geometry.domain.lo[1], geometry.domain.hi[1]);
    } else if (periodic_tangent) {
      coordinate_i = periodic_index(i, geometry.domain.lo[0], geometry.domain.hi[0]);
    }
    const Real x =
        Axis == 0 ? (side < 0 ? geometry.xlo : geometry.xhi) : geometry.x_cell(coordinate_i);
    const Real y =
        Axis == 1 ? (side < 0 ? geometry.ylo : geometry.yhi) : geometry.y_cell(coordinate_j);
    return program.eval_checked(x, y, &time, std::uint8_t{1});
  }

  POPS_HD AnalyticFixedSource source(int i, int j) const {
    if constexpr (Axis == 0) {
      const int boundary = side < 0 ? geometry.domain.lo[0] : geometry.domain.hi[0];
      return {side < 0 ? 2 * boundary - i - 1 : 2 * boundary - i + 1, j};
    } else {
      const int boundary = side < 0 ? geometry.domain.lo[1] : geometry.domain.hi[1];
      return {i, side < 0 ? 2 * boundary - j - 1 : 2 * boundary - j + 1};
    }
  }
};

template <int Axis>
struct AnalyticFixedFaceFiniteKernel {
  AnalyticFixedFaceEvaluator<Axis> evaluator;

  POPS_HD Real operator()(int i, int j) const {
    return evaluator.evaluate(i, j).valid ? Real(0) : Real(1);
  }
};

template <int Axis>
struct AnalyticFixedFaceKernel {
  Array4 state;
  int component;
  AnalyticFixedFaceEvaluator<Axis> evaluator;

  POPS_HD void operator()(int i, int j) const {
    const auto value = evaluator.evaluate(i, j);
    const auto source = evaluator.source(i, j);
    state(i, j, component) = Real(2) * value.value - state(source.i, source.j, component);
  }
};

template <int Dim>
inline HyperbolicComponentTransform<Dim> transform_from_role(std::string_view role) {
  if (role == "MomentumX" || role == "VelocityX")
    return HyperbolicComponentTransform<Dim>::polar_vector(0);
  if (role == "MomentumY" || role == "VelocityY")
    return HyperbolicComponentTransform<Dim>::polar_vector(1);
  if (role == "MomentumZ" || role == "VelocityZ")
    return HyperbolicComponentTransform<Dim>::polar_vector(2);
  if (role == "AxialX")
    return HyperbolicComponentTransform<Dim>::axial_vector(0);
  if (role == "AxialY")
    return HyperbolicComponentTransform<Dim>::axial_vector(1);
  if (role == "AxialZ")
    return HyperbolicComponentTransform<Dim>::axial_vector(2);
  if (role == "Density" || role == "Energy" || role == "Pressure" || role == "Temperature" ||
      role == "Scalar" || role == "Custom")
    return HyperbolicComponentTransform<Dim>::scalar();
  throw std::invalid_argument("unsupported hyperbolic boundary component role '" +
                              std::string(role) + "'");
}

inline HyperbolicBoundaryLaw hyperbolic_law_from_token(std::string_view token) {
  if (token == "periodic")
    return HyperbolicBoundaryLaw::Periodic;
  if (token == "foextrap")
    return HyperbolicBoundaryLaw::Extrapolate;
  if (token == "dirichlet")
    return HyperbolicBoundaryLaw::FixedState;
  if (token == "slip_wall")
    return HyperbolicBoundaryLaw::ReflectiveSlip;
  if (token == "external")
    return HyperbolicBoundaryLaw::External;
  throw std::invalid_argument("unsupported prepared hyperbolic face law '" + std::string(token) +
                              "'");
}

inline HyperbolicStateRepresentation hyperbolic_representation_from_token(std::string_view token) {
  if (token == "conservative")
    return HyperbolicStateRepresentation::Conservative;
  if (token == "primitive")
    return HyperbolicStateRepresentation::Primitive;
  throw std::invalid_argument("unsupported prepared hyperbolic representation '" +
                              std::string(token) + "'");
}

}  // namespace detail

template <int Dim>
class PreparedHyperbolicBoundary {
 public:
  static_assert(Dim >= 1 && Dim <= 3);
  using Transform = HyperbolicComponentTransform<Dim>;

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
    validate(explicit_periodic_identifications);
    prepare_device_tables();
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
  bool has_analytic_state() const {
    return std::any_of(faces_.begin(), faces_.end(), [](const PreparedHyperbolicFace& face) {
      return !face.analytic_state.empty();
    });
  }

  bool requires_fixed_state_conversion() const {
    return std::any_of(faces_.begin(), faces_.end(), [](const PreparedHyperbolicFace& prepared) {
      return prepared.law == HyperbolicBoundaryLaw::FixedState &&
             prepared.authored_representation == HyperbolicStateRepresentation::Primitive &&
             !prepared.fixed_state_converted;
    });
  }

  /// Return a new executable table after converting every pending primitive fixed state.
  ///
  /// Conversion is transactional: this immutable table is untouched if the model conversion
  /// throws or produces a non-finite component.
  PreparedHyperbolicBoundary with_converted_fixed_states(
      const std::function<void(const double*, double*)>& primitive_to_conservative) const {
    if (!requires_fixed_state_conversion())
      return *this;
    if (!primitive_to_conservative)
      throw std::invalid_argument(
          "primitive fixed-state boundary requires the compiled block-model conversion");

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
    return PreparedHyperbolicBoundary(std::move(converted_faces), component_transforms_,
                                      corner_policy_, explicit_periodic_identifications_);
  }

  Periodicity periodicity() const {
    static_assert(Dim == 2, "the current MultiFab topology is two-dimensional");
    const bool xlo = faces_[0].law == HyperbolicBoundaryLaw::Periodic;
    const bool xhi = faces_[1].law == HyperbolicBoundaryLaw::Periodic;
    const bool ylo = faces_[2].law == HyperbolicBoundaryLaw::Periodic;
    const bool yhi = faces_[3].law == HyperbolicBoundaryLaw::Periodic;
    if (xlo != xhi || ylo != yhi)
      throw std::logic_error(
          "axis-permuted periodic topology has no per-axis runtime Periodicity projection");
    return Periodicity{
        xlo,
        ylo,
    };
  }

  /// Fill only physical faces. Same-level/MPI and periodic topology remain owned by fill_boundary.
  ///
  /// The explicit NotRequired corner policy excludes double-physical corners. Periodic tangential
  /// ghosts are included because they were already produced by fill_boundary and are valid inputs.
  void fill_physical(MultiFab& state, const Box2D& domain) const {
    if (has_analytic_state())
      throw std::logic_error(
          "analytic hyperbolic boundary requires physical Geometry at execution");
    fill_physical_impl_(state, domain, nullptr, Real(0), {}, false, world_communicator_view());
  }

  void fill_physical(MultiFab& state, const Geometry& geometry) const {
    fill_physical(state, geometry, world_communicator_view());
  }

  void fill_physical(MultiFab& state, const Geometry& geometry,
                     CommunicatorView communicator) const {
    fill_physical_impl_(state, geometry.domain, &geometry, Real(0), {}, false, communicator);
  }

  void fill_physical(MultiFab& state, const Geometry& geometry, Real physical_time,
                     std::string_view clock) const {
    fill_physical(state, geometry, physical_time, clock, world_communicator_view());
  }

  void fill_physical(MultiFab& state, const Geometry& geometry, Real physical_time,
                     std::string_view clock, CommunicatorView communicator) const {
    fill_physical_impl_(state, geometry.domain, &geometry, physical_time, clock, true,
                        communicator);
  }

 private:
  void fill_physical_impl_(MultiFab& state, const Box2D& domain, const Geometry* geometry,
                           Real physical_time, std::string_view clock, bool has_evaluation_point,
                           CommunicatorView communicator) const {
    static_assert(Dim == 2, "the current MultiFab storage is two-dimensional");
    if (requires_fixed_state_conversion())
      throw std::logic_error(
          "primitive fixed-state boundary reached execution before model conversion");
    if (state.ncomp() != ncomp())
      throw std::invalid_argument(
          "prepared hyperbolic boundary component count differs from the state");
    const int depth = state.n_grow();
    if (depth == 0)
      return;
    if (has_analytic_state()) {
      if (geometry == nullptr || geometry->domain != domain)
        throw std::invalid_argument(
            "analytic hyperbolic boundary requires matching physical Geometry");
      for (int face = 0; face < 2 * Dim; ++face) {
        const auto& prepared = faces_[static_cast<std::size_t>(face)];
        if (prepared.analytic_state.empty())
          continue;
        const int axis_cells = face / 2 == 0 ? domain.nx() : domain.ny();
        if (depth > axis_cells)
          throw std::invalid_argument(
              "analytic hyperbolic boundary does not support multi-reflection ghost depth");
        if (!prepared.analytic_clock.empty() &&
            (!has_evaluation_point || prepared.analytic_clock != clock ||
             !std::isfinite(physical_time)))
          throw std::invalid_argument(
              "analytic hyperbolic boundary requires its exact finite BoundaryEvaluationPoint");
      }
      validate_analytic_values_(state, *geometry, physical_time, communicator);
    }
    const auto table = table_view();
    for (int component = 0; component < ncomp(); ++component) {
      for (int offset = 1; offset <= depth; ++offset) {
        if (detail::is_physical_hyperbolic_law(faces_[0].law))
          detail::validate_hyperbolic_extension(domain.lo[0] - offset, domain.lo[0], domain.hi[0],
                                                0, faces_[0].law, faces_[1].law, table, component);
        if (detail::is_physical_hyperbolic_law(faces_[1].law))
          detail::validate_hyperbolic_extension(domain.hi[0] + offset, domain.lo[0], domain.hi[0],
                                                0, faces_[0].law, faces_[1].law, table, component);
        if (detail::is_physical_hyperbolic_law(faces_[2].law))
          detail::validate_hyperbolic_extension(domain.lo[1] - offset, domain.lo[1], domain.hi[1],
                                                1, faces_[2].law, faces_[3].law, table, component);
        if (detail::is_physical_hyperbolic_law(faces_[3].law))
          detail::validate_hyperbolic_extension(domain.hi[1] + offset, domain.lo[1], domain.hi[1],
                                                1, faces_[2].law, faces_[3].law, table, component);
      }
    }

    for (int local = 0; local < state.local_size(); ++local) {
      Fab2D& fab = state.fab(local);
      const Box2D valid = fab.box();
      const Array4 values = fab.array();

      int tangential_lo = valid.lo[1] - depth;
      int tangential_hi = valid.hi[1] + depth;
      if (faces_[2].law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, domain.lo[1]);
      if (faces_[3].law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, domain.hi[1]);
      if (detail::is_physical_hyperbolic_law(faces_[0].law) && valid.lo[0] == domain.lo[0])
        fill_x_face_(
            values, Box2D{{domain.lo[0] - depth, tangential_lo}, {domain.lo[0] - 1, tangential_hi}},
            0, domain, geometry, physical_time, table);
      if (detail::is_physical_hyperbolic_law(faces_[1].law) && valid.hi[0] == domain.hi[0])
        fill_x_face_(
            values, Box2D{{domain.hi[0] + 1, tangential_lo}, {domain.hi[0] + depth, tangential_hi}},
            1, domain, geometry, physical_time, table);

      tangential_lo = valid.lo[0] - depth;
      tangential_hi = valid.hi[0] + depth;
      if (faces_[0].law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, domain.lo[0]);
      if (faces_[1].law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, domain.hi[0]);
      if (detail::is_physical_hyperbolic_law(faces_[2].law) && valid.lo[1] == domain.lo[1])
        fill_y_face_(
            values, Box2D{{tangential_lo, domain.lo[1] - depth}, {tangential_hi, domain.lo[1] - 1}},
            2, domain, geometry, physical_time, table);
      if (detail::is_physical_hyperbolic_law(faces_[3].law) && valid.hi[1] == domain.hi[1])
        fill_y_face_(
            values, Box2D{{tangential_lo, domain.hi[1] + 1}, {tangential_hi, domain.hi[1] + depth}},
            3, domain, geometry, physical_time, table);
    }
  }

  std::array<PreparedHyperbolicFace, 2 * Dim> faces_{};
  std::vector<Transform> component_transforms_;
  HyperbolicCornerPolicy corner_policy_ = HyperbolicCornerPolicy::NotRequired;
  bool explicit_periodic_identifications_ = false;
#if defined(POPS_HAS_KOKKOS)
  Kokkos::View<Transform*, Kokkos::SharedSpace> device_transforms_;
  Kokkos::View<Real*, Kokkos::SharedSpace> device_fixed_values_;
#else
  std::vector<Transform> device_transforms_;
  std::vector<Real> device_fixed_values_;
#endif

  template <int Axis>
  detail::AnalyticFixedFaceEvaluator<Axis> analytic_evaluator_(int face_ordinal, int component,
                                                               const Geometry& geometry,
                                                               Real physical_time) const {
    const auto& face = faces_[static_cast<std::size_t>(face_ordinal)];
    const bool periodic_tangent = Axis == 0 ? faces_[2].law == HyperbolicBoundaryLaw::Periodic
                                            : faces_[0].law == HyperbolicBoundaryLaw::Periodic;
    return {face.analytic_state[static_cast<std::size_t>(component)].view(), geometry,
            face_ordinal % 2 == 0 ? -1 : 1, periodic_tangent, physical_time};
  }

  void fill_x_face_(const Array4& values, const Box2D& region, int face_ordinal,
                    const Box2D& domain, const Geometry* geometry, Real physical_time,
                    const detail::HyperbolicBoundaryTableView<Dim>& table) const {
    const auto& face = faces_[static_cast<std::size_t>(face_ordinal)];
    if (face.analytic_state.empty()) {
      for_each_cell(region,
                    detail::HyperbolicFaceXKernel<Dim>{values, table, domain.lo[0], domain.hi[0],
                                                       faces_[0].law, faces_[1].law});
      return;
    }
    if (geometry == nullptr)
      throw std::logic_error("analytic x-face execution lost physical Geometry");
    for (int component = 0; component < ncomp(); ++component)
      for_each_cell(region,
                    detail::AnalyticFixedFaceKernel<0>{
                        values, component,
                        analytic_evaluator_<0>(face_ordinal, component, *geometry, physical_time)});
  }

  void fill_y_face_(const Array4& values, const Box2D& region, int face_ordinal,
                    const Box2D& domain, const Geometry* geometry, Real physical_time,
                    const detail::HyperbolicBoundaryTableView<Dim>& table) const {
    const auto& face = faces_[static_cast<std::size_t>(face_ordinal)];
    if (face.analytic_state.empty()) {
      for_each_cell(region,
                    detail::HyperbolicFaceYKernel<Dim>{values, table, domain.lo[1], domain.hi[1],
                                                       faces_[2].law, faces_[3].law});
      return;
    }
    if (geometry == nullptr)
      throw std::logic_error("analytic y-face execution lost physical Geometry");
    for (int component = 0; component < ncomp(); ++component)
      for_each_cell(region,
                    detail::AnalyticFixedFaceKernel<1>{
                        values, component,
                        analytic_evaluator_<1>(face_ordinal, component, *geometry, physical_time)});
  }

  void validate_analytic_values_(const MultiFab& state, const Geometry& geometry,
                                 Real physical_time, CommunicatorView communicator) const {
    const int depth = state.n_grow();
    long invalid_local = 0;
    for (int local = 0; local < state.local_size(); ++local) {
      const Box2D valid = state.fab(local).box();
      int tangential_lo = valid.lo[1] - depth;
      int tangential_hi = valid.hi[1] + depth;
      if (faces_[2].law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, geometry.domain.lo[1]);
      if (faces_[3].law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, geometry.domain.hi[1]);
      for (int face_ordinal = 0; face_ordinal < 2; ++face_ordinal) {
        const auto& face = faces_[static_cast<std::size_t>(face_ordinal)];
        const bool touches = face_ordinal == 0 ? valid.lo[0] == geometry.domain.lo[0]
                                               : valid.hi[0] == geometry.domain.hi[0];
        if (face.analytic_state.empty() || !touches)
          continue;
        const Box2D region = face_ordinal == 0
                                 ? Box2D{{geometry.domain.lo[0] - depth, tangential_lo},
                                         {geometry.domain.lo[0] - 1, tangential_hi}}
                                 : Box2D{{geometry.domain.hi[0] + 1, tangential_lo},
                                         {geometry.domain.hi[0] + depth, tangential_hi}};
        for (int component = 0; component < ncomp(); ++component)
          invalid_local += static_cast<long>(for_each_cell_reduce_sum(
              region, detail::AnalyticFixedFaceFiniteKernel<0>{analytic_evaluator_<0>(
                          face_ordinal, component, geometry, physical_time)}));
      }

      tangential_lo = valid.lo[0] - depth;
      tangential_hi = valid.hi[0] + depth;
      if (faces_[0].law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, geometry.domain.lo[0]);
      if (faces_[1].law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, geometry.domain.hi[0]);
      for (int face_ordinal = 2; face_ordinal < 4; ++face_ordinal) {
        const auto& face = faces_[static_cast<std::size_t>(face_ordinal)];
        const bool touches = face_ordinal == 2 ? valid.lo[1] == geometry.domain.lo[1]
                                               : valid.hi[1] == geometry.domain.hi[1];
        if (face.analytic_state.empty() || !touches)
          continue;
        const Box2D region = face_ordinal == 2
                                 ? Box2D{{tangential_lo, geometry.domain.lo[1] - depth},
                                         {tangential_hi, geometry.domain.lo[1] - 1}}
                                 : Box2D{{tangential_lo, geometry.domain.hi[1] + 1},
                                         {tangential_hi, geometry.domain.hi[1] + depth}};
        for (int component = 0; component < ncomp(); ++component)
          invalid_local += static_cast<long>(for_each_cell_reduce_sum(
              region, detail::AnalyticFixedFaceFiniteKernel<1>{analytic_evaluator_<1>(
                          face_ordinal, component, geometry, physical_time)}));
      }
    }
    const long invalid = all_reduce_sum(invalid_local, communicator);
    if (invalid != 0)
      throw std::runtime_error("analytic hyperbolic boundary produced non-finite values (count=" +
                               std::to_string(invalid) + ")");
  }

  void validate(bool explicit_periodic_identifications) const {
    if (component_transforms_.empty())
      throw std::invalid_argument(
          "prepared hyperbolic boundary requires model-qualified components");
    if (corner_policy_ != HyperbolicCornerPolicy::NotRequired)
      throw std::invalid_argument("unsupported hyperbolic corner policy");
    for (int axis = 0; axis < Dim; ++axis) {
      const auto& low = faces_[static_cast<std::size_t>(2 * axis)];
      const auto& high = faces_[static_cast<std::size_t>(2 * axis + 1)];
      if (!explicit_periodic_identifications && (low.law == HyperbolicBoundaryLaw::Periodic) !=
                                                    (high.law == HyperbolicBoundaryLaw::Periodic))
        throw std::invalid_argument(
            "prepared hyperbolic periodic topology requires complete axis pairs");
    }
    for (int face_ordinal = 0; face_ordinal < 2 * Dim; ++face_ordinal) {
      const auto& prepared_face = faces_[static_cast<std::size_t>(face_ordinal)];
      if (prepared_face.identity.empty() || prepared_face.identity_token == 0)
        throw std::invalid_argument("prepared hyperbolic faces require owner-qualified identities");
      if (!prepared_face.analytic_state.empty()) {
        if (prepared_face.law != HyperbolicBoundaryLaw::FixedState ||
            prepared_face.authored_representation != HyperbolicStateRepresentation::Conservative ||
            !prepared_face.converter_identity.empty() || !prepared_face.fixed_state_converted ||
            prepared_face.analytic_state.size() != component_transforms_.size() ||
            std::any_of(prepared_face.fixed_state.begin(), prepared_face.fixed_state.end(),
                        [](Real value) { return value != Real(0); }) ||
            std::any_of(prepared_face.analytic_state.begin(), prepared_face.analytic_state.end(),
                        [](const analytic::AnalyticProgram& program) {
                          return program.empty() ||
                                 program.result_type() != analytic::AnalyticValueType::Scalar;
                        }))
          throw std::invalid_argument(
              "analytic hyperbolic boundary requires zero fixed-state placeholders and one "
              "conservative scalar program per component");
      } else if (!prepared_face.analytic_clock.empty()) {
        throw std::invalid_argument(
            "only an analytic hyperbolic boundary may carry a logical Clock");
      }
      if (prepared_face.law == HyperbolicBoundaryLaw::FixedState) {
        if (prepared_face.fixed_state.size() != component_transforms_.size() ||
            std::any_of(prepared_face.fixed_state.begin(), prepared_face.fixed_state.end(),
                        [](Real value) { return !std::isfinite(value); }))
          throw std::invalid_argument(
              "fixed-state hyperbolic boundary must provide one finite value per component");
        if (prepared_face.authored_representation == HyperbolicStateRepresentation::Primitive) {
          if (prepared_face.converter_identity.empty())
            throw std::invalid_argument(
                "primitive fixed-state boundary requires one converter identity");
        } else if (!prepared_face.converter_identity.empty() ||
                   !prepared_face.fixed_state_converted) {
          throw std::invalid_argument(
              "conservative fixed-state boundary must not carry conversion metadata");
        }
      } else if (!prepared_face.fixed_state.empty()) {
        throw std::invalid_argument(
            "only a fixed-state hyperbolic boundary may carry component values");
      } else if (prepared_face.authored_representation !=
                     HyperbolicStateRepresentation::Conservative ||
                 !prepared_face.converter_identity.empty() ||
                 !prepared_face.fixed_state_converted) {
        throw std::invalid_argument(
            "only a fixed-state hyperbolic boundary may carry conversion metadata");
      }
      if (prepared_face.law == HyperbolicBoundaryLaw::ReflectiveSlip) {
        const int normal_axis = face_ordinal / 2;
        const bool owns_normal_polar_component =
            std::any_of(component_transforms_.begin(), component_transforms_.end(),
                        [normal_axis](const Transform& transform) {
                          return transform.parity == HyperbolicComponentParity::PolarVector &&
                                 transform.axis == normal_axis;
                        });
        if (!owns_normal_polar_component)
          throw std::invalid_argument(
              "reflective slip wall requires a declared normal polar-vector component");
      }
    }
  }

  void prepare_device_tables() {
    const std::size_t components = component_transforms_.size();
    std::vector<Real> fixed(static_cast<std::size_t>(2 * Dim) * components, Real(0));
    for (int face_ordinal = 0; face_ordinal < 2 * Dim; ++face_ordinal) {
      const auto& source = faces_[static_cast<std::size_t>(face_ordinal)].fixed_state;
      if (source.empty())
        continue;
      std::copy(source.begin(), source.end(),
                fixed.begin() + static_cast<std::ptrdiff_t>(face_ordinal * components));
    }
#if defined(POPS_HAS_KOKKOS)
    detail::ensure_kokkos_initialized();
    device_transforms_ =
        Kokkos::View<Transform*, Kokkos::SharedSpace>("pops_boundary_transforms", components);
    device_fixed_values_ =
        Kokkos::View<Real*, Kokkos::SharedSpace>("pops_boundary_fixed_values", fixed.size());
    auto host_transforms = Kokkos::create_mirror_view(device_transforms_);
    auto host_fixed = Kokkos::create_mirror_view(device_fixed_values_);
    for (std::size_t index = 0; index < components; ++index)
      host_transforms(index) = component_transforms_[index];
    for (std::size_t index = 0; index < fixed.size(); ++index)
      host_fixed(index) = fixed[index];
    Kokkos::deep_copy(device_transforms_, host_transforms);
    Kokkos::deep_copy(device_fixed_values_, host_fixed);
#else
    device_transforms_ = component_transforms_;
    device_fixed_values_ = std::move(fixed);
#endif
  }

  detail::HyperbolicBoundaryTableView<Dim> table_view() const {
    return {
        device_transforms_.data(),
        device_fixed_values_.data(),
        ncomp(),
    };
  }
};

/// Sole built-in parser from the installed Python/native table into the typed hyperbolic plan.
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
  const std::size_t analytic_rows = static_cast<std::size_t>(2 * Dim) * component_roles.size();
  if (face_analytic_opcodes.empty() != face_analytic_literals.empty() ||
      (!face_analytic_opcodes.empty() &&
       (face_analytic_opcodes.size() != analytic_rows ||
        face_analytic_literals.size() != analytic_rows ||
        face_analytic_clocks.size() != static_cast<std::size_t>(2 * Dim))) ||
      (face_analytic_opcodes.empty() && !face_analytic_clocks.empty()))
    throw std::invalid_argument(
        "prepared hyperbolic analytic tables must cover every face/component and Clock");

  std::vector<HyperbolicComponentTransform<Dim>> transforms;
  transforms.reserve(component_roles.size());
  for (const auto& role : component_roles)
    transforms.push_back(detail::transform_from_role<Dim>(role));

  std::string plan_analytic_clock;
  std::array<PreparedHyperbolicFace, 2 * Dim> faces;
  for (int face = 0; face < 2 * Dim; ++face) {
    auto& destination = faces[static_cast<std::size_t>(face)];
    destination.law = detail::hyperbolic_law_from_token(face_types[static_cast<std::size_t>(face)]);
    destination.identity = face_identities[static_cast<std::size_t>(face)];
    destination.identity_token = detail::stable_boundary_identity(destination.identity);
    destination.authored_representation = detail::hyperbolic_representation_from_token(
        face_representations.empty()
            ? std::string_view("conservative")
            : std::string_view(face_representations[static_cast<std::size_t>(face)]));
    destination.converter_identity =
        face_converter_identities.empty()
            ? std::string{}
            : face_converter_identities[static_cast<std::size_t>(face)];
    if (destination.law == HyperbolicBoundaryLaw::FixedState) {
      destination.fixed_state.reserve(component_roles.size());
      for (std::size_t component = 0; component < component_roles.size(); ++component)
        destination.fixed_state.push_back(
            static_cast<Real>(face_values[component * static_cast<std::size_t>(2 * Dim) +
                                          static_cast<std::size_t>(face)]));
      destination.fixed_state_converted =
          destination.authored_representation == HyperbolicStateRepresentation::Conservative;
    }
    if (!face_analytic_opcodes.empty()) {
      bool any_program = false;
      bool every_program = true;
      bool reads_time = false;
      destination.analytic_clock = face_analytic_clocks[static_cast<std::size_t>(face)];
      for (std::size_t component = 0; component < component_roles.size(); ++component) {
        const std::size_t row = static_cast<std::size_t>(face) * component_roles.size() + component;
        const auto& opcodes = face_analytic_opcodes[row];
        const auto& literals = face_analytic_literals[row];
        any_program = any_program || !opcodes.empty();
        every_program = every_program && !opcodes.empty();
        if (opcodes.empty() && literals.empty())
          continue;
        if (opcodes.empty() || opcodes.size() != literals.size())
          throw std::invalid_argument(
              "prepared hyperbolic analytic opcode/literal rows must be non-empty and aligned");
        std::vector<analytic::AnalyticToken> tokens;
        tokens.reserve(opcodes.size());
        for (std::size_t index = 0; index < opcodes.size(); ++index) {
          const analytic::AnalyticOp op = analytic::analytic_op_from_name(opcodes[index]);
          const double raw = literals[index];
          if (!std::isfinite(raw))
            throw std::invalid_argument(
                "prepared hyperbolic analytic token literal must be finite");
          if (op == analytic::AnalyticOp::Input) {
            if (raw != 0.0)
              throw std::invalid_argument(
                  "prepared hyperbolic analytic input is reserved for physical time slot zero");
            reads_time = true;
          }
          tokens.push_back({op, static_cast<Real>(raw)});
        }
        destination.analytic_state.push_back(analytic::compile_analytic_postfix(tokens));
      }
      if (any_program != every_program)
        throw std::invalid_argument(
            "prepared hyperbolic analytic face must cover every state component");
      if (!any_program) {
        if (!destination.analytic_clock.empty())
          throw std::invalid_argument(
              "prepared hyperbolic analytic Clock requires a program on the same face");
        destination.analytic_clock.clear();
      } else if (reads_time != !destination.analytic_clock.empty()) {
        throw std::invalid_argument(
            "prepared hyperbolic analytic physical-time input requires one exact logical Clock");
      } else if (reads_time) {
        if (plan_analytic_clock.empty())
          plan_analytic_clock = destination.analytic_clock;
        else if (plan_analytic_clock != destination.analytic_clock)
          throw std::invalid_argument(
              "prepared hyperbolic analytic plan cannot mix logical Clocks");
      }
    }
  }
  return PreparedHyperbolicBoundary<Dim>(std::move(faces), std::move(transforms),
                                         HyperbolicCornerPolicy::NotRequired,
                                         explicit_periodic_identifications);
}

}  // namespace pops
