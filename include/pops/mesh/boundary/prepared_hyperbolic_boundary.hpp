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
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/multifab.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

enum class HyperbolicBoundaryLaw { Periodic, Extrapolate, FixedState, ReflectiveSlip, External };

enum class HyperbolicComponentParity { Scalar, PolarVector, AxialVector };

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
      HyperbolicCornerPolicy corner_policy = HyperbolicCornerPolicy::NotRequired)
      : faces_(std::move(faces)),
        component_transforms_(std::move(component_transforms)),
        corner_policy_(corner_policy) {
    validate();
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

  Periodicity periodicity() const {
    static_assert(Dim == 2, "the current MultiFab topology is two-dimensional");
    return Periodicity{
        faces_[0].law == HyperbolicBoundaryLaw::Periodic,
        faces_[2].law == HyperbolicBoundaryLaw::Periodic,
    };
  }

  /// Fill only physical faces. Same-level/MPI and periodic topology remain owned by fill_boundary.
  ///
  /// The explicit NotRequired corner policy excludes double-physical corners. Periodic tangential
  /// ghosts are included because they were already produced by fill_boundary and are valid inputs.
  void fill_physical(MultiFab& state, const Box2D& domain) const {
    static_assert(Dim == 2, "the current MultiFab storage is two-dimensional");
    if (state.ncomp() != ncomp())
      throw std::invalid_argument(
          "prepared hyperbolic boundary component count differs from the state");
    const int depth = state.n_grow();
    if (depth == 0)
      return;
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
        for_each_cell(
            Box2D{{domain.lo[0] - depth, tangential_lo}, {domain.lo[0] - 1, tangential_hi}},
            detail::HyperbolicFaceXKernel<Dim>{values, table, domain.lo[0], domain.hi[0],
                                               faces_[0].law, faces_[1].law});
      if (detail::is_physical_hyperbolic_law(faces_[1].law) && valid.hi[0] == domain.hi[0])
        for_each_cell(
            Box2D{{domain.hi[0] + 1, tangential_lo}, {domain.hi[0] + depth, tangential_hi}},
            detail::HyperbolicFaceXKernel<Dim>{values, table, domain.lo[0], domain.hi[0],
                                               faces_[0].law, faces_[1].law});

      tangential_lo = valid.lo[0] - depth;
      tangential_hi = valid.hi[0] + depth;
      if (faces_[0].law != HyperbolicBoundaryLaw::Periodic)
        tangential_lo = std::max(tangential_lo, domain.lo[0]);
      if (faces_[1].law != HyperbolicBoundaryLaw::Periodic)
        tangential_hi = std::min(tangential_hi, domain.hi[0]);
      if (detail::is_physical_hyperbolic_law(faces_[2].law) && valid.lo[1] == domain.lo[1])
        for_each_cell(
            Box2D{{tangential_lo, domain.lo[1] - depth}, {tangential_hi, domain.lo[1] - 1}},
            detail::HyperbolicFaceYKernel<Dim>{values, table, domain.lo[1], domain.hi[1],
                                               faces_[2].law, faces_[3].law});
      if (detail::is_physical_hyperbolic_law(faces_[3].law) && valid.hi[1] == domain.hi[1])
        for_each_cell(
            Box2D{{tangential_lo, domain.hi[1] + 1}, {tangential_hi, domain.hi[1] + depth}},
            detail::HyperbolicFaceYKernel<Dim>{values, table, domain.lo[1], domain.hi[1],
                                               faces_[2].law, faces_[3].law});
    }
  }

 private:
  std::array<PreparedHyperbolicFace, 2 * Dim> faces_{};
  std::vector<Transform> component_transforms_;
  HyperbolicCornerPolicy corner_policy_ = HyperbolicCornerPolicy::NotRequired;
#if defined(POPS_HAS_KOKKOS)
  Kokkos::View<Transform*, Kokkos::SharedSpace> device_transforms_;
  Kokkos::View<Real*, Kokkos::SharedSpace> device_fixed_values_;
#else
  std::vector<Transform> device_transforms_;
  std::vector<Real> device_fixed_values_;
#endif

  void validate() const {
    if (component_transforms_.empty())
      throw std::invalid_argument(
          "prepared hyperbolic boundary requires model-qualified components");
    if (corner_policy_ != HyperbolicCornerPolicy::NotRequired)
      throw std::invalid_argument("unsupported hyperbolic corner policy");
    for (int axis = 0; axis < Dim; ++axis) {
      const auto& low = faces_[static_cast<std::size_t>(2 * axis)];
      const auto& high = faces_[static_cast<std::size_t>(2 * axis + 1)];
      if ((low.law == HyperbolicBoundaryLaw::Periodic) !=
          (high.law == HyperbolicBoundaryLaw::Periodic))
        throw std::invalid_argument(
            "prepared hyperbolic periodic topology requires complete axis pairs");
    }
    for (int face_ordinal = 0; face_ordinal < 2 * Dim; ++face_ordinal) {
      const auto& prepared_face = faces_[static_cast<std::size_t>(face_ordinal)];
      if (prepared_face.identity.empty() || prepared_face.identity_token == 0)
        throw std::invalid_argument("prepared hyperbolic faces require owner-qualified identities");
      if (prepared_face.law == HyperbolicBoundaryLaw::FixedState) {
        if (prepared_face.fixed_state.size() != component_transforms_.size() ||
            std::any_of(prepared_face.fixed_state.begin(), prepared_face.fixed_state.end(),
                        [](Real value) { return !std::isfinite(value); }))
          throw std::invalid_argument(
              "fixed-state hyperbolic boundary must provide one finite value per component");
      } else if (!prepared_face.fixed_state.empty()) {
        throw std::invalid_argument(
            "only a fixed-state hyperbolic boundary may carry component values");
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
    const std::vector<std::string>& component_roles) {
  if (face_types.size() != static_cast<std::size_t>(2 * Dim) ||
      face_identities.size() != static_cast<std::size_t>(2 * Dim))
    throw std::invalid_argument(
        "prepared hyperbolic boundary requires one type and identity per oriented face");
  if (component_roles.empty() ||
      face_values.size() != component_roles.size() * static_cast<std::size_t>(2 * Dim))
    throw std::invalid_argument(
        "prepared hyperbolic boundary values must be component-major and total");

  std::vector<HyperbolicComponentTransform<Dim>> transforms;
  transforms.reserve(component_roles.size());
  for (const auto& role : component_roles)
    transforms.push_back(detail::transform_from_role<Dim>(role));

  std::array<PreparedHyperbolicFace, 2 * Dim> faces;
  for (int face = 0; face < 2 * Dim; ++face) {
    auto& destination = faces[static_cast<std::size_t>(face)];
    destination.law = detail::hyperbolic_law_from_token(face_types[static_cast<std::size_t>(face)]);
    destination.identity = face_identities[static_cast<std::size_t>(face)];
    destination.identity_token = detail::stable_boundary_identity(destination.identity);
    if (destination.law == HyperbolicBoundaryLaw::FixedState) {
      destination.fixed_state.reserve(component_roles.size());
      for (std::size_t component = 0; component < component_roles.size(); ++component)
        destination.fixed_state.push_back(
            static_cast<Real>(face_values[component * static_cast<std::size_t>(2 * Dim) +
                                          static_cast<std::size_t>(face)]));
    }
  }
  return PreparedHyperbolicBoundary<Dim>(std::move(faces), std::move(transforms),
                                         HyperbolicCornerPolicy::NotRequired);
}

}  // namespace pops
