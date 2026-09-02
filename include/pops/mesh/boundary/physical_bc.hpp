/// @file
/// @brief Prepared compile-time-ranked physical boundary fills for scalar and vector fields.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/nd_boundary_schedule.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops {

/// Native affine laws applied at one physical face.  Periodicity belongs exclusively to
/// BoundaryTopology<Dim>; it is deliberately not duplicated here.
enum class PhysicalBoundaryKind : unsigned char {
  external,
  constant_extrapolation,
  dirichlet,
  neumann,
  robin,
};

/// alpha*u + beta*du/dn = value at a physical face.  Dirichlet consumes value, Neumann consumes
/// the outward derivative in value, and constant extrapolation consumes no coefficients.
struct PhysicalBoundaryFace {
  PhysicalBoundaryKind kind = PhysicalBoundaryKind::external;
  Real value = Real(0);
  Real alpha = Real(0);
  Real beta = Real(1);

  bool operator==(const PhysicalBoundaryFace&) const = default;
};

/// Exact-ranked physical laws plus the topology which owns periodic faces.
template <int Dim>
class PhysicalBoundaryConditions {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PhysicalBoundaryConditions only supports dimensions 1, 2, and 3");
  static constexpr std::size_t face_count = static_cast<std::size_t>(2 * Dim);

  PhysicalBoundaryConditions(BoundaryTopology<Dim> topology,
                             std::array<PhysicalBoundaryFace, face_count> faces,
                             RealVector<Dim> spacing)
      : topology_(std::move(topology)), faces_(std::move(faces)), spacing_(spacing) {
    validate_();
  }

  const BoundaryTopology<Dim>& topology() const noexcept { return topology_; }
  const RealVector<Dim>& spacing() const noexcept { return spacing_; }
  const PhysicalBoundaryFace& at(Face<Dim> face) const noexcept {
    return faces_[static_cast<std::size_t>(face.ordinal())];
  }

  bool operator==(const PhysicalBoundaryConditions&) const = default;

 private:
  static bool finite_(Real value) noexcept {
    return value == value && value != std::numeric_limits<Real>::infinity() &&
           value != -std::numeric_limits<Real>::infinity();
  }

  void validate_() const {
    for (int axis = 0; axis < Dim; ++axis) {
      if (!finite_(spacing_[axis]) || !(spacing_[axis] > Real(0)))
        throw std::invalid_argument(
            "physical boundary conditions require finite positive spacing on every axis");
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        const PhysicalBoundaryFace& law = at(face);
        if (topology_.is_periodic(face)) {
          if (law.kind != PhysicalBoundaryKind::external)
            throw std::invalid_argument(
                "periodic topology faces cannot also carry a physical boundary law");
          continue;
        }
        if (law.kind == PhysicalBoundaryKind::external)
          continue;
        if (!finite_(law.value))
          throw std::invalid_argument("physical boundary values must be finite");
        if (law.kind == PhysicalBoundaryKind::robin && (!finite_(law.alpha) || !finite_(law.beta)))
          throw std::invalid_argument("Robin boundary coefficients must be finite");
      }
    }
  }

  BoundaryTopology<Dim> topology_;
  std::array<PhysicalBoundaryFace, face_count> faces_{};
  RealVector<Dim> spacing_{};
};

namespace physical_boundary_detail {

struct AffineSample {
  int source = 0;
  Real scale = Real(1);
  Real offset = Real(0);
};

POPS_HD inline bool is_native(PhysicalBoundaryKind kind) {
  return kind != PhysicalBoundaryKind::external;
}

/// Map an arbitrary-depth ghost coordinate to a valid coordinate and compose every crossed face
/// as one affine transform.  The loop is independent of spatial rank and handles domains shorter
/// than their halo depth without reading a previously written ghost cell.
POPS_HD inline AffineSample sample_axis(int index, int lo, int hi, PhysicalBoundaryFace low,
                                        PhysicalBoundaryFace high, Real spacing) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const PhysicalBoundaryFace law = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (law.kind == PhysicalBoundaryKind::constant_extrapolation) {
      current = boundary;
      break;
    }

    const std::int64_t layer = below ? boundary - current : current - boundary;
    const Real distance = (Real(2) * static_cast<Real>(layer) - Real(1)) * spacing;
    Real face_scale = Real(1);
    Real face_offset = Real(0);
    if (law.kind == PhysicalBoundaryKind::dirichlet) {
      face_scale = Real(-1);
      face_offset = Real(2) * law.value;
    } else if (law.kind == PhysicalBoundaryKind::neumann) {
      face_offset = law.value * distance;
    } else if (law.kind == PhysicalBoundaryKind::robin) {
      const Real denominator = law.alpha / Real(2) + law.beta / distance;
      face_scale = -(law.alpha / Real(2) - law.beta / distance) / denominator;
      face_offset = law.value / denominator;
    }

    offset += scale * face_offset;
    scale *= face_scale;
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
  return {static_cast<int>(current), scale, offset};
}

inline void validate_axis_sample(std::int64_t index, int lo, int hi, PhysicalBoundaryFace low,
                                 PhysicalBoundaryFace high, Real spacing) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const PhysicalBoundaryFace law = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (!is_native(law.kind))
      throw std::invalid_argument(
          "deep physical boundary extension reaches an externally owned opposite face");
    if (law.kind == PhysicalBoundaryKind::constant_extrapolation)
      return;

    const std::int64_t layer = below ? boundary - current : current - boundary;
    const Real distance = (Real(2) * static_cast<Real>(layer) - Real(1)) * spacing;
    Real face_scale = Real(1);
    Real face_offset = Real(0);
    if (law.kind == PhysicalBoundaryKind::dirichlet) {
      face_scale = Real(-1);
      face_offset = Real(2) * law.value;
    } else if (law.kind == PhysicalBoundaryKind::neumann) {
      face_offset = law.value * distance;
    } else if (law.kind == PhysicalBoundaryKind::robin) {
      const Real denominator = law.alpha / Real(2) + law.beta / distance;
      if (!std::isfinite(static_cast<double>(distance)) || !(distance > Real(0)) ||
          !std::isfinite(static_cast<double>(denominator)) || denominator == Real(0))
        throw std::invalid_argument("physical boundary preparation found a singular Robin law");
      face_scale = -(law.alpha / Real(2) - law.beta / distance) / denominator;
      face_offset = law.value / denominator;
    }
    offset += scale * face_offset;
    scale *= face_scale;
    if (!std::isfinite(static_cast<double>(scale)) || !std::isfinite(static_cast<double>(offset)))
      throw std::overflow_error("physical boundary preparation produced a non-finite transform");
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
}

template <int Dim>
struct ApplyPhysicalRegion {
  FieldView<Real, Dim> values{};
  Box<Dim> domain{};
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  BoundaryRegionPlan<Dim> plan{};
  int first_component = 0;
  int component_count = 0;

  POPS_HD void operator()(const Index<Dim>& destination) const {
    Index<Dim> source = destination;

    Real scale = Real(1);
    Real offset = Real(0);
    for (std::size_t operation = 0; operation < plan.operation_count; ++operation) {
      const BoundaryOperation<Dim>& op = plan.operations[operation];
      if (op.kind != BoundaryFaceKind::physical)
        continue;
      const int axis = op.face.axis;
      const PhysicalBoundaryFace low =
          faces[static_cast<std::size_t>(Face<Dim>{axis, BoundarySide::lower}.ordinal())];
      const PhysicalBoundaryFace high =
          faces[static_cast<std::size_t>(Face<Dim>{axis, BoundarySide::upper}.ordinal())];
      const AffineSample sample = sample_axis(destination[axis], domain.lo[axis], domain.hi[axis],
                                              low, high, spacing[axis]);
      source[axis] = sample.source;
      offset = sample.scale * offset + sample.offset;
      scale *= sample.scale;
    }
    for (int component = first_component; component < first_component + component_count;
         ++component)
      values(destination, component) = scale * values(source, component) + offset;
  }
};

}  // namespace physical_boundary_detail

/// Prepared disjoint face/edge/corner regions and authenticated affine laws.
template <int Dim>
class PreparedPhysicalBoundary {
 public:
  PreparedPhysicalBoundary(PhysicalBoundaryConditions<Dim> conditions,
                           BoundarySchedule<Dim> schedule)
      : conditions_(std::move(conditions)), schedule_(std::move(schedule)) {
    if (conditions_.topology() != schedule_.topology())
      throw std::invalid_argument(
          "prepared physical boundary topology disagrees with its region schedule");
    validate_transforms_();
  }

  const PhysicalBoundaryConditions<Dim>& conditions() const noexcept { return conditions_; }
  const BoundarySchedule<Dim>& schedule() const noexcept { return schedule_; }
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    return schedule_.resident_storage_bytes();
  }

  template <class MemorySpace>
  void authenticate(const MultiFab<Dim, MemorySpace>& fields) const {
    if (fields.ghosts() != schedule_.ghosts())
      throw std::invalid_argument(
          "prepared physical boundary ghost extents disagree with the target field");
    for (const Box<Dim>& patch : fields.layout().boxes())
      if (!schedule_.domain().contains(patch))
        throw std::invalid_argument(
            "prepared physical boundary domain does not contain the target field layout");
  }

 private:
  void validate_transforms_() const {
    const Box<Dim>& domain = schedule_.domain();
    const Extent<Dim>& ghosts = schedule_.ghosts();
    for (int axis = 0; axis < Dim; ++axis) {
      const PhysicalBoundaryFace& low = conditions_.at(Face<Dim>{axis, BoundarySide::lower});
      const PhysicalBoundaryFace& high = conditions_.at(Face<Dim>{axis, BoundarySide::upper});
      for (std::int64_t layer = 1; layer <= ghosts[axis]; ++layer) {
        if (physical_boundary_detail::is_native(low.kind))
          physical_boundary_detail::validate_axis_sample(
              static_cast<std::int64_t>(domain.lo[axis]) - layer, domain.lo[axis], domain.hi[axis],
              low, high, conditions_.spacing()[axis]);
        if (physical_boundary_detail::is_native(high.kind))
          physical_boundary_detail::validate_axis_sample(
              static_cast<std::int64_t>(domain.hi[axis]) + layer, domain.lo[axis], domain.hi[axis],
              low, high, conditions_.spacing()[axis]);
      }
    }
  }

  PhysicalBoundaryConditions<Dim> conditions_;
  BoundarySchedule<Dim> schedule_;
};

template <int Dim>
PreparedPhysicalBoundary<Dim> prepare_physical_boundary(const Box<Dim>& domain,
                                                        const Extent<Dim>& ghosts,
                                                        PhysicalBoundaryConditions<Dim> conditions,
                                                        BoundaryScheduleBudget budget) {
  BoundarySchedule<Dim> schedule =
      prepare_boundary_schedule(domain, ghosts, conditions.topology(), budget);
  return PreparedPhysicalBoundary<Dim>{std::move(conditions), std::move(schedule)};
}

/// Fill physical destination regions after the authenticated same-level/periodic halo exchange.
/// External regions remain untouched.  Component selection is exact and validated before launch.
template <int Dim, class MemorySpace>
void fill_physical_boundary(MultiFab<Dim, MemorySpace>& fields,
                            const PreparedPhysicalBoundary<Dim>& prepared, int first_component,
                            int component_count) {
  prepared.authenticate(fields);
  if (first_component < 0 || component_count < 1 || first_component > fields.ncomp() ||
      component_count > fields.ncomp() - first_component)
    throw std::out_of_range(
        "physical boundary component range must be non-empty and inside the target field");

  const auto& conditions = prepared.conditions();
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  for (int axis = 0; axis < Dim; ++axis) {
    faces[static_cast<std::size_t>(Face<Dim>{axis, BoundarySide::lower}.ordinal())] =
        conditions.at(Face<Dim>{axis, BoundarySide::lower});
    faces[static_cast<std::size_t>(Face<Dim>{axis, BoundarySide::upper}.ordinal())] =
        conditions.at(Face<Dim>{axis, BoundarySide::upper});
  }

  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    auto& fab = fields.fab(local);
    for (const BoundaryRegionPlan<Dim>& plan : prepared.schedule().entries()) {
      if (!plan.has_physical())
        continue;
      bool externally_owned = false;
      for (std::size_t operation = 0; operation < plan.operation_count; ++operation) {
        const BoundaryOperation<Dim>& op = plan.operations[operation];
        if (op.kind == BoundaryFaceKind::physical &&
            conditions.at(op.face).kind == PhysicalBoundaryKind::external) {
          externally_owned = true;
          break;
        }
      }
      if (externally_owned)
        continue;
      const Box<Dim> destination = fab.grown_box().intersect(plan.destination);
      for_each_cell(destination, physical_boundary_detail::ApplyPhysicalRegion<Dim>{
                                     fab.view(), prepared.schedule().domain(), faces,
                                     conditions.spacing(), plan, first_component, component_count});
    }
  }
  ::pops::device_fence();
}

template <int Dim, class MemorySpace>
void fill_physical_boundary(MultiFab<Dim, MemorySpace>& fields,
                            const PreparedPhysicalBoundary<Dim>& prepared) {
  fill_physical_boundary(fields, prepared, 0, fields.ncomp());
}

}  // namespace pops
