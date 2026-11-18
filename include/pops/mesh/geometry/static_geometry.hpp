/// @file
/// @brief Static geometry arrays of a mapped structured configuration (N17, C1).
///
/// A coordinate map X that reads neither the clock nor a discrete input sends the logical box of a
/// layout onto its physical image, fixed in time. The arrays below are generated once, at bind,
/// from the analytic programs of X (data, not callbacks) and from a stated quadrature rule on
/// [0, 1] (D0, D0b):
///
/// * node positions X at the logical nodes, the logical node k of an axis of M cells being
///   lower + (upper - lower) (k / M) with the quotient k / M correctly rounded, so that node k of
///   M cells and node 2k of 2M cells are one binary64 point and graded nodes nest bitwise;
/// * cell measure |K| = int_K |det DX|, by the tensor rule (a chart), or the product of the node
///   differences (node positions: component a reads the logical coordinate a alone);
/// * cell centroid int_K X |det DX| / |K| (chart), or the midpoint of the node intervals;
/// * face area vector S_f oriented toward increasing index, so that the faces of every cell close
///   exactly in exact arithmetic: for node positions of any rank, the logical axis times the
///   product of the node differences of the other axes (the node itself in rank one); for a
///   chart, of rank two, the rotated chord; its measure |S_f| and unit normal S_f / |S_f|;
/// * face centroid (the centre of a box face, or the chord midpoint), face distance |c_R - c_L|,
///   and at a physical end of node positions the mirrored distance 2 |m_f - c_K| (a chart states
///   none there: its chord and its curved cell need not keep the centroid inside, so the end
///   distance of a chart is NaN, and a consumer that would read it fails closed); cell width
///   |K| / ((|S_lower| + |S_upper|) / 2), which for node positions is the node difference itself,
///   taken exactly.
///
/// The generation certifies what the consumers rely on, and refuses the configuration otherwise:
/// finite values; positive measures; positive orientation (det DX > 0 at every quadrature point of
/// a chart, S_f . (c_R - c_L) > 0 across every face, node positions increasing along their axis);
/// node positions that keep the bounds of their box (the end nodes of every axis within 4 ulp of
/// its bounds, then set to them: the nodes move inside the box, the box stays the domain); the
/// closure of every cell to its rounding floor; and, on a periodic axis of a chart, one
/// translation T with X(upper) = X(lower) + T at every seam node, to the rounding floor of A3
/// (|X| ulp and ||DX|| ulp of the logical endpoint). The seam nodes are then identified, so the
/// seam is one face, bitwise, on both sides.

#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/runtime/analytic/expression.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

enum class StaticGeometryKind : std::uint8_t { nodes = 0, chart = 1 };

/// A tensor quadrature rule on [0, 1], applied on every logical axis of a chart.
struct StaticGeometryRule {
  std::vector<Real> nodes;
  std::vector<Real> weights;
};

/// Non-owning, trivially copyable reading of the arrays, captured by value in kernels. Cell arrays
/// cover the logical domain; the face arrays of an axis cover its faces lo..hi+1. A reading
/// outside them is NaN, so a consumer that reaches past the arrays fails closed.
template <int Dim>
struct StaticGeometryView {
  static_assert(Dim >= 1 && Dim <= 3);
  Box<Dim> domain{};
  const Real* cell_measure = nullptr;
  std::array<const Real*, Dim> centroid{};
  std::array<const Real*, Dim> width{};
  std::array<std::array<const Real*, Dim>, Dim> face_area{};
  std::array<const Real*, Dim> face_measure{};
  std::array<std::array<const Real*, Dim>, Dim> face_normal{};
  std::array<std::array<const Real*, Dim>, Dim> face_centroid{};
  std::array<const Real*, Dim> face_distance{};

  POPS_HD bool contains_cell(const Index<Dim>& cell) const {
    for (int axis = 0; axis < Dim; ++axis)
      if (cell[axis] < domain.lo[axis] || cell[axis] > domain.hi[axis])
        return false;
    return cell_measure != nullptr;
  }
  POPS_HD std::int64_t cell_offset(const Index<Dim>& cell) const {
    std::int64_t offset = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      offset += static_cast<std::int64_t>(cell[axis] - domain.lo[axis]) * stride;
      stride *= static_cast<std::int64_t>(domain.hi[axis] - domain.lo[axis] + 1);
    }
    return offset;
  }
  POPS_HD bool contains_face(int axis, const Index<Dim>& face) const {
    for (int other = 0; other < Dim; ++other) {
      const auto upper = domain.hi[other] + (other == axis ? 1 : 0);
      if (face[other] < domain.lo[other] || face[other] > upper)
        return false;
    }
    return face_measure[axis] != nullptr;
  }
  POPS_HD std::int64_t face_offset(int axis, const Index<Dim>& face) const {
    std::int64_t offset = 0, stride = 1;
    for (int other = 0; other < Dim; ++other) {
      offset += static_cast<std::int64_t>(face[other] - domain.lo[other]) * stride;
      stride *= static_cast<std::int64_t>(domain.hi[other] - domain.lo[other] + 1 +
                                          (other == axis ? 1 : 0));
    }
    return offset;
  }
  POPS_HD static Real missing() { return std::numeric_limits<Real>::quiet_NaN(); }

  POPS_HD Real measure(const Index<Dim>& cell) const {
    return contains_cell(cell) ? cell_measure[cell_offset(cell)] : missing();
  }
  POPS_HD Real centroid_at(int component, const Index<Dim>& cell) const {
    return contains_cell(cell) ? centroid[component][cell_offset(cell)] : missing();
  }
  POPS_HD Real width_at(int axis, const Index<Dim>& cell) const {
    return contains_cell(cell) ? width[axis][cell_offset(cell)] : missing();
  }
  POPS_HD Real area_at(int axis, int component, const Index<Dim>& face) const {
    return contains_face(axis, face) ? face_area[axis][component][face_offset(axis, face)]
                                     : missing();
  }
  POPS_HD Real face_measure_at(int axis, const Index<Dim>& face) const {
    return contains_face(axis, face) ? face_measure[axis][face_offset(axis, face)] : missing();
  }
  POPS_HD Real normal_at(int axis, int component, const Index<Dim>& face) const {
    return contains_face(axis, face) ? face_normal[axis][component][face_offset(axis, face)]
                                     : missing();
  }
  POPS_HD Real face_centroid_at(int axis, int component, const Index<Dim>& face) const {
    return contains_face(axis, face) ? face_centroid[axis][component][face_offset(axis, face)]
                                     : missing();
  }
  POPS_HD Real distance_at(int axis, const Index<Dim>& face) const {
    return contains_face(axis, face) ? face_distance[axis][face_offset(axis, face)] : missing();
  }
};

static_assert(std::is_trivially_copyable_v<StaticGeometryView<1>>);
static_assert(std::is_trivially_copyable_v<StaticGeometryView<2>>);
static_assert(std::is_trivially_copyable_v<StaticGeometryView<3>>);

namespace static_geometry_detail {

[[noreturn]] inline void refuse(const std::string& message) {
  throw std::invalid_argument("static geometry arrays: " + message);
}

inline Real ulp(Real value) {
  const Real magnitude = std::fabs(value);
  return std::nextafter(magnitude, std::numeric_limits<Real>::infinity()) - magnitude;
}

/// det DX, rows the components of X, columns the logical axes.
template <int Dim>
Real determinant(const std::array<RealVector<Dim>, Dim>& jacobian) {
  if constexpr (Dim == 1) {
    return jacobian[0][0];
  } else if constexpr (Dim == 2) {
    return jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
  } else {
    return jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1]) -
           jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0]) +
           jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);
  }
}

inline std::string number(Real value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6e", static_cast<double>(value));
  return buffer;
}

template <int Dim>
std::string cell_name(const Index<Dim>& cell) {
  std::string result = "(";
  for (int axis = 0; axis < Dim; ++axis)
    result += (axis ? ", " : "") + std::to_string(cell[axis]);
  return result + ")";
}

}  // namespace static_geometry_detail

/// The owning arrays of one layout, immutable once generated.
template <int Dim>
class StaticGeometry {
 public:
  static_assert(Dim >= 1 && Dim <= 3);
  using Storage = std::vector<Real, fab_allocator<Real>>;

  /// Generate and certify the arrays of a map over the logical domain of @p geometry.
  static std::shared_ptr<const StaticGeometry> generate(
      std::string identity, StaticGeometryKind kind,
      const std::vector<analytic::AnalyticProgram>& components, const StaticGeometryRule& rule,
      const Geometry<Dim>& geometry, const std::array<bool, Dim>& periodicity) {
    using static_geometry_detail::refuse;
    if (identity.empty())
      refuse("the layout identity is empty");
    if (static_cast<int>(components.size()) != Dim)
      refuse("the coordinate map needs one program per logical axis");
    for (const auto& program : components)
      if (program.empty() || program.result_type() != analytic::AnalyticValueType::Scalar ||
          program.required_dimension() > Dim)
        refuse("every component of the coordinate map is one scalar program of the layout rank");
    if (kind == StaticGeometryKind::chart) {
      if (rule.nodes.empty() || rule.nodes.size() != rule.weights.size())
        refuse("a chart needs a stated quadrature rule with as many weights as nodes");
      for (std::size_t point = 0; point < rule.nodes.size(); ++point)
        if (!std::isfinite(rule.nodes[point]) || !std::isfinite(rule.weights[point]) ||
            rule.nodes[point] < Real(0) || rule.nodes[point] > Real(1))
          refuse("the quadrature rule must be finite, with nodes on [0, 1]");
      if (Dim != 2)
        refuse("a chart is served on layouts of rank two");
    } else if (!rule.nodes.empty() || !rule.weights.empty()) {
      refuse("node positions have exact node differences; a rule would state nothing");
    }
    std::shared_ptr<StaticGeometry> result(new StaticGeometry());
    result->identity_ = std::move(identity);
    result->kind_ = kind;
    result->domain_ = geometry.domain();
    result->build(components, rule, geometry, periodicity);
    return result;
  }

  [[nodiscard]] const std::string& identity() const noexcept { return identity_; }
  [[nodiscard]] StaticGeometryKind kind() const noexcept { return kind_; }
  [[nodiscard]] const Box<Dim>& domain() const noexcept { return domain_; }
  /// Largest closure residual of a cell relative to its rounding floor (certified <= 1).
  [[nodiscard]] Real closure_ratio() const noexcept { return closure_ratio_; }

  [[nodiscard]] StaticGeometryView<Dim> view() const noexcept {
    StaticGeometryView<Dim> view{};
    view.domain = domain_;
    view.cell_measure = cell_measure_.data();
    for (int axis = 0; axis < Dim; ++axis) {
      view.centroid[axis] = centroid_[axis].data();
      view.width[axis] = width_[axis].data();
      view.face_measure[axis] = face_measure_[axis].data();
      view.face_distance[axis] = face_distance_[axis].data();
      for (int component = 0; component < Dim; ++component) {
        view.face_area[axis][component] = face_area_[axis][component].data();
        view.face_normal[axis][component] = face_normal_[axis][component].data();
        view.face_centroid[axis][component] = face_centroid_[axis][component].data();
      }
    }
    return view;
  }

  /// Logical node count per axis and node positions, for snapshots and tests.
  [[nodiscard]] const std::array<Storage, Dim>& nodes() const noexcept { return nodes_; }

 private:
  StaticGeometry() = default;

  std::array<std::int64_t, Dim> cells_{};
  std::array<std::int64_t, Dim> node_extent_{};

  std::int64_t node_offset(const std::array<std::int64_t, Dim>& node) const {
    std::int64_t offset = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      offset += node[axis] * stride;
      stride *= node_extent_[axis];
    }
    return offset;
  }
  RealVector<Dim> node_position(const std::array<std::int64_t, Dim>& node) const {
    RealVector<Dim> result{};
    const auto offset = node_offset(node);
    for (int component = 0; component < Dim; ++component)
      result[component] = nodes_[component][static_cast<std::size_t>(offset)];
    return result;
  }

  template <class Visit>
  static void for_each_index(const std::array<std::int64_t, Dim>& extent, Visit&& visit) {
    std::array<std::int64_t, Dim> index{};
    std::int64_t total = 1;
    for (int axis = 0; axis < Dim; ++axis)
      total *= extent[axis];
    for (std::int64_t linear = 0; linear < total; ++linear) {
      std::int64_t rest = linear;
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] = rest % extent[axis];
        rest /= extent[axis];
      }
      visit(index, linear);
    }
  }

  void build(const std::vector<analytic::AnalyticProgram>& components,
             const StaticGeometryRule& rule, const Geometry<Dim>& geometry,
             const std::array<bool, Dim>& periodicity) {
    using static_geometry_detail::cell_name;
    using static_geometry_detail::refuse;
    using static_geometry_detail::ulp;
    std::array<analytic::AnalyticProgramView, Dim> programs{};
    for (int axis = 0; axis < Dim; ++axis)
      programs[axis] = components[static_cast<std::size_t>(axis)].view();
    std::array<std::vector<Real>, Dim> logical{};
    std::int64_t cell_count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      cells_[axis] = domain_.length(axis);
      if (cells_[axis] < 1)
        refuse("the logical domain is empty");
      node_extent_[axis] = cells_[axis] + 1;
      cell_count *= cells_[axis];
      // The logical node k sits at lower + (upper - lower) (k / M), the quotient of two integers
      // correctly rounded: node k of M cells and node 2k of 2M cells read one binary64 point, so
      // the nodes of a graded axis nest bitwise across resolutions (design note N17, 3.8). The
      // last node is the upper bound itself, which the sum may miss by its rounding.
      const Real lower = geometry.lower()[axis], upper = geometry.upper()[axis];
      const Real cells = static_cast<Real>(cells_[axis]);
      logical[axis].resize(static_cast<std::size_t>(node_extent_[axis]));
      for (std::int64_t node = 0; node + 1 < node_extent_[axis]; ++node)
        logical[axis][static_cast<std::size_t>(node)] =
            lower + (upper - lower) * (static_cast<Real>(node) / cells);
      logical[axis].back() = upper;
    }
    std::int64_t node_count = 1;
    for (int axis = 0; axis < Dim; ++axis)
      node_count *= node_extent_[axis];
    for (int component = 0; component < Dim; ++component)
      nodes_[component].assign(static_cast<std::size_t>(node_count), Real(0));

    // Node positions. Node positions of a separable map are evaluated once per axis, so that
    // every node difference along an axis is the same binary64 difference on every line.
    if (kind_ == StaticGeometryKind::nodes) {
      std::array<std::vector<Real>, Dim> positions{};
      for (int axis = 0; axis < Dim; ++axis) {
        positions[axis].resize(static_cast<std::size_t>(node_extent_[axis]));
        for (std::int64_t node = 0; node < node_extent_[axis]; ++node) {
          RealVector<Dim> point{};
          for (int other = 0; other < Dim; ++other)
            point[other] = logical[other][0];
          point[axis] = logical[axis][static_cast<std::size_t>(node)];
          const auto value = programs[axis].eval_checked(point);
          if (!value.valid || !std::isfinite(value.value))
            refuse("the coordinate map is not finite at a node of axis " + std::to_string(axis));
          positions[axis][static_cast<std::size_t>(node)] = value.value;
        }
        // Node positions keep the bounds of their box: the end nodes lie within 4 ulp of the
        // bounds of the axis and are set to them (DN3). The nodes move inside the box, and the box
        // stays the domain; a separable map that moves the box would state the domain in the
        // layout (SU6).
        const Real lower = geometry.lower()[axis], upper = geometry.upper()[axis];
        const Real tolerance = Real(4) * ulp(std::max(std::fabs(lower), std::fabs(upper)));
        Real& first = positions[axis].front();
        Real& last = positions[axis].back();
        if (!(std::fabs(first - lower) <= tolerance) || !(std::fabs(last - upper) <= tolerance))
          refuse(
              "node positions of axis " + std::to_string(axis) + " must keep the bounds [" +
              static_geometry_detail::number(lower) + ", " + static_geometry_detail::number(upper) +
              "] of the box within 4 ulp; the map sends them to [" +
              static_geometry_detail::number(first) + ", " + static_geometry_detail::number(last) +
              "]: graded nodes move inside the box, they do not move the box");
        first = lower;
        last = upper;
        for (std::int64_t node = 0; node + 1 < node_extent_[axis]; ++node)
          if (!(positions[axis][static_cast<std::size_t>(node + 1)] >
                positions[axis][static_cast<std::size_t>(node)]))
            refuse("node positions must increase along axis " + std::to_string(axis) + " (node " +
                   std::to_string(node) + "); the map folds or reverses it");
      }
      for_each_index(node_extent_, [&](const auto& node, std::int64_t linear) {
        for (int component = 0; component < Dim; ++component)
          nodes_[component][static_cast<std::size_t>(linear)] =
              positions[component][static_cast<std::size_t>(node[component])];
      });
    } else {
      for_each_index(node_extent_, [&](const auto& node, std::int64_t linear) {
        RealVector<Dim> point{};
        for (int axis = 0; axis < Dim; ++axis)
          point[axis] = logical[axis][static_cast<std::size_t>(node[axis])];
        for (int component = 0; component < Dim; ++component) {
          const auto value = programs[component].eval_checked(point);
          if (!value.valid || !std::isfinite(value.value))
            refuse("the coordinate map is not finite at a logical node");
          nodes_[component][static_cast<std::size_t>(linear)] = value.value;
        }
      });
    }

    // The seam of a periodic axis of a chart: one translation T, then identified nodes.
    std::array<RealVector<Dim>, Dim> seam_translation{};
    for (int axis = 0; axis < Dim; ++axis) {
      if (!periodicity[axis])
        continue;
      if (kind_ == StaticGeometryKind::nodes) {
        seam_translation[axis][axis] = nodes_[axis][static_cast<std::size_t>(node_offset([&] {
                                         std::array<std::int64_t, Dim> upper{};
                                         upper[axis] = cells_[axis];
                                         return upper;
                                       }()))] -
                                       nodes_[axis][0];
        continue;
      }
      bool first = true;
      RealVector<Dim> reference{};
      std::array<std::int64_t, Dim> extent = node_extent_;
      extent[axis] = 1;
      for_each_index(extent, [&](auto node, std::int64_t) {
        auto upper = node;
        upper[axis] = cells_[axis];
        const RealVector<Dim> low = node_position(node), high = node_position(upper);
        RealVector<Dim> point{};
        for (int other = 0; other < Dim; ++other)
          point[other] = logical[other][static_cast<std::size_t>(upper[other])];
        Real jacobian = Real(0), magnitude = Real(0);
        for (int component = 0; component < Dim; ++component) {
          const auto jet = programs[component].eval_jet_checked(point);
          if (!jet.valid)
            refuse("the derivative of the coordinate map is not finite on a periodic seam");
          for (int other = 0; other < Dim; ++other)
            jacobian = std::max(jacobian, std::fabs(jet.partials[other]));
          magnitude = std::max({magnitude, std::fabs(low[component]), std::fabs(high[component])});
        }
        RealVector<Dim> translation{};
        for (int component = 0; component < Dim; ++component)
          translation[component] = high[component] - low[component];
        if (first) {
          reference = translation;
          first = false;
        }
        const Real floor =
            Real(4) * (Real(Dim) * jacobian * ulp(logical[axis].back()) + ulp(magnitude));
        for (int component = 0; component < Dim; ++component)
          if (!(std::fabs(translation[component] - reference[component]) <= floor))
            refuse("periodic axis " + std::to_string(axis) +
                   ": the map does not send the seam onto itself by one translation (" +
                   static_geometry_detail::number(translation[component] - reference[component]) +
                   " above the rounding floor " + static_geometry_detail::number(floor) + ")");
      });
      // A closed seam (T at its rounding floor) is identified exactly.
      Real magnitude = Real(0), scale = Real(0);
      for (int component = 0; component < Dim; ++component) {
        magnitude = std::max(magnitude, std::fabs(reference[component]));
        scale = std::max(scale, std::fabs(nodes_[component][0]));
      }
      for (int component = 0; component < Dim; ++component)
        for (std::size_t node = 0; node < nodes_[component].size(); ++node)
          scale = std::max(scale, std::fabs(nodes_[component][node]));
      if (magnitude <= Real(64) * std::numeric_limits<Real>::epsilon() * scale)
        reference = RealVector<Dim>{};
      seam_translation[axis] = reference;
      for_each_index(extent, [&](auto node, std::int64_t) {
        auto upper = node;
        upper[axis] = cells_[axis];
        const auto low = node_offset(node), high = node_offset(upper);
        for (int component = 0; component < Dim; ++component)
          nodes_[component][static_cast<std::size_t>(high)] =
              nodes_[component][static_cast<std::size_t>(low)] + reference[component];
      });
    }

    // Cells: measure and centroid.
    cell_measure_.assign(static_cast<std::size_t>(cell_count), Real(0));
    for (int axis = 0; axis < Dim; ++axis) {
      centroid_[axis].assign(static_cast<std::size_t>(cell_count), Real(0));
      width_[axis].assign(static_cast<std::size_t>(cell_count), Real(0));
    }
    for_each_index(cells_, [&](const auto& cell, std::int64_t linear) {
      const auto index = static_cast<std::size_t>(linear);
      if (kind_ == StaticGeometryKind::nodes) {
        Real measure = Real(1);
        for (int axis = 0; axis < Dim; ++axis) {
          auto lower = cell, upper = cell;
          ++upper[axis];
          const Real low = nodes_[axis][static_cast<std::size_t>(node_offset(lower))];
          const Real high = nodes_[axis][static_cast<std::size_t>(node_offset(upper))];
          measure *= high - low;
          centroid_[axis][index] = Real(0.5) * (low + high);
        }
        cell_measure_[index] = measure;
      } else {
        Real measure = Real(0);
        RealVector<Dim> moment{};
        const std::size_t points = rule.nodes.size();
        std::array<std::size_t, Dim> point{};
        std::size_t total = 1;
        for (int axis = 0; axis < Dim; ++axis)
          total *= points;
        for (std::size_t linear_point = 0; linear_point < total; ++linear_point) {
          std::size_t rest = linear_point;
          for (int axis = 0; axis < Dim; ++axis) {
            point[axis] = rest % points;
            rest /= points;
          }
          RealVector<Dim> logical_point{};
          Real weight = Real(1);
          for (int axis = 0; axis < Dim; ++axis) {
            const Real low = logical[axis][static_cast<std::size_t>(cell[axis])];
            const Real high = logical[axis][static_cast<std::size_t>(cell[axis] + 1)];
            logical_point[axis] = low + rule.nodes[point[axis]] * (high - low);
            weight *= rule.weights[point[axis]] * (high - low);
          }
          std::array<RealVector<Dim>, Dim> jacobian{};
          RealVector<Dim> position{};
          for (int component = 0; component < Dim; ++component) {
            const auto jet = programs[component].eval_jet_checked(logical_point);
            if (!jet.valid)
              refuse("the coordinate map or its derivative is not finite in cell " +
                     cell_name(cell_index(cell)));
            position[component] = jet.value;
            jacobian[component] = jet.partials;
          }
          const Real determinant = static_geometry_detail::determinant<Dim>(jacobian);
          if (!(determinant > Real(0)))
            refuse("det DX is not positive in cell " + cell_name(cell_index(cell)) +
                   "; the chart folds or reverses orientation there");
          measure += weight * determinant;
          for (int component = 0; component < Dim; ++component)
            moment[component] += weight * determinant * position[component];
        }
        cell_measure_[index] = measure;
        for (int component = 0; component < Dim; ++component)
          centroid_[component][index] = moment[component] / measure;
      }
      if (!std::isfinite(cell_measure_[index]) || !(cell_measure_[index] > Real(0)))
        refuse("cell " + cell_name(cell_index(cell)) + " has no positive finite measure");
    });

    // Faces: area vector, measure, normal, centroid.
    for (int axis = 0; axis < Dim; ++axis) {
      std::array<std::int64_t, Dim> extent = cells_;
      ++extent[axis];
      std::int64_t face_count = 1;
      for (int other = 0; other < Dim; ++other)
        face_count *= extent[other];
      face_measure_[axis].assign(static_cast<std::size_t>(face_count), Real(0));
      face_distance_[axis].assign(static_cast<std::size_t>(face_count), Real(0));
      for (int component = 0; component < Dim; ++component) {
        face_area_[axis][component].assign(static_cast<std::size_t>(face_count), Real(0));
        face_normal_[axis][component].assign(static_cast<std::size_t>(face_count), Real(0));
        face_centroid_[axis][component].assign(static_cast<std::size_t>(face_count), Real(0));
      }
      for_each_index(extent, [&](const auto& face, std::int64_t linear) {
        const auto index = static_cast<std::size_t>(linear);
        RealVector<Dim> area{}, middle{};
        if (kind_ == StaticGeometryKind::nodes) {
          // A face of node positions is a face of a box: the logical axis times the product of
          // the node differences of the other axes (1 in rank one), centred on that face.
          Real product = Real(1);
          for (int other = 0; other < Dim; ++other) {
            const Real low = nodes_[other][static_cast<std::size_t>(node_offset(face))];
            if (other == axis) {
              middle[other] = low;
              continue;
            }
            auto end = face;
            ++end[other];
            const Real high = nodes_[other][static_cast<std::size_t>(node_offset(end))];
            product *= high - low;
            middle[other] = Real(0.5) * (low + high);
          }
          area[axis] = product;
        } else if constexpr (Dim == 2) {
          const int tangent = 1 - axis;
          auto end = face;
          ++end[tangent];
          const RealVector<Dim> start = node_position(face), stop = node_position(end);
          const Real tx = stop[0] - start[0], ty = stop[1] - start[1];
          area = axis == 0 ? RealVector<Dim>{ty, -tx} : RealVector<Dim>{-ty, tx};
          for (int component = 0; component < Dim; ++component)
            middle[component] = Real(0.5) * (start[component] + stop[component]);
        } else {
          refuse("a chart is served on layouts of rank two");
        }
        Real squared = Real(0);
        for (int component = 0; component < Dim; ++component)
          squared += area[component] * area[component];
        const Real measure = std::sqrt(squared);
        if (!std::isfinite(measure) || !(measure > Real(0)))
          refuse("a face of axis " + std::to_string(axis) + " has no positive finite measure");
        face_measure_[axis][index] = measure;
        for (int component = 0; component < Dim; ++component) {
          face_area_[axis][component][index] = area[component];
          face_normal_[axis][component][index] = area[component] / measure;
          face_centroid_[axis][component][index] = middle[component];
        }
      });
      // The seam of a periodic axis is one face: the upper copy reads the lower one bitwise
      // (its centroid translated by T).
      if (periodicity[axis]) {
        std::array<std::int64_t, Dim> seam = extent;
        seam[axis] = 1;
        for_each_index(seam, [&](auto face, std::int64_t) {
          auto upper = face;
          upper[axis] = cells_[axis];
          const auto low = static_cast<std::size_t>(face_linear(axis, face));
          const auto high = static_cast<std::size_t>(face_linear(axis, upper));
          face_measure_[axis][high] = face_measure_[axis][low];
          for (int component = 0; component < Dim; ++component) {
            face_area_[axis][component][high] = face_area_[axis][component][low];
            face_normal_[axis][component][high] = face_normal_[axis][component][low];
            face_centroid_[axis][component][high] =
                face_centroid_[axis][component][low] + seam_translation[axis][component];
          }
        });
      }
      // Distances, and the orientation across every face.
      for_each_index(extent, [&](const auto& face, std::int64_t linear) {
        const auto index = static_cast<std::size_t>(linear);
        const bool lower_end = face[axis] == 0, upper_end = face[axis] == cells_[axis];
        RealVector<Dim> left{}, right{};
        if ((lower_end || upper_end) && !periodicity[axis]) {
          if (kind_ == StaticGeometryKind::chart) {
            face_distance_[axis][index] = std::numeric_limits<Real>::quiet_NaN();
            return;
          }
          auto inside = face;
          if (upper_end)
            --inside[axis];
          const auto cell = static_cast<std::size_t>(cell_linear(inside));
          Real squared = Real(0), alignment = Real(0);
          for (int component = 0; component < Dim; ++component) {
            const Real gap = face_centroid_[axis][component][index] - centroid_[component][cell];
            squared += gap * gap;
            alignment += gap * face_area_[axis][component][index];
          }
          face_distance_[axis][index] = Real(2) * std::sqrt(squared);
          if (!(upper_end ? alignment > Real(0) : alignment < Real(0)))
            refuse("the centroid of an end cell of axis " + std::to_string(axis) +
                   " is not inside its end face");
          return;
        }
        auto left_cell = face, right_cell = face;
        if (lower_end) {
          left_cell[axis] = cells_[axis] - 1;
        } else {
          --left_cell[axis];
        }
        if (upper_end)
          right_cell[axis] = 0;
        const auto left_index = static_cast<std::size_t>(cell_linear(left_cell));
        const auto right_index = static_cast<std::size_t>(cell_linear(right_cell));
        for (int component = 0; component < Dim; ++component) {
          left[component] = centroid_[component][left_index];
          right[component] = centroid_[component][right_index];
          if (lower_end)
            left[component] -= seam_translation[axis][component];
          if (upper_end)
            right[component] += seam_translation[axis][component];
        }
        Real squared = Real(0), alignment = Real(0);
        for (int component = 0; component < Dim; ++component) {
          const Real gap = right[component] - left[component];
          squared += gap * gap;
          alignment += gap * face_area_[axis][component][index];
        }
        face_distance_[axis][index] = std::sqrt(squared);
        if (!(alignment > Real(0)))
          refuse("the area vector of a face of axis " + std::to_string(axis) +
                 " does not point from its left cell to its right cell");
      });
      // The two copies of a seam face are one face: the upper copy takes the distance of the
      // lower one bitwise, so that a two-point flux across the seam is one value on both sides.
      if (periodicity[axis]) {
        std::array<std::int64_t, Dim> seam = extent;
        seam[axis] = 1;
        for_each_index(seam, [&](auto face, std::int64_t) {
          auto upper = face;
          upper[axis] = cells_[axis];
          face_distance_[axis][static_cast<std::size_t>(face_linear(axis, upper))] =
              face_distance_[axis][static_cast<std::size_t>(face_linear(axis, face))];
        });
      }
    }
    for (int axis = 0; axis < Dim; ++axis)
      for (const Real distance : face_distance_[axis])
        if (!(kind_ == StaticGeometryKind::chart && std::isnan(distance)) &&
            (!std::isfinite(distance) || !(distance > Real(0))))
          refuse("a face distance of axis " + std::to_string(axis) + " is not positive");

    // Widths and closure.
    for_each_index(cells_, [&](const auto& cell, std::int64_t linear) {
      const auto index = static_cast<std::size_t>(linear);
      RealVector<Dim> closure{};
      Real floor = Real(0);
      for (int axis = 0; axis < Dim; ++axis) {
        auto upper = cell;
        ++upper[axis];
        const auto low = static_cast<std::size_t>(face_linear(axis, cell));
        const auto high = static_cast<std::size_t>(face_linear(axis, upper));
        if (kind_ == StaticGeometryKind::nodes) {
          // |K| / ((|S_lower| + |S_upper|) / 2) is the node difference along the axis: taken
          // exactly, it is the spacing of a graded box.
          auto node = cell;
          ++node[axis];
          width_[axis][index] = nodes_[axis][static_cast<std::size_t>(node_offset(node))] -
                                nodes_[axis][static_cast<std::size_t>(node_offset(cell))];
        } else {
          width_[axis][index] =
              cell_measure_[index] /
              (Real(0.5) * (face_measure_[axis][low] + face_measure_[axis][high]));
        }
        for (int component = 0; component < Dim; ++component) {
          closure[component] +=
              face_area_[axis][component][high] - face_area_[axis][component][low];
          floor += std::fabs(face_area_[axis][component][high]) +
                   std::fabs(face_area_[axis][component][low]);
        }
      }
      floor *= Real(8) * std::numeric_limits<Real>::epsilon();
      for (int component = 0; component < Dim; ++component) {
        const Real residual = std::fabs(closure[component]);
        if (residual > floor)
          refuse("the faces of cell " + cell_name(cell_index(cell)) + " do not close: residual " +
                 static_geometry_detail::number(residual) + " above the rounding floor " +
                 static_geometry_detail::number(floor));
        if (floor > Real(0))
          closure_ratio_ = std::max(closure_ratio_, residual / floor);
      }
    });
  }

  Index<Dim> cell_index(const std::array<std::int64_t, Dim>& offset) const {
    Index<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = static_cast<int>(domain_.lo[axis] + offset[axis]);
    return result;
  }
  std::int64_t cell_linear(const std::array<std::int64_t, Dim>& cell) const {
    std::int64_t offset = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      offset += cell[axis] * stride;
      stride *= cells_[axis];
    }
    return offset;
  }
  std::int64_t face_linear(int axis, const std::array<std::int64_t, Dim>& face) const {
    std::int64_t offset = 0, stride = 1;
    for (int other = 0; other < Dim; ++other) {
      offset += face[other] * stride;
      stride *= cells_[other] + (other == axis ? 1 : 0);
    }
    return offset;
  }

  std::string identity_;
  StaticGeometryKind kind_ = StaticGeometryKind::nodes;
  Box<Dim> domain_{};
  Real closure_ratio_ = Real(0);
  std::array<Storage, Dim> nodes_{};
  Storage cell_measure_;
  std::array<Storage, Dim> centroid_{};
  std::array<Storage, Dim> width_{};
  std::array<std::array<Storage, Dim>, Dim> face_area_{};
  std::array<Storage, Dim> face_measure_{};
  std::array<std::array<Storage, Dim>, Dim> face_normal_{};
  std::array<std::array<Storage, Dim>, Dim> face_centroid_{};
  std::array<Storage, Dim> face_distance_{};
};

/// The prepared metric of a mapped configuration: cell and face geometry read from the arrays,
/// with the physical unit normal of every face (physical_face_normals). The Jacobian and the
/// inverse map are not provided (capabilities false; NaN and OutsidePatch).
template <int Dim>
class PreparedStaticGeometryMetric {
 public:
  static constexpr int logical_dimension = Dim;
  static constexpr int embedding_dimension = Dim;
  static constexpr bool physical_face_normals = true;
  using PhysicalPoint = RealVector<Dim>;
  using Identity = PreparedMetricIdentity<Dim, Dim>;

  PreparedStaticGeometryMetric() = default;
  explicit PreparedStaticGeometryMetric(StaticGeometryView<Dim> view) : view_(view) {
    if (view_.domain.empty() || view_.cell_measure == nullptr)
      throw std::invalid_argument("prepared static geometry metric requires generated arrays");
    for (int axis = 0; axis < Dim; ++axis)
      inverse_extent_[axis] = Real(1) / static_cast<Real>(view_.domain.length(axis));
  }

  static constexpr PreparedMetricCapabilities capabilities() {
    CoordinateMapCapabilities map{};
    map.logical_dimension = Dim;
    map.embedding_dimension = Dim;
    map.kind = CoordinateMapKind::GeneratedArrays;
    map.cell_centers = true;
    map.face_centers = true;
    map.compile_time_axes = true;
    map.device_callable = true;
    return {map, true, false, true};
  }

  POPS_HD Identity identity() const {
    Identity result{};
    result.coordinate_map.kind = CoordinateMapKind::GeneratedArrays;
    for (int axis = 0; axis < Dim; ++axis) {
      result.coordinate_map.lower[axis] = Real(0);
      result.coordinate_map.upper[axis] = Real(1);
      result.coordinate_map.embedded_axis[axis] = axis;
      result.coordinate_map.orientation[axis] = 1;
    }
    result.domain = view_.domain;
    return result;
  }

  POPS_HD ReferenceCell<Dim> reference_cell(const Index<Dim>& index) const {
    ReferenceCell<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis) {
      const Real offset = static_cast<Real>(index[axis]) - static_cast<Real>(view_.domain.lo[axis]);
      result.lower[axis] = offset * inverse_extent_[axis];
      result.upper[axis] = (offset + Real(1)) * inverse_extent_[axis];
      result.center[axis] = (offset + Real(0.5)) * inverse_extent_[axis];
    }
    return result;
  }

  POPS_HD PhysicalPoint cell_center(const Index<Dim>& index) const {
    PhysicalPoint result{};
    for (int component = 0; component < Dim; ++component)
      result[component] = view_.centroid_at(component, index);
    return result;
  }

  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint face_center(const Index<Dim>& index) const {
    static_assert(Axis >= 0 && Axis < Dim);
    Index<Dim> face = index;
    if constexpr (Side == MetricFaceSide::Upper)
      ++face[Axis];
    PhysicalPoint result{};
    for (int component = 0; component < Dim; ++component)
      result[component] = view_.face_centroid_at(Axis, component, face);
    return result;
  }

  POPS_HD CoordinateJacobian<Dim, Dim> jacobian(const Index<Dim>&) const {
    CoordinateJacobian<Dim, Dim> result{};
    for (int row = 0; row < Dim; ++row)
      for (int column = 0; column < Dim; ++column)
        result[row][column] = std::numeric_limits<Real>::quiet_NaN();
    return result;
  }

  POPS_HD_FORCEINLINE Real cell_measure(const Index<Dim>& index) const {
    return view_.measure(index);
  }

  /// Outward for the cell, as every prepared metric: minus the lower face vector, plus the upper.
  template <int Axis, MetricFaceSide Side>
  POPS_HD_FORCEINLINE PhysicalPoint oriented_face_area_vector(const Index<Dim>& index) const {
    static_assert(Axis >= 0 && Axis < Dim);
    Index<Dim> face = index;
    if constexpr (Side == MetricFaceSide::Upper)
      ++face[Axis];
    constexpr Real sign = Side == MetricFaceSide::Upper ? Real(1) : Real(-1);
    PhysicalPoint result{};
    for (int component = 0; component < Dim; ++component)
      result[component] = sign * view_.area_at(Axis, component, face);
    return result;
  }

  /// Unit normal of a face in its canonical orientation (toward increasing index).
  template <int Axis>
  POPS_HD_FORCEINLINE void face_unit_normal(const Index<Dim>& face, Real (&normal)[3]) const {
    static_assert(Axis >= 0 && Axis < Dim);
    for (int component = 0; component < 3; ++component)
      normal[component] = component < Dim ? view_.normal_at(Axis, component, face) : Real(0);
  }

  POPS_HD InverseMapResult<Dim> inverse_map(const PhysicalPoint&) const {
    return {RealVector<Dim>{}, InverseMapStatus::OutsidePatch};
  }

  POPS_HD const StaticGeometryView<Dim>& arrays() const { return view_; }

 private:
  StaticGeometryView<Dim> view_{};
  RealVector<Dim> inverse_extent_{};
};

static_assert(PreparedMetricProvider<1, PreparedStaticGeometryMetric<1>>);
static_assert(PreparedMetricProvider<2, PreparedStaticGeometryMetric<2>>);
static_assert(PreparedMetricProvider<3, PreparedStaticGeometryMetric<3>>);

}  // namespace pops
