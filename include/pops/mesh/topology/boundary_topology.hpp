/// @file
/// @brief Compile-time-ranked Cartesian boundary topology.

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace pops {

enum class BoundarySide : unsigned char { lower, upper };

/// One oriented Cartesian domain face.  Ordinals are stable and dimension independent:
/// axis 0 lower/upper, axis 1 lower/upper, and so on.
template <int Dim>
struct Face {
  static_assert(Dim >= 1 && Dim <= 3, "pops::Face supports dimensions 1, 2, and 3");

  int axis = 0;
  BoundarySide side = BoundarySide::lower;

  constexpr Face() = default;
  constexpr Face(int face_axis, BoundarySide face_side) : axis(face_axis), side(face_side) {
    if (axis < 0 || axis >= Dim)
      throw std::invalid_argument("pops::Face axis is outside the compile-time rank");
  }

  constexpr int ordinal() const noexcept {
    return 2 * axis + (side == BoundarySide::upper ? 1 : 0);
  }

  constexpr int outward_sign() const noexcept { return side == BoundarySide::lower ? -1 : 1; }

  constexpr Face opposite() const noexcept {
    return Face{axis, side == BoundarySide::lower ? BoundarySide::upper : BoundarySide::lower};
  }

  constexpr bool operator==(const Face&) const = default;
};

template <int Dim>
constexpr bool face_less(Face<Dim> left, Face<Dim> right) noexcept {
  return left.ordinal() < right.ordinal();
}

enum class BoundaryFaceKind : unsigned char { physical, periodic };

/// One ordinary axis-translation periodic pairing.  Mapped/signed identifications deliberately
/// remain outside this value: a translation schedule must never silently approximate one.
template <int Dim>
struct PeriodicFacePair {
  Face<Dim> first{};
  Face<Dim> second{};

  PeriodicFacePair(Face<Dim> left, Face<Dim> right) : first(left), second(right) {
    if (left.axis != right.axis || left.side == right.side)
      throw std::invalid_argument("pops::PeriodicFacePair requires opposite sides of one axis");
    if (face_less(second, first)) {
      const Face<Dim> saved = first;
      first = second;
      second = saved;
    }
  }

  bool operator==(const PeriodicFacePair&) const = default;
};

template <int Dim>
struct BoundaryFaceRecord {
  Face<Dim> face{};
  BoundaryFaceKind kind = BoundaryFaceKind::physical;
  Face<Dim> partner{};

  bool operator==(const BoundaryFaceRecord&) const = default;
};

/// Complete Cartesian topology: every one of the 2*Dim faces is classified exactly once.
/// Unpaired faces are physical.  Periodic pairs are canonicalized and conflicting assignments are
/// rejected before any topology is published.
template <int Dim>
class BoundaryTopology {
  static_assert(Dim >= 1 && Dim <= 3, "pops::BoundaryTopology supports dimensions 1, 2, and 3");

 public:
  static constexpr std::size_t face_count = static_cast<std::size_t>(2 * Dim);

  BoundaryTopology() { initialize_physical_faces(); }

  explicit BoundaryTopology(const std::array<bool, Dim>& periodic_axes) {
    initialize_physical_faces();
    for (int axis = 0; axis < Dim; ++axis) {
      if (!periodic_axes[static_cast<std::size_t>(axis)])
        continue;
      assign_pair(PeriodicFacePair<Dim>{Face<Dim>{axis, BoundarySide::lower},
                                        Face<Dim>{axis, BoundarySide::upper}});
    }
  }

  template <std::size_t Count>
  explicit BoundaryTopology(const std::array<PeriodicFacePair<Dim>, Count>& periodic_pairs) {
    static_assert(Count <= static_cast<std::size_t>(Dim),
                  "a Cartesian topology has at most one periodic pair per axis");
    initialize_physical_faces();
    for (const PeriodicFacePair<Dim>& pair : periodic_pairs)
      assign_pair(pair);
  }

  static BoundaryTopology physical() { return BoundaryTopology{}; }

  static BoundaryTopology axis_periodic(const std::array<bool, Dim>& periodic_axes) {
    return BoundaryTopology{periodic_axes};
  }

  const std::array<BoundaryFaceRecord<Dim>, face_count>& faces() const noexcept { return faces_; }

  const BoundaryFaceRecord<Dim>& at(Face<Dim> face) const noexcept {
    return faces_[static_cast<std::size_t>(face.ordinal())];
  }

  BoundaryFaceKind kind(Face<Dim> face) const noexcept { return at(face).kind; }

  bool is_physical(Face<Dim> face) const noexcept {
    return kind(face) == BoundaryFaceKind::physical;
  }

  bool is_periodic(Face<Dim> face) const noexcept {
    return kind(face) == BoundaryFaceKind::periodic;
  }

  Face<Dim> partner(Face<Dim> face) const {
    if (!is_periodic(face))
      throw std::invalid_argument("pops::BoundaryTopology physical face has no partner");
    return at(face).partner;
  }

  std::size_t periodic_pair_count() const noexcept { return periodic_pair_count_; }

  bool operator==(const BoundaryTopology&) const = default;

 private:
  void initialize_physical_faces() noexcept {
    for (int axis = 0; axis < Dim; ++axis) {
      const Face<Dim> lower{axis, BoundarySide::lower};
      const Face<Dim> upper{axis, BoundarySide::upper};
      faces_[static_cast<std::size_t>(lower.ordinal())] =
          BoundaryFaceRecord<Dim>{lower, BoundaryFaceKind::physical, lower};
      faces_[static_cast<std::size_t>(upper.ordinal())] =
          BoundaryFaceRecord<Dim>{upper, BoundaryFaceKind::physical, upper};
    }
    periodic_pair_count_ = 0;
  }

  void assign_pair(PeriodicFacePair<Dim> pair) {
    const std::size_t first = static_cast<std::size_t>(pair.first.ordinal());
    const std::size_t second = static_cast<std::size_t>(pair.second.ordinal());
    if (faces_[first].kind == BoundaryFaceKind::periodic ||
        faces_[second].kind == BoundaryFaceKind::periodic)
      throw std::invalid_argument(
          "pops::BoundaryTopology assigns one face to multiple periodic pairs");
    faces_[first] = BoundaryFaceRecord<Dim>{pair.first, BoundaryFaceKind::periodic, pair.second};
    faces_[second] = BoundaryFaceRecord<Dim>{pair.second, BoundaryFaceKind::periodic, pair.first};
    ++periodic_pair_count_;
  }

  std::array<BoundaryFaceRecord<Dim>, face_count> faces_{};
  std::size_t periodic_pair_count_ = 0;
};

static_assert(std::is_trivially_copyable_v<Face<1>>);
static_assert(std::is_trivially_copyable_v<BoundaryFaceRecord<3>>);

}  // namespace pops
